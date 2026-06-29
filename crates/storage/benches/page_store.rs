use criterion::{
    BatchSize, BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use pagebox_storage::buffer_frame::PAGE_SIZE;
use pagebox_storage::page_store::{BatchPageStore, FilePageStore, PageStore};

// ---------------------------------------------------------------------------
// Factory trait — implement this for each PageStore backend to plug into
// every benchmark automatically.
// ---------------------------------------------------------------------------

/// A factory that produces a fresh, empty PageStore for benchmarking.
///
/// The `Env` associated type holds any external resources (temp dirs, etc.)
/// that must outlive the store.  It is kept alive but otherwise unused.
trait StoreFactory: Clone + 'static {
    type Env: Send;
    type Store: BatchPageStore + Send + Sync + 'static;

    /// Human-readable label used in benchmark IDs.
    fn label(&self) -> &'static str;

    /// Create a fresh store with no pre-existing pages.
    fn create(&self) -> (Self::Env, Self::Store);
}

// ---------------------------------------------------------------------------
// FilePageStore factory
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct FileFactory;

impl StoreFactory for FileFactory {
    type Env = tempfile::TempDir;
    type Store = FilePageStore;

    fn label(&self) -> &'static str {
        "file"
    }

    fn create(&self) -> (Self::Env, Self::Store) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FilePageStore::open(&dir.path().join("data")).unwrap();
        (dir, store)
    }
}

// ---------------------------------------------------------------------------
// I/O benchmark configuration
// ---------------------------------------------------------------------------

/// Configure a benchmark group for I/O workloads.  Flat sampling avoids
/// Criterion's adaptive iteration-count scaling, which interacts badly
/// with OS page cache and writeback variance.
fn io_group(
    c: &mut Criterion,
    name: String,
) -> criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> {
    let mut g = c.benchmark_group(name);
    g.sampling_mode(SamplingMode::Flat);
    g.sample_size(50);
    g.measurement_time(Duration::from_secs(10));
    g.warm_up_time(Duration::from_secs(3));
    g.noise_threshold(0.10);
    g
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn page_data(seed: u64) -> [u8; PAGE_SIZE] {
    let mut buf = [0u8; PAGE_SIZE];
    let bytes = seed.to_le_bytes();
    for chunk in buf.chunks_exact_mut(8) {
        chunk.copy_from_slice(&bytes);
    }
    buf
}

/// Pre-allocate and write `n` pages into `store` (pids 1..=n).
fn populate(store: &impl PageStore, n: u64) {
    for pid in 1..=n {
        store.allocate(pid).unwrap();
        store.write_page(pid, &page_data(pid)).unwrap();
    }
    store.sync().unwrap();
}

/// Deterministic shuffle of 1..=n using a multiplicative hash.
fn shuffled_pids(n: u64) -> Vec<u64> {
    let mut pids: Vec<u64> = (1..=n).collect();
    // Fisher-Yates with deterministic hash as entropy source.
    for i in (1..pids.len()).rev() {
        let h = (i as u64).wrapping_mul(0x517cc1b727220a95).wrapping_add(1);
        let j = (h as usize) % (i + 1);
        pids.swap(i, j);
    }
    pids
}

// ---------------------------------------------------------------------------
// Single-op benchmarks (PageStore trait)
// ---------------------------------------------------------------------------

fn bench_sequential_write<F: StoreFactory>(c: &mut Criterion, factory: F) {
    let label = factory.label();
    let mut group = io_group(c, format!("page_store/{label}/seq_write"));
    for &n in &[1_000, 10_000] {
        let pages: Vec<[u8; PAGE_SIZE]> = (1..=n).map(|i| page_data(i as u64)).collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            // Allocate once; measure steady-state overwrite throughput.
            let (_env, store) = factory.create();
            populate(&store, n as u64);
            b.iter(|| {
                for (i, pd) in pages.iter().enumerate() {
                    store.write_page((i + 1) as u64, pd).unwrap();
                }
            });
        });
    }
    group.finish();
}

fn bench_random_write<F: StoreFactory>(c: &mut Criterion, factory: F) {
    let label = factory.label();
    let mut group = io_group(c, format!("page_store/{label}/rand_write"));
    for &n in &[1_000, 10_000] {
        let order = shuffled_pids(n as u64);
        let pages: Vec<[u8; PAGE_SIZE]> = order.iter().map(|&pid| page_data(pid)).collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let (_env, store) = factory.create();
            populate(&store, n as u64);
            b.iter(|| {
                for (pd, &pid) in pages.iter().zip(order.iter()) {
                    store.write_page(pid, pd).unwrap();
                }
            });
        });
    }
    group.finish();
}

fn bench_sequential_read<F: StoreFactory>(c: &mut Criterion, factory: F) {
    let label = factory.label();
    let mut group = io_group(c, format!("page_store/{label}/seq_read"));
    for &n in &[1_000, 10_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            // Populate once; reads are non-mutating.
            let (_env, store) = factory.create();
            populate(&store, n as u64);
            let mut buf = [0u8; PAGE_SIZE];
            b.iter(|| {
                for pid in 1..=n as u64 {
                    store.read_page(pid, &mut buf).unwrap();
                }
            });
        });
    }
    group.finish();
}

fn bench_random_read<F: StoreFactory>(c: &mut Criterion, factory: F) {
    let label = factory.label();
    let mut group = io_group(c, format!("page_store/{label}/rand_read"));
    for &n in &[1_000, 10_000] {
        let order = shuffled_pids(n as u64);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let (_env, store) = factory.create();
            populate(&store, n as u64);
            let mut buf = [0u8; PAGE_SIZE];
            b.iter(|| {
                for &pid in &order {
                    store.read_page(pid, &mut buf).unwrap();
                }
            });
        });
    }
    group.finish();
}

fn bench_concurrent_read_write<F: StoreFactory>(c: &mut Criterion, factory: F) {
    let label = factory.label();
    let mut group = io_group(c, format!("page_store/{label}/concurrent_rw"));
    let pages_per_thread = 500u64;
    for &n_threads in &[2, 4, 8] {
        // Half readers, half writers.
        let n_writers = n_threads / 2;
        let n_readers = n_threads - n_writers;
        let total_ops = pages_per_thread * n_threads as u64;
        group.throughput(Throughput::Elements(total_ops));
        group.bench_with_input(
            BenchmarkId::new("threads", n_threads),
            &n_threads,
            |b, _| {
                b.iter_custom(|iters| {
                    let mut total_elapsed = std::time::Duration::ZERO;
                    for _ in 0..iters {
                        let (_env, store) = factory.create();
                        let total_pages = pages_per_thread * n_writers as u64;
                        // Pre-allocate so writers don't contend on alloc_lock.
                        store.allocate(total_pages).unwrap();
                        // Seed some data for readers.
                        populate(&store, total_pages);

                        let store = Arc::new(store);
                        let go = Arc::new(Barrier::new(n_threads + 1));
                        let done = Arc::new(Barrier::new(n_threads + 1));

                        let mut handles = Vec::with_capacity(n_threads);

                        // Spawn writers.
                        for t in 0..n_writers {
                            let store = Arc::clone(&store);
                            let go = Arc::clone(&go);
                            let done = Arc::clone(&done);
                            handles.push(std::thread::spawn(move || {
                                go.wait();
                                let base = t as u64 * pages_per_thread + 1;
                                for i in 0..pages_per_thread {
                                    let pid = base + i;
                                    store.write_page(pid, &page_data(pid)).unwrap();
                                }
                                done.wait();
                            }));
                        }

                        // Spawn readers.
                        for _ in 0..n_readers {
                            let store = Arc::clone(&store);
                            let go = Arc::clone(&go);
                            let done = Arc::clone(&done);
                            handles.push(std::thread::spawn(move || {
                                go.wait();
                                let mut buf = [0u8; PAGE_SIZE];
                                for pid in 1..=pages_per_thread {
                                    store.read_page(pid, &mut buf).unwrap();
                                }
                                done.wait();
                            }));
                        }

                        let start = Instant::now();
                        go.wait();
                        done.wait();
                        total_elapsed += start.elapsed();

                        for h in handles {
                            h.join().unwrap();
                        }
                    }
                    total_elapsed
                });
            },
        );
    }
    group.finish();
}

fn bench_sync_latency<F: StoreFactory>(c: &mut Criterion, factory: F) {
    let label = factory.label();
    let mut group = io_group(c, format!("page_store/{label}/sync"));
    for &dirty_pages in &[1, 32, 256] {
        group.bench_with_input(
            BenchmarkId::new("dirty_pages", dirty_pages),
            &dirty_pages,
            |b, &dirty_pages| {
                b.iter_batched(
                    || {
                        let (env, store) = factory.create();
                        store.allocate(dirty_pages as u64).unwrap();
                        for pid in 1..=dirty_pages as u64 {
                            store.write_page(pid, &page_data(pid)).unwrap();
                        }
                        (env, store)
                    },
                    |(_env, store)| {
                        store.sync().unwrap();
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Batch benchmarks (BatchPageStore trait)
// ---------------------------------------------------------------------------

/// Submit N writes via submit_write + wait_all (no sync).
fn bench_batch_write<F: StoreFactory>(c: &mut Criterion, factory: F) {
    let label = factory.label();
    let mut group = io_group(c, format!("page_store/{label}/batch_write"));
    for &n in &[256, 1_000] {
        let pages: Vec<[u8; PAGE_SIZE]> = (1..=n).map(|i| page_data(i as u64)).collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let (_env, store) = factory.create();
            populate(&store, n as u64);
            b.iter(|| {
                for (i, pd) in pages.iter().enumerate() {
                    unsafe { store.submit_write((i + 1) as u64, pd).unwrap() };
                }
                store.wait_all().unwrap();
            });
        });
    }
    group.finish();
}

/// Submit N writes + fsync in one batch via submit_write + submit_sync +
/// wait_all.  This is the path BufferPool::flush would take.
fn bench_batch_write_sync<F: StoreFactory>(c: &mut Criterion, factory: F) {
    let label = factory.label();
    let mut group = io_group(c, format!("page_store/{label}/batch_write_sync"));
    for &n in &[256, 1_000] {
        let pages: Vec<[u8; PAGE_SIZE]> = (1..=n).map(|i| page_data(i as u64)).collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let (_env, store) = factory.create();
            populate(&store, n as u64);
            b.iter(|| {
                for (i, pd) in pages.iter().enumerate() {
                    unsafe { store.submit_write((i + 1) as u64, pd).unwrap() };
                }
                store.submit_sync().unwrap();
                store.wait_all().unwrap();
            });
        });
    }
    group.finish();
}

/// Submit N reads via submit_read + wait_all on pre-populated store.
fn bench_batch_read<F: StoreFactory>(c: &mut Criterion, factory: F) {
    let label = factory.label();
    let mut group = io_group(c, format!("page_store/{label}/batch_read"));
    for &n in &[256, 1_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let (_env, store) = factory.create();
            populate(&store, n as u64);
            let mut bufs = vec![[0u8; PAGE_SIZE]; n as usize];
            b.iter(|| {
                for (i, buf) in bufs.iter_mut().enumerate() {
                    unsafe { store.submit_read((i + 1) as u64, buf).unwrap() };
                }
                store.wait_all().unwrap();
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Registration — add new backends here.
// ---------------------------------------------------------------------------

macro_rules! register_benches {
    ($($factory:expr),+ $(,)?) => {
        // Generate one Criterion function per (benchmark × factory).
        paste::paste! {
            $(
                fn [<seq_write_ $factory:snake>](c: &mut Criterion) {
                    bench_sequential_write(c, $factory);
                }
                fn [<rand_write_ $factory:snake>](c: &mut Criterion) {
                    bench_random_write(c, $factory);
                }
                fn [<seq_read_ $factory:snake>](c: &mut Criterion) {
                    bench_sequential_read(c, $factory);
                }
                fn [<rand_read_ $factory:snake>](c: &mut Criterion) {
                    bench_random_read(c, $factory);
                }
                fn [<concurrent_rw_ $factory:snake>](c: &mut Criterion) {
                    bench_concurrent_read_write(c, $factory);
                }
                fn [<sync_ $factory:snake>](c: &mut Criterion) {
                    bench_sync_latency(c, $factory);
                }
                fn [<batch_write_ $factory:snake>](c: &mut Criterion) {
                    bench_batch_write(c, $factory);
                }
                fn [<batch_write_sync_ $factory:snake>](c: &mut Criterion) {
                    bench_batch_write_sync(c, $factory);
                }
                fn [<batch_read_ $factory:snake>](c: &mut Criterion) {
                    bench_batch_read(c, $factory);
                }
            )+

            criterion_group!(
                benches,
                $(
                    [<seq_write_ $factory:snake>],
                    [<rand_write_ $factory:snake>],
                    [<seq_read_ $factory:snake>],
                    [<rand_read_ $factory:snake>],
                    [<concurrent_rw_ $factory:snake>],
                    [<sync_ $factory:snake>],
                    [<batch_write_ $factory:snake>],
                    [<batch_write_sync_ $factory:snake>],
                    [<batch_read_ $factory:snake>],
                )+
            );
        }
    };
}

register_benches!(FileFactory);

criterion_main!(benches);
