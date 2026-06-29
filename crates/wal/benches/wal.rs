use criterion::{
    BatchSize, BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use pagebox_frame_kernel::PAGE_SIZE;
use pagebox_wal::{WAL_BUF_RECORDS, Wal};

fn page_data(seed: u64) -> [u8; PAGE_SIZE] {
    let mut buf = [0u8; PAGE_SIZE];
    let bytes = seed.to_le_bytes();
    for chunk in buf.chunks_exact_mut(8) {
        chunk.copy_from_slice(&bytes);
    }
    buf
}

// ---------------------------------------------------------------------------
// WAL factory
// ---------------------------------------------------------------------------

trait WalFactory: Clone + 'static {
    fn label(&self) -> &'static str;
    fn create(&self) -> (tempfile::TempDir, Wal);
}

#[derive(Clone)]
struct PwriteFactory;

impl WalFactory for PwriteFactory {
    fn label(&self) -> &'static str {
        "pwrite"
    }
    fn create(&self) -> (tempfile::TempDir, Wal) {
        let dir = tempfile::tempdir().expect("tempdir");
        let wal = Wal::open_opts(&dir.path().join("wal")).unwrap();
        (dir, wal)
    }
}

// ---------------------------------------------------------------------------
// I/O benchmark configuration
// ---------------------------------------------------------------------------

fn io_group(
    c: &mut Criterion,
    name: String,
) -> criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> {
    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    fn env_secs(name: &str, default: u64) -> Duration {
        Duration::from_secs(
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default),
        )
    }

    let mut g = c.benchmark_group(name);
    g.sampling_mode(SamplingMode::Flat);
    g.sample_size(env_usize("PAGEBOX_WAL_BENCH_SAMPLE_SIZE", 50));
    g.measurement_time(env_secs("PAGEBOX_WAL_BENCH_MEASUREMENT_SECS", 10));
    g.warm_up_time(env_secs("PAGEBOX_WAL_BENCH_WARMUP_SECS", 3));
    g.noise_threshold(0.10);
    g
}

// ---------------------------------------------------------------------------
// 1. Append-only — pure serialization + memcpy into the 256KB buffer
//
// Not parameterized over backend because no I/O happens.
// ---------------------------------------------------------------------------

fn append_buffer_only(c: &mut Criterion) {
    let mut group = io_group(c, "wal/append_buffer_only".to_string());
    let n = WAL_BUF_RECORDS as u64;
    group.throughput(Throughput::Elements(n));
    group.bench_function(format!("{n}_records"), |b| {
        let pages: Vec<[u8; PAGE_SIZE]> = (0..n).map(page_data).collect();
        b.iter_batched(
            || {
                let dir = tempfile::tempdir().expect("tempdir");
                let wal = Wal::open_opts(&dir.path().join("wal")).unwrap();
                (dir, wal)
            },
            |(_dir, wal)| {
                for (i, pd) in pages.iter().enumerate() {
                    wal.append_page_image(i as u64, pd).unwrap();
                }
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// 2. Append + flush — full durability cost per batch
// ---------------------------------------------------------------------------

fn bench_append_flush<F: WalFactory>(c: &mut Criterion, factory: F) {
    let label = factory.label();
    let mut group = io_group(c, format!("wal/{label}/append_flush"));
    for &batch in &[1, 32, 256] {
        let pages: Vec<[u8; PAGE_SIZE]> = (0..batch).map(|i| page_data(i as u64)).collect();
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_with_input(BenchmarkId::new("batch", batch), &batch, |b, _| {
            b.iter_batched(
                || factory.create(),
                |(_dir, wal)| {
                    for (i, pd) in pages.iter().enumerate() {
                        wal.append_page_image(i as u64, pd).unwrap();
                    }
                    wal.flush();
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Flush latency — isolate fdatasync cost
// ---------------------------------------------------------------------------

fn bench_flush_latency<F: WalFactory>(c: &mut Criterion, factory: F) {
    let label = factory.label();
    let mut group = io_group(c, format!("wal/{label}/flush_latency"));
    for &pending in &[1, 8, 32] {
        group.bench_with_input(
            BenchmarkId::new("pending_records", pending),
            &pending,
            |b, &pending| {
                let pd = page_data(42);
                b.iter_batched(
                    || factory.create(),
                    |(_dir, wal)| {
                        for i in 0..pending {
                            wal.append_page_image(i as u64, &pd).unwrap();
                        }
                        wal.flush();
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Group commit effectiveness — N threads flushing concurrently
// ---------------------------------------------------------------------------

fn bench_group_commit<F: WalFactory>(c: &mut Criterion, factory: F) {
    let label = factory.label();
    let mut group = io_group(c, format!("wal/{label}/group_commit"));
    for &n_threads in &[1, 2, 4, 8] {
        group.throughput(Throughput::Elements(n_threads as u64));
        group.bench_with_input(
            BenchmarkId::new("threads", n_threads),
            &n_threads,
            |b, &n_threads| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let wal = Arc::new(factory.create());
                        let go = Arc::new(Barrier::new(n_threads + 1));
                        let done = Arc::new(Barrier::new(n_threads + 1));

                        let handles: Vec<_> = (0..n_threads)
                            .map(|t| {
                                let wal = Arc::clone(&wal);
                                let go = Arc::clone(&go);
                                let done = Arc::clone(&done);
                                std::thread::spawn(move || {
                                    let pd = page_data(t as u64);
                                    go.wait();
                                    let lsn = wal.1.append_page_image(t as u64, &pd).unwrap();
                                    wal.1.flush_at_least(lsn);
                                    done.wait();
                                })
                            })
                            .collect();

                        let start = Instant::now();
                        go.wait();
                        done.wait();
                        total += start.elapsed();

                        for h in handles {
                            h.join().unwrap();
                        }
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 5. Concurrent append throughput — N threads appending, single final flush
// ---------------------------------------------------------------------------

fn bench_concurrent_append<F: WalFactory>(c: &mut Criterion, factory: F) {
    let label = factory.label();
    let records_per_thread = 1_000u64;
    let mut group = io_group(c, format!("wal/{label}/concurrent_append"));
    for &n_threads in &[1, 2, 4, 8] {
        let total = records_per_thread * n_threads as u64;
        group.throughput(Throughput::Elements(total));
        group.bench_with_input(
            BenchmarkId::new("threads", n_threads),
            &n_threads,
            |b, &n_threads| {
                b.iter_custom(|iters| {
                    let mut total_elapsed = Duration::ZERO;
                    for _ in 0..iters {
                        let wal = Arc::new(factory.create());
                        let go = Arc::new(Barrier::new(n_threads + 1));
                        let done = Arc::new(Barrier::new(n_threads + 1));

                        let handles: Vec<_> = (0..n_threads)
                            .map(|t| {
                                let wal = Arc::clone(&wal);
                                let go = Arc::clone(&go);
                                let done = Arc::clone(&done);
                                std::thread::spawn(move || {
                                    let pd = page_data(t as u64);
                                    go.wait();
                                    for i in 0..records_per_thread {
                                        let pid = t as u64 * records_per_thread + i;
                                        wal.1.append_page_image(pid, &pd).unwrap();
                                    }
                                    done.wait();
                                })
                            })
                            .collect();

                        let start = Instant::now();
                        go.wait();
                        done.wait();
                        wal.1.flush();
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

// ---------------------------------------------------------------------------
// 6. Replay throughput — not parameterized (replay uses pread, not flush)
// ---------------------------------------------------------------------------

fn replay(c: &mut Criterion) {
    let mut group = io_group(c, "wal/replay".to_string());
    for &n in &[1_000, 10_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let dir = tempfile::tempdir().unwrap();
            let wal_path = dir.path().join("wal");
            let wal = Wal::open(&wal_path).unwrap();
            let pd = page_data(7);
            for i in 0..n {
                wal.append_page_image(i as u64, &pd).unwrap();
            }
            wal.flush();

            b.iter(|| {
                let mut count = 0u64;
                wal.replay(|_lsn, _pid, _data| {
                    count += 1;
                })
                .unwrap();
                assert_eq!(count, n as u64);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 7. Steady-state: append + periodic flush
// ---------------------------------------------------------------------------

fn bench_steady_state<F: WalFactory>(c: &mut Criterion, factory: F) {
    let label = factory.label();
    let mut group = io_group(c, format!("wal/{label}/steady_state"));
    for &flush_every in &[1, 16, 64] {
        let total = 1_000u64;
        group.throughput(Throughput::Elements(total));
        group.bench_with_input(
            BenchmarkId::new("flush_every", flush_every),
            &flush_every,
            |b, &flush_every| {
                let pd = page_data(99);
                b.iter_batched(
                    || factory.create(),
                    |(_dir, wal)| {
                        for i in 0..total {
                            wal.append_page_image(i, &pd).unwrap();
                            if (i + 1) % flush_every == 0 {
                                wal.flush();
                            }
                        }
                        wal.flush();
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 8. Concurrent durable commits — N threads each doing append +
//    flush_at_least in a loop. This is the most direct approximation of
//    workers committing through a single shared WAL.
// ---------------------------------------------------------------------------

fn bench_concurrent_commit<F: WalFactory>(c: &mut Criterion, factory: F) {
    let label = factory.label();
    let commits_per_thread = 100u64;
    let mut group = io_group(c, format!("wal/{label}/concurrent_commit"));
    for &records_per_commit in &[1u64, 4, 16] {
        for &n_threads in &[1, 2, 4, 8] {
            let total_commits = commits_per_thread * n_threads as u64;
            group.throughput(Throughput::Elements(total_commits));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("records_per_commit={records_per_commit}"),
                    n_threads,
                ),
                &n_threads,
                |b, &n_threads| {
                    b.iter_custom(|iters| {
                        let mut total_elapsed = Duration::ZERO;
                        for _ in 0..iters {
                            let wal = Arc::new(factory.create());
                            let go = Arc::new(Barrier::new(n_threads + 1));
                            let done = Arc::new(Barrier::new(n_threads + 1));

                            let handles: Vec<_> = (0..n_threads)
                                .map(|t| {
                                    let wal = Arc::clone(&wal);
                                    let go = Arc::clone(&go);
                                    let done = Arc::clone(&done);
                                    std::thread::spawn(move || {
                                        let pd = page_data(t as u64);
                                        go.wait();
                                        for c in 0..commits_per_thread {
                                            let mut last_lsn = 0;
                                            for r in 0..records_per_commit {
                                                let pid = t as u64 * 100_000 + c * 100 + r;
                                                last_lsn =
                                                    wal.1.append_page_image(pid, &pd).unwrap();
                                            }
                                            wal.1.flush_at_least(last_lsn);
                                        }
                                        done.wait();
                                    })
                                })
                                .collect();

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
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

macro_rules! register_wal_benches {
    ($($factory:expr),+ $(,)?) => {
        paste::paste! {
            $(
                fn [<append_flush_ $factory:snake>](c: &mut Criterion) {
                    bench_append_flush(c, $factory);
                }
                fn [<flush_latency_ $factory:snake>](c: &mut Criterion) {
                    bench_flush_latency(c, $factory);
                }
                fn [<group_commit_ $factory:snake>](c: &mut Criterion) {
                    bench_group_commit(c, $factory);
                }
                fn [<concurrent_append_ $factory:snake>](c: &mut Criterion) {
                    bench_concurrent_append(c, $factory);
                }
                fn [<steady_state_ $factory:snake>](c: &mut Criterion) {
                    bench_steady_state(c, $factory);
                }
                fn [<concurrent_commit_ $factory:snake>](c: &mut Criterion) {
                    bench_concurrent_commit(c, $factory);
                }
            )+

            criterion_group!(
                benches,
                append_buffer_only,
                $(
                    [<append_flush_ $factory:snake>],
                    [<flush_latency_ $factory:snake>],
                    [<group_commit_ $factory:snake>],
                    [<concurrent_append_ $factory:snake>],
                    [<steady_state_ $factory:snake>],
                    [<concurrent_commit_ $factory:snake>],
                )+
                replay,
            );
        }
    };
}

register_wal_benches!(PwriteFactory);

criterion_main!(benches);
