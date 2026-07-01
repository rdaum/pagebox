//! Microbenchmark for the buffer pool eviction path.
//!
//! Isolates eviction throughput and latency from B-tree traversal logic.
//! Each worker pins a frame, reads one byte, drops the pin, and moves to
//! the next page. When the pool is full, fix_orphan_frame triggers
//! eviction. The backoff strategy and eviction permit budget control
//! how contention resolves.
//!
//! Environment variables:
//!   PAGEBOX_BP_EVICT_PAGES     — total pages in the store (default 100_000)
//!   PAGEBOX_BP_EVICT_POOL      — buffer pool frames (default 1024)
//!   PAGEBOX_BP_EVICT_PAGES_PER_THREAD — pages touched per worker before stopping (default 10_000)

use std::env;
use std::sync::Arc;
use std::time::Duration;

use micromeasure::{
    ConcurrentBenchContext, ConcurrentBenchControl, ConcurrentWorker, ConcurrentWorkerResult,
    Throughput, benchmark_main, black_box,
};
use pagebox_storage::buffer_frame::PAGE_SIZE;
use pagebox_storage::buffer_pool::{BufferPool, NoLatches};
use pagebox_storage::page_store::{FilePageStore, PageStore};

#[repr(align(4096))]
struct AlignedPage([u8; PAGE_SIZE]);

#[derive(Clone, Copy)]
struct EvictBenchConfig {
    num_pages: usize,
    pool_frames: usize,
    pages_per_thread: usize,
}

impl EvictBenchConfig {
    fn from_env() -> Self {
        Self {
            num_pages: env_usize("PAGEBOX_BP_EVICT_PAGES", 100_000),
            pool_frames: env_usize("PAGEBOX_BP_EVICT_POOL", 1_024),
            pages_per_thread: env_usize("PAGEBOX_BP_EVICT_PAGES_PER_THREAD", 10_000),
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn shuffled(n: usize) -> Vec<u64> {
    let mut v: Vec<u64> = (1..=n as u64).collect();
    for i in (1..v.len()).rev() {
        let h = (i as u64)
            .wrapping_mul(0x517c_c1b7_2722_0a95)
            .wrapping_add(1);
        let j = (h as usize) % (i + 1);
        v.swap(i, j);
    }
    v
}

fn populate_store(config: EvictBenchConfig) -> tempfile::TempDir {
    assert!(
        config.num_pages > 0,
        "PAGEBOX_BP_EVICT_PAGES must be greater than zero"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data");
    let store = FilePageStore::open(&path).expect("open page store");
    store.allocate(config.num_pages as u64).expect("allocate");

    let mut page = AlignedPage([0u8; PAGE_SIZE]);
    for pid in 1..=config.num_pages as u64 {
        page.0[0..8].copy_from_slice(&pid.to_le_bytes());
        store.write_page(pid, &page.0).expect("write page");
    }
    store.sync().expect("sync");
    store.drop_cache();
    drop(store);
    dir
}

fn open_pool(dir: &std::path::Path, config: EvictBenchConfig) -> BufferPool {
    let store = FilePageStore::open(&dir.join("data")).expect("open page store");
    store.drop_cache();
    BufferPool::with_store(config.pool_frames, Box::new(store))
}

struct EvictCtx {
    _dir: tempfile::TempDir,
    pool: Arc<BufferPool>,
    thread_pages: Vec<Vec<u64>>,
    eviction_before: u64,
}

impl ConcurrentBenchContext for EvictCtx {
    fn prepare(num_threads: usize) -> Self {
        let config = EvictBenchConfig::from_env();
        let dir = populate_store(config);
        let pool = Arc::new(open_pool(dir.path(), config));

        let order = shuffled(config.num_pages);
        let thread_pages: Vec<Vec<u64>> = (0..num_threads)
            .map(|thread_idx| {
                let offset = (thread_idx * 997) % order.len();
                order
                    .iter()
                    .cycle()
                    .skip(offset)
                    .take(config.pages_per_thread)
                    .copied()
                    .collect()
            })
            .collect();

        let eviction_before = pool.eviction_count();
        Self {
            _dir: dir,
            pool,
            thread_pages,
            eviction_before,
        }
    }
}

/// Sequential access pattern: each thread walks its page list in order.
/// This exercises the eviction path with predictable access patterns
/// (semi-sequential within each thread, random across threads).
fn evict_worker_sequential(
    ctx: &EvictCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let pool = &ctx.pool;
    let pages = &ctx.thread_pages[control.role_thread_index()];
    let mut operations = 0u64;
    let mut checksum = 0u64;
    let mut latencies_us: Vec<u64> = Vec::with_capacity(1024);

    while !control.should_stop() {
        for &pid in pages {
            let start = std::time::Instant::now();
            let frame = unsafe { pool.fix_orphan_frame(pid, NoLatches::new(pool)) };
            checksum ^= u64::from_le_bytes(frame.page[0..8].try_into().expect("pid bytes"));
            let elapsed_us = start.elapsed().as_micros() as u64;
            if latencies_us.len() < 4096 {
                latencies_us.push(elapsed_us);
            }
            drop(frame);
            operations += 1;
            if control.should_stop() {
                break;
            }
        }
    }

    let mut result =
        ConcurrentWorkerResult::operations(operations).with_counter("page_ops", operations);

    if control.role_thread_index() == 0 {
        let evictions = pool.eviction_count();
        let delta = evictions.saturating_sub(ctx.eviction_before);
        result = result.with_counter("evictions", delta);
    }

    if !latencies_us.is_empty() {
        latencies_us.sort_unstable();
        let n = latencies_us.len();
        let p50 = latencies_us[n / 2];
        let p95 = latencies_us[(n * 95) / 100];
        let p99 = latencies_us[(n * 99) / 100];
        result = result.with_counter("p50_us", p50);
        result = result.with_counter("p95_us", p95);
        result = result.with_counter("p99_us", p99);
    }

    black_box(checksum);
    result
}

/// Uniform random access pattern: each thread accesses random pages.
/// This is the worst case for eviction — no locality, every access
/// may trigger eviction.
fn evict_worker_random(ctx: &EvictCtx, control: &ConcurrentBenchControl) -> ConcurrentWorkerResult {
    let pool = &ctx.pool;
    let pages = &ctx.thread_pages[control.role_thread_index()];
    let mut operations = 0u64;
    let mut checksum = 0u64;
    let mut latencies_us: Vec<u64> = Vec::with_capacity(1024);
    let mut idx = 0usize;

    while !control.should_stop() {
        let pid = pages[idx % pages.len()];
        idx = idx.wrapping_add(1);

        let start = std::time::Instant::now();
        let frame = unsafe { pool.fix_orphan_frame(pid, NoLatches::new(pool)) };
        checksum ^= u64::from_le_bytes(frame.page[0..8].try_into().expect("pid bytes"));
        let elapsed_us = start.elapsed().as_micros() as u64;
        if latencies_us.len() < 4096 {
            latencies_us.push(elapsed_us);
        }
        drop(frame);
        operations += 1;
        if control.should_stop() {
            break;
        }
    }

    let mut result =
        ConcurrentWorkerResult::operations(operations).with_counter("page_ops", operations);

    if control.role_thread_index() == 0 {
        let evictions = pool.eviction_count();
        let delta = evictions.saturating_sub(ctx.eviction_before);
        result = result.with_counter("evictions", delta);
    }

    if !latencies_us.is_empty() {
        latencies_us.sort_unstable();
        let n = latencies_us.len();
        let p50 = latencies_us[n / 2];
        let p95 = latencies_us[(n * 95) / 100];
        let p99 = latencies_us[(n * 99) / 100];
        result = result.with_counter("p50_us", p50);
        result = result.with_counter("p95_us", p95);
        result = result.with_counter("p99_us", p99);
    }

    black_box(checksum);
    result
}

benchmark_main!(|runner| {
    for &n_threads in &[1usize, 2, 4, 8, 16] {
        let workers = [ConcurrentWorker {
            name: "evict_worker",
            threads: n_threads,
            run: evict_worker_sequential,
        }];
        runner.concurrent_group::<EvictCtx>("buffer_pool/evict/sequential", |g| {
            g.sample_duration(Duration::from_millis(200))
                .throughput(Throughput::per_operation(1, "pages"))
                .bench(&format!("{n_threads}t"), &workers);
        });

        let workers = [ConcurrentWorker {
            name: "evict_worker",
            threads: n_threads,
            run: evict_worker_random,
        }];
        runner.concurrent_group::<EvictCtx>("buffer_pool/evict/random", |g| {
            g.sample_duration(Duration::from_millis(200))
                .throughput(Throughput::per_operation(1, "pages"))
                .bench(&format!("{n_threads}t"), &workers);
        });
    }
});
