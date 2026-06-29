use std::env;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use micromeasure::{
    ConcurrentBenchContext, ConcurrentBenchControl, ConcurrentWorker, ConcurrentWorkerResult,
    Throughput, benchmark_main, black_box,
};
use pagebox_storage::buffer_frame::{PAGE_SIZE, PageClass};
use pagebox_storage::buffer_pool::BufferPool;
use pagebox_storage::page_store::{FilePageStore, PageStore};

#[repr(align(4096))]
struct AlignedPage([u8; PAGE_SIZE]);

#[derive(Clone, Copy)]
struct FaultBenchConfig {
    num_pages: usize,
    pool_frames: usize,
    pages_per_thread: usize,
    drop_cache: bool,
}

impl FaultBenchConfig {
    fn from_env() -> Self {
        Self {
            num_pages: env_usize("PAGEBOX_BP_FAULT_PAGES", 100_000),
            pool_frames: env_usize("PAGEBOX_BP_FAULT_POOL", 1_024),
            pages_per_thread: env_usize("PAGEBOX_BP_FAULT_PAGES_PER_THREAD", 2_000),
            drop_cache: env_bool("PAGEBOX_BP_FAULT_DROP_CACHE", true),
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(default)
}

fn shuffled(n: usize) -> Vec<u64> {
    let mut v: Vec<u64> = (1..=n as u64)
        .map(|page_number| PageClass::Size4K.encode_page_id(page_number))
        .collect();
    for i in (1..v.len()).rev() {
        let h = (i as u64)
            .wrapping_mul(0x517c_c1b7_2722_0a95)
            .wrapping_add(1);
        let j = (h as usize) % (i + 1);
        v.swap(i, j);
    }
    v
}

fn populate_store(config: FaultBenchConfig) -> tempfile::TempDir {
    assert!(
        config.num_pages > 0,
        "PAGEBOX_BP_FAULT_PAGES must be greater than zero"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data");
    let store = FilePageStore::open(&path).expect("open page store");
    let max_pid = PageClass::Size4K.encode_page_id(config.num_pages as u64);
    store.allocate(max_pid).expect("allocate benchmark pages");

    let mut page = AlignedPage([0u8; PAGE_SIZE]);
    for page_number in 1..=config.num_pages as u64 {
        let pid = PageClass::Size4K.encode_page_id(page_number);
        page.0[0..8].copy_from_slice(&pid.to_le_bytes());
        page.0[8..16].copy_from_slice(&page_number.to_le_bytes());
        store
            .write_page(pid, &page.0)
            .expect("write benchmark page");
    }

    store.sync().expect("sync benchmark page store");
    if config.drop_cache {
        store.drop_cache();
    }
    drop(store);
    dir
}

fn open_sync_pool(dir: &std::path::Path, config: FaultBenchConfig) -> BufferPool {
    let store = FilePageStore::open(&dir.join("data")).expect("open benchmark page store");
    if config.drop_cache {
        store.drop_cache();
    }
    BufferPool::with_store(config.pool_frames, Box::new(store))
}

struct SyncFaultCtx {
    _dir: tempfile::TempDir,
    pool: Arc<BufferPool>,
    eviction_checkpoint: Mutex<u64>,
    thread_pages: Vec<Vec<u64>>,
}

impl ConcurrentBenchContext for SyncFaultCtx {
    fn prepare(num_threads: usize) -> Self {
        let config = FaultBenchConfig::from_env();
        let dir = populate_store(config);
        let pool = Arc::new(open_sync_pool(dir.path(), config));
        let thread_pages = pages_by_thread(config, num_threads);
        Self {
            _dir: dir,
            eviction_checkpoint: Mutex::new(pool.eviction_count()),
            pool,
            thread_pages,
        }
    }
}

fn pages_by_thread(config: FaultBenchConfig, num_threads: usize) -> Vec<Vec<u64>> {
    let order = shuffled(config.num_pages);
    assert!(
        !order.is_empty(),
        "PAGEBOX_BP_FAULT_PAGES must be greater than zero"
    );
    assert!(
        config.pages_per_thread > 0,
        "PAGEBOX_BP_FAULT_PAGES_PER_THREAD must be greater than zero"
    );
    (0..num_threads)
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
        .collect()
}

fn fault_worker(
    pool: &BufferPool,
    eviction_checkpoint: &Mutex<u64>,
    pages: &[u64],
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    let mut checksum = 0_u64;

    while !control.should_stop() {
        for &pid in pages {
            let frame = unsafe { pool.fix_orphan_frame(pid) };
            checksum ^= u64::from_le_bytes(frame.page[0..8].try_into().expect("pid bytes"));
            operations = operations.wrapping_add(1);
            drop(frame);
            if control.should_stop() {
                break;
            }
        }
    }

    let mut result =
        ConcurrentWorkerResult::operations(operations).with_counter("page_ops", operations);
    if control.role_thread_index() == 0 {
        let evictions = pool.eviction_count();
        let mut checkpoint = eviction_checkpoint
            .lock()
            .expect("eviction checkpoint poisoned");
        let previous = *checkpoint;
        *checkpoint = evictions;
        result = result.with_counter("evictions", counter_delta(evictions, previous));
    }
    black_box(checksum);
    result
}

fn counter_delta(current: u64, previous: u64) -> u64 {
    current.saturating_sub(previous)
}

fn sync_fault_worker(
    ctx: &SyncFaultCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    fault_worker(
        &ctx.pool,
        &ctx.eviction_checkpoint,
        &ctx.thread_pages[control.role_thread_index()],
        control,
    )
}

benchmark_main!(|runner| {
    for &n_threads in &[1usize, 2, 4, 8, 16] {
        let workers = [ConcurrentWorker {
            name: "fault_worker",
            threads: n_threads,
            run: sync_fault_worker,
        }];
        runner.concurrent_group::<SyncFaultCtx>("buffer_pool/sync/random_orphan_fault", |g| {
            g.sample_duration(Duration::from_millis(100))
                .throughput(Throughput::per_operation(1, "pages"))
                .bench(&format!("{n_threads}t"), &workers);
        });
    }
});
