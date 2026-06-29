use std::ops::Bound;
use std::sync::Arc;
use std::time::Duration;

use micromeasure::{
    BenchContext, ConcurrentBenchContext, ConcurrentBenchControl, ConcurrentWorker,
    ConcurrentWorkerResult, Throughput, benchmark_main, black_box,
};
use pagebox_btree::BTree;
use pagebox_storage::buffer_pool::BufferPool;

fn random_key(seed: u64) -> [u8; 8] {
    seed.wrapping_mul(0x517cc1b727220a95)
        .wrapping_add(1)
        .to_be_bytes()
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn counter_delta_u64(after: u64, before: u64) -> u64 {
    after.saturating_sub(before)
}

struct SequentialInsertCtx<const N: usize> {
    tree: BTree,
    next_key: u64,
    _pool: Arc<BufferPool>,
}

impl<const N: usize> BenchContext for SequentialInsertCtx<N> {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("sequential insert bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(N)
    }
}

fn sequential_insert<const N: usize>(
    ctx: &mut SequentialInsertCtx<N>,
    chunk_size: usize,
    _chunk_num: usize,
) {
    for _ in 0..chunk_size {
        let key = ctx.next_key.to_be_bytes();
        ctx.tree.insert(&key, &key);
        ctx.next_key += 1;
    }
}

struct RandomInsertCtx<const N: usize> {
    tree: BTree,
    keys: Arc<Vec<[u8; 8]>>,
    _pool: Arc<BufferPool>,
}

impl<const N: usize> BenchContext for RandomInsertCtx<N> {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("random insert bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(N)
    }
}

fn random_insert<const N: usize>(
    ctx: &mut RandomInsertCtx<N>,
    chunk_size: usize,
    _chunk_num: usize,
) {
    for key in ctx.keys.iter().take(chunk_size) {
        ctx.tree.insert(key, key);
    }
}

struct LookupSeed<const N: usize> {
    pool: Arc<BufferPool>,
    root_page_id: u64,
    height: u32,
    keys: Arc<Vec<[u8; 8]>>,
}

struct LookupCtx<const N: usize> {
    tree: BTree,
    keys: Arc<Vec<[u8; 8]>>,
    _pool: Arc<BufferPool>,
}

impl<const N: usize> BenchContext for LookupCtx<N> {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("lookup bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(N)
    }
}

fn random_lookup<const N: usize>(ctx: &mut LookupCtx<N>, chunk_size: usize, chunk_num: usize) {
    let start = (chunk_num * chunk_size) % ctx.keys.len();
    for i in 0..chunk_size {
        let key = &ctx.keys[(start + i) % ctx.keys.len()];
        black_box(ctx.tree.lookup(key));
    }
}

const RANGE_SCAN_SPAN: usize = 256;

struct RangeScanSeed<const N: usize> {
    pool: Arc<BufferPool>,
    root_page_id: u64,
    height: u32,
    max_start: u32,
}

struct RangeScanCtx<const N: usize> {
    tree: BTree,
    max_start: u32,
    _pool: Arc<BufferPool>,
}

impl<const N: usize> BenchContext for RangeScanCtx<N> {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("range scan bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(1_024)
    }
}

fn range_scan<const N: usize>(ctx: &mut RangeScanCtx<N>, chunk_size: usize, chunk_num: usize) {
    let base = (chunk_num * chunk_size) as u32;
    for i in 0..chunk_size as u32 {
        let start = if ctx.max_start == 0 {
            0
        } else {
            base.wrapping_add(i).wrapping_mul(97) % ctx.max_start
        };
        let end = start + RANGE_SCAN_SPAN as u32;
        let lower = start.to_be_bytes();
        let upper = end.to_be_bytes();
        let mut count = 0_u64;
        ctx.tree.scan_range(
            Bound::Included(lower.as_slice()),
            Bound::Excluded(upper.as_slice()),
            |_, _| count += 1,
        );
        black_box(count);
    }
}

struct MixedRwCtx {
    tree: BTree,
    next_temp_key: u64,
    _pool: Arc<BufferPool>,
}

impl BenchContext for MixedRwCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("mixed read/write bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(50_000)
    }
}

fn mixed_read_write(ctx: &mut MixedRwCtx, chunk_size: usize, _chunk_num: usize) {
    for i in 0..chunk_size {
        if i % 2 == 0 {
            let key = ((i / 2) as u32 % 10_000).to_be_bytes();
            black_box(ctx.tree.lookup(&key));
        } else {
            let key = ctx.next_temp_key.to_be_bytes();
            ctx.tree.insert(&key, &key);
            ctx.tree.remove(&key);
            ctx.next_temp_key += 1;
        }
    }
}

struct ConcurrentInsertBuildCtx {
    tree: Arc<BTree>,
    pool: Arc<BufferPool>,
}

impl ConcurrentBenchContext for ConcurrentInsertBuildCtx {
    fn prepare(num_threads: usize) -> Self {
        let _ = num_threads;
        let pool_frames = env_usize("PAGEBOX_BTREE_CONCURRENT_INSERT_BUILD_POOL_FRAMES", 16_384);
        let pool = Arc::new(BufferPool::new(pool_frames));
        let tree = Arc::new(BTree::new(&pool, 0));
        Self { pool, tree }
    }
}

struct ConcurrentInsertSteadyCtx {
    tree: Arc<BTree>,
    preseeded_keys: u64,
    pool: Arc<BufferPool>,
}

impl ConcurrentBenchContext for ConcurrentInsertSteadyCtx {
    fn prepare(num_threads: usize) -> Self {
        let _ = num_threads;
        let preseeded_keys = env_usize("PAGEBOX_BTREE_CONCURRENT_INSERT_STEADY_PRELOAD", 50_000);
        let pool_frames = env_usize("PAGEBOX_BTREE_CONCURRENT_INSERT_STEADY_POOL_FRAMES", 65_536);
        let pool = Arc::new(BufferPool::new(pool_frames));
        let tree = Arc::new(BTree::new(&pool, 0));
        for i in 0..preseeded_keys as u64 {
            let key = i.to_be_bytes();
            tree.insert(&key, &key);
        }
        Self {
            pool,
            tree,
            preseeded_keys: preseeded_keys as u64,
        }
    }
}

fn concurrent_insert_build_worker(
    ctx: &ConcurrentInsertBuildCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    let mut next_key =
        (control.thread_index() as u64) << 48 | ((control.role_thread_index() as u64) << 32);
    let evictions_before = if control.role_thread_index() == 0 {
        Some(ctx.pool.eviction_count())
    } else {
        None
    };

    while !control.should_stop() {
        let key = next_key.to_be_bytes();
        ctx.tree.insert(&key, &key);
        next_key = next_key.wrapping_add(1);
        operations = operations.wrapping_add(1);
    }

    let mut result = ConcurrentWorkerResult::operations(operations);
    if let Some(evictions_before) = evictions_before {
        result = result.with_counter(
            "evictions",
            counter_delta_u64(ctx.pool.eviction_count(), evictions_before),
        );
    }

    result
}

fn concurrent_insert_steady_worker(
    ctx: &ConcurrentInsertSteadyCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    let mut next_key = ctx.preseeded_keys
        + ((control.thread_index() as u64) << 48)
        + ((control.role_thread_index() as u64) << 32);
    let evictions_before = if control.role_thread_index() == 0 {
        Some(ctx.pool.eviction_count())
    } else {
        None
    };

    while !control.should_stop() {
        let key = next_key.to_be_bytes();
        ctx.tree.insert(&key, &key);
        next_key = next_key.wrapping_add(1);
        operations = operations.wrapping_add(1);
    }

    let mut result = ConcurrentWorkerResult::operations(operations);
    if let Some(evictions_before) = evictions_before {
        result = result.with_counter(
            "evictions",
            counter_delta_u64(ctx.pool.eviction_count(), evictions_before),
        );
    }

    result
}

benchmark_main!(|runner| {
    runner.group::<SequentialInsertCtx<1_000>>("btree/sequential_insert", |g| {
        g.throughput(Throughput::per_operation(1_000, "keys"))
            .factory(&|| {
                let pool = Arc::new(BufferPool::new(256));
                let tree = BTree::new(&pool, 0);
                SequentialInsertCtx::<1_000> {
                    _pool: pool,
                    tree,
                    next_key: 0,
                }
            })
            .bench("n1k", sequential_insert::<1_000>);
    });

    runner.group::<SequentialInsertCtx<10_000>>("btree/sequential_insert", |g| {
        g.throughput(Throughput::per_operation(10_000, "keys"))
            .factory(&|| {
                let pool = Arc::new(BufferPool::new(2_560));
                let tree = BTree::new(&pool, 0);
                SequentialInsertCtx::<10_000> {
                    _pool: pool,
                    tree,
                    next_key: 0,
                }
            })
            .bench("n10k", sequential_insert::<10_000>);
    });

    runner.group::<SequentialInsertCtx<100_000>>("btree/sequential_insert", |g| {
        g.throughput(Throughput::per_operation(100_000, "keys"))
            .factory(&|| {
                let pool = Arc::new(BufferPool::new(25_000));
                let tree = BTree::new(&pool, 0);
                SequentialInsertCtx::<100_000> {
                    _pool: pool,
                    tree,
                    next_key: 0,
                }
            })
            .bench("n100k", sequential_insert::<100_000>);
    });

    runner.group::<RandomInsertCtx<1_000>>("btree/random_insert", |g| {
        g.throughput(Throughput::per_operation(1_000, "keys"))
            .factory(&|| {
                let pool = Arc::new(BufferPool::new(256));
                let tree = BTree::new(&pool, 0);
                let keys = Arc::new((0..1_000).map(|i| random_key(i as u64)).collect());
                RandomInsertCtx::<1_000> {
                    _pool: pool,
                    tree,
                    keys,
                }
            })
            .bench("n1k", random_insert::<1_000>);
    });

    runner.group::<RandomInsertCtx<10_000>>("btree/random_insert", |g| {
        g.throughput(Throughput::per_operation(10_000, "keys"))
            .factory(&|| {
                let pool = Arc::new(BufferPool::new(2_560));
                let tree = BTree::new(&pool, 0);
                let keys = Arc::new((0..10_000).map(|i| random_key(i as u64)).collect());
                RandomInsertCtx::<10_000> {
                    _pool: pool,
                    tree,
                    keys,
                }
            })
            .bench("n10k", random_insert::<10_000>);
    });

    runner.group::<RandomInsertCtx<100_000>>("btree/random_insert", |g| {
        g.throughput(Throughput::per_operation(100_000, "keys"))
            .factory(&|| {
                let pool = Arc::new(BufferPool::new(25_000));
                let tree = BTree::new(&pool, 0);
                let keys = Arc::new((0..100_000).map(|i| random_key(i as u64)).collect());
                RandomInsertCtx::<100_000> {
                    _pool: pool,
                    tree,
                    keys,
                }
            })
            .bench("n100k", random_insert::<100_000>);
    });

    let lookup_seed_1k = Arc::new({
        let n = 1_000usize;
        let pool = Arc::new(BufferPool::new((n / 4).max(64)));
        let tree = BTree::new(&pool, 0);
        let keys: Vec<[u8; 8]> = (0..n).map(|i| random_key(i as u64)).collect();
        for key in &keys {
            tree.insert(key, key);
        }
        LookupSeed::<1_000> {
            pool,
            root_page_id: tree.root_page_id(),
            height: tree.height(),
            keys: Arc::new(keys),
        }
    });
    let lookup_seed_10k = Arc::new({
        let n = 10_000usize;
        let pool = Arc::new(BufferPool::new((n / 4).max(64)));
        let tree = BTree::new(&pool, 0);
        let keys: Vec<[u8; 8]> = (0..n).map(|i| random_key(i as u64)).collect();
        for key in &keys {
            tree.insert(key, key);
        }
        LookupSeed::<10_000> {
            pool,
            root_page_id: tree.root_page_id(),
            height: tree.height(),
            keys: Arc::new(keys),
        }
    });
    let lookup_seed_100k = Arc::new({
        let n = 100_000usize;
        let pool = Arc::new(BufferPool::new((n / 4).max(64)));
        let tree = BTree::new(&pool, 0);
        let keys: Vec<[u8; 8]> = (0..n).map(|i| random_key(i as u64)).collect();
        for key in &keys {
            tree.insert(key, key);
        }
        LookupSeed::<100_000> {
            pool,
            root_page_id: tree.root_page_id(),
            height: tree.height(),
            keys: Arc::new(keys),
        }
    });

    runner.group::<LookupCtx<1_000>>("btree/random_lookup", |g| {
        g.throughput(Throughput::per_operation(1_000, "keys"))
            .factory(&{
                let seed = Arc::clone(&lookup_seed_1k);
                move || LookupCtx::<1_000> {
                    _pool: Arc::clone(&seed.pool),
                    tree: BTree::open(&seed.pool, seed.root_page_id, seed.height, 0),
                    keys: Arc::clone(&seed.keys),
                }
            })
            .bench("n1k", random_lookup::<1_000>);
    });

    runner.group::<LookupCtx<10_000>>("btree/random_lookup", |g| {
        g.throughput(Throughput::per_operation(10_000, "keys"))
            .factory(&{
                let seed = Arc::clone(&lookup_seed_10k);
                move || LookupCtx::<10_000> {
                    _pool: Arc::clone(&seed.pool),
                    tree: BTree::open(&seed.pool, seed.root_page_id, seed.height, 0),
                    keys: Arc::clone(&seed.keys),
                }
            })
            .bench("n10k", random_lookup::<10_000>);
    });

    runner.group::<LookupCtx<100_000>>("btree/random_lookup", |g| {
        g.throughput(Throughput::per_operation(100_000, "keys"))
            .factory(&{
                let seed = Arc::clone(&lookup_seed_100k);
                move || LookupCtx::<100_000> {
                    _pool: Arc::clone(&seed.pool),
                    tree: BTree::open(&seed.pool, seed.root_page_id, seed.height, 0),
                    keys: Arc::clone(&seed.keys),
                }
            })
            .bench("n100k", random_lookup::<100_000>);
    });

    let range_scan_seed_1k = Arc::new({
        let n = 1_000usize;
        let pool = Arc::new(BufferPool::new((n / 4).max(64)));
        let tree = BTree::new(&pool, 0);
        for i in 0..n as u32 {
            let key = i.to_be_bytes();
            tree.insert(&key, &key);
        }
        RangeScanSeed::<1_000> {
            pool,
            root_page_id: tree.root_page_id(),
            height: tree.height(),
            max_start: (n.saturating_sub(RANGE_SCAN_SPAN)).max(1) as u32,
        }
    });
    let range_scan_seed_10k = Arc::new({
        let n = 10_000usize;
        let pool = Arc::new(BufferPool::new((n / 4).max(64)));
        let tree = BTree::new(&pool, 0);
        for i in 0..n as u32 {
            let key = i.to_be_bytes();
            tree.insert(&key, &key);
        }
        RangeScanSeed::<10_000> {
            pool,
            root_page_id: tree.root_page_id(),
            height: tree.height(),
            max_start: (n.saturating_sub(RANGE_SCAN_SPAN)).max(1) as u32,
        }
    });
    let range_scan_seed_100k = Arc::new({
        let n = 100_000usize;
        let pool = Arc::new(BufferPool::new((n / 4).max(64)));
        let tree = BTree::new(&pool, 0);
        for i in 0..n as u32 {
            let key = i.to_be_bytes();
            tree.insert(&key, &key);
        }
        RangeScanSeed::<100_000> {
            pool,
            root_page_id: tree.root_page_id(),
            height: tree.height(),
            max_start: (n.saturating_sub(RANGE_SCAN_SPAN)).max(1) as u32,
        }
    });

    runner.group::<RangeScanCtx<1_000>>("btree/range_scan", |g| {
        g.throughput(Throughput::per_operation(1_024, "ranges"))
            .factory(&{
                let seed = Arc::clone(&range_scan_seed_1k);
                move || RangeScanCtx::<1_000> {
                    _pool: Arc::clone(&seed.pool),
                    tree: BTree::open(&seed.pool, seed.root_page_id, seed.height, 0),
                    max_start: seed.max_start,
                }
            })
            .bench("n1k", range_scan::<1_000>);
    });

    runner.group::<RangeScanCtx<10_000>>("btree/range_scan", |g| {
        g.throughput(Throughput::per_operation(1_024, "ranges"))
            .factory(&{
                let seed = Arc::clone(&range_scan_seed_10k);
                move || RangeScanCtx::<10_000> {
                    _pool: Arc::clone(&seed.pool),
                    tree: BTree::open(&seed.pool, seed.root_page_id, seed.height, 0),
                    max_start: seed.max_start,
                }
            })
            .bench("n10k", range_scan::<10_000>);
    });

    runner.group::<RangeScanCtx<100_000>>("btree/range_scan", |g| {
        g.throughput(Throughput::per_operation(1_024, "ranges"))
            .factory(&{
                let seed = Arc::clone(&range_scan_seed_100k);
                move || RangeScanCtx::<100_000> {
                    _pool: Arc::clone(&seed.pool),
                    tree: BTree::open(&seed.pool, seed.root_page_id, seed.height, 0),
                    max_start: seed.max_start,
                }
            })
            .bench("n100k", range_scan::<100_000>);
    });

    runner.group::<MixedRwCtx>("btree/mixed_rw", |g| {
        g.throughput(Throughput::per_operation(50_000, "operations"))
            .factory(&|| {
                let pool = Arc::new(BufferPool::new(10_000));
                let tree = BTree::new(&pool, 0);
                for i in 0..10_000_u32 {
                    let key = i.to_be_bytes();
                    tree.insert(&key, &key);
                }
                MixedRwCtx {
                    _pool: pool,
                    tree,
                    next_temp_key: 10_000,
                }
            })
            .bench("fifty_fifty", mixed_read_write);
    });

    for &n_threads in &[1usize, 2, 4, 8] {
        let build_workers = [ConcurrentWorker {
            name: "insert_worker",
            threads: n_threads,
            run: concurrent_insert_build_worker,
        }];
        let steady_workers = [ConcurrentWorker {
            name: "insert_worker",
            threads: n_threads,
            run: concurrent_insert_steady_worker,
        }];

        runner.concurrent_group::<ConcurrentInsertBuildCtx>(
            "btree/concurrent_insert_build_empty",
            |g| {
                g.sample_duration(Duration::from_millis(100))
                    .throughput(Throughput::per_operation(1, "keys"))
                    .bench(&format!("{n_threads}t"), &build_workers);
            },
        );

        runner.concurrent_group::<ConcurrentInsertSteadyCtx>(
            "btree/concurrent_insert_steady_pre50k",
            |g| {
                g.sample_duration(Duration::from_millis(500))
                    .throughput(Throughput::per_operation(1, "keys"))
                    .bench(&format!("{n_threads}t"), &steady_workers);
            },
        );
    }
});
