use std::sync::Arc;

use micromeasure::{BenchContext, Throughput, benchmark_main, black_box};

use pagebox_betree::{CowBeTree, CowBeTreeConfig, CowBeTreeGcCursor};
use pagebox_storage::buffer_pool::BufferPool;

const LARGE_PAGE_POOL_PAGES: usize = 32;
const STRUCTURAL_POOL_PAGES: usize = 96;
const LOOKUP_SEED_KEYS: usize = 4_096;
const HOT_WINDOW: usize = 64;
const GC_KEYS: usize = 64;
const GC_VERSIONS_PER_PASS: u64 = 4;

fn bench_config() -> CowBeTreeConfig {
    CowBeTreeConfig {
        flush_threshold_messages: 1024,
        flush_threshold_bytes: 128 * 1024,
        ..CowBeTreeConfig::default()
    }
}

fn structural_config() -> CowBeTreeConfig {
    CowBeTreeConfig {
        flush_threshold_messages: 32,
        flush_threshold_bytes: 16 * 1024,
        max_leaf_entries: 256,
        max_internal_children: 64,
        merge_leaf_entries: 512,
        merge_internal_children: 128,
    }
}

fn direct_flush_config() -> CowBeTreeConfig {
    CowBeTreeConfig {
        flush_threshold_messages: 8,
        flush_threshold_bytes: 1024,
        max_leaf_entries: 64,
        max_internal_children: 8,
        merge_leaf_entries: 128,
        merge_internal_children: 16,
    }
}

fn large_pool(pages: usize) -> Arc<BufferPool> {
    Arc::new(BufferPool::new(pages))
}

fn key(n: u64) -> [u8; 8] {
    n.to_be_bytes()
}

fn hashed_key(n: u64) -> [u8; 8] {
    n.wrapping_mul(0x517cc1b727220a95)
        .wrapping_add(1)
        .to_be_bytes()
}

fn seed_internal_root(tree: &CowBeTree, count: u64) {
    for i in 0..count {
        let key = key(i);
        tree.put(&key, i, &key).unwrap();
    }
    tree.flush_all().unwrap();
}

struct LookupSeed {
    pool: Arc<BufferPool>,
    root_page_id: u64,
    hot_keys: Arc<Vec<[u8; 8]>>,
}

struct LookupCtx {
    tree: CowBeTree,
    keys: Arc<Vec<[u8; 8]>>,
    _pool: Arc<BufferPool>,
}

impl BenchContext for LookupCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("lookup bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(10_000)
    }
}

fn lookup_hot(ctx: &mut LookupCtx, chunk_size: usize, chunk_num: usize) {
    let start = (chunk_num * chunk_size) % ctx.keys.len();
    for i in 0..chunk_size {
        let key = &ctx.keys[(start + i) % ctx.keys.len()];
        black_box(ctx.tree.lookup(key));
    }
}

struct InsertCtx {
    tree: CowBeTree,
    next_key: u64,
    _pool: Arc<BufferPool>,
}

impl BenchContext for InsertCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("insert bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(512)
    }
}

fn insert_hot(ctx: &mut InsertCtx, chunk_size: usize, _chunk_num: usize) {
    for _ in 0..chunk_size {
        let key = key(ctx.next_key);
        ctx.tree.put(&key, ctx.next_key, &key).unwrap();
        ctx.next_key += 1;
    }
}

struct PathLookupCtx {
    tree: CowBeTree,
    keys: Arc<Vec<[u8; 8]>>,
    _pool: Arc<BufferPool>,
}

impl BenchContext for PathLookupCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("path lookup bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(10_000)
    }
}

fn lookup_with_path_buffer(ctx: &mut PathLookupCtx, chunk_size: usize, chunk_num: usize) {
    let start = (chunk_num * chunk_size) % ctx.keys.len();
    for i in 0..chunk_size {
        let key = &ctx.keys[(start + i) % ctx.keys.len()];
        black_box(ctx.tree.lookup(key));
    }
}

struct BufferedAppendCtx {
    tree: CowBeTree,
    next_key: u64,
    _pool: Arc<BufferPool>,
}

impl BenchContext for BufferedAppendCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("buffer append bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(512)
    }
}

fn append_to_root_buffer(ctx: &mut BufferedAppendCtx, chunk_size: usize, _chunk_num: usize) {
    for _ in 0..chunk_size {
        let key = key(ctx.next_key);
        ctx.tree.put(&key, ctx.next_key, &key).unwrap();
        ctx.next_key += 1;
    }
}

struct FlushCtx {
    tree: CowBeTree,
    next_key: u64,
    _pool: Arc<BufferPool>,
}

impl BenchContext for FlushCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("flush bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(512)
    }
}

fn threshold_flush_partitioning(ctx: &mut FlushCtx, chunk_size: usize, _chunk_num: usize) {
    for _ in 0..chunk_size {
        let key = key(ctx.next_key);
        ctx.tree.put(&key, ctx.next_key, &key).unwrap();
        ctx.next_key += 1;
    }
}

struct DirectLeafFlushCtx {
    tree: CowBeTree,
    next_offset: usize,
    _pool: Arc<BufferPool>,
}

impl BenchContext for DirectLeafFlushCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("direct leaf flush bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(64)
    }
}

fn direct_leaf_flush_hot_keyset(
    ctx: &mut DirectLeafFlushCtx,
    chunk_size: usize,
    _chunk_num: usize,
) {
    for _ in 0..chunk_size {
        let key = key(512 + (ctx.next_offset % 8) as u64);
        ctx.tree.put(&key, 2_000, &key).unwrap();
        ctx.next_offset += 1;
    }
}

struct SplitCtx {
    tree: CowBeTree,
    next_key: u64,
    _pool: Arc<BufferPool>,
}

impl BenchContext for SplitCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("split bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(256)
    }
}

fn split_rebuild(ctx: &mut SplitCtx, chunk_size: usize, _chunk_num: usize) {
    for _ in 0..chunk_size {
        let key = key(ctx.next_key);
        ctx.tree.put(&key, ctx.next_key, &key).unwrap();
        ctx.next_key += 1;
    }
    ctx.tree.flush_all().unwrap();
}

struct CompactCtx {
    tree: CowBeTree,
    _pool: Arc<BufferPool>,
}

impl BenchContext for CompactCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("compact bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn compact_merge(ctx: &mut CompactCtx, _chunk_size: usize, _chunk_num: usize) {
    ctx.tree.compact().unwrap();
    black_box(ctx.tree.debug_snapshot());
}

struct GcCtx {
    tree: CowBeTree,
    keys: Arc<Vec<[u8; 8]>>,
    cursor: CowBeTreeGcCursor,
    next_ts: u64,
    _pool: Arc<BufferPool>,
}

impl BenchContext for GcCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("GC bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn gc_prune_hot_versions(ctx: &mut GcCtx, chunk_size: usize, _chunk_num: usize) {
    for _ in 0..chunk_size {
        let base_ts = ctx.next_ts;
        for key in ctx.keys.iter() {
            for offset in 0..GC_VERSIONS_PER_PASS {
                let ts = base_ts + offset;
                ctx.tree.put(key, ts, &ts.to_be_bytes()).unwrap();
            }
        }
        let result = ctx
            .tree
            .prune_versions(base_ts + GC_VERSIONS_PER_PASS - 1)
            .unwrap();
        black_box(result.versions_pruned);
        ctx.next_ts += GC_VERSIONS_PER_PASS;
    }
}

fn gc_prune_hot_versions_incremental(ctx: &mut GcCtx, chunk_size: usize, _chunk_num: usize) {
    for _ in 0..chunk_size {
        let base_ts = ctx.next_ts;
        for key in ctx.keys.iter() {
            for offset in 0..GC_VERSIONS_PER_PASS {
                let ts = base_ts + offset;
                ctx.tree.put(key, ts, &ts.to_be_bytes()).unwrap();
            }
        }
        let result = ctx
            .tree
            .prune_versions_incremental(base_ts + GC_VERSIONS_PER_PASS - 1, &mut ctx.cursor, 4)
            .unwrap();
        black_box(result.versions_pruned);
        ctx.next_ts += GC_VERSIONS_PER_PASS;
    }
}

benchmark_main!(|runner| {
    let lookup_seed = {
        let pool = large_pool(LARGE_PAGE_POOL_PAGES);
        let tree = CowBeTree::with_config(&pool, bench_config());
        let keys = (0..LOOKUP_SEED_KEYS)
            .map(|i| hashed_key(i as u64))
            .collect::<Vec<_>>();
        for key in &keys {
            tree.put(key, u64::MAX, key).unwrap();
        }
        tree.flush_all().unwrap();
        Arc::new(LookupSeed {
            pool,
            root_page_id: tree.root_page_id(),
            hot_keys: Arc::new(keys[..HOT_WINDOW].to_vec()),
        })
    };

    runner.group::<LookupCtx>("cow_betree_lookup", |g| {
        g.throughput(Throughput::per_operation(1, "lookup"))
            .factory(&{
                let seed = Arc::clone(&lookup_seed);
                move || LookupCtx {
                    _pool: Arc::clone(&seed.pool),
                    tree: CowBeTree::open_with_config(
                        &seed.pool,
                        seed.root_page_id,
                        bench_config(),
                    ),
                    keys: Arc::clone(&seed.hot_keys),
                }
            })
            .bench("lookup_hot", lookup_hot);
    });

    runner.group::<InsertCtx>("cow_betree_insert", |g| {
        g.throughput(Throughput::per_operation(1, "insert"))
            .factory(&|| {
                let pool = large_pool(LARGE_PAGE_POOL_PAGES);
                let tree = CowBeTree::with_config(&pool, bench_config());
                InsertCtx {
                    _pool: pool,
                    tree,
                    next_key: 0,
                }
            })
            .bench("insert_hot", insert_hot);
    });

    runner.group::<PathLookupCtx>("cow_betree_path_lookup", |g| {
        g.throughput(Throughput::per_operation(1, "lookup"))
            .factory(&|| {
                let pool = large_pool(STRUCTURAL_POOL_PAGES);
                let tree = CowBeTree::with_config(&pool, structural_config());
                seed_internal_root(&tree, 512);
                let keys = (1_000u64..1_064).map(key).collect::<Vec<_>>();
                for key in &keys {
                    tree.put(key, 2_000, b"buffered").unwrap();
                }
                PathLookupCtx {
                    _pool: pool,
                    tree,
                    keys: Arc::new(keys),
                }
            })
            .bench("lookup_with_path_buffer", lookup_with_path_buffer);
    });

    runner.group::<BufferedAppendCtx>("cow_betree_buffer_append", |g| {
        g.throughput(Throughput::per_operation(1, "message"))
            .factory(&|| {
                let pool = large_pool(STRUCTURAL_POOL_PAGES);
                let config = CowBeTreeConfig {
                    flush_threshold_messages: 256,
                    flush_threshold_bytes: 8 * 1024,
                    ..structural_config()
                };
                let tree = CowBeTree::with_config(&pool, config);
                seed_internal_root(&tree, 512);
                BufferedAppendCtx {
                    _pool: pool,
                    tree,
                    next_key: 10_000,
                }
            })
            .bench("append_to_root_buffer", append_to_root_buffer);
    });

    runner.group::<FlushCtx>("cow_betree_flush", |g| {
        g.throughput(Throughput::per_operation(1, "message"))
            .factory(&|| {
                let pool = large_pool(STRUCTURAL_POOL_PAGES);
                let config = CowBeTreeConfig {
                    flush_threshold_messages: 16,
                    flush_threshold_bytes: 1024,
                    ..structural_config()
                };
                let tree = CowBeTree::with_config(&pool, config);
                seed_internal_root(&tree, 512);
                FlushCtx {
                    _pool: pool,
                    tree,
                    next_key: 20_000,
                }
            })
            .bench("threshold_flush_partitioning", threshold_flush_partitioning);
    });

    runner.group::<DirectLeafFlushCtx>("cow_betree_direct_flush", |g| {
        g.throughput(Throughput::per_operation(1, "message"))
            .factory(&|| {
                let pool = large_pool(STRUCTURAL_POOL_PAGES);
                let config = direct_flush_config();
                let tree = CowBeTree::with_config(&pool, config);
                seed_internal_root(&tree, 1024);
                debug_assert!(
                    tree.height() >= 2,
                    "direct flush bench should route through an internal child"
                );
                DirectLeafFlushCtx {
                    _pool: pool,
                    tree,
                    next_offset: 0,
                }
            })
            .bench("direct_leaf_flush_hot_keyset", direct_leaf_flush_hot_keyset);
    });

    runner.group::<SplitCtx>("cow_betree_split", |g| {
        g.throughput(Throughput::per_operation(1, "message"))
            .factory(&|| {
                let pool = large_pool(STRUCTURAL_POOL_PAGES);
                let config = CowBeTreeConfig {
                    flush_threshold_messages: 8,
                    max_leaf_entries: 64,
                    max_internal_children: 16,
                    ..structural_config()
                };
                SplitCtx {
                    _pool: Arc::clone(&pool),
                    tree: CowBeTree::with_config(&pool, config),
                    next_key: 0,
                }
            })
            .bench("split_rebuild", split_rebuild);
    });

    runner.group::<CompactCtx>("cow_betree_compact", |g| {
        g.throughput(Throughput::per_operation(1, "compact"))
            .factory(&|| {
                let pool = large_pool(STRUCTURAL_POOL_PAGES);
                let tree = CowBeTree::with_config(&pool, structural_config());
                seed_internal_root(&tree, 384);
                CompactCtx { _pool: pool, tree }
            })
            .bench("compact_merge", compact_merge);
    });

    runner.group::<GcCtx>("cow_betree_gc", |g| {
        g.throughput(Throughput::per_operation(
            GC_KEYS as u64 * GC_VERSIONS_PER_PASS,
            "version",
        ))
        .factory(&|| {
            let pool = large_pool(STRUCTURAL_POOL_PAGES);
            let tree = CowBeTree::with_config(&pool, structural_config());
            let keys: Arc<Vec<[u8; 8]>> =
                Arc::new((0..GC_KEYS).map(|idx| key(50_000 + idx as u64)).collect());
            for key in keys.iter() {
                tree.put(key, 1, &1u64.to_be_bytes()).unwrap();
            }
            GcCtx {
                _pool: pool,
                tree,
                keys,
                cursor: CowBeTreeGcCursor::default(),
                next_ts: 2,
            }
        })
        .bench("prune_hot_versions", gc_prune_hot_versions);
    });

    runner.group::<GcCtx>("cow_betree_gc_incremental", |g| {
        g.throughput(Throughput::per_operation(1, "step"))
            .factory(&|| {
                let pool = large_pool(STRUCTURAL_POOL_PAGES);
                let tree = CowBeTree::with_config(&pool, structural_config());
                let keys: Arc<Vec<[u8; 8]>> =
                    Arc::new((0..GC_KEYS).map(|idx| key(60_000 + idx as u64)).collect());
                for key in keys.iter() {
                    tree.put(key, 1, &1u64.to_be_bytes()).unwrap();
                }
                GcCtx {
                    _pool: pool,
                    tree,
                    keys,
                    cursor: CowBeTreeGcCursor::default(),
                    next_ts: 2,
                }
            })
            .bench(
                "prune_hot_versions_four_leaf_budget",
                gc_prune_hot_versions_incremental,
            );
    });
});
