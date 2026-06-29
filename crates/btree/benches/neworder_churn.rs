use std::sync::Arc;

use micromeasure::{BenchContext, Throughput, benchmark_main, black_box};
use pagebox_btree::BTree;
use pagebox_storage::buffer_pool::BufferPool;

const PREFIX: [u8; 4] = [0, 1, 0, 1];

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn key_for(order_id: u64) -> [u8; 12] {
    let mut key = [0_u8; 12];
    key[..4].copy_from_slice(&PREFIX);
    key[4..].copy_from_slice(&order_id.to_be_bytes());
    key
}

fn first_order_id(tree: &BTree) -> Option<u64> {
    let mut found = None;
    tree.scan_prefix_borrowed_until(&PREFIX, |key, _| {
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(&key[4..12]);
        found = Some(u64::from_be_bytes(bytes));
        false
    });
    found
}

struct AppendOnlyCtx {
    tree: BTree,
    next_order_id: u64,
    _pool: Arc<BufferPool>,
}

impl BenchContext for AppendOnlyCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("append-only bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(100_000)
    }
}

fn append_only(ctx: &mut AppendOnlyCtx, chunk_size: usize, _chunk_num: usize) {
    for _ in 0..chunk_size {
        let key = key_for(ctx.next_order_id);
        ctx.tree.insert(&key, &ctx.next_order_id.to_be_bytes());
        ctx.next_order_id += 1;
    }
}

struct FindOldestCtx {
    tree: BTree,
    _pool: Arc<BufferPool>,
}

impl BenchContext for FindOldestCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("find-oldest bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(1_000_000)
    }
}

fn find_oldest(ctx: &mut FindOldestCtx, chunk_size: usize, _chunk_num: usize) {
    for _ in 0..chunk_size {
        black_box(first_order_id(&ctx.tree));
    }
}

struct ChurnCtx {
    tree: BTree,
    next_order_id: u64,
    oldest_order_id: u64,
    _pool: Arc<BufferPool>,
}

impl BenchContext for ChurnCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("churn bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(100_000)
    }
}

fn delete_oldest_then_append(ctx: &mut ChurnCtx, chunk_size: usize, _chunk_num: usize) {
    for _ in 0..chunk_size {
        let found = first_order_id(&ctx.tree).expect("preloaded prefix should stay non-empty");
        assert_eq!(
            found, ctx.oldest_order_id,
            "prefix scan should return the oldest key"
        );
        black_box(found);
        let oldest_key = key_for(found);
        let removed = ctx.tree.remove(&oldest_key);
        black_box(removed);
        assert!(removed, "oldest key should be present");
        ctx.oldest_order_id = found + 1;

        let new_key = key_for(ctx.next_order_id);
        ctx.tree.insert(&new_key, &ctx.next_order_id.to_be_bytes());
        ctx.next_order_id += 1;
    }
}

benchmark_main!(|runner| {
    let preload = env_usize("PAGEBOX_BTREE_NEW_ORDER_PRELOAD", 50_000);
    let pool_frames = env_usize("PAGEBOX_BTREE_NEW_ORDER_POOL_FRAMES", 65_536);

    runner.group::<AppendOnlyCtx>("new_order_pk", |g| {
        g.throughput(Throughput::per_operation(100_000, "orders"))
            .factory(&move || {
                let pool = Arc::new(BufferPool::new(pool_frames));
                let tree = BTree::new(&pool, 0);
                for order_id in 1..=preload as u64 {
                    let key = key_for(order_id);
                    tree.insert(&key, &order_id.to_be_bytes());
                }
                AppendOnlyCtx {
                    _pool: pool,
                    tree,
                    next_order_id: preload as u64 + 1,
                }
            })
            .bench("append_monotonic", append_only);
    });

    runner.group::<FindOldestCtx>("new_order_pk", |g| {
        g.throughput(Throughput::per_operation(1_000_000, "lookups"))
            .factory(&move || {
                let pool = Arc::new(BufferPool::new(pool_frames));
                let tree = BTree::new(&pool, 0);
                for order_id in 1..=preload as u64 {
                    let key = key_for(order_id);
                    tree.insert(&key, &order_id.to_be_bytes());
                }
                FindOldestCtx { _pool: pool, tree }
            })
            .bench("find_oldest_prefix", find_oldest);
    });

    runner.group::<ChurnCtx>("new_order_pk", |g| {
        g.throughput(Throughput::per_operation(100_000, "orders"))
            .factory(&move || {
                let pool = Arc::new(BufferPool::new(pool_frames));
                let tree = BTree::new(&pool, 0);
                for order_id in 1..=preload as u64 {
                    let key = key_for(order_id);
                    tree.insert(&key, &order_id.to_be_bytes());
                }
                ChurnCtx {
                    _pool: pool,
                    tree,
                    next_order_id: preload as u64 + 1,
                    oldest_order_id: 1,
                }
            })
            .bench("delete_oldest_then_append", delete_oldest_then_append);
    });
});
