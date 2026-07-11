use std::sync::Arc;

use micromeasure::{BenchContext, Throughput, benchmark_main, black_box};
use pagebox_btree::BTree;
use pagebox_storage::buffer_pool::BufferPool;

struct BtreeLookupSeed {
    pool: Arc<BufferPool>,
    root_page_id: u64,
    height: u32,
    hot_keys: Arc<Vec<[u8; 8]>>,
}

struct BtreeLookupCtx {
    tree: BTree,
    keys: Arc<Vec<[u8; 8]>>,
    _pool: Arc<BufferPool>,
}

impl BenchContext for BtreeLookupCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("btree lookup bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(1_000_000)
    }
}

fn lookup_hot(ctx: &mut BtreeLookupCtx, chunk_size: usize, chunk_num: usize) {
    let start = (chunk_num * chunk_size) % ctx.keys.len();
    for i in 0..chunk_size {
        let key = &ctx.keys[(start + i) % ctx.keys.len()];
        black_box(ctx.tree.lookup(key));
    }
}

struct BtreeInsertCtx {
    tree: BTree,
    next_key: u64,
    _pool: Arc<BufferPool>,
}

impl BenchContext for BtreeInsertCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("btree insert bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(100_000)
    }
}

fn insert_hot(ctx: &mut BtreeInsertCtx, chunk_size: usize, _chunk_num: usize) {
    for _ in 0..chunk_size {
        let key = ctx.next_key.to_be_bytes();
        ctx.tree.insert(&key, &key);
        ctx.next_key += 1;
    }
}

benchmark_main!(|runner| {
    let lookup_seed = {
        let n = 100_000usize;
        let hot_window = 64usize;
        let pool = Arc::new(BufferPool::new((n / 4).max(64)));
        let tree = BTree::new(&pool, 0);
        let keys: Vec<[u8; 8]> = (0..n)
            .map(|i| {
                let h = (i as u64).wrapping_mul(0x517cc1b727220a95).wrapping_add(1);
                h.to_be_bytes()
            })
            .collect();
        for key in &keys {
            tree.insert(key, key);
        }
        Arc::new(BtreeLookupSeed {
            pool,
            root_page_id: tree.root_page_id(),
            height: tree.height(),
            hot_keys: Arc::new(keys[..hot_window].to_vec()),
        })
    };

    runner.group::<BtreeLookupCtx>("btree_lookup", |g| {
        g.throughput(Throughput::per_operation(1, "keys"))
            .factory(&{
                let seed = Arc::clone(&lookup_seed);
                move || BtreeLookupCtx {
                    _pool: Arc::clone(&seed.pool),
                    tree: BTree::open(&seed.pool, seed.root_page_id, seed.height, 0),
                    keys: Arc::clone(&seed.hot_keys),
                }
            })
            .bench("lookup_hot", lookup_hot);
    });

    runner.group::<BtreeInsertCtx>("btree_insert", |g| {
        g.throughput(Throughput::per_operation(1, "keys"))
            .factory(&|| {
                let pool = Arc::new(BufferPool::new(4096));
                let tree = BTree::new(&pool, 0);
                let seeded = 100_000u64;
                for i in 0..seeded {
                    let key = i.to_be_bytes();
                    tree.insert(&key, &key);
                }
                BtreeInsertCtx {
                    _pool: pool,
                    tree,
                    next_key: seeded,
                }
            })
            .bench("insert_hot", insert_hot);
    });
});
