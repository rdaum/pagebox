use std::sync::Arc;
use std::time::Duration;

use micromeasure::{
    ConcurrentBenchContext, ConcurrentBenchControl, ConcurrentWorker, ConcurrentWorkerResult,
    Throughput, benchmark_main, black_box,
};
use pagebox_btree::BTree;
use pagebox_storage::buffer_pool::BufferPool;

const N_RECORDS: usize = 10_000;

fn hash_key(seed: u64) -> [u8; 8] {
    seed.wrapping_mul(0x517cc1b727220a95)
        .wrapping_add(0x9e3779b97f4a7c15)
        .to_be_bytes()
}

struct YcsbCtx<const READ_PCT: u32> {
    tree: Arc<BTree>,
    _pool: Arc<BufferPool>,
}

impl<const READ_PCT: u32> ConcurrentBenchContext for YcsbCtx<READ_PCT> {
    fn prepare(num_threads: usize) -> Self {
        let pool = Arc::new(BufferPool::new(N_RECORDS.max(num_threads * 1024)));
        let tree = Arc::new(BTree::new(&pool, 0));
        let value = [0xAA_u8; 100];

        for i in 0..N_RECORDS {
            let key = hash_key(i as u64);
            tree.insert(&key, &value);
        }

        Self { _pool: pool, tree }
    }
}

fn ycsb_worker<const READ_PCT: u32>(
    ctx: &YcsbCtx<READ_PCT>,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    let mut reads = 0_u64;
    let mut updates = 0_u64;
    let update_value = [0xBB_u8; 100];
    let thread_seed = ((control.thread_index() as u64) << 32) | control.role_thread_index() as u64;

    while !control.should_stop() {
        let op_id = thread_seed.wrapping_add(operations);
        let is_read = (op_id.wrapping_mul(2_654_435_761) % 100) < READ_PCT as u64;
        let key_idx = op_id % N_RECORDS as u64;
        let key = hash_key(key_idx);

        if is_read {
            black_box(ctx.tree.lookup(&key));
            reads = reads.wrapping_add(1);
        } else {
            ctx.tree.remove(&key);
            ctx.tree.insert(&key, &update_value);
            updates = updates.wrapping_add(1);
        }

        operations = operations.wrapping_add(1);
    }

    ConcurrentWorkerResult::operations(operations)
        .with_counter("reads", reads)
        .with_counter("updates", updates)
}

benchmark_main!(|runner| {
    for &n_threads in &[1usize, 2, 4] {
        let workers_a = [ConcurrentWorker {
            name: "ycsb_worker",
            threads: n_threads,
            run: ycsb_worker::<50>,
        }];
        let workers_b = [ConcurrentWorker {
            name: "ycsb_worker",
            threads: n_threads,
            run: ycsb_worker::<95>,
        }];
        let workers_c = [ConcurrentWorker {
            name: "ycsb_worker",
            threads: n_threads,
            run: ycsb_worker::<100>,
        }];

        runner.concurrent_group::<YcsbCtx<50>>("ycsb/A_50r50w", |g| {
            g.sample_duration(Duration::from_millis(100))
                .throughput(Throughput::per_operation(1, "operations"))
                .bench(&format!("{n_threads}t"), &workers_a);
        });
        runner.concurrent_group::<YcsbCtx<95>>("ycsb/B_95r5w", |g| {
            g.sample_duration(Duration::from_millis(100))
                .throughput(Throughput::per_operation(1, "operations"))
                .bench(&format!("{n_threads}t"), &workers_b);
        });
        runner.concurrent_group::<YcsbCtx<100>>("ycsb/C_100r", |g| {
            g.sample_duration(Duration::from_millis(100))
                .throughput(Throughput::per_operation(1, "operations"))
                .bench(&format!("{n_threads}t"), &workers_c);
        });
    }
});
