#![allow(
    unused_unsafe,
    reason = "NoLatches construction is always explicit at benchmark call sites"
)]

use std::sync::Arc;
use std::time::Duration;

use micromeasure::{
    BenchContext, ConcurrentBenchContext, ConcurrentBenchControl, ConcurrentWorker,
    ConcurrentWorkerResult, Throughput, benchmark_main, black_box,
};
use pagebox_storage::buffer_frame::StableSwip;
use pagebox_storage::buffer_pool::{BufferPool, NoLatches};

const OPS_PER_CHUNK: usize = 10_000;

struct StableFixCtx {
    pool: Arc<BufferPool>,
    edge: StableSwip,
}

impl StableFixCtx {
    fn resident() -> Self {
        let pool = Arc::new(BufferPool::new(1));
        let edge = pool.allocate_page();
        drop(pool.fix_stable(&edge, unsafe { NoLatches::new(&pool) }));
        Self { pool, edge }
    }
}

impl BenchContext for StableFixCtx {
    fn prepare(_num_chunks: usize) -> Self {
        Self::resident()
    }

    fn chunk_size() -> Option<usize> {
        Some(OPS_PER_CHUNK)
    }
}

impl ConcurrentBenchContext for StableFixCtx {
    fn prepare(_num_threads: usize) -> Self {
        Self::resident()
    }
}

fn hot_fix(ctx: &mut StableFixCtx, chunk_size: usize, _chunk_num: usize) {
    for _ in 0..chunk_size {
        let frame = ctx
            .pool
            .fix_stable(&ctx.edge, unsafe { NoLatches::new(&ctx.pool) });
        black_box(frame.pid());
    }
}

fn hot_try_fix(ctx: &mut StableFixCtx, chunk_size: usize, _chunk_num: usize) {
    for _ in 0..chunk_size {
        let frame = ctx
            .pool
            .try_fix_stable(&ctx.edge)
            .expect("benchmark stable edge must remain resident");
        black_box(frame.pid());
    }
}

fn concurrent_hot_fix(
    ctx: &StableFixCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0u64;
    while !control.should_stop() {
        let frame = ctx
            .pool
            .fix_stable(&ctx.edge, unsafe { NoLatches::new(&ctx.pool) });
        black_box(frame.pid());
        operations = operations.wrapping_add(1);
    }
    ConcurrentWorkerResult::operations(operations)
}

benchmark_main!(|runner| {
    runner.group::<StableFixCtx>("stable_swip/resident", |g| {
        g.throughput(Throughput::per_operation(1, "fixes"))
            .bench("fix", hot_fix);
        g.throughput(Throughput::per_operation(1, "fixes"))
            .bench("try_fix", hot_try_fix);
    });

    for &threads in &[1usize, 2, 4, 8] {
        let workers = [ConcurrentWorker {
            name: "fix",
            threads,
            run: concurrent_hot_fix,
        }];
        runner.concurrent_group::<StableFixCtx>("stable_swip/concurrent", |g| {
            g.sample_duration(Duration::from_millis(100))
                .throughput(Throughput::per_operation(1, "fixes"))
                .bench(&format!("{threads}t"), &workers);
        });
    }
});
