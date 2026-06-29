use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use micromeasure::{
    BenchContext, ConcurrentBenchContext, ConcurrentBenchControl, ConcurrentWorker,
    ConcurrentWorkerResult, Throughput, benchmark_main, black_box,
};
use pagebox_storage::buffer_frame::{PageClass, physical_page_number};
use pagebox_storage::buffer_pool::BufferPool;
use pagebox_storage::free_page_allocator::{FreeExtent, FreePageAllocator};

const OPS_PER_CHUNK: usize = 10_000;
const OPS_PER_CHUNK_U64: u64 = OPS_PER_CHUNK as u64;

struct FreeAllocatorCtx {
    allocator: FreePageAllocator,
    next_reusable_page_number: u64,
    reusable_len: u64,
    page_class: PageClass,
}

impl BenchContext for FreeAllocatorCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("free-page allocator bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(OPS_PER_CHUNK)
    }
}

struct BufferPoolAllocCtx {
    pool: Arc<BufferPool>,
    reusable_page_number: u64,
    page_class: PageClass,
}

struct ConcurrentAllocatorMonotonicCtx {
    allocator: FreePageAllocator,
}

struct ConcurrentAllocatorPrefilledReuseCtx {
    allocator: FreePageAllocator,
}

struct ConcurrentAllocatorPromoteReuseCtx {
    allocator: FreePageAllocator,
    next_reusable_page_number: AtomicU64,
}

struct ConcurrentBufferPoolMonotonicCtx {
    pool: BufferPool,
}

impl BenchContext for BufferPoolAllocCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("buffer-pool allocation bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(OPS_PER_CHUNK)
    }
}

impl ConcurrentBenchContext for ConcurrentAllocatorMonotonicCtx {
    fn prepare(num_threads: usize) -> Self {
        Self {
            allocator: FreePageAllocator::new(1, num_threads),
        }
    }
}

impl ConcurrentBenchContext for ConcurrentAllocatorPrefilledReuseCtx {
    fn prepare(num_threads: usize) -> Self {
        let allocator = FreePageAllocator::new(100_000_000, num_threads);
        allocator.promote_reusable_extent(FreeExtent::new(1_000_000, 50_000_000));
        Self { allocator }
    }
}

impl ConcurrentBenchContext for ConcurrentAllocatorPromoteReuseCtx {
    fn prepare(num_threads: usize) -> Self {
        Self {
            allocator: FreePageAllocator::new(100_000_000, num_threads),
            next_reusable_page_number: AtomicU64::new(1_000_000),
        }
    }
}

impl ConcurrentBenchContext for ConcurrentBufferPoolMonotonicCtx {
    fn prepare(_num_threads: usize) -> Self {
        Self {
            pool: BufferPool::new(1024),
        }
    }
}

fn allocator_monotonic_4k(ctx: &mut FreeAllocatorCtx, chunk_size: usize, chunk_num: usize) {
    for i in 0..chunk_size {
        let pid = ctx
            .allocator
            .allocate_page(PageClass::Size4K, i + chunk_num);
        black_box(pid);
    }
}

fn allocator_reuse_extent(ctx: &mut FreeAllocatorCtx, chunk_size: usize, chunk_num: usize) {
    for i in 0..chunk_size {
        ctx.allocator.promote_reusable_extent(FreeExtent::new(
            ctx.next_reusable_page_number,
            ctx.reusable_len,
        ));
        ctx.next_reusable_page_number += ctx.reusable_len;
        let pid = ctx.allocator.allocate_page(ctx.page_class, i + chunk_num);
        black_box(pid);
    }
}

fn buffer_pool_monotonic_allocate_page(
    ctx: &mut BufferPoolAllocCtx,
    chunk_size: usize,
    _chunk_num: usize,
) {
    for _ in 0..chunk_size {
        black_box(ctx.pool.allocate_page());
    }
}

fn buffer_pool_reuse_allocate_page(
    ctx: &mut BufferPoolAllocCtx,
    chunk_size: usize,
    _chunk_num: usize,
) {
    for _ in 0..chunk_size {
        ctx.pool.promote_reusable_extent(FreeExtent::new(
            ctx.reusable_page_number,
            ctx.page_class.base_page_count() as u64,
        ));
        let swip = ctx.pool.allocate_page_class(ctx.page_class);
        let pid = swip.load(std::sync::atomic::Ordering::Acquire).as_page_id();
        ctx.reusable_page_number = physical_page_number(pid);
        black_box(pid);
    }
}

fn buffer_pool_reuse_allocate_and_fix_64k(
    ctx: &mut BufferPoolAllocCtx,
    chunk_size: usize,
    _chunk_num: usize,
) {
    for _ in 0..chunk_size {
        ctx.pool.promote_reusable_extent(FreeExtent::new(
            ctx.reusable_page_number,
            ctx.page_class.base_page_count() as u64,
        ));
        let (pid, frame) = ctx.pool.allocate_and_fix_class(ctx.page_class);
        drop(frame);
        ctx.reusable_page_number = physical_page_number(pid);
        black_box(pid);
    }
}

fn concurrent_allocator_monotonic_4k(
    ctx: &ConcurrentAllocatorMonotonicCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    while !control.should_stop() {
        let pid = ctx
            .allocator
            .allocate_page(PageClass::Size4K, control.thread_index());
        black_box(pid);
        operations = operations.wrapping_add(1);
    }
    ConcurrentWorkerResult::operations(operations)
}

fn concurrent_allocator_prefilled_reuse_4k(
    ctx: &ConcurrentAllocatorPrefilledReuseCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    while !control.should_stop() {
        let pid = ctx
            .allocator
            .allocate_page(PageClass::Size4K, control.thread_index());
        black_box(pid);
        operations = operations.wrapping_add(1);
    }
    ConcurrentWorkerResult::operations(operations)
}

fn concurrent_allocator_promote_reuse_4k(
    ctx: &ConcurrentAllocatorPromoteReuseCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    while !control.should_stop() {
        let page_number = ctx
            .next_reusable_page_number
            .fetch_add(1, Ordering::Relaxed);
        ctx.allocator
            .promote_reusable_extent(FreeExtent::new(page_number, 1));
        let pid = ctx
            .allocator
            .allocate_page(PageClass::Size4K, control.thread_index());
        black_box(pid);
        operations = operations.wrapping_add(1);
    }
    ConcurrentWorkerResult::operations(operations)
}

fn concurrent_buffer_pool_monotonic_allocate_page_4k(
    ctx: &ConcurrentBufferPoolMonotonicCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    while !control.should_stop() {
        black_box(ctx.pool.allocate_page());
        operations = operations.wrapping_add(1);
    }
    ConcurrentWorkerResult::operations(operations)
}

benchmark_main!(|runner| {
    runner.group::<FreeAllocatorCtx>("free_page_allocator", |g| {
        g.throughput(Throughput::per_operation(OPS_PER_CHUNK_U64, "allocations"))
            .factory(&|| FreeAllocatorCtx {
                allocator: FreePageAllocator::new(1, 16),
                next_reusable_page_number: 1_000_000,
                reusable_len: 1,
                page_class: PageClass::Size4K,
            })
            .bench("monotonic_4k", allocator_monotonic_4k);

        g.throughput(Throughput::per_operation(OPS_PER_CHUNK_U64, "allocations"))
            .factory(&|| FreeAllocatorCtx {
                allocator: FreePageAllocator::new(2_000_000, 16),
                next_reusable_page_number: 1_000_000,
                reusable_len: 1,
                page_class: PageClass::Size4K,
            })
            .bench("promote_reuse_4k", allocator_reuse_extent);

        g.throughput(Throughput::per_operation(OPS_PER_CHUNK_U64, "allocations"))
            .factory(&|| FreeAllocatorCtx {
                allocator: FreePageAllocator::new(2_000_000, 16),
                next_reusable_page_number: 1_000_000,
                reusable_len: PageClass::Size64K.base_page_count() as u64,
                page_class: PageClass::Size64K,
            })
            .bench("promote_reuse_64k", allocator_reuse_extent);
    });

    runner.group::<BufferPoolAllocCtx>("buffer_pool_allocator", |g| {
        g.throughput(Throughput::per_operation(OPS_PER_CHUNK_U64, "allocations"))
            .factory(&|| BufferPoolAllocCtx {
                pool: Arc::new(BufferPool::new(OPS_PER_CHUNK * 2)),
                reusable_page_number: 0,
                page_class: PageClass::Size4K,
            })
            .bench(
                "monotonic_allocate_page_4k",
                buffer_pool_monotonic_allocate_page,
            );

        g.throughput(Throughput::per_operation(OPS_PER_CHUNK_U64, "allocations"))
            .factory(&|| {
                let pool = Arc::new(BufferPool::new(OPS_PER_CHUNK * 2));
                let swip = pool.allocate_page();
                let pid = swip.load(std::sync::atomic::Ordering::Acquire).as_page_id();
                BufferPoolAllocCtx {
                    pool,
                    reusable_page_number: physical_page_number(pid),
                    page_class: PageClass::Size4K,
                }
            })
            .bench(
                "promote_reuse_allocate_page_4k",
                buffer_pool_reuse_allocate_page,
            );

        g.throughput(Throughput::per_operation(OPS_PER_CHUNK_U64, "allocations"))
            .factory(&|| {
                let pool = Arc::new(BufferPool::new(PageClass::Size64K.base_page_count() * 128));
                let class = PageClass::Size64K;
                let (pid, frame) = pool.allocate_and_fix_class(class);
                drop(frame);
                BufferPoolAllocCtx {
                    pool,
                    reusable_page_number: physical_page_number(pid),
                    page_class: class,
                }
            })
            .bench(
                "promote_reuse_allocate_and_fix_64k",
                buffer_pool_reuse_allocate_and_fix_64k,
            );
    });

    for &n_threads in &[1usize, 2, 4, 8, 16] {
        let workers = [ConcurrentWorker {
            name: "allocator_worker",
            threads: n_threads,
            run: concurrent_allocator_monotonic_4k,
        }];
        runner.concurrent_group::<ConcurrentAllocatorMonotonicCtx>(
            "free_page_allocator/concurrent/monotonic_4k",
            |g| {
                g.sample_duration(Duration::from_millis(50))
                    .throughput(Throughput::per_operation(1, "allocations"))
                    .bench(&format!("{n_threads}t"), &workers);
            },
        );
    }

    for &n_threads in &[1usize, 2, 4, 8, 16] {
        let workers = [ConcurrentWorker {
            name: "allocator_worker",
            threads: n_threads,
            run: concurrent_allocator_prefilled_reuse_4k,
        }];
        runner.concurrent_group::<ConcurrentAllocatorPrefilledReuseCtx>(
            "free_page_allocator/concurrent/prefilled_reuse_4k",
            |g| {
                g.sample_duration(Duration::from_millis(50))
                    .throughput(Throughput::per_operation(1, "allocations"))
                    .bench(&format!("{n_threads}t"), &workers);
            },
        );
    }

    for &n_threads in &[1usize, 2, 4, 8, 16] {
        let workers = [ConcurrentWorker {
            name: "allocator_worker",
            threads: n_threads,
            run: concurrent_allocator_promote_reuse_4k,
        }];
        runner.concurrent_group::<ConcurrentAllocatorPromoteReuseCtx>(
            "free_page_allocator/concurrent/promote_reuse_4k",
            |g| {
                g.sample_duration(Duration::from_millis(50))
                    .throughput(Throughput::per_operation(1, "allocations"))
                    .bench(&format!("{n_threads}t"), &workers);
            },
        );
    }

    for &n_threads in &[1usize, 2, 4, 8, 16] {
        let workers = [ConcurrentWorker {
            name: "buffer_pool_worker",
            threads: n_threads,
            run: concurrent_buffer_pool_monotonic_allocate_page_4k,
        }];
        runner.concurrent_group::<ConcurrentBufferPoolMonotonicCtx>(
            "buffer_pool_allocator/concurrent/monotonic_allocate_page_4k",
            |g| {
                g.sample_duration(Duration::from_millis(50))
                    .throughput(Throughput::per_operation(1, "allocations"))
                    .bench(&format!("{n_threads}t"), &workers);
            },
        );
    }
});
