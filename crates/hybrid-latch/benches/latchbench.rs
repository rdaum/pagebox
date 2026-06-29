use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use micromeasure::{
    ConcurrentBenchContext, ConcurrentBenchControl, ConcurrentWorker, ConcurrentWorkerResult,
    Throughput, benchmark_main, black_box,
};
use pagebox_hybrid_latch::HybridLatch;

const HOTSET_SIZE: usize = 16;
const HOT_LATCHES: usize = 2;
const HOT_BIAS_MASK: u64 = 0x7;
const WRITER_HOLD_SPINS_SHORT: usize = 64;
const WRITER_HOLD_SPINS_LONG: usize = 512;

struct HybridLatchCtx {
    latch: HybridLatch,
    hotset: Vec<HybridLatch>,
    writer_try_attempts: AtomicU64,
    writer_try_successes: AtomicU64,
    writer_try_failures: AtomicU64,
}

impl ConcurrentBenchContext for HybridLatchCtx {
    fn prepare(_num_threads: usize) -> Self {
        Self {
            latch: HybridLatch::new(),
            hotset: (0..HOTSET_SIZE).map(|_| HybridLatch::new()).collect(),
            writer_try_attempts: AtomicU64::new(0),
            writer_try_successes: AtomicU64::new(0),
            writer_try_failures: AtomicU64::new(0),
        }
    }
}

fn next_rng(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn hold_exclusive(spins: usize) {
    for _ in 0..spins {
        std::hint::spin_loop();
    }
}

fn hot_latch<'a>(ctx: &'a HybridLatchCtx, rng: &mut u64) -> &'a HybridLatch {
    let sample = next_rng(rng);
    let idx = if sample & HOT_BIAS_MASK != 0 {
        (sample as usize) % HOT_LATCHES
    } else {
        HOT_LATCHES + ((sample as usize) % (HOTSET_SIZE - HOT_LATCHES))
    };
    &ctx.hotset[idx]
}

fn optimistic_reader(
    ctx: &HybridLatchCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    let mut restarts = 0_u64;
    while !control.should_stop() {
        if let Ok(guard) = ctx.latch.optimistic_or_restart() {
            black_box(guard.validate().is_ok());
            operations = operations.wrapping_add(1);
        } else {
            restarts = restarts.wrapping_add(1);
        }
    }
    ConcurrentWorkerResult::operations(operations).with_counter("restarts", restarts)
}

fn shared_reader(ctx: &HybridLatchCtx, control: &ConcurrentBenchControl) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    while !control.should_stop() {
        let guard = ctx.latch.lock_shared();
        black_box(guard.version() ^ control.thread_index() as u64);
        operations = operations.wrapping_add(1);
    }
    ConcurrentWorkerResult::operations(operations)
}

fn exclusive_writer(
    ctx: &HybridLatchCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    while !control.should_stop() {
        let guard = ctx.latch.lock_exclusive();
        black_box(guard.version() ^ control.thread_index() as u64);
        operations = operations.wrapping_add(1);
    }
    ConcurrentWorkerResult::operations(operations)
}

fn upgrade_shared_reader(
    ctx: &HybridLatchCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    let mut optimistic_restarts = 0_u64;
    let mut upgrade_restarts = 0_u64;
    while !control.should_stop() {
        match ctx.latch.optimistic_or_restart() {
            Ok(guard) => match guard.upgrade_to_shared() {
                Ok(shared) => {
                    black_box(shared.version() ^ control.role_thread_index() as u64);
                    operations = operations.wrapping_add(1);
                }
                Err(_) => {
                    upgrade_restarts = upgrade_restarts.wrapping_add(1);
                }
            },
            Err(_) => {
                optimistic_restarts = optimistic_restarts.wrapping_add(1);
            }
        }
    }
    ConcurrentWorkerResult::operations(operations)
        .with_counter("optimistic_restarts", optimistic_restarts)
        .with_counter("upgrade_restarts", upgrade_restarts)
}

fn try_upgrade_exclusive_reader(
    ctx: &HybridLatchCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    let mut optimistic_restarts = 0_u64;
    let mut failed_try_upgrade = 0_u64;
    while !control.should_stop() {
        match ctx.latch.optimistic_or_restart() {
            Ok(guard) => match guard.try_upgrade_to_exclusive() {
                Ok(exclusive) => {
                    black_box(exclusive.version() ^ control.role_thread_index() as u64);
                    operations = operations.wrapping_add(1);
                }
                Err(_) => {
                    failed_try_upgrade = failed_try_upgrade.wrapping_add(1);
                }
            },
            Err(_) => {
                optimistic_restarts = optimistic_restarts.wrapping_add(1);
            }
        }
    }
    ConcurrentWorkerResult::operations(operations)
        .with_counter("optimistic_restarts", optimistic_restarts)
        .with_counter("failed_try_upgrade", failed_try_upgrade)
}

fn hotset_shared_reader(
    ctx: &HybridLatchCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    let mut rng = (control.thread_index() as u64 + 1) * 0x9E37_79B9_7F4A_7C15;
    while !control.should_stop() {
        let guard = hot_latch(ctx, &mut rng).lock_shared();
        black_box(guard.version() ^ control.thread_index() as u64);
        operations = operations.wrapping_add(1);
    }
    ConcurrentWorkerResult::operations(operations)
}

fn hotset_exclusive_writer_short(
    ctx: &HybridLatchCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    let mut rng = (control.thread_index() as u64 + 1) * 0xD1B5_4A32_D192_ED03;
    while !control.should_stop() {
        let guard = hot_latch(ctx, &mut rng).lock_exclusive();
        black_box(guard.version() ^ control.thread_index() as u64);
        hold_exclusive(WRITER_HOLD_SPINS_SHORT);
        operations = operations.wrapping_add(1);
    }
    ConcurrentWorkerResult::operations(operations)
}

fn hotset_try_exclusive_writer_short(
    ctx: &HybridLatchCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    let mut rng = (control.thread_index() as u64 + 1) * 0x7C15_9E37_79B9_4A7B;
    while !control.should_stop() {
        let latch = hot_latch(ctx, &mut rng);
        ctx.writer_try_attempts.fetch_add(1, Ordering::Relaxed);
        match latch.try_lock_exclusive() {
            Some(guard) => {
                black_box(guard.version() ^ control.thread_index() as u64);
                hold_exclusive(WRITER_HOLD_SPINS_SHORT);
                ctx.writer_try_successes.fetch_add(1, Ordering::Relaxed);
                operations = operations.wrapping_add(1);
            }
            None => {
                ctx.writer_try_failures.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    ConcurrentWorkerResult::operations(operations)
}

fn hotset_exclusive_writer_long(
    ctx: &HybridLatchCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    let mut rng = (control.thread_index() as u64 + 1) * 0x94D0_49BB_1331_11EB;
    while !control.should_stop() {
        let guard = hot_latch(ctx, &mut rng).lock_exclusive();
        black_box(guard.version() ^ control.thread_index() as u64);
        hold_exclusive(WRITER_HOLD_SPINS_LONG);
        operations = operations.wrapping_add(1);
    }
    ConcurrentWorkerResult::operations(operations)
}

fn hotset_upgrade_shared_reader(
    ctx: &HybridLatchCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    let mut optimistic_restarts = 0_u64;
    let mut upgrade_restarts = 0_u64;
    let mut rng = (control.thread_index() as u64 + 1) * 0x2545_F491_4F6C_DD1D;
    while !control.should_stop() {
        let latch = hot_latch(ctx, &mut rng);
        match latch.optimistic_or_restart() {
            Ok(guard) => match guard.upgrade_to_shared() {
                Ok(shared) => {
                    black_box(shared.version() ^ control.role_thread_index() as u64);
                    operations = operations.wrapping_add(1);
                }
                Err(_) => {
                    upgrade_restarts = upgrade_restarts.wrapping_add(1);
                }
            },
            Err(_) => {
                optimistic_restarts = optimistic_restarts.wrapping_add(1);
            }
        }
    }
    ConcurrentWorkerResult::operations(operations)
        .with_counter("optimistic_restarts", optimistic_restarts)
        .with_counter("upgrade_restarts", upgrade_restarts)
}

fn contended_exclusive_writer(
    ctx: &HybridLatchCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    while !control.should_stop() {
        let guard = ctx.latch.lock_exclusive();
        black_box(guard.version() ^ control.thread_index() as u64);
        hold_exclusive(WRITER_HOLD_SPINS_SHORT);
        operations = operations.wrapping_add(1);
    }
    ConcurrentWorkerResult::operations(operations)
}

fn contended_shared_reader(
    ctx: &HybridLatchCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    while !control.should_stop() {
        let guard = ctx.latch.lock_shared();
        black_box(guard.version() ^ control.thread_index() as u64);
        operations = operations.wrapping_add(1);
    }
    ConcurrentWorkerResult::operations(operations)
}

fn hotset_mixed_reader(
    ctx: &HybridLatchCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    let mut optimistic_restarts = 0_u64;
    let mut shared_fallbacks = 0_u64;
    let mut rng = (control.thread_index() as u64 + 1) * 0xA24B_AED4_963E_E407;
    while !control.should_stop() {
        let latch = hot_latch(ctx, &mut rng);
        match latch.optimistic_or_restart() {
            Ok(guard) => {
                if guard.validate().is_ok() {
                    black_box(control.thread_index() as u64);
                    operations = operations.wrapping_add(1);
                    continue;
                }
                optimistic_restarts = optimistic_restarts.wrapping_add(1);
            }
            Err(_) => {
                optimistic_restarts = optimistic_restarts.wrapping_add(1);
            }
        }
        let shared = latch.lock_shared();
        black_box(shared.version() ^ control.thread_index() as u64);
        shared_fallbacks = shared_fallbacks.wrapping_add(1);
        operations = operations.wrapping_add(1);
    }
    ConcurrentWorkerResult::operations(operations)
        .with_counter("optimistic_restarts", optimistic_restarts)
        .with_counter("shared_fallbacks", shared_fallbacks)
}

fn hotset_try_shared_mixed_reader(
    ctx: &HybridLatchCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0_u64;
    let mut optimistic_restarts = 0_u64;
    let mut shared_fallbacks = 0_u64;
    let mut shared_fallback_restarts = 0_u64;
    let mut rng = (control.thread_index() as u64 + 1) * 0x62A9_D9ED_7997_05F5;
    while !control.should_stop() {
        let latch = hot_latch(ctx, &mut rng);
        match latch.optimistic_or_restart() {
            Ok(guard) => {
                if guard.validate().is_ok() {
                    black_box(control.thread_index() as u64);
                    operations = operations.wrapping_add(1);
                    continue;
                }
                optimistic_restarts = optimistic_restarts.wrapping_add(1);
            }
            Err(_) => {
                optimistic_restarts = optimistic_restarts.wrapping_add(1);
            }
        }
        match latch.optimistic_or_restart() {
            Ok(guard) => match guard.try_upgrade_to_shared() {
                Ok(shared) => {
                    black_box(shared.version() ^ control.thread_index() as u64);
                    shared_fallbacks = shared_fallbacks.wrapping_add(1);
                    operations = operations.wrapping_add(1);
                }
                Err(_) => {
                    shared_fallback_restarts = shared_fallback_restarts.wrapping_add(1);
                }
            },
            Err(_) => {
                shared_fallback_restarts = shared_fallback_restarts.wrapping_add(1);
            }
        }
    }
    ConcurrentWorkerResult::operations(operations)
        .with_counter("optimistic_restarts", optimistic_restarts)
        .with_counter("shared_fallbacks", shared_fallbacks)
        .with_counter("shared_fallback_restarts", shared_fallback_restarts)
        .with_counter(
            "writer_try_attempts",
            ctx.writer_try_attempts.load(Ordering::Relaxed),
        )
        .with_counter(
            "writer_try_successes",
            ctx.writer_try_successes.load(Ordering::Relaxed),
        )
        .with_counter(
            "writer_try_failures",
            ctx.writer_try_failures.load(Ordering::Relaxed),
        )
}

benchmark_main!(|runner| {
    let optimistic_readers_vs_writer = [
        ConcurrentWorker {
            name: "optimistic_reader",
            threads: 3,
            run: optimistic_reader,
        },
        ConcurrentWorker {
            name: "exclusive_writer",
            threads: 1,
            run: exclusive_writer,
        },
    ];

    let shared_readers_vs_writer = [
        ConcurrentWorker {
            name: "shared_reader",
            threads: 3,
            run: shared_reader,
        },
        ConcurrentWorker {
            name: "exclusive_writer",
            threads: 1,
            run: exclusive_writer,
        },
    ];

    let upgrade_shared_vs_writer = [
        ConcurrentWorker {
            name: "upgrade_shared_reader",
            threads: 3,
            run: upgrade_shared_reader,
        },
        ConcurrentWorker {
            name: "exclusive_writer",
            threads: 1,
            run: exclusive_writer,
        },
    ];

    let try_upgrade_exclusive_vs_writer = [
        ConcurrentWorker {
            name: "try_upgrade_exclusive_reader",
            threads: 3,
            run: try_upgrade_exclusive_reader,
        },
        ConcurrentWorker {
            name: "exclusive_writer",
            threads: 1,
            run: exclusive_writer,
        },
    ];

    let hotset_shared_heavy_short_writer = [
        ConcurrentWorker {
            name: "hotset_shared_reader",
            threads: 11,
            run: hotset_shared_reader,
        },
        ConcurrentWorker {
            name: "hotset_exclusive_writer_short",
            threads: 1,
            run: hotset_exclusive_writer_short,
        },
    ];

    let hotset_shared_heavy_long_writer = [
        ConcurrentWorker {
            name: "hotset_shared_reader",
            threads: 11,
            run: hotset_shared_reader,
        },
        ConcurrentWorker {
            name: "hotset_exclusive_writer_long",
            threads: 1,
            run: hotset_exclusive_writer_long,
        },
    ];

    let hotset_upgrade_heavy = [
        ConcurrentWorker {
            name: "hotset_upgrade_shared_reader",
            threads: 11,
            run: hotset_upgrade_shared_reader,
        },
        ConcurrentWorker {
            name: "hotset_exclusive_writer_short",
            threads: 1,
            run: hotset_exclusive_writer_short,
        },
    ];

    let hotset_mixed_readers = [
        ConcurrentWorker {
            name: "hotset_mixed_reader",
            threads: 11,
            run: hotset_mixed_reader,
        },
        ConcurrentWorker {
            name: "hotset_exclusive_writer_short",
            threads: 1,
            run: hotset_exclusive_writer_short,
        },
    ];

    let hotset_mixed_readers_20 = [
        ConcurrentWorker {
            name: "hotset_mixed_reader",
            threads: 19,
            run: hotset_mixed_reader,
        },
        ConcurrentWorker {
            name: "hotset_exclusive_writer_short",
            threads: 1,
            run: hotset_exclusive_writer_short,
        },
    ];

    let hotset_try_shared_mixed_readers_20 = [
        ConcurrentWorker {
            name: "hotset_try_shared_mixed_reader",
            threads: 19,
            run: hotset_try_shared_mixed_reader,
        },
        ConcurrentWorker {
            name: "hotset_exclusive_writer_short",
            threads: 1,
            run: hotset_exclusive_writer_short,
        },
    ];

    let hotset_try_shared_mixed_readers_20_probe_writer = [
        ConcurrentWorker {
            name: "hotset_try_shared_mixed_reader",
            threads: 19,
            run: hotset_try_shared_mixed_reader,
        },
        ConcurrentWorker {
            name: "hotset_try_exclusive_writer_short",
            threads: 1,
            run: hotset_try_exclusive_writer_short,
        },
    ];

    let contended_exclusive_4 = [ConcurrentWorker {
        name: "contended_exclusive_writer",
        threads: 4,
        run: contended_exclusive_writer,
    }];

    let contended_exclusive_8 = [ConcurrentWorker {
        name: "contended_exclusive_writer",
        threads: 8,
        run: contended_exclusive_writer,
    }];

    let contended_shared_4_vs_exclusive_4 = [
        ConcurrentWorker {
            name: "contended_shared_reader",
            threads: 4,
            run: contended_shared_reader,
        },
        ConcurrentWorker {
            name: "contended_exclusive_writer",
            threads: 4,
            run: contended_exclusive_writer,
        },
    ];

    runner.concurrent_group::<HybridLatchCtx>("hybrid_latch", |g| {
        g.sample_duration(Duration::from_millis(100))
            .throughput(Throughput::per_operation(1, "lock_ops"))
            .bench(
                "optimistic_readers_vs_writer",
                &optimistic_readers_vs_writer,
            );
        g.sample_duration(Duration::from_millis(100))
            .throughput(Throughput::per_operation(1, "lock_ops"))
            .bench("shared_readers_vs_writer", &shared_readers_vs_writer);
        g.sample_duration(Duration::from_millis(100))
            .throughput(Throughput::per_operation(1, "lock_ops"))
            .bench("upgrade_shared_vs_writer", &upgrade_shared_vs_writer);
        g.sample_duration(Duration::from_millis(100))
            .throughput(Throughput::per_operation(1, "lock_ops"))
            .bench(
                "try_upgrade_exclusive_vs_writer",
                &try_upgrade_exclusive_vs_writer,
            );
        g.sample_duration(Duration::from_millis(100))
            .throughput(Throughput::per_operation(1, "lock_ops"))
            .bench(
                "hotset_shared_heavy_short_writer",
                &hotset_shared_heavy_short_writer,
            );
        g.sample_duration(Duration::from_millis(100))
            .throughput(Throughput::per_operation(1, "lock_ops"))
            .bench(
                "hotset_shared_heavy_long_writer",
                &hotset_shared_heavy_long_writer,
            );
        g.sample_duration(Duration::from_millis(100))
            .throughput(Throughput::per_operation(1, "lock_ops"))
            .bench("hotset_upgrade_heavy", &hotset_upgrade_heavy);
        g.sample_duration(Duration::from_millis(100))
            .throughput(Throughput::per_operation(1, "lock_ops"))
            .bench("hotset_mixed_readers", &hotset_mixed_readers);
        g.sample_duration(Duration::from_millis(100))
            .throughput(Throughput::per_operation(1, "lock_ops"))
            .bench("hotset_mixed_readers_20", &hotset_mixed_readers_20);
        g.sample_duration(Duration::from_millis(100))
            .throughput(Throughput::per_operation(1, "lock_ops"))
            .bench(
                "hotset_try_shared_mixed_readers_20",
                &hotset_try_shared_mixed_readers_20,
            );
        g.sample_duration(Duration::from_millis(100))
            .throughput(Throughput::per_operation(1, "lock_ops"))
            .bench(
                "hotset_try_shared_mixed_readers_20_probe_writer",
                &hotset_try_shared_mixed_readers_20_probe_writer,
            );
        g.sample_duration(Duration::from_millis(100))
            .throughput(Throughput::per_operation(1, "lock_ops"))
            .bench("contended_exclusive_4", &contended_exclusive_4);
        g.sample_duration(Duration::from_millis(100))
            .throughput(Throughput::per_operation(1, "lock_ops"))
            .bench("contended_exclusive_8", &contended_exclusive_8);
        g.sample_duration(Duration::from_millis(100))
            .throughput(Throughput::per_operation(1, "lock_ops"))
            .bench(
                "contended_shared_4_vs_exclusive_4",
                &contended_shared_4_vs_exclusive_4,
            );
    });
});
