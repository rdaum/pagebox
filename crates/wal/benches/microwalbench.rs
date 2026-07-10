use std::sync::{Arc, OnceLock};
use std::time::Duration;

use micromeasure::{
    BenchContext, ConcurrentBenchContext, ConcurrentBenchControl, ConcurrentWorker,
    ConcurrentWorkerResult, Throughput, benchmark_main, black_box,
};
use pagebox_frame_kernel::PAGE_SIZE;
use pagebox_wal::{CommitMode, WAL_BUF_RECORDS, Wal};

struct AppendBufferOnlyCtx {
    wal: Wal,
    _dir: tempfile::TempDir,
    pages: Vec<[u8; PAGE_SIZE]>,
}

impl BenchContext for AppendBufferOnlyCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("append buffer bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(WAL_BUF_RECORDS)
    }
}

struct AppendFlushCtx {
    wal: Wal,
    _dir: tempfile::TempDir,
    pages: Vec<[u8; PAGE_SIZE]>,
}

impl BenchContext for AppendFlushCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("append flush bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(64)
    }
}

struct ReplayCtx {
    wal: Wal,
    _dir: tempfile::TempDir,
    expected: usize,
}

impl BenchContext for ReplayCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("replay bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(16)
    }
}

struct ConcurrentWalState {
    wal: Arc<Wal>,
    _dir: tempfile::TempDir,
    pages: Arc<Vec<[u8; PAGE_SIZE]>>,
}

#[derive(Clone)]
struct ConcurrentWalCtx(Arc<ConcurrentWalState>);

impl std::ops::Deref for ConcurrentWalCtx {
    type Target = ConcurrentWalState;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ConcurrentWalCtx {
    fn prepare_with_mode(commit_mode: CommitMode, num_threads: usize) -> Self {
        let dir = tempfile::tempdir().unwrap();
        // Keep backend comparisons on the same single-shard topology. The
        // production default may create one shard per available CPU, which
        // makes per-sample setup dominate this benchmark and is not comparable
        // with io_uring's required single-shard layout.
        let wal = Arc::new(Wal::open_with_shards(&dir.path().join("wal"), 1).unwrap());
        wal.set_commit_mode(commit_mode);
        let page_count = (num_threads.max(1) * 64).next_power_of_two();
        let pages = Arc::new((0..page_count as u64).map(page_data).collect());
        Self(Arc::new(ConcurrentWalState {
            wal,
            _dir: dir,
            pages,
        }))
    }
}

#[derive(Clone)]
struct ConcurrentWalAppendCtx(ConcurrentWalCtx);

impl ConcurrentBenchContext for ConcurrentWalAppendCtx {
    fn prepare(_num_threads: usize) -> Self {
        panic!("concurrent WAL benchmarks must use their shared factory")
    }
}

#[derive(Clone)]
struct ConcurrentWalCommitStrictCtx(ConcurrentWalCtx);

impl ConcurrentBenchContext for ConcurrentWalCommitStrictCtx {
    fn prepare(_num_threads: usize) -> Self {
        panic!("concurrent WAL benchmarks must use their shared factory")
    }
}

#[derive(Clone)]
struct ConcurrentWalCommitRelaxedCtx(ConcurrentWalCtx);

impl ConcurrentBenchContext for ConcurrentWalCommitRelaxedCtx {
    fn prepare(_num_threads: usize) -> Self {
        panic!("concurrent WAL benchmarks must use their shared factory")
    }
}

fn page_data(seed: u64) -> [u8; PAGE_SIZE] {
    let mut buf = [0u8; PAGE_SIZE];
    let bytes = seed.to_le_bytes();
    for chunk in buf.chunks_exact_mut(8) {
        chunk.copy_from_slice(&bytes);
    }
    buf
}

fn append_buffer_only(ctx: &mut AppendBufferOnlyCtx, chunk_size: usize, _chunk_num: usize) {
    for i in 0..chunk_size {
        let pid = i as u64;
        let lsn = ctx.wal.append_page_image(pid, &ctx.pages[i]).unwrap();
        black_box(lsn);
    }
}

fn append_flush(ctx: &mut AppendFlushCtx, chunk_size: usize, chunk_num: usize) {
    let base = chunk_num * chunk_size;
    for i in 0..chunk_size {
        let idx = (base + i) % ctx.pages.len();
        let lsn = ctx
            .wal
            .append_page_image(idx as u64, &ctx.pages[idx])
            .unwrap();
        black_box(lsn);
    }
    black_box(ctx.wal.flush());
}

fn replay_walk(ctx: &mut ReplayCtx, _chunk_size: usize, _chunk_num: usize) {
    let mut seen = 0usize;
    ctx.wal
        .replay(|lsn, pid, data| {
            seen += 1;
            black_box((lsn, pid, data[0]));
        })
        .unwrap();
    assert_eq!(seen, ctx.expected, "replay should scan the full WAL");
}

fn concurrent_append_worker(
    ctx: &ConcurrentWalAppendCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut operations = 0u64;
    let base = ((control.thread_index() as u64) << 32) | control.role_thread_index() as u64;

    while !control.should_stop() {
        let idx = (operations as usize) % ctx.0.pages.len();
        let pid = base.wrapping_add(operations);
        let lsn = ctx.0.wal.append_page_image(pid, &ctx.0.pages[idx]).unwrap();
        black_box(lsn);
        operations = operations.wrapping_add(1);
    }

    ConcurrentWorkerResult::operations(operations)
}

fn concurrent_commit_worker(
    ctx: &ConcurrentWalCommitStrictCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    concurrent_commit_worker_inner(&ctx.0, control)
}

fn concurrent_commit_relaxed_worker(
    ctx: &ConcurrentWalCommitRelaxedCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    concurrent_commit_worker_inner(&ctx.0, control)
}

fn concurrent_commit_worker_inner(
    ctx: &ConcurrentWalCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    let mut commits = 0u64;
    let base = ((control.thread_index() as u64) << 32) | control.role_thread_index() as u64;

    while !control.should_stop() {
        let idx = (commits as usize) % ctx.0.pages.len();
        let pid = base.wrapping_add(commits);
        let lsn = ctx.0.wal.append_page_image(pid, &ctx.0.pages[idx]).unwrap();
        black_box(ctx.0.wal.flush_at_least(lsn));
        commits = commits.wrapping_add(1);
    }

    ConcurrentWorkerResult::operations(commits)
}

benchmark_main!(|runner| {
    runner.group::<AppendBufferOnlyCtx>("wal_append", |g| {
        g.throughput(Throughput::per_operation(WAL_BUF_RECORDS as u64, "pages"))
            .factory(&|| {
                let dir = tempfile::tempdir().unwrap();
                let wal = Wal::open_opts(&dir.path().join("wal")).unwrap();
                let pages = (0..WAL_BUF_RECORDS as u64).map(page_data).collect();
                AppendBufferOnlyCtx {
                    wal,
                    _dir: dir,
                    pages,
                }
            })
            .bench("buffer_only_full_window", append_buffer_only);
    });

    runner.group::<AppendFlushCtx>("wal_commit", |g| {
        g.throughput(Throughput::per_operation(64, "pages"))
            .factory(&|| {
                let dir = tempfile::tempdir().unwrap();
                let wal = Wal::open_opts(&dir.path().join("wal")).unwrap();
                let pages = (0..1u64).map(page_data).collect();
                AppendFlushCtx {
                    wal,
                    _dir: dir,
                    pages,
                }
            })
            .bench("append_flush_batch_1", append_flush);
        g.throughput(Throughput::per_operation(64, "pages"))
            .factory(&|| {
                let dir = tempfile::tempdir().unwrap();
                let wal = Wal::open_opts(&dir.path().join("wal")).unwrap();
                let pages = (0..8u64).map(page_data).collect();
                AppendFlushCtx {
                    wal,
                    _dir: dir,
                    pages,
                }
            })
            .bench("append_flush_batch_8", append_flush);
        g.throughput(Throughput::per_operation(64, "pages"))
            .factory(&|| {
                let dir = tempfile::tempdir().unwrap();
                let wal = Wal::open_opts(&dir.path().join("wal")).unwrap();
                let pages = (0..64u64).map(page_data).collect();
                AppendFlushCtx {
                    wal,
                    _dir: dir,
                    pages,
                }
            })
            .bench("append_flush_batch_64", append_flush);
    });

    runner.group::<ReplayCtx>("wal_replay", |g| {
        g.throughput(Throughput::per_operation(1_024, "records"))
            .factory(&|| {
                let dir = tempfile::tempdir().unwrap();
                let wal = Wal::open_opts(&dir.path().join("wal")).unwrap();
                let page = page_data(7);
                for i in 0..1_024usize {
                    wal.append_page_image(i as u64, &page).unwrap();
                }
                wal.flush();
                ReplayCtx {
                    wal,
                    _dir: dir,
                    expected: 1_024,
                }
            })
            .bench("records_1024", replay_walk);
        g.throughput(Throughput::per_operation(8_192, "records"))
            .factory(&|| {
                let dir = tempfile::tempdir().unwrap();
                let wal = Wal::open_opts(&dir.path().join("wal")).unwrap();
                let page = page_data(7);
                for i in 0..8_192usize {
                    wal.append_page_image(i as u64, &page).unwrap();
                }
                wal.flush();
                ReplayCtx {
                    wal,
                    _dir: dir,
                    expected: 8_192,
                }
            })
            .bench("records_8192", replay_walk);
    });

    for &n_threads in &[1usize, 2, 4, 8] {
        // micromeasure invokes the factory outside each warm-up or measured
        // sample. Keep the threads and ring alive, but truncate the WAL before
        // returning the context so every timed sample starts from a bounded,
        // quiescent log instead of inheriting an ever-growing file.
        let append_ctx = OnceLock::<ConcurrentWalAppendCtx>::new();
        let append_factory = |num_threads| {
            let ctx = append_ctx.get_or_init(|| {
                ConcurrentWalAppendCtx(ConcurrentWalCtx::prepare_with_mode(
                    CommitMode::Strict,
                    num_threads,
                ))
            });
            ctx.0
                .wal
                .reset()
                .expect("reset concurrent append WAL between samples");
            ctx.clone()
        };
        let append_workers = [ConcurrentWorker {
            name: "append_worker",
            threads: n_threads,
            run: concurrent_append_worker,
        }];
        let strict_ctx = OnceLock::<ConcurrentWalCommitStrictCtx>::new();
        let strict_factory = |num_threads| {
            let ctx = strict_ctx.get_or_init(|| {
                ConcurrentWalCommitStrictCtx(ConcurrentWalCtx::prepare_with_mode(
                    CommitMode::Strict,
                    num_threads,
                ))
            });
            ctx.0
                .wal
                .reset()
                .expect("reset strict concurrent WAL between samples");
            ctx.clone()
        };
        let commit_workers = [ConcurrentWorker {
            name: "commit_worker",
            threads: n_threads,
            run: concurrent_commit_worker,
        }];

        runner.concurrent_group::<ConcurrentWalAppendCtx>("wal_concurrent_append", |g| {
            g.sample_duration(Duration::from_millis(200))
                .throughput(Throughput::per_operation(1, "pages"))
                .factory(&append_factory)
                .bench(&format!("{n_threads}t"), &append_workers);
        });

        runner.concurrent_group::<ConcurrentWalCommitStrictCtx>(
            "wal_concurrent_commit_strict",
            |g| {
                g.sample_duration(Duration::from_millis(200))
                    .throughput(Throughput::per_operation(1, "commits"))
                    .factory(&strict_factory)
                    .bench(&format!("{n_threads}t"), &commit_workers);
            },
        );

        let relaxed_ctx = OnceLock::<ConcurrentWalCommitRelaxedCtx>::new();
        let relaxed_factory = |num_threads| {
            let ctx = relaxed_ctx.get_or_init(|| {
                ConcurrentWalCommitRelaxedCtx(ConcurrentWalCtx::prepare_with_mode(
                    CommitMode::Relaxed,
                    num_threads,
                ))
            });
            ctx.0
                .wal
                .reset()
                .expect("reset relaxed concurrent WAL between samples");
            ctx.clone()
        };
        let relaxed_workers = [ConcurrentWorker {
            name: "commit_worker",
            threads: n_threads,
            run: concurrent_commit_relaxed_worker,
        }];

        runner.concurrent_group::<ConcurrentWalCommitRelaxedCtx>(
            "wal_concurrent_commit_relaxed",
            |g| {
                g.sample_duration(Duration::from_millis(200))
                    .throughput(Throughput::per_operation(1, "commits"))
                    .factory(&relaxed_factory)
                    .bench(&format!("{n_threads}t"), &relaxed_workers);
            },
        );
    }
});
