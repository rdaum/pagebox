use std::sync::{Arc, OnceLock};
use std::time::Duration;

#[cfg(feature = "metrics")]
use fast_telemetry::{HistogramSnapshot, MetricLabels, MetricMeta, MetricVisitor};
use micromeasure::{
    BenchContext, BenchmarkCaseOrder, ConcurrentBenchContext, ConcurrentBenchControl,
    ConcurrentSampleInfo, ConcurrentSampleLifecycle, ConcurrentWorker, ConcurrentWorkerResult,
    MeasurementDomain, MetricValue, Throughput, benchmark_main, black_box,
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

#[cfg(feature = "metrics")]
#[derive(Clone, Copy, Default)]
struct WalSampleMetrics {
    flush_calls: u64,
    write_calls: u64,
    write_bytes: u64,
    sync_calls: u64,
    sync_coalesced: u64,
    durable_advances: u64,
    write_latency_count: u64,
    write_latency_ns: u64,
    sync_latency_count: u64,
    sync_latency_ns: u64,
    drain_latency_count: u64,
    drain_latency_ns: u64,
    fsync_latency_count: u64,
    fsync_latency_ns: u64,
}

#[cfg(feature = "metrics")]
impl WalSampleMetrics {
    fn capture(wal: &Wal) -> Self {
        struct Visitor(WalSampleMetrics);

        impl MetricVisitor for Visitor {
            fn counter(&mut self, meta: MetricMeta<'_>, labels: MetricLabels<'_>, value: i64) {
                if meta.name != "wal_events" {
                    return;
                }
                let value = value.max(0) as u64;
                let event = labels
                    .iter()
                    .find(|label| label.name == "event")
                    .map(|label| label.value);
                match event {
                    Some("flush_call") => self.0.flush_calls += value,
                    Some("write_call") => self.0.write_calls += value,
                    Some("write_bytes") => self.0.write_bytes += value,
                    Some("sync_call") => self.0.sync_calls += value,
                    Some("sync_coalesced") => self.0.sync_coalesced += value,
                    Some("durable_advance") => self.0.durable_advances += value,
                    _ => {}
                }
            }

            fn gauge_i64(&mut self, _meta: MetricMeta<'_>, _labels: MetricLabels<'_>, _value: i64) {
            }

            fn gauge_f64(&mut self, _meta: MetricMeta<'_>, _labels: MetricLabels<'_>, _value: f64) {
            }

            fn histogram(
                &mut self,
                meta: MetricMeta<'_>,
                labels: MetricLabels<'_>,
                histogram: &dyn HistogramSnapshot,
            ) {
                if meta.name != "wal_latencies" {
                    return;
                }
                let latency = labels
                    .iter()
                    .find(|label| label.name == "latency")
                    .map(|label| label.value);
                match latency {
                    Some("write") => {
                        self.0.write_latency_count += histogram.count();
                        self.0.write_latency_ns += histogram.sum();
                    }
                    Some("sync") => {
                        self.0.sync_latency_count += histogram.count();
                        self.0.sync_latency_ns += histogram.sum();
                    }
                    Some("sync_drain") => {
                        self.0.drain_latency_count += histogram.count();
                        self.0.drain_latency_ns += histogram.sum();
                    }
                    Some("sync_fsync") => {
                        self.0.fsync_latency_count += histogram.count();
                        self.0.fsync_latency_ns += histogram.sum();
                    }
                    _ => {}
                }
            }
        }

        let mut visitor = Visitor(Self::default());
        wal.visit_metrics(&mut visitor);
        visitor.0
    }

    fn delta(self, start: Self) -> Self {
        Self {
            flush_calls: self.flush_calls.saturating_sub(start.flush_calls),
            write_calls: self.write_calls.saturating_sub(start.write_calls),
            write_bytes: self.write_bytes.saturating_sub(start.write_bytes),
            sync_calls: self.sync_calls.saturating_sub(start.sync_calls),
            sync_coalesced: self.sync_coalesced.saturating_sub(start.sync_coalesced),
            durable_advances: self.durable_advances.saturating_sub(start.durable_advances),
            write_latency_count: self
                .write_latency_count
                .saturating_sub(start.write_latency_count),
            write_latency_ns: self.write_latency_ns.saturating_sub(start.write_latency_ns),
            sync_latency_count: self
                .sync_latency_count
                .saturating_sub(start.sync_latency_count),
            sync_latency_ns: self.sync_latency_ns.saturating_sub(start.sync_latency_ns),
            drain_latency_count: self
                .drain_latency_count
                .saturating_sub(start.drain_latency_count),
            drain_latency_ns: self.drain_latency_ns.saturating_sub(start.drain_latency_ns),
            fsync_latency_count: self
                .fsync_latency_count
                .saturating_sub(start.fsync_latency_count),
            fsync_latency_ns: self.fsync_latency_ns.saturating_sub(start.fsync_latency_ns),
        }
    }

    fn into_metrics(self) -> Vec<MetricValue> {
        vec![
            count_metric("commits", self.flush_calls, "commits", "Commits"),
            count_metric("writes", self.write_calls, "writes", "Writes"),
            count_metric("barriers", self.sync_calls, "barriers", "Barriers"),
            count_metric(
                "coalesced_targets",
                self.sync_coalesced,
                "targets",
                "Coalesced targets",
            ),
            count_metric(
                "durable_advances",
                self.durable_advances,
                "advances",
                "Durable advances",
            ),
            wal_metric(
                "commits_per_durable",
                ratio(self.flush_calls, self.durable_advances),
                "commits/durable",
                "Commits/durable",
            ),
            wal_metric(
                "writes_per_durable",
                ratio(self.write_calls, self.durable_advances),
                "writes/durable",
                "Writes/durable",
            ),
            wal_metric(
                "durable_latency_us",
                if self.sync_latency_count > 0 {
                    ratio(self.sync_latency_ns, self.sync_latency_count) / 1_000.0
                } else {
                    ratio(self.write_latency_ns, self.write_latency_count) / 1_000.0
                },
                "us",
                "Durable latency",
            ),
            wal_metric(
                "commits_per_barrier",
                ratio(self.flush_calls, self.sync_calls),
                "commits/barrier",
                "Commits/barrier",
            ),
            wal_metric(
                "writes_per_barrier",
                ratio(self.write_calls, self.sync_calls),
                "writes/barrier",
                "Writes/barrier",
            ),
            wal_metric(
                "coalesced_per_barrier",
                ratio(self.sync_coalesced, self.sync_calls),
                "targets/barrier",
                "Coalesced/barrier",
            ),
            wal_metric(
                "mean_write_kib",
                ratio(self.write_bytes, self.write_calls) / 1024.0,
                "KiB",
                "Mean write",
            ),
            wal_metric(
                "write_completion_us",
                ratio(self.write_latency_ns, self.write_latency_count) / 1_000.0,
                "us",
                "Write completion",
            ),
            wal_metric(
                "barrier_total_us",
                ratio(self.sync_latency_ns, self.sync_latency_count) / 1_000.0,
                "us",
                "Barrier total",
            ),
            wal_metric(
                "drain_wait_us",
                ratio(self.drain_latency_ns, self.drain_latency_count) / 1_000.0,
                "us",
                "Drain wait",
            ),
            wal_metric(
                "fsync_service_us",
                ratio(self.fsync_latency_ns, self.fsync_latency_count) / 1_000.0,
                "us",
                "Fsync service",
            ),
        ]
    }
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

    fn reset(&self) {
        self.wal
            .reset()
            .expect("reset concurrent WAL between samples");
    }
}

#[cfg(feature = "metrics")]
fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(feature = "metrics")]
fn wal_metric(
    name: &'static str,
    value: f64,
    unit: &'static str,
    display_name: &'static str,
) -> MetricValue {
    MetricValue::new(name, value, unit)
        .with_display_name(display_name)
        .with_section("wal")
}

#[cfg(feature = "metrics")]
fn count_metric(
    name: &'static str,
    value: u64,
    unit: &'static str,
    display_name: &'static str,
) -> MetricValue {
    MetricValue::integer(name, value.min(i64::MAX as u64) as i64, unit)
        .with_display_name(display_name)
        .with_section("wal")
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

trait ConcurrentWalBenchContext {
    fn wal_context(&self) -> &ConcurrentWalCtx;
}

impl ConcurrentWalBenchContext for ConcurrentWalAppendCtx {
    fn wal_context(&self) -> &ConcurrentWalCtx {
        &self.0
    }
}

impl ConcurrentWalBenchContext for ConcurrentWalCommitRelaxedCtx {
    fn wal_context(&self) -> &ConcurrentWalCtx {
        &self.0
    }
}

struct ResetWalLifecycle;

impl<T: ConcurrentWalBenchContext> ConcurrentSampleLifecycle<T> for ResetWalLifecycle {
    fn before_sample(&mut self, context: &mut T, _sample: ConcurrentSampleInfo) {
        context.wal_context().reset();
    }
}

#[derive(Default)]
struct StrictWalLifecycle {
    #[cfg(feature = "metrics")]
    start_metrics: Option<WalSampleMetrics>,
}

impl ConcurrentSampleLifecycle<ConcurrentWalCommitStrictCtx> for StrictWalLifecycle {
    fn before_sample(
        &mut self,
        context: &mut ConcurrentWalCommitStrictCtx,
        _sample: ConcurrentSampleInfo,
    ) {
        context.0.reset();
        #[cfg(feature = "metrics")]
        {
            self.start_metrics = Some(WalSampleMetrics::capture(&context.0.wal));
        }
    }

    fn after_sample(
        &mut self,
        context: &mut ConcurrentWalCommitStrictCtx,
        _sample: ConcurrentSampleInfo,
    ) -> Vec<MetricValue> {
        #[cfg(feature = "metrics")]
        {
            let start = self
                .start_metrics
                .take()
                .expect("strict WAL lifecycle missing sample start metrics");
            WalSampleMetrics::capture(&context.0.wal)
                .delta(start)
                .into_metrics()
        }

        #[cfg(not(feature = "metrics"))]
        {
            let _ = context;
            Vec::new()
        }
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

fn wal_sync_backend_metadata() -> &'static str {
    match std::env::var("PAGEBOX_WAL_SYNC_BACKEND").ok().as_deref() {
        Some("pwritev2_dsync") | Some("pwritev2-dsync") | Some("rwf_dsync") => "pwritev2_dsync",
        #[cfg(target_os = "linux")]
        Some("io_uring") | Some("iouring") => "io_uring",
        _ => "fdatasync",
    }
}

fn wal_direct_io_metadata() -> &'static str {
    if matches!(
        std::env::var("PAGEBOX_WAL_DIRECT_IO").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    ) {
        "true"
    } else {
        "false"
    }
}

fn env_u64_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
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
    ConcurrentWorkerResult::operations(concurrent_commit_worker_inner(&ctx.0, control))
}

fn concurrent_commit_relaxed_worker(
    ctx: &ConcurrentWalCommitRelaxedCtx,
    control: &ConcurrentBenchControl,
) -> ConcurrentWorkerResult {
    ConcurrentWorkerResult::operations(concurrent_commit_worker_inner(&ctx.0, control))
}

fn concurrent_commit_worker_inner(ctx: &ConcurrentWalCtx, control: &ConcurrentBenchControl) -> u64 {
    let mut commits = 0u64;
    let base = ((control.thread_index() as u64) << 32) | control.role_thread_index() as u64;

    while !control.should_stop() {
        let idx = (commits as usize) % ctx.0.pages.len();
        let pid = base.wrapping_add(commits);
        let lsn = ctx.0.wal.append_page_image(pid, &ctx.0.pages[idx]).unwrap();
        black_box(ctx.0.wal.flush_at_least(lsn));
        commits = commits.wrapping_add(1);
    }

    commits
}

benchmark_main!(|runner| {
    let sync_backend = wal_sync_backend_metadata();
    let direct_io = wal_direct_io_metadata();
    let delay_min_us = env_u64_or("PAGEBOX_WAL_GROUP_COMMIT_DELAY_MIN_US", 100);
    let delay_max_default_us = if sync_backend == "fdatasync" { 0 } else { 250 };
    let delay_max_us = env_u64_or(
        "PAGEBOX_WAL_GROUP_COMMIT_DELAY_MAX_US",
        delay_max_default_us,
    );
    let target_default_records = if sync_backend == "fdatasync" { 256 } else { 32 };
    let target_records = env_u64_or(
        "PAGEBOX_WAL_GROUP_COMMIT_TARGET_RECORDS",
        target_default_records,
    );
    let case_seed = env_u64_or("PAGEBOX_BENCH_CASE_SEED", 0x7061_6765_626f_7809);
    let case_cooldown_ms = env_u64_or("PAGEBOX_BENCH_CASE_COOLDOWN_MS", 1_000);

    runner.group::<AppendBufferOnlyCtx>("wal_append", |g| {
        g.throughput(Throughput::per_operation(1, "pages"))
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

    runner.set_case_cooldown(Duration::from_millis(case_cooldown_ms));
    let thread_counts = [1usize, 2, 4, 8];
    let case_order = runner.ordered_case_indices(
        thread_counts.len(),
        BenchmarkCaseOrder::Randomized { seed: case_seed },
    );
    for case_index in case_order {
        let n_threads = thread_counts[case_index];
        let benchmark_name = format!("{n_threads}t");
        // Keep each WAL and its background threads alive across samples. The
        // lifecycle hook resets persistent state outside the timed window.
        let append_ctx = OnceLock::<ConcurrentWalAppendCtx>::new();
        let append_factory = |num_threads| {
            append_ctx
                .get_or_init(|| {
                    ConcurrentWalAppendCtx(ConcurrentWalCtx::prepare_with_mode(
                        CommitMode::Strict,
                        num_threads,
                    ))
                })
                .clone()
        };
        let append_workers = [ConcurrentWorker {
            name: "append_worker",
            threads: n_threads,
            run: concurrent_append_worker,
        }];
        let strict_ctx = OnceLock::<ConcurrentWalCommitStrictCtx>::new();
        let strict_factory = |num_threads| {
            strict_ctx
                .get_or_init(|| {
                    ConcurrentWalCommitStrictCtx(ConcurrentWalCtx::prepare_with_mode(
                        CommitMode::Strict,
                        num_threads,
                    ))
                })
                .clone()
        };
        let commit_workers = [ConcurrentWorker {
            name: "commit_worker",
            threads: n_threads,
            run: concurrent_commit_worker,
        }];

        runner.concurrent_group::<ConcurrentWalAppendCtx>("wal_concurrent_append", |g| {
            g.sample_duration(Duration::from_millis(200))
                .throughput(Throughput::per_operation(1, "pages"))
                .metadata("sync_backend", sync_backend)
                .metadata("direct_io", direct_io)
                .metadata("delay_min_us", delay_min_us.to_string())
                .metadata("delay_max_us", delay_max_us.to_string())
                .metadata("target_records", target_records.to_string())
                .metadata("case_seed", case_seed.to_string())
                .metadata("case_cooldown_ms", case_cooldown_ms.to_string())
                .lifecycle(|| ResetWalLifecycle)
                .factory(&append_factory)
                .bench(&benchmark_name, &append_workers);
        });

        runner.concurrent_group::<ConcurrentWalCommitStrictCtx>(
            "wal_concurrent_commit_strict",
            |g| {
                g.sample_duration(Duration::from_millis(200))
                    .throughput(Throughput::per_operation(1, "commits"))
                    .measurement_domain(MeasurementDomain::Io)
                    .metadata("sync_backend", sync_backend)
                    .metadata("direct_io", direct_io)
                    .metadata("delay_min_us", delay_min_us.to_string())
                    .metadata("delay_max_us", delay_max_us.to_string())
                    .metadata("target_records", target_records.to_string())
                    .metadata("case_seed", case_seed.to_string())
                    .metadata("case_cooldown_ms", case_cooldown_ms.to_string())
                    .lifecycle(StrictWalLifecycle::default)
                    .factory(&strict_factory)
                    .bench(&benchmark_name, &commit_workers);
            },
        );

        let relaxed_ctx = OnceLock::<ConcurrentWalCommitRelaxedCtx>::new();
        let relaxed_factory = |num_threads| {
            relaxed_ctx
                .get_or_init(|| {
                    ConcurrentWalCommitRelaxedCtx(ConcurrentWalCtx::prepare_with_mode(
                        CommitMode::Relaxed,
                        num_threads,
                    ))
                })
                .clone()
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
                    .metadata("sync_backend", sync_backend)
                    .metadata("direct_io", direct_io)
                    .metadata("delay_min_us", delay_min_us.to_string())
                    .metadata("delay_max_us", delay_max_us.to_string())
                    .metadata("target_records", target_records.to_string())
                    .metadata("case_seed", case_seed.to_string())
                    .metadata("case_cooldown_ms", case_cooldown_ms.to_string())
                    .lifecycle(|| ResetWalLifecycle)
                    .factory(&relaxed_factory)
                    .bench(&benchmark_name, &relaxed_workers);
            },
        );
    }
});
