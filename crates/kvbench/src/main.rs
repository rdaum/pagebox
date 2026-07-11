//! `kvbench` — KV benchmark harness.
//!
//! Drives synthetic YCSB / db_bench workloads against embedded KV engines
//! through a uniform adapter trait. Non-interactive mode (`--no-tui`) takes
//! a TOML spec file and writes a JSON report.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

mod comparison;
mod distribution;
mod driver;
mod engine;
mod engines;
mod report;
mod stats;
mod workload;

use comparison::{ComparisonContract, MemoryRegime};
use driver::run_phase;
use engine::{EngineOpts, EngineStats, KvEngine};
use engines::kvstore_adapter::KvstoreAdapter;
use report::{Report, ReportMeasurements};
use workload::{WorkloadSpec, generate_load_ops, generate_run_ops, validate_spec};

#[cfg(feature = "fjall")]
use engines::fjall::FjallAdapter;
#[cfg(feature = "lmdb")]
use engines::lmdb::LmdbAdapter;
#[cfg(feature = "redb")]
use engines::redb::RedbAdapter;
#[cfg(feature = "rocksdb")]
use engines::rocksdb::RocksdbAdapter;

#[derive(Parser)]
#[command(name = "kvbench")]
#[command(about = "KV benchmark harness for pagebox")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a benchmark spec and write a JSON report.
    Run {
        /// Path to the TOML spec file.
        #[arg(long)]
        spec: PathBuf,
        /// Output JSON report path.
        #[arg(long, default_value = "report.json")]
        output: PathBuf,
        /// Engine to run ("kvstore", "fjall", "redb", "rocksdb", "lmdb").
        #[arg(long)]
        engine: String,
        /// Temp directory root (each run gets a fresh subdir).
        #[arg(long, default_value = "/tmp/kvbench")]
        tmpdir: PathBuf,
        /// Verify correctness against a shadow HashMap.
        #[arg(long)]
        verify: bool,
        /// Override thread count from spec. Useful for scaling sweeps.
        #[arg(long)]
        threads: Option<usize>,
        /// Override WAL backend from spec ("fdatasync", "pwritev2_dsync",
        /// "io_uring"). kvstore only.
        #[arg(long)]
        wal_backend: Option<String>,
        /// Repetition number recorded in the report.
        #[arg(long, default_value_t = 1)]
        iteration: u32,
    },
    /// Compare two JSON reports.
    Compare {
        /// First report.
        a: PathBuf,
        /// Second report.
        b: PathBuf,
    },
    /// Summarize all JSON reports in a directory into a consolidated table.
    Summarize {
        /// Directory containing `*.json` report files.
        dir: PathBuf,
    },
    /// Print the engine cohort declared by a benchmark spec.
    Cohort {
        /// Path to the TOML spec file.
        #[arg(long)]
        spec: PathBuf,
    },
    /// List available workloads.
    List,
}

/// A run configuration parsed from the spec file.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpecFile {
    #[serde(flatten)]
    spec: WorkloadSpec,
    comparison: ComparisonContract,
    #[serde(default)]
    engine_opts: EngineOpts,
}

struct RunInvocation {
    spec_path: PathBuf,
    output: PathBuf,
    engine_name: String,
    tmpdir: PathBuf,
    verify: bool,
    threads_override: Option<usize>,
    wal_backend_override: Option<String>,
    iteration: u32,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Run {
            spec,
            output,
            engine,
            tmpdir,
            verify,
            threads,
            wal_backend,
            iteration,
        } => run_spec_file(RunInvocation {
            spec_path: spec,
            output,
            engine_name: engine,
            tmpdir,
            verify,
            threads_override: threads,
            wal_backend_override: wal_backend,
            iteration,
        }),
        Command::Compare { a, b } => {
            let report_a: Report = read_report(&a)?;
            let report_b: Report = read_report(&b)?;
            report::print_comparison(&report_a, &report_b).map_err(invalid_input)?;
            Ok(())
        }
        Command::Summarize { dir } => {
            let reports = read_all_reports(&dir)?;
            report::print_summary_table(&reports);
            Ok(())
        }
        Command::Cohort { spec } => {
            let spec_file = read_spec_file(&spec)?;
            for engine in spec_file.comparison.engines {
                println!("{engine}");
            }
            Ok(())
        }
        Command::List => {
            println!("Available workloads:");
            for w in workload::Workload::ALL {
                println!("  {} - {}", w.name(), workload_description(*w));
            }
            println!("\nAvailable engines:");
            println!("  kvstore (always)");
            #[cfg(feature = "fjall")]
            println!("  fjall (--features fjall)");
            #[cfg(feature = "redb")]
            println!("  redb (--features redb)");
            #[cfg(feature = "lmdb")]
            println!("  lmdb (--features lmdb)");
            #[cfg(feature = "rocksdb")]
            println!("  rocksdb (--features rocksdb)");
            Ok(())
        }
    }
}

fn workload_description(w: workload::Workload) -> &'static str {
    match w {
        workload::Workload::FillRandom => "Random-order bulk put",
        workload::Workload::FillSeq => "Sequential bulk put",
        workload::Workload::ReadRandom => "Random point lookups",
        workload::Workload::Overwrite => "In-place update churn",
        workload::Workload::YcsbA => "50% read / 50% update (uniform)",
        workload::Workload::YcsbB => "95% read / 5% update (uniform)",
        workload::Workload::YcsbC => "100% read (uniform)",
        workload::Workload::YcsbD => "95% read / 5% insert (latest)",
        workload::Workload::YcsbE => "95% short scan / 5% insert (zipfian)",
        workload::Workload::YcsbF => "100% read-modify-write (zipfian)",
        workload::Workload::ReadSeq => "Sequential reads",
        workload::Workload::ReadWhileWriting => "90% read / 10% write mix",
        workload::Workload::DeleteRandom => "Random deletes",
        workload::Workload::DeleteSeq => "Sequential deletes",
        workload::Workload::SeekRandom => "Random range seeks",
    }
}

fn run_spec_file(invocation: RunInvocation) -> std::io::Result<()> {
    let spec_file = read_spec_file(&invocation.spec_path)?;

    let mut spec = spec_file.spec.clone();
    if let Some(t) = invocation.threads_override {
        spec.threads = t.max(1);
    }
    validate_spec(&spec).map_err(invalid_input)?;
    let mut opts = spec_file.engine_opts.clone();
    if let Some(ref b) = invocation.wal_backend_override {
        opts.wal_backend = Some(b.clone());
    }

    eprintln!(
        "Running {} / {} (comparison={}, records={}, ops={}, threads={}, iteration={})",
        invocation.engine_name,
        spec.workload.name(),
        spec_file.comparison.id,
        spec.record_count,
        spec.operation_count,
        spec.threads,
        invocation.iteration,
    );

    let dir = create_fresh_dir(&invocation.tmpdir, &invocation.engine_name)?;
    let report = match invocation.engine_name.as_str() {
        "kvstore" => run_engine::<KvstoreAdapter>(
            dir.path(),
            &spec_file.comparison,
            invocation.iteration,
            &spec,
            &opts,
            invocation.verify,
        )?,
        #[cfg(feature = "fjall")]
        "fjall" => run_engine::<FjallAdapter>(
            dir.path(),
            &spec_file.comparison,
            invocation.iteration,
            &spec,
            &opts,
            invocation.verify,
        )?,
        #[cfg(feature = "redb")]
        "redb" => run_engine::<RedbAdapter>(
            dir.path(),
            &spec_file.comparison,
            invocation.iteration,
            &spec,
            &opts,
            invocation.verify,
        )?,
        #[cfg(feature = "lmdb")]
        "lmdb" => run_engine::<LmdbAdapter>(
            dir.path(),
            &spec_file.comparison,
            invocation.iteration,
            &spec,
            &opts,
            invocation.verify,
        )?,
        #[cfg(feature = "rocksdb")]
        "rocksdb" => run_engine::<RocksdbAdapter>(
            dir.path(),
            &spec_file.comparison,
            invocation.iteration,
            &spec,
            &opts,
            invocation.verify,
        )?,
        _ => {
            let available = available_engines();
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "unknown engine '{}'. Available: {available}",
                    invocation.engine_name
                ),
            ));
        }
    };

    report.write_to_file(&invocation.output)?;
    eprintln!("Report written to {}", invocation.output.display());

    // Print summary to stderr.
    eprintln!();
    eprintln!(
        "=== Results: {} / {} ===",
        invocation.engine_name,
        spec.workload.name()
    );
    print_summary(&report);

    Ok(())
}

fn read_spec_file(spec_path: &std::path::Path) -> std::io::Result<SpecFile> {
    let spec_contents = std::fs::read_to_string(spec_path)?;
    toml::from_str(&spec_contents).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("spec parse error: {e}"),
        )
    })
}

fn run_engine<E: KvEngine>(
    dir: &std::path::Path,
    comparison: &ComparisonContract,
    iteration: u32,
    spec: &WorkloadSpec,
    opts: &EngineOpts,
    verify: bool,
) -> std::io::Result<Report> {
    comparison
        .validate::<E>(spec, opts)
        .map_err(invalid_input)?;
    let load_opts = load_engine_opts(opts, comparison.memory_regime);
    let mut engine = E::open(dir, &load_opts)?;
    let engine_name = E::NAME;
    let shadow = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));

    // Load phase.
    let load_summary = if spec.workload.needs_load_phase() {
        eprintln!("  Loading {} records...", spec.record_count);
        let load_ops = generate_load_ops(spec);
        let load_stats = if verify {
            driver::run_load_phase_verify(&engine, &load_ops, spec.threads.max(1), &shadow)
        } else {
            driver::run_load_phase(&engine, &load_ops, spec.threads.max(1))
        };
        eprintln!(
            "  Load: {:.0} ops/sec, {:.2}ms",
            load_stats.ops_per_sec(),
            load_stats.duration.as_secs_f64() * 1000.0
        );
        Some((&load_stats).into())
    } else {
        None
    };

    let load_durability_drain_secs = if load_summary.is_some() {
        let drain_start = Instant::now();
        if uses_bounded_cold_start(comparison.memory_regime) {
            engine.prepare_for_reopen()?;
        } else {
            engine.sync()?;
        }
        Some(drain_start.elapsed().as_secs_f64())
    } else {
        None
    };

    if load_summary.is_some() && uses_bounded_cold_start(comparison.memory_regime) {
        drop(engine);
        engine = E::open(dir, opts)?;
    }

    // Run phase.
    eprintln!("  Running {} operations...", spec.operation_count);
    let run_ops = generate_run_ops(spec);
    let minimum_duration = Duration::from_secs(spec.minimum_duration_secs);
    let run_stats = if verify {
        driver::run_phase_verify(
            &engine,
            &run_ops,
            spec.threads.max(1),
            minimum_duration,
            &shadow,
        )
    } else {
        run_phase(&engine, &run_ops, spec.threads.max(1), minimum_duration)
    };

    let durability_drain_secs =
        if spec.workload.has_mutations() && opts.sync_mode == crate::engine::SyncMode::Relaxed {
            let drain_start = Instant::now();
            engine.sync()?;
            drain_start.elapsed().as_secs_f64()
        } else {
            0.0
        };

    let engine_stats = engine.stats();
    if uses_bounded_cold_start(comparison.memory_regime) {
        validate_cache_pressure_evidence(E::NAME, opts, &engine_stats)?;
    }

    let report = Report::new(
        engine_name,
        comparison.clone(),
        iteration,
        spec.clone(),
        opts.clone(),
        ReportMeasurements {
            load_phase: load_summary,
            load_durability_drain_secs,
            run_phase: (&run_stats).into(),
            durability_drain_secs,
            engine_stats,
        },
    );

    // Clean up.
    drop(engine);
    let _ = std::fs::remove_dir_all(dir);

    Ok(report)
}

fn uses_bounded_cold_start(regime: MemoryRegime) -> bool {
    matches!(
        regime,
        MemoryRegime::ApplicationCache | MemoryRegime::DirectIoApplicationCache
    )
}

fn load_engine_opts(opts: &EngineOpts, regime: MemoryRegime) -> EngineOpts {
    let mut load_opts = opts.clone();
    if regime == MemoryRegime::DirectIoApplicationCache {
        // The direct-I/O contract applies to the measured phase. Building the
        // same persisted data set through buffered writes avoids turning an
        // untimed setup phase into a direct-write/WAL benchmark.
        load_opts.direct_io = false;
    }
    load_opts
}

fn validate_cache_pressure_evidence(
    engine: &str,
    opts: &EngineOpts,
    stats: &EngineStats,
) -> std::io::Result<()> {
    let capacity = stats.cache_capacity_bytes.ok_or_else(|| {
        invalid_input(format!(
            "engine '{engine}' did not report cache capacity for a cache-pressure run"
        ))
    })?;
    if capacity != opts.cache_budget_bytes as u64 {
        return Err(invalid_input(format!(
            "engine '{engine}' reported {capacity} cache bytes, expected {}",
            opts.cache_budget_bytes
        )));
    }
    let working_set = stats.live_data_bytes.or(stats.persisted_data_bytes).ok_or_else(|| {
        invalid_input(format!(
            "engine '{engine}' did not report live or persisted data size for a cache-pressure run"
        ))
    })?;
    if working_set <= capacity {
        return Err(invalid_input(format!(
            "engine '{engine}' did not establish cache pressure: {working_set} working-set bytes fit in the {capacity}-byte cache"
        )));
    }
    let misses = stats.cache_misses.ok_or_else(|| {
        invalid_input(format!(
            "engine '{engine}' did not report cache misses for a cache-pressure run"
        ))
    })?;
    if misses == 0 {
        return Err(invalid_input(format!(
            "engine '{engine}' reported zero cache misses during a cache-pressure run"
        )));
    }
    let observed_turnover = stats.cache_evictions.is_some_and(|evictions| evictions > 0)
        || stats
            .cache_insert_bytes
            .is_some_and(|inserted| inserted > capacity);
    if !observed_turnover {
        return Err(invalid_input(format!(
            "engine '{engine}' did not prove cache turnover with evictions or more than one cache capacity of inserted bytes"
        )));
    }
    if opts.direct_io && stats.direct_io != Some(true) {
        return Err(invalid_input(format!(
            "engine '{engine}' did not confirm direct data-file I/O"
        )));
    }
    Ok(())
}

fn create_fresh_dir(
    tmpdir: &std::path::Path,
    engine_name: &str,
) -> std::io::Result<tempfile::TempDir> {
    std::fs::create_dir_all(tmpdir)?;
    tempfile::Builder::new()
        .prefix(&format!("{engine_name}-"))
        .tempdir_in(tmpdir)
}

fn read_report(path: &std::path::Path) -> std::io::Result<Report> {
    let contents = std::fs::read_to_string(path)?;
    serde_json::from_str(&contents).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("report parse error: {e}"),
        )
    })
}

/// Read all `*.json` report files from a directory, sorted by filename.
fn read_all_reports(dir: &std::path::Path) -> std::io::Result<Vec<Report>> {
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();

    let mut reports = Vec::new();
    for path in &paths {
        match read_report(path) {
            Ok(r) => reports.push(r),
            Err(e) => eprintln!("  warning: skipping {}: {}", path.display(), e),
        }
    }
    Ok(reports)
}

fn available_engines() -> String {
    let mut engines = vec!["kvstore"];
    if cfg!(feature = "fjall") {
        engines.push("fjall");
    }
    if cfg!(feature = "redb") {
        engines.push("redb");
    }
    if cfg!(feature = "lmdb") {
        engines.push("lmdb");
    }
    if cfg!(feature = "rocksdb") {
        engines.push("rocksdb");
    }
    engines.join(", ")
}

fn print_summary(report: &Report) {
    let r = &report.run_phase;
    eprintln!("  visible:    {:.0} ops/sec", r.ops_per_sec);
    eprintln!("  p50:        {:.2} us", r.p50_us);
    eprintln!("  p95:        {:.2} us", r.p95_us);
    eprintln!("  p99:        {:.2} us", r.p99_us);
    eprintln!("  p99.9:      {:.2} us", r.p999_us);
    eprintln!("  duration:   {:.3} s", r.duration_secs);
    eprintln!("  drain:      {:.3} s", report.durability_drain_secs);
    eprintln!("  incl drain: {:.0} ops/sec", report.drained_ops_per_sec);
    eprintln!(
        "  puts/gets/dels/scans/rmws: {}/{}/{}/{}/{}",
        r.puts, r.gets, r.dels, r.scans, r.rmws
    );
    let stats = &report.engine_stats;
    if let (Some(used), Some(capacity), Some(persisted)) = (
        stats.cache_used_bytes,
        stats.cache_capacity_bytes,
        stats.persisted_data_bytes,
    ) {
        eprintln!(
            "  cache:      {:.1}/{:.1} MiB; persisted {:.1} MiB",
            used as f64 / 1_048_576.0,
            capacity as f64 / 1_048_576.0,
            persisted as f64 / 1_048_576.0,
        );
    }
    if let Some(live) = stats.live_data_bytes {
        eprintln!("  live data:  {:.1} MiB", live as f64 / 1_048_576.0);
    }
    if let (Some(hits), Some(misses)) = (stats.cache_hits, stats.cache_misses) {
        eprintln!("  cache h/m: {hits}/{misses}");
    } else if let Some(misses) = stats.cache_misses {
        eprintln!("  cache miss: {misses}");
    }
    if let Some(evictions) = stats.cache_evictions {
        eprintln!("  evictions:  {evictions}");
    }
}

fn invalid_input(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_specs_have_valid_comparison_contracts() {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let specs_dir = manifest_dir.join("specs");
        let mut spec_paths: Vec<_> = std::fs::read_dir(specs_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "toml")
            })
            .collect();
        spec_paths.sort();

        assert!(
            !spec_paths.is_empty(),
            "kvbench should have checked-in specs"
        );
        for path in spec_paths {
            let spec_file = read_spec_file(&path)
                .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()));
            validate_spec(&spec_file.spec)
                .unwrap_or_else(|error| panic!("invalid workload in {}: {error}", path.display()));
            spec_file
                .comparison
                .validate_workload(&spec_file.spec, &spec_file.engine_opts)
                .unwrap_or_else(|error| {
                    panic!("invalid comparison in {}: {error}", path.display())
                });
            if spec_file.comparison.memory_regime
                == crate::comparison::MemoryRegime::ApplicationCache
            {
                assert!(
                    !spec_file
                        .comparison
                        .engines
                        .iter()
                        .any(|engine| engine == "lmdb"),
                    "{} must exclude LMDB from application-cache comparisons",
                    path.display()
                );
            }
        }
    }

    #[test]
    fn cache_pressure_requires_oversized_persisted_data_and_misses() {
        let opts = EngineOpts {
            cache_budget_bytes: 64,
            ..EngineOpts::default()
        };
        let valid = EngineStats {
            cache_capacity_bytes: Some(64),
            persisted_data_bytes: Some(256),
            cache_misses: Some(1),
            cache_evictions: Some(1),
            ..EngineStats::default()
        };
        validate_cache_pressure_evidence("mock", &opts, &valid).unwrap();

        let fits = EngineStats {
            persisted_data_bytes: Some(64),
            ..valid.clone()
        };
        assert!(
            validate_cache_pressure_evidence("mock", &opts, &fits).is_err(),
            "a data set that fits in cache is not cache pressure"
        );
        let allocated_but_not_live = EngineStats {
            live_data_bytes: Some(32),
            ..valid.clone()
        };
        assert!(
            validate_cache_pressure_evidence("mock", &opts, &allocated_but_not_live).is_err(),
            "allocated file high-water space must not substitute for a live working set"
        );
        let no_misses = EngineStats {
            cache_misses: Some(0),
            ..valid.clone()
        };
        assert!(
            validate_cache_pressure_evidence("mock", &opts, &no_misses).is_err(),
            "cache-pressure reports need observed cache misses"
        );
        let no_turnover = EngineStats {
            cache_evictions: None,
            cache_insert_bytes: Some(64),
            ..valid
        };
        assert!(
            validate_cache_pressure_evidence("mock", &opts, &no_turnover).is_err(),
            "cache pressure must prove that bounded cache contents turned over"
        );
    }

    #[test]
    fn direct_io_pressure_requires_runtime_confirmation() {
        let opts = EngineOpts {
            cache_budget_bytes: 64,
            direct_io: true,
            ..EngineOpts::default()
        };
        let stats = EngineStats {
            direct_io: Some(false),
            cache_capacity_bytes: Some(64),
            persisted_data_bytes: Some(256),
            cache_misses: Some(1),
            cache_evictions: Some(1),
            ..EngineStats::default()
        };
        assert!(
            validate_cache_pressure_evidence("mock", &opts, &stats).is_err(),
            "requesting direct I/O must not silently accept buffered fallback"
        );
    }

    #[test]
    fn direct_io_pressure_builds_the_data_set_with_buffered_io() {
        let opts = EngineOpts {
            direct_io: true,
            ..EngineOpts::default()
        };
        let load_opts = load_engine_opts(&opts, MemoryRegime::DirectIoApplicationCache);
        assert!(!load_opts.direct_io);
        assert!(
            opts.direct_io,
            "the measured-phase options must stay direct"
        );
    }
}
