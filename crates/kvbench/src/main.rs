//! `kvbench` — KV benchmark harness.
//!
//! Drives synthetic YCSB / db_bench workloads against embedded KV engines
//! through a uniform adapter trait. Non-interactive mode (`--no-tui`) takes
//! a TOML spec file and writes a JSON report.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

mod distribution;
mod driver;
mod engine;
mod engines;
mod report;
mod stats;
mod workload;

use driver::run_phase;
use engine::{EngineOpts, KvEngine};
use engines::kvstore_adapter::KvstoreAdapter;
use report::Report;
use workload::{WorkloadSpec, generate_load_ops, generate_run_ops};

#[cfg(feature = "fjall")]
use engines::fjall::FjallAdapter;
#[cfg(feature = "redb")]
use engines::redb::RedbAdapter;
#[cfg(feature = "sled")]
use engines::sled::SledAdapter;

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
        /// Engine to run ("kvstore", "fjall", "redb", "sled").
        #[arg(long)]
        engine: String,
        /// Temp directory root (each run gets a fresh subdir).
        #[arg(long, default_value = "/tmp/kvbench")]
        tmpdir: PathBuf,
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
    /// List available workloads.
    List,
}

/// A run configuration parsed from the spec file.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpecFile {
    #[serde(flatten)]
    spec: WorkloadSpec,
    #[serde(default)]
    engine_opts: EngineOpts,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Run {
            spec,
            output,
            engine,
            tmpdir,
        } => run_spec_file(&spec, &output, &engine, &tmpdir),
        Command::Compare { a, b } => {
            let report_a: Report = read_report(&a)?;
            let report_b: Report = read_report(&b)?;
            report::print_comparison(&report_a, &report_b);
            Ok(())
        }
        Command::Summarize { dir } => {
            let reports = read_all_reports(&dir)?;
            report::print_summary_table(&reports);
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
            #[cfg(feature = "sled")]
            println!("  sled (--features sled)");
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

fn run_spec_file(
    spec_path: &std::path::Path,
    output: &std::path::Path,
    engine_name: &str,
    tmpdir: &std::path::Path,
) -> std::io::Result<()> {
    let spec_contents = std::fs::read_to_string(spec_path)?;
    let spec_file: SpecFile = toml::from_str(&spec_contents).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("spec parse error: {e}"),
        )
    })?;

    let spec = spec_file.spec.clone();
    let opts = spec_file.engine_opts.clone();
    eprintln!(
        "Running {} / {} (records={}, ops={}, threads={})",
        engine_name,
        spec.workload.name(),
        spec.record_count,
        spec.operation_count,
        spec.threads
    );

    let dir = create_fresh_dir(tmpdir, engine_name)?;
    let report = match engine_name {
        "kvstore" => run_engine::<KvstoreAdapter>(&dir, &spec, &opts)?,
        "betree" => run_engine_custom::<KvstoreAdapter>(&dir, &spec, &opts, "betree")?,
        "betree-nw" => run_engine_custom_nw::<KvstoreAdapter>(&dir, &spec, &opts, "betree-nw")?,
        #[cfg(feature = "fjall")]
        "fjall" => run_engine::<FjallAdapter>(&dir, &spec, &opts)?,
        #[cfg(feature = "redb")]
        "redb" => run_engine::<RedbAdapter>(&dir, &spec, &opts)?,
        #[cfg(feature = "sled")]
        "sled" => run_engine::<SledAdapter>(&dir, &spec, &opts)?,
        _ => {
            let available = available_engines();
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unknown engine '{engine_name}'. Available: {available}",),
            ));
        }
    };

    report.write_to_file(output)?;
    eprintln!("Report written to {}", output.display());

    // Print summary to stderr.
    eprintln!();
    eprintln!(
        "=== Results: {} / {} ===",
        engine_name,
        spec.workload.name()
    );
    print_summary(&report);

    Ok(())
}

fn run_engine_custom<E: KvEngine>(
    dir: &std::path::Path,
    spec: &WorkloadSpec,
    opts: &EngineOpts,
    display_name: &str,
) -> std::io::Result<Report> {
    let mut custom_opts = opts.clone();
    custom_opts
        .engine_specific
        .insert("tree_backend".to_string(), "betree".to_string());
    let engine = E::open(dir, &custom_opts)?;
    let load_summary = if spec.workload.needs_load_phase() {
        eprintln!("  Loading {} records...", spec.record_count);
        let load_ops = generate_load_ops(spec, &custom_opts);
        let load_stats = run_phase(&engine, &load_ops, spec.threads.max(1));
        eprintln!(
            "  Load: {:.0} ops/sec, {:.2}ms",
            load_stats.ops_per_sec(),
            load_stats.duration.as_secs_f64() * 1000.0
        );
        Some((&load_stats).into())
    } else {
        None
    };
    let _ = engine.sync();
    eprintln!("  Running {} operations...", spec.operation_count);
    let run_ops = generate_run_ops(spec, &custom_opts);
    let run_stats = run_phase(&engine, &run_ops, spec.threads.max(1));
    let report = Report::new(
        display_name,
        spec.clone(),
        custom_opts.clone(),
        load_summary,
        (&run_stats).into(),
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(dir);
    Ok(report)
}

fn run_engine_custom_nw<E: KvEngine>(
    dir: &std::path::Path,
    spec: &WorkloadSpec,
    opts: &EngineOpts,
    display_name: &str,
) -> std::io::Result<Report> {
    let mut custom_opts = opts.clone();
    custom_opts
        .engine_specific
        .insert("tree_backend".to_string(), "betree-nw".to_string());
    let engine = E::open(dir, &custom_opts)?;
    let load_summary = if spec.workload.needs_load_phase() {
        eprintln!("  Loading {} records...", spec.record_count);
        let load_ops = generate_load_ops(spec, &custom_opts);
        let load_stats = run_phase(&engine, &load_ops, spec.threads.max(1));
        eprintln!(
            "  Load: {:.0} ops/sec, {:.2}ms",
            load_stats.ops_per_sec(),
            load_stats.duration.as_secs_f64() * 1000.0
        );
        Some((&load_stats).into())
    } else {
        None
    };
    let _ = engine.sync();
    eprintln!("  Running {} operations...", spec.operation_count);
    let run_ops = generate_run_ops(spec, &custom_opts);
    let run_stats = run_phase(&engine, &run_ops, spec.threads.max(1));
    let report = Report::new(
        display_name,
        spec.clone(),
        custom_opts.clone(),
        load_summary,
        (&run_stats).into(),
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(dir);
    Ok(report)
}

fn run_engine<E: KvEngine>(
    dir: &std::path::Path,
    spec: &WorkloadSpec,
    opts: &EngineOpts,
) -> std::io::Result<Report> {
    let engine = E::open(dir, opts)?;
    let engine_name = E::NAME;

    // Load phase.
    let load_summary = if spec.workload.needs_load_phase() {
        eprintln!("  Loading {} records...", spec.record_count);
        let load_ops = generate_load_ops(spec, opts);
        let load_stats = run_phase(&engine, &load_ops, spec.threads.max(1));
        eprintln!(
            "  Load: {:.0} ops/sec, {:.2}ms",
            load_stats.ops_per_sec(),
            load_stats.duration.as_secs_f64() * 1000.0
        );
        Some((&load_stats).into())
    } else {
        None
    };

    // Sync after load.
    let _ = engine.sync();

    // Run phase.
    eprintln!("  Running {} operations...", spec.operation_count);
    let run_ops = generate_run_ops(spec, opts);
    let run_stats = run_phase(&engine, &run_ops, spec.threads.max(1));

    let report = Report::new(
        engine_name,
        spec.clone(),
        opts.clone(),
        load_summary,
        (&run_stats).into(),
    );

    // Clean up.
    drop(engine);
    let _ = std::fs::remove_dir_all(dir);

    Ok(report)
}

fn create_fresh_dir(
    tmpdir: &std::path::Path,
    engine_name: &str,
) -> std::io::Result<std::path::PathBuf> {
    let dir = tmpdir.join(format!("{}-{}", engine_name, std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
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
    let mut engines = vec!["kvstore", "betree", "betree-nw"];
    if cfg!(feature = "fjall") {
        engines.push("fjall");
    }
    if cfg!(feature = "redb") {
        engines.push("redb");
    }
    if cfg!(feature = "sled") {
        engines.push("sled");
    }
    engines.join(", ")
}

fn print_summary(report: &Report) {
    let r = &report.run_phase;
    eprintln!("  ops/sec:    {:.0}", r.ops_per_sec);
    eprintln!("  p50:        {:.2} us", r.p50_us);
    eprintln!("  p95:        {:.2} us", r.p95_us);
    eprintln!("  p99:        {:.2} us", r.p99_us);
    eprintln!("  p99.9:      {:.2} us", r.p999_us);
    eprintln!("  duration:   {:.3} s", r.duration_secs);
    eprintln!(
        "  puts/gets/dels/scans/rmws: {}/{}/{}/{}/{}",
        r.puts, r.gets, r.dels, r.scans, r.rmws
    );
}
