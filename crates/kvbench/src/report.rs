//! JSON report serialization.
//!
//! Each report is a self-contained JSON file with schema version, engine name,
//! full workload spec, git commit hash, and load/run phase summaries. A
//! reported number is reproducible from the report alone.

use serde::{Deserialize, Serialize};

use crate::comparison::ComparisonContract;
use crate::engine::{EngineOpts, EngineStats};
use crate::stats::PhaseSummary;
use crate::workload::WorkloadSpec;

/// JSON report schema version.
pub const SCHEMA_VERSION: u32 = 4;

/// A complete benchmark report for one run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub schema: u32,
    pub engine: String,
    pub comparison: ComparisonContract,
    pub iteration: u32,
    pub spec: WorkloadSpec,
    pub engine_opts: EngineOpts,
    pub git_commit: String,
    /// Git object hash of the running executable. This disambiguates dirty
    /// builds whose source commit alone is not reproducible.
    pub binary_hash: String,
    pub load_phase: Option<PhaseSummary>,
    pub load_durability_drain_secs: Option<f64>,
    pub load_drained_ops_per_sec: Option<f64>,
    pub run_phase: PhaseSummary,
    /// Time required after the measured phase to make every preceding
    /// mutation durable. This is near-zero for read-only and strict runs.
    pub durability_drain_secs: f64,
    /// Run-phase operations divided by run time plus durability drain time.
    pub drained_ops_per_sec: f64,
    /// Cache and persisted-data evidence captured after the measured phase.
    pub engine_stats: EngineStats,
}

/// Phase measurements supplied when constructing a report.
pub struct ReportMeasurements {
    pub load_phase: Option<PhaseSummary>,
    pub load_durability_drain_secs: Option<f64>,
    pub run_phase: PhaseSummary,
    pub durability_drain_secs: f64,
    pub engine_stats: EngineStats,
}

impl Report {
    /// Create a new report with the current git commit hash.
    pub fn new(
        engine: &str,
        comparison: ComparisonContract,
        iteration: u32,
        spec: WorkloadSpec,
        engine_opts: EngineOpts,
        measurements: ReportMeasurements,
    ) -> Self {
        let ReportMeasurements {
            load_phase,
            load_durability_drain_secs,
            run_phase,
            durability_drain_secs,
            engine_stats,
        } = measurements;
        let drained_duration = run_phase.duration_secs + durability_drain_secs;
        let drained_ops_per_sec = if drained_duration == 0.0 {
            0.0
        } else {
            run_phase.operations as f64 / drained_duration
        };
        let load_drained_ops_per_sec = load_phase.as_ref().and_then(|phase| {
            load_durability_drain_secs
                .map(|drain| phase.operations as f64 / (phase.duration_secs + drain))
        });
        Self {
            schema: SCHEMA_VERSION,
            engine: engine.to_string(),
            comparison,
            iteration,
            spec,
            engine_opts,
            git_commit: git_commit_hash(),
            binary_hash: binary_hash(),
            load_phase,
            load_durability_drain_secs,
            load_drained_ops_per_sec,
            run_phase,
            durability_drain_secs,
            drained_ops_per_sec,
            engine_stats,
        }
    }

    /// Serialize to JSON pretty-printed string.
    pub fn to_json_pretty(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Write to a file.
    pub fn write_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = self
            .to_json_pretty()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// Hash the executable with Git's object hash algorithm. Unlike the source
/// commit, this changes when kvbench is built from uncommitted changes.
fn binary_hash() -> String {
    let Ok(executable) = std::env::current_exe() else {
        return "unknown".to_string();
    };
    std::process::Command::new("git")
        .arg("hash-object")
        .arg(executable)
        .output()
        .ok()
        .and_then(|output| {
            output
                .status
                .success()
                .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get the current git commit hash (short). Returns `"unknown"` if git is
/// unavailable or this isn't a git repo.
fn git_commit_hash() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Compare two reports and return a summary of the differences.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonRow {
    pub metric: String,
    pub a: String,
    pub b: String,
    pub delta_pct: f64,
}

/// Print a side-by-side comparison of two compatible reports to stdout.
pub fn print_comparison(a: &Report, b: &Report) -> Result<(), String> {
    if comparison_identity(a, false) != comparison_identity(b, false) {
        return Err(format!(
            "reports are not comparable: '{}' and '{}' have different contracts or workload settings",
            a.comparison.id, b.comparison.id
        ));
    }
    println!(
        "{:<20} {:>20} {:>20} {:>10}",
        "metric", a.engine, b.engine, "delta %"
    );
    println!("{}", "-".repeat(72));

    let rows = [
        (
            "visible ops/sec",
            a.run_phase.ops_per_sec,
            b.run_phase.ops_per_sec,
        ),
        ("p50_us", a.run_phase.p50_us, b.run_phase.p50_us),
        ("p95_us", a.run_phase.p95_us, b.run_phase.p95_us),
        ("p99_us", a.run_phase.p99_us, b.run_phase.p99_us),
        (
            "drained ops/sec",
            a.drained_ops_per_sec,
            b.drained_ops_per_sec,
        ),
        (
            "drain_ms",
            a.durability_drain_secs * 1000.0,
            b.durability_drain_secs * 1000.0,
        ),
    ];

    for (name, va, vb) in rows {
        let delta = if va != 0.0 {
            (vb - va) / va * 100.0
        } else {
            0.0
        };
        println!("{:<20} {:>20.2} {:>20.2} {:>9.1}%", name, va, vb, delta);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Consolidated summary table
// ---------------------------------------------------------------------------

/// A short label identifying one comparison configuration.
fn scenario_label(report: &Report) -> String {
    let binary = report.binary_hash.get(..8).unwrap_or(&report.binary_hash);
    format!(
        "{} {}t {}/{}",
        report.comparison.id, report.spec.threads, report.git_commit, binary
    )
}

fn comparison_identity(report: &Report, include_commit: bool) -> String {
    let commit = include_commit.then_some(report.git_commit.as_str());
    serde_json::to_string(&(
        commit,
        include_commit.then_some(report.binary_hash.as_str()),
        &report.comparison,
        &report.spec,
        &report.engine_opts,
    ))
    .expect("comparison identity should serialize")
}

/// Format a number with thousands separators (e.g. 7813308 -> "7,813,308").
fn fmt_thousands(n: f64) -> String {
    let s = format!("{:.0}", n);
    let bytes = s.as_bytes();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    let len = bytes.len();
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            result.push(',');
        }
        result.push(b as char);
    }
    result
}

#[derive(Debug)]
struct SummaryRow<'a> {
    engine: &'a str,
    runs: usize,
    ops_per_sec: f64,
    drained_ops_per_sec: f64,
    drain_ms: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    p999_us: f64,
    load_ops_per_sec: Option<f64>,
    load_drained_ops_per_sec: Option<f64>,
    load_drain_ms: Option<f64>,
    cache_capacity_mib: Option<f64>,
    cache_used_mib: Option<f64>,
    live_data_mib: Option<f64>,
    persisted_data_mib: Option<f64>,
    cache_hit_pct: Option<f64>,
    cache_misses: Option<f64>,
    cache_evictions: Option<f64>,
    cache_turnovers: Option<f64>,
    storage_read_mib: Option<f64>,
    direct_io: Option<bool>,
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn median_some(values: Vec<f64>) -> Option<f64> {
    (!values.is_empty()).then(|| median(values))
}

fn summarize_engine<'a>(engine: &'a str, reports: &[&Report]) -> SummaryRow<'a> {
    let load_values: Vec<f64> = reports
        .iter()
        .filter_map(|report| report.load_phase.as_ref().map(|phase| phase.ops_per_sec))
        .collect();
    let load_drained_values: Vec<f64> = reports
        .iter()
        .filter_map(|report| report.load_drained_ops_per_sec)
        .collect();
    let load_drain_values: Vec<f64> = reports
        .iter()
        .filter_map(|report| {
            report
                .load_durability_drain_secs
                .map(|duration| duration * 1000.0)
        })
        .collect();
    SummaryRow {
        engine,
        runs: reports.len(),
        ops_per_sec: median(
            reports
                .iter()
                .map(|report| report.run_phase.ops_per_sec)
                .collect(),
        ),
        drained_ops_per_sec: median(
            reports
                .iter()
                .map(|report| report.drained_ops_per_sec)
                .collect(),
        ),
        drain_ms: median(
            reports
                .iter()
                .map(|report| report.durability_drain_secs * 1000.0)
                .collect(),
        ),
        p50_us: median(
            reports
                .iter()
                .map(|report| report.run_phase.p50_us)
                .collect(),
        ),
        p95_us: median(
            reports
                .iter()
                .map(|report| report.run_phase.p95_us)
                .collect(),
        ),
        p99_us: median(
            reports
                .iter()
                .map(|report| report.run_phase.p99_us)
                .collect(),
        ),
        p999_us: median(
            reports
                .iter()
                .map(|report| report.run_phase.p999_us)
                .collect(),
        ),
        load_ops_per_sec: (!load_values.is_empty()).then(|| median(load_values)),
        load_drained_ops_per_sec: (!load_drained_values.is_empty())
            .then(|| median(load_drained_values)),
        load_drain_ms: (!load_drain_values.is_empty()).then(|| median(load_drain_values)),
        cache_capacity_mib: median_some(
            reports
                .iter()
                .filter_map(|report| report.engine_stats.cache_capacity_bytes)
                .map(|bytes| bytes as f64 / 1_048_576.0)
                .collect(),
        ),
        cache_used_mib: median_some(
            reports
                .iter()
                .filter_map(|report| report.engine_stats.cache_used_bytes)
                .map(|bytes| bytes as f64 / 1_048_576.0)
                .collect(),
        ),
        live_data_mib: median_some(
            reports
                .iter()
                .filter_map(|report| report.engine_stats.live_data_bytes)
                .map(|bytes| bytes as f64 / 1_048_576.0)
                .collect(),
        ),
        persisted_data_mib: median_some(
            reports
                .iter()
                .filter_map(|report| report.engine_stats.persisted_data_bytes)
                .map(|bytes| bytes as f64 / 1_048_576.0)
                .collect(),
        ),
        cache_hit_pct: median_some(
            reports
                .iter()
                .filter_map(|report| {
                    let hits = report.engine_stats.cache_hits?;
                    let misses = report.engine_stats.cache_misses?;
                    let accesses = hits + misses;
                    (accesses != 0).then_some(hits as f64 / accesses as f64 * 100.0)
                })
                .collect(),
        ),
        cache_misses: median_some(
            reports
                .iter()
                .filter_map(|report| report.engine_stats.cache_misses)
                .map(|value| value as f64)
                .collect(),
        ),
        cache_evictions: median_some(
            reports
                .iter()
                .filter_map(|report| report.engine_stats.cache_evictions)
                .map(|value| value as f64)
                .collect(),
        ),
        cache_turnovers: median_some(
            reports
                .iter()
                .filter_map(|report| {
                    let inserted = report.engine_stats.cache_insert_bytes?;
                    let capacity = report.engine_stats.cache_capacity_bytes?;
                    (capacity != 0).then_some(inserted as f64 / capacity as f64)
                })
                .collect(),
        ),
        storage_read_mib: median_some(
            reports
                .iter()
                .filter_map(|report| report.engine_stats.storage_read_bytes)
                .map(|bytes| bytes as f64 / 1_048_576.0)
                .collect(),
        ),
        direct_io: reports
            .iter()
            .filter_map(|report| report.engine_stats.direct_io)
            .next(),
    }
}

/// Print a consolidated table of per-engine medians. Reports are grouped only
/// when their full comparison contract, workload settings, engine options,
/// source commit, and executable hash match.
pub fn print_summary_table(reports: &[Report]) {
    if reports.is_empty() {
        println!("(no reports found)");
        return;
    }

    let mut scenarios: Vec<(String, String, Vec<&Report>)> = Vec::new();
    for r in reports {
        let identity = comparison_identity(r, true);
        let label = scenario_label(r);
        if let Some(entry) = scenarios.iter_mut().find(|(key, _, _)| key == &identity) {
            entry.2.push(r);
        } else {
            scenarios.push((identity, label, vec![r]));
        }
    }

    println!("=== Consolidated Benchmark Report ===");
    println!(
        "{} runs across {} comparison configurations",
        reports.len(),
        scenarios.len(),
    );
    println!();

    // Column widths.
    let w_scenario = scenarios
        .iter()
        .map(|(_, label, _)| label.len())
        .max()
        .unwrap_or(8)
        .max(8);
    let w_engine = reports
        .iter()
        .map(|report| report.engine.len())
        .max()
        .unwrap_or(6)
        .max(6);

    // Header.
    println!(
        "{:w1$}  {:w2$}  {:>3}  {:>12}  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "scenario",
        "engine",
        "n",
        "visible/sec",
        "incl.drain",
        "drain_ms",
        "p50_us",
        "p95_us",
        "p99_us",
        "p99.9_us",
        w1 = w_scenario,
        w2 = w_engine,
    );
    println!(
        "{}",
        "-".repeat(
            w_scenario
                + w_engine
                + 2
                + 3
                + 2
                + 12
                + 2
                + 12
                + 2
                + 10
                + 2
                + 10
                + 2
                + 10
                + 2
                + 10
                + 2
                + 10
        )
    );

    for (_, label, group) in &scenarios {
        let mut rows = Vec::new();
        for engine in &group[0].comparison.engines {
            let engine_reports: Vec<&Report> = group
                .iter()
                .copied()
                .filter(|report| &report.engine == engine)
                .collect();
            if !engine_reports.is_empty() {
                rows.push(summarize_engine(engine, &engine_reports));
            }
        }
        let best_ops = rows
            .iter()
            .map(|row| row.ops_per_sec)
            .fold(0.0_f64, f64::max);
        let complete_cohort = rows.len() == group[0].comparison.engines.len();
        if !complete_cohort {
            let missing: Vec<&str> = group[0]
                .comparison
                .engines
                .iter()
                .filter(|engine| !rows.iter().any(|row| row.engine == engine.as_str()))
                .map(String::as_str)
                .collect();
            println!("# incomplete cohort; missing: {}", missing.join(", "));
        }
        let balanced_repetitions = rows
            .first()
            .is_some_and(|first| rows.iter().all(|row| row.runs == first.runs));
        if !balanced_repetitions {
            println!("# unbalanced repetitions; no winner selected");
        }
        let rankable = complete_cohort && balanced_repetitions;

        let engine_count = rows.len();
        for row in rows {
            let is_best = rankable && row.ops_per_sec == best_ops && engine_count > 1;
            let star = if is_best { " ★" } else { "  " };
            println!(
                "{:w1$}  {:w2$}  {:>3}  {:>12}  {:>12}  {:>9.2}  {:>9.2}  {:>9.2}  {:>9.2}  {:>9.2}{}",
                label,
                row.engine,
                row.runs,
                fmt_thousands(row.ops_per_sec),
                fmt_thousands(row.drained_ops_per_sec),
                row.drain_ms,
                row.p50_us,
                row.p95_us,
                row.p99_us,
                row.p999_us,
                star,
                w1 = w_scenario,
                w2 = w_engine,
            );
        }
    }

    println!();
    println!(
        "Values are medians when n > 1; ★ marks best visible/sec in a complete, balanced cohort"
    );
    println!();
    println!("Cache-pressure evidence (medians):");
    println!(
        "{:w1$}  {:w2$}  {:>13}  {:>10}  {:>10}  {:>8}  {:>12}  {:>12}  {:>10}  {:>10}  {:>6}",
        "scenario",
        "engine",
        "used/cap MiB",
        "live MiB",
        "file MiB",
        "hit %",
        "misses",
        "evictions",
        "turnover x",
        "read MiB",
        "direct",
        w1 = w_scenario,
        w2 = w_engine,
    );
    println!(
        "{}",
        "-".repeat(
            w_scenario
                + w_engine
                + 2
                + 13
                + 2
                + 10
                + 2
                + 10
                + 2
                + 8
                + 2
                + 12
                + 2
                + 12
                + 2
                + 10
                + 2
                + 10
                + 2
                + 6
        )
    );
    for (_, label, group) in &scenarios {
        if group[0].comparison.memory_regime == crate::comparison::MemoryRegime::Resident {
            continue;
        }
        for engine in &group[0].comparison.engines {
            let engine_reports: Vec<&Report> = group
                .iter()
                .copied()
                .filter(|report| &report.engine == engine)
                .collect();
            if engine_reports.is_empty() {
                continue;
            }
            let row = summarize_engine(engine, &engine_reports);
            if row.cache_capacity_mib.is_none() {
                continue;
            }
            let cache = match (row.cache_used_mib, row.cache_capacity_mib) {
                (Some(used), Some(capacity)) => format!("{used:.1}/{capacity:.1}"),
                _ => "-".to_string(),
            };
            let optional = |value: Option<f64>| {
                value.map_or_else(|| "-".to_string(), |value| format!("{value:.1}"))
            };
            let count = |value: Option<f64>| value.map_or_else(|| "-".to_string(), fmt_thousands);
            let direct = row
                .direct_io
                .map_or("-", |enabled| if enabled { "yes" } else { "no" });
            println!(
                "{:w1$}  {:w2$}  {:>13}  {:>10}  {:>10}  {:>8}  {:>12}  {:>12}  {:>10}  {:>10}  {:>6}",
                label,
                engine,
                cache,
                optional(row.live_data_mib),
                optional(row.persisted_data_mib),
                optional(row.cache_hit_pct),
                count(row.cache_misses),
                count(row.cache_evictions),
                optional(row.cache_turnovers),
                optional(row.storage_read_mib),
                direct,
                w1 = w_scenario,
                w2 = w_engine,
            );
        }
    }
    println!();
    println!("Load phase medians:");
    println!(
        "{:w1$}  {:w2$}  {:>12}  {:>12}  {:>10}",
        "scenario",
        "engine",
        "visible/sec",
        "incl.drain",
        "drain_ms",
        w1 = w_scenario,
        w2 = w_engine,
    );
    println!(
        "{}",
        "-".repeat(w_scenario + w_engine + 2 + 12 + 2 + 12 + 2 + 10)
    );
    for (_, label, group) in &scenarios {
        for engine in &group[0].comparison.engines {
            let engine_reports: Vec<&Report> = group
                .iter()
                .copied()
                .filter(|report| &report.engine == engine)
                .collect();
            if engine_reports.is_empty() {
                continue;
            }
            let row = summarize_engine(engine, &engine_reports);
            let load_str = match row.load_ops_per_sec {
                Some(ops_per_sec) => fmt_thousands(ops_per_sec),
                None => "-".to_string(),
            };
            let drained_str = match row.load_drained_ops_per_sec {
                Some(ops_per_sec) => fmt_thousands(ops_per_sec),
                None => "-".to_string(),
            };
            let drain_str = match row.load_drain_ms {
                Some(drain_ms) => format!("{drain_ms:.2}"),
                None => "-".to_string(),
            };
            println!(
                "{:w1$}  {:w2$}  {:>12}  {:>12}  {:>10}",
                label,
                engine,
                load_str,
                drained_str,
                drain_str,
                w1 = w_scenario,
                w2 = w_engine,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::PhaseStats;
    use crate::workload::Workload;

    #[test]
    fn report_serializes_to_json() {
        let mut stats = PhaseStats::new();
        stats.record(1000, crate::stats::OpKind::Get);
        stats.duration = std::time::Duration::from_millis(10);

        let spec = WorkloadSpec {
            workload: Workload::ReadRandom,
            ..WorkloadSpec::default()
        };
        let report = Report::new(
            "mock",
            ComparisonContract {
                id: "test".to_string(),
                engines: vec!["mock".to_string()],
                memory_regime: crate::comparison::MemoryRegime::Resident,
                operation_contract: crate::comparison::OperationContract::PointRead,
            },
            1,
            spec,
            EngineOpts::default(),
            ReportMeasurements {
                load_phase: None,
                load_durability_drain_secs: None,
                run_phase: (&stats).into(),
                durability_drain_secs: 0.0,
                engine_stats: EngineStats::default(),
            },
        );
        let json = report.to_json_pretty().unwrap();
        assert!(json.contains("\"schema\""));
        assert!(json.contains("\"engine\""));
        assert!(json.contains("\"mock\""));
        assert!(json.contains("\"run_phase\""));
        assert!(json.contains("\"durability_drain_secs\""));
        assert!(json.contains("\"binary_hash\""));
        assert!(json.contains("\"engine_stats\""));
    }

    #[test]
    fn median_aggregates_repetitions_without_reusing_one_sample() {
        assert_eq!(median(vec![30.0, 10.0, 20.0]), 20.0);
        assert_eq!(median(vec![40.0, 10.0, 30.0, 20.0]), 25.0);
    }

    #[test]
    fn comparison_identity_includes_cache_and_contract() {
        let mut stats = PhaseStats::new();
        stats.record(1000, crate::stats::OpKind::Get);
        stats.duration = std::time::Duration::from_millis(1);
        let contract = ComparisonContract {
            id: "resident-read".to_string(),
            engines: vec!["a".to_string(), "b".to_string()],
            memory_regime: crate::comparison::MemoryRegime::Resident,
            operation_contract: crate::comparison::OperationContract::PointRead,
        };
        let a = Report::new(
            "a",
            contract.clone(),
            1,
            WorkloadSpec::default(),
            EngineOpts::default(),
            ReportMeasurements {
                load_phase: None,
                load_durability_drain_secs: None,
                run_phase: (&stats).into(),
                durability_drain_secs: 0.0,
                engine_stats: EngineStats::default(),
            },
        );
        let mut different_opts = EngineOpts::default();
        different_opts.cache_budget_bytes /= 2;
        let b = Report::new(
            "b",
            contract,
            1,
            WorkloadSpec::default(),
            different_opts,
            ReportMeasurements {
                load_phase: None,
                load_durability_drain_secs: None,
                run_phase: (&stats).into(),
                durability_drain_secs: 0.0,
                engine_stats: EngineStats::default(),
            },
        );
        assert_ne!(
            comparison_identity(&a, false),
            comparison_identity(&b, false),
            "different cache budgets must never share a comparison group"
        );
    }
}
