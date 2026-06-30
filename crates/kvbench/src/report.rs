//! JSON report serialization.
//!
//! Each report is a self-contained JSON file with schema version, engine name,
//! full workload spec, git commit hash, and load/run phase summaries. A
//! reported number is reproducible from the report alone.

use serde::{Deserialize, Serialize};

use crate::engine::EngineOpts;
use crate::stats::PhaseSummary;
use crate::workload::WorkloadSpec;

/// JSON report schema version.
pub const SCHEMA_VERSION: u32 = 1;

/// A complete benchmark report for one run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub schema: u32,
    pub engine: String,
    pub spec: WorkloadSpec,
    pub engine_opts: EngineOpts,
    pub git_commit: String,
    pub load_phase: Option<PhaseSummary>,
    pub run_phase: PhaseSummary,
}

impl Report {
    /// Create a new report with the current git commit hash.
    pub fn new(
        engine: &str,
        spec: WorkloadSpec,
        engine_opts: EngineOpts,
        load_phase: Option<PhaseSummary>,
        run_phase: PhaseSummary,
    ) -> Self {
        Self {
            schema: SCHEMA_VERSION,
            engine: engine.to_string(),
            spec,
            engine_opts,
            git_commit: git_commit_hash(),
            load_phase,
            run_phase,
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

/// Print a side-by-side comparison of two reports to stdout.
pub fn print_comparison(a: &Report, b: &Report) {
    println!(
        "{:<20} {:>20} {:>20} {:>10}",
        "metric", a.engine, b.engine, "delta %"
    );
    println!("{}", "-".repeat(72));

    let rows = [
        ("ops/sec", a.run_phase.ops_per_sec, b.run_phase.ops_per_sec),
        ("p50_us", a.run_phase.p50_us, b.run_phase.p50_us),
        ("p95_us", a.run_phase.p95_us, b.run_phase.p95_us),
        ("p99_us", a.run_phase.p99_us, b.run_phase.p99_us),
    ];

    for (name, va, vb) in rows {
        let delta = if va != 0.0 {
            (vb - va) / va * 100.0
        } else {
            0.0
        };
        println!("{:<20} {:>20.2} {:>20.2} {:>9.1}%", name, va, vb, delta);
    }
}

// ---------------------------------------------------------------------------
// Consolidated summary table
// ---------------------------------------------------------------------------

/// A short label identifying a scenario: `{workload} {records}/{ops}/{threads}t {dist}`.
fn scenario_label(report: &Report) -> String {
    let s = &report.spec;
    format!(
        "{} {}/{}/{}t {}",
        s.workload.name(),
        abbreviate(s.record_count),
        abbreviate(s.operation_count),
        s.threads,
        distribution_label(&s.distribution),
    )
}

/// Abbreviate a count with k/M suffix.
fn abbreviate(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}k", n / 1_000)
    } else {
        n.to_string()
    }
}

/// Short distribution label.
fn distribution_label(d: &crate::distribution::Distribution) -> &'static str {
    match d {
        crate::distribution::Distribution::Uniform => "uniform",
        crate::distribution::Distribution::Zipfian { .. } => "zipfian",
        crate::distribution::Distribution::Latest => "latest",
    }
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

/// Print a consolidated summary table from multiple reports, grouped by
/// scenario. Within each scenario, the engine with the highest ops/sec is
/// marked with `★`.
pub fn print_summary_table(reports: &[Report]) {
    if reports.is_empty() {
        println!("(no reports found)");
        return;
    }

    let commit = reports[0].git_commit.as_str();
    let engines: Vec<&str> = {
        let mut seen = Vec::new();
        for r in reports {
            if !seen.contains(&r.engine.as_str()) {
                seen.push(r.engine.as_str());
            }
        }
        seen
    };

    // Group reports by scenario label, preserving first-seen order.
    let mut scenarios: Vec<(String, Vec<&Report>)> = Vec::new();
    for r in reports {
        let label = scenario_label(r);
        if let Some(entry) = scenarios.iter_mut().find(|(l, _)| l == &label) {
            entry.1.push(r);
        } else {
            scenarios.push((label, vec![r]));
        }
    }

    println!("=== Consolidated Benchmark Report ===");
    println!(
        "{} runs across {} engines ({}), commit {}",
        reports.len(),
        engines.len(),
        engines.join(", "),
        commit,
    );
    println!();

    // Column widths.
    let w_scenario = scenarios
        .iter()
        .map(|(l, _)| l.len())
        .max()
        .unwrap_or(8)
        .max(8);
    let w_engine = engines.iter().map(|e| e.len()).max().unwrap_or(6).max(6);

    // Header.
    println!(
        "{:w1$}  {:w2$}  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}",
        "scenario",
        "engine",
        "ops/sec",
        "p50_us",
        "p95_us",
        "p99_us",
        "p99.9_us",
        w1 = w_scenario,
        w2 = w_engine,
    );
    println!(
        "{}",
        "-".repeat(w_scenario + w_engine + 2 + 12 + 2 + 10 + 2 + 10 + 2 + 10 + 2 + 10)
    );

    for (label, group) in &scenarios {
        // Find best ops/sec in this scenario.
        let best_ops = group
            .iter()
            .map(|r| r.run_phase.ops_per_sec)
            .fold(0.0_f64, f64::max);

        for r in group {
            let is_best = r.run_phase.ops_per_sec == best_ops && group.len() > 1;
            let star = if is_best { " ★" } else { "  " };
            println!(
                "{:w1$}  {:w2$}  {:>12}  {:>9.2}us  {:>9.2}us  {:>9.2}us  {:>9.2}us{}",
                label,
                r.engine,
                fmt_thousands(r.run_phase.ops_per_sec),
                r.run_phase.p50_us,
                r.run_phase.p95_us,
                r.run_phase.p99_us,
                r.run_phase.p999_us,
                star,
                w1 = w_scenario,
                w2 = w_engine,
            );
        }
    }

    println!();
    println!("★ = best ops/sec for scenario");
    println!();
    println!("Load phase (ops/sec):");
    println!(
        "{:w1$}  {:w2$}  {:>12}",
        "scenario",
        "engine",
        "load ops/sec",
        w1 = w_scenario,
        w2 = w_engine,
    );
    println!("{}", "-".repeat(w_scenario + w_engine + 2 + 12));
    for (label, group) in &scenarios {
        for r in group {
            let load_str = match r.load_phase.as_ref() {
                Some(p) => fmt_thousands(p.ops_per_sec),
                None => "-".to_string(),
            };
            println!(
                "{:w1$}  {:w2$}  {:>12}",
                label,
                r.engine,
                load_str,
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
        let report = Report::new("mock", spec, EngineOpts::default(), None, (&stats).into());
        let json = report.to_json_pretty().unwrap();
        assert!(json.contains("\"schema\""));
        assert!(json.contains("\"engine\""));
        assert!(json.contains("\"mock\""));
        assert!(json.contains("\"run_phase\""));
    }
}
