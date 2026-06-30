//! Latency and throughput statistics.
//!
//! Uses `HdrHistogram` for latency percentiles (p50/p95/p99/p99.9).

use std::time::Duration;

use hdrhistogram::Histogram;

/// Latency histogram + aggregate counters for one benchmark phase.
#[derive(Clone)]
pub struct PhaseStats {
    pub operations: u64,
    pub duration: Duration,
    /// Latency histogram in nanoseconds.
    pub latencies: Histogram<u64>,
    /// Per-operation-type counters.
    pub puts: u64,
    pub gets: u64,
    pub dels: u64,
    pub scans: u64,
    pub rmws: u64,
}

impl PhaseStats {
    /// Create an empty stats container.
    pub fn new() -> Self {
        Self {
            operations: 0,
            duration: Duration::ZERO,
            latencies: Histogram::new_with_bounds(1, 60_000_000_000, 3).unwrap(),
            puts: 0,
            gets: 0,
            dels: 0,
            scans: 0,
            rmws: 0,
        }
    }

    /// Record a single operation's latency.
    pub fn record(&mut self, latency_ns: u64, op_kind: OpKind) {
        let _ = self.latencies.record(latency_ns);
        self.operations += 1;
        match op_kind {
            OpKind::Put => self.puts += 1,
            OpKind::Get => self.gets += 1,
            OpKind::Del => self.dels += 1,
            OpKind::Scan => self.scans += 1,
            OpKind::Rmw => self.rmws += 1,
        }
    }

    /// Merge another phase's stats into this one.
    pub fn merge(&mut self, other: &Self) {
        self.operations += other.operations;
        self.duration = self.duration.max(other.duration);
        self.latencies += &other.latencies;
        self.puts += other.puts;
        self.gets += other.gets;
        self.dels += other.dels;
        self.scans += other.scans;
        self.rmws += other.rmws;
    }

    /// Throughput in operations per second.
    pub fn ops_per_sec(&self) -> f64 {
        if self.duration.is_zero() {
            return 0.0;
        }
        self.operations as f64 / self.duration.as_secs_f64()
    }

    /// p50 latency in microseconds.
    pub fn p50_us(&self) -> f64 {
        self.latencies.value_at_quantile(0.5) as f64 / 1000.0
    }

    /// p95 latency in microseconds.
    pub fn p95_us(&self) -> f64 {
        self.latencies.value_at_quantile(0.95) as f64 / 1000.0
    }

    /// p99 latency in microseconds.
    pub fn p99_us(&self) -> f64 {
        self.latencies.value_at_quantile(0.99) as f64 / 1000.0
    }

    /// p99.9 latency in microseconds.
    pub fn p999_us(&self) -> f64 {
        self.latencies.value_at_quantile(0.999) as f64 / 1000.0
    }
}

impl Default for PhaseStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Which kind of operation was executed (for per-type counting).
#[derive(Debug, Clone, Copy)]
pub enum OpKind {
    Put,
    Get,
    Del,
    Scan,
    Rmw,
}

/// Serializable summary of a phase's results.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PhaseSummary {
    pub operations: u64,
    pub duration_secs: f64,
    pub ops_per_sec: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub p999_us: f64,
    pub puts: u64,
    pub gets: u64,
    pub dels: u64,
    pub scans: u64,
    pub rmws: u64,
}

impl From<&PhaseStats> for PhaseSummary {
    fn from(s: &PhaseStats) -> Self {
        Self {
            operations: s.operations,
            duration_secs: s.duration.as_secs_f64(),
            ops_per_sec: s.ops_per_sec(),
            p50_us: s.p50_us(),
            p95_us: s.p95_us(),
            p99_us: s.p99_us(),
            p999_us: s.p999_us(),
            puts: s.puts,
            gets: s.gets,
            dels: s.dels,
            scans: s.scans,
            rmws: s.rmws,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_merge() {
        let mut a = PhaseStats::new();
        a.record(1000, OpKind::Get);
        a.record(2000, OpKind::Put);
        a.duration = Duration::from_millis(10);

        assert_eq!(a.operations, 2);
        assert_eq!(a.gets, 1);
        assert_eq!(a.puts, 1);
        assert!((a.ops_per_sec() - 200.0).abs() < 1.0);
        assert!((a.p50_us() - 1.0).abs() < 0.5);

        let mut b = PhaseStats::new();
        b.record(3000, OpKind::Get);
        b.duration = Duration::from_millis(15);

        a.merge(&b);
        assert_eq!(a.operations, 3);
        assert_eq!(a.gets, 2);
        assert_eq!(a.puts, 1);
    }
}
