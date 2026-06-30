//! Workload specification and operation generator.
//!
//! Each spec produces two phases:
//! - **Load**: bulk `Put` of `record_count` keys (timed separately).
//! - **Run**: `operation_count` mixed operations (the headline measurement).
//!
//! This matches the YCSB load/run convention and isolates bulk-load cost from
//! steady-state throughput.

use serde::{Deserialize, Serialize};

use crate::distribution::{Distribution, format_key, hash64};
use crate::engine::EngineOpts;

// ---------------------------------------------------------------------------
// Specs
// ---------------------------------------------------------------------------

/// All workload types. YCSB A–F follow the Cooper et al. (SoCC 2010) spec.
/// db_bench scenarios follow the LevelDB/RocksDB `db_bench` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Workload {
    // Phase 0
    /// Random-order bulk put.
    FillRandom,
    /// Sequential bulk put.
    FillSeq,
    /// Random point lookups.
    ReadRandom,
    /// In-place update churn (existing keys).
    Overwrite,
    // Phase 1: YCSB core
    /// 50% read / 50% update (uniform).
    YcsbA,
    /// 95% read / 5% update (uniform).
    YcsbB,
    /// 100% read (uniform).
    YcsbC,
    /// 95% read / 5% insert (latest distribution).
    YcsbD,
    /// 95% short-range scan / 5% insert (zipfian).
    YcsbE,
    /// 100% read-modify-write (zipfian).
    YcsbF,
    // Phase 1: db_bench scenarios
    /// Sequential reads.
    ReadSeq,
    /// Mixed read/write (approx. 90% read / 10% write).
    ReadWhileWriting,
    /// Random deletes.
    DeleteRandom,
    /// Sequential deletes.
    DeleteSeq,
    /// Random range seeks.
    SeekRandom,
}

impl Workload {
    /// All workload variants, in declaration order.
    pub const ALL: &'static [Workload] = &[
        Workload::FillRandom,
        Workload::FillSeq,
        Workload::ReadRandom,
        Workload::Overwrite,
        Workload::YcsbA,
        Workload::YcsbB,
        Workload::YcsbC,
        Workload::YcsbD,
        Workload::YcsbE,
        Workload::YcsbF,
        Workload::ReadSeq,
        Workload::ReadWhileWriting,
        Workload::DeleteRandom,
        Workload::DeleteSeq,
        Workload::SeekRandom,
    ];

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Workload::FillRandom => "fillrandom",
            Workload::FillSeq => "fillseq",
            Workload::ReadRandom => "readrandom",
            Workload::Overwrite => "overwrite",
            Workload::YcsbA => "ycsb_a",
            Workload::YcsbB => "ycsb_b",
            Workload::YcsbC => "ycsb_c",
            Workload::YcsbD => "ycsb_d",
            Workload::YcsbE => "ycsb_e",
            Workload::YcsbF => "ycsb_f",
            Workload::ReadSeq => "readseq",
            Workload::ReadWhileWriting => "readwhilewriting",
            Workload::DeleteRandom => "deleterandom",
            Workload::DeleteSeq => "deleteseq",
            Workload::SeekRandom => "seekrandom",
        }
    }

    /// Whether this workload needs a populated database before the run phase.
    pub fn needs_load_phase(self) -> bool {
        !matches!(self, Workload::FillRandom | Workload::FillSeq)
    }

    /// What fraction of the run-phase ops are reads (0.0–1.0).
    pub fn read_pct(self) -> f64 {
        match self {
            Workload::FillRandom | Workload::FillSeq | Workload::Overwrite => 0.0,
            Workload::ReadRandom | Workload::ReadSeq | Workload::YcsbC => 1.0,
            Workload::YcsbA => 0.50,
            Workload::YcsbB => 0.95,
            Workload::YcsbD => 0.95,
            Workload::YcsbE => 0.95,
            Workload::YcsbF => 0.0, // 100% RMW
            Workload::ReadWhileWriting => 0.90,
            Workload::DeleteRandom | Workload::DeleteSeq => 0.0,
            Workload::SeekRandom => 0.0, // scans, not point reads
        }
    }
}

impl std::fmt::Display for Workload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Default zipfian theta used by YCSB E.
#[allow(dead_code)]
const DEFAULT_THETA: f64 = Distribution::DEFAULT_ZIPF_THETA;

/// Full specification of a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadSpec {
    /// Which workload to run.
    pub workload: Workload,
    /// Key distribution.
    #[serde(default)]
    pub distribution: Distribution,
    /// Number of keys to insert in the load phase.
    pub record_count: u64,
    /// Value size in bytes.
    #[serde(default = "default_value_size")]
    pub value_size: usize,
    /// Number of operations in the run phase.
    pub operation_count: u64,
    /// PRNG seed (for reproducibility).
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Number of worker threads.
    #[serde(default = "default_threads")]
    pub threads: usize,
}

impl Default for WorkloadSpec {
    fn default() -> Self {
        Self {
            workload: Workload::ReadRandom,
            distribution: Distribution::Uniform,
            record_count: 10_000,
            value_size: default_value_size(),
            operation_count: 10_000,
            seed: default_seed(),
            threads: default_threads(),
        }
    }
}

fn default_value_size() -> usize {
    100
}
fn default_seed() -> u64 {
    42
}
fn default_threads() -> usize {
    1
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

/// A single operation the driver will execute against the engine.
#[derive(Debug, Clone)]
pub enum WorkloadOp {
    /// Insert or update.
    Put { key: Vec<u8>, value: Vec<u8> },
    /// Point lookup.
    Get { key: Vec<u8> },
    /// Delete.
    Del { key: Vec<u8> },
    /// Scan `count` keys starting from `start` (inclusive).
    Scan { start: Vec<u8>, count: usize },
    /// Read-modify-write: read then update (measured as one op).
    Rmw { key: Vec<u8>, value: Vec<u8> },
}

impl WorkloadOp {
    /// Whether this op is read-only (no mutation).
    #[allow(dead_code)]
    pub fn is_read(&self) -> bool {
        matches!(self, WorkloadOp::Get { .. } | WorkloadOp::Scan { .. })
    }

    /// Whether this op mutates the database.
    #[allow(dead_code)]
    pub fn is_write(&self) -> bool {
        matches!(
            self,
            WorkloadOp::Put { .. } | WorkloadOp::Del { .. } | WorkloadOp::Rmw { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// Generate the load-phase ops (sequential puts of `record_count` keys).
pub fn generate_load_ops(spec: &WorkloadSpec, opts: &EngineOpts) -> Vec<WorkloadOp> {
    let value = make_value(opts.value_size, spec.seed);
    let mut ops = Vec::with_capacity(spec.record_count as usize);
    for i in 0..spec.record_count {
        ops.push(WorkloadOp::Put {
            key: format_key(i).to_vec(),
            value: value.clone(),
        });
    }
    ops
}

/// Generate the run-phase ops.
pub fn generate_run_ops(spec: &WorkloadSpec, _opts: &EngineOpts) -> Vec<WorkloadOp> {
    let value = make_value(_opts.value_size, spec.seed);
    let n = spec.operation_count;
    let mut sampler = spec.distribution.sampler(spec.record_count, spec.seed);

    // For YcsbD / YcsbE which insert new keys during the run phase, we start
    // inserting at record_count and grow.
    let mut next_insert_id = spec.record_count;

    let mut ops = Vec::with_capacity(n as usize);
    for i in 0..n {
        let op = match spec.workload {
            Workload::FillRandom => {
                let k = sampler.sample(i);
                WorkloadOp::Put {
                    key: format_key(k).to_vec(),
                    value: value.clone(),
                }
            }
            Workload::FillSeq => WorkloadOp::Put {
                key: format_key(i % spec.record_count.max(1)).to_vec(),
                value: value.clone(),
            },
            Workload::ReadRandom => {
                let k = sampler.sample(i);
                WorkloadOp::Get {
                    key: format_key(k).to_vec(),
                }
            }
            Workload::ReadSeq => WorkloadOp::Get {
                key: format_key(i % spec.record_count.max(1)).to_vec(),
            },
            Workload::Overwrite => {
                let k = sampler.sample(i);
                WorkloadOp::Put {
                    key: format_key(k).to_vec(),
                    value: value.clone(),
                }
            }
            Workload::YcsbA | Workload::YcsbB | Workload::YcsbC => {
                let k = sampler.sample(i);
                if is_read(spec.workload.read_pct(), spec.seed, i) {
                    WorkloadOp::Get {
                        key: format_key(k).to_vec(),
                    }
                } else {
                    WorkloadOp::Put {
                        key: format_key(k).to_vec(),
                        value: value.clone(),
                    }
                }
            }
            Workload::YcsbD => {
                let k = sampler.sample(i);
                if is_read(spec.workload.read_pct(), spec.seed, i) {
                    WorkloadOp::Get {
                        key: format_key(k).to_vec(),
                    }
                } else {
                    // Insert a new key (grows dataset).
                    let new_key = format_key(next_insert_id).to_vec();
                    next_insert_id += 1;
                    sampler.notify_insert();
                    WorkloadOp::Put {
                        key: new_key,
                        value: value.clone(),
                    }
                }
            }
            Workload::YcsbE => {
                let k = sampler.sample(i);
                if is_read(spec.workload.read_pct(), spec.seed, i) {
                    // Short-range scan: 10 keys starting from k.
                    WorkloadOp::Scan {
                        start: format_key(k).to_vec(),
                        count: 10,
                    }
                } else {
                    let new_key = format_key(next_insert_id).to_vec();
                    next_insert_id += 1;
                    sampler.notify_insert();
                    WorkloadOp::Put {
                        key: new_key,
                        value: value.clone(),
                    }
                }
            }
            Workload::YcsbF => {
                let k = sampler.sample(i);
                WorkloadOp::Rmw {
                    key: format_key(k).to_vec(),
                    value: value.clone(),
                }
            }
            Workload::ReadWhileWriting => {
                let k = sampler.sample(i);
                if is_read(spec.workload.read_pct(), spec.seed, i) {
                    WorkloadOp::Get {
                        key: format_key(k).to_vec(),
                    }
                } else {
                    WorkloadOp::Put {
                        key: format_key(k).to_vec(),
                        value: value.clone(),
                    }
                }
            }
            Workload::DeleteRandom => {
                let k = sampler.sample(i);
                WorkloadOp::Del {
                    key: format_key(k).to_vec(),
                }
            }
            Workload::DeleteSeq => WorkloadOp::Del {
                key: format_key(i % spec.record_count.max(1)).to_vec(),
            },
            Workload::SeekRandom => {
                let k = sampler.sample(i);
                WorkloadOp::Scan {
                    start: format_key(k).to_vec(),
                    count: 1,
                }
            }
        };
        ops.push(op);
    }
    ops
}

/// Deterministically decide whether op `i` is a read based on `read_pct`
/// (0.0 = 0% reads, 1.0 = 100% reads).
fn is_read(read_pct: f64, seed: u64, i: u64) -> bool {
    let h = hash64(seed.wrapping_add(i).wrapping_mul(2_654_435_761));
    ((h % 1000) as f64) < read_pct * 1000.0
}

/// Generate a value of the given size, filled with a deterministic pattern.
fn make_value(size: usize, seed: u64) -> Vec<u8> {
    let mut v = vec![0u8; size];
    // Fill with a deterministic pattern so values are compressible but not
    // all-zero (which some engines may special-case).
    let mut state = seed | 0xAA_55_AA_55;
    for byte in &mut v {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *byte = (state >> 32) as u8;
    }
    v
}

/// End-key for a scan_range that starts at `start_key` and covers `count` keys.
/// Returns the key immediately after `start_key + count` (exclusive upper bound).
pub fn scan_end_key(start_key: &[u8], count: u64) -> Vec<u8> {
    if start_key.len() >= 8 {
        let mut start = u64::from_be_bytes(start_key[..8].try_into().unwrap());
        start = start.saturating_add(count);
        start.to_be_bytes().to_vec()
    } else {
        // Fallback: just return start + count as a Vec.
        start_key.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_phase_generates_record_count_puts() {
        let spec = WorkloadSpec {
            workload: Workload::FillRandom,
            record_count: 100,
            operation_count: 0,
            ..WorkloadSpec::default()
        };
        let opts = EngineOpts::default();
        let ops = generate_load_ops(&spec, &opts);
        assert_eq!(ops.len(), 100);
        // All should be puts.
        assert!(ops.iter().all(|op| op.is_write()));
    }

    #[test]
    fn readrandom_all_gets() {
        let spec = WorkloadSpec {
            workload: Workload::ReadRandom,
            record_count: 100,
            operation_count: 500,
            ..WorkloadSpec::default()
        };
        let opts = EngineOpts::default();
        let ops = generate_run_ops(&spec, &opts);
        assert_eq!(ops.len(), 500);
        assert!(ops.iter().all(|op| op.is_read()));
    }

    #[test]
    fn overwrite_all_puts() {
        let spec = WorkloadSpec {
            workload: Workload::Overwrite,
            record_count: 100,
            operation_count: 500,
            ..WorkloadSpec::default()
        };
        let opts = EngineOpts::default();
        let ops = generate_run_ops(&spec, &opts);
        assert_eq!(ops.len(), 500);
        assert!(ops.iter().all(|op| op.is_write()));
    }

    #[test]
    fn ycsba_mix_is_50_50() {
        let spec = WorkloadSpec {
            workload: Workload::YcsbA,
            record_count: 1000,
            operation_count: 10_000,
            ..WorkloadSpec::default()
        };
        let opts = EngineOpts::default();
        let ops = generate_run_ops(&spec, &opts);
        let reads = ops.iter().filter(|op| op.is_read()).count();
        let writes = ops.iter().filter(|op| op.is_write()).count();
        // Should be approximately 50/50 (within 5%).
        assert!(
            reads > 4500 && reads < 5500,
            "YcsbA reads should be ~5000, got {}",
            reads
        );
        assert!(
            writes > 4500 && writes < 5500,
            "YcsbA writes should be ~5000, got {}",
            writes
        );
    }

    #[test]
    fn ycsbc_all_reads() {
        let spec = WorkloadSpec {
            workload: Workload::YcsbC,
            record_count: 100,
            operation_count: 500,
            ..WorkloadSpec::default()
        };
        let opts = EngineOpts::default();
        let ops = generate_run_ops(&spec, &opts);
        assert!(ops.iter().all(|op| op.is_read()));
    }

    #[test]
    fn deleterandom_all_deletes() {
        let spec = WorkloadSpec {
            workload: Workload::DeleteRandom,
            record_count: 100,
            operation_count: 50,
            ..WorkloadSpec::default()
        };
        let opts = EngineOpts::default();
        let ops = generate_run_ops(&spec, &opts);
        assert_eq!(ops.len(), 50);
        assert!(ops.iter().all(|op| matches!(op, WorkloadOp::Del { .. })));
    }

    #[test]
    fn seekrandom_all_scans() {
        let spec = WorkloadSpec {
            workload: Workload::SeekRandom,
            record_count: 100,
            operation_count: 50,
            ..WorkloadSpec::default()
        };
        let opts = EngineOpts::default();
        let ops = generate_run_ops(&spec, &opts);
        assert_eq!(ops.len(), 50);
        assert!(ops.iter().all(|op| matches!(op, WorkloadOp::Scan { .. })));
    }

    #[test]
    fn ops_are_deterministic_given_seed() {
        let spec = WorkloadSpec {
            workload: Workload::YcsbA,
            record_count: 100,
            operation_count: 100,
            seed: 42,
            ..WorkloadSpec::default()
        };
        let opts = EngineOpts::default();
        let ops1 = generate_run_ops(&spec, &opts);
        let ops2 = generate_run_ops(&spec, &opts);
        assert_eq!(ops1.len(), ops2.len());
        for (a, b) in ops1.iter().zip(ops2.iter()) {
            match (a, b) {
                (WorkloadOp::Get { key: ka }, WorkloadOp::Get { key: kb }) => {
                    assert_eq!(ka, kb);
                }
                (
                    WorkloadOp::Put { key: ka, value: va },
                    WorkloadOp::Put { key: kb, value: vb },
                ) => {
                    assert_eq!(ka, kb);
                    assert_eq!(va, vb);
                }
                _ => panic!("op type mismatch or unexpected variant"),
            }
        }
    }
}
