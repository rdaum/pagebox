//! Thread-pool workload executor with per-operation latency capture.
//!
//! N worker threads pull from a shared operation vector via an atomic index,
//! call the engine, and record per-op latency via `std::time::Instant`. Uses
//! `std::thread::scope` so borrowed references can be shared across threads
//! without `Arc` or raw pointers.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::engine::KvEngine;
use crate::stats::{OpKind, PhaseStats};
use crate::workload::{WorkloadOp, scan_end_key};

/// Run a phase (load or run) against the given engine.
///
/// The `ops` slice is partitioned across `threads` worker threads; each
/// worker atomically claims the next op index and executes it.
pub fn run_phase<E: KvEngine + Sync>(engine: &E, ops: &[WorkloadOp], threads: usize) -> PhaseStats {
    let threads = threads.max(1);
    let next_idx = Arc::new(AtomicU64::new(0));
    let total = ops.len() as u64;

    let start = Instant::now();
    let mut combined = PhaseStats::new();

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            let next_idx = Arc::clone(&next_idx);
            handles.push(s.spawn(move || {
                let mut local = PhaseStats::new();
                loop {
                    let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                    if idx >= total {
                        break;
                    }
                    let op = &ops[idx as usize];
                    let op_start = Instant::now();
                    let kind = execute_op(engine, op);
                    let elapsed = op_start.elapsed().as_nanos() as u64;
                    local.record(elapsed, kind);
                }
                local
            }));
        }
        for handle in handles {
            let local = handle.join().expect("worker thread panicked");
            combined.merge(&local);
        }
    });

    combined.duration = start.elapsed();
    combined
}

/// Execute a single operation against the engine. Returns the op kind for
/// counting.
fn execute_op<E: KvEngine>(engine: &E, op: &WorkloadOp) -> OpKind {
    match op {
        WorkloadOp::Put { key, value } => {
            engine.put(key, value);
            OpKind::Put
        }
        WorkloadOp::Get { key } => {
            black_box(engine.get(key));
            OpKind::Get
        }
        WorkloadOp::Del { key } => {
            black_box(engine.del(key));
            OpKind::Del
        }
        WorkloadOp::Scan { start, count } => {
            let end = scan_end_key(start, *count as u64);
            let mut n = 0usize;
            engine.scan_range(start, &end, &mut |_, _| {
                n += 1;
            });
            black_box(n);
            OpKind::Scan
        }
        WorkloadOp::Rmw { key, value } => {
            let _ = engine.get(key);
            engine.put(key, value);
            OpKind::Rmw
        }
    }
}

/// Prevent the compiler from optimising away the computation.
fn black_box<T>(t: T) -> T {
    std::hint::black_box(t)
}

/// Estimate the time to run a single warmup operation (for sanity checks).
#[allow(dead_code)]
pub fn time_single_op<E: KvEngine>(engine: &E, op: &WorkloadOp) -> Duration {
    let start = Instant::now();
    let _ = execute_op(engine, op);
    start.elapsed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineOpts;
    use std::collections::BTreeMap;
    use std::path::Path;

    /// A mock engine backed by a `BTreeMap`, used for testing the driver
    /// and workload generator without a real engine.
    pub struct MockEngine {
        data: std::sync::Mutex<BTreeMap<Vec<u8>, Vec<u8>>>,
    }

    impl MockEngine {
        pub fn new() -> Self {
            Self {
                data: std::sync::Mutex::new(BTreeMap::new()),
            }
        }
    }

    impl KvEngine for MockEngine {
        const NAME: &'static str = "mock";

        fn open(_dir: &Path, _opts: &EngineOpts) -> std::io::Result<Self> {
            Ok(Self::new())
        }

        fn put(&self, key: &[u8], value: &[u8]) -> bool {
            let mut data = self.data.lock().unwrap();
            let inserted = !data.contains_key(key);
            data.insert(key.to_vec(), value.to_vec());
            inserted
        }

        fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
            self.data.lock().unwrap().get(key).cloned()
        }

        fn del(&self, key: &[u8]) -> bool {
            self.data.lock().unwrap().remove(key).is_some()
        }

        fn scan_range(&self, start: &[u8], end: &[u8], f: &mut dyn FnMut(&[u8], &[u8])) {
            let data = self.data.lock().unwrap();
            for (k, v) in data.range::<[u8], _>((
                std::ops::Bound::Included(start),
                std::ops::Bound::Excluded(end),
            )) {
                f(k, v);
            }
        }

        fn sync(&self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Default for MockEngine {
        fn default() -> Self {
            Self::new()
        }
    }

    #[test]
    fn mock_engine_adapter_contract() {
        let engine = MockEngine::new();
        assert!(engine.put(b"A", b"val_a"));
        assert!(!engine.put(b"A", b"val_a2"));
        assert!(engine.put(b"B", b"val_b"));
        assert_eq!(engine.get(b"A"), Some(b"val_a2".to_vec()));
        assert!(engine.del(b"A"));
        assert_eq!(engine.get(b"A"), None);

        let mut results = Vec::new();
        engine.scan_range(b"A", b"C", &mut |k, v| {
            results.push((k.to_vec(), v.to_vec()));
        });
        assert_eq!(results, vec![(b"B".to_vec(), b"val_b".to_vec())]);
    }

    #[test]
    fn run_phase_executes_all_ops() {
        let engine = MockEngine::new();
        let ops = vec![
            WorkloadOp::Put {
                key: b"k1".to_vec(),
                value: b"v1".to_vec(),
            },
            WorkloadOp::Put {
                key: b"k2".to_vec(),
                value: b"v2".to_vec(),
            },
            WorkloadOp::Get {
                key: b"k1".to_vec(),
            },
            WorkloadOp::Get {
                key: b"k3".to_vec(),
            },
        ];
        let stats = run_phase(&engine, &ops, 2);
        assert_eq!(stats.operations, 4);
        assert_eq!(stats.puts, 2);
        assert_eq!(stats.gets, 2);
        assert_eq!(stats.dels, 0);
    }

    #[test]
    fn run_phase_single_thread() {
        let engine = MockEngine::new();
        let ops = vec![
            WorkloadOp::Put {
                key: b"k1".to_vec(),
                value: b"v1".to_vec(),
            },
            WorkloadOp::Get {
                key: b"k1".to_vec(),
            },
        ];
        let stats = run_phase(&engine, &ops, 1);
        assert_eq!(stats.operations, 2);
        assert!((stats.ops_per_sec() - 2.0 / stats.duration.as_secs_f64()).abs() < 1.0);
    }

    #[test]
    fn run_phase_scans() {
        let engine = MockEngine::new();
        engine.put(b"k1", b"v1");
        engine.put(b"k2", b"v2");
        engine.put(b"k3", b"v3");
        let ops = vec![WorkloadOp::Scan {
            start: b"k1".to_vec(),
            count: 2,
        }];
        let stats = run_phase(&engine, &ops, 1);
        assert_eq!(stats.scans, 1);
    }

    #[test]
    fn run_phase_rmw() {
        let engine = MockEngine::new();
        engine.put(b"k1", b"v1");
        let ops = vec![WorkloadOp::Rmw {
            key: b"k1".to_vec(),
            value: b"v2".to_vec(),
        }];
        let stats = run_phase(&engine, &ops, 1);
        assert_eq!(stats.rmws, 1);
        assert_eq!(engine.get(b"k1"), Some(b"v2".to_vec()));
    }
}
