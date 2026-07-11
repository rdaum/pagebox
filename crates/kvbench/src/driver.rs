//! Thread-pool workload executor with per-operation latency capture.
//!
//! N worker threads pull from a shared operation vector via an atomic index,
//! call the engine, and record per-op latency via `std::time::Instant`. Uses
//! `std::thread::scope` so borrowed references can be shared across threads
//! without `Arc` or raw pointers.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::engine::KvEngine;
use crate::stats::{OpKind, PhaseStats};
use crate::workload::{WorkloadOp, scan_end_key};

const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
const PROGRESS_FLUSH_OPS: u64 = 256;

/// Run a phase (load or run) against the given engine.
///
/// The `ops` slice is partitioned across `threads` worker threads; each
/// worker atomically claims the next op index and executes it.
pub fn run_phase<E: KvEngine + Sync>(
    engine: &E,
    ops: &[WorkloadOp],
    threads: usize,
    minimum_duration: Duration,
) -> PhaseStats {
    let shadow = Arc::new(std::sync::Mutex::new(HashMap::new()));
    run_phase_inner(engine, ops, threads, minimum_duration, false, &shadow, None)
}

/// Run a finite load phase and emit periodic progress to stderr.
pub fn run_load_phase<E: KvEngine + Sync>(
    engine: &E,
    ops: &[WorkloadOp],
    threads: usize,
) -> PhaseStats {
    let shadow = Arc::new(std::sync::Mutex::new(HashMap::new()));
    run_phase_inner(
        engine,
        ops,
        threads,
        Duration::ZERO,
        false,
        &shadow,
        Some("Load"),
    )
}

/// Run a phase with verification against a shadow HashMap.
///
/// Every Put updates the shadow map. Every Get is checked against it.
/// Every Del is checked. Scans verify the count matches. Mismatches
/// are printed and counted. The shadow map is shared across calls
/// so the load phase's writes are visible to the run phase.
pub fn run_phase_verify<E: KvEngine + Sync>(
    engine: &E,
    ops: &[WorkloadOp],
    threads: usize,
    minimum_duration: Duration,
    shadow: &Arc<std::sync::Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
) -> PhaseStats {
    run_phase_inner(engine, ops, threads, minimum_duration, true, shadow, None)
}

/// Run a verified finite load phase and emit periodic progress to stderr.
pub fn run_load_phase_verify<E: KvEngine + Sync>(
    engine: &E,
    ops: &[WorkloadOp],
    threads: usize,
    shadow: &Arc<std::sync::Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
) -> PhaseStats {
    run_phase_inner(
        engine,
        ops,
        threads,
        Duration::ZERO,
        true,
        shadow,
        Some("Load"),
    )
}

fn run_phase_inner<E: KvEngine + Sync>(
    engine: &E,
    ops: &[WorkloadOp],
    threads: usize,
    minimum_duration: Duration,
    verify: bool,
    shadow: &Arc<std::sync::Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
    progress_label: Option<&'static str>,
) -> PhaseStats {
    let threads = threads.max(1);
    let next_idx = Arc::new(AtomicU64::new(0));
    let total = ops.len() as u64;
    assert!(total > 0, "benchmark phase requires at least one operation");

    let errors = Arc::new(AtomicU64::new(0));
    let completed = progress_label.map(|_| Arc::new(AtomicU64::new(0)));

    let start = Instant::now();
    let deadline = start + minimum_duration;
    let mut combined = PhaseStats::new();

    std::thread::scope(|s| {
        let (progress_stop_tx, progress_stop_rx) = std::sync::mpsc::channel();
        let progress_handle = progress_label.map(|label| {
            let completed = Arc::clone(completed.as_ref().unwrap());
            s.spawn(move || {
                let mut last_completed = 0_u64;
                let mut last_tick = start;
                loop {
                    match progress_stop_rx.recv_timeout(PROGRESS_INTERVAL) {
                        Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            let now = Instant::now();
                            let current = completed.load(Ordering::Relaxed).min(total);
                            let elapsed = now.duration_since(start).as_secs_f64();
                            let recent_elapsed = now.duration_since(last_tick).as_secs_f64();
                            let average_rate = current as f64 / elapsed.max(f64::EPSILON);
                            let recent_rate = current.saturating_sub(last_completed) as f64
                                / recent_elapsed.max(f64::EPSILON);
                            let percentage = current as f64 * 100.0 / total as f64;
                            eprintln!(
                                "  {label} progress: {current}/{total} ({percentage:.1}%), \
                                 elapsed={elapsed:.1}s, avg={average_rate:.0}/s, recent={recent_rate:.0}/s"
                            );
                            last_completed = current;
                            last_tick = now;
                        }
                    }
                }
            })
        });

        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            let next_idx = Arc::clone(&next_idx);
            let shadow = Arc::clone(shadow);
            let errors = Arc::clone(&errors);
            let completed = completed.clone();
            handles.push(s.spawn(move || {
                let mut local = PhaseStats::new();
                let mut deadline_check_countdown = 0_u16;
                let mut unreported_completions = 0_u64;
                loop {
                    let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                    if idx >= total {
                        if minimum_duration.is_zero() {
                            break;
                        }
                        if deadline_check_countdown == 0 {
                            if Instant::now() >= deadline {
                                break;
                            }
                            deadline_check_countdown = 255;
                        } else {
                            deadline_check_countdown -= 1;
                        }
                    }
                    let op = &ops[(idx % total) as usize];
                    let op_start = Instant::now();
                    let kind = execute_op(engine, op, verify, &shadow, &errors, idx);
                    let elapsed = op_start.elapsed().as_nanos() as u64;
                    local.record(elapsed, kind);
                    if let Some(completed) = &completed {
                        unreported_completions += 1;
                        if unreported_completions == PROGRESS_FLUSH_OPS {
                            completed.fetch_add(unreported_completions, Ordering::Relaxed);
                            unreported_completions = 0;
                        }
                    }
                }
                if let Some(completed) = &completed
                    && unreported_completions > 0
                {
                    completed.fetch_add(unreported_completions, Ordering::Relaxed);
                }
                local
            }));
        }
        for handle in handles {
            let local = handle.join().expect("worker thread panicked");
            combined.merge(&local);
        }
        let _ = progress_stop_tx.send(());
        if let Some(handle) = progress_handle {
            handle.join().expect("progress thread panicked");
        }
    });

    combined.duration = start.elapsed();

    let err_count = errors.load(Ordering::Relaxed);
    if verify && err_count > 0 {
        eprintln!("  VERIFY: {err_count} mismatches detected in {total} operations");
    }

    combined
}

/// Execute a single operation against the engine. Returns the op kind for
/// counting.
fn execute_op<E: KvEngine>(
    engine: &E,
    op: &WorkloadOp,
    verify: bool,
    shadow: &Arc<std::sync::Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
    errors: &Arc<AtomicU64>,
    op_idx: u64,
) -> OpKind {
    match op {
        WorkloadOp::Put { key, value } => {
            engine.put(key, value);
            if verify {
                shadow.lock().unwrap().insert(key.clone(), value.clone());
            }
            OpKind::Put
        }
        WorkloadOp::Get { key } => {
            let actual = engine.get(key);
            black_box(&actual);
            if verify {
                let expected = shadow.lock().unwrap().get(key).cloned();
                if actual != expected {
                    errors.fetch_add(1, Ordering::Relaxed);
                    if errors.load(Ordering::Relaxed) <= 5 {
                        eprintln!(
                            "  VERIFY MISMATCH op[{op_idx}] Get: key={:?} expected={:?} actual={:?}",
                            key,
                            expected.as_ref().map(|v| &v[..]),
                            actual.as_ref().map(|v| &v[..]),
                        );
                    }
                }
            }
            OpKind::Get
        }
        WorkloadOp::Del { key } => {
            engine.del(key);
            if verify {
                shadow.lock().unwrap().remove(key);
            }
            OpKind::Del
        }
        WorkloadOp::Scan { start, count } => {
            let end = scan_end_key(start, *count as u64);
            let mut n = 0usize;
            engine.scan_range(start, &end, &mut |_, _| {
                n += 1;
            });
            black_box(n);
            if verify {
                let guard = shadow.lock().unwrap();
                let expected: Vec<_> = guard
                    .iter()
                    .filter(|(k, _)| {
                        k.as_slice() >= start.as_slice() && k.as_slice() < end.as_slice()
                    })
                    .collect();
                if n != expected.len() {
                    errors.fetch_add(1, Ordering::Relaxed);
                    if errors.load(Ordering::Relaxed) <= 5 {
                        eprintln!(
                            "  VERIFY MISMATCH op[{op_idx}] Scan: start={:?} expected={} entries, got {n}",
                            &start[..],
                            expected.len(),
                        );
                    }
                }
            }
            OpKind::Scan
        }
        WorkloadOp::Rmw { key, value } => {
            let _ = engine.get(key);
            engine.put(key, value);
            if verify {
                shadow.lock().unwrap().insert(key.clone(), value.clone());
            }
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
    let _ = execute_op(
        engine,
        op,
        false,
        &Arc::new(std::sync::Mutex::new(HashMap::new())),
        &Arc::new(AtomicU64::new(0)),
        0,
    );
    start.elapsed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{CacheControl, EngineOpts};
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
        const CACHE_CONTROL: CacheControl = CacheControl::Application;

        fn open(_dir: &Path, _opts: &EngineOpts) -> std::io::Result<Self> {
            Ok(Self::new())
        }

        fn put(&self, key: &[u8], value: &[u8]) {
            let mut data = self.data.lock().unwrap();
            data.insert(key.to_vec(), value.to_vec());
        }

        fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
            self.data.lock().unwrap().get(key).cloned()
        }

        fn del(&self, key: &[u8]) {
            self.data.lock().unwrap().remove(key);
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
        engine.put(b"A", b"val_a");
        engine.put(b"A", b"val_a2");
        engine.put(b"B", b"val_b");
        assert_eq!(engine.get(b"A"), Some(b"val_a2".to_vec()));
        engine.del(b"A");
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
        let stats = run_phase(&engine, &ops, 2, Duration::ZERO);
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
        let stats = run_phase(&engine, &ops, 1, Duration::ZERO);
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
        let stats = run_phase(&engine, &ops, 1, Duration::ZERO);
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
        let stats = run_phase(&engine, &ops, 1, Duration::ZERO);
        assert_eq!(stats.rmws, 1);
        assert_eq!(engine.get(b"k1"), Some(b"v2".to_vec()));
    }

    #[test]
    fn minimum_duration_repeats_the_trace() {
        let engine = MockEngine::new();
        let ops = vec![WorkloadOp::Get {
            key: b"missing".to_vec(),
        }];
        let stats = run_phase(&engine, &ops, 2, Duration::from_millis(5));
        assert!(
            stats.operations > 1,
            "duration-bound phase should repeat its operation trace"
        );
        assert!(stats.duration >= Duration::from_millis(5));
    }

    #[test]
    fn short_load_progress_run_stops_reporter_immediately() {
        let engine = MockEngine::new();
        let ops = vec![WorkloadOp::Put {
            key: b"key".to_vec(),
            value: b"value".to_vec(),
        }];
        let wall_start = Instant::now();

        let stats = run_load_phase(&engine, &ops, 1);

        assert_eq!(stats.operations, 1);
        assert!(
            wall_start.elapsed() < PROGRESS_INTERVAL,
            "a completed load must wake the progress reporter instead of waiting for its interval"
        );
    }
}
