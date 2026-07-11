#[cfg(loom)]
use loom::sync::atomic::{AtomicU64, Ordering};
#[cfg(all(not(loom), feature = "latch-metrics"))]
use std::collections::HashMap;
#[cfg(all(not(loom), feature = "latch-metrics"))]
use std::panic::Location;
#[cfg(not(loom))]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(all(not(loom), feature = "latch-metrics"))]
use std::sync::{Mutex, OnceLock};
#[cfg(all(not(loom), feature = "latch-metrics"))]
use std::time::Instant;

#[cfg(feature = "latch-metrics")]
use fast_telemetry::DeriveLabel;
#[cfg(all(not(loom), feature = "latch-metrics"))]
use fast_telemetry::{LabeledCounter, LabeledHistogram};

use crate::helpers::{
    can_advance_readable_version, enter_exclusive_version, exit_exclusive_version,
    optimistic_restart_required, optimistic_snapshot, optimistic_snapshot_still_valid,
    version_is_exclusive,
};
use crate::lock::imp as lock;

/// Signal that an optimistic read section observed a version change and must
/// restart.
///
/// Returned by [`HybridLatch::optimistic_or_restart`] (when a writer is
/// currently in its critical section), by [`OptimisticGuard::validate`] (when a
/// writer committed during the reader's section), and by the `upgrade_to_*`
/// paths (when a concurrent writer moved the version word between the snapshot
/// and the lock acquisition). Callers are obligated to loop on this — the
/// latch does not retry internally.
#[derive(Debug)]
pub struct Restart;

/// The kind of blocking acquire a waiter performed. Used as the label key for
/// the contention histograms emitted with the `latch-metrics` feature.
#[cfg_attr(feature = "latch-metrics", derive(DeriveLabel))]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "latch-metrics", label_name = "mode")]
pub enum LatchWaitMode {
    Shared,
    Exclusive,
    UpgradeShared,
    UpgradeExclusive,
}

/// One call site's aggregated contention, returned from
/// [`top_latch_wait_sites`]. Only populated when the `latch-metrics` feature is
/// enabled **and** `PAGEBOX_TRACE_LATCH_WAITS` is set at runtime; otherwise the
/// registry is never installed and [`top_latch_wait_sites`] returns an empty
/// vector.
#[derive(Clone, Debug)]
pub struct LatchWaitSite {
    pub mode: LatchWaitMode,
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
    pub contended_acquires: u64,
    pub total_wait_ns: u64,
    pub max_wait_ns: u64,
}

#[cfg(all(not(loom), feature = "latch-metrics"))]
struct LatchWaitMetrics {
    counts: LabeledCounter<LatchWaitMode>,
    latencies: LabeledHistogram<LatchWaitMode>,
}

#[cfg(all(not(loom), feature = "latch-metrics"))]
impl LatchWaitMetrics {
    fn new() -> Self {
        let shards = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        Self {
            counts: LabeledCounter::new(shards),
            latencies: LabeledHistogram::new(&latch_wait_latency_bounds_ns(), shards),
        }
    }

    #[inline]
    fn record(&self, mode: LatchWaitMode, wait_ns: u64) {
        self.counts.inc(mode);
        self.latencies.record(mode, wait_ns);
    }
}

#[cfg(all(not(loom), feature = "latch-metrics"))]
fn latch_wait_metrics() -> &'static LatchWaitMetrics {
    static METRICS: OnceLock<LatchWaitMetrics> = OnceLock::new();
    METRICS.get_or_init(LatchWaitMetrics::new)
}

#[cfg(all(not(loom), feature = "latch-metrics"))]
fn latch_wait_latency_bounds_ns() -> [u64; 13] {
    [
        100,
        500,
        1_000,
        5_000,
        10_000,
        50_000,
        100_000,
        500_000,
        1_000_000,
        5_000_000,
        10_000_000,
        50_000_000,
        100_000_000,
    ]
}

#[cfg(all(not(loom), feature = "latch-metrics"))]
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct LatchWaitSiteKey {
    mode: LatchWaitMode,
    file: &'static str,
    line: u32,
    column: u32,
}

#[cfg(all(not(loom), feature = "latch-metrics"))]
#[derive(Default)]
struct LatchWaitSiteStats {
    contended_acquires: u64,
    total_wait_ns: u64,
    max_wait_ns: u64,
}

#[cfg(all(not(loom), feature = "latch-metrics"))]
fn latch_site_registry() -> &'static Mutex<HashMap<LatchWaitSiteKey, LatchWaitSiteStats>> {
    static REGISTRY: OnceLock<Mutex<HashMap<LatchWaitSiteKey, LatchWaitSiteStats>>> =
        OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(all(not(loom), feature = "latch-metrics"))]
fn trace_latch_waits_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("PAGEBOX_TRACE_LATCH_WAITS").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
        )
    })
}

#[cfg(all(not(loom), feature = "latch-metrics"))]
#[inline]
fn record_contended_latch_wait(
    mode: LatchWaitMode,
    location: &'static Location<'static>,
    wait_ns: u64,
) {
    latch_wait_metrics().record(mode, wait_ns);
    if trace_latch_waits_enabled() {
        record_latch_wait_site(mode, location, wait_ns);
    }
}

#[cfg(all(not(loom), feature = "latch-metrics"))]
fn record_latch_wait_site(mode: LatchWaitMode, location: &'static Location<'static>, wait_ns: u64) {
    let key = LatchWaitSiteKey {
        mode,
        file: location.file(),
        line: location.line(),
        column: location.column(),
    };
    let mut registry = latch_site_registry().lock().unwrap();
    let entry = registry.entry(key).or_default();
    entry.contended_acquires = entry.contended_acquires.saturating_add(1);
    entry.total_wait_ns = entry.total_wait_ns.saturating_add(wait_ns);
    entry.max_wait_ns = entry.max_wait_ns.max(wait_ns);
}

/// Returns the `limit` call sites ranked by total wait time, descending.
///
/// This is the inspection surface for the optional `PAGEBOX_TRACE_LATCH_WAITS`
/// registry. It is cheap to call when the registry is absent (the function
/// short-circuits to an empty `Vec`), but it locks the global site registry
/// while iterating, so do not call it from a hot path.
///
/// Returns an empty vector when either:
/// - the `latch-metrics` feature is off (or `cfg(loom)` is set), or
/// - `PAGEBOX_TRACE_LATCH_WAITS` was not set at process start.
pub fn top_latch_wait_sites(limit: usize) -> Vec<LatchWaitSite> {
    #[cfg(any(loom, not(feature = "latch-metrics")))]
    {
        let _ = limit;
        Vec::new()
    }

    #[cfg(all(not(loom), feature = "latch-metrics"))]
    {
        if !trace_latch_waits_enabled() {
            return Vec::new();
        }
        let registry = latch_site_registry().lock().unwrap();
        let mut entries: Vec<_> = registry
            .iter()
            .map(|(key, stats)| LatchWaitSite {
                mode: key.mode,
                file: key.file,
                line: key.line,
                column: key.column,
                contended_acquires: stats.contended_acquires,
                total_wait_ns: stats.total_wait_ns,
                max_wait_ns: stats.max_wait_ns,
            })
            .collect();
        entries.sort_by(|a, b| {
            b.total_wait_ns
                .cmp(&a.total_wait_ns)
                .then_with(|| b.contended_acquires.cmp(&a.contended_acquires))
                .then_with(|| a.file.cmp(b.file))
                .then_with(|| a.line.cmp(&b.line))
                .then_with(|| a.column.cmp(&b.column))
        });
        entries.truncate(limit);
        entries
    }
}

/// A hybrid optimistic/shared/exclusive latch.
///
/// A `HybridLatch` is the combination of a 64-bit version word (the optimistic
/// fast path) and a `parking_lot` reader/writer lock (the fallback). See the
/// [crate-level documentation](crate) for the version-word encoding and the
/// rationale.
///
/// A reader that can tolerate restarts should reach for
/// [`optimistic_or_restart`](Self::optimistic_or_restart) or
/// [`optimistic_spin`](Self::optimistic_spin); a reader needing a stable
/// snapshot should use [`lock_shared`](Self::lock_shared); a mutating section
/// should use [`lock_exclusive`](Self::lock_exclusive). The `try_*` variants
/// report contention via [`Restart`] / `None` rather than blocking.
///
/// One latch instance per buffer frame is the intended deployment: see
/// `pagebox-storage::BufferFrame`. The latch is `Send + Sync`.
pub struct HybridLatch {
    version: AtomicU64,
    mutex: lock::InnerLock,
}

#[cfg(not(loom))]
unsafe impl Send for HybridLatch {}
#[cfg(not(loom))]
unsafe impl Sync for HybridLatch {}

impl Default for HybridLatch {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridLatch {
    /// Construct a latch at version `0` (readable, never written).
    pub fn new() -> Self {
        HybridLatch {
            version: AtomicU64::new(0),
            mutex: lock::InnerLock::new(),
        }
    }

    /// Begin an optimistic read section.
    ///
    /// Loads the version word with `Acquire`. If the exclusive bit is set (a
    /// writer is in its critical section) returns [`Err`](Restart); otherwise
    /// returns an [`OptimisticGuard`] holding the snapshot. The snapshot must be
    /// re-checked with [`OptimisticGuard::validate`] before any decision based
    /// on the read is committed — between the snapshot and `validate` a writer
    /// may have entered and left its critical section, advancing the base
    /// version by two.
    ///
    /// This performs no mutex traffic and no CAS; the cost is two atomic loads.
    /// Use [`optimistic_spin`](Self::optimistic_spin) when you would rather spin
    /// than restart on a busy latch.
    pub fn optimistic_or_restart(&self) -> Result<OptimisticGuard<'_>, Restart> {
        let v = self.version.load(Ordering::Acquire);
        let Some(snapshot) = optimistic_snapshot(v) else {
            return Err(Restart);
        };
        Ok(OptimisticGuard {
            latch: self,
            version: snapshot,
        })
    }

    /// Begin an optimistic read section, spinning while a writer holds the
    /// exclusive bit.
    ///
    /// Unlike [`optimistic_or_restart`](Self::optimistic_or_restart) this never
    /// returns [`Restart`]; it busy-waits (`std::hint::spin_loop`, or
    /// `loom::thread::yield_now` under `cfg(loom)`) until the version word is
    /// readable. Use it when the caller cannot easily roll back partial work —
    /// for example when an optimistic snapshot was already used to dereference
    /// a swizzled pointer that must be re-pinned on restart — but prefer
    /// `optimistic_or_restart` in general so the scheduler can make progress.
    pub fn optimistic_spin(&self) -> OptimisticGuard<'_> {
        loop {
            let v = self.version.load(Ordering::Acquire);
            if !version_is_exclusive(v) {
                return OptimisticGuard {
                    latch: self,
                    version: v,
                };
            }
            #[cfg(not(loom))]
            std::hint::spin_loop();
            #[cfg(loom)]
            loom::thread::yield_now();
        }
    }

    /// Acquire the latch exclusively, blocking until exclusive access is
    /// available.
    ///
    /// Takes the underlying writer lock (a `try_lock` first, then a blocking
    /// acquire), then sets the exclusive bit by storing `base | 1` with
    /// `Release`. While this guard is live, [`optimistic_or_restart`](Self::optimistic_or_restart)
    /// returns [`Restart`] and `validate` fails for any in-flight optimistic
    /// guard. On drop the exclusive bit is cleared and the base version is
    /// advanced by two, publishing the write to subsequent readers.
    ///
    /// Panics if the version word has reached its terminal value
    /// (`u64::MAX - 1` readable, i.e. one step from overflow); see the
    /// crate-level overflow note.
    #[track_caller]
    pub fn lock_exclusive(&self) -> ExclusiveGuard<'_> {
        #[cfg(all(not(loom), feature = "latch-metrics"))]
        let caller = Location::caller();
        let _inner = if let Some(inner) = self.mutex.try_lock_exclusive() {
            inner
        } else {
            #[cfg(all(not(loom), feature = "latch-metrics"))]
            let start = Instant::now();
            let inner = self.mutex.lock_exclusive();
            #[cfg(all(not(loom), feature = "latch-metrics"))]
            record_contended_latch_wait(
                LatchWaitMode::Exclusive,
                caller,
                start.elapsed().as_nanos() as u64,
            );
            inner
        };
        let current = self.version.load(Ordering::Relaxed);
        assert!(
            can_advance_readable_version(current),
            "hybrid latch version overflow on exclusive entry"
        );
        let v = enter_exclusive_version(current);
        self.version.store(v, Ordering::Release);
        ExclusiveGuard {
            latch: self,
            _inner,
            version: v,
        }
    }

    /// Try to acquire the latch exclusively without blocking.
    ///
    /// Returns `None` if the underlying writer lock is contended. On success
    /// behaves exactly like [`lock_exclusive`](Self::lock_exclusive) except
    /// that no contention telemetry is recorded (there was no wait).
    ///
    /// Use this on eviction and parent-link publication paths where the caller
    /// would rather skip the page than sleep; an `None` here is reported to the
    /// caller and the B+tree traversal in `pagebox-btree` retries via a
    /// different route.
    pub fn try_lock_exclusive(&self) -> Option<ExclusiveGuard<'_>> {
        let _inner = self.mutex.try_lock_exclusive()?;
        let current = self.version.load(Ordering::Relaxed);
        assert!(
            can_advance_readable_version(current),
            "hybrid latch version overflow on exclusive entry"
        );
        let v = enter_exclusive_version(current);
        self.version.store(v, Ordering::Release);
        Some(ExclusiveGuard {
            latch: self,
            _inner,
            version: v,
        })
    }

    /// Acquire the latch shared, blocking until a shared lock is available.
    ///
    /// Takes the underlying reader lock and snapshots the version word with
    /// `Acquire`. While a `SharedGuard` is live the exclusive bit cannot be
    /// set (the reader lock excludes writers), so an optimistic guard snapshotted
    /// *after* `lock_shared` returns is guaranteed to validate for the lifetime
    /// of the shared section. Multiple shared holders may coexist.
    ///
    /// Shared access is appropriate for scans that read more than a single slot
    /// — long enough that an optimistic section would restart too often, but
    /// not mutating.
    #[track_caller]
    pub fn lock_shared(&self) -> SharedGuard<'_> {
        #[cfg(all(not(loom), feature = "latch-metrics"))]
        let caller = Location::caller();
        let _inner = if let Some(inner) = self.mutex.try_lock_shared() {
            inner
        } else {
            #[cfg(all(not(loom), feature = "latch-metrics"))]
            let start = Instant::now();
            let inner = self.mutex.lock_shared();
            #[cfg(all(not(loom), feature = "latch-metrics"))]
            record_contended_latch_wait(
                LatchWaitMode::Shared,
                caller,
                start.elapsed().as_nanos() as u64,
            );
            inner
        };
        let v = self.version.load(Ordering::Acquire);
        SharedGuard {
            latch: self,
            _inner,
            version: v,
        }
    }

    /// Try to acquire the latch shared without blocking.
    ///
    /// Returns `None` if a writer (or a would-be writer waiting) holds the
    /// underlying lock; otherwise returns a [`SharedGuard`] equivalent to the
    /// blocking form. No telemetry is recorded.
    pub fn try_lock_shared(&self) -> Option<SharedGuard<'_>> {
        let _inner = self.mutex.try_lock_shared()?;
        let v = self.version.load(Ordering::Acquire);
        Some(SharedGuard {
            latch: self,
            _inner,
            version: v,
        })
    }

    /// Read the live version word with `Acquire`.
    ///
    /// Exposed for diagnostics and for the B+tree traversal's restart budget
    /// bookkeeping; it is **not** a substitute for an optimistic section. A
    /// value read here is stale the moment it is observed and must not be used
    /// to make decisions about page contents without an
    /// [`OptimisticGuard::validate`] step against a snapshot taken on the same
    /// thread.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }
}

/// An in-flight optimistic read of a [`HybridLatch`].
///
/// Construction snapshots the version word (`Acquire`); the guard's lifetime
/// bounds the optimistic section. Drop is a no-op — there is nothing to
/// release, because no mutex was acquired. The reader's obligation is to call
/// [`validate`](Self::validate) before acting on anything it read; if
/// validation returns [`Err`](Restart) the read interleaved with a writer and
/// must be repeated.
///
/// Alternatively, an optimistic guard can be promoted via
/// [`upgrade_to_shared`](Self::upgrade_to_shared) or
/// [`upgrade_to_exclusive`](Self::upgrade_to_exclusive); both promotion paths
/// re-check the snapshot and return [`Restart`] on a version mismatch.
pub struct OptimisticGuard<'a> {
    latch: &'a HybridLatch,
    version: u64,
}

impl<'a> OptimisticGuard<'a> {
    /// Re-load the version word and confirm no writer committed during the
    /// optimistic section.
    ///
    /// Returns `Ok(())` iff the live version still equals the snapshot, i.e.
    /// no exclusive section was entered and left between construction and this
    /// call. Returning [`Err`](Restart) is a request to retry the whole
    /// section; partial work based on the optimistic read must be discarded.
    ///
    /// Memory ordering: `Acquire` on the load pairs with the `Release` store
    /// in [`ExclusiveGuard::drop`], so anything the writer published before
    /// dropping its guard is visible to this reader once `validate` succeeds.
    pub fn validate(&self) -> Result<(), Restart> {
        let current = self.latch.version.load(Ordering::Acquire);
        if optimistic_restart_required(current, self.version) {
            Err(Restart)
        } else {
            Ok(())
        }
    }

    /// Promote the optimistic guard to an exclusive guard, blocking until the
    /// writer lock is acquired.
    ///
    /// Takes the writer lock (a `try_lock` first, then a blocking acquire),
    /// then atomically transitions the version word from the snapshotted value
    /// to `snapshot | 1` with a `compare_exchange`. If the CAS fails a
    /// concurrent writer moved the version between our snapshot and the lock
    /// acquisition; we drop the lock and return `Err(Restart)` so the caller
    /// can restart the section with a fresh optimistic snapshot.
    ///
    /// Panics at the terminal version boundary (same as
    /// [`HybridLatch::lock_exclusive`]).
    ///
    /// Contended waits are recorded under [`LatchWaitMode::UpgradeExclusive`]
    /// when the `latch-metrics` feature is enabled.
    #[track_caller]
    pub fn upgrade_to_exclusive(self) -> Result<ExclusiveGuard<'a>, Restart> {
        #[cfg(all(not(loom), feature = "latch-metrics"))]
        let caller = Location::caller();
        let inner = if let Some(inner) = self.latch.mutex.try_lock_exclusive() {
            inner
        } else {
            #[cfg(all(not(loom), feature = "latch-metrics"))]
            let start = Instant::now();
            let inner = self.latch.mutex.lock_exclusive();
            #[cfg(all(not(loom), feature = "latch-metrics"))]
            record_contended_latch_wait(
                LatchWaitMode::UpgradeExclusive,
                caller,
                start.elapsed().as_nanos() as u64,
            );
            inner
        };
        assert!(
            can_advance_readable_version(self.version),
            "hybrid latch version overflow on exclusive entry"
        );
        let new_version = enter_exclusive_version(self.version);
        let Ok(_) = self.latch.version.compare_exchange(
            self.version,
            new_version,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) else {
            drop(inner);
            return Err(Restart);
        };

        Ok(ExclusiveGuard {
            latch: self.latch,
            _inner: inner,
            version: new_version,
        })
    }

    /// Try to promote the optimistic guard to an exclusive guard without
    /// blocking.
    ///
    /// Reports contention from the writer lock as [`Restart`] rather than
    /// sleeping. On a successful CAS this is equivalent to
    /// [`upgrade_to_exclusive`](Self::upgrade_to_exclusive); the version-word
    /// transition is the same.
    pub fn try_upgrade_to_exclusive(self) -> Result<ExclusiveGuard<'a>, Restart> {
        let Some(inner) = self.latch.mutex.try_lock_exclusive() else {
            return Err(Restart);
        };
        assert!(
            can_advance_readable_version(self.version),
            "hybrid latch version overflow on exclusive entry"
        );
        let new_version = enter_exclusive_version(self.version);
        let Ok(_) = self.latch.version.compare_exchange(
            self.version,
            new_version,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) else {
            drop(inner);
            return Err(Restart);
        };

        Ok(ExclusiveGuard {
            latch: self.latch,
            _inner: inner,
            version: new_version,
        })
    }

    /// Promote the optimistic guard to a shared guard, blocking until a shared
    /// lock is acquired.
    ///
    /// Takes the reader lock (a `try_lock` first, then a blocking acquire),
    /// then re-checks that the live version word still equals the snapshot.
    /// If it moved, a writer committed during the window between snapshot and
    /// lock acquisition; we drop the shared lock and return `Err(Restart)`.
    ///
    /// On success the returned [`SharedGuard`] holds a real reader lock, so
    /// the version word cannot change for its lifetime — the snapshot is
    /// effectively frozen. The shared version is the optimistic snapshot, not a
    /// re-load.
    ///
    /// Contended waits are recorded under [`LatchWaitMode::UpgradeShared`].
    #[track_caller]
    pub fn upgrade_to_shared(self) -> Result<SharedGuard<'a>, Restart> {
        #[cfg(all(not(loom), feature = "latch-metrics"))]
        let caller = Location::caller();
        let inner = if let Some(inner) = self.latch.mutex.try_lock_shared() {
            inner
        } else {
            #[cfg(all(not(loom), feature = "latch-metrics"))]
            let start = Instant::now();
            let inner = self.latch.mutex.lock_shared();
            #[cfg(all(not(loom), feature = "latch-metrics"))]
            record_contended_latch_wait(
                LatchWaitMode::UpgradeShared,
                caller,
                start.elapsed().as_nanos() as u64,
            );
            inner
        };
        if !optimistic_snapshot_still_valid(
            self.latch.version.load(Ordering::Acquire),
            self.version,
        ) {
            drop(inner);
            return Err(Restart);
        }
        Ok(SharedGuard {
            latch: self.latch,
            _inner: inner,
            version: self.version,
        })
    }

    /// Try to promote the optimistic guard to a shared guard without blocking.
    ///
    /// Reports contention from the reader lock as [`Restart`] rather than
    /// sleeping. On success equivalent to
    /// [`upgrade_to_shared`](Self::upgrade_to_shared).
    pub fn try_upgrade_to_shared(self) -> Result<SharedGuard<'a>, Restart> {
        let Some(inner) = self.latch.mutex.try_lock_shared() else {
            return Err(Restart);
        };
        if !optimistic_snapshot_still_valid(
            self.latch.version.load(Ordering::Acquire),
            self.version,
        ) {
            drop(inner);
            return Err(Restart);
        }
        Ok(SharedGuard {
            latch: self.latch,
            _inner: inner,
            version: self.version,
        })
    }
}

/// A shared (read) guard on a [`HybridLatch`].
///
/// Constructed by [`HybridLatch::lock_shared`](HybridLatch::lock_shared),
/// [`HybridLatch::try_lock_shared`](HybridLatch::try_lock_shared), or by
/// promoting an [`OptimisticGuard`] via
/// [`OptimisticGuard::upgrade_to_shared`]. Holds the underlying reader lock,
/// so no exclusive section can be entered concurrently — optimistic readers
/// snapshotted after this guard is live will validate for its lifetime.
///
/// Drop releases the reader lock. There is no version advance: the version
/// word is unchanged across a shared section's entry and exit.
pub struct SharedGuard<'a> {
    #[allow(dead_code)]
    latch: &'a HybridLatch,
    #[cfg(not(loom))]
    _inner: lock::SharedInnerGuard<'a>,
    #[cfg(loom)]
    _inner: lock::InnerGuard<'a>,
    version: u64,
}

impl<'a> SharedGuard<'a> {
    /// The version word snapshotted at acquisition. Stable for the guard's
    /// lifetime, because holding the reader lock excludes writers.
    pub fn version(&self) -> u64 {
        self.version
    }
}

/// An exclusive (write) guard on a [`HybridLatch`].
///
/// Constructed by [`HybridLatch::lock_exclusive`](HybridLatch::lock_exclusive),
/// [`HybridLatch::try_lock_exclusive`](HybridLatch::try_lock_exclusive), or by
/// promoting an [`OptimisticGuard`] via
/// [`OptimisticGuard::upgrade_to_exclusive`]. While live, the version word has
/// its exclusive bit set, so optimistic readers reject their snapshots and
/// in-flight optimistic guards fail [`OptimisticGuard::validate`].
///
/// On drop the exclusive bit is cleared and the base version is advanced by
/// two with a `Release` store — this is the publish step that makes the writes
/// performed under the guard visible to subsequent optimistic readers (the
/// `Acquire` in `OptimisticGuard::validate` pairs with this store).
pub struct ExclusiveGuard<'a> {
    latch: &'a HybridLatch,
    #[cfg(not(loom))]
    _inner: lock::ExclusiveInnerGuard<'a>,
    #[cfg(loom)]
    _inner: lock::InnerGuard<'a>,
    version: u64,
}

impl<'a> ExclusiveGuard<'a> {
    /// The version word in effect under this guard (exclusive bit set). For
    /// diagnostics; rarely useful to callers — the guard itself is the proof
    /// of exclusive access.
    pub fn version(&self) -> u64 {
        self.version
    }
}

impl Drop for ExclusiveGuard<'_> {
    fn drop(&mut self) {
        let new_version = exit_exclusive_version(self.version);
        self.latch.version.store(new_version, Ordering::Release);
    }
}

/// Either an [`OptimisticGuard`] or a [`SharedGuard`], for callers that can
/// commit to either but want a single type to thread through.
///
/// `validate` is a no-op for the shared variant (a held shared lock cannot be
/// invalidated) and delegates to [`OptimisticGuard::validate`] for the
/// optimistic variant. There is no exclusive variant: an exclusive section is
/// always terminal in a flow where this enum is useful.
pub enum LatchGuard<'a> {
    Optimistic(OptimisticGuard<'a>),
    Shared(SharedGuard<'a>),
}

impl<'a> LatchGuard<'a> {
    /// Cheap when shared, may return [`Restart`] when optimistic. See
    /// [`OptimisticGuard::validate`].
    pub fn validate(&self) -> Result<(), Restart> {
        match self {
            LatchGuard::Optimistic(g) => g.validate(),
            LatchGuard::Shared(_) => Ok(()),
        }
    }
}

#[cfg(test)]
#[cfg(not(loom))]
mod tests {
    use super::*;
    use crate::helpers::TEST_EXCLUSIVE_BIT as EXCLUSIVE_BIT;
    use std::sync::atomic::AtomicU64;
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn exclusive_invalidates_optimistic() {
        let latch = HybridLatch::new();
        let guard = latch.optimistic_or_restart().unwrap();

        {
            let _exc = latch.lock_exclusive();
            assert!(latch.optimistic_or_restart().is_err());
        }

        assert!(guard.validate().is_err());

        let guard2 = latch.optimistic_or_restart().unwrap();
        assert!(guard2.validate().is_ok());
    }

    #[test]
    fn version_increments_on_exclusive() {
        let latch = HybridLatch::new();
        assert_eq!(latch.version(), 0);

        {
            let _exc = latch.lock_exclusive();
            assert_eq!(latch.version(), 1);
        }
        assert_eq!(latch.version(), 2);

        {
            let _exc = latch.lock_exclusive();
        }
        assert_eq!(latch.version(), 4);
    }

    #[test]
    #[should_panic(expected = "hybrid latch version overflow on exclusive entry")]
    fn lock_exclusive_panics_at_terminal_readable_version() {
        let latch = HybridLatch::new();
        latch
            .version
            .store(u64::MAX - EXCLUSIVE_BIT, Ordering::Relaxed);
        let _exc = latch.lock_exclusive();
    }

    #[test]
    #[should_panic(expected = "hybrid latch version overflow on exclusive entry")]
    fn upgrade_to_exclusive_panics_at_terminal_readable_version() {
        let latch = HybridLatch::new();
        latch
            .version
            .store(u64::MAX - EXCLUSIVE_BIT, Ordering::Relaxed);
        let guard = latch.optimistic_or_restart().unwrap();
        let _exc = guard.upgrade_to_exclusive();
    }

    #[test]
    fn upgrade_optimistic_to_exclusive() {
        let latch = HybridLatch::new();
        let guard = latch.optimistic_or_restart().unwrap();
        let exc = guard.upgrade_to_exclusive().unwrap();
        assert_eq!(exc.version(), 1);
        drop(exc);
        assert_eq!(latch.version(), 2);
    }

    #[test]
    fn upgrade_fails_if_version_changed() {
        let latch = HybridLatch::new();
        let guard = latch.optimistic_or_restart().unwrap();

        {
            let _exc = latch.lock_exclusive();
        }

        assert!(guard.upgrade_to_exclusive().is_err());
    }

    #[test]
    fn concurrent_readers_and_writer() {
        let latch = Arc::new(HybridLatch::new());
        let data = Arc::new(AtomicU64::new(0));
        let barrier = Arc::new(Barrier::new(3));

        let num_iterations = 10_000;

        let w_latch = latch.clone();
        let w_data = data.clone();
        let w_barrier = barrier.clone();
        let writer = thread::spawn(move || {
            w_barrier.wait();
            for i in 1..=num_iterations {
                let _exc = w_latch.lock_exclusive();
                w_data.store(i, Ordering::Release);
            }
        });

        let mut readers = Vec::new();
        for _ in 0..2 {
            let r_latch = latch.clone();
            let r_data = data.clone();
            let r_barrier = barrier.clone();
            readers.push(thread::spawn(move || {
                r_barrier.wait();
                let mut restarts = 0u64;
                let mut reads = 0u64;
                loop {
                    let guard = match r_latch.optimistic_or_restart() {
                        Ok(g) => g,
                        Err(Restart) => {
                            restarts += 1;
                            continue;
                        }
                    };
                    let val = r_data.load(Ordering::Acquire);
                    if guard.validate().is_err() {
                        restarts += 1;
                        continue;
                    }
                    reads += 1;
                    if val == num_iterations {
                        break;
                    }
                }
                (reads, restarts)
            }));
        }

        writer.join().unwrap();
        for r in readers {
            let (reads, restarts) = r.join().unwrap();
            assert!(reads > 0, "reader should have completed some reads");
            eprintln!("reads: {reads}, restarts: {restarts}");
        }
    }

    #[test]
    fn concurrent_upgrade_contention() {
        let latch = Arc::new(HybridLatch::new());
        let barrier = Arc::new(Barrier::new(2));

        let results: Vec<_> = (0..2)
            .map(|_| {
                let l = latch.clone();
                let b = barrier.clone();
                thread::spawn(move || {
                    let guard = l.optimistic_spin();
                    b.wait();
                    guard.upgrade_to_exclusive().is_ok()
                })
            })
            .collect();

        let outcomes: Vec<bool> = results.into_iter().map(|h| h.join().unwrap()).collect();
        let successes = outcomes.iter().filter(|&&x| x).count();
        assert_eq!(
            successes, 1,
            "expected exactly one upgrade to succeed, got {outcomes:?}"
        );
    }

    #[test]
    fn upgrade_to_shared_succeeds_uncontended() {
        let latch = HybridLatch::new();
        let guard = latch.optimistic_or_restart().unwrap();
        let shared = guard.upgrade_to_shared().unwrap();
        assert_eq!(shared.version(), 0);
    }

    #[test]
    fn try_upgrade_to_shared_succeeds_uncontended() {
        let latch = HybridLatch::new();
        let guard = latch.optimistic_or_restart().unwrap();
        let shared = guard.try_upgrade_to_shared().unwrap();
        assert_eq!(shared.version(), 0);
    }

    #[test]
    fn try_upgrade_to_shared_fails_if_version_changed() {
        let latch = HybridLatch::new();
        let guard = latch.optimistic_or_restart().unwrap();
        {
            let _exc = latch.lock_exclusive();
        }
        assert!(guard.try_upgrade_to_shared().is_err());
    }

    #[test]
    fn multiple_shared_holders_coexist() {
        let latch = Arc::new(HybridLatch::new());
        let s1 = latch.lock_shared();
        let s2 = latch.lock_shared();
        // Both shared guards should coexist; optimistic reads should also work.
        let opt = latch.optimistic_or_restart().unwrap();
        assert!(opt.validate().is_ok());
        drop(s1);
        drop(s2);
    }

    #[test]
    fn shared_blocks_exclusive_try_lock() {
        let latch = HybridLatch::new();
        let _shared = latch.lock_shared();
        // try_lock_exclusive should fail while shared is held.
        assert!(latch.try_lock_exclusive().is_none());
        drop(_shared);
        // After dropping shared, exclusive should succeed.
        assert!(latch.try_lock_exclusive().is_some());
    }

    // -----------------------------------------------------------------------
    // helpers.rs bit-math roundtrips
    // -----------------------------------------------------------------------

    use crate::helpers::{
        advance_readable_version, can_advance_readable_version, enter_exclusive_version,
        exclusive_base_version, exclusive_mask, exit_exclusive_version, optimistic_read_allowed,
        optimistic_snapshot, version_is_exclusive,
    };

    #[test]
    fn helpers_exclusive_mask_roundtrip() {
        for v in [0u64, 1, 2, 4, 100, u64::MAX - 1, u64::MAX] {
            let masked = exclusive_mask(v);
            assert_eq!(masked, v & 1);
            assert_eq!(version_is_exclusive(v), masked != 0);
        }
    }

    #[test]
    fn helpers_enter_exit_exclusive_roundtrip() {
        for base in [0u64, 2, 4, 100, 1000] {
            let exc = enter_exclusive_version(base);
            assert!(version_is_exclusive(exc));
            assert_eq!(exclusive_base_version(exc), base);
            let next = exit_exclusive_version(exc);
            assert_eq!(next, base + 2);
            assert!(!version_is_exclusive(next));
        }
    }

    #[test]
    fn helpers_optimistic_snapshot_allowed_when_not_exclusive() {
        for v in [0u64, 2, 4, 100] {
            assert!(optimistic_read_allowed(v));
            assert_eq!(optimistic_snapshot(v), Some(v));
        }
    }

    #[test]
    fn helpers_optimistic_snapshot_none_when_exclusive() {
        for v in [1u64, 3, 5, 101] {
            assert!(!optimistic_read_allowed(v));
            assert_eq!(optimistic_snapshot(v), None);
        }
    }

    #[test]
    fn helpers_can_advance_readable_version_boundary() {
        assert!(can_advance_readable_version(0));
        assert!(can_advance_readable_version(u64::MAX - 2));
        assert!(!can_advance_readable_version(u64::MAX - 1));
        assert!(!can_advance_readable_version(u64::MAX));
    }

    #[test]
    fn helpers_advance_readable_version_steps_by_two() {
        assert_eq!(advance_readable_version(0), 2);
        assert_eq!(advance_readable_version(2), 4);
        assert_eq!(advance_readable_version(u64::MAX - 2), u64::MAX);
    }

    #[test]
    #[should_panic(expected = "hybrid latch version overflow on exclusive release")]
    fn exit_exclusive_version_panics_at_terminal_version() {
        // u64::MAX is exclusive (bit 0 set); base is u64::MAX - 1;
        // advance_readable_version(u64::MAX - 1) overflows.
        let _ = exit_exclusive_version(u64::MAX);
    }
}

#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;
    use loom::sync::Arc;
    use loom::thread;

    #[test]
    fn loom_exclusive_invalidates_optimistic() {
        loom::model(|| {
            let latch = Arc::new(HybridLatch::new());

            let l2 = latch.clone();
            let writer = thread::spawn(move || {
                let _exc = l2.lock_exclusive();
            });

            let guard = latch.optimistic_or_restart();
            if let Ok(g) = guard {
                let _ = g.validate();
            }

            writer.join().unwrap();
        });
    }

    #[test]
    fn loom_concurrent_upgrade_same_snapshot() {
        loom::model(|| {
            let latch = Arc::new(HybridLatch::new());
            let ready = Arc::new(loom::sync::atomic::AtomicU32::new(0));

            let l1 = latch.clone();
            let r1 = ready.clone();
            let l2 = latch.clone();
            let r2 = ready.clone();

            let t1 = thread::spawn(move || {
                let g = l1.optimistic_or_restart().unwrap();
                r1.fetch_add(1, Ordering::Release);
                while r1.load(Ordering::Acquire) < 2 {
                    loom::thread::yield_now();
                }
                g.upgrade_to_exclusive().is_ok()
            });

            let t2 = thread::spawn(move || {
                let g = l2.optimistic_or_restart().unwrap();
                r2.fetch_add(1, Ordering::Release);
                while r2.load(Ordering::Acquire) < 2 {
                    loom::thread::yield_now();
                }
                g.upgrade_to_exclusive().is_ok()
            });

            let r1 = t1.join().unwrap();
            let r2 = t2.join().unwrap();

            assert!(!(r1 && r2), "both upgrades from same snapshot succeeded");
        });
    }

    #[test]
    fn loom_writer_reader_version_consistency() {
        loom::model(|| {
            let latch = Arc::new(HybridLatch::new());
            let data = Arc::new(loom::sync::atomic::AtomicU64::new(0));

            let wl = latch.clone();
            let wd = data.clone();
            let writer = thread::spawn(move || {
                let _exc = wl.lock_exclusive();
                wd.store(42, Ordering::Release);
            });

            let guard = latch.optimistic_or_restart();
            if let Ok(g) = guard {
                let val = data.load(Ordering::Acquire);
                if g.validate().is_ok() {
                    assert_eq!(val, 0);
                }
            }

            writer.join().unwrap();
        });
    }

    #[test]
    fn loom_upgrade_to_shared_restarts_on_version_change() {
        loom::model(|| {
            let latch = Arc::new(HybridLatch::new());
            let ready = Arc::new(loom::sync::atomic::AtomicU32::new(0));

            let reader_latch = latch.clone();
            let reader_ready = ready.clone();
            let reader = thread::spawn(move || {
                let guard = reader_latch.optimistic_or_restart().unwrap();
                reader_ready.store(1, Ordering::Release);
                while reader_ready.load(Ordering::Acquire) < 2 {
                    loom::thread::yield_now();
                }
                guard.upgrade_to_shared().is_ok()
            });

            let writer_latch = latch.clone();
            let writer_ready = ready.clone();
            let writer = thread::spawn(move || {
                while writer_ready.load(Ordering::Acquire) < 1 {
                    loom::thread::yield_now();
                }
                {
                    let _exc = writer_latch.lock_exclusive();
                }
                writer_ready.store(2, Ordering::Release);
            });

            let upgraded = reader.join().unwrap();
            writer.join().unwrap();

            if latch.version() != 0 {
                assert!(
                    !upgraded,
                    "upgrade_to_shared succeeded across version change"
                );
            }
        });
    }

    #[test]
    fn loom_try_upgrade_to_exclusive_same_snapshot() {
        loom::model(|| {
            let latch = Arc::new(HybridLatch::new());
            let ready = Arc::new(loom::sync::atomic::AtomicU32::new(0));

            let l1 = latch.clone();
            let r1 = ready.clone();
            let l2 = latch.clone();
            let r2 = ready.clone();

            let t1 = thread::spawn(move || {
                let g = l1.optimistic_or_restart().unwrap();
                r1.fetch_add(1, Ordering::Release);
                while r1.load(Ordering::Acquire) < 2 {
                    loom::thread::yield_now();
                }
                g.try_upgrade_to_exclusive().is_ok()
            });

            let t2 = thread::spawn(move || {
                let g = l2.optimistic_or_restart().unwrap();
                r2.fetch_add(1, Ordering::Release);
                while r2.load(Ordering::Acquire) < 2 {
                    loom::thread::yield_now();
                }
                g.try_upgrade_to_exclusive().is_ok()
            });

            let r1 = t1.join().unwrap();
            let r2 = t2.join().unwrap();

            assert!(
                !(r1 && r2),
                "both try-upgrades from same snapshot succeeded"
            );
        });
    }
}
