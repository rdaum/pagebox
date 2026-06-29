//! Optimistic/shared/exclusive latch used by Pagebox's hot storage and tree paths.
//!
//! `pagebox-hybrid-latch` implements a LeanStore-style "optimistic lock coupling"
//! primitive that lets the B+tree and buffer pool take reads on a single 64-bit
//! version word without touching a mutex, while still falling back to a real
//! reader/writer lock when optimistic access is impossible or when a mutation
//! needs mutual exclusion. The fast path is a single relaxed atomic load and a
//! final acquire load; the slow path delegates to `parking_lot::RawRwLock`.
//!
//! ## Why a hybrid latch
//!
//! Tree traversal in a swizzled-pointer buffer pool is dominated by read-only
//! descents that touch many pages per lookup. A plain `RwLock` read on every
//! node would be correct but cache-hostile: each acquire dirtyies a cache line in
//! the mutex word and forces an atomic CAS that serialises unrelated readers.
//! The hybrid latch keeps the common case free of write-side traffic — readers
//! snapshot the version word, do their work against the page, then call
//! [`OptimisticGuard::validate`]. If validation fails the reader discards its
//! work and restarts; otherwise it has observed a consistent view.
//!
//! ## Version-word encoding
//!
//! The latch is a single `AtomicU64` plus a reader/writer lock. Bit 0 is the
//! **exclusive flag**; the remaining bits hold an even **base version**:
//!
//! ```text
//!   readable:      base is even,   bit 0 == 0   →  optimistic reads allowed
//!   exclusive:     base is even,   bit 0 == 1   →  optimistic reads rejected
//! ```
//!
//! Entering an exclusive section sets bit 0 (`base | 1`); leaving clears it and
//! advances the base by two (`base + 2`). Stepping by two guarantees that a
//! readable version visible before a write `v` is never equal to any readable
//! version visible after it, so a stored snapshot that mismatches the live word
//! is unambiguous proof that the optimistic reader interleaved with a writer.
//! The encoding is checked by `helpers.rs` and exercised by loom.
//!
//! ## Modes and guards
//!
//! | Operation                  | Guard                | Mutex cost             |
//! |---------------------------|----------------------|------------------------|
//! | [`HybridLatch::optimistic_or_restart`] | [`OptimisticGuard`]  | none (load + load)     |
//! | [`HybridLatch::lock_shared`]            | [`SharedGuard`]      | shared (read)          |
//! | [`HybridLatch::lock_exclusive`]        | [`ExclusiveGuard`]  | exclusive (write)      |
//!
//! Optimistic guards can be promoted in two directions:
//!
//! - [`OptimisticGuard::upgrade_to_shared`] — take a shared lock, then re-check
//!   that the version word is still the one we snapshotted. If a writer moved
//!   it, restart; otherwise we now hold a real shared lock.
//! - [`OptimisticGuard::upgrade_to_exclusive`] — take the exclusive lock, then
//!   CAS the version word from our snapshot to `snapshot | 1`. A failing CAS
//!   means a concurrent writer beat us, so we drop the lock and restart.
//!
//! Each variant has a `try_*` form that never blocks; a blocked acquire is
//! reported as [`Restart`] rather than sleeping.
//!
//! ## Restart contract
//!
//! [`Restart`] is the unit of flow control returned wherever an optimistic
//! section observed (or would observe) writes it cannot safely ignore.
//! **Callers are obligated to loop on `Restart`** — the latch does not retry
//! internally. The B+tree bounds this loop with `LOOKUP_OPTIMISTIC_RESTART_LIMIT`
//! before falling back to a shared latch; see `pagebox-btree` for that policy.
//!
//! ## Overflow
//!
//! The base version advances by two per exclusive section and never rewinds. At
//! `u64::MAX - 1` the latch is one step from overflow; `lock_exclusive` and
//! `upgrade_to_exclusive` panic at that boundary (guarded by
//! `#[should_panic]` tests) rather than silently wrapping. In practice the
//! budget is ~9.2 × 10¹⁸ write critical sections per latch instance.
//!
//! ## Telemetry
//!
//! With the `latch-metrics` feature (default on, pulled in by the workspace
//! `metrics` feature) contended acquires record wait-time histograms keyed by
//! [`LatchWaitMode`]. Setting `PAGEBOX_TRACE_LATCH_WAITS=1` additionally
//! aggregates per-call-site contention; read it back with
//! [`top_latch_wait_sites`]. Both are no-ops under `cfg(loom)` and when the
//! feature is off, so the hot path stays dependency-free.
//!
//! ## Loom and Miri
//!
//! State transitions here are subtle enough to enumerate rather than stress:
//! the `loom_tests` module in `latch.rs` models exclusive-vs-optimistic
//! invalidation, concurrent upgrades from the same snapshot, and version
//! consistency across a writer and an optimistic reader. Build with
//! `RUSTFLAGS="--cfg loom" cargo test -p pagebox-hybrid-latch`.

mod helpers;
mod latch;
mod lock;

pub use crate::latch::{
    ExclusiveGuard, HybridLatch, LatchGuard, LatchWaitMode, LatchWaitSite, OptimisticGuard,
    Restart, SharedGuard, top_latch_wait_sites,
};
