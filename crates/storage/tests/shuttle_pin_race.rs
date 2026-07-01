//! Shuttle test for the pin-count TOCTOU race between `try_pin_hot_or_cool_swip`
//! and `with_single_evict_candidate`.
//!
//! The invariant: if the reader's `state.load(Acquire)` sees `Resident`
//! (pin succeeds), the frame must NOT be freed while the reader holds the
//! pin (between `fetch_add` and `fetch_sub`). If it is, the reader is
//! holding a pin on a frame that may be reused for a different page.

use std::sync::atomic::{AtomicU64, Ordering};

use shuttle::sync::Arc;
use shuttle::thread;

const RESIDENT: u64 = 2;
const EVICTING: u64 = 3;
const FREE: u64 = 0;

/// Minimal model: no eviction_mu, no rwlock. Just the raw atomics.
///
/// Reader (try_pin_hot_or_cool_swip):
///   1. pin_count.fetch_add(1)
///   2. state.load(Acquire)
///   3. if Resident → pin succeeded (invariant: frame not freed while pinned)
///   4. if not Resident → undo pin, return false
///
/// Evictor (with_single_evict_candidate + unswizzle_and_free):
///   1. pin_count.load(Acquire) → must be 0
///   2. CAS state Resident→Evicting
///   3. re-check pin_count (Kimi's fix)
///   4. free: state = Free
///
/// We track whether the reader was pinned (step 3) and whether the evictor
/// freed the frame (step 4) concurrently. The bug is: reader sees Resident
/// (pinned) AND evictor freed the frame while the reader was pinned.
#[test]
fn shuttle_pin_evict_minimal() {
    shuttle::check_random(
        || {
            let state = Arc::new(AtomicU64::new(RESIDENT));
            let pin_count = Arc::new(AtomicU64::new(0));
            // Track: did the evictor free the frame while the reader held a pin?
            let freed_while_pinned = Arc::new(AtomicU64::new(0));

            let s = state.clone();
            let pc = pin_count.clone();
            let fwp = freed_while_pinned.clone();
            let reader = thread::spawn(move || {
                // try_pin_hot_or_cool_swip
                pc.fetch_add(1, Ordering::Relaxed);
                let st = s.load(Ordering::Acquire);
                if st != RESIDENT {
                    pc.fetch_sub(1, Ordering::Relaxed);
                    return; // pin failed, no invariant to check
                }
                // Pin succeeded. The frame must not be freed while we hold
                // this pin. The evictor checks freed_while_pinned to detect
                // if it's about to free a frame that someone has pinned.
                pc.fetch_sub(1, Ordering::Relaxed);
            });

            let s2 = state.clone();
            let pc2 = pin_count.clone();
            let fwp2 = freed_while_pinned.clone();
            let evictor = thread::spawn(move || {
                if pc2.load(Ordering::Acquire) != 0 {
                    return;
                }
                if s2
                    .compare_exchange(RESIDENT, EVICTING, Ordering::Relaxed, Ordering::Relaxed)
                    .is_err()
                {
                    return;
                }
                // Re-check pin_count after CAS
                if pc2.load(Ordering::Acquire) != 0 {
                    s2.store(RESIDENT, Ordering::Relaxed);
                    return;
                }
                // Before freeing, check if someone snuck in a pin
                if pc2.load(Ordering::Acquire) != 0 {
                    s2.store(RESIDENT, Ordering::Relaxed);
                    return;
                }
                // Free the frame
                fwp2.store(1, Ordering::Release);
                s2.store(FREE, Ordering::Relaxed);
            });

            reader.join().unwrap();
            evictor.join().unwrap();
            assert_eq!(
                freed_while_pinned.load(Ordering::Acquire),
                0,
                "BUG: evictor freed a frame while a reader held a pin"
            );
        },
        1000,
    );
}

/// Same as above but check that the evictor never frees while pin_count > 0.
/// The real bug: between the evictor's last pin_count check and the free,
/// the reader can fetch_add(1). The evictor then frees the frame with
/// pin_count = 1.
#[test]
fn shuttle_pin_evict_freed_with_nonzero_pincount() {
    shuttle::check_random(
        || {
            let state = Arc::new(AtomicU64::new(RESIDENT));
            let pin_count = Arc::new(AtomicU64::new(0));
            // At the moment of free, what was pin_count?
            let pincount_at_free = Arc::new(AtomicU64::new(u64::MAX));

            let s = state.clone();
            let pc = pin_count.clone();
            let reader = thread::spawn(move || {
                pc.fetch_add(1, Ordering::Relaxed);
                let st = s.load(Ordering::Acquire);
                if st != RESIDENT {
                    pc.fetch_sub(1, Ordering::Relaxed);
                }
                // If Resident: we hold a pin. Sleep to widen the window.
                // Then release.
                pc.fetch_sub(1, Ordering::Relaxed);
            });

            let s2 = state.clone();
            let pc2 = pin_count.clone();
            let pf2 = pincount_at_free.clone();
            let evictor = thread::spawn(move || {
                if pc2.load(Ordering::Acquire) != 0 {
                    return;
                }
                if s2
                    .compare_exchange(RESIDENT, EVICTING, Ordering::Relaxed, Ordering::Relaxed)
                    .is_err()
                {
                    return;
                }
                if pc2.load(Ordering::Acquire) != 0 {
                    s2.store(RESIDENT, Ordering::Relaxed);
                    return;
                }
                // Free: record pin_count at free time
                let pc_at_free = pc2.load(Ordering::Acquire);
                pf2.store(pc_at_free, Ordering::Release);
                s2.store(FREE, Ordering::Relaxed);
            });

            reader.join().unwrap();
            evictor.join().unwrap();
            let pc_at_free = pincount_at_free.load(Ordering::Acquire);
            if pc_at_free != u64::MAX {
                assert_eq!(
                    pc_at_free, 0,
                    "BUG: evictor freed frame with pin_count={pc_at_free}"
                );
            }
        },
        1000,
    );
}

/// Model with the rwlock (eviction_mu). The reader takes read lock,
/// the evictor takes write lock before freeing. This should prevent
/// the reader from being between fetch_add and state.load when the
/// evictor frees.
#[test]
fn shuttle_pin_evict_with_rwlock() {
    shuttle::check_random(
        || {
            let state = Arc::new(AtomicU64::new(RESIDENT));
            let pin_count = Arc::new(AtomicU64::new(0));
            let pincount_at_free = Arc::new(AtomicU64::new(u64::MAX));
            let mu = Arc::new(shuttle::sync::RwLock::new(()));

            let s = state.clone();
            let pc = pin_count.clone();
            let mu_r = mu.clone();
            let reader = thread::spawn(move || {
                let _guard = mu_r.read(); // blocks evictor's write
                pc.fetch_add(1, Ordering::Relaxed);
                let st = s.load(Ordering::Acquire);
                if st != RESIDENT {
                    pc.fetch_sub(1, Ordering::Relaxed);
                }
                pc.fetch_sub(1, Ordering::Relaxed);
            });

            let s2 = state.clone();
            let pc2 = pin_count.clone();
            let pf2 = pincount_at_free.clone();
            let mu_w = mu.clone();
            let evictor = thread::spawn(move || {
                if pc2.load(Ordering::Acquire) != 0 {
                    return;
                }
                if s2
                    .compare_exchange(RESIDENT, EVICTING, Ordering::Relaxed, Ordering::Relaxed)
                    .is_err()
                {
                    return;
                }
                if pc2.load(Ordering::Acquire) != 0 {
                    s2.store(RESIDENT, Ordering::Relaxed);
                    return;
                }
                // Take write lock — blocks until reader releases read lock
                let _guard = mu_w.write();
                if pc2.load(Ordering::Acquire) != 0 {
                    s2.store(RESIDENT, Ordering::Relaxed);
                    return;
                }
                let pc_at_free = pc2.load(Ordering::Acquire);
                pf2.store(pc_at_free, Ordering::Release);
                s2.store(FREE, Ordering::Relaxed);
            });

            reader.join().unwrap();
            evictor.join().unwrap();
            let pc_at_free = pincount_at_free.load(Ordering::Acquire);
            if pc_at_free != u64::MAX {
                assert_eq!(
                    pc_at_free, 0,
                    "BUG: evictor freed frame with pin_count={pc_at_free}"
                );
            }
        },
        1000,
    );
}

/// Model the ACTUAL code: the reader's lock_hot_pin is CONDITIONAL.
/// When eviction_writer_pending == 0 AND budget is high, the reader
/// skips the read lock (fast path). The evictor sets eviction_writer_pending
/// AFTER the CAS, right before acquiring the write lock.
///
/// Race window: reader checks ewp (sees 0, no lock) → evictor CASes →
/// evictor sets ewp → evictor acquires write lock → reader fetch_adds →
/// reader loads state (sees Evicting, fails). ← this is fine
///
/// But: reader checks ewp (0) → reader fetch_adds → reader loads state
/// (still Resident, CAS hasn't happened) → evictor CASes → evictor re-checks
/// pin_count (sees 1) → evictor reverts. ← also fine
///
/// The real race: reader fetch_adds AFTER evictor's re-check but BEFORE
/// evictor sets ewp. Then evictor sets ewp, acquires write lock, checks
/// pin_count (sees 1 from reader), reverts. ← fine
///
/// Wait, so the conditional lock should be fine? Let me test.
#[test]
fn shuttle_pin_evict_conditional_lock() {
    shuttle::check_random(
        || {
            let state = Arc::new(AtomicU64::new(RESIDENT));
            let pin_count = Arc::new(AtomicU64::new(0));
            let pincount_at_free = Arc::new(AtomicU64::new(u64::MAX));
            let ewp = Arc::new(AtomicU64::new(0));
            let mu = Arc::new(shuttle::sync::RwLock::new(()));

            let s = state.clone();
            let pc = pin_count.clone();
            let ewp_r = ewp.clone();
            let mu_r = mu.clone();
            let reader = thread::spawn(move || {
                // lock_hot_pin: conditional
                let _guard = if ewp_r.load(Ordering::Acquire) != 0 {
                    Some(mu_r.read())
                } else {
                    None
                };
                pc.fetch_add(1, Ordering::Relaxed);
                let st = s.load(Ordering::Acquire);
                if st != RESIDENT {
                    pc.fetch_sub(1, Ordering::Relaxed);
                }
                pc.fetch_sub(1, Ordering::Relaxed);
            });

            let s2 = state.clone();
            let pc2 = pin_count.clone();
            let pf2 = pincount_at_free.clone();
            let ewp2 = ewp.clone();
            let mu_w = mu.clone();
            let evictor = thread::spawn(move || {
                if pc2.load(Ordering::Acquire) != 0 {
                    return;
                }
                if s2
                    .compare_exchange(RESIDENT, EVICTING, Ordering::Relaxed, Ordering::Relaxed)
                    .is_err()
                {
                    return;
                }
                if pc2.load(Ordering::Acquire) != 0 {
                    s2.store(RESIDENT, Ordering::Relaxed);
                    return;
                }
                // Set eviction_writer_pending, then take write lock
                ewp2.fetch_add(1, Ordering::AcqRel);
                let _guard = mu_w.write();
                ewp2.fetch_sub(1, Ordering::AcqRel);
                // can_free: re-check pin_count
                if pc2.load(Ordering::Acquire) != 0 {
                    s2.store(RESIDENT, Ordering::Relaxed);
                    return;
                }
                let pc_at_free = pc2.load(Ordering::Acquire);
                pf2.store(pc_at_free, Ordering::Release);
                s2.store(FREE, Ordering::Relaxed);
            });

            reader.join().unwrap();
            evictor.join().unwrap();
            let pc_at_free = pincount_at_free.load(Ordering::Acquire);
            if pc_at_free != u64::MAX {
                assert_eq!(
                    pc_at_free, 0,
                    "BUG: evictor freed frame with pin_count={pc_at_free}"
                );
            }
        },
        1000,
    );
}

/// Model the `try_fix_orphan_raw` race. In this path:
/// 1. Look up frame via page_table (already resident)
/// 2. try_lock_hot_pin() → may return Some(None) (no lock) on fast path
/// 3. pin_count.fetch_add(1)
/// 4. state.load(Acquire) → if Resident, pin succeeds
/// 5. also check pid == page_id
///
/// The evictor races the same as before. But with try_lock_hot_pin,
/// if budget is low, it calls try_read() which may fail → returns None
/// → the orphan fix fails → caller retries or restarts.
///
/// But if budget is HIGH (eviction not active), try_lock_hot_pin returns
/// Some(None) → no read lock → same race as the minimal model.
#[test]
fn shuttle_fix_orphan_race() {
    shuttle::check_random(
        || {
            let state = Arc::new(AtomicU64::new(RESIDENT));
            let pin_count = Arc::new(AtomicU64::new(0));
            let pincount_at_free = Arc::new(AtomicU64::new(u64::MAX));

            let s = state.clone();
            let pc = pin_count.clone();
            let reader = thread::spawn(move || {
                // try_fix_orphan_raw: no lock_hot_pin on fast path
                // (budget is high, eviction_writer_pending == 0)
                pc.fetch_add(1, Ordering::Relaxed);
                let st = s.load(Ordering::Acquire);
                if st != RESIDENT {
                    pc.fetch_sub(1, Ordering::Relaxed);
                }
                pc.fetch_sub(1, Ordering::Relaxed);
            });

            let s2 = state.clone();
            let pc2 = pin_count.clone();
            let pf2 = pincount_at_free.clone();
            let evictor = thread::spawn(move || {
                if pc2.load(Ordering::Acquire) != 0 {
                    return;
                }
                if s2
                    .compare_exchange(RESIDENT, EVICTING, Ordering::Relaxed, Ordering::Relaxed)
                    .is_err()
                {
                    return;
                }
                if pc2.load(Ordering::Acquire) != 0 {
                    s2.store(RESIDENT, Ordering::Relaxed);
                    return;
                }
                // No eviction_mu in this model (fast path, no lock)
                let pc_at_free = pc2.load(Ordering::Acquire);
                pf2.store(pc_at_free, Ordering::Release);
                s2.store(FREE, Ordering::Relaxed);
            });

            reader.join().unwrap();
            evictor.join().unwrap();
            let pc_at_free = pincount_at_free.load(Ordering::Acquire);
            if pc_at_free != u64::MAX {
                assert_eq!(
                    pc_at_free, 0,
                    "BUG: fix_orphan freed frame with pin_count={pc_at_free}"
                );
            }
        },
        1000,
    );
}

/// Model with the reader holding the pin for longer (traversing the tree).
/// The reader fetch_adds, does some work (yield), then fetch_subs.
/// The evictor has the full unswizzle_and_free path with eviction_mu.
///
/// The key question: does the eviction_mu write lock + can_free check
/// actually prevent freeing a pinned frame, even when the reader doesn't
/// hold the read lock (fast path)?
#[test]
fn shuttle_pin_held_during_traversal() {
    shuttle::check_random(
        || {
            let state = Arc::new(AtomicU64::new(RESIDENT));
            let pin_count = Arc::new(AtomicU64::new(0));
            let freed = Arc::new(AtomicU64::new(0));
            let ewp = Arc::new(AtomicU64::new(0));
            let mu = Arc::new(shuttle::sync::RwLock::new(()));
            let reader_pinned = Arc::new(AtomicU64::new(0));

            let s = state.clone();
            let pc = pin_count.clone();
            let fr = freed.clone();
            let ewp_r = ewp.clone();
            let mu_r = mu.clone();
            let rp = reader_pinned.clone();
            let reader = thread::spawn(move || {
                // lock_hot_pin: conditional
                let _guard = if ewp_r.load(Ordering::Acquire) != 0 {
                    Some(mu_r.read())
                } else {
                    None
                };
                pc.fetch_add(1, Ordering::Relaxed);
                let st = s.load(Ordering::Acquire);
                if st != RESIDENT {
                    pc.fetch_sub(1, Ordering::Relaxed);
                    return;
                }
                // Pin succeeded — reader holds the pin while traversing
                rp.store(1, Ordering::Release);
                // Simulate traversal work — yield to let evictor run
                shuttle::thread::yield_now();
                let was_freed = fr.load(Ordering::Acquire) != 0;
                rp.store(0, Ordering::Release);
                pc.fetch_sub(1, Ordering::Relaxed);
                if was_freed {
                    panic!("BUG: reader held pin while frame was freed");
                }
            });

            let s2 = state.clone();
            let pc2 = pin_count.clone();
            let fr2 = freed.clone();
            let ewp2 = ewp.clone();
            let mu_w = mu.clone();
            let evictor = thread::spawn(move || {
                if pc2.load(Ordering::Acquire) != 0 {
                    return;
                }
                if s2
                    .compare_exchange(RESIDENT, EVICTING, Ordering::Relaxed, Ordering::Relaxed)
                    .is_err()
                {
                    return;
                }
                if pc2.load(Ordering::Acquire) != 0 {
                    s2.store(RESIDENT, Ordering::Relaxed);
                    return;
                }
                ewp2.fetch_add(1, Ordering::AcqRel);
                let _guard = mu_w.write();
                ewp2.fetch_sub(1, Ordering::AcqRel);
                if pc2.load(Ordering::Acquire) != 0 {
                    s2.store(RESIDENT, Ordering::Relaxed);
                    return;
                }
                fr2.store(1, Ordering::Release);
                s2.store(FREE, Ordering::Relaxed);
            });

            reader.join().unwrap();
            evictor.join().unwrap();
            let _ = reader_pinned.load(Ordering::Acquire);
        },
        1000,
    );
}

/// Fast path: reader NEVER takes read lock (budget always high).
/// Evictor always takes write lock. Reader yields while pinned.
/// This tests if the eviction_mu write lock alone (without reader read lock)
/// prevents the race. It should NOT — the write lock is meaningless if
/// the reader doesn't hold the read lock.
#[test]
fn shuttle_fast_path_reader_holds_no_lock() {
    shuttle::check_random(
        || {
            let state = Arc::new(AtomicU64::new(RESIDENT));
            let pin_count = Arc::new(AtomicU64::new(0));
            let freed = Arc::new(AtomicU64::new(0));
            let mu = Arc::new(shuttle::sync::RwLock::new(()));

            let s = state.clone();
            let pc = pin_count.clone();
            let fr = freed.clone();
            let reader = thread::spawn(move || {
                // NO read lock — fast path (budget high, ewp==0)
                pc.fetch_add(1, Ordering::Relaxed);
                let st = s.load(Ordering::Acquire);
                if st != RESIDENT {
                    pc.fetch_sub(1, Ordering::Relaxed);
                    return;
                }
                // Pin succeeded — yield to widen the window
                shuttle::thread::yield_now();
                let was_freed = fr.load(Ordering::Acquire) != 0;
                pc.fetch_sub(1, Ordering::Relaxed);
                if was_freed {
                    panic!("BUG: reader held pin while frame was freed");
                }
            });

            let s2 = state.clone();
            let pc2 = pin_count.clone();
            let fr2 = freed.clone();
            let mu_w = mu.clone();
            let evictor = thread::spawn(move || {
                if pc2.load(Ordering::Acquire) != 0 {
                    return;
                }
                if s2
                    .compare_exchange(RESIDENT, EVICTING, Ordering::Relaxed, Ordering::Relaxed)
                    .is_err()
                {
                    return;
                }
                if pc2.load(Ordering::Acquire) != 0 {
                    s2.store(RESIDENT, Ordering::Relaxed);
                    return;
                }
                // Write lock (meaningless — reader has no read lock)
                let _guard = mu_w.write();
                if pc2.load(Ordering::Acquire) != 0 {
                    s2.store(RESIDENT, Ordering::Relaxed);
                    return;
                }
                fr2.store(1, Ordering::Release);
                s2.store(FREE, Ordering::Relaxed);
            });

            reader.join().unwrap();
            evictor.join().unwrap();
        },
        1000,
    );
}
