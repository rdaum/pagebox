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
            let _fwp = freed_while_pinned.clone();
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

/// Model the stale-SWIP-after-revert race.
///
/// Scenario:
/// 1. Evictor: CAS child state Resident→Evicting (succeeds)
/// 2. Evictor: re-check pin_count → 0
/// 3. Evictor: unswizzle_parent → writes EVICTED(pid) to parent's SWIP
/// 4. Reader: reads parent's SWIP → sees EVICTED(pid) (cold)
/// 5. Reader: calls fix_orphan_frame(pid) → page_table.lookup(pid)
///    → finds child frame → state is Evicting → tries to pin
/// 6. Evictor: eviction_mu.write() → can_free → pin_count > 0 (reader pinned it in step 5's fetch_add?)
///
/// Wait: in step 5, fix_orphan_raw does:
///   a. page_table.lookup(pid) → finds bf
///   b. state.load(Acquire) → Evicting (not Resident) → enters Loading/Evicting handling
///   c. In the Evicting branch: try_rescue_evicting_orphan → tries to revert to Resident
///   d. If rescue succeeds: continues with the pin
///   e. If rescue fails: returns None
///
/// But between step 3 (unswizzle_parent) and step 6 (eviction_mu.write),
/// the reader can:
///   - Read the parent's SWIP (EVICTED)
///   - Look up the child in page_table (finds it, state=Evicting)
///   - Try try_rescue_evicting_orphan (exclusively latches the child)
///     But the child is already exclusively latched by the evictor!
///     So try_lock_exclusive FAILS → rescue fails → returns None.
///
/// Then the evictor:
///   - eviction_mu.write()
///   - can_free: pin_count == 0 (reader didn't pin) → frees the frame
///   - page_table.remove(pid)
///
/// Now the reader retries fix_orphan_frame(pid):
///   - page_table.lookup(pid) → NOT FOUND (removed) → tries free_list
///   - Pops a free frame → tries try_insert(pid, bf) → succeeds
///   - Loads the page from disk → Resident
///
/// This should work. But what if the frame was reused for a DIFFERENT page
/// before the reader retries? Then page_table.lookup(pid) fails, the reader
/// pops a new free frame, loads the page from disk, and everything is fine.
///
/// So the stale-SWIP-after-revert path seems correct...
/// Unless the reader sees EVICTED but the frame is REVERTED to Resident
/// (pin_count > 0 at can_free check). Then:
///   - Reader reads parent SWIP → EVICTED(pid)
///   - Reader calls fix_orphan_frame(pid)
///   - page_table.lookup(pid) → finds bf (still in page_table, not freed)
///   - state.load(Acquire) → Resident (reverted by evictor)
///   - Tries to pin: pin_count.fetch_add(1) → state still Resident → pin succeeds
///   - Reader uses the frame → correct!
///
/// So this path is also fine. The race must be elsewhere.
///
/// Let me test a different hypothesis: the parent's SWIP is read by
/// try_route_to_child under a SHARED guard (optimistic or shared latch).
/// The unswizzle_parent writes EVICTED to the parent's page bytes under
/// an exclusive latch. But if the reader holds a shared guard, the
/// exclusive write should block... unless it's optimistic.
///
/// In find_leaf_optimistic, the reader uses an optimistic guard on the
/// parent. Optimistic guards DON'T block exclusive writers. The writer
/// (unswizzle_parent) does try_lock_exclusive on the parent. If the
/// optimistic reader is mid-read, the exclusive write succeeds, and the
/// reader's subsequent validate() fails → Restart.
///
/// But in find_leaf_optimistic, between reading the SWIP and validating,
/// the reader may have already called try_pin_child on the STALE hot SWIP.
/// If the child was evicted (SWIP changed to EVICTED), the reader's
/// try_pin_child would use the OLD hot SWIP (from before the unswizzle).
/// The hot SWIP points to a frame that may have been freed and reused.
///
/// THIS is the real race: optimistic reader reads hot SWIP → unswizzle
/// changes it to EVICTED → reader tries try_pin_child on stale hot SWIP
/// → if the frame was freed and reused, try_pin_child pins the wrong page.
#[test]
fn shuttle_optimistic_stale_swip_race() {
    shuttle::check_random(
        || {
            // Model: parent has a SWIP that can be read optimistically.
            // The SWIP is either HOT (points to child frame) or EVICTED (page_id).
            // Reader reads SWIP optimistically; evictor changes HOT→EVICTED.

            let swip = Arc::new(AtomicU64::new(0)); // 0 = HOT (points to frame 0)
            let child_state = Arc::new(AtomicU64::new(RESIDENT));
            let child_pid = Arc::new(AtomicU64::new(42));
            let child_freed = Arc::new(AtomicU64::new(0));
            let child_reused_pid = Arc::new(AtomicU64::new(99));

            let swip_r = swip.clone();
            let cs_r = child_state.clone();
            let cp_r = child_pid.clone();
            let cf_r = child_freed.clone();
            let cr_r = child_reused_pid.clone();
            let reader = thread::spawn(move || {
                // Optimistic read of SWIP
                let raw = swip_r.load(Ordering::Acquire);
                let is_hot = raw == 0; // 0 = HOT

                if is_hot {
                    // try_pin_child on the hot SWIP
                    // (no lock_hot_pin on fast path)
                    // Check child state
                    let st = cs_r.load(Ordering::Acquire);
                    if st != RESIDENT {
                        return false; // pin failed
                    }
                    // Pin succeeded — read the child's pid
                    let pid = cp_r.load(Ordering::Acquire);
                    let was_freed = cf_r.load(Ordering::Acquire) != 0;
                    if was_freed {
                        // The child was freed and reused for a different page.
                        // We're reading the WRONG pid!
                        let reused_pid = cr_r.load(Ordering::Acquire);
                        if reused_pid != pid {
                            panic!(
                                "BUG: reader pinned freed frame, read pid={pid} but actual={reused_pid}"
                            );
                        }
                    }
                }
                true
            });

            let swip_e = swip.clone();
            let cs_e = child_state.clone();
            let cp_e = child_pid.clone();
            let cf_e = child_freed.clone();
            let _cr_e = child_reused_pid.clone();
            let evictor = thread::spawn(move || {
                // Evict the child:
                // 1. CAS state Resident→Evicting
                if cs_e
                    .compare_exchange(RESIDENT, EVICTING, Ordering::Relaxed, Ordering::Relaxed)
                    .is_err()
                {
                    return;
                }
                // 2. Re-check pin_count (not modeled — assume 0)
                // 3. unswizzle_parent: write EVICTED(pid) to parent's SWIP
                let pid = cp_e.load(Ordering::Relaxed);
                swip_e.store(pid, Ordering::Release); // EVICTED(pid)
                // 4. eviction_mu.write() + can_free + free (not modeled with lock)
                // 5. free: state = Free, page_table.remove
                cf_e.store(1, Ordering::Release);
                cs_e.store(FREE, Ordering::Relaxed);
                // 6. Frame is reused for a different page
                cp_e.store(99, Ordering::Relaxed); // new pid
                cs_e.store(RESIDENT, Ordering::Relaxed); // back to Resident
            });

            reader.join().unwrap();
            evictor.join().unwrap();
        },
        1000,
    );
}

/// Model the page_table race between fix_orphan_raw (reader loading a cold page)
/// and free_evicting_frame (evictor freeing the same page).
///
/// fix_orphan_raw:
///   1. page_table.lookup(pid) → finds existing frame (Resident)
///   2. pin_count.fetch_add(1) → pin succeeds (state == Resident)
///   3. return pinned frame
///
/// free_evicting_frame (called from unswizzle_and_free after can_free passes):
///   1. page_table.remove(pid)
///   2. state = Free
///
/// The race: between step 1 and step 2 of fix_orphan_raw, the evictor may
/// run free_evicting_frame. But can_free_evicting_frame checks pin_count == 0,
/// and the reader's fetch_add in step 2 happens before the evictor's can_free
/// check (because eviction_mu.write() blocks until the reader's lock_hot_pin
/// read lock is released... but on the fast path, there IS no read lock).
///
/// Wait — fix_orphan_raw does NOT use try_pin_hot_or_cool_swip. It uses its
/// own pin logic at line 3695:
///   pin_count.fetch_add(1, Relaxed)
///   state.load(Acquire)
///   if Resident: Some
///   else: undo, None
///
/// And it uses try_lock_hot_pin (line 3692) which is conditional.
/// On the fast path (budget high), try_lock_hot_pin returns Some(None) → no lock.
/// Then pin_count.fetch_add + state.load race with the evictor.
///
/// But the evictor's can_free check (under eviction_mu.write) should catch
/// any non-zero pin_count... UNLESS the reader is on the fast path (no read
/// lock) and the evictor's eviction_mu.write doesn't block the reader.
///
/// The key question: can the reader's fetch_add happen AFTER the evictor's
/// can_free check but BEFORE the evictor's page_table.remove + state=Free?
///
/// Timeline:
/// 1. Evictor: CAS state Resident→Evicting
/// 2. Evictor: re-check pin_count → 0
/// 3. Evictor: unswizzle_parent (modifies parent SWIP)
/// 4. Evictor: eviction_mu.write()
/// 5. Evictor: can_free → pin_count == 0
/// 6. Reader: (fast path, no read lock) page_table.lookup → finds frame
/// 7. Reader: try_lock_hot_pin → Some(None) (fast path, ewp was set by evictor at step 4... wait, ewp is set BEFORE eviction_mu.write())
///
/// Actually: evp.fetch_add(1) at line 4709, THEN eviction_mu.write() at 4710.
/// So the reader's lock_hot_pin checks ewp:
///   - If ewp > 0: takes read lock → blocks on eviction_mu.write()
///   - If ewp == 0: fast path, no lock
///
/// The reader can check ewp BEFORE the evictor sets it (step 4). Then:
/// 6. Reader: ewp.load → 0 → fast path
/// 7. Reader: page_table.lookup → finds frame
/// 8. Evictor: ewp.fetch_add(1) (step 4a)
/// 9. Evictor: eviction_mu.write() (step 4b)
/// 10. Evictor: can_free → pin_count still 0 (reader hasn't fetched yet)
/// 11. Evictor: page_table.remove(pid), state = Free
/// 12. Reader: pin_count.fetch_add(1) → pins a FREE frame!
/// 13. Reader: state.load → Free (not Resident) → undo pin, return None
///
/// Step 13 returns None, so the reader retries. No harm.
/// But what if step 12 and 13 happen BETWEEN step 10 and 11?
/// 10. Evictor: can_free → pin_count == 0
/// 11. Reader: pin_count.fetch_add(1) → pin_count = 1
/// 12. Reader: state.load → Evicting → undo pin, return None
/// 13. Evictor: page_table.remove, state = Free
///
/// Also fine — reader sees Evicting, backs off.
///
/// What if step 12 sees Resident? State was CAS'd to Evicting at step 1.
/// So state.load can't see Resident after the CAS. Unless the frame was
/// reverted to Resident by another path...
///
/// I can't find the race through this path either. Let me just commit the
/// tests and move on to running the actual benchmark with the Kimi fix.
#[test]
fn shuttle_page_table_race() {
    shuttle::check_random(
        || {
            let state = Arc::new(AtomicU64::new(RESIDENT));
            let pin_count = Arc::new(AtomicU64::new(0));
            let in_page_table = Arc::new(AtomicU64::new(1)); // 1 = yes
            let ewp = Arc::new(AtomicU64::new(0));
            let mu = Arc::new(shuttle::sync::RwLock::new(()));
            let freed_while_pinned = Arc::new(AtomicU64::new(0));

            let s = state.clone();
            let pc = pin_count.clone();
            let ipt = in_page_table.clone();
            let ewp_r = ewp.clone();
            let mu_r = mu.clone();
            let fwp_r = freed_while_pinned.clone();
            let reader = thread::spawn(move || {
                // fix_orphan_raw: lookup → pin
                // try_lock_hot_pin: conditional
                let _guard = if ewp_r.load(Ordering::Acquire) != 0 {
                    Some(mu_r.read())
                } else {
                    None
                };
                // page_table.lookup → found (in_page_table == 1)
                if ipt.load(Ordering::Acquire) == 0 {
                    return; // not in page_table
                }
                pc.fetch_add(1, Ordering::Relaxed);
                let st = s.load(Ordering::Acquire);
                if st != RESIDENT {
                    pc.fetch_sub(1, Ordering::Relaxed);
                    return; // pin failed
                }
                // Pin succeeded — check if frame was freed while pinned
                let was_freed = fwp_r.load(Ordering::Acquire) != 0;
                pc.fetch_sub(1, Ordering::Relaxed);
                if was_freed {
                    panic!("BUG: page_table reader pinned a freed frame");
                }
            });

            let s2 = state.clone();
            let pc2 = pin_count.clone();
            let ipt2 = in_page_table.clone();
            let ewp2 = ewp.clone();
            let mu_w = mu.clone();
            let fwp2 = freed_while_pinned.clone();
            let evictor = thread::spawn(move || {
                // CAS Resident→Evicting
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
                // unswizzle_and_free:
                ewp2.fetch_add(1, Ordering::AcqRel);
                let _g = mu_w.write();
                ewp2.fetch_sub(1, Ordering::AcqRel);
                if pc2.load(Ordering::Acquire) != 0 {
                    s2.store(RESIDENT, Ordering::Relaxed);
                    return;
                }
                // free: page_table.remove, state = Free
                ipt2.store(0, Ordering::Release);
                fwp2.store(1, Ordering::Release);
                s2.store(FREE, Ordering::Relaxed);
            });

            reader.join().unwrap();
            evictor.join().unwrap();
        },
        1000,
    );
}

/// Model the optimistic traversal race in find_leaf_optimistic.
///
/// Reader (find_leaf_optimistic, cold SWIP path):
///   1. optimistic guard on parent (version = v)
///   2. route_to_child(key) → reads SWIP from parent's page bytes
///   3. validate (version still v?) → OK
///   4. child_swip is COLD → try_fix_orphan_frame(page_id)
///   5. [MISSING: inner.validate() — NOW ADDED]
///   6. set_parent_link + current = child
///
/// Writer (unswizzle_parent → try_unswizzle_inner_node_fast_path):
///   1. try_lock_exclusive on parent (bumps version from v to v|1)
///   2. write EVICTED(pid) to parent's page bytes (modifies routing)
///   3. drop exclusive guard (bumps version to v+2)
///
/// The race: reader reads SWIP (step 2), writer modifies it (W2-3),
/// reader validates (step 3). If the writer's exclusive guard is held
/// during step 3, validate sees version v|1 → fails → Restart. Good.
/// If the writer completes before step 3, validate sees v+2 → fails. Good.
/// If the writer hasn't started yet, validate sees v → succeeds. Then
/// the writer runs between step 3 and step 4. The reader proceeds with
/// the old SWIP (which was HOT, not COLD). try_pin_child on HOT SWIP
/// might succeed (child was evicted, SWIP changed, but reader has stale
/// SWIP pointing to old frame position). The frame may have been reused.
///
/// Wait — but the reader reads the SWIP from the parent's page bytes.
/// If the writer modifies the SWIP in the parent's page bytes (W2),
/// the reader already has the OLD SWIP in a local variable (step 2).
/// The SWIP is a u64 value copied into `child_swip`. The writer's
/// modification to the parent's page bytes doesn't affect the reader's
/// local copy. But `validate()` checks if the parent was modified.
/// If validate passes (no modification yet), the reader's local SWIP
/// is correct. If validate fails, the reader restarts.
///
/// The question: can the writer modify the parent BETWEEN the reader's
/// validate (step 3) and the reader's use of child_swip (step 4)?
/// YES — that's the race window. But the SWIP value is already in the
/// reader's local variable. If the SWIP was HOT, the reader calls
/// try_pin_child, which calls try_pin_hot_or_cool_swip. The hot SWIP
/// points to a frame. If the frame was evicted between step 3 and step 4,
/// the state would be Evicting (not Resident), and try_pin_hot_or_cool_swip
/// would fail (fetch_add, load Evicting, undo, return None). Restart.
///
/// If the SWIP was COLD (EVICTED), the reader calls try_fix_orphan_frame.
/// The page_id in the COLD SWIP is stable (it was written by a previous
/// unswizzle). try_fix_orphan_frame loads the page from disk. The loaded
/// page is the correct child. No race here.
///
/// So the optimistic traversal should be correct IF validate is called
/// after every read of the parent's data. The bug is that the COLD path
/// doesn't call validate after try_fix_orphan_frame.
///
/// With my fix (adding validate after try_fix_orphan_frame), the race
/// should be closed. But it still fails. Let me model this in shuttle.
#[test]
fn shuttle_optimistic_cold_swip_race() {
    shuttle::check_random(
        || {
            // Model: parent version, parent SWIP (HOT or COLD),
            // child frame state.
            let parent_version = Arc::new(AtomicU64::new(0));
            let parent_swip = Arc::new(AtomicU64::new(0)); // 0 = HOT
            let child_state = Arc::new(AtomicU64::new(RESIDENT));
            let child_pid = Arc::new(AtomicU64::new(42));

            let pv = parent_version.clone();
            let ps = parent_swip.clone();
            let cs = child_state.clone();
            let _cp = child_pid.clone();
            let reader = thread::spawn(move || {
                // 1. optimistic guard (snapshot version)
                let snapshot = pv.load(Ordering::Acquire);

                // 2. route_to_child: read SWIP
                let swip = ps.load(Ordering::Acquire);
                let is_cold = swip != 0;

                // 3. validate
                if pv.load(Ordering::Acquire) != snapshot {
                    return; // restart — parent was modified
                }

                // 4. resolve child
                if is_cold {
                    // COLD path: try_fix_orphan_frame(page_id)
                    // page_id = swip (the EVICTED page_id)
                    let _page_id = swip;
                    // Load page from disk — no race here, page_id is stable
                } else {
                    // HOT path: try_pin_child(hot_swip)
                    let st = cs.load(Ordering::Acquire);
                    if st != RESIDENT {
                        return; // pin failed — restart
                    }
                    // Pin succeeded — but parent may have been modified
                    // between validate (step 3) and here
                    // Re-validate in HOT path (line 863 in the real code)
                    if pv.load(Ordering::Acquire) != snapshot {
                        return; // restart
                    }
                }

                // 5. [MISSING in cold path] validate
                // With the fix, this is now present for COLD path too
                if is_cold && pv.load(Ordering::Acquire) != snapshot { // restart — parent was modified
                }
            });

            let pv2 = parent_version.clone();
            let ps2 = parent_swip.clone();
            let _cs2 = child_state.clone();
            let cp2 = child_pid.clone();
            let writer = thread::spawn(move || {
                // unswizzle_parent: try_lock_exclusive (bump version)
                let old = pv2.load(Ordering::Acquire);
                pv2.store(old + 2, Ordering::Release); // exclusive version bump

                // Write EVICTED(pid) to parent's SWIP
                let pid = cp2.load(Ordering::Acquire);
                ps2.store(pid, Ordering::Release); // now COLD

                // drop exclusive guard (bump version again)
                // already done by the store above
            });

            reader.join().unwrap();
            writer.join().unwrap();
        },
        1000,
    );
}
