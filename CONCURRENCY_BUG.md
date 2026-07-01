# B+tree Concurrent Lookup Race Under Eviction Pressure

## Symptoms

The kvbench `ycsb_c_evicting` spec (200k records, 64 buffer-pool frames, 8
threads, 100% reads) intermittently returns `None` for keys that exist in the
tree. The `--verify` shadow HashMap catches these as `VERIFY MISMATCH` —
`expected=Some(...)` but `actual=None`. The failure rate is ~50% of runs at
8 threads, 0% at ≤4 threads. A post-load single-threaded check of all 200k
keys always passes, confirming the keys are genuinely present — the loss
happens during concurrent lookup, not during insert.

A second lookup immediately after the failing one (`retry`) sometimes
succeeds and sometimes also returns `None`, indicating the race is timing-
dependent: the traversal lands on the wrong leaf under concurrent page
eviction/loading, not a permanently corrupted tree.

A separate symptom is a livelock at 8+ threads: all worker threads spin at
100% CPU in `pop_free_frame` → `try_evict_one`, with zero I/O. The
second-chance eviction algorithm cannot find an unreferenced frame because
8 threads continuously re-reference all 64 frames faster than the clock hand
can clear them. This is a "clock hand stuck" pathology, not a deadlock.

## Reproduction

```bash
cargo build -p kvbench --release --all-features
for i in $(seq 1 20); do
  timeout 30 ./target/release/kvbench run \
    --spec crates/kvbench/specs/ycsb_c_evicting.toml \
    --output /tmp/kv_$i.json \
    --engine kvstore \
    --tmpdir /tmp/kvbench-$i \
    --threads 8 --verify 2>&1 | grep -c "VERIFY MISMATCH"
done
```

The bug does not reproduce with ≤4 threads, with ≥256 frames, or with an
`InMemoryPageStore` (no WAL/FilePageStore). It requires the full KvStore
stack and extreme memory pressure (64 frames for 200k records ≈ 5.4×
oversize).

## Z3 Analysis

Formal modeling with Z3 proved that `convert_swips_in_buf` (the SWIP-to-PID
conversion during eviction writeback) is safe: the parent frame is
exclusively latched during writeback, which prevents `unswizzle_parent` from
succeeding on the child (its `try_lock_exclusive` on the parent fails), so
the child frame is never freed or reused while the parent is latched. The
child's PID is stable. The race is therefore not in SWIP conversion but
elsewhere in the lookup traversal path under concurrent eviction.

## Attempted Fixes (Still Broken)

### 1. Clock-hand livelock (partially addressed)

**Root cause:** `RandomSecondChance` eviction uses `attempts = (max_batch * 4).max(8)`
(64 attempts with `max_batch=16`). Under 8-thread pressure on 64 frames,
referenced bits are re-set faster than 64 random samples can clear them.

**Gemini's fix:** Scale attempts to `allocated_slots() * 2` (128 for 64
frames), ensuring a full sweep. Also changed `finish_latched_evicting_frame`
to call `unswizzle_and_free` (which holds `eviction_mu.write()` and calls
`can_free_evicting_frame` to re-check `pin_count == 0` before freeing), and
changed `eviction_mu.try_write()` to blocking `eviction_mu.write()`.

**Result:** Livelock is resolved (runs complete instead of hanging), but
verify mismatches persist at ~50% rate. The clock-hand fix is necessary but
insufficient — it makes the system fast enough to expose the underlying pin
race more frequently.

### 2. Pin-count race (not fixed)

**Root cause:** There is a TOCTOU window between `pin_count.load()` in the
evictor and the `CAS state Resident→Evicting`. The CAS checks `state`, not
`pin_count`, so a concurrent `try_pin_hot_or_cool_swip` can increment
`pin_count` (via `fetch_add`) and see `state == Resident` (pin succeeds)
between the evictor's `pin_count` check and its `CAS`:

```
Evictor (with_single_evict_candidate):     Reader (try_pin_hot_or_cool_swip):
  pin_count.load() → 0
                                            pin_count.fetch_add(1) → 0 (now 1)
                                            state.load(Acquire) → Resident
                                            → returns Some(bf) — PIN SUCCEEDED
  CAS state Resident→Evicting → OK
  (does NOT re-check pin_count)
  → writeback, unswizzle_parent, free frame
                                            ← reader holds pin on FREED frame
                                            frame is reused for different page
                                            reader reads wrong PID
                                            → traverses wrong subtree → None
```

`unswizzle_and_free` does call `can_free_evicting_frame` (which checks
`pin_count == 0`), but this check happens **after** `unswizzle_parent` has
already modified the parent's SWIP from HOT to EVICTED. If the check fails
(pin_count > 0), the frame is reverted to Resident, but the parent's SWIP is
already stale — it now points to EVICTED(pid) while the child is actually
Resident. The next lookup that routes through this parent calls
`fix_orphan_frame(pid)` which finds the frame via `page_table` — but the
`page_table` entry may have been removed by `free_evicting_frame` before the
revert. This creates a state where the child is Resident but unreachable
through the parent's routing, causing lookups to fail.

**What's needed:** A re-check of `pin_count == 0` immediately **after** the
CAS to Evicting (before any unswizzle or writeback), with revert to Resident
if a pin sneaked in. This prevents the frame from being freed while pinned,
without the stale-SWIP problem of the `unswizzle_and_free` approach.

The batch eviction path (`try_finalize_evicting_candidate`) already has this
re-check (line ~4477-4488), but the single-eviction path
(`with_single_evict_candidate`) does not.

## Shuttle Testing

Shuttle tests were written in `crates/storage/tests/shuttle_pin_race.rs` to
model the race deterministically. Results:

- **`shuttle_pin_evict_minimal`**: FAILS — model without eviction_mu confirms
  the raw race exists: the evictor can free a frame while a reader holds a pin.
- **`shuttle_pin_evict_with_rwlock`**: PASSES — the eviction_mu write lock
  (acquired before freeing) blocks readers holding the read lock, and
  `can_free_evicting_frame` re-checks pin_count.
- **`shuttle_pin_evict_conditional_lock`**: PASSES — models the conditional
  `lock_hot_pin` (fast path skips read lock when budget is high). The
  evictor's `eviction_writer_pending` + write lock closes the race.
- **`shuttle_fast_path_reader_holds_no_lock`**: PASSES — even without the read
  lock, the `can_free_evicting_frame` re-check after `eviction_mu.write()`
  catches any pin that snuck in.

**Conclusion:** The eviction_mu + `can_free_evicting_frame` mechanism in
`unswizzle_and_free` correctly prevents the pin-count race in isolation.
The bug must be in a code path that bypasses `unswizzle_and_free` — either
a direct call to `free_evicting_frame` without the `can_free` check, or a
race in the SWIP unswizzle that corrupts routing before the frame is freed.

### Z3 verification of `try_fix_orphan_raw` vs `free_evicting_frame`

Z3 proved UNSAT that a reader can pin a frame in Free state. The state
machine transitions (Resident→Evicting→Free) are monotonic, and the reader's
`state.load(Acquire)` always sees the current or a later state. If the
state is Evicting, the reader's pin fails (returns None). If Free, also
fails. The reader can only pin when state == Resident, which means the
frame was not yet evicted or was reverted to Resident (not freed).

### Remaining hypothesis

Since shuttle and Z3 both confirm the pin/evict paths are individually
correct, the bug likely lies in the **interaction between the optimistic
read path and the eviction path at the B+tree level** — specifically in
`find_leaf_optimistic` (btree.rs:763), where the reader holds an
optimistic guard on the inner node, reads the routed child SWIP, and
then calls `try_pin_child` or `try_fix_orphan_frame` — all without
re-validating the optimistic guard after the child pin (at least in the
cold-SWIP path at btree.rs:870-884). If the inner node is concurrently
evicted and the optimistic guard is invalidated, the reader may traverse
to a child page that is no longer the correct routing target.

This is distinct from the buffer pool's pin/evict race — it's a
tree-level traversal correctness issue under concurrent eviction.
