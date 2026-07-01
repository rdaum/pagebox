# RAII Frame Reference Design

## Problem Statement

The B+tree and buffer pool have a data race under concurrent eviction because
frame references are `Copy` raw pointers with no lifetime binding to the
guards or pins that make them valid. The borrow checker cannot enforce the
invariant "page bytes may only be accessed while a guard is held."

## Current Architecture

### Type hierarchy (all `Copy`, all raw `*mut BufferFrame` wrappers)

```
BufferFrameRef          — *mut BufferFrame, Copy, no lifetime
├── BufferFrameReadRef  — same ptr, provides page() -> &'static [u8]
├── BufferFrameWriteRef — same ptr, provides page_mut() -> &'static mut [u8]
└── ResidentFrame       — wraps BufferFrameRef, Copy, no lifetime

RoutedChild             — { swip: Swip, edge: ParentEdge }, Copy, no lifetime
```

### Guard types (have lifetimes, own the pin/latch)

```
PinnedFrame<'a>         — owns pin_count increment, Drop decrements
OptimisticFrame<'a>     — PinnedFrame + OptimisticGuard (version snapshot)
SharedFrame<'a>         — PinnedFrame + SharedGuard (rwlock read)
ExclusiveFrame<'a>      — PinnedFrame + ExclusiveGuard (rwlock write)
```

### The race

`ResidentFrame` is created from a guard, then the guard is dropped, then
`ResidentFrame` is used to access page bytes. The `&'static` lifetime on
`page()` means the borrow checker doesn't see the use-after-drop:

```rust
let guard = current.shared_guard();                    // shared latch held
let current_frame = ResidentFrame::from_pinned(&current); // copies raw ptr
let routed_child = current_frame.try_route_to_child(key);  // reads page bytes
let _ = guard;                                          // DROPS shared latch
let child = pool.fix_orphan_frame(routed_child.swip().as_page_id()); // BLOCKING
self.set_parent_link_for_routed_child(
    ResidentFrame::from_pinned(&child),
    current_frame,          // ← uses raw ptr to parent AFTER guard dropped
    routed_child,           // ← reads parent.num_slots() on stale page
);
```

Between `guard` drop and `set_parent_link_for_routed_child`, the parent frame
can be evicted, freed, and reused for a different page. `current_frame.pid()`
returns the wrong PID. `current_frame.num_slots()` reads a different page's
slotted-page header.

### Scale of the problem

- **64 creation sites** for `ResidentFrame::from_*` (55 in btree.rs, 8 in
  node.rs, 1 in split_child.rs)
- **20 call sites** for `set_parent_link_for_routed_child` (all pass
  `ResidentFrame` by value)
- **9 explicit `let _ = guard` drops** followed by `ResidentFrame` use
- Every traversal function (`find_leaf_optimistic`, `find_leaf_exclusive`,
  `with_lookup_fallback_leaf`, `find_leaf_shared_*`, all scan variants) has
  the same pattern

### `'static` leak surface (verified)

The race is enabled by `&'static` returns that decouple page-byte borrows from
any guard lifetime. The full surface, not just `sp()`/`sp_mut()`:

- `BufferFrameReadRef::page` (buffer_frame.rs:258) -> `&'static [u8; PAGE_SIZE]`
- `BufferFrameWriteRef::page` (buffer_frame.rs:276) -> `&'static [u8; PAGE_SIZE]`
- `BufferFrameWriteRef::page_mut` (buffer_frame.rs:280) -> `&'static mut [u8; PAGE_SIZE]`
- `BTreeNode::sp` (node.rs:54) -> `&'static SlottedPage`
- `ResidentFrame::sp` (node.rs:264) -> `&'static SlottedPage`
- `ResidentFrame::sp_mut` (node.rs:268) -> `&'static mut SlottedPage`
- `ResidentFrame::get_key` / `try_get_key` / `get_value` / `try_get_value`
  (node.rs:284-298) -> `&'static [u8]` (4 sites)
- `OptimisticNode::try_value_at` (node.rs:511) -> `Option<&'static [u8]>`
- `SharedNode::key_at` / `try_key_at` / `value_at` / `try_value_at`
  (node.rs:626-638) -> `&'static [u8]` / `Option<&'static [u8]>` (4 sites)
- `ExclusiveNode::key_at` / `value_at` (node.rs:698-702) -> `&'static [u8]`
- `ExclusiveNode::key_at` (node.rs:807) -> `&'static [u8]`

**19 `'static` return sites total.** Phase 1 must retarget all of them, or the
race remains through any accessor the design did not list.

### Related escape hatches (must be covered, not just `ResidentFrame`)

- **`ResidentSharedFrame` / `ResidentOptimisticFrame`** (buffer_pool.rs:1656,
  :1670): both expose `page() -> &'static [u8; PAGE_SIZE]` (buffer_pool.rs:1696,
  :1776) and `Deref` to `BufferFrame`. If Phase 2 only replaces `ResidentFrame`,
  these remain as `'static`-leaking alternatives used by the
  `try_shared_resident_*` / `try_optimistic_resident_*` shortcut paths.
- **`ChildRef`** (node.rs:450) holds a `ResidentFrame` and feeds
  `child_edge_for` / `parent_edge_for_child` — same escape hatch, needs the same
  lifetime treatment.
- **`with_lookup_fallback_leaf` callback** (btree.rs:2581) receives
  `Option<ResidentFrame>` and reads page bytes inside the callback. Phase 2 must
  retype this to `Option<FrameRef<'g>>` and thread the lifetime through
  `lookup_fallback`, `lookup_with_fallback`, `lookup_fixed_fallback`.
- **`extend_*_guard` transmutes** (buffer_pool.rs:420-432) erase guard
  lifetimes to a caller-chosen unbounded `'a`. This is the mechanism by which
  unbounded lifetimes reach the btree: `ResidentFrame::optimistic_guard`
  (node.rs:232) calls `BufferFrameRef::optimistic_guard` (buffer_frame.rs:233),
  which transmutes the guard's lifetime to whatever the caller declared. After
  Phase 1, these transmutes are the residual soundness hole — the guard
  lifetime must not be allowed to exceed the pin (and the pin must not exceed
  the pool borrow). Phase 1 should constrain or remove these transmutes rather
  than leave them as the new `'static`.

## Design: Lifetime-Encoded Frame References

### Core principle

A frame reference must borrow from the guard that makes it valid. The borrow
checker enforces that page bytes are only accessed while the guard is held.

### New types

```rust
/// A reference to a resident frame, bound to the guard that validates it.
/// Replaces the Copy `ResidentFrame`.
///
/// The lifetime `'g` is tied to the guard (OptimisticGuard, SharedGuard, or
/// ExclusiveGuard) that was held when this reference was created.
/// The reference is only valid while the guard is alive.
pub struct FrameRef<'g> {
    bf: BufferFrameRef,
    _marker: PhantomData<&'g ()>,
}

/// A routing decision read from an inner node under a guard.
/// The lifetime ties it to the guard that read the SWIP from the parent's
/// page bytes.
pub struct RoutedChildRef<'g> {
    swip: Swip,
    edge: ParentEdge,
    _marker: PhantomData<&'g ()>,
}
```

### Guard API changes

Guards already have lifetimes. The change is that they produce `FrameRef<'g>`
instead of `ResidentFrame` (Copy):

```rust
impl<'a> PinnedFrame<'a> {
    /// Returns a frame reference bound to this pin.
    /// The reference is valid as long as the PinnedFrame is alive.
    pub fn frame_ref(&self) -> FrameRef<'a> { ... }
}

impl<'a> OptimisticFrame<'a> {
    /// Returns a frame reference bound to this optimistic guard.
    pub fn frame_ref(&self) -> FrameRef<'a> { ... }

    /// Read a routing decision from the inner node.
    /// Returns a RoutedChildRef bound to this guard — the SWIP value
    /// is only valid if the guard is still valid (no version change).
    pub fn route_to_child<'g>(&'g self, key: &[u8])
        -> Option<RoutedChildRef<'g>> { ... }
}
```

### Page access

The `'static` lifetime on `page()` is the core enabler of the race. Replace
with a lifetime tied to the guard:

```rust
impl<'g> FrameRef<'g> {
    /// Access the page bytes. The borrow is tied to the guard.
    pub fn page(&self) -> &'g [u8; PAGE_SIZE] { ... }
    pub fn sp(&self) -> &'g SlottedPage { ... }
}
```

This is the key change: `page()` returns `&'g` instead of `&'static`. The
SlottedPage reference can no longer outlive the guard.

### Traversal pattern (before and after)

**Before (racy):**
```rust
let opt = current.optimistic().map_err(|_| Restart)?;
let current_frame = ResidentFrame::from_optimistic(&opt);
let routed_child = inner.route_to_child(key).ok_or(Restart)?;
inner.validate()?;
let child = pool.try_fix_orphan_frame(routed_child.swip().as_page_id())?;
// ← no re-validate after blocking call
set_parent_link_for_routed_child(
    ResidentFrame::from_pinned(&child),
    current_frame,    // ← stale, guard may be dropped
    routed_child,     // ← stale
);
```

**After (safe):**
```rust
let opt = current.optimistic().map_err(|_| Restart)?;
let routed_child = opt.route_to_child(key).ok_or(Restart)?; // borrows from opt
opt.validate()?;
let child = pool.try_fix_orphan_frame(routed_child.swip().as_page_id())?;
// ← opt is still alive (child doesn't borrow from it)
// ← but routed_child borrows from opt, so we must re-validate:
opt.validate()?;  // ← ideally the compiler forces this
// ← now safe to use routed_child
set_parent_link_for_routed_child(&child, &opt, routed_child)?;
```

### The `set_parent_link_for_routed_child` signature change

```rust
// Before:
unsafe fn set_parent_link_for_routed_child(
    &self,
    child: ResidentFrame,       // ← raw ptr, no guard
    parent: ResidentFrame,      // ← raw ptr, no guard
    routed: RoutedChild,        // ← raw values, no guard
)

// After:
fn set_parent_link_for_routed_child(
    &self,
    child: &PinnedFrame<'_>,   // ← pin is alive
    parent: &OptimisticFrame<'_>, // ← guard is alive (for read access)
    routed: RoutedChildRef<'_>, // ← borrows from the guard
)
```

The parent must be passed as `&OptimisticFrame` (or `&SharedFrame`) so the
caller must still hold a guard. No `unsafe` needed — the lifetimes enforce it.

### The blocking-call problem

The hard case is when a blocking operation (like `fix_orphan_frame`) happens
between reading from the parent and writing back to it:

```rust
let routed_child = opt.route_to_child(key)?;  // borrows from opt
opt.validate()?;
let child = pool.fix_orphan_frame(routed_child.swip().as_page_id())?;
    // ← BLOCKING: parent may be evicted during this call
// At this point, `opt` is still alive (not dropped), but the parent's
// page bytes may have changed (eviction modifies SWIPs under exclusive
// latch, which bumps the version). The guard's version is stale.
opt.validate()?;  // ← MUST re-validate; the type system can't force this
set_parent_link(&child, &opt, routed_child)?;
```

The issue: `opt` is alive (the `OptimisticFrame` owns the pin), but its
optimistic snapshot may be stale. The type system ensures the guard is alive,
but can't force a re-validate. Options:

1. **Make `validate` return a new type** (a `ValidatedFrameRef<'a>` token)
   that `set_parent_link` requires. This forces the caller to call
   `validate()` between the blocking call and `set_parent_link`:

   ```rust
   impl<'a> OptimisticFrame<'a> {
       fn validate(&self) -> Result<ValidatedToken<'a>, Restart> { ... }
   }
   fn set_parent_link_for_routed_child(
       child: &PinnedFrame<'_>,
       parent: &OptimisticFrame<'_>,
       routed: RoutedChildRef<'_>,
       _token: ValidatedToken<'_>,  // ← forces validate() call
   )
   ```

2. **Use shared guards instead of optimistic** for the fallback path. Shared
   guards block exclusive writers, so the parent can't be modified while the
   child is being loaded. This is simpler but may reduce concurrency.

3. **Split into two phases**: read routing under a short-lived guard, load
   the child without any guard, then re-fix the parent under a new guard and
   re-validate the routing. This avoids holding any guard during blocking I/O.

Option 2 is the simplest and most robust. The fallback path already uses
shared guards in `with_lookup_fallback_leaf`. The in-loop optimistic path
(`find_leaf_optimistic`, btree.rs:872) uses non-blocking `try_fix_orphan_frame`
plus a re-validate (btree.rs:882), so the token is not needed there.

**But the root-handling prologue of `find_leaf_optimistic` (btree.rs:802) calls
blocking `fix_orphan_frame` under an optimistic guard and then uses
`routed_child` without re-validating.** That site needs either Option 1
(`ValidatedToken`), a switch to a shared guard for the cold-root path, or a
re-validate before `set_parent_link_for_routed_child`. Option 1 is genuinely
needed, not optional — Phase 4 is real work, not conditional.

### Migration plan

1. **Phase 1: Eliminate `'static` on the page-byte surface** — *DONE*
   - Gave `BufferFrameReadRef<'a>` / `BufferFrameWriteRef<'a>` a lifetime
     parameter; `page()` / `page_mut()` now return `&'a [u8; PAGE_SIZE]` /
     `&'a mut [u8; PAGE_SIZE]` (buffer_frame.rs).
   - `BufferFrameRef::read_ref<'a>` / `write_ref<'a>` now return the
     lifetime-bound types; the safety contract shifted to "caller asserts
     protection for `'a`."
   - Retargeted all 19 `'static` return sites: `BTreeNode::{sp,sp_mut,...}`,
     `ResidentFrame::{sp,sp_mut,get_key,...}`, and the
     `OptimisticNode`/`SharedNode`/`ExclusiveNode` accessors now return `&'a`
     tied to the guard's lifetime.
   - Guard types (`PinnedFrame`/`OptimisticFrame`/`SharedFrame`/
     `ExclusiveFrame`/`ResidentSharedFrame`/`ResidentOptimisticFrame`) produce
     `BufferFrameReadRef<'a>`/`BufferFrameWriteRef<'a>` bound to their own `'a`.
   - **Residual escape hatch (to be removed in Phase 2):** `ResidentFrame`'s
     `read_ref<'a>(self)` / `write_ref<'a>(self)` (node.rs:241, :245) let the
     *caller* pick `'a` with no constraint — effectively `'static` renamed.
     Because of this, the btree still compiles and the race sites are **not
     yet compile errors**. Phase 2 removes this hatch by giving `ResidentFrame`
     a lifetime that borrows from the guard.
   - **`extend_*_guard` transmutes (buffer_pool.rs:420-432):** not yet
     constrained. They remain a residual soundness hole; Phase 2 must address
     them so a guard lifetime cannot be lengthened past the pin that owns it.
   - Verified: `cargo build --workspace` passes; `cargo test --workspace`
     passes (1 pre-existing `shuttle_pin_evict_minimal` failure, unrelated);
     `cargo clippy --workspace --all-targets` introduces no new warnings;
     `cargo fmt --all` is a no-op.

2. **Phase 2: Replace `ResidentFrame` with `ResidentFrame<'g>`** — *DONE*
   - Added lifetime parameter `'g` to `ResidentFrame`; removed `Copy` impl.
   - `from_pinned` / `from_optimistic` / `from_shared` / `from_exclusive` now
     take `&Guard<'g>` and return `ResidentFrame<'g>` borrowing the guard's
     own lifetime `'g` (not the `&self` borrow). Page-byte accessors
     (`sp`, `get_key`, `num_slots`, `is_leaf`, etc.) return `&'g`.
   - `from_hot_swip` is now `unsafe` — the caller asserts the frame is
     protected for `'g`.
   - All function signatures that took `ResidentFrame` by value now take
     `&ResidentFrame<'_>`: `set_parent_link_for_routed_child`,
     `set_parent_link_for_edge`, `set_inner_parent_link`, `set_root_parent_link`,
     `collapse_empty_root_to_child`, `unlink_merged_*`, `eviction_unswizzle`,
     `find_parent`, `repair_separators_after_delete`,
     `refresh_inner_child_parent_links_for_frame`, `collect_evicted_child_pids`.
   - `ChildRef` now carries a lifetime `'g` and holds a `ResidentFrame<'g>` by
     value. `from_frame` borrows from a `&'g ResidentFrame<'g>`; `from_pid`
     constructs from identity-only `BufferFrameRef` + pid (no page-byte access).
   - `with_lookup_fallback_leaf` callback retyped to
     `FnOnce(Option<&ResidentFrame<'_>>) -> R`, threaded through `lookup_fallback`,
     `lookup_with_fallback`, `lookup_fixed_fallback`.
   - `SplitChild::resident_frame` returns `ResidentFrame<'guard>`.
   - **Race sites surfaced and fixed:**
     - `find_leaf_*` functions: `current_frame` previously borrowed from `opt`
       but `opt` was moved into `OptimisticNode`. Fixed by checking `is_leaf`
       via `BTreeNode::is_leaf(opt.read_ref())` before moving `opt`, then using
       `inner.resident_frame()` as `current_frame` in the inner path.
     - `find_leaf_exclusive_fallback` / `find_leaf_exclusive_with_path_fallback`:
       `current_frame` was used after `drop(current_shared)`. Fixed by wrapping
       in `SharedNode` and keeping the guard alive through the inner path.
     - `find_leaf_shared_fallback`: `current_frame` was used after
       `drop(current_shared)` in the cold-SWIP path. Fixed by keeping `shared`
       alive until after `set_parent_link_for_routed_child`.
     - `repair_separators_after_delete`: `child` was a copied `ResidentFrame`
       used across loop iterations after the parent guard was dropped. Fixed
       by using `BufferFrameRef` + pid for identity-only matching across
       iterations.
     - `rebalance_delete_path`: `leaf_frame` was created before `leaf.exclusive()`
       consumed `leaf`. Fixed by extracting `leaf_bf` / `leaf_pid` before
       consuming `leaf`.
     - `eviction_unswizzle`: `node_frame` was created before `node.optimistic()`
       consumed `node`. Fixed by using `node.frame_ref()` for the state check
       and `BTreeNode::is_leaf(opt.read_ref())` for the leaf check.
   - **`extend_*_guard` transmutes (buffer_pool.rs:420-432):** still
     unconstrained. The `from_*` constructors now tie `'g` to the guard's
     lifetime parameter (not the `&self` borrow), which prevents the guard
     lifetime from being lengthened past the pool borrow it was created from.
     The transmutes remain as the mechanism by which the guard's lifetime is
     set to `'a` (the pool borrow), but `'a` is now constrained by the pin/pool
     it was created from. Further constraining the transmutes is Phase 3/4
     work.
   - Verified: `cargo build --workspace` passes; `cargo test --workspace`
     passes (1 pre-existing `shuttle_pin_evict_minimal` failure, unrelated);
     `cargo clippy --workspace --all-targets` introduces no new warnings;
     `cargo fmt --all` is a no-op.

3. **Phase 3: Replace `RoutedChild` with `RoutedChildRef<'g>`** — *DONE*
   - Renamed `RoutedChild` to `RoutedChildRef<'g>` with `PhantomData<&'g ()>`
     marker. Removed `Copy` impl.
   - `RoutedChildRef::new(swip, edge)` creates `RoutedChildRef<'g>` where `'g`
     is chosen by the caller. The `Swip` and `ParentEdge` are `Copy` values
     that don't actually borrow anything; the lifetime `'g` is a compile-time
     marker that documents which guard authorized reading the SWIP.
   - `try_route_to_child` / `route_to_child` / `for_each_child_route` /
     `child_routes` now return `RoutedChildRef<'g>` (or `Vec<RoutedChildRef<'g>>`)
     tied to the guard's lifetime `'g`.
   - `set_parent_link_for_routed_child` takes `&RoutedChildRef<'_>` (by
     reference, not by value) so the caller can use `routed_child.edge()` after
     the call.
   - **What the compiler catches:** `RoutedChildRef` is not `Copy`, so using
     it after moving it (e.g., passing by value to a function and then using
     `routed_child.edge()` again) is a compile error (E0382). This surfaced
     5 sites where `routed_child` was moved into
     `set_parent_link_for_routed_child` and then used again for
     `routed_child.edge()` — all fixed by passing `&routed_child`.
   - **Limitation:** because `RoutedChildRef`'s data is all owned (`Copy`
     `Swip` + `ParentEdge`), the `PhantomData<&'g ()>` marker doesn't create a
     real borrow. The compiler cannot catch use-after-guard-drop for
     `RoutedChildRef` alone — the `'g` lifetime is a documentation marker,
     not an enforced borrow. Enforcing use-after-guard-drop for routing
     decisions requires tying `'g` to a real borrow (e.g., a `ValidatedToken`
     from Phase 4), or making `RoutedChildRef::new` require a `&Guard<'g>`
     argument. The `child_routes()` → `Vec<RoutedChildRef<'a>>` pattern in
     `try_publish_leaf_split_via_blocking_search` (btree.rs:1555) uses
     `RoutedChildRef` values after `drop(current)` — this is acceptable because
     the values are used only for their `swip()` (identity, not routing
     decision), and the caller re-validates by re-fixing the parent.

4. **Phase 4: Add `ValidatedToken` for post-blocking-call validation**
   - Needed: the `find_leaf_optimistic` root prologue (btree.rs:802) uses
     blocking `fix_orphan_frame` under an optimistic guard and then uses
     `routed_child` without re-validating. Either the token forces a
     re-validate, or that specific site is rewritten to use a shared guard or
     a non-blocking `try_fix_orphan_frame` + re-validate (matching the in-loop
     path at btree.rs:872-884).
   - The in-loop optimistic path already does the right thing; no token needed
     there. Audit other blocking-call-under-optimistic sites as part of this
     phase rather than assuming the root prologue is the only one.

### What the compiler will catch after this design

- **Use after guard drop**: `FrameRef<'g>` can't be used after the guard
  (which owns the lifetime `'g`) is dropped
- **Use after blocking call without re-validate**: `RoutedChildRef<'g>` borrows
  from the optimistic guard; if the guard's version is stale after a blocking
  call, `validate()` must be called before using the reference
- **Stale page reads**: `page()` returning `&'g` instead of `&'static` means
  the SlottedPage reference can't outlive the guard
- **Parent modification during child load**: `set_parent_link_for_routed_child`
  requires a live guard on the parent, so the parent can't be evicted while
  the child is being loaded (if using a shared guard)

### What the compiler won't catch

- **Re-validate after non-blocking calls**: `try_fix_orphan_frame` is
  non-blocking, but the parent's version can still change between `validate()`
  and `try_fix_orphan_frame`. The `ValidatedToken` approach (option 1) can
  catch this if the token is consumed by the blocking call and a new one is
  required afterward.
- **Cross-level races**: if the parent's routing is correct at level N, but
  the child at level N+1 is evicted while descending, the reader reaches the
  wrong leaf. The B-link chase at the leaf level should handle this, but
  only if `should_chase_right` is called correctly.
