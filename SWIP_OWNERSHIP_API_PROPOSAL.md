# Safer SWIP and Buffer-Frame Ownership

Status: implemented and verified in the current worktree.

## Implementation status

The implementation follows the independent workstreams below rather than treating the document as
one atomic refactor:

- Stable edges are now non-`Clone` `StableSwip` values backed by an internal `Arc`; resident frame
  backlinks retain that owner, validate process-unique pool provenance, and are explicitly cleared
  before arena teardown.
- `NewStablePage` creates the owner and backlink together, installs into a pre-reserved `Option`
  slot before unpinning, and uses a separate unpublished-allocation abort path on drop. Published
  retirement consumes the stable owner and remains checkpoint-gated.
- `NewUnlinkedPage` now owns split preallocations. It can move between pinned and exclusively
  latched construction states, aborts automatically while unpublished, and only releases its
  tracked state through publication. The B+tree no longer receives general unowned allocation
  frames; unused split reservations abort on drop.
- The B+tree root keeps the same stable owner allocation across replacement. Split-root transfer is
  centralised, keeps the old root, new root, and both children pinned, and updates backlinks before
  releasing those pins.
- Mutable frame views are no longer copyable, borrow their receiver for page access, and bare pins
  no longer expose page bytes. `FrameId` separates copyable slot/generation identity from raw frame
  dereference authority. Optimistic byte access remains an explicitly unsafe internal boundary.
- Child routes borrow the parent guard and no longer expose their raw SWIP word. Route-scoped
  methods perform resident pinning, page-table resolution, and PID interpretation. Code that
  formerly carried routes past parent release now pins resident children first. Cold reswizzling
  consumes a `RoutedChildPublication` capability before upgrading the parent, then installs the
  child backlink and HOT parent word together. Ordinary HOT traversal performs neither operation.
  Split publication releases child latches while retaining pins; delete sibling acquisition is
  resident-only and nonblocking.
- `NoLatches::new` is now an unsafe constructor: each blocking call site must visibly prove that
  no shared/exclusive frame guard is live. There is no thread-local tracking on the hot guard path.
  Structural code releases latches before blocking or uses nonblocking acquisition and restarts.
  This audit exposed and corrected additional sibling-link updates that blocked while structural
  latches were held.

The stable owner currently uses the compact direct-`Arc` layout. A pinned two-thread lifecycle
contention probe on the development AArch64 host did not show reader harm from colocating the word
with the control block; the isolated 128-byte word was consistently slower because of its extra
indirection. The split layout therefore remains conditional rather than becoming a per-edge memory
cost without evidence. The HOT-fix regression test also asserts that repeated fix/drop cycles leave
the stable-owner strong count unchanged.

A release micromeasure comparison against the pre-change `738eb6b` implementation measured the
same resident stable-fix operation at 23.89 ns median / 93.1 cycles before and 14.75 ns / 57.6
cycles after. The owned implementation therefore improves this measured HOT path by about 37%
rather than regressing it. The checked-in `microstableswipbench` also covers resident-only
`try_fix_stable` and 1/2/4/8-thread contention. Its throughput unit is one fix per loop operation;
the chunk size is not multiplied into the reported rate.

The final B+tree measurements against the same clean baseline were 110.00 ns median versus 111.67
ns for `insert_hot`, and 109.44 ns versus 138.10 ns for `lookup_hot`. An intermediate implementation
that attempted to refresh a child backlink on every resident descent regressed insert latency to
140.87 ns. Removing lifecycle mutation from the HOT path and isolating cold backlink/HOT-edge
publication in a non-inlined cold helper eliminated that regression.

Verification completed for both compile-time page sizes: `cargo test --workspace`, `cargo test
--workspace --features page-4k`, and clippy over all workspace targets in both configurations with
warnings denied. The five hybrid-latch loom models and two SWIP-kernel loom models pass. Storage's
ordinary tests are not loom models and correctly cannot run after substituting loom primitives
outside `loom::model`. Miri was not run because the installed stable AArch64 toolchain does not
provide the component.

## Purpose

Pagebox deliberately uses raw, tagged frame pointers in resident SWIPs. That is central to its
in-memory performance model and should remain. The unsafe boundary around those pointers is,
however, wider than it needs to be. In particular, the current public API asks callers to prove
that an externally stored `AtomicSwip` will remain at a stable address until every frame backlink
has disappeared.

That contract recently failed in a downstream table implementation. Newly allocated tuple,
delta, and directory pages were published through boxed `AtomicSwip` values, but their frames
retained `ParentLink::None`. Eviction therefore recycled the frames without rewriting the stable
owner edge to `Evicted(pid)`. The still-HOT owner later addressed a frame containing another page.

This document examines the broader SWIP and frame API, identifies which invariants Rust can
encode, and proposes an ownership-bearing stable-SWIP API plus guard-bounded frame views. The goal
is not to remove the small unsafe kernel required by pointer swizzling. The goal is to prevent
ordinary storage and B+tree code from having to reproduce its lifetime proof.

## Summary of the recommendation

1. Keep `SwipWord` as the eight-byte HOT/COOL/EVICTED representation.
2. Replace externally owned `AtomicSwip` plus `StableSwipRef` with an opaque, non-`Clone`
   `StableSwip` handle backed by `Arc<StableSwipInner>`.
3. Let a resident frame's stable parent link retain an internal strong reference to the same
   `StableSwipInner`. Dropping or moving the public owner can then never leave a dangling frame
   backlink.
4. Add a `NewStablePage<'pool>` typestate guard that creates the owner edge and frame backlink as
   one operation. Dropping it aborts an unpublished allocation; publication installs it directly
   into an owning container or root operation.
5. Make safe frame access borrow the pin or latch guard for the duration of each view. A mutable
   frame view must not be `Copy` and must require `&mut self` to return mutable page bytes.
6. Keep raw frame pointers and decoded child SWIPs inside the pool/B+tree implementation. Expose
   guard-bound routing capabilities rather than freely copyable, dereferenceable frame handles.
7. Record pool provenance in stable SWIPs and validate it at the API boundary. A fully
   compile-time pool brand is possible but is not recommended because it would parameterise the
   entire database object graph.
8. Do not permit blocking frame/latch acquisition through a safe, freely constructible
   `NoLatches` witness. Multi-page structural operations must either release latches before
   blocking or use nonblocking acquisition and restart.

The only reference-count operations on the recommended path occur when a stable link is installed,
replaced, reloaded, evicted, retired, or destroyed. Ordinary HOT fixes continue to load the same
eight-byte SWIP and pin the same frame pointer.

Two implementation constraints are non-negotiable:

- A HOT/COOL fix must never clone, drop, or otherwise modify the stable-owner `Arc` count. It
  borrows the public `StableSwip`, loads the word, validates the frame, and pins it. Ownership work
  is restricted to stable-link installation, cold reload, eviction, retirement, abort, and
  teardown.
- Keep the compact direct-`Arc` layout unless lifecycle refcount traffic measurably contends with
  edge-word readers. If that contention is observed, move the `Arc` control block and frequently
  loaded SWIP word onto different cache lines so eviction's ownership writes do not invalidate the
  cache line used by every HOT fix. Cache-line separation is a measured optimisation, not an
  unconditional layout requirement.

## Pre-migration representation

### SWIP words

`pagebox-swip-kernel` defines a transparent eight-byte `SwipWord` and `AtomicSwipWord`:

- `00`: HOT frame pointer;
- `01`: COOL frame pointer;
- `1x`: EVICTED page ID.

`SwipWord` is intentionally `Copy`. It is a representation value, not proof that a pointed-to
frame is currently resident. `SwipWord::as_ptr` is unsafe, but `SwipWord::hot_ptr` and
`SwipWord::from_raw` can construct arbitrary words through safe calls. That gives callers a
concrete safe path to a forged HOT word, so a safe `BufferFrameRef::from_hot_swip` followed by a
safe dereferencing method is an actual unsound boundary, not merely a theoretical concern.

### Frame lifecycle

The frame state machine is:

```text
Free -> Loading -> Resident -> Evicting -> Free
                      ^            |
                      +------------+
                         rescue/revert
```

Pins prevent `Resident -> Evicting` from completing. Shared/exclusive latches protect page bytes
and non-atomic header fields from mutually latched access. The eviction fence closes the final race
between rewriting an owner edge and recycling the frame.

Before this migration, `PinnedFrame<'pool>` tied the pool borrow to the pin guard. This was the right ownership
shape: dropping the value decrements `pin_count`. The problem is that some views produced from the
pin use `'pool` rather than the shorter borrow of the pin itself.

### Parent-link classes

The former `ParentLink` recorded how eviction invalidated the edge that routes to a frame:

| Variant | Owner location | Eviction action |
| --- | --- | --- |
| `None` | No registered owner | Index pages require discovery; non-index pages may be freed directly |
| `Unswizzled` | Owner edge is still EVICTED | Free directly |
| `Stable(StableSwipRef)` | External `AtomicSwip` | Store `Evicted(pid)` through a raw pointer |
| `InnerNode(InnerParentLink)` | Slot in an inner B+tree page | Validate the hint, latch the parent, and rewrite the slot |

The distinction is important. Inner-node edges live inside a pinned/latch-protected page and can be
rediscovered from structural identity. Stable edges live in ordinary Rust objects such as a B+tree
root field or table page directory. They require an ownership solution, not a tree search.

## Findings

### 1. `StableSwipRef` erases the only lifetime that matters

`StableSwipRef` stores `NonNull<AtomicSwip>` and is `Copy`. Its unsafe constructor requires the
pointed-to edge to remain allocated and unmoved while any frame contains the backlink
(`crates/storage/src/buffer_frame.rs`, `StableSwipRef::from_ref`). Once constructed, no lifetime or
owner is retained.

This is not a contract that a local comment can reliably maintain. It spans:

- the address stability of a caller-owned collection;
- publication order between the owner edge and frame;
- every reload and reswizzle;
- root replacement and page unlinking;
- data-structure teardown;
- buffer-pool teardown.

The current comment that a stable edge "outlives the pool" is also stronger than the actual model
and generally false for table-owned directories. What is required is that the edge outlive every
frame backlink to it. That end point is dynamic and controlled by eviction or retirement.

### 2. Allocation separates creation from ownership registration

`BufferPool::allocate_and_fix` creates a resident, pinned frame with `ParentLink::None`. Its
documentation tells the caller to set the correct owner after publication. The API permits all of
the following states:

```text
allocated frame -> HOT edge published -> pin dropped -> stable parent installed later
allocated frame -> HOT edge published -> pin dropped -> stable parent never installed
allocated frame -> stable parent points at stack or movable storage
```

Only the first ordering in which the stable owner is allocated, its backlink is installed, and the
owner is published before the pin is dropped is valid. The type system currently treats every
ordering alike.

`BufferPool::allocate_page` followed by `fix_frame` installs a stable parent during the load, but it
still accepts a caller-owned `AtomicSwip` whose lifetime is not represented. It also performs a
store allocation and subsequent read for a page that could have been initialised directly in its
new frame.

### 3. Embedded stable edges make their containing object address-sensitive

`BTree` stores `meta_swip: AtomicSwip` directly in the struct. `set_root_parent_link` creates a
`StableSwipRef` to that field. The `BTree` type is otherwise freely movable: constructors return it
by value, and downstream compositions can move it into tables, maps, boxes, or `Arc`s.

The code can be correct when registration happens only after the final move, but that sequencing is
not encoded by `BTree`. A subsequent move after the last backlink installation invalidates the raw
address. An independently allocated stable owner removes address sensitivity from the containing
object.

### 4. Stable-edge teardown is not represented

Replacing a frame's `ParentLink` with `None`, freeing an evicting frame, aborting a load, retiring a
page, and dropping the pool all need to release stable-edge ownership in the proposed model.

Today `ParentLink` is `Copy`, so assignments can simply overwrite it. That is compatible with raw
pointers but incompatible with owning resources. The pool's `mmap` arena also calls `munmap`
without running `BufferFrame` destructors. This is harmless only while every frame field is
trivially destructible. An owning parent link requires explicit frame destruction or explicit
parent-link clearing before `munmap`.

### 5. Safe frame views are not consistently bounded by guard borrows

There are related lifetime-erasure problems in the frame view API:

- `PinnedFrame` is documented as unlatched but safely exposes `page`, `page_bytes`, and
  `read_ref`. A pin prevents eviction; it does not prevent a concurrent writer from changing page
  bytes.
- `PinnedFrame::read_ref(&self) -> BufferFrameReadRef<'pool>` returns the pool lifetime, not the
  borrow lifetime of `&self`. The resulting view can in principle outlive the pin that justified
  it.
- `ExclusiveFrame::read_ref` and `ExclusiveFrame::write_ref` have the same shape.
- `BufferFrameWriteRef<'a>` is `Copy` and `page_mut(self) -> &'a mut [u8; PAGE_SIZE]`. Copying the
  view and calling `page_mut` twice can create two live mutable references to the same page through
  safe methods.
- The B+tree's `ResidentFrame<'g>` repeats this pattern: it can construct a `'g` write view from
  `&self`, and `sp_mut(&self) -> &'g mut SlottedPage` can be called more than once.

These are type-level soundness issues, separate from whether the current callers happen to use the
views briefly. Safe APIs must make the invalid program unrepresentable.

Optimistic reads require an additional, separate audit. Version validation detects that a writer
overlapped a read at the algorithmic level, but it does not by itself make concurrent ordinary Rust
reads and writes to the same byte array data-race-free. The proposed API must not claim that an
optimistic version guard can safely manufacture an ordinary `&[u8]` unless the underlying storage
and access primitives satisfy Rust's memory model. Until that proof exists, optimistic byte access
should remain an explicitly unsafe internal boundary rather than a safe reference-producing API.

### 6. `BufferFrameRef` combines identity with unchecked dereference authority

`BufferFrameRef` is a lifetime-erased `NonNull<BufferFrame>`. `BufferFrameRef::from_hot_swip` is
safe and checks only the SWIP tag, a minimum address, and alignment. Since callers can safely build
an arbitrary `SwipWord`, they can obtain a `BufferFrameRef` for an arbitrary aligned address and
then call safe methods such as `pid` that dereference it.

Even a genuine frame pointer can become logically stale after frame reuse. A copyable identity
token should therefore support equality and diagnostics, not unguarded access to frame fields.
Dereference authority must come from the pool after it validates membership, state, PID or
generation, and the relevant pin/latch protocol.

### 7. Pool provenance is runtime state but is not checked by the stable-edge type

The safety contract for `fix_frame` says that the `AtomicSwip` came from the same pool. The type is
publicly constructible and carries no pool identity. Passing a HOT edge to another pool can make
that pool interpret an unrelated address as one of its frames.

Rust can encode a pool brand with a generative lifetime, but doing so would require
`BTree<'pool>`, `Table<'pool>`, and most higher-level database objects to carry that lifetime. A
private runtime `PoolId` in the stable handle provides a narrower and still safe boundary: the pool
checks provenance before decoding a pointer.

`PoolId` must be globally unique for the lifetime of the process. It must not be derived from the
pool's address, allocator state that can be reset, or a counter that wraps and reuses values. A
process-global monotonic nonzero integer is sufficient if pool creation permanently fails before
wraparound. A stale `StableSwip` that outlives its original pool can then never acquire the identity
of a later pool through ABA.

### 8. In-page child SWIPs need a different capability from stable SWIPs

An inner-node child SWIP is valid because it was read from a page under an optimistic/shared/
exclusive protocol, not because the eight-byte word owns anything. The B+tree already has a
`RoutedChildRef<'g>` marker, but it exposes the raw `SwipWord`, allowing the value to escape the
guard-bound wrapper.

The stable-edge redesign should not add `Arc` or another allocation to every inner-node slot.
Instead, child traversal should consume a guard-bound routing capability while the buffer pool
keeps raw pointer decoding internal.

### 9. Latch ownership is represented, but blocking permission is forgeable

`ExclusiveFrame` correctly represents that a frame latch is held. The pool also has a
`NoLatches<'_>` witness whose documentation forbids constructing it while any frame latch is live.
However, `NoLatches::new` is safe and freely callable, so the compiler cannot enforce that
contract.

The current B+tree exposes a concrete cycle:

```text
split:        exclusive(left) + exclusive(right) -> blocks on exclusive(parent)
delete merge: exclusive(leaf) + exclusive(parent) -> blocks on exclusive(sibling)
```

If the split owns the merge's sibling and the merge owns the split's parent, neither can progress.
The split path constructs `SplitChild` values borrowing both exclusive child frames and passes them
through blocking parent publication. The merge path holds the target leaf and parent, then calls
`pin_exclusive_child`, which can block both while pinning/loading and while acquiring the sibling
latch.

`SplitChildIdentity` documents an intended refactor that releases child latches before parent
publication, but its current shape retains only copyable `BufferFrameRef`/HOT SWIP identity. Once
the pins are also released, eviction can make those values stale. The useful concept is releasing
the child latches, not necessarily releasing their pins.

This is an API-design concern as well as a local deadlock. A safe, freely constructible witness is
documentation, not a capability. Blocking pool operations should not accept it as proof that the
caller holds no latches.

## Proposed ownership model

### `StableSwip`: unique logical owner, shared internal allocation

```rust
pub struct StableSwip {
    inner: Arc<StableSwipInner>,
}

struct StableSwipInner {
    word: AtomicSwip,
    pool_id: PoolId,
}
```

`StableSwip` should not implement `Clone`. It represents one logical routing edge. Higher-level
objects share access by borrowing the edge or by sharing the object that owns it. Internally, a
frame can clone the `Arc<StableSwipInner>` when installing a stable parent link.

The allocation is independently addressed, so moving a `BTree`, table directory, `Vec`, or map no
longer moves the edge. `Arc` is used here for lifetime ownership, not for per-fix synchronisation.
The SWIP word remains atomic and unchanged.

The sketch above describes ownership, not a required physical layout. A direct
`Arc<StableSwipInner>` may place the `Arc` strong/weak counters close enough to `word` for lifecycle
count changes to invalidate the word's cache line. The baseline implementation must measure this.
If it is observable, use a split layout such as:

```rust
#[repr(align(128))]
struct StableSwipWord(AtomicSwip);

struct StableSwipOwner {
    pool_id: PoolId,
    word: Pin<Box<StableSwipWord>>,
}

pub struct StableSwip {
    owner: Arc<StableSwipOwner>,
    word: NonNull<StableSwipWord>,
}
```

The frame retains `Arc<StableSwipOwner>`, which owns the aligned word allocation. The public,
non-`Clone` handle caches the word pointer, so HOT fixing does not clone the `Arc` or add a dependent
control-block access. The separately allocated, 128-byte-aligned word cannot share a cache line
with the `Arc` counters. The extra allocation and padding are justified only by measured lifecycle
contention and must be included in the large-directory memory benchmark.

The public surface should expose only operations needed by an owner:

```rust
impl StableSwip {
    pub fn page_id(&self) -> PageId;
    pub fn state(&self) -> SwipState;

    // Kept private or crate-visible for structural algorithms such as
    // root replacement. Arbitrary stores must not be public.
    fn load_word(&self, order: Ordering) -> SwipWord;
}
```

### Owning stable parent links

```rust
enum ParentLink {
    None,
    Unswizzled,
    Stable(Arc<StableSwipInner>),
    InnerNode(InnerParentLink),
}
```

`ParentLink` is no longer `Copy`. Code that only needs to classify or inspect it should borrow it
or take a non-owning snapshot:

```rust
enum ParentLinkSnapshot {
    None,
    Unswizzled,
    Stable(NonNull<StableSwipInner>),
    InnerNode(InnerParentLink),
}
```

The snapshot remains an internal, short-lived value used while the frame is exclusively latched or
otherwise protected. The stored `ParentLink` owns the strong reference.

On stable eviction the pool:

1. borrows the `StableSwipInner` from the frame's owning link;
2. changes its word from HOT/COOL to `Evicted(pid)`;
3. closes the hot-pin/free race as it does today;
4. replaces the frame link with `None`, dropping the frame's strong reference;
5. recycles the frame.

If the higher-level owner was already dropped, step 4 frees the edge. If the owner remains, the edge
stays EVICTED and can reload the page later. There is no strong-reference cycle: the edge contains a
non-owning tagged frame pointer, while only the frame owns an `Arc` to the edge.

### Stable allocation typestate

The common allocation path should not expose an unowned resident frame:

```rust
#[must_use]
pub struct NewStablePage<'pool> {
    pool: &'pool BufferPool,
    edge: Option<StableSwip>,
    frame: Option<ExclusiveFrame<'pool>>,
}

impl BufferPool {
    pub fn allocate_stable(
        &self,
        no_latches: NoLatches<'_>,
    ) -> NewStablePage<'_>;
}

impl NewStablePage<'_> {
    pub fn pid(&self) -> PageId;
    pub fn page_mut(&mut self) -> &mut [u8; PAGE_SIZE];
    pub fn mark_dirty(&mut self);
}
```

`allocate_stable` allocates the page ID, reserves a frame, creates the stable edge with the frame's
HOT word, and installs the frame's internal strong backlink before returning.

Dropping `NewStablePage` runs an internal **abort-unpublished-allocation** path. This is not normal
page retirement and has no durable-unlink semantics: the page was never reachable and need not
have been logged or written. The abort path clears the temporary stable backlink, removes resident
and page-table state, returns the frame and resident budget, and either returns the never-published
page ID directly to an unpublished-allocation pool or leaves an explicit allocation hole. It must
not enter the checkpoint-gated reusable-extent path used for pages removed from a durable owner.

The high-level API should not expose `publish() -> StableSwip` as the ordinary operation. Such an
API creates a panic window: after `publish` returns but before the caller stores the edge in its
directory, dropping the public owner is memory-safe because the frame retains an internal `Arc`,
but the page becomes unreachable until eviction or pool teardown.

Publication should instead consume both the new page and a pre-reserved owner slot:

```rust
let slot = directory.reserve_stable_slot()?;
let mut new_page = pool.allocate_stable(NoLatches::new(pool));
DirectoryPage::init(new_page.page_mut());
new_page.mark_dirty();
slot.install(new_page);
```

`reserve_stable_slot` performs any allocation or fallible preparation before the page exists.
`install` moves the unique `StableSwip` into the directory before releasing the frame pin and is
infallible after it begins. Different owner types can provide different slot capabilities; the
B+tree root is not an ordinary slot and needs the explicit transfer operation described below.

If a low-level `into_edge` escape hatch remains for specialised integrations, it must be
crate-private or unsafe and document the intentionally deferred unreachable-page case on panic.
The safe common path must preserve the invariant that a page leaves `NewStablePage` only by being
installed into its owner.

A downstream directory allocation becomes:

```rust
let slot = directory.reserve_stable_slot()?;
let mut new_page = pool.allocate_stable(NoLatches::new(pool));
DirectoryPage::init(new_page.page_mut());
new_page.mark_dirty();
slot.install(new_page);
```

There is no API call corresponding to `set_parent_link_stable` in ordinary allocation code.

Split siblings and other pages that are intentionally unlinked during construction should use a
separate `NewUnlinkedPage<'pool>` type. Publishing one into an inner-node edge consumes it and the
latched parent-edge capability together. Dropping it uses the same unpublished-allocation abort
class, not durable retirement. The names should express the state rather than returning the same
general-purpose frame for both paths.

### Safe fixing of stable edges

```rust
impl BufferPool {
    pub fn fix_stable<'pool>(
        &'pool self,
        edge: &StableSwip,
        no_latches: NoLatches<'_>,
    ) -> PinnedFrame<'pool>;

    pub fn try_fix_stable<'pool>(
        &'pool self,
        edge: &StableSwip,
    ) -> Option<PinnedFrame<'pool>>;
}
```

These methods validate `edge.inner.pool_id == self.id` before interpreting a HOT/COOL word. Reload
installs an internal `Arc` clone as the frame's stable parent before publishing HOT. The existing
public `unsafe fix_frame(&AtomicSwip)` should become crate-private and eventually disappear from
downstream APIs.

The HOT/COOL branch of both methods borrows `edge`; it must not clone or drop its owner `Arc`.
Only the EVICTED reload branch installs a new internal strong reference in the loaded frame.

Page retirement should consume the unique owner:

```rust
pub fn retire_stable(
    &self,
    edge: StableSwip,
    no_latches: NoLatches<'_>,
) -> RetiredPage;
```

This operation can assert pool provenance, prevent further fixes through the owner, pin and
exclusively latch the current frame if resident, detach the backlink, and defer page-ID reuse until
the durable unlink boundary. Consuming a non-`Clone` owner encodes that the routing edge is no
longer reachable.

Published stable-page retirement is therefore separate from `NewStablePage::drop`. It is legal
only after the owning directory/root metadata has been durably unlinked according to the caller's
protocol, and it participates in WAL/checkpoint-gated page-ID reuse.

### B+tree root transfer

Root replacement is not an ordinary `Arc` assignment and should remain an explicit B+tree design
subproblem. The `StableSwip` is the identity of the root routing edge and must remain the same
allocation across root replacement. A successful root split changes the word stored in that edge
and transfers which frame retains its internal strong backlink.

The root-transfer operation must enforce this sequence while the old and new roots remain pinned:

1. Build and exclusively latch the candidate new root without giving it stable-root ownership.
2. CAS the existing stable root edge from the expected old-root word to the new-root HOT word.
3. On CAS failure, abort the unpublished new root; the old root keeps the stable backlink.
4. On CAS success, install an internal reference to the **same** `StableSwipInner` in the new root.
5. Publish the old root as a child of the new root and replace its stable backlink with the
   corresponding `InnerParentLink`.
6. Release the new-root/old-root pins only after both backlink states agree with the published root
   edge.

The exact latch ordering needs a dedicated implementation design alongside split publication. In
particular, no evictor may observe the old root still carrying stable ownership after the CAS and
use it to write `Evicted(old_pid)` back into the now-new-root edge. Retaining pins makes that
transition preventable, but the transfer must be centralised rather than assembled from public
`set_parent_link_*` calls.

### Guard-bounded frame views

The frame API should distinguish a copyable identity from access views:

```rust
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct FrameId {
    slot: u32,
    generation: u32,
}

pub struct FrameRead<'guard> {
    frame: NonNull<BufferFrame>,
    _borrow: PhantomData<&'guard BufferFrame>,
}

pub struct FrameWrite<'guard> {
    frame: NonNull<BufferFrame>,
    _borrow: PhantomData<&'guard mut BufferFrame>,
}
```

`FrameId` supports equality, tracing, and validated lookup. It cannot dereference a frame. A
generation protects diagnostics and cached identity from slot reuse.

Read and write views are created only by latch guards and borrow those guards. A bare pin exposes
identity and latch acquisition, but not page bytes:

```rust
impl PinnedFrame<'_> {
    pub fn id(&self) -> FrameId;
    pub fn pid(&self) -> PageId;
    pub fn optimistic(self) -> Result<OptimisticFrame<'_>, PinnedFrame<'_>>;
    pub fn shared(self) -> SharedFrame<'_>;
    pub fn exclusive(self) -> ExclusiveFrame<'_>;
}

impl SharedFrame<'_> {
    pub fn read(&self) -> FrameRead<'_>;
}

impl ExclusiveFrame<'_> {
    pub fn read(&self) -> FrameRead<'_>;
    pub fn write(&mut self) -> FrameWrite<'_>;
}

impl<'guard> FrameWrite<'guard> {
    pub fn page_mut(&mut self) -> &mut [u8; PAGE_SIZE];
    pub fn read(&self) -> FrameRead<'_>;
}
```

`FrameWrite` is neither `Copy` nor `Clone`. Returning `FrameRead<'_>` and `FrameWrite<'_>` ties the
view to the method receiver borrow, not to the longer pool lifetime. Mutation methods on B+tree
resident nodes should likewise require `&mut self`, or take a short-lived `FrameWrite<'_>` by
mutable borrow. They must not return `&'pool mut SlottedPage` from `&self`.

Optimistic views additionally retain the validation obligation. A possible shape, contingent on a
separate Rust-memory-model proof for page-byte access, is:

```rust
pub struct OptimisticRead<'guard> {
    frame: FrameRead<'guard>,
    version: OptimisticGuard<'guard>,
}

impl OptimisticRead<'_> {
    pub fn page(&self) -> &[u8; PAGE_SIZE];
    pub fn validate(self) -> Result<(), Restart>;
}
```

Consuming the optimistic view on validation makes it harder to accidentally use page-derived
decisions after a failed validation. It does not solve concurrent non-atomic byte access. Whether
this wrapper is sound and practical should be determined during the B+tree migration; the minimum
required change is to bind every page borrow to `&self`, remove copyable mutable views, and stop
exposing page bytes from an unlatched `PinnedFrame`.

### Guard-bound child routing

The buffer pool should not accept an arbitrary public `SwipWord` as safe dereference authority.
The B+tree can expose a route borrowed from its parent guard:

```rust
pub(crate) struct ChildRoute<'parent> {
    word: SwipWord,
    edge: ParentEdge,
    _parent: PhantomData<&'parent BufferFrame>,
}
```

The route should not expose `word()` publicly. Instead, a B+tree/pool integration method consumes
or borrows `ChildRoute<'_>` and performs HOT pointer validation and pinning internally. This keeps
the parent-derived lifetime visible even though the encoded word itself is eight copyable bytes.

`BufferFrameRef::from_hot_swip` should become private and unsafe, or be replaced by a pool method
that returns a guard-bound frame after checking arena membership, state, and expected PID or
generation. `ParentFinder` should receive an `EvictingFrame<'_>` capability rather than a freely
copyable `BufferFrameRef`.

### Latch-safe structural operations

The immediate delete-merge correction should be narrow and conservative. While holding the target
leaf and parent, sibling pinning and latching must both be nonblocking:

```rust
fn try_pin_exclusive_child(&self, route: ChildRoute<'_>) -> Option<ExclusiveFrame<'_>> {
    let pinned = self.pool().try_pin_resident_child(route)?;
    pinned.try_exclusive().ok()
}
```

`try_pin_resident_child` is deliberately resident-only: the current `try_pin_child` can still issue
a synchronous page-store read for an EVICTED word when a free frame and resident budget are
immediately available. The raw-word detail remains internal in the final API. A non-resident page,
eviction transition, or contended sibling latch aborts that merge attempt. Deletion remains
committed; underfull pages are valid, and a later operation or explicit rebalance retry can start
from a fresh path. The same rule applies to recursive inner-node merges.

The structural split correction should release both child latches before any blocking parent
acquisition while retaining child pins:

```rust
struct SplitChildren<'pool> {
    left: PinnedFrame<'pool>,
    right: PinnedFrame<'pool>,
    left_pid: PageId,
    right_pid: PageId,
}
```

The pins keep both HOT frame identities valid. Parent publication can then block without owning a
child latch. After the parent edge is published, the parent latch is released before either child
latch is reacquired to install `InnerParentLink` hints. Because both children remain pinned, they
cannot be evicted during that window. Parent-link hints are eviction accelerators; parent routing
is the structural source of truth.

This is safer than using the current `SplitChildIdentity` after releasing both latches and pins. An
alternative is to publish EVICTED PIDs and allow later traversal to reswizzle, but that deliberately
gives up the newly resident HOT edges and needs separate performance evidence.

At the general API level, one of these policies should replace the forgeable witness:

1. Blocking fix/latch methods remain `unsafe` and explicitly require a no-latches proof; safe code
   uses nonblocking `try_*` operations whenever it already owns a latch.
2. Structural code runs inside closure-based pool operations that do not make a blocking
   capability available inside a latched callback.
3. A linear operation context tracks `Unlatched`/`Latched` typestate. This offers the strongest
   static model but is likely too invasive because Rust destructors cannot automatically return a
   consumed context token.

The first policy is the practical initial target. The type system then distinguishes "may block"
from "already holds a latch" at the unsafe boundary instead of presenting `NoLatches::new` as a
proof it cannot provide.

## Why not encode everything with a borrow lifetime?

A tempting definition is:

```rust
struct StableSwipRef<'owner>(&'owner AtomicSwip);
enum ParentLink<'owner> {
    Stable(StableSwipRef<'owner>),
    // ...
}
```

It does not fit the actual ownership graph. Buffer frames live in a homogeneous pool arena, while
stable owners live in independently created and destroyed B-trees and tables. A frame can retain
the link long after the pin that installed it has been dropped. Parameterising `BufferFrame`, the
arena, and the pool by one owner lifetime would incorrectly require every stable edge to share that
lifetime and would create self-referential higher-level structures.

Lifetimes are still the right tool for frame access and routing decisions, whose validity is scoped
to a live guard. Dynamic ownership (`Arc`) is the right tool for a frame backlink whose end is a
future eviction or retirement event.

## Alternatives considered

### `Pin<Box<AtomicSwip>>`

Pinning guarantees that the edge does not move. It does not guarantee that it remains allocated.
The caller could drop the pinned box while a frame retains its raw pointer. `Pin` therefore removes
only half of the unsafe contract.

### An RAII stable-page wrapper without shared ownership

A wrapper could own `Pin<Box<AtomicSwip>>` and synchronously detach or retire its frame in `Drop`.
That makes dropping a table edge potentially block on pins, latches, dirty-page WAL ordering, and
I/O. It also makes ordinary container destruction perform hidden storage operations. The approach
is possible but is a poor default ownership model.

### Pool-owned stable-edge registry

Frames could store a copyable generational registry ID, and eviction could resolve the edge through
the pool. This keeps `ParentLink` trivial but adds a registry lookup and synchronisation to stable
eviction/reload. Reclaiming registry slots safely still needs ownership accounting between the
caller and every frame. It is more machinery than storing the strong reference directly.

### A generative pool lifetime or type brand

`BufferPool<'id>` and `StableSwip<'id>` could make cross-pool use a compile error. To obtain a truly
unique brand, pool construction must occur inside a generative scope, and every object retaining a
pool handle must carry `'id`. This would substantially change the public composition model. A
private runtime `PoolId` retains the important safety check without parameterising the database.

### Keeping the unsafe API and adding a helper

A helper that allocates a `Box<AtomicSwip>` and calls `StableSwipRef::from_ref` would fix the common
allocation sequence but not teardown, object movement, arbitrary safe frame dereference, or mutable
view aliasing. It would make the dangerous operation less visible without closing its invariant.

## Performance and layout impact

The proposal preserves the principal hot-path properties:

- SWIP words remain eight bytes.
- Inner-node child slots remain allocation-free.
- HOT fixes still perform an atomic load, frame-state validation, and pin increment.
- Page bytes remain in place in the frame arena.
- No reference-count operation is added to lookup, scan, or update traversal.

Costs are limited to stable-edge lifecycle operations:

- one `Arc` allocation per externally owned stable edge;
- one internal strong count while its page is resident with that stable parent;
- reference-count changes on install, replacement, eviction, reload, retirement, and teardown.

Downstream table directories already box each stable `AtomicSwip`, so the incremental allocation
overhead is the `Arc` control block rather than a new allocation where none existed. This should
still be measured for workloads with hundreds of thousands of table pages.

`ParentLink` already occupies the header-resident half of a frame. An `Arc` is one pointer, but
making the enum non-`Copy` changes code generation and cleanup requirements. The implementation
must confirm `HEADER_BYTES <= PAGE_SIZE` for both 4 KiB and 64 KiB builds.

Because arena frames are constructed with `ptr::write` and later unmapped, the pool must explicitly
drop every initialised `BufferFrame` before `Arena::drop`, or explicitly replace every owning
parent link with `None`. This cleanup occurs only during pool destruction.

## Implementation sequencing and independent workstreams

This architecture should not be implemented as one serial refactor. Stable-SWIP ownership,
frame-view soundness, child-route capabilities, and general latch-acquisition policy have related
boundaries but separate risks and verification requirements. In particular, the stable ownership
repair must not wait for the frame-view redesign.

### First: correct the reproduced structural latch cycle

This work is independently justified and precedes the API migrations:

1. Make delete merge's sibling lookup resident-only and nonblocking.
2. Use `try_exclusive`; abandon the merge attempt on a missing/transitioning/contended sibling.
3. Apply the same rule to recursive inner-node merging.
4. Replace split's exclusive-child borrows with owned child pins before any blocking parent
   acquisition.
5. Publish the parent edge, release the parent latch, then reacquire child latches individually to
   install parent-link hints before releasing the child pins.
6. Add a deterministic regression model for the captured split/merge cycle.

This correction does not depend on `StableSwip`, new frame views, or a redesigned child-route API.

### Workstream A: stable-SWIP ownership

This is the direct repair for lifetime-erased stable backlinks:

1. Add a process-global, non-reusing `PoolId` and private provenance checks.
2. Add `StableSwipInner` and non-`Clone` `StableSwip`.
3. Change the stored stable `ParentLink` to retain an internal strong reference.
4. Centralise replacement of owning parent links so overwrites always drop the previous owner.
5. Add explicit arena-frame destruction during pool teardown.
6. Add safe `fix_stable` and `try_fix_stable` methods.
7. Add `NewStablePage` with an unpublished-allocation abort path.
8. Add pre-reserved directory/container installation rather than returning an uninstalled edge.
9. Add consuming, durability-aware `retire_stable` separately from allocation abort.
10. Migrate downstream tuple, delta, metadata, and directory edge collections.
11. Verify generated HOT/COOL fix code performs no reference-count operations.
12. Measure cache-line contention between edge-word reads and lifecycle refcount changes; select
    the split aligned-word layout if contention is observable.

The B+tree root is a dedicated subproject within this workstream. Replace
`BTree::meta_swip: AtomicSwip` only after the root-transfer protocol described above has tests for
CAS success, CAS failure, concurrent eviction exclusion, and old-root/new-root backlink state.

Once stable owners migrate, remove public `StableSwipRef`, fixing by arbitrary public
`AtomicSwip`, and parent-link mutation by raw stable identity.

### Workstream B: frame-view soundness

This work can proceed independently and should have its own review and Miri evidence:

1. Remove `Copy` and `Clone` from `BufferFrameWriteRef`.
2. Change `page_mut` and parent-link mutation to require `&mut self`.
3. Return `BufferFrameReadRef<'_>`/`BufferFrameWriteRef<'_>` tied to the receiver borrow.
4. Remove page access from bare `PinnedFrame`; require a shared, exclusive, or explicitly audited
   optimistic guard.
5. Change B+tree mutation wrappers so mutable page access requires a mutable guard-bound receiver.
6. Separate copyable frame identity from dereference-capable, guard-bounded views.

The optimistic-byte-access memory-model question is explicitly part of this workstream, but it
should not block Workstream A's stable ownership repair.

### Workstream C: raw child-route capabilities

After or alongside the frame-identity split:

1. Make `BufferFrameRef::from_hot_swip` private/unsafe and prevent safe dereference from an
   unvalidated identity.
2. Replace exposed raw child words with guard-bound `ChildRoute<'_>` capabilities.
3. Add resident-only route pinning for code that already holds latches.
4. Change `ParentFinder` to receive an eviction-scoped frame capability.

This work should preserve eight-byte in-page SWIPs and avoid ownership allocation in inner-node
slots.

### Workstream D: general latch-deadlock policy

The immediate split/merge correction should land before this broader policy work. Afterwards:

1. Audit every blocking pool fix and latch acquisition performed while a frame latch is live.
2. Replace the safe, forgeable `NoLatches` witness with an explicit unsafe boundary, closure-scoped
   operation API, or genuine capability.
3. Establish the rule that latched multi-page operations use nonblocking acquisition and restart
   unless they first release all conflicting latches.
4. Add loom models for each small acquisition-order invariant and behavioural stress tests with
   post-operation tree oracles for larger structures.

## Verification plan

### Compile-time API checks

Use `compile_fail` doctests rather than adding a new testing dependency. They should demonstrate
that:

- a frame read view cannot outlive its pin or latch guard;
- a mutable frame view cannot be copied;
- two mutable page borrows cannot coexist;
- a `StableSwip` cannot be cloned or constructed from an arbitrary `AtomicSwip`;
- a stable page cannot leave `NewStablePage` without installation into an owner slot;
- stable retirement consumes the owner edge;
- page bytes cannot be read through a bare pin;
- blocking acquisition is unavailable from safe latched-operation helpers.

### Behavioural tests

1. Allocate a stable page, drop the higher-level owner while its frame is resident, evict it, and
   verify that no dangling access occurs and the internal strong reference is released.
2. Move the containing B+tree repeatedly after root creation, force root access and pressure, and
   verify all keys against `BTreeMap`.
3. Fill a small pool with stable pages, force repeated eviction/reload from multiple threads, and
   verify page identity and contents.
4. Abort `NewStablePage` before publication and verify that frame capacity, resident budget, page
   table state, and unpublished-allocation accounting recover without entering durable retirement.
5. Attempt cross-pool fixing and assert a deterministic provenance error before pointer decoding.
6. Replace a B+tree root under concurrent traversal and verify that the old root receives an inner
   parent while the new root owns the stable edge.
7. Drop a pool containing resident stable links and use a drop counter in test-only
   `StableSwipInner` state to verify every internal strong reference is released before `munmap`.
8. Model the split/merge cycle with two threads and barriers: the split owns both child latches
   while requesting the parent, and delete owns the parent while requesting the sibling. The fixed
   merge must restart rather than wait, and the pinned-child split must publish after the parent is
   released.

The small stable-edge state machine is a good candidate for a loom model: owner drop, fix, stable
unswizzle, eviction rescue, and final frame release can be represented without page contents.

Run Miri over frame-view unit tests in Workstream B. The purpose is to catch reference aliasing and
use-after-guard mistakes, not merely to establish that a stress test did not panic.

Optimistic page access requires a dedicated Miri and memory-model investigation. A passing
algorithmic latch test is not evidence that concurrent ordinary byte references are valid Rust.

### Performance checks

Measure before and after each implementation workstream or independently landed change:

- resident stable-page HOT fix latency and instructions retired;
- reference-count operations on the HOT/COOL fix path (required result: zero);
- stable page allocation/publication cost;
- stable eviction and reload cost;
- B+tree point lookup and insert microbenchmarks;
- 4 KiB cache-pressure load and read/write workloads;
- cache-to-cache transfers or equivalent coherence evidence while stable edges are repeatedly
  evicted/reloaded and concurrently fixed;
- memory overhead with a large stable table-page directory, including an aligned split-word
  layout if selected.

The acceptance condition is no measurable HOT-fix regression outside noise. Lifecycle operations
may become modestly more expensive in exchange for owned lifetime safety, but their cost and
frequency must be reported rather than assumed.

## Resulting safety boundary

After the migration, unsafe code remains where Pagebox genuinely needs it:

- encoding and decoding frame pointers in `SwipWord`;
- accessing the `mmap` frame arena;
- extending latch guards over pool-owned latch storage;
- validating and pinning a HOT/COOL pointer under the eviction protocol;
- converting in-page SWIPs on writeback.

Ordinary B+tree and downstream storage code no longer performs these proofs:

- keeping an external SWIP allocation alive for an unknown eviction lifetime;
- ensuring a containing object never moves;
- remembering to install a stable backlink after allocation;
- preventing a frame view from escaping its guard;
- preventing copied write views from aliasing mutable page bytes;
- proving that a stable edge belongs to the pool interpreting its HOT pointer.
- proving by convention that a blocking acquisition cannot participate in a multi-page latch
  cycle.

That is the appropriate division: raw tagged pointers remain an internal performance mechanism,
while public types carry the ownership, provenance, and borrow information required to use them
safely.
