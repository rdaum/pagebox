//! The per-page in-memory slot: [`BufferFrame`], [`BufferFrameRef`] /
//! [`BufferFrameReadRef`] / [`BufferFrameWriteRef`] identity handles, and the
//! [`ParentLink`] enum that drives eviction.
//!
//! A `BufferFrame` is a 4096-aligned, two-page structure: the first
//! `PAGE_SIZE` bytes hold the [`HybridLatch`], [`FrameHeader`], and the
//! header-resident `parent_link`; the second `PAGE_SIZE` bytes are the page's
//! data region (accessed as `&[u8; PAGE_SIZE]`). This layout keeps a frame
//! and its page on adjacent cache lines with no indirection.
//!
//! [`HybridLatch`]: pagebox_hybrid_latch::HybridLatch
//!
//! ## Identity handles
//!
//! Frames are addressed through small copyable reference types rather than
//! Rust references with lifetimes, because they need to outlive any individual
//! borrow (the buffer pool owns them as `Box<[BufferFrame]>`):
//!
//! - [`BufferFrameRef`] — unchecked identity for a frame. Construction is
//!   `unsafe` (caller asserts the frame is live for the use). Cheap to copy;
//!   no `Drop`. This is the type stored inside `Swip::Hot`/`Cool` words.
//! - [`BufferFrameReadRef`] — produced by [`BufferFrameRef::read_ref`] under the
//!   caller's pin/latch protocol; exposes the page bytes as `&'a [u8;
//!   PAGE_SIZE]` where `'a` is the lifetime established at construction (tied
//!   to the guard/pin that authorizes access).
//! - [`BufferFrameWriteRef`] — produced by [`BufferFrameRef::write_ref`] under
//!   an exclusive latch; exposes mutable page bytes and the parent-link
//!   mutators.
//!
//! ## Parent links
//!
//! Eviction needs to find and rewrite the routing edge in the parent that
//! points at the frame being evicted — without it, the parent's `Hot`/`Cool`
//! SWIP would dangle. [`ParentLink`] enumerates the four cases:
//!
//! - [`ParentLink::None`] — orphan (freshly allocated, never published).
//! - [`ParentLink::Unswizzled`] — loaded by page ID while its owner edge
//!   remains evicted, so eviction does not need to rewrite a parent pointer.
//! - [`ParentLink::Stable`] — a routing edge stored outside any tree page
//!   (e.g. `BTree::meta_swip`, a table directory `Vec` entry). The pool can
//!   write `Swip::evicted(pid)` directly through the [`StableSwipRef`] without
//!   needing to walk the tree. These edges must remain live for as long as any
//!   frame refers to them through a `Stable` link — building one is `unsafe`.
//! - [`ParentLink::InnerNode`] — a cached hint pointing at a slot in a B-tree
//!   inner page. The hint is validated during eviction; if stale, the pool
//!   falls back to a registered [`ParentFinder`] tree walk.
//!
//! ## Page-writeback hooks
//!
//! When a frame is evicted with `Hot`/`Cool` child swizzles in its page bytes
//! (B-tree inner nodes), those must be converted to `Evicted(pid)` before being
//! written to disk — the on-disk format contains page IDs, not memory
//! pointers. [`BufferFrame::convert_hot_swips_to_evicted`] does this in place
//! under the exclusive latch; [`BufferFrame::convert_swips_in_buf`] does it on a
//! copy without touching the live frame, which is the form used by the
//! writeback path so concurrent optimistic readers keep observing the live
//! swizzled pointers.

pub use pagebox_frame_kernel::{
    AtomicFrameState, FrameCoreHeader, FrameState, InnerParentLink, Lsn, PAGE_SIZE,
    PaddedAtomicU32, PageId, page_base_span, page_end_base_page, page_size, physical_page_number,
};

use pagebox_hybrid_latch::{HybridLatch, OptimisticGuard, Restart};
use std::ptr::NonNull;
use std::sync::atomic::Ordering;

use pagebox_swip_kernel::{AtomicSwipWord as AtomicSwip, SwipWord as Swip};

// ---------------------------------------------------------------------------
// ParentLink — how eviction finds the routing edge in the parent
// ---------------------------------------------------------------------------

/// How this frame's routing edge in its parent can be found during eviction.
#[derive(Clone, Copy)]
pub enum ParentLink {
    /// No parent tracking (orphan, freshly allocated).
    None,
    /// Loaded by page ID without installing a hot pointer in an owner edge.
    /// The frame can be evicted directly until a traversal swizzles it and
    /// replaces this state with `Stable` or `InnerNode`.
    Unswizzled,
    /// Stable routing edge that outlives the pool
    /// (e.g., BTree::meta_swip, table directory Vec entry).
    /// Eviction writes Swip::evicted(pid) directly through this edge.
    Stable(StableSwipRef),
    /// Child of a B-tree inner node. Cached hint for fast unswizzle.
    /// Eviction validates this hint, and falls back to a tree walk
    /// via the pool's registered ParentFinder if stale.
    InnerNode(InnerParentLink),
}

/// Identity handle for a stable, externally owned `&AtomicSwip` routing edge.
///
/// Stored in [`ParentLink::Stable`] so eviction can publish
/// `Swip::evicted(pid)` directly through the edge without tree-walking. The
/// pointed-at storage must outlive every frame that carries the link;
/// construction is `unsafe` to encode that contract.
///
/// See the module-level docs and [`ParentLink::Stable`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StableSwipRef {
    ptr: NonNull<AtomicSwip>,
}

/// Identity handle for a [`BufferFrame`].
///
/// Construction is `unsafe` (caller asserts the frame remains live for the
/// duration of use). Internally a `NonNull<BufferFrame>` — no `Drop` cost, no
/// lifetime. The buffer pool hands these out from `fix` / `try_evict` paths
/// and B-tree swizzled pointers decode into them via
/// [`BufferFrameRef::from_hot_swip`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferFrameRef {
    ptr: NonNull<BufferFrame>,
}

/// Read view on a [`BufferFrame`] produced under a pin / shared-latch /
/// optimistic-guard protocol.
///
/// Construction is `unsafe` (caller asserts the frame is pinned or otherwise
/// protected from eviction for the lifetime `'a`). Exposes [`BufferFrameReadRef::page`]
/// as a `&'a [u8; PAGE_SIZE]`: the page-byte borrow is bounded by the lifetime
/// established at construction, which the guard types tie to their own borrow
/// of the pool. This prevents the page bytes from outliving the guard that
/// makes them valid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferFrameReadRef<'a> {
    frame: BufferFrameRef,
    _marker: std::marker::PhantomData<&'a ()>,
}

/// Write view on a [`BufferFrame`] produced under an exclusive latch or
/// eviction-time ownership.
///
/// Construction is `unsafe` (caller asserts exclusive access for the lifetime
/// `'a`). Exposes mutable page bytes and the parent-link mutators used by the
/// eviction / publish-split paths. The lifetime `'a` bounds the mutable page
/// borrow to the guard that authorizes mutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferFrameWriteRef<'a> {
    frame: BufferFrameRef,
    _marker: std::marker::PhantomData<&'a ()>,
}

// SAFETY: BufferFrameRef is an identity reference to a BufferFrame. The frame
// itself is Send + Sync; callers still need the same latch/pin/eviction
// protocol that was required for the underlying frame pointer.
unsafe impl Send for BufferFrameRef {}
unsafe impl Sync for BufferFrameRef {}
unsafe impl Send for BufferFrameReadRef<'_> {}
unsafe impl Sync for BufferFrameReadRef<'_> {}
unsafe impl Send for BufferFrameWriteRef<'_> {}
unsafe impl Sync for BufferFrameWriteRef<'_> {}
// SAFETY: StableSwipRef carries identity for a caller-proven stable routing
// edge. AtomicSwip provides its own synchronization; the lifetime invariant is
// established at construction.
unsafe impl Send for StableSwipRef {}
unsafe impl Sync for StableSwipRef {}

impl StableSwipRef {
    /// # Safety
    /// `swip` must be a stable routing edge whose storage will not move or be
    /// freed while any buffer frame may refer to it through `ParentLink`.
    pub unsafe fn from_ref(swip: &AtomicSwip) -> Self {
        Self {
            ptr: NonNull::from(swip),
        }
    }

    pub fn store_evicted(self, pid: PageId) {
        unsafe {
            self.ptr
                .as_ref()
                .store(Swip::evicted(pid), Ordering::Release)
        };
    }

    pub fn load(self, order: Ordering) -> Swip {
        unsafe { self.ptr.as_ref().load(order) }
    }

    pub fn ptr_eq(self, swip: &AtomicSwip) -> bool {
        std::ptr::eq(self.ptr.as_ptr().cast_const(), swip)
    }

    pub(crate) fn as_ref(&self) -> &AtomicSwip {
        unsafe { self.ptr.as_ref() }
    }
}

unsafe fn extend_optimistic_guard<'from, 'to>(
    guard: OptimisticGuard<'from>,
) -> OptimisticGuard<'to> {
    unsafe { std::mem::transmute::<OptimisticGuard<'from>, OptimisticGuard<'to>>(guard) }
}

impl BufferFrameRef {
    /// # Safety
    /// `ptr` must identify a live buffer-pool frame for every use of the
    /// returned reference value.
    pub(crate) unsafe fn from_raw(ptr: *mut BufferFrame) -> Self {
        Self {
            ptr: NonNull::new(ptr).expect("buffer frame pointer must not be null"),
        }
    }

    pub fn from_hot_swip(swip: Swip) -> Option<Self> {
        if !(swip.is_hot() || swip.is_cool()) {
            return None;
        }
        let ptr = unsafe { swip.as_ptr::<BufferFrame>() };
        let addr = ptr as usize;
        if addr < PAGE_SIZE || !addr.is_multiple_of(std::mem::align_of::<BufferFrame>()) {
            return None;
        }
        Some(unsafe { Self::from_raw(ptr) })
    }

    pub(crate) fn as_ptr(self) -> *mut BufferFrame {
        self.ptr.as_ptr()
    }

    pub fn same_frame(self, other: Self) -> bool {
        self.ptr == other.ptr
    }

    pub fn pid(self) -> u64 {
        unsafe { self.ptr.as_ref().header.core.pid }
    }

    pub fn state(self) -> FrameState {
        unsafe { self.ptr.as_ref().header.core.state.load(Ordering::Acquire) }
    }

    pub fn hot_swip(self) -> Swip {
        Swip::hot(self.as_ptr())
    }

    /// # Safety
    /// The caller must ensure the frame remains live while the optimistic guard
    /// is used, and must validate the guard before trusting data read through
    /// it.
    pub unsafe fn optimistic_guard<'a>(self) -> Result<OptimisticGuard<'a>, Restart> {
        let guard = unsafe { self.ptr.as_ref().latch.optimistic_or_restart()? };
        Ok(unsafe { extend_optimistic_guard(guard) })
    }

    /// # Safety
    /// The caller must ensure the frame is live and protected for reads by an
    /// appropriate pin/latch/eviction protocol for the duration `'a`.
    pub unsafe fn read_ref<'a>(self) -> BufferFrameReadRef<'a> {
        BufferFrameReadRef {
            frame: self,
            _marker: std::marker::PhantomData,
        }
    }

    /// # Safety
    /// The caller must ensure the frame is live and protected for mutation by
    /// an exclusive latch or equivalent eviction-time ownership for the
    /// duration `'a`.
    pub unsafe fn write_ref<'a>(self) -> BufferFrameWriteRef<'a> {
        BufferFrameWriteRef {
            frame: self,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a> BufferFrameReadRef<'a> {
    pub fn frame(self) -> BufferFrameRef {
        self.frame
    }

    pub fn page(self) -> &'a [u8; PAGE_SIZE] {
        // SAFETY: caller of read_ref asserted the frame is pinned/protected
        // for 'a. The page bytes are stable while the frame is resident.
        unsafe { &self.frame.ptr.as_ref().page }
    }

    pub fn parent_link(self) -> ParentLink {
        unsafe { self.frame.ptr.as_ref().header.parent_link }
    }
}

impl<'a> BufferFrameWriteRef<'a> {
    pub fn frame(self) -> BufferFrameRef {
        self.frame
    }

    pub fn read_ref(self) -> BufferFrameReadRef<'a> {
        BufferFrameReadRef {
            frame: self.frame,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn page(self) -> &'a [u8; PAGE_SIZE] {
        self.read_ref().page()
    }

    pub fn page_mut(self) -> &'a mut [u8; PAGE_SIZE] {
        // SAFETY: caller of write_ref asserted exclusive access for 'a.
        unsafe { &mut (*self.frame.as_ptr()).page }
    }

    pub fn parent_link(self) -> ParentLink {
        self.read_ref().parent_link()
    }

    pub fn set_parent_link_none(self) {
        unsafe { (*self.frame.as_ptr()).header.parent_link = ParentLink::None };
    }

    pub fn set_parent_link_stable(self, meta_swip: StableSwipRef) {
        unsafe { (*self.frame.as_ptr()).header.parent_link = ParentLink::Stable(meta_swip) };
    }

    pub fn set_parent_link_inner(
        self,
        parent_pid: u64,
        slot_index: u16,
        is_upper: bool,
        dt_id: u16,
    ) {
        unsafe {
            (*self.frame.as_ptr()).header.parent_link = ParentLink::InnerNode(InnerParentLink {
                parent_pid,
                slot_index,
                is_upper,
                dt_id,
            });
        }
    }

    pub fn mark_header_dirty(self) {
        unsafe {
            self.frame
                .ptr
                .as_ref()
                .header
                .core
                .dirty
                .store(true, Ordering::Relaxed);
        }
    }
}

/// Callback trait for finding and unswizzling a child's parent routing
/// edge during eviction. Registered on the BufferPool by the B-tree.
///
/// The implementation must:
/// 1. Walk the tree to find which inner node contains `child.hot_swip()`
/// 2. Exclusively latch that parent
/// 3. Write `Swip::evicted(child_pid)` to the correct slot
/// 4. Mark the parent dirty
/// 5. Release the latch
///
/// Returns true if the parent was found and updated, false if not
/// (e.g., child is the root, or tree structure changed).
pub trait ParentFinder: Send + Sync {
    fn find_and_unswizzle_with_hint(
        &self,
        _child: BufferFrameRef,
        _child_pid: u64,
        _parent_pid: u64,
        _slot_index: u16,
        _is_upper: bool,
    ) -> Option<bool> {
        None
    }

    fn find_and_unswizzle(&self, child: BufferFrameRef, child_pid: u64) -> bool;
}

/// Callback trait for best-effort page reclamation just before eviction.
///
/// Intended for cold-page maintenance work that needs ownership context
/// outside the buffer pool, such as watermark-driven delta pruning on
/// table-owned delta pages.
pub trait PageReclaimer: Send + Sync {
    fn try_reclaim_before_evict(&self, page_pid: u64, page: BufferFrameWriteRef<'_>);
}

/// Callback trait for converting resident-only page bytes in a writeback copy.
///
/// The buffer pool calls this for WAL page images and data-file writeback. The
/// callback must only mutate the supplied copy, not the resident frame.
pub trait PageWritebackPreparer: Send + Sync {
    fn prepare_page_copy_for_writeback(
        &self,
        page: &mut [u8],
        pool: &crate::buffer_pool::BufferPool,
    );
}

// SAFETY: ParentLink stores frame identity hints and stable SWIP edges. Users
// still validate or dereference those identities under the eviction protocol.
unsafe impl Send for ParentLink {}
unsafe impl Sync for ParentLink {}

#[repr(C)]
struct HeaderPrefix {
    latch: HybridLatch,
    header: FrameHeader,
}

const HEADER_BYTES: usize = std::mem::size_of::<HeaderPrefix>();
const HEADER_PAD: usize = PAGE_SIZE - HEADER_BYTES;
const _: () = assert!(HEADER_BYTES <= PAGE_SIZE);

// ---------------------------------------------------------------------------
// BufferFrame
// ---------------------------------------------------------------------------

/// The per-page in-memory slot.
///
/// Layout (4096-aligned, two PAGE_SIZE halves):
///
/// ```text
///   ┌──────────────────────────────┐  ← 4096-aligned base
///   │ HybridLatch                  │
///   │ FrameHeader (core +          │
///   │              parent_link)    │
///   │ header_pad (zeroed)          │
///   ├──────────────────────────────┤  ← offset PAGE_SIZE
///   │ page: [u8; PAGE_SIZE]        │
///   └──────────────────────────────┘
/// ```
///
/// `BufferFrame` is `#[repr(C, align(4096))]`; the `page` field sits at a
/// `PAGE_SIZE` offset from the struct base, which lets swizzled pointers
/// (which encode `&BufferFrame` as a `Hot`/`Cool` SWIP) reach the page bytes
/// with a single pointer arithmetic step. Construction is via
/// [`BufferFrame::new`] / [`BufferFrame::default`]; the buffer pool owns the
/// frames as `Box<[BufferFrame]>` and routes the buffers via
/// [`BufferFrameRef`].
#[repr(C, align(4096))]
pub struct BufferFrame {
    pub latch: HybridLatch,
    pub header: FrameHeader,
    _header_pad: [u8; HEADER_PAD],
    pub page: [u8; PAGE_SIZE],
}

/// Header-resident state shared by every frame.
///
/// `core` (re-exported from `pagebox-frame-kernel`) holds the atomic
/// `FrameState`, `pin_count`, `dirty` bit, `referenced` bit, page LSN, and WAL
/// buffer locality; `parent_link` is the eviction routing edge described at
/// the module level.
pub struct FrameHeader {
    pub core: FrameCoreHeader,
    /// How eviction can find and unswizzle the parent's routing edge
    /// that points to this frame. Only modified under exclusive latch.
    pub parent_link: ParentLink,
}

// SAFETY: BufferFrame is designed for concurrent access. The latch guards
// mutable fields; pin_count, dirty, and state are atomic. Raw pointers
// Parent-link raw pointers are only dereferenced under appropriate
// synchronization.
unsafe impl Send for BufferFrame {}
unsafe impl Sync for BufferFrame {}

impl Default for BufferFrame {
    /// ```anneal
    /// ensures: ret.latch = HybridLatch::new() ∧ ret.header = FrameHeader::new_free()
    /// proof (h_anon):
    ///   unfold Default::default at h_returns
    ///   unfold BufferFrame::new at h_returns
    ///   simp_all
    /// proof (h_progress):
    ///   unfold Default::default
    ///   unfold BufferFrame::new
    ///   refine ⟨BufferFrame { latch: HybridLatch::new(), header: FrameHeader::new_free(), parent_link: ParentLink::None, _header_pad: [0u8; HEADER_PAD], page: [0u8; PAGE_SIZE] }, ?_⟩
    ///   rfl
    /// ```
    fn default() -> Self {
        Self::new()
    }
}

impl BufferFrame {
    /// ```anneal
    /// ensures:
    ///   ret.latch = HybridLatch::new() ∧
    ///   ret.header.core.pin_count = 0 ∧
    ///   ret.header.core.pid = 0 ∧
    ///   ret.header.core.dirty = false ∧
    ///   ret.header.core.referenced = false ∧
    ///   ret.header.core.state = FrameState::Free ∧
    ///   ret.header.parent_link = ParentLink::None ∧
    ///   ret.page = [0u8; PAGE_SIZE]
    /// proof (h_anon):
    ///   unfold BufferFrame::new at h_returns
    ///   simp_all [FrameCoreHeader::new_free, HybridLatch::new]
    /// proof (h_progress):
    ///   unfold BufferFrame::new
    ///   refine ⟨BufferFrame { latch: HybridLatch::new(), header: FrameHeader { core: FrameCoreHeader::new_free(), parent_link: ParentLink::None }, _header_pad: [0u8; HEADER_PAD], page: [0u8; PAGE_SIZE] }, ?_⟩
    ///   constructor <;> rfl
    /// ```
    pub fn new() -> Self {
        BufferFrame {
            latch: HybridLatch::new(),
            header: FrameHeader {
                core: FrameCoreHeader::new_free(),
                parent_link: ParentLink::None,
            },
            _header_pad: [0u8; HEADER_PAD],
            page: [0u8; PAGE_SIZE],
        }
    }

    /// ```anneal
    /// ensures:
    ///   let page_start = (self as *const Self).cast::<u8>().wrapping_add(PAGE_SIZE);
    ///   ret = std::slice::from_raw_parts(page_start, class.page_size())
    /// proof (h_anon):
    ///   unfold BufferFrame::page_bytes at h_returns
    ///   simp_all [PAGE_SIZE]
    /// proof (h_progress):
    ///   unfold BufferFrame::page_bytes
    ///   refine ⟨std::slice::from_raw_parts((self as *const Self).cast::<u8>().wrapping_add(PAGE_SIZE), class.page_size()), ?_⟩
    ///   rfl
    /// ```
    pub fn page_bytes(&self) -> &[u8] {
        &self.page
    }

    pub fn page_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.page
    }

    /// Convert `Hot`/`Cool` child swips in this frame's page bytes to
    /// `Evicted(pid)` before writing to disk.
    ///
    /// The resident in-memory format uses pointer-encoding SWIPs; the on-disk
    /// format uses page IDs. Inner-node B-tree pages embed child swips in slot
    /// values plus the upper-link swip at the page tail; both are rewritten
    /// here. Leaves, scan-only pages, and non-Index page types short-circuit.
    ///
    /// # Safety
    ///
    /// Must be called under the frame's exclusive latch so child swips cannot
    /// change concurrently during in-place conversion.
    pub unsafe fn convert_hot_swips_to_evicted(&mut self) {
        use crate::slotted_page::{self, SlottedPage};
        use pagebox_swip_kernel::SwipWord as Swip;

        let page_type = slotted_page::read_page_type(&self.page);
        if page_type != slotted_page::PageType::Index {
            return;
        }
        let sp = SlottedPage::from_page(&self.page);
        // FLAG_IS_LEAF = 1 << 1
        if sp.has_custom_flag(1 << 1) {
            return; // leaf node — no child swips to convert
        }
        let n = sp.num_slots();
        let sp_mut = SlottedPage::from_page_mut(&mut self.page);
        for i in 0..n {
            let val = sp_mut.get_value(i);
            if val.len() >= 8 {
                let raw = u64::from_ne_bytes(val[..8].try_into().unwrap());
                let swip = Swip::from_raw(raw);
                if swip.is_hot() || swip.is_cool() {
                    let child_pid = unsafe { (*swip.as_ptr::<BufferFrame>()).header.core.pid };
                    sp_mut.update_value_if_same_length(
                        i,
                        &Swip::evicted(child_pid).raw().to_ne_bytes(),
                    );
                }
            }
        }
        // Also convert upper swip at PAGE_SIZE - 8.
        let upper_off = PAGE_SIZE - 8;
        let upper_raw = u64::from_ne_bytes(self.page[upper_off..].try_into().unwrap());
        let upper_swip = Swip::from_raw(upper_raw);
        if upper_swip.is_hot() || upper_swip.is_cool() {
            let upper_pid = unsafe { (*upper_swip.as_ptr::<BufferFrame>()).header.core.pid };
            self.page[upper_off..].copy_from_slice(&Swip::evicted(upper_pid).raw().to_ne_bytes());
        }
    }

    /// Convert `Hot`/`Cool` child swips to `Evicted(pid)` in a *copy* of the
    /// page bytes, leaving the live frame untouched.
    ///
    /// Used on the writeback path where optimistic readers may still be
    /// observing the live page — they keep seeing swizzled pointers, while the
    /// bytes written to disk carry page IDs. The pool is consulted to validate
    /// each swip resolves to a frame it actually owns before rewriting, so a
    /// stale or aliased word is left as-is rather than being silently
    /// rewritten to a wrong page ID.
    pub fn convert_swips_in_buf(buf: &mut [u8; PAGE_SIZE], pool: &crate::buffer_pool::BufferPool) {
        use crate::slotted_page::{self, SlottedPage};
        use pagebox_swip_kernel::SwipWord as Swip;

        let page_type = slotted_page::read_page_type(buf);
        if page_type != slotted_page::PageType::Index {
            return;
        }
        let sp = SlottedPage::from_page(buf);
        if sp.has_custom_flag(1 << 1) {
            return; // leaf
        }
        let n = sp.num_slots();
        let sp_mut = SlottedPage::from_page_mut(buf);
        for i in 0..n {
            let val = sp_mut.get_value(i);
            if val.len() >= 8 {
                let raw = u64::from_ne_bytes(val[..8].try_into().unwrap());
                let swip = Swip::from_raw(raw);
                if let Some(frame) = BufferFrameRef::from_hot_swip(swip)
                    && pool.contains_frame(frame)
                {
                    let child_pid = frame.pid();
                    sp_mut.update_value_if_same_length(
                        i,
                        &Swip::evicted(child_pid).raw().to_ne_bytes(),
                    );
                }
            }
        }
        let upper_off = PAGE_SIZE - 8;
        let upper_raw = u64::from_ne_bytes(buf[upper_off..].try_into().unwrap());
        let upper_swip = Swip::from_raw(upper_raw);
        if let Some(frame) = BufferFrameRef::from_hot_swip(upper_swip)
            && pool.contains_frame(frame)
        {
            let upper_pid = frame.pid();
            buf[upper_off..].copy_from_slice(&Swip::evicted(upper_pid).raw().to_ne_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_id_is_identity_for_plain_pids() {
        assert_eq!(physical_page_number(42), 42);
        assert_eq!(page_size(42), PAGE_SIZE);
        assert_eq!(page_base_span(42), 1);
        assert_eq!(page_end_base_page(42), 42);
    }

    #[test]
    fn buffer_frame_uses_dedicated_page_aligned_data_region() {
        let bf = BufferFrame::new();
        let base = &bf as *const BufferFrame as usize;
        let page = bf.page.as_ptr() as usize;

        assert_eq!(
            page - base,
            PAGE_SIZE,
            "page bytes should start at the PAGE_SIZE offset from the frame base"
        );
        assert_eq!(page % 4096, 0, "page bytes must be 4096-aligned");
        assert_eq!(
            std::mem::size_of::<BufferFrame>(),
            PAGE_SIZE * 2,
            "buffer frame should occupy one header page plus one data page"
        );
    }
}
