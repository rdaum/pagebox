#![allow(
    unused_unsafe,
    reason = "NoLatches construction stays explicitly unsafe inside broader unsafe operations"
)]

//! The concurrent buffer pool: page fixing, residency, eviction, prefetch, and
//! dirty-page tracking.
//!
//! This is the heart of `pagebox-storage`. The [`BufferPool`] owns an
//! array of [`BufferFrame`]s and serves four kinds of
//! caller request against a page:
//!
//! 1. **fix** — make a page resident (load it from the page store if needed)
//!    and pin it so it cannot be evicted. Returns a [`PinnedFrame`].
//! 2. **latch** — acquire an optimistic / shared / exclusive guard on the
//!    frame's [`HybridLatch`]. Higher layers usually compose step 1 + 2: see
//!    [`PinnedFrame::optimistic`] / [`PinnedFrame::shared`] /
//!    [`PinnedFrame::exclusive`].
//! 3. **evict** — pick a resident unpinned page and transition it through
//!    `Resident → Evicting → Free`, writing back its bytes if dirty and
//!    unswizzling the routing edge in its parent (see [`ParentFinder`]).
//! 4. **flush** — write all dirty pages back to the page store and advance
//!    WAL checkpointing. Used at quiescence and during checkpoint.
//!
//! Page references are *swizzled*: a pointer-sized `Swip` word is either a
//! `Hot`/`Cool` direct pointer to a resident frame or an `Evicted(pid)`. Fix
//! takes an `&AtomicSwip` (the parent's routing edge) and either pins the
//! hot frame directly (CASing it to `Cool` if needed) or loads the page from
//! disk into a free frame and atomically swaps the SWIP hot. Cross-structure
//! parent locating is delegated to a `dt_registry` of [`ParentFinder`]s, one
//! per data structure (B-tree) registered via [`BufferPool::register_dt`].
//!
//! ## Resident budget & eviction
//!
//! A fixed count of base pages is allowed to be resident at once. When the
//! budget is exhausted `fix` invokes the configured eviction policy to free a
//! frame:
//!
//! - `RandomSecondChance` (default) — sample a frame, skip it if its referenced
//!   bit is set (and clear the bit), otherwise evict.
//! - `BatchClock` — clock-style scan over a batch of candidate frames. Select
//!   with `PAGEBOX_EVICTION_POLICY=batch_clock`.
//!
//! Dirty pages are **no-steal**: eviction refuses them until
//! [`BufferPool::flush`] (or `flush_dirty_batch`) writes them to the page store
//! (and, when a WAL is attached, pages are logged before being written back).
//! Stable parent-link pages — e.g. the B-tree root wired through
//! [`ParentLink::Stable`] — are not eligible for eviction.
//!
//! ## Pin count guarantee
//!
//! Pin count is incremented on `fix` and decremented on `unfix` / `Drop` of
//! the [`PinnedFrame`]. When repeated full-pool sweeps cannot release a frame
//! because candidates remain pinned, dirty, latched, or cannot be unswizzled,
//! a new fix panics with a detailed `buffer pool exhausted` snapshot. There is
//! no on-demand growth; size the pool to fit the working set.
//!
//! ## Frame state machine
//!
//! Each frame's `FrameCoreHeader.state` cycles through:
//!
//! ```text
//!   Free ──fix──▶ Loading ──load complete──▶ Resident
//!   Resident ──try_evict──▶ Evicting ──writeback + unswizzle──▶ Free
//! ```
//!
//! Concurrent `fix` calls on the same `Evicting` frame spin until the eviction
//! finishes (the frame becomes `Free`); they then re-enter the `Loading`
//! path. Loading frame waits are tracked per-page so contention can be
//! inspected post-hoc (`fix_orphan_latch_wait_top_page`,
//! `fix_orphan_evicting_retry_top_page`).
//!
//! ## Optimistic-frame protocol
//!
//! The hot B+tree read path takes an optimistic guard on a hot frame, then
//! re-validates before committing. Because a child can be evicted between the
//! snapshot and the pin, the canonical pattern is *snapshot → pin → validate
//! → access → validate*. [`OptimisticFrame::validate`] re-loads both the latch
//! version word and the frame state; either moving invalidates the section.
//! Repeated failures fall back to a true shared latch. See `pagebox-btree`'s
//! `find_leaf_optimistic` for the reference loop bounded by
//! `LOOKUP_OPTIMISTIC_RESTART_LIMIT`.
//!
//! ## Telemetry
//!
//! `metrics` (default): every fix, eviction, prefetch, latch wait, and page
//! load is recorded under labeled counters / histograms. With the feature
//! disabled these become no-op calls. Visit them via
//! [`BufferPool::visit_metrics`].
//!
//! [`HybridLatch`]: pagebox_hybrid_latch::HybridLatch
//! [`BufferFrame`]: crate::buffer_frame::BufferFrame

use std::collections::HashMap;
use std::collections::HashSet;
use std::num::NonZeroU64;
use std::ops::Deref;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use crossbeam_queue::ArrayQueue;
#[cfg(feature = "metrics")]
use fast_telemetry::{
    Counter, DeriveLabel, ExportMetrics, Gauge, Histogram, LabeledCounter, LabeledGauge,
    MetricVisitor,
};
use pagebox_hybrid_latch::{ExclusiveGuard, OptimisticGuard, Restart, SharedGuard};
use pagebox_threading as threading;
#[cfg(not(miri))]
use pagebox_wal::{BufferedWalRecord, Wal};

#[cfg(not(feature = "metrics"))]
use crate::metrics_stub::{Counter, Gauge, Histogram, LabeledCounter, LabeledGauge, MetricVisitor};

use crate::buffer_frame::PAGE_SIZE;
use crate::buffer_frame::{
    BufferFrame, BufferFrameReadRef, BufferFrameRef, BufferFrameWriteRef, EvictingFrame, FrameId,
    FrameState, PageReclaimer, PageWritebackPreparer, ParentFinder, ParentLink, PoolId, StableSwip,
    StableSwipOwner, physical_page_number,
};
use crate::buffer_frame::{Lsn, PageId};
use crate::free_page_allocator::{FreeExtent, FreePageAllocator};
use crate::page_header::{self, PageType};
use crate::page_provider;
use crate::page_store::{InMemoryPageStore, PageStore};
use pagebox_swip_kernel::{AtomicSwipWord as AtomicSwip, SwipWord as Swip};

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);
const FRAME_CAPACITY_STALL_LIMIT: u32 = 256;

fn allocate_pool_id() -> PoolId {
    let raw = NEXT_POOL_ID
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            current.checked_add(1)
        })
        .expect("buffer-pool identity space exhausted");
    PoolId::new(NonZeroU64::new(raw).expect("buffer-pool identity must be nonzero"))
}

// ---------------------------------------------------------------------------
// FreeList — recycled buffer frames
// ---------------------------------------------------------------------------

/// Bounded MPMC queue for recycling evicted frames back to the pool.
///
/// Frames are reused immediately, so a raw-pointer Treiber stack is vulnerable
/// to ABA: the same pointer can leave and re-enter while a concurrent pop still
/// holds its old successor. `ArrayQueue` provides ABA-safe ownership without
/// serializing concurrent cache misses or allocating after construction.
struct FreeList {
    frames: ArrayQueue<usize>,
}

impl FreeList {
    fn new(capacity: usize) -> Self {
        Self {
            frames: ArrayQueue::new(capacity),
        }
    }

    /// Push a frame back to the free list.
    /// The frame must be in `Free` state with no outstanding references.
    fn push(&self, bf: *mut BufferFrame) {
        assert!(
            self.frames.push(bf as usize).is_ok(),
            "free list overflow: frame returned more than once"
        );
    }

    /// Pop a frame from the free list. Returns `None` if the list is empty.
    fn try_pop(&self) -> Option<*mut BufferFrame> {
        self.frames.pop().map(|bf| bf as *mut BufferFrame)
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.frames.len()
    }
}

// ---------------------------------------------------------------------------
// PageTable — sharded PID→frame map for orphan fixes and resident lookups
// ---------------------------------------------------------------------------

const PAGE_TABLE_SHARDS: usize = 8;

/// Concurrent PID→frame map used to find resident/in-flight pages without
/// slot addressing. Replaces the `slot(pid)` direct-addressing mechanism.
/// Sharded by PID for scalability (like LeanStore's `Partition::io_ht`).
struct PageTable {
    shards: [parking_lot::Mutex<HashMap<u64, *mut BufferFrame>>; PAGE_TABLE_SHARDS],
}

impl PageTable {
    fn new() -> Self {
        Self {
            shards: std::array::from_fn(|_| parking_lot::Mutex::new(HashMap::new())),
        }
    }

    fn shard(&self, pid: u64) -> &parking_lot::Mutex<HashMap<u64, *mut BufferFrame>> {
        let idx = (pid as usize) % PAGE_TABLE_SHARDS;
        &self.shards[idx]
    }

    /// Look up a page by PID. Returns the frame if resident or in-flight.
    fn lookup(&self, pid: u64) -> Option<*mut BufferFrame> {
        self.shard(pid).lock().get(&pid).copied()
    }

    /// Try to insert a PID→frame mapping. Returns `false` if the PID is
    /// already mapped (another thread won the race to load this page).
    fn try_insert(&self, pid: u64, bf: *mut BufferFrame) -> bool {
        let mut map = self.shard(pid).lock();
        if map.contains_key(&pid) {
            return false;
        }
        map.insert(pid, bf);
        true
    }

    /// Remove a PID→frame mapping. Called on eviction.
    fn remove(&self, pid: u64) {
        self.shard(pid).lock().remove(&pid);
    }

    /// Unconditionally insert a PID→frame mapping (for freshly allocated pages).
    fn insert(&self, pid: u64, bf: *mut BufferFrame) {
        self.shard(pid).lock().insert(pid, bf);
    }
}

// ---------------------------------------------------------------------------
// Arena — mmap on real builds, Vec fallback under miri
// ---------------------------------------------------------------------------

// With a single page class, every frame has the same stride: one header page
// plus one data page (both PAGE_SIZE). This equals `size_of::<BufferFrame>()`.
const FRAME_STRIDE: usize = 2 * PAGE_SIZE;

struct Arena {
    ptr: *mut BufferFrame,
    len: usize,
    byte_len: usize,
    frame_stride: usize,
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

#[cfg(not(miri))]
impl Arena {
    fn new(num_frames: usize) -> Self {
        let frame_stride = FRAME_STRIDE;
        let byte_len = frame_stride * num_frames;
        assert!(byte_len > 0, "cannot create empty frame array");

        let ptr = unsafe {
            #[cfg(target_os = "linux")]
            let flags = libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_NORESERVE;
            #[cfg(not(target_os = "linux"))]
            let flags = libc::MAP_ANONYMOUS | libc::MAP_PRIVATE;
            libc::mmap(
                std::ptr::null_mut(),
                byte_len,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED, "mmap failed");

        // Request Transparent Huge Pages for the frame arena to reduce TLB
        // pressure. The arena is large (16× the resident budget) and accessed
        // frequently on the hot path.
        #[cfg(target_os = "linux")]
        {
            let ret = unsafe { libc::madvise(ptr, byte_len, libc::MADV_HUGEPAGE) };
            // Non-fatal: the kernel may refuse THP for this region.
            debug_assert_eq!(
                ret,
                0,
                "madvise(MADV_HUGEPAGE) failed: {}",
                std::io::Error::last_os_error()
            );

            // Prevent the mapping from being inherited by fork(). Required
            // for O_DIRECT correctness: fork() creates a COW copy of the
            // address space, and direct I/O into a COW'd page may write
            // from the wrong physical page.
            let ret = unsafe { libc::madvise(ptr, byte_len, libc::MADV_DONTFORK) };
            debug_assert_eq!(
                ret,
                0,
                "madvise(MADV_DONTFORK) failed: {}",
                std::io::Error::last_os_error()
            );
        }

        let ptr = ptr as *mut BufferFrame;

        Arena {
            ptr,
            len: num_frames,
            byte_len,
            frame_stride,
        }
    }
}

#[cfg(not(miri))]
impl Drop for Arena {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.byte_len) };
    }
}

#[cfg(miri)]
impl Arena {
    fn new(num_frames: usize) -> Self {
        let frame_stride = FRAME_STRIDE;
        let byte_len = frame_stride * num_frames;
        let layout = std::alloc::Layout::from_size_align(byte_len, PAGE_SIZE).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut BufferFrame };
        assert!(!ptr.is_null(), "frame arena allocation failed");
        Arena {
            ptr,
            len: num_frames,
            byte_len,
            frame_stride,
        }
    }
}

#[cfg(miri)]
impl Drop for Arena {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.byte_len, PAGE_SIZE).unwrap();
        unsafe { std::alloc::dealloc(self.ptr as *mut u8, layout) };
    }
}

// Compile-time checks for alignment compatibility with mmap/madvise.
const _: () = assert!(align_of::<BufferFrame>() <= 4096);

struct AlignedPageCopy {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
}

impl AlignedPageCopy {
    fn copy_from(page: &[u8]) -> Self {
        let layout = std::alloc::Layout::from_size_align(page.len(), PAGE_SIZE).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        let ptr = std::ptr::NonNull::new(ptr).expect("aligned page copy allocation failed");
        unsafe {
            std::ptr::copy_nonoverlapping(page.as_ptr(), ptr.as_ptr(), page.len());
        }
        Self {
            ptr,
            len: page.len(),
        }
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedPageCopy {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, PAGE_SIZE).unwrap();
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

thread_local! {
    static SPARE_PAGE_BUF: std::cell::RefCell<Option<Box<[u8; PAGE_SIZE]>>> = const { std::cell::RefCell::new(None) };
}

/// A reusable thread-local page-level scratch buffer.
/// Avoids alloc/free on every mark_dirty_raw call. The scratch is
/// filled and prepared OUTSIDE the WAL mutex to avoid holding the WAL
/// lock during the copy and prepare_page_copy_for_writeback (which
/// walks the page to convert swizzled pointers). Holding the mutex for
/// that extra work increases contention and degrades throughput under
/// concurrent writers.
struct PageScratch {
    buf: Box<[u8; PAGE_SIZE]>,
}

impl PageScratch {
    fn take() -> Self {
        let buf = SPARE_PAGE_BUF.with(|cell| cell.borrow_mut().take());
        Self {
            buf: buf.unwrap_or_else(|| Box::new([0u8; PAGE_SIZE])),
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buf[..]
    }
}

impl Drop for PageScratch {
    fn drop(&mut self) {
        let buf = std::mem::replace(&mut self.buf, Box::new([0u8; PAGE_SIZE]));
        SPARE_PAGE_BUF.with(|cell| {
            *cell.borrow_mut() = Some(buf);
        });
    }
}

unsafe fn extend_shared_guard<'from, 'to>(guard: SharedGuard<'from>) -> SharedGuard<'to> {
    unsafe { std::mem::transmute::<SharedGuard<'from>, SharedGuard<'to>>(guard) }
}

unsafe fn extend_exclusive_guard<'from, 'to>(guard: ExclusiveGuard<'from>) -> ExclusiveGuard<'to> {
    unsafe { std::mem::transmute::<ExclusiveGuard<'from>, ExclusiveGuard<'to>>(guard) }
}

unsafe fn extend_optimistic_guard<'from, 'to>(
    guard: OptimisticGuard<'from>,
) -> OptimisticGuard<'to> {
    unsafe { std::mem::transmute::<OptimisticGuard<'from>, OptimisticGuard<'to>>(guard) }
}

fn prepare_page_copy_for_writeback(page: &mut [u8], pool: &BufferPool) {
    let page_type = page_header::read_page_type(page);
    let preparer = pool
        .page_writeback_preparers
        .lock()
        .get(&page_type)
        .cloned();
    if let Some(preparer) = preparer {
        preparer.prepare_page_copy_for_writeback(page, pool);
        return;
    }
    if page_type != PageType::Index {
        return;
    }
    let Some(head) = page
        .get_mut(..PAGE_SIZE)
        .and_then(|bytes| bytes.try_into().ok())
    else {
        return;
    };
    BufferFrame::convert_swips_in_buf(head, pool);
}

fn is_stable_index_root(page: &[u8], parent_link: ParentLink) -> bool {
    if !matches!(parent_link, ParentLink::Stable(_)) {
        return false;
    }
    matches!(
        page_header::read_page_type(page),
        PageType::Index | PageType::BeTreeInternal | PageType::BeTreeLeaf
    )
}

#[cfg(not(miri))]
#[derive(Clone, Copy)]
struct ParentChildEdge {
    slot: u16,
    is_upper: bool,
}

#[cfg(not(miri))]
impl ParentChildEdge {
    fn new(slot: u16, is_upper: bool) -> Self {
        Self { slot, is_upper }
    }

    unsafe fn read_raw(self, parent_bf: *mut BufferFrame) -> Option<u64> {
        if self.is_upper {
            let off = PAGE_SIZE - 8;
            let bytes: [u8; 8] = unsafe { (&(*parent_bf).page)[off..off + 8].try_into().unwrap() };
            return Some(u64::from_ne_bytes(bytes));
        }

        let sp = crate::slotted_page::SlottedPage::from_page(unsafe { &(*parent_bf).page });
        if self.slot >= sp.num_slots() {
            return None;
        }
        let val = sp.get_value(self.slot);
        if val.len() < 8 {
            return None;
        }
        Some(u64::from_ne_bytes(val[..8].try_into().unwrap()))
    }

    unsafe fn write_evicted(self, parent_bf: *mut BufferFrame, child_pid: u64) {
        let raw = Swip::evicted(child_pid).raw().to_ne_bytes();
        if self.is_upper {
            let off = PAGE_SIZE - 8;
            unsafe { (&mut (*parent_bf).page)[off..off + 8].copy_from_slice(&raw) };
            return;
        }

        let sp = crate::slotted_page::SlottedPage::from_page_mut(unsafe { &mut (*parent_bf).page });
        sp.update_value_if_same_length(self.slot, &raw);
    }
}

struct PoolState {
    arena: Arena,
    slot_init: Box<[AtomicU8]>,
    free_list: FreeList,
    eviction_hand: AtomicUsize,
}

// ---------------------------------------------------------------------------
// BufferPool
// ---------------------------------------------------------------------------

/// A concurrent buffer pool with page-owned slots and resident-budget eviction.
///
/// All public methods take `&self` — concurrency is handled internally
/// via atomic operations, the [`HybridLatch`](pagebox_hybrid_latch::HybridLatch)
/// on each slot, and mutexes on the page provider and page store.
///
/// Construction is via [`BufferPool::new`] (in-memory page store, used by tests
/// and the `kvstore` restart path) or [`BufferPool::with_store`] (caller
/// supplies an [`crate::page_store::PageStore`] implementation,
/// typically a [`crate::page_store::FilePageStore`]). Returns to
/// the caller not a `BufferPool` directly but a [`BufferPoolHandle`] for shared
/// ownership; hot-path callers borrow through `as_pool`/`Deref` to avoid
/// touching the `Arc` reference count.
///
/// Composition hooks:
/// - [`BufferPool::register_dt`] — register a [`ParentFinder`] per data
///   structure ID, so eviction can unswizzle a child's parent edge in-place.
/// - [`BufferPool::register_page_reclaimer`] — best-effort reclaim callback
///   invoked just before eviction (e.g. delta-page pruning on table-owned
///   pages).
/// - [`BufferPool::register_page_writeback_preparer`] — rewrite page bytes in
///   a writeback copy (e.g. swip → page-id conversion).
/// - [`BufferPool::set_wal`] — attach a WAL; after this, dirty pages are
///   logged before being written to the data file.
pub struct BufferPool {
    id: PoolId,
    self_weak: OnceLock<Weak<BufferPool>>,
    state: PoolState,
    page_table: PageTable,
    page_store: Box<dyn PageStore>,
    next_page_id: AtomicU64,
    free_page_allocator: FreePageAllocator,
    resident_base_pages: usize,
    resident_base_pages_available: AtomicUsize,
    #[cfg(not(miri))]
    wal: Option<Arc<Wal>>,
    prefetch_workers: std::sync::Mutex<PrefetchWorkers>,
    prefetch_inflight: parking_lot::Mutex<HashSet<PageId>>,
    metrics: BufferPoolMetrics,
    loading_frame_wait_peak_pages: parking_lot::Mutex<HashMap<u64, u32>>,
    loading_frame_wait_current_pages: parking_lot::Mutex<HashMap<u64, u32>>,
    fix_orphan_latch_wait_sample_clock: AtomicU64,
    fix_orphan_latch_wait_sampled_pages: parking_lot::Mutex<HashMap<u64, u64>>,
    fix_orphan_evicting_retry_sample_clock: AtomicU64,
    fix_orphan_evicting_retry_sampled_pages: parking_lot::Mutex<HashMap<u64, u64>>,
    /// Registry of parent finders by data structure ID.
    /// Each B-tree registers itself so eviction can tree-walk to find parents.
    dt_registry: parking_lot::Mutex<HashMap<u16, Arc<dyn ParentFinder>>>,
    /// Coordinate hot-frame pins against eviction.
    eviction_mu: parking_lot::RwLock<()>,
    /// Number of evictors waiting to enter the final write-locked free window.
    /// Readers back off when this is non-zero so they don't starve
    /// `Evicting -> Free` transitions.
    eviction_writer_pending: AtomicUsize,
    /// Bounded budget for concurrent evictions. At most `eviction_budget`
    /// frames can be in the Evicting state simultaneously. Acquired before
    /// transitioning a frame to Evicting; released when the frame reaches
    /// Free or reverts to Resident.
    eviction_budget: usize,
    eviction_in_flight: AtomicUsize,
    /// Page IDs currently owned by dirty-page cleaning batches. Claims avoid
    /// duplicate writes while allowing concurrent batches to clean disjoint
    /// pages.
    dirty_flush_inflight: parking_lot::Mutex<HashSet<PageId>>,
    /// Best-effort reclaim callbacks keyed by owning page ID.
    page_reclaimers: parking_lot::Mutex<HashMap<u64, Arc<dyn PageReclaimer>>>,
    /// Page-image preparation callbacks keyed by common page type.
    page_writeback_preparers: parking_lot::Mutex<HashMap<PageType, Arc<dyn PageWritebackPreparer>>>,
    /// Page extents retired from data structures but not reusable until the
    /// next buffer-pool flush has made the unlink durable in the data file.
    pending_reusable_extents: parking_lot::Mutex<Vec<FreeExtent>>,
    /// Background page provider thread handle. Disabled by default; set
    /// `PAGEBOX_ENABLE_BACKGROUND_PAGE_PROVIDER=1` to enable it for experiments.
    page_provider: std::sync::Mutex<page_provider::PageProviderHandle>,
}

/// Shared ownership handle for a buffer pool.
///
/// Cloning the handle is for structural ownership only. Hot paths should
/// borrow the underlying pool through `as_pool`/`Deref`, which does not touch
/// the `Arc` reference count.
///
/// Construction ([`BufferPoolHandle::new`]) also wires the pool's internal
/// `Weak<BufferPool>` self-reference, which background workers (prefetch,
/// page-provider) need in order to upgrade-and-stop on drop.
#[derive(Clone)]
pub struct BufferPoolHandle {
    inner: Arc<BufferPool>,
}

impl BufferPoolHandle {
    pub fn new(pool: Arc<BufferPool>) -> Self {
        let _ = pool.self_weak.set(Arc::downgrade(&pool));
        // Start the background page provider thread.
        if background_page_provider_enabled() {
            let weak = Arc::downgrade(&pool);
            pool.page_provider.lock().unwrap().start(weak);
        }
        Self { inner: pool }
    }

    pub fn as_pool(&self) -> &BufferPool {
        &self.inner
    }

    pub fn into_arc(self) -> Arc<BufferPool> {
        self.inner
    }
}

impl From<Arc<BufferPool>> for BufferPoolHandle {
    fn from(pool: Arc<BufferPool>) -> Self {
        Self::new(pool)
    }
}

impl From<&Arc<BufferPool>> for BufferPoolHandle {
    fn from(pool: &Arc<BufferPool>) -> Self {
        Self::new(pool.clone())
    }
}

impl Deref for BufferPoolHandle {
    type Target = BufferPool;

    fn deref(&self) -> &Self::Target {
        self.as_pool()
    }
}

#[cfg_attr(feature = "metrics", derive(DeriveLabel))]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "metrics", label_name = "unswizzle_parent_event")]
enum UnswizzleParentEvent {
    FastPathHits,
    FastPathLatchMisses,
    DfsFallbacks,
    DfsSuccesses,
    DfsFailures,
}

// SAFETY: All fields are either Send+Sync or behind Mutex/atomic.
unsafe impl Send for BufferPool {}
unsafe impl Sync for BufferPool {}

fn background_page_provider_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        // Disabled by default. The background thread competes with worker
        // threads for CPU under high contention, reducing throughput. The
        // inline backoff strategy in pop_free_frame handles eviction
        // contention without a separate thread.
        // Set PAGEBOX_ENABLE_BACKGROUND_PAGE_PROVIDER=1 to enable for
        // experiments.
        matches!(
            std::env::var("PAGEBOX_ENABLE_BACKGROUND_PAGE_PROVIDER")
                .ok()
                .as_deref(),
            Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
        )
    })
}

type EvictionReadGuard<'a> = parking_lot::lock_api::RwLockReadGuard<'a, parking_lot::RawRwLock, ()>;

#[cfg_attr(feature = "metrics", derive(DeriveLabel))]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "metrics", label_name = "page_kind")]
enum BufferPoolLoadedPageKind {
    InnerIndex,
    LeafIndex,
    Tuple,
    Delta,
    ResidentMeta,
    Unknown,
}

#[cfg_attr(feature = "metrics", derive(DeriveLabel))]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "metrics", label_name = "fix_orphan_event")]
enum BufferPoolFixOrphanEvent {
    LatchWait,
    LatchWaitFree,
    LatchWaitLoading,
    LatchWaitResident,
    LatchWaitOther,
    HotPinWait,
    LoadingRetry,
    EvictingRetry,
}

#[cfg_attr(feature = "metrics", derive(DeriveLabel))]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "metrics", label_name = "eviction_event")]
enum BufferPoolEvictionEvent {
    Evictions,
    SecondChanceSkips,
    DirtyMarks,
    DirtyRelogs,
}

#[cfg_attr(feature = "metrics", derive(DeriveLabel))]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "metrics", label_name = "frame_state")]
enum BufferPoolFrameState {
    Free,
    Resident,
    Loading,
    Evicting,
}

#[cfg_attr(feature = "metrics", derive(ExportMetrics))]
#[cfg_attr(feature = "metrics", metric_prefix = "buffer_pool")]
struct BufferPoolMetrics {
    #[cfg_attr(feature = "metrics", help = "Configured resident frame budget")]
    frames_total: Gauge,
    #[cfg_attr(feature = "metrics", help = "Occupied buffer-pool frames")]
    frames_occupied: Gauge,
    #[cfg_attr(feature = "metrics", help = "Buffer-pool frames by current state")]
    frame_state_frames: LabeledGauge<BufferPoolFrameState>,
    #[cfg_attr(
        feature = "metrics",
        help = "Resident frame budget pages currently available"
    )]
    resident_budget_available: Gauge,
    #[cfg_attr(
        feature = "metrics",
        help = "Simple prefetch pages currently in flight"
    )]
    simple_prefetch_inflight: Gauge,
    #[cfg_attr(
        feature = "metrics",
        help = "Pages currently present in the page store"
    )]
    pages_on_disk: Gauge,
    #[cfg_attr(feature = "metrics", help = "Buffer pool eviction events")]
    eviction_events: LabeledCounter<BufferPoolEvictionEvent>,
    #[cfg_attr(
        feature = "metrics",
        help = "Synchronous orphan fix load latency in nanoseconds"
    )]
    fix_orphan_sync_load_latency: Histogram,
    #[cfg_attr(
        feature = "metrics",
        help = "Synchronous orphan fix loaded pages by page kind"
    )]
    fix_orphan_sync_load_pages: LabeledCounter<BufferPoolLoadedPageKind>,
    #[cfg_attr(
        feature = "metrics",
        help = "Simple prefetch load latency in nanoseconds"
    )]
    simple_prefetch_load_latency: Histogram,
    #[cfg_attr(
        feature = "metrics",
        help = "Simple prefetch loaded pages by page kind"
    )]
    simple_prefetch_load_pages: LabeledCounter<BufferPoolLoadedPageKind>,
    #[cfg_attr(
        feature = "metrics",
        help = "Simple prefetch loads claimed by demand reads"
    )]
    simple_prefetch_demand_steals: Counter,
    #[cfg_attr(
        feature = "metrics",
        help = "Simple prefetch queue wait latency in nanoseconds"
    )]
    simple_prefetch_queue_wait_latency: Histogram,
    #[cfg_attr(
        feature = "metrics",
        help = "Simple prefetch service latency in nanoseconds"
    )]
    simple_prefetch_service_latency: Histogram,
    #[cfg_attr(
        feature = "metrics",
        help = "SWIP fix synchronous load latency in nanoseconds"
    )]
    fix_swip_sync_load_latency: Histogram,
    #[cfg_attr(
        feature = "metrics",
        help = "SWIP fix synchronous loaded pages by page kind"
    )]
    fix_swip_sync_load_pages: LabeledCounter<BufferPoolLoadedPageKind>,
    #[cfg_attr(feature = "metrics", help = "Dirty WAL page images by page kind")]
    dirty_wal_page_image_pages: LabeledCounter<BufferPoolLoadedPageKind>,
    #[cfg_attr(feature = "metrics", help = "Dirty WAL page image relogs by page kind")]
    dirty_wal_page_image_relog_pages: LabeledCounter<BufferPoolLoadedPageKind>,
    #[cfg_attr(feature = "metrics", help = "Compact dirty WAL page-patch records")]
    dirty_wal_page_patch_records: Counter,
    #[cfg_attr(feature = "metrics", help = "Encoded bytes in dirty WAL page patches")]
    dirty_wal_page_patch_bytes: Counter,
    #[cfg_attr(feature = "metrics", help = "Dirty writeback batches")]
    dirty_flush_batches: Counter,
    #[cfg_attr(feature = "metrics", help = "Pages copied for dirty writeback")]
    dirty_flush_pages: Counter,
    #[cfg_attr(
        feature = "metrics",
        help = "Nanoseconds spent waiting for WAL durability during dirty writeback"
    )]
    dirty_flush_wal_wait_ns: Counter,
    #[cfg_attr(
        feature = "metrics",
        help = "Nanoseconds spent writing dirty pages to the page store"
    )]
    dirty_flush_data_write_ns: Counter,
    #[cfg_attr(
        feature = "metrics",
        help = "Dirty pages cleaned at their copied mutation generation"
    )]
    dirty_flush_cleaned_pages: Counter,
    #[cfg_attr(
        feature = "metrics",
        help = "Dirty-page copies stale after writeback I/O"
    )]
    dirty_flush_stale_pages: Counter,
    #[cfg_attr(
        feature = "metrics",
        help = "Hot frame transition wait latency in nanoseconds"
    )]
    hot_frame_transition_wait_latency: Histogram,
    #[cfg_attr(
        feature = "metrics",
        help = "Loading frame transition wait latency in nanoseconds"
    )]
    loading_frame_transition_wait_latency: Histogram,
    #[cfg_attr(feature = "metrics", help = "Fix-orphan events")]
    fix_orphan_events: LabeledCounter<BufferPoolFixOrphanEvent>,
    #[cfg_attr(feature = "metrics", help = "Unswizzle parent lookup events")]
    unswizzle_parent_events: LabeledCounter<UnswizzleParentEvent>,
}

impl BufferPoolMetrics {
    fn new(shards: usize) -> Self {
        Self {
            frames_total: Gauge::new(),
            frames_occupied: Gauge::new(),
            frame_state_frames: LabeledGauge::new(),
            resident_budget_available: Gauge::new(),
            simple_prefetch_inflight: Gauge::new(),
            pages_on_disk: Gauge::new(),
            eviction_events: LabeledCounter::new(shards),
            fix_orphan_sync_load_latency: Histogram::new(&buffer_pool_latency_bounds_ns(), shards),
            fix_orphan_sync_load_pages: LabeledCounter::new(shards),
            simple_prefetch_load_latency: Histogram::new(&buffer_pool_latency_bounds_ns(), shards),
            simple_prefetch_load_pages: LabeledCounter::new(shards),
            simple_prefetch_demand_steals: Counter::new(shards),
            simple_prefetch_queue_wait_latency: Histogram::new(
                &buffer_pool_latency_bounds_ns(),
                shards,
            ),
            simple_prefetch_service_latency: Histogram::new(
                &buffer_pool_latency_bounds_ns(),
                shards,
            ),
            fix_swip_sync_load_latency: Histogram::new(&buffer_pool_latency_bounds_ns(), shards),
            fix_swip_sync_load_pages: LabeledCounter::new(shards),
            dirty_wal_page_image_pages: LabeledCounter::new(shards),
            dirty_wal_page_image_relog_pages: LabeledCounter::new(shards),
            dirty_wal_page_patch_records: Counter::new(shards),
            dirty_wal_page_patch_bytes: Counter::new(shards),
            dirty_flush_batches: Counter::new(shards),
            dirty_flush_pages: Counter::new(shards),
            dirty_flush_wal_wait_ns: Counter::new(shards),
            dirty_flush_data_write_ns: Counter::new(shards),
            dirty_flush_cleaned_pages: Counter::new(shards),
            dirty_flush_stale_pages: Counter::new(shards),
            hot_frame_transition_wait_latency: Histogram::new(
                &buffer_pool_latency_bounds_ns(),
                shards,
            ),
            loading_frame_transition_wait_latency: Histogram::new(
                &buffer_pool_latency_bounds_ns(),
                shards,
            ),
            fix_orphan_events: LabeledCounter::new(shards),
            unswizzle_parent_events: LabeledCounter::new(shards),
        }
    }
}

#[cfg(not(feature = "metrics"))]
impl BufferPoolMetrics {
    fn visit_metrics<V: MetricVisitor + ?Sized>(&self, _visitor: &mut V) {}
}

#[derive(Default)]
struct BufferPoolFrameStateCounts {
    resident: usize,
    loading: usize,
    evicting: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvictionPolicy {
    BatchClock,
    RandomSecondChance,
}

fn parse_eviction_policy(value: &str) -> Option<EvictionPolicy> {
    if value.eq_ignore_ascii_case("batch_clock")
        || value.eq_ignore_ascii_case("clock")
        || value.eq_ignore_ascii_case("batch")
    {
        return Some(EvictionPolicy::BatchClock);
    }
    if value.eq_ignore_ascii_case("random_second_chance")
        || value.eq_ignore_ascii_case("random")
        || value.eq_ignore_ascii_case("rsc")
    {
        return Some(EvictionPolicy::RandomSecondChance);
    }
    None
}

/// Exponential backoff for `pop_free_frame` contention.
///
/// Starts with `yield_now` ( cooperative scheduling hint, ~0us), then
/// doubles the sleep duration on each failed attempt: 1us, 2us, 4us, ...
/// capped at 10ms. Resets to yield when progress is made. This lets
/// latch-holding threads release their latches and run, instead of
/// burning CPU in a tight spin loop.
struct Backoff {
    attempt: u32,
    sleep_us: u64,
}

impl Backoff {
    fn new() -> Self {
        Self {
            attempt: 0,
            sleep_us: 0,
        }
    }

    fn reset(&mut self) {
        self.attempt = 0;
        self.sleep_us = 0;
    }

    fn snooze(&mut self) {
        self.attempt += 1;
        if self.sleep_us == 0 {
            #[cfg(not(loom))]
            std::thread::yield_now();
            #[cfg(loom)]
            loom::thread::yield_now();
            self.sleep_us = 1;
        } else {
            #[cfg(not(loom))]
            std::thread::sleep(std::time::Duration::from_micros(self.sleep_us));
            #[cfg(loom)]
            loom::thread::yield_now();
            self.sleep_us = (self.sleep_us * 2).min(10_000);
        }
    }
}

fn eviction_policy() -> EvictionPolicy {
    static VALUE: OnceLock<EvictionPolicy> = OnceLock::new();
    *VALUE.get_or_init(|| {
        let Ok(raw) = std::env::var("PAGEBOX_EVICTION_POLICY") else {
            return EvictionPolicy::RandomSecondChance;
        };
        parse_eviction_policy(raw.trim()).unwrap_or(EvictionPolicy::RandomSecondChance)
    })
}

/// A permit to transition a frame into the Evicting state.
///
/// Bounds the number of frames simultaneously in Evicting to
/// `BufferPool::eviction_budget` (pool_size / 4, min 1). Without this bound,
/// concurrent eviction calls on a small pool can transition most frames to
/// Evicting before any finishes the unswizzle → Free transition, leaving no
/// Resident frames for readers and panicking with "buffer pool exhausted."
///
/// Acquired via [`BufferPool::try_acquire_eviction_permit`] before the CAS
/// that transitions a frame to Evicting. Released on Drop — whether the frame
/// reaches Free (eviction succeeded) or reverts to Resident (eviction
/// aborted), the permit is released when the Evicting state ends.
struct EvictionPermit<'a> {
    in_flight: &'a AtomicUsize,
}
impl Drop for EvictionPermit<'_> {
    fn drop(&mut self) {
        self.in_flight.fetch_sub(1, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// NoLatches — witness token for blocking pool methods
// ---------------------------------------------------------------------------

/// A witness that no blocking frame latches (shared or exclusive) are
/// currently held by the calling thread.
///
/// Blocking pool methods like [`BufferPool::fix_orphan_frame`] and
/// [`BufferPool::fix_frame`] require this token because they may trigger
/// eviction, which needs to exclusively latch frames. If the caller holds a
/// shared or exclusive latch on a frame and calls a blocking method, eviction
/// may be unable to acquire the exclusive latch it needs, causing pool
/// exhaustion or deadlock.
///
/// `PinnedFrame` does NOT require this token — a pin prevents eviction of
/// the pinned frame but does not block exclusive-latch acquisition on other
/// frames.
///
/// The token is `!Copy` and `!Clone`, and construction is `unsafe`: the borrow
/// checker cannot prove the absence of guards because shared borrows of
/// `&self` coexist. Making the proof boundary explicit keeps latch accounting
/// off every shared/exclusive hot path.
///
/// # Safety contract
///
/// Constructing `NoLatches` while any of these are alive is a contract
/// violation that may cause pool exhaustion or deadlock:
/// - [`SharedFrame`], [`ExclusiveFrame`]
/// - raw [`SharedGuard`] or [`ExclusiveGuard`] values obtained through unsafe
///   internal APIs
///
/// ```compile_fail
/// use pagebox_storage::buffer_pool::{BufferPool, NoLatches};
///
/// let pool = BufferPool::new(1);
/// let witness = NoLatches::new(&pool);
/// ```
pub struct NoLatches<'a> {
    _marker: core::marker::PhantomData<&'a BufferPool>,
}

impl<'a> NoLatches<'a> {
    /// Construct a `NoLatches` witness.
    ///
    /// The caller asserts that no blocking frame latches are currently held by
    /// this thread.
    ///
    /// # Safety
    /// No [`SharedFrame`], [`ExclusiveFrame`], raw [`SharedGuard`], or raw
    /// [`ExclusiveGuard`] may be live on the calling thread. Violating this
    /// contract may cause pool exhaustion or deadlock.
    #[must_use]
    pub unsafe fn new(pool: &'a BufferPool) -> Self {
        let _ = pool;
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

/// Name of the eviction policy in effect for this process.
///
/// Resolved once from `PAGEBOX_EVICTION_POLICY` at first call; one of
/// `"random_second_chance"` (default) or `"batch_clock"`. Exposed for
/// diagnostics and benchmark labels rather than runtime switching.
pub fn eviction_policy_name() -> &'static str {
    match eviction_policy() {
        EvictionPolicy::BatchClock => "batch_clock",
        EvictionPolicy::RandomSecondChance => "random_second_chance",
    }
}

const PREFETCH_QUEUE_CAPACITY: usize = 2048;
const PREFETCH_WORKERS: usize = 16;

struct PrefetchRequest {
    pid: PageId,
    enqueued_at: Instant,
}

struct PrefetchWorkers {
    threads: Vec<std::thread::JoinHandle<()>>,
    txs: Vec<SyncSender<PrefetchRequest>>,
    rxs: Option<Vec<Receiver<PrefetchRequest>>>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl PrefetchWorkers {
    fn new() -> Self {
        let mut txs = Vec::with_capacity(PREFETCH_WORKERS);
        let mut rxs = Vec::with_capacity(PREFETCH_WORKERS);
        for _ in 0..PREFETCH_WORKERS {
            let (tx, rx) = sync_channel(PREFETCH_QUEUE_CAPACITY);
            txs.push(tx);
            rxs.push(rx);
        }
        Self {
            threads: Vec::new(),
            txs,
            rxs: Some(rxs),
            shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    fn start(&mut self, pool: Weak<BufferPool>) {
        if !self.threads.is_empty() {
            return;
        }
        let rxs = self
            .rxs
            .take()
            .expect("prefetch worker receivers already taken");
        let shutdown = Arc::clone(&self.shutdown);
        self.threads = rxs
            .into_iter()
            .enumerate()
            .map(|(idx, rx)| {
                let shutdown = Arc::clone(&shutdown);
                let pool = Weak::clone(&pool);
                threading::spawn_efficient(format!("prefetch-{idx}"), move || {
                    prefetch_worker_loop(pool, &shutdown, rx);
                })
                .expect("failed to spawn prefetch worker")
            })
            .collect();
    }

    fn stop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }

    fn try_send(&self, pid: PageId) -> bool {
        let worker_idx = (pid as usize) % self.txs.len().max(1);
        self.txs[worker_idx]
            .try_send(PrefetchRequest {
                pid,
                enqueued_at: Instant::now(),
            })
            .is_ok()
    }
}

impl Drop for PrefetchWorkers {
    fn drop(&mut self) {
        self.stop();
    }
}

struct PrefetchInflightGuard<'a> {
    pool: &'a BufferPool,
    pid: PageId,
    active: bool,
}

impl<'a> PrefetchInflightGuard<'a> {
    fn new(pool: &'a BufferPool, pid: PageId) -> Self {
        Self {
            pool,
            pid,
            active: true,
        }
    }

    fn claim_if_present(pool: &'a BufferPool, pid: PageId) -> Option<Self> {
        pool.prefetch_inflight_contains(pid)
            .then(|| Self::new(pool, pid))
    }

    fn disarm(mut self) {
        self.active = false;
    }
}

impl Drop for PrefetchInflightGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            self.pool.prefetch_inflight_remove(self.pid);
        }
    }
}

struct LoadingFrameReservation<'a> {
    pool: &'a BufferPool,
    bf: *mut BufferFrame,
    active: bool,
}

impl<'a> LoadingFrameReservation<'a> {
    fn new(pool: &'a BufferPool, bf: *mut BufferFrame) -> Self {
        Self {
            pool,
            bf,
            active: true,
        }
    }

    fn disarm(mut self) {
        self.active = false;
    }
}

impl Drop for LoadingFrameReservation<'_> {
    fn drop(&mut self) {
        if self.active {
            unsafe { self.pool.release_loading_frame(self.bf) };
        }
    }
}

fn saturating_duration_nanos(elapsed: Duration) -> u64 {
    elapsed.as_nanos().min(u64::MAX as u128) as u64
}

fn saturating_usize_to_i64(value: usize) -> i64 {
    value.min(i64::MAX as usize) as i64
}

impl BufferPool {
    fn page_kind(page: &[u8]) -> BufferPoolLoadedPageKind {
        if page_header::is_inner_index_page(page) {
            return BufferPoolLoadedPageKind::InnerIndex;
        }
        if page_header::should_remain_resident(page) {
            return BufferPoolLoadedPageKind::ResidentMeta;
        }
        match page_header::read_page_type(page) {
            PageType::Index => BufferPoolLoadedPageKind::LeafIndex,
            PageType::BeTreeInternal => BufferPoolLoadedPageKind::InnerIndex,
            PageType::BeTreeLeaf => BufferPoolLoadedPageKind::LeafIndex,
            PageType::Tuple => BufferPoolLoadedPageKind::Tuple,
            PageType::Delta => BufferPoolLoadedPageKind::Delta,
            PageType::Meta | PageType::RootMeta => BufferPoolLoadedPageKind::ResidentMeta,
            PageType::Unknown => BufferPoolLoadedPageKind::Unknown,
        }
    }

    fn record_page_kind(page: &[u8], page_kinds: &LabeledCounter<BufferPoolLoadedPageKind>) {
        page_kinds.inc(Self::page_kind(page));
    }

    fn record_page_load(
        &self,
        page: &[u8],
        elapsed: Duration,
        latency: &Histogram,
        page_kinds: &LabeledCounter<BufferPoolLoadedPageKind>,
    ) {
        latency.record(saturating_duration_nanos(elapsed));
        Self::record_page_kind(page, page_kinds);
    }

    fn record_simple_prefetch_load(&self, page: &[u8], elapsed: Duration) {
        self.record_page_load(
            page,
            elapsed,
            &self.metrics.simple_prefetch_load_latency,
            &self.metrics.simple_prefetch_load_pages,
        );
    }

    fn record_fix_swip_sync_load(&self, page: &[u8], elapsed: Duration) {
        self.record_page_load(
            page,
            elapsed,
            &self.metrics.fix_swip_sync_load_latency,
            &self.metrics.fix_swip_sync_load_pages,
        );
    }

    fn record_fix_orphan_sync_load(&self, page: &[u8], elapsed: Duration) {
        self.record_page_load(
            page,
            elapsed,
            &self.metrics.fix_orphan_sync_load_latency,
            &self.metrics.fix_orphan_sync_load_pages,
        );
    }

    fn record_hot_frame_transition_wait(&self, elapsed: Duration) {
        self.metrics
            .hot_frame_transition_wait_latency
            .record(saturating_duration_nanos(elapsed));
    }

    fn record_loading_frame_transition_wait(&self, elapsed: Duration) {
        self.metrics
            .loading_frame_transition_wait_latency
            .record(saturating_duration_nanos(elapsed));
    }

    fn record_simple_prefetch_queue_wait(&self, elapsed: Duration) {
        self.metrics
            .simple_prefetch_queue_wait_latency
            .record(saturating_duration_nanos(elapsed));
    }

    fn record_simple_prefetch_service(&self, elapsed: Duration) {
        self.metrics
            .simple_prefetch_service_latency
            .record(saturating_duration_nanos(elapsed));
    }

    unsafe fn install_loaded_frame_metadata(
        &self,
        bf: *mut BufferFrame,
        pid: PageId,
        parent_link: ParentLink,
        pin_count: u32,
    ) {
        unsafe {
            (*bf).header.core.pid = pid;
            (*bf).header.parent_link = parent_link;
            (*bf)
                .header
                .core
                .pin_count
                .store(pin_count, Ordering::Relaxed);
            (*bf).header.core.referenced.store(true, Ordering::Relaxed);
            (*bf).header.core.dirty.store(false, Ordering::Relaxed);

            let on_disk_lsn = page_header::read_page_lsn((*bf).page_bytes());
            (*bf)
                .header
                .core
                .page_lsn
                .store(on_disk_lsn, Ordering::Relaxed);
            (*bf)
                .header
                .core
                .wal_buffer_epoch
                .store(0, Ordering::Relaxed);
            (*bf)
                .header
                .core
                .wal_buffer_offset
                .store(0, Ordering::Relaxed);
        }
    }

    unsafe fn release_loading_frame(&self, bf: *mut BufferFrame) {
        let pid = unsafe { (*bf).header.core.pid };
        if pid != 0 {
            self.page_table.remove(pid);
        }
        unsafe {
            (*bf).header.core.pin_count.store(0, Ordering::Relaxed);
            (*bf).header.parent_link = ParentLink::None;
            (*bf)
                .header
                .core
                .state
                .store(FrameState::Free, Ordering::Release);
        }
        self.release_resident_budget(bf);
        // Recycle the frame to the FreeList (eliminates MADV_DONTNEED).
        self.state.free_list.push(bf);
    }

    fn enter_loading_frame_wait(&self, page_id: u64) {
        if page_id == 0 {
            return;
        }
        let mut current = self.loading_frame_wait_current_pages.lock();
        let next = current
            .get(&page_id)
            .copied()
            .unwrap_or(0)
            .saturating_add(1);
        current.insert(page_id, next);
        drop(current);

        let mut peaks = self.loading_frame_wait_peak_pages.lock();
        let peak = peaks.entry(page_id).or_insert(0);
        if next > *peak {
            *peak = next;
        }
    }

    fn exit_loading_frame_wait(&self, page_id: u64) {
        if page_id == 0 {
            return;
        }
        let mut current = self.loading_frame_wait_current_pages.lock();
        let Some(waiters) = current.get_mut(&page_id) else {
            return;
        };
        if *waiters <= 1 {
            current.remove(&page_id);
            return;
        }
        *waiters -= 1;
    }

    fn sample_fix_orphan_latch_wait_page(&self, page_id: u64) {
        let tick = self
            .fix_orphan_latch_wait_sample_clock
            .fetch_add(1, Ordering::Relaxed);
        if tick & 63 != 0 {
            return;
        }
        let mut sampled_pages = self.fix_orphan_latch_wait_sampled_pages.lock();
        *sampled_pages.entry(page_id).or_insert(0) += 1;
    }

    pub fn fix_orphan_latch_wait_top_page(&self) -> Option<(u64, u64)> {
        let sampled_pages = self.fix_orphan_latch_wait_sampled_pages.lock();
        sampled_pages
            .iter()
            .max_by_key(|(_, samples)| **samples)
            .map(|(&page_id, &samples)| (page_id, samples))
    }

    fn sample_fix_orphan_evicting_retry_page(&self, page_id: u64) {
        let tick = self
            .fix_orphan_evicting_retry_sample_clock
            .fetch_add(1, Ordering::Relaxed);
        if tick & 63 != 0 {
            return;
        }
        let mut sampled_pages = self.fix_orphan_evicting_retry_sampled_pages.lock();
        *sampled_pages.entry(page_id).or_insert(0) += 1;
    }

    pub fn fix_orphan_evicting_retry_top_page(&self) -> Option<(u64, u64)> {
        let sampled_pages = self.fix_orphan_evicting_retry_sampled_pages.lock();
        sampled_pages
            .iter()
            .max_by_key(|(_, samples)| **samples)
            .map(|(&page_id, &samples)| (page_id, samples))
    }

    fn lock_hot_pin(&self) -> Option<EvictionReadGuard<'_>> {
        // Only take the eviction fence when budget is low and the
        // background thread may be evicting. This keeps the in-memory
        // hot path lock-free when eviction isn't needed.
        if self.eviction_writer_pending.load(Ordering::Acquire) == 0
            && self.resident_base_pages_available.load(Ordering::Relaxed)
                > (self.resident_base_pages / 10).max(16)
        {
            return None;
        }
        loop {
            if self.eviction_writer_pending.load(Ordering::Acquire) == 0 {
                return Some(self.eviction_mu.read());
            }
            #[cfg(not(loom))]
            std::thread::yield_now();
            #[cfg(loom)]
            loom::thread::yield_now();
        }
    }

    fn try_lock_hot_pin(&self) -> Option<Option<EvictionReadGuard<'_>>> {
        if self.eviction_writer_pending.load(Ordering::Acquire) == 0
            && self.resident_base_pages_available.load(Ordering::Relaxed)
                > (self.resident_base_pages / 10).max(16)
        {
            return Some(None);
        }
        if self.eviction_writer_pending.load(Ordering::Acquire) != 0 {
            return None;
        }
        self.eviction_mu.try_read().map(Some)
    }
}

fn prefetch_worker_loop(
    pool: Weak<BufferPool>,
    shutdown: &std::sync::atomic::AtomicBool,
    rx: Receiver<PrefetchRequest>,
) {
    loop {
        match rx.recv_timeout(Duration::from_millis(10)) {
            Ok(req) => {
                let Some(pool) = pool.upgrade() else {
                    break;
                };
                pool.record_simple_prefetch_queue_wait(req.enqueued_at.elapsed());
                let service_start = Instant::now();
                let bf = if let Some(existing) = pool.page_table.lookup(req.pid) {
                    existing
                } else {
                    match pool.state.free_list.try_pop() {
                        Some(bf) => bf,
                        None => continue,
                    }
                };
                let Some(_inflight) = PrefetchInflightGuard::claim_if_present(&pool, req.pid)
                else {
                    continue;
                };
                let Some(_guard) = (unsafe { pool.try_lock_frame_exclusive_at(bf, req.pid) })
                else {
                    continue;
                };
                let Some(loading) = try_claim_prefetch_frame(&pool, bf, req.pid) else {
                    continue;
                };

                let read_start = Instant::now();
                let found = read_prefetch_page(&pool, bf, req.pid);
                finish_prefetch_frame(&pool, bf, found, read_start.elapsed(), loading);
                pool.record_simple_prefetch_service(service_start.elapsed());
            }
            Err(RecvTimeoutError::Timeout) => {
                if shutdown.load(Ordering::Acquire) {
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn try_claim_prefetch_frame(
    pool: &BufferPool,
    bf: *mut BufferFrame,
    pid: PageId,
) -> Option<LoadingFrameReservation<'_>> {
    let state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
    if state != FrameState::Free {
        return None;
    }
    if !pool.try_reserve_resident_budget() {
        return None;
    }

    unsafe {
        (*bf).header.core.state.transition(
            FrameState::Free,
            FrameState::Loading,
            Ordering::Relaxed,
        );
        (*bf).header.core.pid = pid;
        (*bf).header.core.pin_count.store(0, Ordering::Relaxed);
        (*bf).header.core.dirty.store(false, Ordering::Relaxed);
        (*bf).header.parent_link = ParentLink::None;
    }
    Some(LoadingFrameReservation::new(pool, bf))
}

fn read_prefetch_page(pool: &BufferPool, bf: *mut BufferFrame, pid: PageId) -> bool {
    let page = unsafe { (*bf).page_bytes_mut() };
    pool.page_store
        .read_page(pid, page)
        .expect("prefetch read failed")
}

fn finish_prefetch_frame(
    pool: &BufferPool,
    bf: *mut BufferFrame,
    found: bool,
    read_elapsed: Duration,
    loading: LoadingFrameReservation<'_>,
) {
    if found {
        pool.record_simple_prefetch_load(unsafe { (*bf).page_bytes() }, read_elapsed);
        unsafe {
            let on_disk_lsn = page_header::read_page_lsn((*bf).page_bytes());
            (*bf)
                .header
                .core
                .page_lsn
                .store(on_disk_lsn, Ordering::Relaxed);
            (*bf).header.core.dirty.store(false, Ordering::Relaxed);
            (*bf).header.core.pin_count.store(0, Ordering::Relaxed);
            (*bf).header.parent_link = ParentLink::Unswizzled;
            (*bf).header.core.state.transition(
                FrameState::Loading,
                FrameState::Resident,
                Ordering::Release,
            );
        }
        loading.disarm();
    }
}

/// A pinned, unlatched handle on a resident frame.
///
/// Returned by [`BufferPool::fix_orphan_frame`], [`BufferPool::fix_stable`],
/// and their `try_*` variants. New-page pins remain inside
/// [`NewStablePage`] or [`NewUnlinkedPage`] until publication. The frame
/// cannot be evicted for the lifetime of this guard (pin count is incremented
/// by one on construction and decremented on `Drop`).
///
/// A `PinnedFrame` does **not** carry a latch. To read or write page contents,
/// the caller must obtain a guard via [`PinnedFrame::optimistic`] (read),
/// [`PinnedFrame::shared`] (read), [`PinnedFrame::exclusive`] (write), or one
/// of the `try_*` variants on the resulting `OptimisticFrame` /
/// `SharedFrame` / `ExclusiveFrame`. The composed optimistic section's
/// contract is the one documented at the module level: snapshot, access,
/// [`OptimisticFrame::validate`]; restart on `Restart`.
///
/// `Clone` and [`PinnedFrame::clone_pin`] increment the pin count again, so
/// multiple handles may reference the same pinned frame.
///
/// ```compile_fail
/// use pagebox_storage::buffer_pool::PinnedFrame;
///
/// fn unlatched_read(frame: &PinnedFrame<'_>) {
///     let _ = frame.page_bytes();
/// }
/// ```
pub struct PinnedFrame<'a> {
    pool: &'a BufferPool,
    bf: *mut BufferFrame,
}

impl Clone for PinnedFrame<'_> {
    fn clone(&self) -> Self {
        unsafe {
            (*self.bf)
                .header
                .core
                .pin_count
                .fetch_add(1, Ordering::Relaxed)
        };
        Self {
            pool: self.pool,
            bf: self.bf,
        }
    }
}

impl<'a> PinnedFrame<'a> {
    /// # Safety
    /// `bf` must be a live pinned frame managed by `pool`.
    pub(crate) unsafe fn new(pool: &'a BufferPool, bf: *mut BufferFrame) -> Self {
        Self { pool, bf }
    }

    fn raw(&self) -> *mut BufferFrame {
        self.bf
    }

    /// # Safety
    /// The returned lifetime-erased identity must not be used after this pin
    /// is released or after its arena slot is recycled. Prefer [`Self::id`]
    /// for identity that needs to escape the pin.
    pub unsafe fn frame_ref(&self) -> BufferFrameRef {
        unsafe { BufferFrameRef::from_raw(self.bf) }
    }

    pub fn pid(&self) -> u64 {
        unsafe { (*self.bf).header.core.pid }
    }

    pub fn id(&self) -> FrameId {
        self.pool.frame_id(self.bf)
    }

    pub fn hot_swip(&self) -> Swip {
        Swip::hot(self.bf)
    }

    pub fn mark_dirty(&self) {
        unsafe { self.pool.mark_dirty_raw(self.bf) };
    }

    pub fn mark_dirty_with_lsn(&self, lsn: Lsn) {
        unsafe { self.pool.mark_dirty_with_lsn_raw(self.bf, lsn) };
    }
}

impl PoolState {
    fn new(slot_capacity: usize) -> Self {
        #[cfg(not(miri))]
        let slot_init = (0..slot_capacity)
            .map(|_| AtomicU8::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        #[cfg(miri)]
        let slot_init = (0..slot_capacity)
            .map(|_| AtomicU8::new(2))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            arena: Arena::new(slot_capacity),
            slot_init,
            free_list: FreeList::new(slot_capacity),
            eviction_hand: AtomicUsize::new(0),
        }
    }
}

impl Drop for PinnedFrame<'_> {
    fn drop(&mut self) {
        unsafe { self.pool.unfix_raw(self.bf) };
    }
}

/// A pinned frame plus an in-flight optimistic read guard.
///
/// Constructed via [`PinnedFrame::optimistic`]; on success the underlying
/// [`HybridLatch`](pagebox_hybrid_latch::HybridLatch) has snapshotted its
/// version word and the underlying frame is pinned so it cannot be evicted
/// mid-section. The reader's obligation is to call [`OptimisticFrame::validate`]
/// before committing to any decision based on page bytes read through the
/// guard — a writer may have entered and left its critical section, advancing
/// the base version by two.
///
/// Upgrades preserve the pin: [`OptimisticFrame::upgrade_to_shared`],
/// [`OptimisticFrame::upgrade_to_exclusive`], and the `try_*` variant promote
/// the optimistic guard into a [`SharedFrame`] or [`ExclusiveFrame`]. A failed
/// upgrade returns the original [`PinnedFrame`] so the caller can retry.
pub struct OptimisticFrame<'a> {
    pinned: PinnedFrame<'a>,
    guard: OptimisticGuard<'a>,
}

impl<'a> OptimisticFrame<'a> {
    /// Return this guard's lifetime-erased frame identity.
    ///
    /// # Safety
    ///
    /// The returned reference must not outlive this guard's pin, and it must
    /// not be used to read page bytes without preserving this optimistic
    /// guard's validation obligation.
    pub unsafe fn frame_ref(&self) -> BufferFrameRef {
        unsafe { self.pinned.frame_ref() }
    }

    /// # Safety
    /// The returned ordinary byte view may race with a writer. The caller must
    /// treat all reads as speculative and validate before using any result.
    pub unsafe fn read_ref(&self) -> BufferFrameReadRef<'_> {
        unsafe { self.frame_ref().read_ref() }
    }

    pub fn pid(&self) -> u64 {
        self.pinned.pid()
    }

    /// # Safety
    /// See [`OptimisticFrame::read_ref`].
    pub unsafe fn page(&self) -> &[u8; PAGE_SIZE] {
        unsafe { &(*self.pinned.raw()).page }
    }

    /// # Safety
    /// See [`OptimisticFrame::read_ref`].
    pub unsafe fn page_bytes(&self) -> &[u8] {
        unsafe { (*self.pinned.raw()).page_bytes() }
    }

    pub fn validate(&self) -> Result<(), Restart> {
        self.guard.validate()
    }

    pub fn into_pinned(self) -> PinnedFrame<'a> {
        let this = std::mem::ManuallyDrop::new(self);
        let _guard = unsafe { std::ptr::read(&this.guard) };
        unsafe { std::ptr::read(&this.pinned) }
    }

    pub fn upgrade_to_shared(self) -> Result<SharedFrame<'a>, PinnedFrame<'a>> {
        let guard = match self.guard.upgrade_to_shared() {
            Ok(guard) => guard,
            Err(Restart) => return Err(self.pinned),
        };
        Ok(SharedFrame {
            pinned: self.pinned,
            guard,
        })
    }

    pub fn upgrade_to_exclusive(self) -> Result<ExclusiveFrame<'a>, PinnedFrame<'a>> {
        let guard = match self.guard.upgrade_to_exclusive() {
            Ok(guard) => guard,
            Err(Restart) => return Err(self.pinned),
        };
        Ok(ExclusiveFrame {
            pinned: self.pinned,
            guard,
        })
    }

    pub fn try_upgrade_to_exclusive(self) -> Result<ExclusiveFrame<'a>, PinnedFrame<'a>> {
        let guard = match self.guard.try_upgrade_to_exclusive() {
            Ok(guard) => guard,
            Err(Restart) => return Err(self.pinned),
        };
        Ok(ExclusiveFrame {
            pinned: self.pinned,
            guard,
        })
    }
}

/// A pinned frame plus a shared (read) latch guard.
///
/// Constructed via [`PinnedFrame::shared`] or by promoting an
/// [`OptimisticFrame`] via [`OptimisticFrame::upgrade_to_shared`]. Because a
/// real reader lock is held, no exclusive section can be entered concurrently —
/// the snapshot is frozen for the guard's lifetime. `Clone` re-acquires a fresh
/// shared lock on the same pinned frame.
///
/// ```compile_fail
/// use pagebox_storage::buffer_pool::SharedFrame;
///
/// fn escape<'pool>(frame: SharedFrame<'pool>) -> &'pool [u8] {
///     frame.page_bytes()
/// }
/// ```
pub struct SharedFrame<'a> {
    pinned: PinnedFrame<'a>,
    guard: SharedGuard<'a>,
}

impl<'a> Clone for SharedFrame<'a> {
    fn clone(&self) -> Self {
        let pinned = self.pinned.clone();
        let guard = unsafe { (&*pinned.raw()).latch.lock_shared() };
        let guard = unsafe { extend_shared_guard(guard) };
        Self { pinned, guard }
    }
}

impl<'a> SharedFrame<'a> {
    pub fn try_clone(&self) -> Option<Self> {
        let pinned = self.pinned.clone();
        let guard = unsafe { (&*pinned.raw()).latch.try_lock_shared()? };
        let guard = unsafe { extend_shared_guard(guard) };
        Some(Self { pinned, guard })
    }
}

impl<'a> SharedFrame<'a> {
    /// Return this guard's lifetime-erased frame identity.
    ///
    /// # Safety
    ///
    /// The returned reference must not outlive this guard's pin and shared
    /// latch. It is identity, not independent page-access authority.
    pub unsafe fn frame_ref(&self) -> BufferFrameRef {
        unsafe { self.pinned.frame_ref() }
    }

    pub fn read_ref(&self) -> BufferFrameReadRef<'_> {
        unsafe { self.frame_ref().read_ref() }
    }

    pub fn pid(&self) -> u64 {
        self.pinned.pid()
    }

    pub fn page(&self) -> &[u8; PAGE_SIZE] {
        unsafe { &(*self.pinned.raw()).page }
    }

    pub fn page_bytes(&self) -> &[u8] {
        unsafe { (*self.pinned.raw()).page_bytes() }
    }

    pub fn into_pinned(self) -> PinnedFrame<'a> {
        let this = std::mem::ManuallyDrop::new(self);
        let _guard = unsafe { std::ptr::read(&this.guard) };
        unsafe { std::ptr::read(&this.pinned) }
    }

    pub fn guard(&self) -> &SharedGuard<'a> {
        &self.guard
    }
}

impl Deref for SharedFrame<'_> {
    type Target = BufferFrame;

    fn deref(&self) -> &Self::Target {
        // The shared guard excludes writers for the lifetime of this borrow.
        unsafe { &*self.pinned.raw() }
    }
}

/// A pinned frame plus an exclusive (write) latch guard.
///
/// Constructed via [`PinnedFrame::exclusive`], [`PinnedFrame::try_exclusive`],
/// or by promoting an [`OptimisticFrame`] via
/// [`OptimisticFrame::upgrade_to_exclusive`]. While live the underlying
/// [`HybridLatch`](pagebox_hybrid_latch::HybridLatch) has set its exclusive
/// bit, so in-flight optimistic guards fail `validate`. Drop releases the
/// exclusive latch (advancing the base version by two and publishing the
/// write to subsequent optimistic readers) and unpins the frame.
///
/// While exclusive access is held the guard exposes its parent-link mutators
/// ([`ExclusiveFrame::set_parent_link_none`] /
/// [`ExclusiveFrame::set_parent_link_stable`]) and write-reference views onto
/// both the page and frame header.
pub struct ExclusiveFrame<'a> {
    pinned: PinnedFrame<'a>,
    guard: ExclusiveGuard<'a>,
}

/// A newly allocated stable page that has not yet been installed in an owner.
///
/// Dropping the guard aborts the unpublished allocation and returns its frame
/// capacity without entering durable page-retirement accounting.
#[must_use]
pub struct NewStablePage<'pool> {
    pool: &'pool BufferPool,
    edge: Option<StableSwip>,
    frame: Option<ExclusiveFrame<'pool>>,
}

/// A newly allocated page that has not yet been made reachable by a routing
/// edge.
///
/// The page retains a pin but no latch. Dropping it aborts the unpublished
/// allocation immediately rather than entering checkpoint-gated retirement.
/// Callers may exclusively initialise it and return to this unlatched state
/// before acquiring a parent latch.
///
/// ```compile_fail
/// use pagebox_storage::buffer_pool::{BufferPool, NoLatches};
///
/// let pool = BufferPool::new(1);
/// let page = pool.allocate_unlinked(unsafe { NoLatches::new(&pool) });
/// let untracked_pin = page.into_pinned();
/// ```
#[must_use]
pub struct NewUnlinkedPage<'pool> {
    pool: &'pool BufferPool,
    frame: Option<PinnedFrame<'pool>>,
}

/// Exclusively latched construction state for a [`NewUnlinkedPage`].
#[must_use]
pub struct NewUnlinkedExclusivePage<'pool> {
    pool: &'pool BufferPool,
    frame: Option<ExclusiveFrame<'pool>>,
}

impl NewStablePage<'_> {
    pub fn pid(&self) -> PageId {
        self.frame
            .as_ref()
            .expect("new stable page frame already consumed")
            .pid()
    }

    pub fn page_mut(&mut self) -> &mut [u8; PAGE_SIZE] {
        self.frame
            .as_mut()
            .expect("new stable page frame already consumed")
            .page_mut()
    }

    pub fn mark_dirty(&mut self) {
        self.frame
            .as_ref()
            .expect("new stable page frame already consumed")
            .mark_dirty();
    }

    /// Run page initialisation through a write view bounded by this unpublished
    /// page guard. The stable backlink is already installed and cannot be
    /// separated from the frame during the callback.
    pub fn with_write<R>(&mut self, f: impl FnOnce(&mut BufferFrameWriteRef<'_>) -> R) -> R {
        let frame = self
            .frame
            .as_mut()
            .expect("new stable page frame already consumed");
        let mut write = frame.write_ref();
        f(&mut write)
    }

    /// Install the stable edge into a pre-reserved owner slot before releasing
    /// the frame latch and pin.
    pub fn install(mut self, slot: &mut Option<StableSwip>) {
        assert!(slot.is_none(), "stable owner slot must be vacant");
        let edge = self
            .edge
            .take()
            .expect("new stable page edge already consumed");
        *slot = Some(edge);
        drop(self.frame.take());
    }
}

impl Drop for NewStablePage<'_> {
    fn drop(&mut self) {
        let Some(frame) = self.frame.take() else {
            return;
        };
        let edge = self
            .edge
            .take()
            .expect("unpublished stable frame must retain its owner edge");
        self.pool.abort_unpublished_stable_frame(frame, edge);
    }
}

impl<'pool> NewUnlinkedPage<'pool> {
    pub fn pid(&self) -> PageId {
        self.frame
            .as_ref()
            .expect("new unlinked page frame already consumed")
            .pid()
    }

    pub fn hot_swip(&self) -> Swip {
        self.frame
            .as_ref()
            .expect("new unlinked page frame already consumed")
            .hot_swip()
    }

    /// Return the unpublished page's lifetime-erased frame identity.
    ///
    /// # Safety
    /// The identity must not outlive this guard or be used as independent
    /// dereference authority.
    pub unsafe fn frame_ref(&self) -> BufferFrameRef {
        unsafe {
            self.frame
                .as_ref()
                .expect("new unlinked page frame already consumed")
                .frame_ref()
        }
    }

    pub fn exclusive(mut self) -> NewUnlinkedExclusivePage<'pool> {
        let frame = self
            .frame
            .take()
            .expect("new unlinked page frame already consumed")
            .exclusive();
        NewUnlinkedExclusivePage {
            pool: self.pool,
            frame: Some(frame),
        }
    }

    /// Finish a publication operation that was performed by a specialised
    /// structural primitive.
    ///
    /// # Safety
    /// An owning parent, root, or sibling edge must already make this page
    /// reachable, and that edge must remain valid after the returned pin is
    /// released.
    pub unsafe fn finish_publication(mut self) -> PinnedFrame<'pool> {
        self.frame
            .take()
            .expect("new unlinked page frame already consumed")
    }
}

impl Drop for NewUnlinkedPage<'_> {
    fn drop(&mut self) {
        let Some(frame) = self.frame.take() else {
            return;
        };
        self.pool
            .abort_unpublished_unlinked_frame(frame.exclusive());
    }
}

impl<'pool> NewUnlinkedExclusivePage<'pool> {
    pub fn frame(&self) -> &ExclusiveFrame<'pool> {
        self.frame
            .as_ref()
            .expect("new unlinked page frame already consumed")
    }

    pub fn frame_mut(&mut self) -> &mut ExclusiveFrame<'pool> {
        self.frame
            .as_mut()
            .expect("new unlinked page frame already consumed")
    }

    pub fn into_unlatched(mut self) -> NewUnlinkedPage<'pool> {
        let frame = self
            .frame
            .take()
            .expect("new unlinked page frame already consumed")
            .into_pinned();
        NewUnlinkedPage {
            pool: self.pool,
            frame: Some(frame),
        }
    }

    /// Escape into the raw exclusive-frame protocol for a specialised,
    /// atomic publication operation such as B+tree root transfer.
    ///
    /// # Safety
    /// The caller must either make the page reachable before releasing the
    /// returned frame or abort/retire it on every failure path. Ordinary
    /// split-page publication must retain the typestate guard until its edge
    /// is installed.
    pub unsafe fn into_exclusive_frame(mut self) -> ExclusiveFrame<'pool> {
        self.frame
            .take()
            .expect("new unlinked page frame already consumed")
    }
}

impl Drop for NewUnlinkedExclusivePage<'_> {
    fn drop(&mut self) {
        let Some(frame) = self.frame.take() else {
            return;
        };
        self.pool.abort_unpublished_unlinked_frame(frame);
    }
}

impl<'a> ExclusiveFrame<'a> {
    /// # Safety
    ///
    /// The caller must ensure `pinned` and `guard` refer to the same frame and
    /// that `guard` provides exclusive access for the lifetime `'a`.
    pub unsafe fn from_parts(pinned: PinnedFrame<'a>, guard: ExclusiveGuard<'a>) -> Self {
        Self { pinned, guard }
    }

    fn raw(&self) -> *mut BufferFrame {
        self.pinned.raw()
    }

    /// Return this guard's lifetime-erased frame identity.
    ///
    /// # Safety
    ///
    /// The returned reference must not outlive this guard's pin and exclusive
    /// latch. It is identity, not independent page-access authority.
    pub unsafe fn frame_ref(&self) -> BufferFrameRef {
        unsafe { self.pinned.frame_ref() }
    }

    pub fn read_ref(&self) -> BufferFrameReadRef<'_> {
        unsafe { self.frame_ref().read_ref() }
    }

    pub fn write_ref(&mut self) -> BufferFrameWriteRef<'_> {
        unsafe { self.frame_ref().write_ref() }
    }

    pub fn pid(&self) -> u64 {
        self.pinned.pid()
    }

    pub fn hot_swip(&self) -> Swip {
        self.pinned.hot_swip()
    }

    pub fn page(&self) -> &[u8; PAGE_SIZE] {
        unsafe { &(*self.pinned.raw()).page }
    }

    pub fn page_bytes(&self) -> &[u8] {
        unsafe { (*self.pinned.raw()).page_bytes() }
    }

    pub fn page_bytes_mut(&mut self) -> &mut [u8] {
        let bf = self.raw();
        unsafe { (*bf).page_bytes_mut() }
    }

    pub fn page_mut(&mut self) -> &mut [u8; PAGE_SIZE] {
        let bf = self.raw();
        unsafe { &mut (*bf).page }
    }

    pub fn set_parent_link_none(&mut self) {
        let bf = self.raw();
        unsafe { (*bf).header.parent_link = ParentLink::None };
    }

    pub(crate) fn set_parent_link_stable(&mut self, swip: &StableSwip) {
        let bf = self.raw();
        unsafe { (*bf).header.parent_link = ParentLink::Stable(Arc::clone(swip.owner())) };
    }

    pub fn guard(&self) -> &ExclusiveGuard<'a> {
        &self.guard
    }

    pub fn mark_dirty(&self) {
        self.pinned.mark_dirty();
    }

    pub fn mark_dirty_with_lsn(&self, lsn: Lsn) {
        self.pinned.mark_dirty_with_lsn(lsn);
    }

    pub fn mark_dirty_patch(&self, offset: usize, data: &[u8]) {
        unsafe {
            self.pinned
                .pool
                .mark_dirty_patch_raw(self.raw(), offset, data)
        };
    }

    pub fn into_pinned(self) -> PinnedFrame<'a> {
        let this = std::mem::ManuallyDrop::new(self);
        let _guard = unsafe { std::ptr::read(&this.guard) };
        unsafe { std::ptr::read(&this.pinned) }
    }
}

impl Deref for ExclusiveFrame<'_> {
    type Target = BufferFrame;

    fn deref(&self) -> &Self::Target {
        // The exclusive guard provides sole frame access for this borrow.
        unsafe { &*self.pinned.raw() }
    }
}

impl<'a> PinnedFrame<'a> {
    pub fn clone_pin(&self) -> PinnedFrame<'a> {
        unsafe {
            (*self.bf)
                .header
                .core
                .pin_count
                .fetch_add(1, Ordering::Relaxed)
        };
        unsafe { PinnedFrame::new(self.pool, self.bf) }
    }

    pub fn optimistic(self) -> Result<OptimisticFrame<'a>, PinnedFrame<'a>> {
        let guard = match unsafe { (&*self.bf).latch.optimistic_or_restart() } {
            Ok(guard) => unsafe { extend_optimistic_guard(guard) },
            Err(Restart) => return Err(self),
        };
        Ok(OptimisticFrame {
            pinned: self,
            guard,
        })
    }

    #[track_caller]
    pub fn shared(self) -> SharedFrame<'a> {
        let guard = unsafe { (&*self.bf).latch.lock_shared() };
        let guard = unsafe { extend_shared_guard(guard) };
        SharedFrame {
            pinned: self,
            guard,
        }
    }

    #[track_caller]
    pub fn exclusive(self) -> ExclusiveFrame<'a> {
        let guard = unsafe { (&*self.bf).latch.lock_exclusive() };
        let guard = unsafe { extend_exclusive_guard(guard) };
        ExclusiveFrame {
            pinned: self,
            guard,
        }
    }

    #[track_caller]
    pub fn try_exclusive(self) -> Result<ExclusiveFrame<'a>, PinnedFrame<'a>> {
        let Some(guard) = (unsafe { (&*self.bf).latch.try_lock_exclusive() }) else {
            return Err(self);
        };
        let guard = unsafe { extend_exclusive_guard(guard) };
        Ok(ExclusiveFrame {
            pinned: self,
            guard,
        })
    }
}

// Thread-local shard hint for allocation.
thread_local! {
    static ALLOC_SHARD_HINT: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

static NEXT_ALLOC_SHARD_HINT: AtomicUsize = AtomicUsize::new(0);

fn thread_alloc_shard_hint() -> usize {
    ALLOC_SHARD_HINT.with(|cell| {
        let hint = cell.get();
        if hint != usize::MAX {
            return hint;
        }
        let hint = NEXT_ALLOC_SHARD_HINT.fetch_add(1, Ordering::Relaxed);
        cell.set(hint);
        hint
    })
}

impl BufferPool {
    fn yield_for_contention() {
        #[cfg(not(loom))]
        std::thread::yield_now();
        #[cfg(loom)]
        loom::thread::yield_now();
    }

    fn fix_frame_backoff(attempts: u32) {
        if attempts < 16 {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }

    fn with_fix_orphan_hot_pin<T>(&self, f: impl FnOnce() -> T) -> T {
        let _pin_guard = loop {
            if let Some(guard) = self.try_lock_hot_pin() {
                break guard;
            }
            self.metrics
                .fix_orphan_events
                .inc(BufferPoolFixOrphanEvent::HotPinWait);
            Self::yield_for_contention();
        };
        f()
    }

    fn record_fix_orphan_latch_wait(&self, bf: *mut BufferFrame, page_id: PageId) {
        self.metrics
            .fix_orphan_events
            .inc(BufferPoolFixOrphanEvent::LatchWait);
        let state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
        match state {
            FrameState::Free => self
                .metrics
                .fix_orphan_events
                .inc(BufferPoolFixOrphanEvent::LatchWaitFree),
            FrameState::Loading => self
                .metrics
                .fix_orphan_events
                .inc(BufferPoolFixOrphanEvent::LatchWaitLoading),
            FrameState::Resident => self
                .metrics
                .fix_orphan_events
                .inc(BufferPoolFixOrphanEvent::LatchWaitResident),
            _ => self
                .metrics
                .fix_orphan_events
                .inc(BufferPoolFixOrphanEvent::LatchWaitOther),
        }
        self.sample_fix_orphan_latch_wait_page(page_id);
        Self::yield_for_contention();
    }

    fn with_fix_orphan_exclusive_at<T>(
        &self,
        bf: *mut BufferFrame,
        page_id: PageId,
        f: impl FnOnce() -> T,
    ) -> T {
        let _guard = loop {
            if let Some(guard) = unsafe { self.try_lock_frame_exclusive_at(bf, page_id) } {
                break guard;
            }
            self.record_fix_orphan_latch_wait(bf, page_id);
        };
        f()
    }

    fn wait_for_hot_frame_transition(
        &self,
        swip: &AtomicSwip,
        expected: Swip,
        bf: *mut BufferFrame,
    ) {
        let start = Instant::now();
        for attempts in 0..64 {
            if swip.load(Ordering::Acquire).raw() != expected.raw() {
                self.record_hot_frame_transition_wait(start.elapsed());
                return;
            }
            let state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
            if state == FrameState::Resident {
                self.record_hot_frame_transition_wait(start.elapsed());
                return;
            }
            Self::fix_frame_backoff(attempts);
        }
        self.record_hot_frame_transition_wait(start.elapsed());
    }

    fn wait_for_loading_frame_transition(&self, bf: *mut BufferFrame) {
        let start = Instant::now();
        let page_id = unsafe { (*bf).header.core.pid };
        self.enter_loading_frame_wait(page_id);
        for attempts in 0..64 {
            let state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
            if state != FrameState::Loading {
                self.exit_loading_frame_wait(page_id);
                self.record_loading_frame_transition_wait(start.elapsed());
                return;
            }
            Self::fix_frame_backoff(attempts);
        }
        while unsafe { (*bf).header.core.state.load(Ordering::Acquire) } == FrameState::Loading {
            #[cfg(not(loom))]
            std::thread::yield_now();
            #[cfg(loom)]
            loom::thread::yield_now();
        }
        self.exit_loading_frame_wait(page_id);
        self.record_loading_frame_transition_wait(start.elapsed());
    }

    fn try_repair_nonresident_hot_swip(
        &self,
        swip: &AtomicSwip,
        expected: Swip,
        bf: *mut BufferFrame,
        state: FrameState,
    ) -> bool {
        if state != FrameState::Free {
            return false;
        }
        let page_id = unsafe { (*bf).header.core.pid };
        if page_id == 0 {
            return false;
        }
        swip.compare_exchange(
            expected,
            Swip::evicted(page_id),
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_ok()
    }

    fn mark_referenced(&self, bf: *mut BufferFrame) {
        unsafe { (*bf).header.core.referenced.store(true, Ordering::Relaxed) };
    }

    fn arena(&self) -> &Arena {
        &self.state.arena
    }

    fn allocated_slots(&self) -> usize {
        // The full arena is pre-constructed at init.
        self.arena().len
    }

    /// Pin a child frame referenced by `swip`, handling HOT/COOL/EVICTED
    /// resolution in one place.
    ///
    /// # Safety
    /// `swip` must come from a valid page routing edge managed by this pool.
    pub unsafe fn pin_child(
        &self,
        swip: Swip,
        _no_latches: NoLatches<'_>,
    ) -> Option<PinnedFrame<'_>> {
        unsafe { self.pin_child_internal(swip, false) }
    }

    /// Pin a child frame without blocking on orphan page faults.
    ///
    /// Returns `None` when the child is not immediately pinnable,
    /// allowing read-only traversal paths to restart instead of
    /// spinning in `fix_orphan()`.
    ///
    /// # Safety
    /// `swip` must come from a valid page routing edge managed by this pool.
    pub unsafe fn try_pin_child(&self, swip: Swip) -> Option<PinnedFrame<'_>> {
        if swip.raw() == 0 {
            return None;
        }
        let bf = if swip.is_hot() || swip.is_cool() {
            let _pin_guard = self.lock_hot_pin();
            unsafe { self.try_pin_hot_or_cool_swip(swip) }?
        } else {
            unsafe { self.try_fix_orphan_raw(swip.as_page_id()) }?
        };
        Some(unsafe { PinnedFrame::new(self, bf) })
    }

    /// Pin a child only when its routing edge already names a resident frame.
    ///
    /// Unlike [`BufferPool::try_pin_child`], this method never resolves an
    /// EVICTED SWIP through the page table and never starts a page-store read.
    /// It is intended for structural operations that already hold another
    /// frame latch and must restart instead of blocking on I/O.
    ///
    /// # Safety
    /// `swip` must come from a valid page routing edge managed by this pool.
    pub unsafe fn try_pin_resident_child(&self, swip: Swip) -> Option<PinnedFrame<'_>> {
        if !(swip.is_hot() || swip.is_cool()) {
            return None;
        }
        let _pin_guard = self.lock_hot_pin();
        let bf = unsafe { self.try_pin_hot_or_cool_swip(swip) }?;
        Some(unsafe { PinnedFrame::new(self, bf) })
    }

    /// Pin a child frame while the caller already holds `eviction_mu.write()`.
    ///
    /// # Safety
    /// `swip` must come from a valid page routing edge managed by this pool.
    /// The caller must already hold the eviction write lock.
    pub unsafe fn pin_child_during_eviction(&self, swip: Swip) -> Option<PinnedFrame<'_>> {
        unsafe { self.pin_child_internal(swip, true) }
    }

    /// # Safety
    /// `swip` must come from a valid page routing edge managed by this pool.
    /// When `skip_hot_pin_lock` is true, the caller must already hold
    /// `eviction_mu.write()`.
    unsafe fn pin_child_internal(
        &self,
        swip: Swip,
        skip_hot_pin_lock: bool,
    ) -> Option<PinnedFrame<'_>> {
        if swip.raw() == 0 {
            return None;
        }
        let bf = if swip.is_hot() || swip.is_cool() {
            unsafe { self.pin_hot_or_cool_child_swip(swip, skip_hot_pin_lock) }?
        } else {
            unsafe { self.fix_orphan_raw(swip.as_page_id()) }
        };
        Some(unsafe { PinnedFrame::new(self, bf) })
    }

    unsafe fn pin_hot_or_cool_child_swip(
        &self,
        swip: Swip,
        skip_hot_pin_lock: bool,
    ) -> Option<*mut BufferFrame> {
        let bf = unsafe { swip.as_ptr::<BufferFrame>() };
        if !self.contains_frame_ptr(bf) {
            return None;
        }

        let mut attempts = 0u32;
        loop {
            let pinned = {
                let _pin_guard = (!skip_hot_pin_lock).then(|| self.lock_hot_pin());
                if !self.contains_frame_ptr(bf) {
                    return None;
                }
                unsafe { (*bf).header.core.pin_count.fetch_add(1, Ordering::Relaxed) };
                let state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
                if state == FrameState::Resident {
                    Some(bf)
                } else {
                    unsafe { (*bf).header.core.pin_count.fetch_sub(1, Ordering::Relaxed) };
                    None
                }
            };
            if let Some(bf) = pinned {
                return Some(bf);
            }

            let state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
            if state == FrameState::Evicting {
                let page_id = unsafe { (*bf).header.core.pid };
                if page_id != 0 && unsafe { self.try_rescue_evicting_orphan(bf, page_id) } {
                    attempts = 0;
                    continue;
                }
            }
            if state == FrameState::Loading {
                self.wait_for_loading_frame_transition(bf);
                attempts = 0;
                continue;
            }
            if state == FrameState::Free {
                return None;
            }

            attempts = attempts.saturating_add(1);
            Self::fix_frame_backoff(attempts);
        }
    }

    unsafe fn try_pin_hot_or_cool_swip(&self, swip: Swip) -> Option<*mut BufferFrame> {
        let bf = unsafe { swip.as_ptr::<BufferFrame>() };
        debug_assert!(
            self.contains_frame_ptr(bf),
            "pool.try_pin_hot_or_cool_swip: stale HOT/COOL pointer: raw={:#x} ptr={:#x}",
            swip.raw(),
            bf as usize
        );
        if !self.contains_frame_ptr(bf) {
            return None;
        }

        unsafe { (*bf).header.core.pin_count.fetch_add(1, Ordering::Relaxed) };
        let state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
        if state != FrameState::Resident {
            unsafe { (*bf).header.core.pin_count.fetch_sub(1, Ordering::Relaxed) };
            return None;
        }

        Some(bf)
    }

    pub fn fix_stable<'a>(
        &'a self,
        swip: &StableSwip,
        _no_latches: NoLatches<'_>,
    ) -> PinnedFrame<'a> {
        self.assert_stable_provenance(swip);
        unsafe { PinnedFrame::new(self, self.fix_stable_raw(swip.owner())) }
    }

    fn assert_stable_provenance(&self, swip: &StableSwip) {
        assert_eq!(
            swip.pool_id(),
            self.id,
            "stable SWIP belongs to a different buffer pool"
        );
    }

    pub fn with_fixed_stable_exclusive<T>(
        &self,
        swip: &StableSwip,
        no_latches: NoLatches<'_>,
        f: impl FnOnce(&mut ExclusiveFrame<'_>) -> T,
    ) -> T {
        let mut frame = self.fix_stable(swip, no_latches).exclusive();
        f(&mut frame)
    }

    pub fn mark_dirty_frame(&self, frame: BufferFrameWriteRef<'_>) {
        unsafe { self.mark_dirty_raw(frame.frame().as_ptr()) };
    }

    pub fn contains_frame(&self, frame: BufferFrameRef) -> bool {
        self.contains_frame_ptr(frame.as_ptr())
    }

    pub fn contains_hot_or_cool_swip_frame(&self, swip: Swip) -> bool {
        if !(swip.is_hot() || swip.is_cool()) {
            return false;
        }
        let bf = unsafe { swip.as_ptr::<BufferFrame>() };
        self.contains_frame_ptr(bf)
    }

    /// Returns `None` instead of blocking when the stable edge's frame is
    /// currently contested or not already resident.
    unsafe fn try_fix_stable_word<'a>(&'a self, swip: &AtomicSwip) -> Option<PinnedFrame<'a>> {
        let s = swip.load(Ordering::Acquire);
        if s.is_hot() || s.is_cool() {
            let bf = unsafe { s.as_ptr::<BufferFrame>() };
            debug_assert!(
                self.contains_frame_ptr(bf),
                "pool.try_fix_frame: stale HOT/COOL pointer: raw={:#x} ptr={:#x}",
                s.raw(),
                bf as usize,
            );
            if !self.contains_frame_ptr(bf) {
                return None;
            }
            let pre_state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
            if pre_state != FrameState::Resident {
                let current = swip.load(Ordering::Acquire);
                if current.raw() == s.raw() {
                    let _ = self.try_repair_nonresident_hot_swip(swip, s, bf, pre_state);
                }
                return None;
            }
            let _pin_guard = self.try_lock_hot_pin()?;
            unsafe { (*bf).header.core.pin_count.fetch_add(1, Ordering::Relaxed) };
            let current = swip.load(Ordering::Acquire);
            let state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
            if current.raw() == s.raw() && state == FrameState::Resident {
                return Some(unsafe { PinnedFrame::new(self, bf) });
            }
            unsafe { (*bf).header.core.pin_count.fetch_sub(1, Ordering::Relaxed) };
            if current.raw() == s.raw() {
                let _ = self.try_repair_nonresident_hot_swip(swip, s, bf, state);
            }
            return None;
        }
        unsafe { self.try_fix_resident_page(s.as_page_id()) }
            .map(|bf| unsafe { PinnedFrame::new(self, bf) })
    }

    /// Pin a stable edge only if its page is already resident.
    ///
    /// This never performs storage I/O and therefore needs no [`NoLatches`]
    /// witness. The HOT/COOL path borrows the stable owner without modifying
    /// its `Arc` count.
    pub fn try_fix_stable<'a>(&'a self, swip: &StableSwip) -> Option<PinnedFrame<'a>> {
        self.assert_stable_provenance(swip);
        unsafe { self.try_fix_stable_word(swip.word()) }
    }

    /// # Safety
    /// `page_id` must refer to a valid allocated page.
    pub unsafe fn fix_orphan_frame<'a>(
        &'a self,
        page_id: u64,
        _no_latches: NoLatches<'_>,
    ) -> PinnedFrame<'a> {
        unsafe { PinnedFrame::new(self, self.fix_orphan_raw(page_id)) }
    }

    /// # Safety
    /// `page_id` must refer to a valid allocated page.
    pub unsafe fn try_fix_orphan_frame<'a>(&'a self, page_id: u64) -> Option<PinnedFrame<'a>> {
        let bf = unsafe { self.try_fix_orphan_raw(page_id) }?;
        Some(unsafe { PinnedFrame::new(self, bf) })
    }

    /// Try to pin an already-resident page by page id without faulting it in.
    ///
    /// This is for data-structure parent-link fast paths during eviction. If
    /// the hinted parent is not resident or is currently transitioning, callers
    /// should report that the parent could not be updated and let eviction retry
    /// or use a broader fallback.
    ///
    /// # Safety
    /// `page_id` must refer to a valid allocated page in this buffer pool.
    pub unsafe fn try_fix_resident_page_frame<'a>(
        &'a self,
        page_id: u64,
    ) -> Option<PinnedFrame<'a>> {
        let bf = unsafe { self.try_fix_resident_page(page_id) }?;
        Some(unsafe { PinnedFrame::new(self, bf) })
    }

    fn raw_frame(&self, idx: usize) -> *mut BufferFrame {
        let arena = self.arena();
        debug_assert!(idx < arena.len);
        unsafe { (arena.ptr as *mut u8).add(idx * arena.frame_stride) as *mut BufferFrame }
    }

    fn is_slot_initialized(&self, _idx: usize) -> bool {
        // All frames are pre-constructed at init; every slot is initialized.
        true
    }

    fn try_reserve_resident_budget(&self) -> bool {
        // With a single page class, each frame consumes one base page.
        let mut available = self.resident_base_pages_available.load(Ordering::Relaxed);
        while available >= 1 {
            match self.resident_base_pages_available.compare_exchange_weak(
                available,
                available - 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(current) => available = current,
            }
        }
        false
    }

    fn flush_dirty_for_capacity(&self, max_batch: usize) -> usize {
        self.try_flush_dirty_batch(max_batch)
            .unwrap_or_else(|error| {
                panic!("buffer pool dirty flush failed while waiting for frame capacity: {error}")
            })
    }

    fn wait_for_resident_budget(&self) {
        if self.resident_base_pages_available.load(Ordering::Relaxed) >= 1 {
            return;
        }
        // When the background page provider is enabled, it handles eviction
        // and dirty-page flushing. Workers just wake it and wait.
        // Otherwise, workers do inline eviction.
        let frames_available = if background_page_provider_enabled() {
            let pp = self.page_provider.lock().unwrap();
            pp.need_frames_notify();
            Some(pp.frames_available.clone())
        } else {
            None
        };

        let mut stalled_rounds = 0u32;
        loop {
            if self.resident_base_pages_available.load(Ordering::Relaxed) >= 1 {
                return;
            }
            stalled_rounds += 1;
            if stalled_rounds >= FRAME_CAPACITY_STALL_LIMIT {
                self.panic_pool_exhausted("resident_budget", stalled_rounds);
            }

            // Inline eviction (primary path when bg provider is off,
            // safety valve when it's on).
            if (self.try_evict_any_policy(1) > 0 || self.try_evict_any_batch(1) > 0)
                && self.resident_base_pages_available.load(Ordering::Relaxed) >= 1
            {
                return;
            }

            if self.flush_dirty_for_capacity(64) > 0 {
                continue;
            }

            if let Some(frames_available) = &frames_available {
                let guard = frames_available.0.lock().unwrap();
                let _ = frames_available
                    .1
                    .wait_timeout(guard, Duration::from_micros(100));
            } else {
                #[cfg(not(loom))]
                std::thread::yield_now();
                #[cfg(loom)]
                loom::thread::yield_now();
            }
        }
    }

    fn reserve_resident_budget(&self) {
        loop {
            if self.try_reserve_resident_budget() {
                return;
            }
            self.wait_for_resident_budget();
        }
    }

    /// Pop a free frame from the FreeList. If empty, evicts to replenish.
    /// Panics if all frames are pinned (pool exhausted).
    fn pop_free_frame(&self) -> *mut BufferFrame {
        if let Some(bf) = self.state.free_list.try_pop() {
            return self.claim_free_frame(bf);
        }
        let mut backoff = Backoff::new();
        let mut stalled_rounds = 0u32;
        loop {
            if let Some(bf) = self.state.free_list.try_pop() {
                return self.claim_free_frame(bf);
            }
            stalled_rounds += 1;
            if stalled_rounds >= FRAME_CAPACITY_STALL_LIMIT {
                self.panic_pool_exhausted("free_frame", stalled_rounds);
            }

            // Notify the background page provider that we need frames.
            // It will evict clean pages and flush dirty ones in the
            // background, replenishing the free list without blocking
            // the caller.
            if stalled_rounds == 1 {
                let pp = self.page_provider.lock().unwrap();
                pp.need_frames_notify();
            }

            // Backpressure: if all eviction permits are taken, other
            // threads are already evicting. Wait for them to complete
            // rather than competing for the same frames.
            let in_flight = self.eviction_in_flight.load(Ordering::Acquire);
            if in_flight >= self.eviction_budget {
                backoff.snooze();
                continue;
            }

            let evicted = self.try_evict_any_policy(1) > 0 || self.try_evict_any_batch(1) > 0;
            if evicted {
                // A frame was released globally, but another waiter can take
                // it before this caller reaches the free list. Reset only the
                // scheduling delay; this allocation request remains stalled.
                backoff.reset();
            } else {
                // No frame was evictable — all are referenced, pinned,
                // or latched. Back off to let latch-holders release.
                if self.flush_dirty_for_capacity(64) > 0 {
                    backoff.reset();
                } else {
                    backoff.snooze();
                }
            }
        }
    }

    fn claim_free_frame(&self, bf: *mut BufferFrame) -> *mut BufferFrame {
        let generation = unsafe { &(*bf).header.generation };
        generation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .expect("buffer-frame generation exhausted");
        bf
    }

    unsafe fn try_fix_resident_page(&self, page_id: u64) -> Option<*mut BufferFrame> {
        let _pin_guard = self.try_lock_hot_pin()?;
        let bf = self.page_table.lookup(page_id)?;
        let state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
        if state != FrameState::Resident {
            return None;
        }
        let Ok(guard) = (unsafe { (*bf).latch.optimistic_or_restart() }) else {
            return None;
        };
        if unsafe { (*bf).header.core.state.load(Ordering::Acquire) } != FrameState::Resident {
            return None;
        }
        unsafe { (*bf).header.core.pin_count.fetch_add(1, Ordering::Relaxed) };
        if guard.validate().is_err() {
            unsafe { (*bf).header.core.pin_count.fetch_sub(1, Ordering::Relaxed) };
            return None;
        }
        Some(bf)
    }

    unsafe fn try_rescue_evicting_orphan(&self, bf: *mut BufferFrame, page_id: u64) -> bool {
        let _guard = match unsafe { self.try_lock_frame_exclusive_at(bf, page_id) } {
            Some(guard) => guard,
            None => return false,
        };
        let state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
        if state != FrameState::Evicting {
            return false;
        }
        if unsafe { (*bf).header.core.pin_count.load(Ordering::Acquire) } != 0
            || unsafe { (*bf).header.core.pid } != page_id
        {
            return false;
        }
        unsafe {
            (*bf).header.core.referenced.store(true, Ordering::Relaxed);
            (*bf).header.core.state.transition(
                FrameState::Loading,
                FrameState::Resident,
                Ordering::Release,
            );
        }
        true
    }

    /// If `swip` is a stable edge and `bf` is already resident for the
    /// referenced `page_id`, try to re-swizzle the edge back to HOT so
    /// future traversals avoid the resident-page scan path.
    ///
    /// Caller must already hold the slot's exclusive latch.
    unsafe fn reswizzle_stable_resident_locked(
        &self,
        swip: &AtomicSwip,
        owner: &Arc<StableSwipOwner>,
        expected: Swip,
        bf: *mut BufferFrame,
        page_id: u64,
    ) {
        if unsafe { (*bf).header.core.state.load(Ordering::Acquire) } != FrameState::Resident
            || unsafe { (*bf).header.core.pid } != page_id
        {
            return;
        }
        match unsafe { &(*bf).header.parent_link } {
            ParentLink::None | ParentLink::Unswizzled => {}
            ParentLink::Stable(edge) if Arc::ptr_eq(edge, owner) => {}
            _ => return,
        }
        if swip
            .compare_exchange(expected, Swip::hot(bf), Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            unsafe { (*bf).header.parent_link = ParentLink::Stable(Arc::clone(owner)) };
        }
    }

    fn contains_frame_ptr(&self, bf: *mut BufferFrame) -> bool {
        let addr = bf as usize;
        let arena = self.arena();
        let base = arena.ptr as usize;
        let byte_len = arena.len.saturating_mul(arena.frame_stride);
        addr >= base
            && addr < base.saturating_add(byte_len)
            && (addr - base).is_multiple_of(arena.frame_stride)
    }

    fn frame_id(&self, bf: *mut BufferFrame) -> FrameId {
        assert!(
            self.contains_frame_ptr(bf),
            "frame does not belong to this pool"
        );
        let arena = self.arena();
        let slot = ((bf as usize - arena.ptr as usize) / arena.frame_stride) as u32;
        let generation = unsafe { (*bf).header.generation.load(Ordering::Acquire) };
        FrameId::new(slot, generation)
    }

    pub fn new(num_frames: usize) -> Self {
        Self::with_store(num_frames, Box::new(InMemoryPageStore::new()))
    }

    pub fn with_store(num_frames: usize, page_store: Box<dyn PageStore>) -> Self {
        let start_pid = page_store.next_page_id();
        Self::with_store_from_start_pid(num_frames, page_store, start_pid)
    }

    fn with_store_from_start_pid(
        num_frames: usize,
        page_store: Box<dyn PageStore>,
        start_pid: PageId,
    ) -> Self {
        let num_shards = num_cpus().max(1);
        let frame_count = num_frames.max(1);
        let state = PoolState::new(frame_count);

        let pool = BufferPool {
            id: allocate_pool_id(),
            self_weak: OnceLock::new(),
            state,
            page_table: PageTable::new(),
            page_store,
            next_page_id: AtomicU64::new(start_pid),
            free_page_allocator: FreePageAllocator::new(start_pid, num_shards),
            resident_base_pages: num_frames,
            resident_base_pages_available: AtomicUsize::new(num_frames),
            #[cfg(not(miri))]
            wal: None,
            prefetch_workers: std::sync::Mutex::new(PrefetchWorkers::new()),
            prefetch_inflight: parking_lot::Mutex::new(HashSet::new()),
            metrics: BufferPoolMetrics::new(num_shards),
            loading_frame_wait_peak_pages: parking_lot::Mutex::new(HashMap::new()),
            loading_frame_wait_current_pages: parking_lot::Mutex::new(HashMap::new()),
            fix_orphan_latch_wait_sample_clock: AtomicU64::new(0),
            fix_orphan_latch_wait_sampled_pages: parking_lot::Mutex::new(HashMap::new()),
            fix_orphan_evicting_retry_sample_clock: AtomicU64::new(0),
            fix_orphan_evicting_retry_sampled_pages: parking_lot::Mutex::new(HashMap::new()),
            dt_registry: parking_lot::Mutex::new(HashMap::new()),
            eviction_mu: parking_lot::RwLock::new(()),
            eviction_writer_pending: AtomicUsize::new(0),
            eviction_budget: (num_frames / 4).max(1),
            eviction_in_flight: AtomicUsize::new(0),
            dirty_flush_inflight: parking_lot::Mutex::new(HashSet::new()),
            page_reclaimers: parking_lot::Mutex::new(HashMap::new()),
            page_writeback_preparers: parking_lot::Mutex::new(HashMap::new()),
            pending_reusable_extents: parking_lot::Mutex::new(Vec::new()),
            page_provider: std::sync::Mutex::new(page_provider::PageProviderHandle::new()),
        };

        // Pre-construct frames for all slots and seed the FreeList.
        // This replaces the lazy slot-init mechanism: frames are recycled
        // via the FreeList rather than addressed by page ID.
        let arena = &pool.state.arena;
        for idx in 0..arena.len {
            let bf =
                unsafe { (arena.ptr as *mut u8).add(idx * arena.frame_stride) as *mut BufferFrame };
            unsafe { bf.write(BufferFrame::new()) };
            pool.state.slot_init[idx].store(2, Ordering::Release);
            pool.state.free_list.push(bf);
        }

        pool
    }

    pub fn prefetch_inflight_len(&self) -> usize {
        self.prefetch_inflight.lock().len()
    }

    pub fn visit_metrics<V: MetricVisitor + ?Sized>(&self, visitor: &mut V) {
        self.update_sampled_metrics();
        self.metrics.visit_metrics(visitor);
    }

    fn update_sampled_metrics(&self) {
        let counts = self.current_frame_state_counts();
        let occupied = counts
            .resident
            .saturating_add(counts.loading)
            .saturating_add(counts.evicting);
        let free = self.resident_base_pages.saturating_sub(occupied);
        self.metrics
            .frames_total
            .set(saturating_usize_to_i64(self.resident_base_pages));
        self.metrics
            .frames_occupied
            .set(saturating_usize_to_i64(occupied));
        self.metrics
            .frame_state_frames
            .set(BufferPoolFrameState::Free, saturating_usize_to_i64(free));
        self.metrics.frame_state_frames.set(
            BufferPoolFrameState::Resident,
            saturating_usize_to_i64(counts.resident),
        );
        self.metrics.frame_state_frames.set(
            BufferPoolFrameState::Loading,
            saturating_usize_to_i64(counts.loading),
        );
        self.metrics.frame_state_frames.set(
            BufferPoolFrameState::Evicting,
            saturating_usize_to_i64(counts.evicting),
        );
        self.metrics
            .resident_budget_available
            .set(saturating_usize_to_i64(
                self.resident_base_pages_available.load(Ordering::Relaxed),
            ));
        self.metrics
            .simple_prefetch_inflight
            .set(saturating_usize_to_i64(self.prefetch_inflight_len()));
        self.metrics
            .pages_on_disk
            .set(saturating_usize_to_i64(self.page_store.len()));
    }

    fn current_frame_state_counts(&self) -> BufferPoolFrameStateCounts {
        let mut counts = BufferPoolFrameStateCounts::default();
        for i in 0..self.allocated_slots() {
            if !self.is_slot_initialized(i) {
                continue;
            }
            let bf = self.raw_frame(i);
            match unsafe { (*bf).header.core.state.load(Ordering::Relaxed) } {
                FrameState::Free => {}
                FrameState::Resident => counts.resident += 1,
                FrameState::Loading => counts.loading += 1,
                FrameState::Evicting => counts.evicting += 1,
            }
        }
        counts
    }

    /// Register a parent finder for a data structure ID.
    pub fn register_dt(&self, dt_id: u16, finder: Arc<dyn ParentFinder>) {
        self.dt_registry.lock().insert(dt_id, finder);
    }

    /// Remove a parent finder. Used by `DROP TABLE` so that subsequent
    /// eviction work doesn't try to walk parent pointers in a tree whose
    /// owning Table struct has been destroyed.
    pub fn unregister_dt(&self, dt_id: u16) {
        self.dt_registry.lock().remove(&dt_id);
    }

    /// Whether a data structure currently has an eviction parent finder.
    pub fn has_registered_dt(&self, dt_id: u16) -> bool {
        self.dt_registry.lock().contains_key(&dt_id)
    }

    pub fn register_page_reclaimer(&self, page_pid: u64, reclaimer: Arc<dyn PageReclaimer>) {
        self.page_reclaimers.lock().insert(page_pid, reclaimer);
    }

    pub fn unregister_page_reclaimer(&self, page_pid: u64) {
        self.page_reclaimers.lock().remove(&page_pid);
    }

    pub fn register_page_writeback_preparer(
        &self,
        page_type: PageType,
        preparer: Arc<dyn PageWritebackPreparer>,
    ) {
        self.page_writeback_preparers
            .lock()
            .insert(page_type, preparer);
    }

    #[track_caller]
    unsafe fn try_lock_frame_exclusive_at<'a>(
        &self,
        bf: *mut BufferFrame,
        _page_id_hint: PageId,
    ) -> Option<ExclusiveGuard<'a>> {
        unsafe { (*bf).latch.try_lock_exclusive() }
    }

    #[track_caller]
    unsafe fn lock_frame_exclusive_at<'a>(
        &self,
        bf: *mut BufferFrame,
        _page_id_hint: PageId,
    ) -> ExclusiveGuard<'a> {
        unsafe { (*bf).latch.lock_exclusive() }
    }

    fn prefetch_inflight_contains(&self, pid: PageId) -> bool {
        self.prefetch_inflight.lock().contains(&pid)
    }

    #[allow(dead_code)]
    fn prefetch_inflight_take(&self, pid: PageId) -> bool {
        self.prefetch_inflight.lock().remove(&pid)
    }

    fn prefetch_inflight_remove(&self, pid: PageId) {
        self.prefetch_inflight.lock().remove(&pid);
    }

    unsafe fn prepare_orphan_loading_frame(
        &self,
        bf: *mut BufferFrame,
        page_id: PageId,
    ) -> LoadingFrameReservation<'_> {
        unsafe {
            (*bf).header.core.pid = page_id;
            (*bf).header.parent_link = ParentLink::None;
            (*bf).header.core.referenced.store(true, Ordering::Relaxed);
            (*bf).header.core.state.transition(
                FrameState::Free,
                FrameState::Loading,
                Ordering::Release,
            );
        }
        LoadingFrameReservation::new(self, bf)
    }

    fn try_reclaim_before_evict(&self, page_pid: u64, page_bf: *mut BufferFrame) {
        let reclaimer = self.page_reclaimers.lock().get(&page_pid).cloned();
        if let Some(reclaimer) = reclaimer {
            let page = unsafe { BufferFrameRef::from_raw(page_bf).write_ref() };
            reclaimer.try_reclaim_before_evict(page_pid, page);
        }
    }

    /// Attach a WAL to this buffer pool. When set, dirty pages are logged
    /// to the WAL before being written to the data file (WAL protocol).
    #[cfg(not(miri))]
    pub fn set_wal(&mut self, wal: Arc<Wal>) {
        self.wal = Some(wal);
    }

    #[cfg(not(miri))]
    pub fn append_logical_wal(&self, kind: u64, payload: &[u8]) -> std::io::Result<Option<Lsn>> {
        let Some(wal) = &self.wal else {
            return Ok(None);
        };
        wal.append_logical(kind, payload).map(Some)
    }

    #[cfg(miri)]
    pub fn append_logical_wal(&self, _kind: u64, _payload: &[u8]) -> std::io::Result<Option<Lsn>> {
        Ok(None)
    }

    pub fn prefetch_pid_async(&self, pid: PageId) {
        {
            let mut inflight = self.prefetch_inflight.lock();
            if inflight.contains(&pid) {
                return;
            }
            inflight.insert(pid);
        }
        let inflight = PrefetchInflightGuard::new(self, pid);

        let Some(pool) = self.self_weak.get().cloned() else {
            return;
        };
        let mut workers = self.prefetch_workers.lock().unwrap();
        workers.start(pool);
        if workers.try_send(pid) {
            inflight.disarm();
        }
    }

    /// Make a quiescent retired page extent available for future allocations.
    ///
    /// The caller must ensure no live owner can still reach the pages and no
    /// resident frame still contains live state for them.
    pub fn promote_reusable_extent(&self, extent: FreeExtent) {
        self.free_page_allocator.promote_reusable_extent(extent);
    }

    fn promote_pending_reusable_extents(&self) {
        let pending = {
            let mut pending = self.pending_reusable_extents.lock();
            if pending.is_empty() {
                return;
            }
            std::mem::take(&mut *pending)
        };
        for extent in pending {
            self.promote_reusable_extent(extent);
        }
    }

    /// Retire a page that has already been unlinked from its owner.
    ///
    /// This consumes the exclusive frame so the buffer pool can clear resident
    /// state. The retired page id is held back from the reusable-page allocator
    /// until `flush` makes the unlink durable in the data file. Any
    /// already-pinned users are allowed to drain before the frame is made free.
    ///
    /// # Safety
    ///
    /// No live data structure may still be able to discover this page id or
    /// frame through a stable root, parent edge, sibling edge, side table, or
    /// any other durable owner metadata.
    pub unsafe fn retire_unlinked_exclusive_frame(&self, frame: ExclusiveFrame<'_>) -> PageId {
        let bf = frame.raw();
        let pid = frame.pid();
        assert!(
            !matches!(Self::frame_parent_link(bf), ParentLink::Stable(_)),
            "stable-root pages cannot be retired through unlink retirement"
        );
        while unsafe { (*bf).header.core.pin_count.load(Ordering::Acquire) } != 1 {
            std::thread::yield_now();
        }

        unsafe {
            (*bf).header.parent_link = ParentLink::None;
            (*bf).header.core.dirty.store(false, Ordering::Relaxed);
            (*bf).header.core.referenced.store(false, Ordering::Relaxed);
            (*bf).header.core.page_lsn.store(0, Ordering::Relaxed);
            (*bf)
                .header
                .core
                .wal_buffer_epoch
                .store(0, Ordering::Relaxed);
            (*bf)
                .header
                .core
                .wal_buffer_offset
                .store(0, Ordering::Relaxed);
            (*bf)
                .header
                .core
                .state
                .store(FrameState::Free, Ordering::Release);
        }
        self.page_table.remove(pid);
        self.release_resident_budget(bf);
        self.state.free_list.push(bf);

        drop(frame);

        // With a single page class, one page retires one base page.
        self.pending_reusable_extents
            .lock()
            .push(FreeExtent::new(physical_page_number(pid), 1));
        pid
    }

    fn abort_unpublished_stable_frame(&self, mut frame: ExclusiveFrame<'_>, edge: StableSwip) {
        self.assert_stable_provenance(&edge);
        frame.set_parent_link_none();
        drop(edge);
        self.abort_unpublished_unlinked_frame(frame);
    }

    fn abort_unpublished_unlinked_frame(&self, frame: ExclusiveFrame<'_>) {
        let bf = frame.raw();
        let pid = frame.pid();
        assert_eq!(
            unsafe { (*bf).header.core.pin_count.load(Ordering::Acquire) },
            1,
            "unpublished page unexpectedly acquired another pin"
        );
        assert!(
            matches!(Self::frame_parent_link(bf), ParentLink::None),
            "unpublished unlinked page acquired an owning backlink"
        );
        unsafe {
            (*bf).header.core.dirty.store(false, Ordering::Relaxed);
            (*bf).header.core.referenced.store(false, Ordering::Relaxed);
            (*bf).header.core.page_lsn.store(0, Ordering::Relaxed);
            (*bf)
                .header
                .core
                .wal_buffer_epoch
                .store(0, Ordering::Relaxed);
            (*bf)
                .header
                .core
                .wal_buffer_offset
                .store(0, Ordering::Relaxed);
            (*bf)
                .header
                .core
                .state
                .store(FrameState::Free, Ordering::Release);
        }
        self.page_table.remove(pid);
        self.release_resident_budget(bf);
        self.state.free_list.push(bf);
        drop(frame);
    }

    /// Retire a stable page after its owning metadata has been durably
    /// unlinked. The edge is consumed so no future fix can discover the page.
    ///
    /// # Safety
    /// The caller must have removed every durable and in-memory route to this
    /// stable edge before calling. Reuse remains checkpoint-gated.
    ///
    /// ```compile_fail
    /// use pagebox_storage::buffer_pool::{BufferPool, NoLatches};
    ///
    /// let pool = BufferPool::new(1);
    /// let edge = pool.allocate_page();
    /// unsafe { pool.retire_stable(edge, NoLatches::new(&pool)) };
    /// let _ = edge.page_id();
    /// ```
    pub unsafe fn retire_stable(&self, edge: StableSwip, no_latches: NoLatches<'_>) -> PageId {
        self.assert_stable_provenance(&edge);
        let mut frame = self.fix_stable(&edge, no_latches).exclusive();
        frame.set_parent_link_none();
        drop(edge);
        unsafe { self.retire_unlinked_exclusive_frame(frame) }
    }

    fn allocate_page_id(&self) -> PageId {
        let prior_next_page_number = self.free_page_allocator.next_page_number();
        let pid = self
            .free_page_allocator
            .allocate_page(thread_alloc_shard_hint());
        self.next_page_id.fetch_max(
            self.free_page_allocator.next_page_number(),
            Ordering::Release,
        );
        self.page_store
            .allocate(pid)
            .expect("page store allocate failed");
        let page_number = physical_page_number(pid);
        if page_number < prior_next_page_number {
            let zeros = vec![0u8; PAGE_SIZE];
            self.page_store
                .write_page(pid, &zeros)
                .expect("page store zero reused page failed");
        }
        pid
    }

    /// Allocate a new page for an internal publication typestate guard.
    ///
    /// Returns `(page_id, frame)`. The frame is pinned (pin_count=1), not
    /// exclusively latched, and in Resident state. Public callers use
    /// [`BufferPool::allocate_stable`] or [`BufferPool::allocate_unlinked`],
    /// which cannot lose track of the unpublished allocation.
    pub(crate) fn allocate_and_fix(&self, _no_latches: NoLatches<'_>) -> (u64, PinnedFrame<'_>) {
        let pid = self.allocate_page_id();

        self.reserve_resident_budget();
        let bf = self.pop_free_frame();
        let _guard = unsafe { self.lock_frame_exclusive_at(bf, pid) };

        unsafe {
            (*bf).header.core.pid = pid;
            (*bf).header.parent_link = ParentLink::None;
            (*bf).page_bytes_mut().fill(0);
            (*bf).header.core.pin_count.store(1, Ordering::Relaxed);
            (*bf).header.core.referenced.store(true, Ordering::Relaxed);
            (*bf).header.core.dirty.store(false, Ordering::Relaxed);
            (*bf).header.core.page_lsn.store(0, Ordering::Relaxed);
            (*bf)
                .header
                .core
                .wal_buffer_epoch
                .store(0, Ordering::Relaxed);
            (*bf)
                .header
                .core
                .wal_buffer_offset
                .store(0, Ordering::Relaxed);
            (*bf).header.core.state.transition(
                FrameState::Free,
                FrameState::Resident,
                Ordering::Relaxed,
            );
        }
        // Register in page_table for orphan-fix lookups.
        self.page_table.insert(pid, bf);
        (pid, unsafe { PinnedFrame::new(self, bf) })
    }

    /// Allocate a resident page whose stable owner and frame backlink are
    /// created atomically before the caller can observe it.
    pub fn allocate_stable(&self, no_latches: NoLatches<'_>) -> NewStablePage<'_> {
        let (_, frame) = self.allocate_and_fix(no_latches);
        let mut frame = frame.exclusive();
        let edge = self.stable_frame(&mut frame);
        NewStablePage {
            pool: self,
            edge: Some(edge),
            frame: Some(frame),
        }
    }

    /// Allocate an unlinked page whose publication state is tracked by its
    /// returned guard.
    pub fn allocate_unlinked(&self, no_latches: NoLatches<'_>) -> NewUnlinkedPage<'_> {
        let (_, frame) = self.allocate_and_fix(no_latches);
        NewUnlinkedPage {
            pool: self,
            frame: Some(frame),
        }
    }

    /// Allocate a new page, returning an evicted swip for it.
    /// The page is written to the store but not loaded into a frame.
    pub fn allocate_page(&self) -> StableSwip {
        let page_id = self.allocate_page_id();
        StableSwip::new(self.id, page_id, Swip::evicted(page_id))
    }

    /// Create a stable owner for an already allocated page ID.
    pub fn stable_page(&self, page_id: PageId) -> StableSwip {
        let page_number = physical_page_number(page_id);
        assert!(page_number > 0, "stable page ID must be nonzero");
        assert!(
            page_number < self.next_page_id.load(Ordering::Acquire),
            "stable page ID {page_id} was not allocated by this pool"
        );
        StableSwip::new(self.id, page_id, Swip::evicted(page_id))
    }

    /// Create a stable owner for a newly initialised resident frame and install
    /// the frame's owning backlink before the exclusive latch can be released.
    fn stable_frame(&self, frame: &mut ExclusiveFrame<'_>) -> StableSwip {
        assert!(
            self.contains_frame_ptr(frame.raw()),
            "stable frame belongs to a different buffer pool"
        );
        let stable = StableSwip::new(self.id, frame.pid(), frame.hot_swip());
        frame.set_parent_link_stable(&stable);
        stable
    }

    /// Fix a swip: ensure its page is resident and return a pointer to the frame.
    /// Increments the pin count; caller must call `unfix` when done.
    ///
    /// # Safety
    /// `swip` must be a valid AtomicSwip previously returned by this pool.
    /// If the swip is hot, its buffer frame pointer must be live.
    unsafe fn fix_stable_raw(&self, owner: &Arc<StableSwipOwner>) -> *mut BufferFrame {
        // HOT/COOL fixing only borrows the stable owner. Do not clone or drop
        // this Arc on the resident fast path: refcount traffic belongs to
        // stable-link lifecycle transitions, never ordinary fixing.
        unsafe { self.fix_raw(owner.word(), owner) }
    }

    /// # Safety
    /// `owner` must belong to this pool.
    unsafe fn fix_raw(&self, swip: &AtomicSwip, owner: &Arc<StableSwipOwner>) -> *mut BufferFrame {
        loop {
            let s = swip.load(Ordering::Acquire);

            if s.is_hot() || s.is_cool() {
                let bf = unsafe { s.as_ptr::<BufferFrame>() };
                debug_assert!(
                    self.contains_frame_ptr(bf),
                    "pool.fix: stale HOT pointer: raw={:#x} ptr={:#x}",
                    s.raw(),
                    bf as usize,
                );
                assert!(
                    self.contains_frame_ptr(bf),
                    "pool.fix: stale HOT pointer: raw={:#x} ptr={:#x}",
                    s.raw(),
                    bf as usize,
                );
                let pre_state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
                if pre_state != FrameState::Resident {
                    let current = swip.load(Ordering::Acquire);
                    if current.raw() == s.raw()
                        && pre_state == FrameState::Evicting
                        && unsafe { self.try_rescue_evicting_orphan(bf, (*bf).header.core.pid) }
                    {
                        continue;
                    }
                    if current.raw() == s.raw() && pre_state == FrameState::Loading {
                        self.wait_for_hot_frame_transition(swip, s, bf);
                    }
                    continue;
                }
                let Ok(guard) = (unsafe { (*bf).latch.optimistic_or_restart() }) else {
                    continue;
                };
                let current = swip.load(Ordering::Acquire);
                if current.raw() != s.raw() {
                    continue;
                }
                unsafe { (*bf).header.core.pin_count.fetch_add(1, Ordering::Relaxed) };
                let current_state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
                if current_state != FrameState::Resident {
                    unsafe { (*bf).header.core.pin_count.fetch_sub(1, Ordering::Relaxed) };
                    if current_state == FrameState::Evicting
                        && unsafe { self.try_rescue_evicting_orphan(bf, (*bf).header.core.pid) }
                    {
                        continue;
                    }
                    if current_state == FrameState::Loading {
                        self.wait_for_hot_frame_transition(swip, s, bf);
                    }
                    continue;
                }
                if guard.validate().is_err() {
                    unsafe { (*bf).header.core.pin_count.fetch_sub(1, Ordering::Relaxed) };
                    continue;
                }
                self.mark_referenced(bf);
                return bf;
            }

            let page_id = s.as_page_id();
            let page_number = physical_page_number(page_id);
            assert!(page_id > 0, "invalid encoded page id {page_id}");
            let next_pid = self.next_page_id.load(Ordering::Relaxed);
            assert!(
                page_number > 0 && page_number < next_pid,
                "pool.fix: EVICTED swip has invalid page_id={} (next_pid={}), \
                 raw={:#x}, swip_addr={:p}",
                page_id,
                next_pid,
                s.raw(),
                swip,
            );

            let (bf, popped_from_free_list): (*mut BufferFrame, bool) = {
                // Check page_table: maybe another thread already loaded
                // this page (via fix_raw or fix_orphan_raw).
                if let Some(existing_bf) = self.page_table.lookup(page_id) {
                    (existing_bf, false)
                } else {
                    (self.pop_free_frame(), true)
                }
            };
            let _guard = unsafe { self.lock_frame_exclusive_at(bf, page_id) };
            let state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
            if state == FrameState::Resident && unsafe { (*bf).header.core.pid } == page_id {
                unsafe {
                    (*bf).header.core.pin_count.fetch_add(1, Ordering::Relaxed);
                    (*bf).header.core.referenced.store(true, Ordering::Relaxed);
                    self.reswizzle_stable_resident_locked(swip, owner, s, bf, page_id);
                }
                return bf;
            }
            if state == FrameState::Resident {
                continue;
            }
            if state == FrameState::Free && !popped_from_free_list {
                continue;
            }
            if state == FrameState::Loading || state == FrameState::Evicting {
                #[cfg(not(loom))]
                std::thread::yield_now();
                #[cfg(loom)]
                loom::thread::yield_now();
                continue;
            }

            // Register in page_table before loading (dedup for orphan
            // fixes). If another thread won the race, try again.
            if !self.page_table.try_insert(page_id, bf) {
                // Another thread is loading this page. Return the unclaimed
                // free frame; no resident-budget token has been reserved yet.
                drop(_guard);
                if popped_from_free_list {
                    self.state.free_list.push(bf);
                }
                continue;
            }

            if !self.try_reserve_resident_budget() {
                self.page_table.remove(page_id);
                drop(_guard);
                if popped_from_free_list {
                    self.state.free_list.push(bf);
                }
                self.wait_for_resident_budget();
                continue;
            }
            unsafe {
                (*bf).header.core.pid = page_id;
                (*bf).header.core.referenced.store(true, Ordering::Relaxed);
                (*bf).header.core.state.transition(
                    FrameState::Free,
                    FrameState::Loading,
                    Ordering::Relaxed,
                );
            }
            let loading = LoadingFrameReservation::new(self, bf);

            let read_start = Instant::now();
            let found = match unsafe { self.page_store.read_page(page_id, (*bf).page_bytes_mut()) }
            {
                Ok(f) => f,
                Err(e) => {
                    eprintln!(
                        "fix_swip_raw: read_page failed for page_id={page_id} (0x{page_id:016x}) err={e:?}",
                    );
                    panic!("page store read failed for page_id={page_id} (fix_swip_raw)");
                }
            };
            self.record_fix_swip_sync_load(unsafe { (*bf).page_bytes() }, read_start.elapsed());
            assert!(found, "page {page_id} not found in store");

            unsafe {
                let parent_link = ParentLink::Stable(Arc::clone(owner));
                self.install_loaded_frame_metadata(bf, page_id, parent_link, 1);
            }

            // CAS swip from EVICTED to HOT. Transition Loading → Resident.
            let Ok(_) =
                swip.compare_exchange(s, Swip::hot(bf), Ordering::AcqRel, Ordering::Acquire)
            else {
                continue;
            };
            unsafe {
                (*bf).header.core.state.transition(
                    FrameState::Loading,
                    FrameState::Resident,
                    Ordering::Relaxed,
                );
            }
            loading.disarm();
            return bf;
        }
    }

    /// Load a page by ID into its slot without setting `parent_swip`.
    ///
    /// Used for child traversal where no stable edge owner exists yet.
    /// The frame's `parent_swip` is null, so eviction will not attempt
    /// to unswizzle any parent edge — the frame is simply freed.  The
    /// next traversal through the parent will re-fault.
    ///
    /// # Safety
    /// Caller must ensure `page_id` refers to a valid allocated page.
    unsafe fn fix_orphan_raw(&self, page_id: u64) -> *mut BufferFrame {
        enum FixOrphanAction<'a> {
            Pinned(*mut BufferFrame),
            Retry,
            YieldRetry,
            WaitLoading,
            WaitForBudget,
            Load(LoadingFrameReservation<'a>),
        }

        loop {
            // Use page_table + FreeList.
            let (bf, popped_from_free_list) =
                if let Some(existing) = self.page_table.lookup(page_id) {
                    (existing, false)
                } else {
                    let bf = self.pop_free_frame();
                    (bf, true)
                };

            let state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
            if state == FrameState::Loading {
                self.metrics
                    .fix_orphan_events
                    .inc(BufferPoolFixOrphanEvent::LoadingRetry);
                self.wait_for_loading_frame_transition(bf);
                if popped_from_free_list {
                    self.state.free_list.push(bf);
                }
                continue;
            }
            if state == FrameState::Evicting {
                self.metrics
                    .fix_orphan_events
                    .inc(BufferPoolFixOrphanEvent::EvictingRetry);
                self.sample_fix_orphan_evicting_retry_page(page_id);
                if unsafe { self.try_rescue_evicting_orphan(bf, page_id) } {
                    if popped_from_free_list {
                        self.state.free_list.push(bf);
                    }
                    continue;
                }
                Self::yield_for_contention();
                if popped_from_free_list {
                    self.state.free_list.push(bf);
                }
                continue;
            }

            let pinned = self.with_fix_orphan_hot_pin(|| {
                let pid = unsafe { (*bf).header.core.pid };
                if state != FrameState::Resident || pid != page_id {
                    return None;
                }

                unsafe { (*bf).header.core.pin_count.fetch_add(1, Ordering::Relaxed) };
                let current_state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
                let current_pid = unsafe { (*bf).header.core.pid };
                if current_state == FrameState::Resident && current_pid == page_id {
                    self.mark_referenced(bf);
                    return Some(bf);
                }

                unsafe { (*bf).header.core.pin_count.fetch_sub(1, Ordering::Relaxed) };
                None
            });
            if let Some(bf) = pinned {
                // Frame was already resident (from page_table lookup).
                // If we popped from FreeList, that shouldn't happen here —
                // the page_table entry would have been found.
                debug_assert!(!popped_from_free_list || state == FrameState::Resident);
                if popped_from_free_list {
                    self.state.free_list.push(bf);
                }
                return bf;
            }

            let action = self.with_fix_orphan_exclusive_at(bf, page_id, || {
                let pinned = self.with_fix_orphan_hot_pin(|| {
                    let state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
                    let pid = unsafe { (*bf).header.core.pid };
                    if state != FrameState::Resident || pid != page_id {
                        return None;
                    }

                    unsafe { (*bf).header.core.pin_count.fetch_add(1, Ordering::Relaxed) };
                    self.mark_referenced(bf);
                    Some(bf)
                });
                if let Some(bf) = pinned {
                    return FixOrphanAction::Pinned(bf);
                }

                let state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
                if state == FrameState::Resident {
                    return FixOrphanAction::Retry;
                }
                if state == FrameState::Free && !popped_from_free_list {
                    return FixOrphanAction::Retry;
                }
                if state == FrameState::Loading {
                    self.metrics
                        .fix_orphan_events
                        .inc(BufferPoolFixOrphanEvent::LoadingRetry);
                    return FixOrphanAction::WaitLoading;
                }
                if state == FrameState::Evicting {
                    if unsafe { self.try_rescue_evicting_orphan(bf, page_id) } {
                        return FixOrphanAction::Retry;
                    }
                    self.metrics
                        .fix_orphan_events
                        .inc(BufferPoolFixOrphanEvent::EvictingRetry);
                    return FixOrphanAction::YieldRetry;
                }

                // Register in page_table before loading (dedup for
                // concurrent orphan fixes). If another thread won the
                // race, retry with a fresh frame.
                if !self.page_table.try_insert(page_id, bf) {
                    return FixOrphanAction::Retry;
                }

                // Do not block for resident budget while this frame is
                // exclusively latched.  A waiter for the same orphan page
                // will spin on that latch, while eviction may need this
                // thread to relinquish it before it can free a victim.
                if !self.try_reserve_resident_budget() {
                    self.page_table.remove(page_id);
                    return FixOrphanAction::WaitForBudget;
                }

                FixOrphanAction::Load(unsafe { self.prepare_orphan_loading_frame(bf, page_id) })
            });

            let loading = match action {
                FixOrphanAction::Pinned(bf) => {
                    debug_assert!(!popped_from_free_list);
                    return bf;
                }
                FixOrphanAction::Retry => {
                    if popped_from_free_list {
                        self.state.free_list.push(bf);
                    }
                    continue;
                }
                FixOrphanAction::YieldRetry => {
                    if popped_from_free_list {
                        self.state.free_list.push(bf);
                    }
                    Self::yield_for_contention();
                    continue;
                }
                FixOrphanAction::WaitLoading => {
                    self.wait_for_loading_frame_transition(bf);
                    if popped_from_free_list {
                        self.state.free_list.push(bf);
                    }
                    continue;
                }
                FixOrphanAction::WaitForBudget => {
                    if popped_from_free_list {
                        self.state.free_list.push(bf);
                    }
                    self.wait_for_resident_budget();
                    continue;
                }
                FixOrphanAction::Load(loading) => loading,
            };

            let read_start = Instant::now();
            let found = match unsafe { self.page_store.read_page(page_id, (*bf).page_bytes_mut()) }
            {
                Ok(f) => f,
                Err(e) => {
                    eprintln!(
                        "fix_orphan_raw: read_page failed for page_id={page_id} (0x{page_id:016x}) err={e:?}",
                    );
                    panic!("page store read failed for page_id={page_id} (fix_orphan_raw)");
                }
            };
            self.record_fix_orphan_sync_load(unsafe { (*bf).page_bytes() }, read_start.elapsed());
            assert!(found, "page {page_id} not found in store");

            unsafe {
                self.install_loaded_frame_metadata(bf, page_id, ParentLink::Unswizzled, 1);
                (*bf).header.core.state.transition(
                    FrameState::Evicting,
                    FrameState::Resident,
                    Ordering::Release,
                );
            }
            loading.disarm();
            return bf;
        }
    }

    /// Non-blocking variant of `fix_orphan`. Returns `None` if the page
    /// is not already resident and no resident budget is available. Used by
    /// the eviction DFS to resolve EVICTED children without risking
    /// deadlock on the page-provider thread.
    ///
    /// # Safety
    /// `page_id` must refer to a valid allocated page.
    unsafe fn try_fix_orphan_raw(&self, page_id: u64) -> Option<*mut BufferFrame> {
        let (bf, popped_from_free_list) = if let Some(existing) = self.page_table.lookup(page_id) {
            (existing, false)
        } else {
            let bf = self.state.free_list.try_pop()?;
            (bf, true)
        };
        let mut state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
        if state == FrameState::Loading {
            if popped_from_free_list {
                self.state.free_list.push(bf);
            }
            return None;
        }
        if state == FrameState::Evicting {
            self.metrics
                .fix_orphan_events
                .inc(BufferPoolFixOrphanEvent::EvictingRetry);
            self.sample_fix_orphan_evicting_retry_page(page_id);
            if !unsafe { self.try_rescue_evicting_orphan(bf, page_id) } {
                if popped_from_free_list {
                    self.state.free_list.push(bf);
                }
                return None;
            }
            state = FrameState::Resident;
        }
        let pinned = {
            let Some(_pin_guard) = self.try_lock_hot_pin() else {
                if popped_from_free_list {
                    self.state.free_list.push(bf);
                }
                return None;
            };
            let pid = unsafe { (*bf).header.core.pid };
            if state == FrameState::Resident && pid == page_id {
                unsafe { (*bf).header.core.pin_count.fetch_add(1, Ordering::Relaxed) };
                let current_state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
                let current_pid = unsafe { (*bf).header.core.pid };
                if current_state == FrameState::Resident && current_pid == page_id {
                    self.mark_referenced(bf);
                    Some(bf)
                } else {
                    unsafe { (*bf).header.core.pin_count.fetch_sub(1, Ordering::Relaxed) };
                    None
                }
            } else {
                None
            }
        };
        if let Some(bf) = pinned {
            debug_assert!(!popped_from_free_list);
            return Some(bf);
        }

        let loading = {
            let Some(_guard) = (unsafe { self.try_lock_frame_exclusive_at(bf, page_id) }) else {
                if popped_from_free_list {
                    self.state.free_list.push(bf);
                }
                return None;
            };
            let state = {
                let _pin_guard = match self.try_lock_hot_pin() {
                    Some(guard) => guard,
                    None => {
                        self.metrics
                            .fix_orphan_events
                            .inc(BufferPoolFixOrphanEvent::HotPinWait);
                        if popped_from_free_list {
                            self.state.free_list.push(bf);
                        }
                        return None;
                    }
                };
                let state = unsafe { (*bf).header.core.state.load(Ordering::Acquire) };
                let pid = unsafe { (*bf).header.core.pid };
                if state == FrameState::Resident && pid == page_id {
                    unsafe { (*bf).header.core.pin_count.fetch_add(1, Ordering::Relaxed) };
                    self.mark_referenced(bf);
                    return Some(bf);
                }
                state
            };
            if state == FrameState::Resident {
                debug_assert!(!popped_from_free_list);
                return None;
            }
            if state == FrameState::Free && !popped_from_free_list {
                return None;
            }
            if state == FrameState::Loading || state == FrameState::Evicting {
                if popped_from_free_list {
                    self.state.free_list.push(bf);
                }
                return None;
            }
            // Register in page_table before loading.
            if popped_from_free_list && !self.page_table.try_insert(page_id, bf) {
                self.state.free_list.push(bf);
                return None;
            }
            if !self.try_reserve_resident_budget() {
                if popped_from_free_list {
                    self.page_table.remove(page_id);
                    self.state.free_list.push(bf);
                }
                return None;
            }

            unsafe { self.prepare_orphan_loading_frame(bf, page_id) }
        };

        let read_start = Instant::now();
        let found = match unsafe { self.page_store.read_page(page_id, (*bf).page_bytes_mut()) } {
            Ok(f) => f,
            Err(e) => {
                eprintln!(
                    "fix_orphan_raw: read_page failed for page_id={page_id} (0x{page_id:016x}) err={e:?}",
                );
                panic!("page store read failed for page_id={page_id}");
            }
        };
        self.record_fix_orphan_sync_load(unsafe { (*bf).page_bytes() }, read_start.elapsed());
        if !found {
            return None;
        }

        unsafe {
            self.install_loaded_frame_metadata(bf, page_id, ParentLink::Unswizzled, 1);
            (*bf).header.core.state.transition(
                FrameState::Evicting,
                FrameState::Resident,
                Ordering::Release,
            );
        }
        loading.disarm();
        Some(bf)
    }

    /// Unfix a frame, decrementing its pin count.
    ///
    /// # Safety
    /// `bf` must point to a live, pinned frame managed by this pool.
    unsafe fn unfix_raw(&self, bf: *mut BufferFrame) {
        let old = unsafe { (*bf).header.core.pin_count.fetch_sub(1, Ordering::Release) };
        debug_assert!(old > 0, "unfix on unpinned frame");
        let _ = old;
        // Parent pin management is handled during eviction.
    }

    /// Mark a frame as dirty (modified). If a WAL is attached, appends a
    /// full-page image to the WAL and records the LSN on the frame.
    ///
    /// # Safety
    /// `bf` must point to a live, pinned frame managed by this pool.
    unsafe fn mark_dirty_raw(&self, bf: *mut BufferFrame) {
        debug_assert!(
            unsafe { (*bf).header.core.pin_count.load(Ordering::Relaxed) } > 0,
            "must be pinned to mark dirty"
        );
        self.metrics
            .eviction_events
            .inc(BufferPoolEvictionEvent::DirtyMarks);
        let was_dirty = unsafe { (*bf).header.core.dirty.load(Ordering::Relaxed) };
        if was_dirty {
            self.metrics
                .eviction_events
                .inc(BufferPoolEvictionEvent::DirtyRelogs);
        }
        self.mark_referenced(bf);
        #[cfg(not(miri))]
        if let Some(ref wal) = self.wal {
            let pid = unsafe { (*bf).header.core.pid };
            let page = unsafe { (*bf).page_bytes_mut() };
            if page.len() == PAGE_SIZE {
                Self::record_page_kind(page, &self.metrics.dirty_wal_page_image_pages);
                if was_dirty {
                    Self::record_page_kind(page, &self.metrics.dirty_wal_page_image_relog_pages);
                }
                let prev_epoch_raw =
                    unsafe { (*bf).header.core.wal_buffer_epoch.load(Ordering::Relaxed) };
                let prev_epoch = prev_epoch_raw & ((1u64 << 48) - 1);
                let prev_shard_idx = (prev_epoch_raw >> 48) as u16;
                let prev_offset =
                    unsafe { (*bf).header.core.wal_buffer_offset.load(Ordering::Relaxed) };
                let prev_record = (was_dirty && prev_epoch != 0 && prev_offset != 0).then_some(
                    BufferedWalRecord {
                        epoch: prev_epoch,
                        offset: prev_offset,
                        shard_idx: prev_shard_idx,
                    },
                );
                let lsn = wal.claim_lsn();
                page_header::write_page_lsn(page, lsn);
                let mut scratch = PageScratch::take();
                scratch.as_mut_slice().copy_from_slice(page);
                prepare_page_copy_for_writeback(scratch.as_mut_slice(), self);
                let mut record = BufferedWalRecord {
                    epoch: 0,
                    offset: 0,
                    shard_idx: 0,
                };
                let mut logged = false;

                if let Some(prev_record) = prev_record {
                    logged = wal
                        .try_overwrite_page_image_with_lsn(
                            prev_record,
                            lsn,
                            pid,
                            |_lsn, page_image| {
                                page_image.copy_from_slice(scratch.as_mut_slice());
                            },
                        )
                        .expect("WAL overwrite failed");
                    if logged {
                        record = prev_record;
                    }
                }

                if !logged {
                    record = wal
                        .append_page_image_with_lsn(lsn, pid, |_lsn, page_image| {
                            page_image.copy_from_slice(scratch.as_mut_slice());
                        })
                        .expect("WAL append failed");
                }
                unsafe {
                    (*bf).header.core.page_lsn.store(lsn, Ordering::Relaxed);
                    (*bf).header.core.wal_buffer_epoch.store(
                        record.epoch | ((u64::from(record.shard_idx)) << 48),
                        Ordering::Relaxed,
                    );
                    (*bf)
                        .header
                        .core
                        .wal_buffer_offset
                        .store(record.offset, Ordering::Relaxed);
                };
            } else {
                let lsn = wal.claim_lsn();
                page_header::write_page_lsn(page, lsn);
                let mut page_copy = AlignedPageCopy::copy_from(page);
                prepare_page_copy_for_writeback(page_copy.as_mut_slice(), self);
                let page_len = page_copy.as_slice().len();
                wal.append_page_image_bytes_with_lsn(lsn, pid, page_len, |lsn, page_image| {
                    page_header::write_page_lsn(page_image, lsn);
                    page_image.copy_from_slice(page_copy.as_slice());
                })
                .expect("WAL append failed");
                unsafe {
                    (*bf).header.core.page_lsn.store(lsn, Ordering::Relaxed);
                    (*bf)
                        .header
                        .core
                        .wal_buffer_epoch
                        .store(0, Ordering::Relaxed);
                    (*bf)
                        .header
                        .core
                        .wal_buffer_offset
                        .store(0, Ordering::Relaxed);
                };
            }
        }
        unsafe {
            (*bf)
                .header
                .core
                .dirty_generation
                .fetch_add(1, Ordering::Relaxed);
            (*bf).header.core.dirty.store(true, Ordering::Release);
        };
    }

    /// Mark a page dirty after an in-place byte-range update and log only the
    /// page LSN plus the changed range. The caller must hold the frame's
    /// exclusive latch and `data` must already match the page at `offset`.
    unsafe fn mark_dirty_patch_raw(&self, bf: *mut BufferFrame, offset: usize, data: &[u8]) {
        #[cfg(not(miri))]
        let Some(wal) = self.wal.as_ref() else {
            unsafe { self.mark_dirty_raw(bf) };
            return;
        };
        #[cfg(miri)]
        {
            let _ = (offset, data);
            unsafe { self.mark_dirty_raw(bf) };
            return;
        }

        #[cfg(not(miri))]
        {
            debug_assert!(
                unsafe { (*bf).header.core.pin_count.load(Ordering::Relaxed) } > 0,
                "must be pinned to mark dirty"
            );
            assert!(offset >= std::mem::size_of::<Lsn>());
            let end = offset
                .checked_add(data.len())
                .expect("dirty patch range overflow");
            assert!(end <= PAGE_SIZE, "dirty patch extends beyond page");
            let page = unsafe { (*bf).page_bytes_mut() };
            assert_eq!(
                &page[offset..end],
                data,
                "dirty patch bytes must already be installed in the page"
            );

            self.metrics
                .eviction_events
                .inc(BufferPoolEvictionEvent::DirtyMarks);
            if unsafe { (*bf).header.core.dirty.load(Ordering::Relaxed) } {
                self.metrics
                    .eviction_events
                    .inc(BufferPoolEvictionEvent::DirtyRelogs);
            }
            self.mark_referenced(bf);

            let pid = unsafe { (*bf).header.core.pid };
            let lsn = wal.claim_lsn();
            let lsn_bytes = lsn.to_le_bytes();
            let encoded_bytes = if data.is_empty() {
                wal.append_page_patch_ranges_with_lsn(lsn, pid, &[(0, &lsn_bytes)])
            } else {
                wal.append_page_patch_ranges_with_lsn(lsn, pid, &[(0, &lsn_bytes), (offset, data)])
            }
            .expect("WAL page-patch append failed")
            .expect("valid page-local ranges must use patch encoding");

            page_header::write_page_lsn(page, lsn);
            unsafe {
                (*bf).header.core.page_lsn.store(lsn, Ordering::Relaxed);
                (*bf)
                    .header
                    .core
                    .wal_buffer_epoch
                    .store(0, Ordering::Relaxed);
                (*bf)
                    .header
                    .core
                    .wal_buffer_offset
                    .store(0, Ordering::Relaxed);
                (*bf)
                    .header
                    .core
                    .dirty_generation
                    .fetch_add(1, Ordering::Relaxed);
                (*bf).header.core.dirty.store(true, Ordering::Release);
            }
            self.metrics.dirty_wal_page_patch_records.inc();
            self.metrics
                .dirty_wal_page_patch_bytes
                .add(encoded_bytes.min(isize::MAX as usize) as isize);
        }
    }

    /// Mark a frame dirty after the caller has appended an equivalent logical
    /// WAL record and has the record LSN.
    ///
    /// This is for callers that log an equivalent logical mutation instead of
    /// a full-page image. Eviction and checkpoint still enforce WAL-before-data
    /// by flushing this LSN before writing the page.
    ///
    /// # Safety
    /// `bf` must point to a live, pinned frame managed by this pool.
    unsafe fn mark_dirty_with_lsn_raw(&self, bf: *mut BufferFrame, lsn: Lsn) {
        debug_assert!(
            unsafe { (*bf).header.core.pin_count.load(Ordering::Relaxed) } > 0,
            "must be pinned to mark dirty"
        );
        self.metrics
            .eviction_events
            .inc(BufferPoolEvictionEvent::DirtyMarks);
        let was_dirty = unsafe { (*bf).header.core.dirty.load(Ordering::Relaxed) };
        if was_dirty {
            self.metrics
                .eviction_events
                .inc(BufferPoolEvictionEvent::DirtyRelogs);
        }
        self.mark_referenced(bf);

        let page = unsafe { (*bf).page_bytes_mut() };
        page_header::write_page_lsn(page, lsn);
        unsafe {
            (*bf).header.core.page_lsn.store(lsn, Ordering::Relaxed);
            (*bf)
                .header
                .core
                .wal_buffer_epoch
                .store(0, Ordering::Relaxed);
            (*bf)
                .header
                .core
                .wal_buffer_offset
                .store(0, Ordering::Relaxed);
            (*bf)
                .header
                .core
                .dirty_generation
                .fetch_add(1, Ordering::Relaxed);
            (*bf).header.core.dirty.store(true, Ordering::Release);
        }
    }

    /// Panic with a detailed snapshot of slot-state distribution.
    /// Called when resident-budget waiting detects true exhaustion:
    /// no budget tokens available and no evictable resident slot.
    fn panic_pool_exhausted(&self, waiting_for: &'static str, stalled_rounds: u32) -> ! {
        let arena = self.arena();
        let num_slots = self.allocated_slots();
        let mut pinned = 0usize;
        #[cfg_attr(miri, allow(unused_mut))]
        let mut inner = 0usize;
        let mut dirty = 0usize;
        let mut referenced = 0usize;
        let mut resident = 0usize;
        let mut free_actual = 0usize;
        let mut evicting = 0usize;
        let mut loading = 0usize;
        for i in 0..num_slots {
            if !self.is_slot_initialized(i) {
                free_actual += 1;
                continue;
            }
            let bf = self.raw_frame(i);
            let state = unsafe { (*bf).header.core.state.load(Ordering::Relaxed) };
            match state {
                FrameState::Free => free_actual += 1,
                FrameState::Resident => {
                    resident += 1;
                    if unsafe { (*bf).header.core.pin_count.load(Ordering::Relaxed) } > 0 {
                        pinned += 1;
                    }
                    if unsafe { (*bf).header.core.dirty.load(Ordering::Relaxed) } {
                        dirty += 1;
                    }
                    if unsafe { (*bf).header.core.referenced.load(Ordering::Relaxed) } {
                        referenced += 1;
                    }
                    #[cfg(not(miri))]
                    if page_header::is_inner_index_page(unsafe { &(*bf).page }) {
                        inner += 1;
                    }
                }
                FrameState::Evicting => evicting += 1,
                FrameState::Loading => loading += 1,
            }
        }
        let parent_hint_latch_misses = self
            .metrics
            .unswizzle_parent_events
            .get(UnswizzleParentEvent::FastPathLatchMisses);
        let parent_dfs_fallbacks = self
            .metrics
            .unswizzle_parent_events
            .get(UnswizzleParentEvent::DfsFallbacks);
        let parent_dfs_failures = self
            .metrics
            .unswizzle_parent_events
            .get(UnswizzleParentEvent::DfsFailures);
        let second_chance_skips = self
            .metrics
            .eviction_events
            .get(BufferPoolEvictionEvent::SecondChanceSkips);
        panic!(
            "buffer pool exhausted: no frame became evictable \
             (waiting_for={}, stalled_rounds={}, total={}, allocated={}, \
             resident_budget_available={}, free_actual={}, resident={}, \
             pinned={}, dirty={}, referenced={}, inner_idx={}, \
             evicting={}, loading={}, eviction_budget={}, eviction_in_flight={}, \
             parent_hint_latch_misses={}, parent_dfs_fallbacks={}, parent_dfs_failures={}, \
             second_chance_skips={})",
            waiting_for,
            stalled_rounds,
            arena.len,
            num_slots,
            self.resident_base_pages_available.load(Ordering::Relaxed),
            free_actual,
            resident,
            pinned,
            dirty,
            referenced,
            inner,
            evicting,
            loading,
            self.eviction_budget,
            self.eviction_in_flight.load(Ordering::Relaxed),
            parent_hint_latch_misses,
            parent_dfs_fallbacks,
            parent_dfs_failures,
            second_chance_skips,
        )
    }

    pub(crate) fn release_resident_budget(&self, bf: *mut BufferFrame) {
        let _ = bf;
        // With a single page class, each frame consumes one base page.
        self.resident_base_pages_available
            .fetch_add(1, Ordering::Relaxed);
        // Notify any worker waiting in pop_free_frame/wait_for_resident_budget.
        if background_page_provider_enabled()
            && let Ok(pp) = self.page_provider.lock()
        {
            pp.frames_available_notify();
        }
    }

    /// Approximate number of resident-budget tokens still available
    /// before eviction must replenish capacity.
    pub fn approx_available_budget(&self) -> usize {
        self.resident_base_pages_available.load(Ordering::Relaxed)
    }

    pub fn try_evict_any_policy_for_provider(&self, max_batch: usize) -> usize {
        self.try_evict_any_policy(max_batch)
    }

    /// Flush up to `max_batch` dirty pages for the background provider.
    ///
    /// Returns only pages whose dirty generation was unchanged and therefore
    /// transitioned to clean. Pages redirtied after their writeback copy was
    /// taken are persisted but do not count as eviction-capacity progress.
    #[cfg(not(miri))]
    pub fn try_flush_dirty_batch_for_provider(&self, max_batch: usize) -> std::io::Result<usize> {
        self.try_flush_dirty_batch(max_batch)
    }

    #[cfg(miri)]
    pub fn try_flush_dirty_batch_for_provider(&self, _max_batch: usize) -> std::io::Result<usize> {
        Ok(0)
    }

    /// Try to evict one randomly-selected frame. Returns true if a frame
    /// was evicted and released a resident-budget token. Used by the
    /// page provider thread to proactively replenish available budget.
    /// Try to evict one frame using a sequential clock-hand scan.
    ///
    /// Advances a shared clock hand through the frame array. On each probe:
    /// - If the frame is not Resident, skip (no cache-line bounces).
    /// - If the `referenced` bit is set, clear it (second chance) and skip.
    /// - If unreferenced, attempt eviction (exclusive latch, CAS to Evicting).
    ///
    /// Sequential scanning has better cache behavior than random probing:
    /// adjacent frames are likely in the same cache-line group, and the
    /// hardware prefetcher can warm the next group while we process the
    /// current one. It also avoids repeatedly probing the same hot frames
    /// that random selection would hit.
    #[cfg(not(miri))]
    pub fn try_evict_one(&self) -> bool {
        let num_slots = self.allocated_slots();
        if num_slots == 0 {
            return false;
        }
        let start = self.state.eviction_hand.fetch_add(1, Ordering::Relaxed) % num_slots;
        for offset in 0..num_slots {
            let idx = (start + offset) % num_slots;
            if !self.is_slot_initialized(idx) {
                continue;
            }
            let bf = self.raw_frame(idx);
            if self
                .with_single_evict_candidate(bf, |pid| self.finish_latched_evicting_frame(bf, pid))
                .is_some()
            {
                return true;
            }
        }
        false
    }

    /// Try to acquire an eviction permit. Returns `None` if the pool already
    /// has `eviction_budget` frames in the Evicting state. The permit must be
    /// held for the entire lifetime of the Evicting state — from the CAS that
    /// transitions Resident → Evicting until the frame reaches Free or reverts
    /// to Resident.
    fn try_acquire_eviction_permit(&self) -> Option<EvictionPermit<'_>> {
        loop {
            let current = self.eviction_in_flight.load(Ordering::Acquire);
            if current >= self.eviction_budget {
                return None;
            }
            if self
                .eviction_in_flight
                .compare_exchange(current, current + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Some(EvictionPermit {
                    in_flight: &self.eviction_in_flight,
                });
            }
        }
    }

    #[cfg(not(miri))]
    fn with_single_evict_candidate<T>(
        &self,
        bf: *mut BufferFrame,
        f: impl FnOnce(u64) -> T,
    ) -> Option<T> {
        if unsafe { (*bf).header.core.state.load(Ordering::Relaxed) } != FrameState::Resident {
            return None;
        }
        if !Self::frame_page_allows_eviction(bf) {
            return None;
        }

        let Ok(guard) = (unsafe { (*bf).latch.optimistic_or_restart() }) else {
            return None;
        };
        if unsafe { (*bf).header.core.state.load(Ordering::Relaxed) } != FrameState::Resident {
            return None;
        }
        if guard.validate().is_err() {
            return None;
        }

        // Eviction is opportunistic; blocking here can deadlock if callers
        // hold shared page latches while trying to reserve resident budget.
        let Ok(_exc) = guard.try_upgrade_to_exclusive() else {
            return None;
        };
        if unsafe { (*bf).header.core.pin_count.load(Ordering::Acquire) } != 0 {
            return None;
        }
        if unsafe { (*bf).header.core.referenced.swap(false, Ordering::Relaxed) } {
            self.metrics
                .eviction_events
                .inc(BufferPoolEvictionEvent::SecondChanceSkips);
            return None;
        }
        if unsafe { (*bf).header.core.dirty.load(Ordering::Acquire) } {
            return None;
        }

        // Acquire eviction permit before transitioning to Evicting.
        // This bounds the number of frames simultaneously in Evicting to
        // eviction_budget, preventing pool exhaustion under concurrent
        // eviction attempts.
        let _permit = self.try_acquire_eviction_permit()?;

        let Ok(_) = (unsafe {
            (*bf).header.core.state.compare_exchange_transition(
                FrameState::Resident,
                FrameState::Evicting,
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
        }) else {
            return None;
        };

        // A concurrent reader can increment pin_count between the earlier
        // inspection and the CAS above while still seeing Resident state.
        // Re-check pin_count under the Evicting state before any writeback or
        // parent unswizzle; if a pin sneaked in, revert to Resident so the
        // reader keeps a valid frame and the parent SWIP stays correct.
        if unsafe { (*bf).header.core.pin_count.load(Ordering::Acquire) } != 0 {
            Self::revert_frame_to_resident(bf);
            return None;
        }

        let pid = unsafe { (*bf).header.core.pid };

        if page_header::read_page_type(unsafe { (*bf).page_bytes() }) == PageType::Delta {
            self.try_reclaim_before_evict(pid, bf);
        }

        Some(f(pid))
    }

    #[cfg(not(miri))]
    fn finish_latched_evicting_frame(&self, bf: *mut BufferFrame, pid: u64) -> bool {
        self.unswizzle_and_free(bf, pid)
    }

    pub(crate) fn try_evict_policy(&self, max_batch: usize) -> usize {
        match eviction_policy() {
            EvictionPolicy::BatchClock => {
                #[cfg(miri)]
                {
                    return usize::from(self.try_evict_one());
                }
                #[cfg(not(miri))]
                {
                    let evicted = self.try_evict_batch(max_batch);
                    if evicted > 0 {
                        return evicted;
                    }
                    usize::from(self.try_evict_one())
                }
            }
            EvictionPolicy::RandomSecondChance => {
                // try_evict_one already does a full clock-hand sweep
                // (up to num_slots probes). Call it just once — the
                // sweep clears referenced bits and evicts the first
                // unreferenced frame it finds. If it returns false,
                // no frame in the pool is currently evictable.
                let mut evicted = 0usize;
                for _ in 0..max_batch {
                    if self.try_evict_one() {
                        evicted += 1;
                    } else {
                        break;
                    }
                }
                evicted
            }
        }
    }

    #[cfg(not(miri))]
    fn try_evict_any_policy(&self, max_batch: usize) -> usize {
        // With a single page class there is one arena to evict from.
        self.try_evict_policy(max_batch)
    }

    #[cfg(miri)]
    fn try_evict_any_policy(&self, _max_batch: usize) -> usize {
        0
    }

    #[cfg(not(miri))]
    fn try_evict_any_batch(&self, max_batch: usize) -> usize {
        // With a single page class there is one arena to evict from.
        self.try_evict_batch(max_batch)
    }

    #[cfg(miri)]
    fn try_evict_any_batch(&self, _max_batch: usize) -> usize {
        0
    }

    #[cfg(not(miri))]
    /// Return the number of pages that actually transitioned to clean.
    fn try_flush_dirty_batch(&self, max_batch: usize) -> std::io::Result<usize> {
        if max_batch == 0 {
            return Ok(0);
        }

        struct DirtyFlushClaim<'a> {
            pool: &'a BufferPool,
            pid: PageId,
        }

        impl Drop for DirtyFlushClaim<'_> {
            fn drop(&mut self) {
                self.pool.dirty_flush_inflight.lock().remove(&self.pid);
            }
        }

        struct DirtyPage<'a> {
            _claim: DirtyFlushClaim<'a>,
            pinned: PinnedFrame<'a>,
            pid: u64,
            page_lsn: u64,
            dirty_generation: u64,
            copy_idx: usize,
        }

        struct DirtyPageCopy<'a> {
            page: DirtyPage<'a>,
            copy: AlignedPageCopy,
        }

        fn try_copy_dirty_resident_page<'a>(
            pool: &'a BufferPool,
            bf: *mut BufferFrame,
        ) -> Option<DirtyPageCopy<'a>> {
            if unsafe { (*bf).header.core.state.load(Ordering::Acquire) } != FrameState::Resident {
                return None;
            }
            if !unsafe { (*bf).header.core.dirty.load(Ordering::Relaxed) } {
                return None;
            }

            let _guard = unsafe { (*bf).latch.try_lock_exclusive() }?;
            if unsafe { (*bf).header.core.state.load(Ordering::Acquire) } != FrameState::Resident
                || !unsafe { (*bf).header.core.dirty.load(Ordering::Relaxed) }
            {
                return None;
            }

            let pid = unsafe { (*bf).header.core.pid };
            if !pool.dirty_flush_inflight.lock().insert(pid) {
                return None;
            }
            let claim = DirtyFlushClaim { pool, pid };
            unsafe { (*bf).header.core.pin_count.fetch_add(1, Ordering::Relaxed) };
            let pinned = unsafe { PinnedFrame::new(pool, bf) };
            let page_lsn = unsafe { (*bf).header.core.page_lsn.load(Ordering::Relaxed) };
            let dirty_generation =
                unsafe { (*bf).header.core.dirty_generation.load(Ordering::Relaxed) };
            let mut copy = AlignedPageCopy::copy_from(unsafe { (*bf).page_bytes() });
            prepare_page_copy_for_writeback(copy.as_mut_slice(), pool);

            let page = DirtyPage {
                _claim: claim,
                pinned,
                pid,
                page_lsn,
                dirty_generation,
                copy_idx: 0,
            };
            Some(DirtyPageCopy { page, copy })
        }

        let mut dirty_pages: Vec<DirtyPage<'_>> = Vec::with_capacity(max_batch);
        let mut copies: Vec<AlignedPageCopy> = Vec::with_capacity(max_batch);
        let mut max_lsn = 0u64;

        let num_slots = self.allocated_slots();
        for i in 0..num_slots {
            if dirty_pages.len() >= max_batch {
                break;
            }
            if !self.is_slot_initialized(i) {
                continue;
            }

            let bf = self.raw_frame(i);
            let Some(mut dirty_copy) = try_copy_dirty_resident_page(self, bf) else {
                continue;
            };

            max_lsn = max_lsn.max(dirty_copy.page.page_lsn);
            let copy_idx = copies.len();
            dirty_copy.page.copy_idx = copy_idx;
            copies.push(dirty_copy.copy);
            dirty_pages.push(dirty_copy.page);
            if dirty_pages.len() >= max_batch {
                break;
            }
        }

        if dirty_pages.is_empty() {
            return Ok(0);
        }

        self.metrics.dirty_flush_batches.inc();
        self.metrics.dirty_flush_pages.add(
            dirty_pages
                .len()
                .min(isize::MAX as usize)
                .try_into()
                .unwrap_or(isize::MAX),
        );

        #[cfg(not(miri))]
        if max_lsn > 0
            && let Some(ref wal) = self.wal
        {
            let started = Instant::now();
            wal.flush_at_least(max_lsn);
            self.metrics
                .dirty_flush_wal_wait_ns
                .add(saturating_duration_nanos(started.elapsed()).min(isize::MAX as u64) as isize);
        }

        let pages = dirty_pages
            .iter()
            .map(|page| (page.pid, copies[page.copy_idx].as_slice()))
            .collect::<Vec<_>>();
        // WAL is durable through max_lsn before these data-file writes. This
        // cleaning pass only needs pages to be readable for a subsequent
        // eviction fault; checkpoint/flush owns the data-file sync boundary.
        let write_started = Instant::now();
        self.page_store.write_pages(&pages)?;
        self.metrics.dirty_flush_data_write_ns.add(
            saturating_duration_nanos(write_started.elapsed()).min(isize::MAX as u64) as isize,
        );

        let mut cleaned = 0usize;
        for page in &dirty_pages {
            let bf = page.pinned.raw();
            let _guard = unsafe { (*bf).latch.lock_exclusive() };
            let unchanged_since_copy = unsafe { (*bf).header.core.state.load(Ordering::Acquire) }
                == FrameState::Resident
                && unsafe { (*bf).header.core.pid } == page.pid
                && unsafe { (*bf).header.core.dirty.load(Ordering::Relaxed) }
                && unsafe { (*bf).header.core.dirty_generation.load(Ordering::Relaxed) }
                    == page.dirty_generation;
            if unchanged_since_copy {
                Self::clear_frame_dirty_metadata(bf);
                cleaned += 1;
            }
        }
        let stale = dirty_pages.len() - cleaned;
        self.metrics
            .dirty_flush_cleaned_pages
            .add(cleaned.min(isize::MAX as usize) as isize);
        self.metrics
            .dirty_flush_stale_pages
            .add(stale.min(isize::MAX as usize) as isize);

        Ok(cleaned)
    }

    #[cfg(miri)]
    fn try_flush_dirty_batch(&self, _max_batch: usize) -> std::io::Result<usize> {
        Ok(0)
    }

    /// Stub for miri — try_evict_one is not supported under miri.
    #[cfg(miri)]
    pub fn try_evict_one(&self) -> bool {
        false
    }

    /// Batch eviction: scan frames sequentially from a clock-hand position,
    /// collect up to `max_batch` clean candidates, unswizzle their parents,
    /// and release resident-budget tokens. Dirty pages remain Resident and
    /// are cleaned separately by `try_flush_dirty_batch` so page-store I/O
    /// never strands readers behind an Evicting frame.
    /// Returns the number of slots successfully evicted.
    #[cfg(not(miri))]
    pub fn try_evict_batch(&self, max_batch: usize) -> usize {
        use pagebox_hybrid_latch::ExclusiveGuard;

        let num_slots = self.allocated_slots();
        if num_slots == 0 {
            return 0;
        }
        let start = self
            .state
            .eviction_hand
            .fetch_add(max_batch * 2, Ordering::Relaxed)
            % num_slots;

        // Scan and collect clean candidates under their exclusive latches.
        struct Candidate {
            bf: *mut BufferFrame,
            pid: u64,
        }

        struct LatchedCandidate {
            candidate: Candidate,
            _guard: ExclusiveGuard<'static>,
            _permit: EvictionPermit<'static>,
        }

        impl LatchedCandidate {
            unsafe fn new(
                candidate: Candidate,
                guard: ExclusiveGuard<'_>,
                permit: EvictionPermit<'_>,
            ) -> Self {
                // SAFETY: frame latches live in the pool's frame array and
                // outlive this eviction batch. The eviction_in_flight counter
                // lives in the pool's struct and outlives this batch.
                let guard = unsafe { extend_exclusive_guard(guard) };
                let permit = unsafe {
                    std::mem::transmute::<EvictionPermit<'_>, EvictionPermit<'static>>(permit)
                };
                Self {
                    candidate,
                    _guard: guard,
                    _permit: permit,
                }
            }

            fn release(self) -> Candidate {
                self.candidate
            }
        }

        fn try_select_evict_candidate(
            pool: &BufferPool,
            bf: *mut BufferFrame,
        ) -> Option<LatchedCandidate> {
            if unsafe { (*bf).header.core.state.load(Ordering::Relaxed) } != FrameState::Resident {
                return None;
            }

            if !BufferPool::frame_page_allows_eviction(bf) {
                return None;
            }

            let guard = unsafe { (*bf).latch.try_lock_exclusive() }?;
            if unsafe { (*bf).header.core.state.load(Ordering::Relaxed) } != FrameState::Resident
                || unsafe { (*bf).header.core.pin_count.load(Ordering::Acquire) } != 0
            {
                return None;
            }
            if unsafe { (*bf).header.core.referenced.swap(false, Ordering::Relaxed) } {
                pool.metrics
                    .eviction_events
                    .inc(BufferPoolEvictionEvent::SecondChanceSkips);
                return None;
            }

            if unsafe { (*bf).header.core.dirty.load(Ordering::Acquire) } {
                return None;
            }

            // Acquire eviction permit before transitioning to Evicting.
            let permit = pool.try_acquire_eviction_permit()?;

            let Ok(_) = (unsafe {
                (*bf).header.core.state.compare_exchange_transition(
                    FrameState::Resident,
                    FrameState::Evicting,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
            }) else {
                return None;
            };

            let pid = unsafe { (*bf).header.core.pid };
            if page_header::read_page_type(unsafe { (*bf).page_bytes() }) == PageType::Delta {
                pool.try_reclaim_before_evict(pid, bf);
            }
            let candidate = Candidate { bf, pid };
            Some(unsafe { LatchedCandidate::new(candidate, guard, permit) })
        }

        fn try_finalize_evicting_candidate(pool: &BufferPool, candidate: &Candidate) -> bool {
            let Some(_guard) = (unsafe { (*candidate.bf).latch.try_lock_exclusive() }) else {
                BufferPool::revert_frame_to_resident(candidate.bf);
                return false;
            };

            if unsafe { (*candidate.bf).header.core.state.load(Ordering::Acquire) }
                != FrameState::Evicting
            {
                return false;
            }

            if unsafe {
                (*candidate.bf)
                    .header
                    .core
                    .pin_count
                    .load(Ordering::Acquire)
            } != 0
                || unsafe { (*candidate.bf).header.core.dirty.load(Ordering::Acquire) }
            {
                BufferPool::revert_frame_to_resident(candidate.bf);
                return false;
            }

            pool.unswizzle_and_free(candidate.bf, candidate.pid)
        }

        let mut candidates: Vec<Candidate> = Vec::with_capacity(max_batch);

        for i in 0..num_slots {
            if candidates.len() >= max_batch {
                break;
            }
            let idx = (start + i) % num_slots;
            if !self.is_slot_initialized(idx) {
                continue;
            }
            let bf = self.raw_frame(idx);
            let Some(latched) = try_select_evict_candidate(self, bf) else {
                continue;
            };
            candidates.push(latched.release());
        }

        if candidates.is_empty() {
            return 0;
        }

        let mut evicted = 0usize;
        for c in &candidates {
            if try_finalize_evicting_candidate(self, c) {
                evicted += 1;
            }
        }

        evicted
    }

    fn frame_parent_link(bf: *mut BufferFrame) -> ParentLink {
        unsafe { (*bf).header.parent_link.clone() }
    }

    #[cfg(not(miri))]
    fn revert_frame_to_resident(bf: *mut BufferFrame) {
        unsafe {
            (*bf).header.core.state.transition(
                FrameState::Evicting,
                FrameState::Resident,
                Ordering::Relaxed,
            );
        }
    }

    #[cfg(not(miri))]
    fn clear_frame_dirty_metadata(bf: *mut BufferFrame) {
        unsafe {
            (*bf).header.core.dirty.store(false, Ordering::Relaxed);
            (*bf)
                .header
                .core
                .wal_buffer_epoch
                .store(0, Ordering::Relaxed);
            (*bf)
                .header
                .core
                .wal_buffer_offset
                .store(0, Ordering::Relaxed);
        }
    }

    #[cfg(not(miri))]
    fn frame_is_index_page(bf: *mut BufferFrame) -> bool {
        page_header::read_page_type(unsafe { (*bf).page_bytes() }) == PageType::Index
    }

    #[cfg(not(miri))]
    fn frame_page_allows_eviction(bf: *mut BufferFrame) -> bool {
        let page = unsafe { (*bf).page_bytes() };
        !page_header::is_inner_index_page(page)
            && !page_header::should_remain_resident(page)
            && !is_stable_index_root(page, Self::frame_parent_link(bf))
    }

    #[cfg(not(miri))]
    fn parent_link_allows_free(&self, bf: *mut BufferFrame, parent_updated: bool) -> bool {
        match Self::frame_parent_link(bf) {
            ParentLink::InnerNode(_) => parent_updated,
            ParentLink::None => !Self::frame_is_index_page(bf) || parent_updated,
            ParentLink::Stable(_) | ParentLink::Unswizzled => true,
        }
    }

    #[cfg(not(miri))]
    fn can_free_evicting_frame(bf: *mut BufferFrame) -> bool {
        (unsafe { (*bf).header.core.pin_count.load(Ordering::Acquire) }) == 0
            && !unsafe { (*bf).header.core.dirty.load(Ordering::Relaxed) }
            && (unsafe { (*bf).header.core.state.load(Ordering::Relaxed) }) == FrameState::Evicting
    }

    #[cfg(not(miri))]
    fn free_evicting_frame(&self, bf: *mut BufferFrame) {
        let pid = unsafe { (*bf).header.core.pid };
        self.page_table.remove(pid);
        unsafe {
            (*bf).header.parent_link = ParentLink::None;
            (*bf).header.core.state.transition(
                FrameState::Evicting,
                FrameState::Free,
                Ordering::Relaxed,
            );
        }

        self.metrics
            .eviction_events
            .inc(BufferPoolEvictionEvent::Evictions);
        self.release_resident_budget(bf);
        // Recycle the frame to the FreeList (eliminates MADV_DONTNEED).
        self.state.free_list.push(bf);
    }

    /// Unswizzle the parent pointer and free the frame. Returns true if
    /// eviction succeeded (frame freed), false if it had to be reverted
    /// to Resident. Uses non-blocking latch acquisition on the parent
    /// to avoid deadlock when multiple frames are batch-evicted.
    #[cfg(not(miri))]
    fn unswizzle_and_free(&self, bf: *mut BufferFrame, pid: u64) -> bool {
        let parent_updated = self.unswizzle_parent(bf, pid);

        if !self.parent_link_allows_free(bf, parent_updated) {
            Self::revert_frame_to_resident(bf);
            return false;
        }

        // Block new hot pins only for the final free window. DFS parent
        // discovery can run without stopping the world; once the parent
        // swip is updated, we only need to ensure no in-flight hot pin
        // observes the Evicting-to-Free transition.
        self.eviction_writer_pending.fetch_add(1, Ordering::AcqRel);
        let _eviction_guard = self.eviction_mu.write();
        self.eviction_writer_pending.fetch_sub(1, Ordering::AcqRel);

        if !Self::can_free_evicting_frame(bf) {
            Self::revert_frame_to_resident(bf);
            return false;
        }

        self.free_evicting_frame(bf);
        true
    }

    /// Pick an unpinned resident frame, write it back if dirty, update its parent
    /// swip to EVICTED, and return the frame pointer.
    ///
    /// Scans from a random start looking for resident frames with pin_count=0.
    /// Panics if no frame is evictable.
    /// Update the parent's routing edge from HOT(bf) → EVICTED(pid).
    /// Returns true if the parent was successfully updated or no update
    /// was needed (Stable/None). Returns false for InnerNode if the
    /// parent couldn't be found or latched.
    #[cfg(not(miri))]
    fn unswizzle_parent(&self, bf: *mut BufferFrame, pid: u64) -> bool {
        self.unswizzle_parent_link(bf, pid, Self::frame_parent_link(bf))
    }

    #[cfg(not(miri))]
    fn unswizzle_parent_link(&self, bf: *mut BufferFrame, pid: u64, link: ParentLink) -> bool {
        match link {
            ParentLink::None if Self::frame_is_index_page(bf) => {
                // Sibling/orphan loads can install a reachable leaf without a
                // cached parent hint. Discover and unswizzle the live parent
                // edge before allowing the index frame to be recycled.
                self.metrics
                    .unswizzle_parent_events
                    .inc(UnswizzleParentEvent::DfsFallbacks);
                self.find_and_unswizzle_with_all_registered_finders(
                    unsafe { BufferFrameRef::from_raw(bf) },
                    pid,
                )
            }
            ParentLink::None => true,
            ParentLink::Unswizzled => true,
            ParentLink::Stable(edge) => {
                edge.store_evicted(pid);
                true
            }
            ParentLink::InnerNode(link) => self.try_unswizzle_inner_node(
                unsafe { BufferFrameRef::from_raw(bf) },
                pid,
                link.parent_pid,
                link.slot_index,
                link.is_upper,
                link.dt_id,
            ),
        }
    }

    /// Try to unswizzle an InnerNode parent edge. Uses non-blocking
    /// latch on the parent to avoid deadlock during batch eviction.
    #[cfg(not(miri))]
    fn try_unswizzle_inner_node(
        &self,
        child: BufferFrameRef,
        pid: u64,
        parent_pid: u64,
        slot_index: u16,
        is_upper: bool,
        dt_id: u16,
    ) -> bool {
        if let Some(result) =
            self.try_unswizzle_inner_node_fast_path(child, pid, parent_pid, slot_index, is_upper)
        {
            return result;
        }

        let hinted_finder = self.dt_registry.lock().get(&dt_id).cloned();
        if let Some(finder) = hinted_finder
            && let Some(result) = finder.find_and_unswizzle_with_hint(
                unsafe { EvictingFrame::new(child) },
                pid,
                parent_pid,
                slot_index,
                is_upper,
            )
        {
            if result {
                self.metrics
                    .unswizzle_parent_events
                    .inc(UnswizzleParentEvent::FastPathHits);
            }
            return result;
        }

        self.metrics
            .unswizzle_parent_events
            .inc(UnswizzleParentEvent::DfsFallbacks);
        self.find_and_unswizzle_with_registered_finders(child, pid, dt_id)
    }

    #[cfg(not(miri))]
    fn try_unswizzle_inner_node_fast_path(
        &self,
        child: BufferFrameRef,
        pid: u64,
        parent_pid: u64,
        slot_index: u16,
        is_upper: bool,
    ) -> Option<bool> {
        // Use page_table to check if parent is resident.
        let parent_bf = self.page_table.lookup(parent_pid)?;
        // Inner index pages are not eviction candidates. Latch the stable
        // frame slot directly instead of passing through the global hot-pin
        // fence, which deliberately rejects new pins while an eviction writer
        // is pending. Revalidate state and PID under the latch to cover a
        // concurrent inner-page retirement and frame reuse.
        let Some(_guard) = (unsafe { (*parent_bf).latch.try_lock_exclusive() }) else {
            self.metrics
                .unswizzle_parent_events
                .inc(UnswizzleParentEvent::FastPathLatchMisses);
            return Some(false);
        };
        if unsafe { (*parent_bf).header.core.state.load(Ordering::Acquire) } != FrameState::Resident
            || unsafe { (*parent_bf).header.core.pid } != parent_pid
        {
            return None;
        }

        let edge = (unsafe {
            self.find_child_pos_in_parent(parent_bf, child.as_ptr(), pid, slot_index, is_upper)
        })?;

        self.metrics
            .unswizzle_parent_events
            .inc(UnswizzleParentEvent::FastPathHits);
        unsafe { self.unswizzle_parent_child_at(parent_bf, edge, pid) };
        unsafe {
            (*parent_bf)
                .header
                .core
                .dirty
                .store(true, Ordering::Relaxed)
        };
        Some(true)
    }

    #[cfg(not(miri))]
    fn find_and_unswizzle_with_registered_finders(
        &self,
        child: BufferFrameRef,
        pid: u64,
        dt_id: u16,
    ) -> bool {
        let hinted_finder = self.dt_registry.lock().get(&dt_id).cloned();
        if let Some(finder) = hinted_finder
            && finder.find_and_unswizzle(unsafe { EvictingFrame::new(child) }, pid)
        {
            self.metrics
                .unswizzle_parent_events
                .inc(UnswizzleParentEvent::DfsSuccesses);
            return true;
        }

        // If the hint is missing or stale (e.g., after reopen/rebind), search all
        // registered DTs as a bounded fallback. This is uncommon in healthy runs,
        // but avoids getting stuck when a single registry lookup is no longer
        // sufficient for an evicting leaf.
        let fallback_finders = {
            let registry = self.dt_registry.lock();
            registry
                .iter()
                .filter_map(|(candidate_dt_id, finder)| {
                    (*candidate_dt_id != dt_id).then_some(Arc::clone(finder))
                })
                .collect::<Vec<_>>()
        };
        for finder in fallback_finders {
            if finder.find_and_unswizzle(unsafe { EvictingFrame::new(child) }, pid) {
                self.metrics
                    .unswizzle_parent_events
                    .inc(UnswizzleParentEvent::DfsSuccesses);
                return true;
            }
        }

        self.metrics
            .unswizzle_parent_events
            .inc(UnswizzleParentEvent::DfsFailures);
        false
    }

    #[cfg(not(miri))]
    fn find_and_unswizzle_with_all_registered_finders(
        &self,
        child: BufferFrameRef,
        pid: u64,
    ) -> bool {
        let finders = self
            .dt_registry
            .lock()
            .values()
            .map(Arc::clone)
            .collect::<Vec<_>>();
        for finder in finders {
            if finder.find_and_unswizzle(unsafe { EvictingFrame::new(child) }, pid) {
                self.metrics
                    .unswizzle_parent_events
                    .inc(UnswizzleParentEvent::DfsSuccesses);
                return true;
            }
        }

        self.metrics
            .unswizzle_parent_events
            .inc(UnswizzleParentEvent::DfsFailures);
        false
    }

    #[cfg(not(miri))]
    unsafe fn find_child_pos_in_parent(
        &self,
        parent_bf: *mut BufferFrame,
        child_bf: *mut BufferFrame,
        child_pid: u64,
        hinted_slot: u16,
        hinted_upper: bool,
    ) -> Option<ParentChildEdge> {
        if page_header::read_page_type(unsafe { (*parent_bf).page_bytes() }) != PageType::Index {
            return None;
        }

        let expected = Swip::hot(child_bf).raw();
        let evicted = Swip::evicted(child_pid).raw();

        let hinted_edge = ParentChildEdge::new(hinted_slot, hinted_upper);
        if let Some(raw) = unsafe { hinted_edge.read_raw(parent_bf) }
            && (raw == expected || raw == evicted)
        {
            return Some(hinted_edge);
        }

        let sp = crate::slotted_page::SlottedPage::from_page(unsafe { &(*parent_bf).page });
        let count = sp.num_slots();
        for pos in 0..count {
            let edge = ParentChildEdge::new(pos, false);
            if (hinted_upper || pos != hinted_slot)
                && let Some(raw) = unsafe { edge.read_raw(parent_bf) }
                && (raw == expected || raw == evicted)
            {
                return Some(edge);
            }
        }
        let upper_edge = ParentChildEdge::new(count, true);
        if !hinted_upper
            && let Some(raw) = unsafe { upper_edge.read_raw(parent_bf) }
            && (raw == expected || raw == evicted)
        {
            return Some(upper_edge);
        }
        None
    }

    #[cfg(not(miri))]
    unsafe fn unswizzle_parent_child_at(
        &self,
        parent_bf: *mut BufferFrame,
        edge: ParentChildEdge,
        child_pid: u64,
    ) {
        unsafe { edge.write_evicted(parent_bf, child_pid) };
    }

    // -- stats for testing --

    pub fn num_frames(&self) -> usize {
        self.resident_base_pages
    }

    pub fn num_slots(&self) -> usize {
        self.state.arena.len
    }

    pub fn num_occupied(&self) -> usize {
        (0..self.state.arena.len)
            .filter(|&i| {
                if !self.is_slot_initialized(i) {
                    return false;
                }
                let s = unsafe {
                    (*self.raw_frame(i))
                        .header
                        .core
                        .state
                        .load(Ordering::Relaxed)
                };
                s != FrameState::Free
            })
            .count()
    }

    pub fn num_occupied_estimate(&self) -> usize {
        self.resident_base_pages
            .saturating_sub(self.resident_base_pages_available.load(Ordering::Relaxed))
    }

    /// Count resident pages without a cached parent link.
    ///
    /// A page in this state can still be reachable from a parent SWIP. Eviction
    /// must use the registered data structure's parent finder before freeing an
    /// unlinked index page.
    pub fn num_unlinked_resident_frames(&self) -> usize {
        (0..self.allocated_slots())
            .filter(|&idx| self.is_slot_initialized(idx))
            .filter(|&idx| {
                let bf = self.raw_frame(idx);
                unsafe {
                    (*bf).header.core.state.load(Ordering::Acquire) == FrameState::Resident
                        && (*bf).header.core.pid != 0
                        && matches!((*bf).header.parent_link, ParentLink::None)
                }
            })
            .count()
    }

    /// Advise the kernel to drop cached pages for the underlying store.
    pub fn drop_cache(&self) {
        self.page_store.drop_cache()
    }

    pub fn num_pages_on_disk(&self) -> usize {
        self.page_store.len()
    }

    pub fn eviction_count(&self) -> u64 {
        self.metrics
            .eviction_events
            .get(BufferPoolEvictionEvent::Evictions) as u64
    }

    /// Detailed cache activity for benchmark diagnostics.
    pub fn diagnostic_stats(&self) -> BufferPoolDiagnosticStats {
        let loaded = |kind| {
            (self.metrics.fix_swip_sync_load_pages.get(kind)
                + self.metrics.fix_orphan_sync_load_pages.get(kind)) as u64
        };
        let mut resident_frames = 0u64;
        let mut pinned_frames = 0u64;
        let mut dirty_frames = 0u64;
        let mut referenced_frames = 0u64;
        let mut eviction_allowed_frames = 0u64;
        for idx in 0..self.allocated_slots() {
            let bf = self.raw_frame(idx);
            if unsafe { (*bf).header.core.state.load(Ordering::Relaxed) } != FrameState::Resident {
                continue;
            }
            resident_frames += 1;
            pinned_frames +=
                u64::from(unsafe { (*bf).header.core.pin_count.load(Ordering::Relaxed) } > 0);
            dirty_frames += u64::from(unsafe { (*bf).header.core.dirty.load(Ordering::Relaxed) });
            referenced_frames +=
                u64::from(unsafe { (*bf).header.core.referenced.load(Ordering::Relaxed) });
            #[cfg(not(miri))]
            {
                let eviction_allowed = Self::frame_page_allows_eviction(bf);
                eviction_allowed_frames += u64::from(eviction_allowed);
            }
        }
        BufferPoolDiagnosticStats {
            inner_index_loads: loaded(BufferPoolLoadedPageKind::InnerIndex),
            leaf_index_loads: loaded(BufferPoolLoadedPageKind::LeafIndex),
            tuple_loads: loaded(BufferPoolLoadedPageKind::Tuple),
            delta_loads: loaded(BufferPoolLoadedPageKind::Delta),
            resident_meta_loads: loaded(BufferPoolLoadedPageKind::ResidentMeta),
            unknown_loads: loaded(BufferPoolLoadedPageKind::Unknown),
            parent_hint_hits: self
                .metrics
                .unswizzle_parent_events
                .get(UnswizzleParentEvent::FastPathHits) as u64,
            parent_hint_latch_misses: self
                .metrics
                .unswizzle_parent_events
                .get(UnswizzleParentEvent::FastPathLatchMisses)
                as u64,
            parent_dfs_fallbacks: self
                .metrics
                .unswizzle_parent_events
                .get(UnswizzleParentEvent::DfsFallbacks) as u64,
            parent_dfs_failures: self
                .metrics
                .unswizzle_parent_events
                .get(UnswizzleParentEvent::DfsFailures) as u64,
            second_chance_skips: self
                .metrics
                .eviction_events
                .get(BufferPoolEvictionEvent::SecondChanceSkips)
                as u64,
            resident_frames,
            pinned_frames,
            dirty_frames,
            referenced_frames,
            eviction_allowed_frames,
            free_list_frames: self.state.free_list.len() as u64,
            resident_budget_available: self.resident_base_pages_available.load(Ordering::Relaxed)
                as u64,
            eviction_in_flight: self.eviction_in_flight.load(Ordering::Relaxed) as u64,
            dirty_flush_batches: self.metrics.dirty_flush_batches.sum().max(0) as u64,
            dirty_flush_pages: self.metrics.dirty_flush_pages.sum().max(0) as u64,
            dirty_flush_wal_wait_ns: self.metrics.dirty_flush_wal_wait_ns.sum().max(0) as u64,
            dirty_flush_data_write_ns: self.metrics.dirty_flush_data_write_ns.sum().max(0) as u64,
            dirty_flush_cleaned_pages: self.metrics.dirty_flush_cleaned_pages.sum().max(0) as u64,
            dirty_flush_stale_pages: self.metrics.dirty_flush_stale_pages.sum().max(0) as u64,
            dirty_wal_page_patch_records: self.metrics.dirty_wal_page_patch_records.sum().max(0)
                as u64,
            dirty_wal_page_patch_bytes: self.metrics.dirty_wal_page_patch_bytes.sum().max(0) as u64,
        }
    }

    /// Number of pages synchronously loaded into the pool since it opened.
    pub fn page_load_count(&self) -> u64 {
        const KINDS: [BufferPoolLoadedPageKind; 6] = [
            BufferPoolLoadedPageKind::InnerIndex,
            BufferPoolLoadedPageKind::LeafIndex,
            BufferPoolLoadedPageKind::Tuple,
            BufferPoolLoadedPageKind::Delta,
            BufferPoolLoadedPageKind::ResidentMeta,
            BufferPoolLoadedPageKind::Unknown,
        ];
        KINDS
            .into_iter()
            .map(|kind| {
                (self.metrics.fix_swip_sync_load_pages.get(kind)
                    + self.metrics.fix_orphan_sync_load_pages.get(kind)) as u64
            })
            .sum()
    }

    /// Write back all dirty resident pages to the page store and sync.
    ///
    /// For each dirty page, ensures its WAL record is durable before
    /// writing the page to the data file (same WAL-before-data rule as
    /// eviction). This is safe under concurrent writers: a page dirtied
    /// after flush starts will have its WAL record flushed on demand
    /// when that page is reached.
    ///
    /// An initial bulk WAL flush is done as an optimization to reduce
    /// per-page flush_at_least calls for pages already durable.
    pub fn flush(&self) -> std::io::Result<()> {
        // Optimization: flush all currently-buffered WAL records up front.
        // This makes most per-page flush_at_least calls no-ops.
        #[cfg(not(miri))]
        if let Some(ref wal) = self.wal {
            wal.flush();
        }

        // Collect dirty frames under exclusive latches.  Guards are held
        // for the duration so page data pointers remain valid through the
        // batched write_pages_and_sync call.
        let mut dirty: Vec<(ExclusiveGuard<'_>, *mut BufferFrame)> = Vec::new();

        for i in 0..self.state.arena.len {
            if !self.is_slot_initialized(i) {
                continue;
            }
            let bf = self.raw_frame(i);
            if unsafe { (*bf).header.core.state.load(Ordering::Relaxed) } == FrameState::Free {
                continue;
            }
            if !unsafe { (*bf).header.core.dirty.load(Ordering::Relaxed) } {
                continue;
            }
            let guard = unsafe { (*bf).latch.lock_exclusive() };
            // Re-check dirty under exclusive latch.
            if !unsafe { (*bf).header.core.dirty.load(Ordering::Relaxed) } {
                continue;
            }
            // WAL-before-data: ensure this page's WAL record is durable.
            #[cfg(not(miri))]
            if let Some(ref wal) = self.wal {
                let page_lsn = unsafe { (*bf).header.core.page_lsn.load(Ordering::Relaxed) };
                if page_lsn > 0 {
                    wal.flush_at_least(page_lsn);
                }
            }
            dirty.push((guard, bf));
        }

        if dirty.is_empty() {
            self.promote_pending_reusable_extents();
            return Ok(());
        }

        let mut page_copies: Vec<(u64, AlignedPageCopy)> = dirty
            .iter()
            .map(|(_guard, bf)| unsafe {
                let pid = (**bf).header.core.pid;
                let mut copy = AlignedPageCopy::copy_from((**bf).page_bytes());
                prepare_page_copy_for_writeback(copy.as_mut_slice(), self);
                (pid, copy)
            })
            .collect();
        let pages: Vec<(u64, &[u8])> = page_copies
            .iter_mut()
            .map(|(pid, copy)| (*pid, copy.as_slice()))
            .collect();

        self.page_store.write_pages_and_sync(&pages)?;

        // Clear dirty flags now that all pages are durable.
        for (_guard, bf) in &dirty {
            unsafe {
                (**bf).header.core.dirty.store(false, Ordering::Relaxed);
                (**bf)
                    .header
                    .core
                    .wal_buffer_epoch
                    .store(0, Ordering::Relaxed);
                (**bf)
                    .header
                    .core
                    .wal_buffer_offset
                    .store(0, Ordering::Relaxed);
            };
        }

        self.promote_pending_reusable_extents();
        Ok(())
    }

    pub fn simulate_crash(&mut self) {
        self.page_provider.lock().unwrap().stop();
        #[cfg(not(miri))]
        {
            self.wal = None;
        }
    }
}

/// Snapshot of detailed buffer-pool activity since the pool was opened.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BufferPoolDiagnosticStats {
    pub inner_index_loads: u64,
    pub leaf_index_loads: u64,
    pub tuple_loads: u64,
    pub delta_loads: u64,
    pub resident_meta_loads: u64,
    pub unknown_loads: u64,
    pub parent_hint_hits: u64,
    pub parent_hint_latch_misses: u64,
    pub parent_dfs_fallbacks: u64,
    pub parent_dfs_failures: u64,
    pub second_chance_skips: u64,
    pub resident_frames: u64,
    pub pinned_frames: u64,
    pub dirty_frames: u64,
    pub referenced_frames: u64,
    pub eviction_allowed_frames: u64,
    pub free_list_frames: u64,
    pub resident_budget_available: u64,
    pub eviction_in_flight: u64,
    pub dirty_flush_batches: u64,
    pub dirty_flush_pages: u64,
    pub dirty_flush_wal_wait_ns: u64,
    pub dirty_flush_data_write_ns: u64,
    pub dirty_flush_cleaned_pages: u64,
    pub dirty_flush_stale_pages: u64,
    pub dirty_wal_page_patch_records: u64,
    pub dirty_wal_page_patch_bytes: u64,
}

fn buffer_pool_latency_bounds_ns() -> [u64; 13] {
    [
        250,
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
        500_000_000,
    ]
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        self.prefetch_workers.lock().unwrap().stop();

        // Stop the page provider thread before flushing.
        self.page_provider.lock().unwrap().stop();
        // Best-effort on drop — caller should use flush() explicitly
        // when durability matters.
        let _ = self.flush();

        // Arena::drop unmaps raw frame storage without running BufferFrame
        // destructors. Release every owning stable backlink explicitly first.
        for idx in 0..self.state.arena.len {
            let bf = self.raw_frame(idx);
            unsafe {
                let old = std::mem::replace(&mut (*bf).header.parent_link, ParentLink::None);
                if let ParentLink::Stable(owner) = &old {
                    owner.store_evicted((*bf).header.core.pid);
                }
                drop(old);
            }
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::buffer_frame::physical_page_number;

    #[cfg(not(miri))]
    struct BlockingPageStore {
        inner: InMemoryPageStore,
        write_started: std::sync::Mutex<Option<std::sync::mpsc::Sender<()>>>,
        release: (std::sync::Mutex<bool>, std::sync::Condvar),
    }

    #[cfg(not(miri))]
    impl BlockingPageStore {
        fn new(write_started: std::sync::mpsc::Sender<()>) -> Self {
            Self {
                inner: InMemoryPageStore::new(),
                write_started: std::sync::Mutex::new(Some(write_started)),
                release: (std::sync::Mutex::new(false), std::sync::Condvar::new()),
            }
        }

        fn release_write(&self) {
            *self.release.0.lock().unwrap() = true;
            self.release.1.notify_all();
        }
    }

    #[cfg(not(miri))]
    impl PageStore for BlockingPageStore {
        fn read_page(&self, pid: PageId, buf: &mut [u8]) -> std::io::Result<bool> {
            self.inner.read_page(pid, buf)
        }

        fn write_page(&self, pid: PageId, data: &[u8]) -> std::io::Result<()> {
            if let Some(started) = self.write_started.lock().unwrap().take() {
                started.send(()).unwrap();
                let mut released = self.release.0.lock().unwrap();
                while !*released {
                    released = self.release.1.wait(released).unwrap();
                }
            }
            self.inner.write_page(pid, data)
        }

        fn allocate(&self, pid: PageId) -> std::io::Result<()> {
            self.inner.allocate(pid)
        }

        fn sync(&self) -> std::io::Result<()> {
            self.inner.sync()
        }

        fn len(&self) -> usize {
            self.inner.len()
        }

        fn next_page_id(&self) -> PageId {
            self.inner.next_page_id()
        }
    }

    #[cfg(not(miri))]
    struct FailingWritePageStore {
        inner: InMemoryPageStore,
        fail_writes: Arc<std::sync::atomic::AtomicBool>,
    }

    #[cfg(not(miri))]
    impl FailingWritePageStore {
        fn new(fail_writes: Arc<std::sync::atomic::AtomicBool>) -> Self {
            Self {
                inner: InMemoryPageStore::new(),
                fail_writes,
            }
        }
    }

    #[cfg(not(miri))]
    impl PageStore for FailingWritePageStore {
        fn read_page(&self, pid: PageId, buf: &mut [u8]) -> std::io::Result<bool> {
            self.inner.read_page(pid, buf)
        }

        fn write_page(&self, pid: PageId, data: &[u8]) -> std::io::Result<()> {
            if self.fail_writes.load(Ordering::Acquire) {
                return Err(std::io::Error::other("injected dirty writeback failure"));
            }
            self.inner.write_page(pid, data)
        }

        fn allocate(&self, pid: PageId) -> std::io::Result<()> {
            self.inner.allocate(pid)
        }

        fn sync(&self) -> std::io::Result<()> {
            self.inner.sync()
        }

        fn len(&self) -> usize {
            self.inner.len()
        }

        fn next_page_id(&self) -> PageId {
            self.inner.next_page_id()
        }
    }

    // -----------------------------------------------------------------------
    // Single-threaded tests (verify &self API)
    // -----------------------------------------------------------------------

    #[test]
    fn allocate_and_fix() {
        let pool = BufferPool::new(4);
        let swip = pool.allocate_page();
        assert!(swip.load(Ordering::Relaxed).is_evicted());

        let bf = pool.fix_stable(&swip, unsafe { NoLatches::new(&pool) });
        let raw = bf.raw();
        assert!(swip.load(Ordering::Relaxed).is_hot());
        assert_eq!(
            unsafe { (*raw).header.core.pin_count.load(Ordering::Relaxed) },
            1
        );
        assert_eq!(bf.pid(), 1);
        drop(bf);
        assert_eq!(
            unsafe { (*raw).header.core.pin_count.load(Ordering::Relaxed) },
            0
        );
    }

    #[test]
    fn dropping_new_stable_page_aborts_unpublished_allocation() {
        let pool = BufferPool::new(2);
        let pid = {
            let page = pool.allocate_stable(unsafe { NoLatches::new(&pool) });
            let pid = page.pid();
            assert!(pool.page_table.lookup(pid).is_some());
            pid
        };

        assert!(
            pool.page_table.lookup(pid).is_none(),
            "aborted unpublished page must leave no page-table entry"
        );
        assert_eq!(pool.num_occupied(), 0);
        assert_eq!(
            pool.resident_base_pages_available.load(Ordering::Acquire),
            pool.resident_base_pages
        );
        assert!(
            pool.pending_reusable_extents.lock().is_empty(),
            "unpublished abort must not enter durable retirement accounting"
        );
    }

    #[test]
    fn dropping_new_unlinked_page_aborts_without_durable_retirement() {
        let pool = BufferPool::new(2);
        let pid = {
            let page = pool.allocate_unlinked(unsafe { NoLatches::new(&pool) });
            let pid = page.pid();
            assert!(pool.page_table.lookup(pid).is_some());
            pid
        };

        assert!(pool.page_table.lookup(pid).is_none());
        assert_eq!(pool.num_occupied(), 0);
        assert_eq!(
            pool.resident_base_pages_available.load(Ordering::Acquire),
            pool.resident_base_pages
        );
        assert!(pool.pending_reusable_extents.lock().is_empty());
    }

    #[test]
    fn panicking_unlinked_construction_aborts_the_page() {
        let pool = BufferPool::new(1);
        let mut pid = 0;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let page = pool.allocate_unlinked(unsafe { NoLatches::new(&pool) });
            pid = page.pid();
            panic!("construction failed");
        }));

        assert!(result.is_err());
        assert!(pool.page_table.lookup(pid).is_none());
        assert_eq!(pool.num_occupied(), 0);
    }

    #[test]
    fn frame_id_generation_changes_when_slot_is_reused() {
        let pool = BufferPool::new(1);
        let first = pool.allocate_stable(unsafe { NoLatches::new(&pool) });
        let first_id = first
            .frame
            .as_ref()
            .expect("new page must retain its frame")
            .pinned
            .id();
        drop(first);

        let second = pool.allocate_stable(unsafe { NoLatches::new(&pool) });
        let second_id = second
            .frame
            .as_ref()
            .expect("new page must retain its frame")
            .pinned
            .id();

        assert_eq!(first_id.slot(), second_id.slot());
        assert_ne!(first_id.generation(), second_id.generation());
        assert_ne!(first_id, second_id, "reused slots need distinct identities");
    }

    #[test]
    fn installing_new_stable_page_publishes_owner_before_unpin() {
        let pool = BufferPool::new(2);
        let mut slot = None;
        let pid = {
            let mut page = pool.allocate_stable(unsafe { NoLatches::new(&pool) });
            page.page_mut()[64] = 0x5a;
            page.mark_dirty();
            let pid = page.pid();
            page.install(&mut slot);
            pid
        };
        let edge = slot.as_ref().expect("stable owner slot must be installed");
        assert_eq!(edge.page_id(), pid);

        let frame = pool
            .fix_stable(edge, unsafe { NoLatches::new(&pool) })
            .shared();
        assert_eq!(frame.page()[64], 0x5a);
        assert!(matches!(
            unsafe { &(*frame.frame_ref().as_ptr()).header.parent_link },
            ParentLink::Stable(owner) if Arc::ptr_eq(owner, edge.owner())
        ));
    }

    #[test]
    #[should_panic(expected = "stable SWIP belongs to a different buffer pool")]
    fn cross_pool_stable_fix_is_rejected_before_pointer_decode() {
        let first = BufferPool::new(2);
        let second = BufferPool::new(2);
        let edge = first.allocate_page();

        drop(second.fix_stable(&edge, unsafe { NoLatches::new(&second) }));
    }

    #[test]
    fn frame_keeps_stable_owner_alive_until_eviction() {
        let pool = BufferPool::new(1);
        let edge = pool.allocate_page();
        let frame = pool.fix_stable(&edge, unsafe { NoLatches::new(&pool) });
        let raw = frame.raw();
        let owner = Arc::downgrade(edge.owner());
        drop(frame);
        drop(edge);
        unsafe {
            (*raw)
                .header
                .core
                .referenced
                .store(false, Ordering::Relaxed);
        }

        assert!(owner.upgrade().is_some());
        assert!(pool.try_evict_one());
        assert!(
            owner.upgrade().is_none(),
            "eviction must release the frame's final stable-owner reference"
        );
    }

    #[test]
    fn hot_stable_fix_does_not_touch_owner_refcount() {
        let pool = BufferPool::new(2);
        let edge = pool.allocate_page();
        drop(pool.fix_stable(&edge, unsafe { NoLatches::new(&pool) }));
        let strong_before = Arc::strong_count(edge.owner());

        for _ in 0..32 {
            let frame = pool.fix_stable(&edge, unsafe { NoLatches::new(&pool) });
            assert_eq!(
                Arc::strong_count(edge.owner()),
                strong_before,
                "HOT fix must borrow the stable owner without cloning its Arc"
            );
            drop(frame);
            assert_eq!(
                Arc::strong_count(edge.owner()),
                strong_before,
                "dropping a HOT pin must not drop a stable-owner Arc"
            );
        }
    }

    #[test]
    fn try_fix_stable_is_resident_only_and_refcount_neutral() {
        let pool = BufferPool::new(1);
        let edge = pool.allocate_page();

        assert!(
            pool.try_fix_stable(&edge).is_none(),
            "resident-only stable fix must not fault an evicted edge"
        );

        drop(pool.fix_stable(&edge, unsafe { NoLatches::new(&pool) }));
        let strong_before = Arc::strong_count(edge.owner());
        let frame = pool
            .try_fix_stable(&edge)
            .expect("resident stable edge should pin without faulting");
        assert_eq!(frame.pid(), edge.page_id());
        assert_eq!(Arc::strong_count(edge.owner()), strong_before);
        drop(frame);
        assert_eq!(Arc::strong_count(edge.owner()), strong_before);
    }

    #[test]
    fn concurrent_stable_edges_survive_repeated_eviction_and_reload() {
        let pool = Arc::new(BufferPool::new(2));
        let mut slots = Vec::new();
        for value in 0..6u8 {
            let mut slot = None;
            let mut page = pool.allocate_stable(unsafe { NoLatches::new(&pool) });
            page.page_mut()[0] = value;
            page.mark_dirty();
            page.install(&mut slot);
            pool.flush().expect("stable-page setup flush failed");
            slots.push(slot.expect("stable page must install its owner"));
        }
        let edges = Arc::new(slots);
        let start = Arc::new(std::sync::Barrier::new(4));

        std::thread::scope(|scope| {
            for thread_index in 0..4usize {
                let pool = Arc::clone(&pool);
                let edges = Arc::clone(&edges);
                let start = Arc::clone(&start);
                scope.spawn(move || {
                    start.wait();
                    for iteration in 0..100usize {
                        let index = (thread_index * 5 + iteration * 3) % edges.len();
                        let frame = pool
                            .fix_stable(&edges[index], unsafe { NoLatches::new(&pool) })
                            .shared();
                        assert_eq!(frame.page()[0], index as u8);
                    }
                });
            }
        });

        assert!(
            edges
                .iter()
                .any(|edge| edge.load(Ordering::Acquire).is_evicted()),
            "tight resident budget should evict at least one stable edge"
        );
    }

    #[test]
    fn pool_drop_releases_resident_stable_backlinks_before_unmap() {
        let owner = {
            let pool = BufferPool::new(1);
            let edge = pool.allocate_page();
            drop(pool.fix_stable(&edge, unsafe { NoLatches::new(&pool) }));
            let owner = Arc::downgrade(edge.owner());
            drop(edge);
            assert!(owner.upgrade().is_some());
            owner
        };

        assert!(
            owner.upgrade().is_none(),
            "pool teardown must clear owning backlinks before unmapping frames"
        );
    }

    #[test]
    fn stable_edge_surviving_pool_drop_is_evicted_before_unmap() {
        let (edge, pid) = {
            let pool = BufferPool::new(1);
            let edge = pool.allocate_page();
            let pid = edge.page_id();
            drop(pool.fix_stable(&edge, unsafe { NoLatches::new(&pool) }));
            (edge, pid)
        };

        let word = edge.load(Ordering::Acquire);
        assert!(word.is_evicted());
        assert_eq!(word.as_page_id(), pid);
        assert_eq!(edge.page_id(), pid);
    }

    #[test]
    fn retire_stable_consumes_owner_and_defers_page_reuse() {
        let pool = BufferPool::new(2);
        let edge = pool.allocate_page();
        let pid = edge.page_id();
        drop(pool.fix_stable(&edge, unsafe { NoLatches::new(&pool) }));

        let retired = unsafe { pool.retire_stable(edge, unsafe { NoLatches::new(&pool) }) };

        assert_eq!(retired, pid);
        assert!(pool.page_table.lookup(pid).is_none());
        assert!(
            pool.pending_reusable_extents
                .lock()
                .iter()
                .any(|extent| extent.start_page_number == physical_page_number(pid)),
            "published stable retirement must remain checkpoint-gated"
        );
    }

    #[test]
    fn try_pin_resident_child_does_not_fault_an_evicted_page() {
        let pool = BufferPool::new(4);
        let swip = pool.allocate_page();
        let evicted = swip.load(Ordering::Acquire);
        assert!(evicted.is_evicted());

        let pinned = unsafe { pool.try_pin_resident_child(evicted) };

        assert!(
            pinned.is_none(),
            "resident-only pinning must restart instead of loading an evicted child"
        );
        assert_eq!(
            swip.load(Ordering::Acquire),
            evicted,
            "resident-only pinning must not reswizzle an evicted edge"
        );
        assert!(
            pool.page_table.lookup(evicted.as_page_id()).is_none(),
            "resident-only pinning must not install a frame in the page table"
        );
    }

    #[test]
    fn allocate_page_reuses_promoted_extent_before_monotonic_growth() {
        let pool = BufferPool::new(4);
        let retired = pool.allocate_page();
        let retired_pid = retired.load(Ordering::Acquire).as_page_id();
        let high_water = pool.next_page_id.load(Ordering::Acquire);

        pool.promote_reusable_extent(FreeExtent::new(physical_page_number(retired_pid), 1));
        let reused = pool.allocate_page();

        assert_eq!(
            reused.load(Ordering::Acquire).as_page_id(),
            retired_pid,
            "buffer pool should allocate promoted reusable pages before growing the store"
        );
        assert_eq!(
            pool.next_page_id.load(Ordering::Acquire),
            high_water,
            "reusing a page must not advance the monotonic high-water mark"
        );
    }

    #[test]
    fn retire_unlinked_exclusive_frame_reuses_page_id_and_frame() {
        let pool = BufferPool::new(4);
        let (retired_pid, retired_frame) = pool.allocate_and_fix(unsafe { NoLatches::new(&pool) });
        let retired_raw = retired_frame.raw();
        let retired_frame = retired_frame.exclusive();

        let returned_pid = unsafe { pool.retire_unlinked_exclusive_frame(retired_frame) };
        assert_eq!(returned_pid, retired_pid);
        assert_eq!(
            unsafe { (*retired_raw).header.core.state.load(Ordering::Acquire) },
            FrameState::Free
        );
        assert_eq!(
            unsafe { (*retired_raw).header.core.pin_count.load(Ordering::Acquire) },
            0
        );

        let (before_flush_pid, before_flush_frame) =
            pool.allocate_and_fix(unsafe { NoLatches::new(&pool) });
        assert_ne!(
            before_flush_pid, retired_pid,
            "retired page id must not be reusable before the next flush"
        );
        drop(before_flush_frame);

        let high_water = pool.next_page_id.load(Ordering::Acquire);
        pool.flush().unwrap();

        let (reused_pid, reused_frame) = pool.allocate_and_fix(unsafe { NoLatches::new(&pool) });
        assert_eq!(reused_pid, retired_pid);
        // Under FreeList addressing, the frame pointer is decoupled from
        // the page ID — we only assert the page ID is reused, not the frame.
        assert_ne!(
            reused_frame.raw(),
            std::ptr::null_mut(),
            "reused frame must be valid"
        );
        assert_eq!(
            pool.next_page_id.load(Ordering::Acquire),
            high_water,
            "retiring and reusing a page must not advance the high-water mark"
        );
    }

    #[test]
    #[cfg(not(miri))]
    fn allocate_and_fix_reuses_promoted_extent() {
        let pool = BufferPool::new(32);
        let (retired_pid, retired_bf) = pool.allocate_and_fix(unsafe { NoLatches::new(&pool) });
        drop(retired_bf);
        let high_water = pool.next_page_id.load(Ordering::Acquire);

        // With a single page class, one retired page yields one base page.
        pool.promote_reusable_extent(FreeExtent::new(physical_page_number(retired_pid), 1));
        let (reused_pid, reused_bf) = pool.allocate_and_fix(unsafe { NoLatches::new(&pool) });

        assert_eq!(
            reused_pid, retired_pid,
            "allocation should reuse a matching promoted extent"
        );
        assert_eq!(
            pool.next_page_id.load(Ordering::Acquire),
            high_water,
            "reuse must not advance the monotonic high-water mark"
        );
        assert_eq!(reused_bf.pid(), retired_pid);
    }

    #[test]
    fn fix_hot_increments_pin() {
        let pool = BufferPool::new(4);
        let swip = pool.allocate_page();

        let bf = pool.fix_stable(&swip, unsafe { NoLatches::new(&pool) });
        let raw = bf.raw();
        assert_eq!(
            unsafe { (*raw).header.core.pin_count.load(Ordering::Relaxed) },
            1
        );

        let bf2 = pool.fix_stable(&swip, unsafe { NoLatches::new(&pool) });
        assert_eq!(bf.raw(), bf2.raw());
        assert_eq!(
            unsafe { (*raw).header.core.pin_count.load(Ordering::Relaxed) },
            2
        );
    }

    #[test]
    fn fix_hot_rescues_evicting_frame() {
        let pool = BufferPool::new(4);
        let swip = pool.allocate_page();

        unsafe {
            let bf = pool.fix_stable(&swip, unsafe { NoLatches::new(&pool) });
            let raw = bf.raw();
            let pid = bf.pid();
            drop(bf);

            assert!(swip.load(Ordering::Acquire).is_hot());
            {
                let _guard = (*raw).latch.lock_exclusive();
                (*raw).header.core.state.transition(
                    FrameState::Resident,
                    FrameState::Evicting,
                    Ordering::Release,
                );
                (*raw).header.core.pin_count.store(0, Ordering::Relaxed);
                (*raw)
                    .header
                    .core
                    .referenced
                    .store(false, Ordering::Relaxed);
            }

            let rescued = pool.fix_stable(&swip, unsafe { NoLatches::new(&pool) });
            assert_eq!(
                rescued.raw(),
                raw,
                "hot fix should reuse the evicting frame"
            );
            assert_eq!(rescued.pid(), pid);
            assert_eq!(
                (*rescued.raw()).header.core.state.load(Ordering::Acquire),
                FrameState::Resident,
                "hot fix should abort eviction instead of waiting on it",
            );
            assert_eq!(
                (*rescued.raw())
                    .header
                    .core
                    .pin_count
                    .load(Ordering::Acquire),
                1
            );
        }
    }

    #[test]
    fn fix_reswizzles_stable_resident_page() {
        let pool = BufferPool::new(4);
        let swip = pool.allocate_page();

        unsafe {
            let bf = pool.fix_stable(&swip, unsafe { NoLatches::new(&pool) });
            let raw = bf.raw();
            let pid = bf.pid();
            drop(bf);

            swip.store(Swip::evicted(pid), Ordering::Release);
            assert!(swip.load(Ordering::Acquire).is_evicted());

            let bf2 = pool.fix_stable(&swip, unsafe { NoLatches::new(&pool) });
            assert_eq!(raw, bf2.raw(), "resident page should be reused");
            let current = swip.load(Ordering::Acquire);
            assert!(current.is_hot(), "stable swip should re-swizzle to HOT");
            assert_eq!(current.as_ptr::<BufferFrame>(), bf2.raw());
            match &(*bf2.raw()).header.parent_link {
                ParentLink::Stable(edge) => {
                    assert!(
                        Arc::ptr_eq(edge, swip.owner()),
                        "stable parent link should point back to the swip",
                    );
                }
                _ => panic!("expected stable parent link after re-swizzle"),
            }
        }
    }

    #[test]
    #[cfg(not(miri))]
    fn page_id_load_is_directly_evictable_until_reswizzled() {
        struct AcceptOrphan;

        impl ParentFinder for AcceptOrphan {
            fn find_and_unswizzle(&self, _child: EvictingFrame<'_>, _child_pid: u64) -> bool {
                true
            }
        }

        let pool = BufferPool::new(1);
        pool.register_dt(0, Arc::new(AcceptOrphan));
        let (pid, frame) = pool.allocate_and_fix(unsafe { NoLatches::new(&pool) });
        let raw = frame.raw();
        let mut frame = frame.exclusive();
        let page = frame.page_bytes_mut();
        let sp = crate::slotted_page::SlottedPage::init(page.try_into().unwrap());
        sp.set_flag(1 << 1);
        page_header::write_page_type(page, PageType::Index);
        frame.mark_dirty();
        drop(frame);
        assert_eq!(pool.try_flush_dirty_batch(1).unwrap(), 1);
        unsafe {
            (*raw)
                .header
                .core
                .referenced
                .store(false, Ordering::Relaxed);
        }
        assert!(pool.try_evict_one());
        pool.unregister_dt(0);

        let frame = unsafe { pool.fix_orphan_frame(pid, unsafe { NoLatches::new(&pool) }) };
        assert!(matches!(
            unsafe { (*frame.raw()).header.parent_link.clone() },
            ParentLink::Unswizzled
        ));
        let raw = frame.raw();
        drop(frame);
        unsafe {
            (*raw)
                .header
                .core
                .referenced
                .store(false, Ordering::Relaxed);
        }
        assert!(
            pool.try_evict_one(),
            "a page-ID load has no hot owner edge to unswizzle"
        );
    }

    #[test]
    #[cfg(not(miri))]
    fn dirty_write_back() {
        let pool = BufferPool::new(1);
        let swip = pool.allocate_page();

        let mut bf = pool
            .fix_stable(&swip, unsafe { NoLatches::new(&pool) })
            .exclusive();
        bf.page_mut()[0] = 42;
        bf.page_mut()[4095] = 99;
        bf.mark_dirty();
        drop(bf);

        // Allocate another page — forces eviction of the first.
        let swip2 = pool.allocate_page();
        let bf2 = pool.fix_stable(&swip2, unsafe { NoLatches::new(&pool) });
        assert!(swip.load(Ordering::Relaxed).is_evicted());
        drop(bf2);

        // Fix it again — should reload from store with our data.
        let bf = pool
            .fix_stable(&swip, unsafe { NoLatches::new(&pool) })
            .shared();
        assert_eq!(bf.page()[0], 42);
        assert_eq!(bf.page()[4095], 99);
    }

    #[test]
    #[cfg(not(miri))]
    fn page_frame_roundtrips_through_evict_and_reload() {
        let pool = BufferPool::new(1);
        let (pid, bf) = pool.allocate_and_fix(unsafe { NoLatches::new(&pool) });

        assert_eq!(pid, 1);
        let raw = bf.raw();
        let mut bf = bf.exclusive();
        let page = bf.page_bytes_mut();
        assert_eq!(page.len(), PAGE_SIZE);
        page[0] = 11;
        page[PAGE_SIZE - 1] = 33;
        bf.mark_dirty();
        drop(bf);
        unsafe {
            (*raw)
                .header
                .core
                .referenced
                .store(false, Ordering::Relaxed);
        }

        assert!(
            !pool.try_evict_one(),
            "dirty pages must be cleaned before eviction"
        );
        assert_eq!(pool.try_flush_dirty_batch(1).unwrap(), 1);
        assert!(pool.try_evict_one(), "clean page should be evictable");
        assert_eq!(
            pool.approx_available_budget(),
            1,
            "evicting one page should return its full base-page budget"
        );

        let bf = unsafe { pool.fix_orphan_frame(pid, unsafe { NoLatches::new(&pool) }) }.shared();
        let page = bf.page_bytes();
        assert_eq!(page[0], 11);
        assert_eq!(page[PAGE_SIZE - 1], 33);
    }

    #[test]
    #[cfg(not(miri))]
    fn page_can_be_marked_dirty_with_logical_lsn() {
        let pool = BufferPool::new(1);
        let (pid, bf) = pool.allocate_and_fix(unsafe { NoLatches::new(&pool) });

        let raw = bf.raw();
        let mut bf = bf.exclusive();
        let page = bf.page_bytes_mut();
        // Avoid bytes [0, 8): the page-LSN field lives there and is
        // overwritten by mark_dirty_with_lsn.
        page[64] = 17;
        bf.mark_dirty_with_lsn(123);
        assert_eq!(
            unsafe { (*raw).header.core.page_lsn.load(Ordering::Relaxed) },
            123,
            "frame LSN should track the logical WAL record"
        );
        assert_eq!(
            page_header::read_page_lsn(bf.page_bytes()),
            123,
            "page image should carry the logical WAL LSN for checkpoint/recovery"
        );
        drop(bf);
        unsafe {
            (*raw)
                .header
                .core
                .referenced
                .store(false, Ordering::Relaxed);
        }

        assert!(
            !pool.try_evict_one(),
            "logically logged dirty pages must be cleaned before eviction"
        );
        assert_eq!(pool.try_flush_dirty_batch(1).unwrap(), 1);
        assert!(pool.try_evict_one(), "clean page should be evictable");
        let bf = unsafe { pool.fix_orphan_frame(pid, unsafe { NoLatches::new(&pool) }) }.shared();
        let page = bf.page_bytes();
        assert_eq!(page[64], 17, "page payload should survive reload");
        assert_eq!(
            page_header::read_page_lsn(page),
            123,
            "page LSN should survive reload"
        );
    }

    #[test]
    #[cfg(not(miri))]
    fn dirty_betree_page_is_not_stolen_before_checkpoint_flush() {
        let pool = BufferPool::new(1);
        let (pid, bf) = pool.allocate_and_fix(unsafe { NoLatches::new(&pool) });

        let raw = bf.raw();
        let mut bf = bf.exclusive();
        let page = bf.page_bytes_mut();
        page_header::write_page_type(page, PageType::BeTreeLeaf);
        page[64] = 33;
        bf.mark_dirty_with_lsn(77);
        drop(bf);
        unsafe {
            (*raw)
                .header
                .core
                .referenced
                .store(false, Ordering::Relaxed);
        }

        assert!(
            !pool.try_evict_one(),
            "dirty B-e pages are logically logged and must not be stolen before checkpoint"
        );
        #[cfg(not(miri))]
        assert_eq!(
            pool.try_evict_batch(1),
            0,
            "batch eviction must honour the dirty B-e no-steal rule"
        );

        pool.flush().unwrap();
        unsafe {
            (*raw)
                .header
                .core
                .referenced
                .store(false, Ordering::Relaxed);
        }
        assert!(
            pool.try_evict_one(),
            "checkpoint-flushed B-e pages should become evictable"
        );

        let bf = unsafe { pool.fix_orphan_frame(pid, unsafe { NoLatches::new(&pool) }) }.shared();
        let page = bf.page_bytes();
        assert_eq!(page[64], 33, "checkpoint-flushed B-e page should reload");
        assert_eq!(page_header::read_page_lsn(page), 77);
    }

    #[test]
    #[cfg(not(miri))]
    fn dirty_betree_page_flush_batch_makes_page_evictable() {
        let pool = BufferPool::new(1);
        let (pid, bf) = pool.allocate_and_fix(unsafe { NoLatches::new(&pool) });

        let raw = bf.raw();
        let mut bf = bf.exclusive();
        let page = bf.page_bytes_mut();
        page_header::write_page_type(page, PageType::BeTreeLeaf);
        page[64] = 44;
        bf.mark_dirty_with_lsn(88);
        drop(bf);
        unsafe {
            (*raw)
                .header
                .core
                .referenced
                .store(false, Ordering::Relaxed);
        }

        assert!(
            !pool.try_evict_one(),
            "dirty B-e page should not be stolen before it is flushed"
        );
        assert_eq!(
            pool.try_flush_dirty_batch(1).unwrap(),
            1,
            "non-blocking dirty flush should clean the available B-e page"
        );
        unsafe {
            (*raw)
                .header
                .core
                .referenced
                .store(false, Ordering::Relaxed);
        }
        assert!(
            pool.try_evict_one(),
            "flushed B-e page should become evictable"
        );

        let bf = unsafe { pool.fix_orphan_frame(pid, unsafe { NoLatches::new(&pool) }) }.shared();
        let page = bf.page_bytes();
        assert_eq!(page[64], 44, "flushed B-e page should reload");
        assert_eq!(page_header::read_page_lsn(page), 88);
    }

    #[test]
    #[cfg(not(miri))]
    fn dirty_flush_releases_page_latch_during_store_io() {
        let (write_started_tx, write_started_rx) = std::sync::mpsc::channel();
        let store = Arc::new(BlockingPageStore::new(write_started_tx));
        let pool = Arc::new(BufferPool::with_store(1, Box::new(store.clone())));
        let swip = Arc::new(pool.allocate_page());

        let raw = {
            let mut frame = pool
                .fix_stable(&swip, unsafe { NoLatches::new(&pool) })
                .exclusive();
            frame.page_bytes_mut()[64] = 1;
            frame.mark_dirty_with_lsn(1);
            unsafe { frame.frame_ref() }.as_ptr()
        };

        let flush_pool = pool.clone();
        let flush = std::thread::spawn(move || flush_pool.try_flush_dirty_batch(1).unwrap());
        write_started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("dirty flush must reach page-store I/O");
        assert_eq!(
            pool.try_flush_dirty_batch(1).unwrap(),
            0,
            "a concurrent caller must not duplicate an in-flight cleaning batch"
        );

        let (writer_done_tx, writer_done_rx) = std::sync::mpsc::channel();
        let writer_pool = pool.clone();
        let writer_swip = swip.clone();
        let writer = std::thread::spawn(move || {
            let mut frame = writer_pool
                .fix_stable(&writer_swip, unsafe { NoLatches::new(&writer_pool) })
                .exclusive();
            frame.page_bytes_mut()[64] = 2;
            frame.mark_dirty_with_lsn(2);
            writer_done_tx.send(()).unwrap();
        });

        let writer_completed_during_io =
            writer_done_rx.recv_timeout(Duration::from_secs(1)).is_ok();
        store.release_write();
        writer.join().unwrap();
        assert_eq!(
            flush.join().unwrap(),
            0,
            "a stale writeback copy must not count as capacity progress"
        );
        assert!(
            writer_completed_during_io,
            "page-store I/O must not retain the dirty page's exclusive latch"
        );
        assert!(
            unsafe { (*raw).header.core.dirty.load(Ordering::Acquire) },
            "a page modified after its copy was taken must remain dirty"
        );
        assert_eq!(
            unsafe { (*raw).header.core.page_lsn.load(Ordering::Acquire) },
            2,
            "cleaning an older copy must preserve the newer page LSN"
        );

        assert_eq!(pool.try_flush_dirty_batch(1).unwrap(), 1);
        unsafe {
            (*raw)
                .header
                .core
                .referenced
                .store(false, Ordering::Relaxed);
        }
        assert!(pool.try_evict_one());
        let frame = unsafe { pool.fix_orphan_frame(1, unsafe { NoLatches::new(&pool) }) }.shared();
        assert_eq!(frame.page_bytes()[64], 2);
        assert_eq!(page_header::read_page_lsn(frame.page_bytes()), 2);

        let diagnostics = pool.diagnostic_stats();
        assert_eq!(diagnostics.dirty_flush_batches, 2);
        assert_eq!(diagnostics.dirty_flush_pages, 2);
        assert_eq!(diagnostics.dirty_flush_cleaned_pages, 1);
        assert_eq!(diagnostics.dirty_flush_stale_pages, 1);
        assert!(diagnostics.dirty_flush_data_write_ns > 0);
    }

    #[test]
    #[cfg(not(miri))]
    #[should_panic(
        expected = "buffer pool dirty flush failed while waiting for frame capacity: injected dirty writeback failure"
    )]
    fn capacity_wait_reports_dirty_flush_io_error() {
        let fail_writes = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pool = BufferPool::with_store(
            1,
            Box::new(FailingWritePageStore::new(Arc::clone(&fail_writes))),
        );
        let first = pool.allocate_page();
        let second = pool.allocate_page();
        {
            let mut frame = pool
                .fix_stable(&first, unsafe { NoLatches::new(&pool) })
                .exclusive();
            frame.page_mut()[0] = 1;
            frame.mark_dirty();
        }

        fail_writes.store(true, Ordering::Release);
        let _ = pool.fix_stable(&second, unsafe { NoLatches::new(&pool) });
    }

    #[test]
    #[cfg(not(miri))]
    fn stable_index_root_is_not_evicted() {
        let pool = BufferPool::new(1);
        let swip = pool.allocate_page();

        unsafe {
            let bf = pool.fix_stable(&swip, unsafe { NoLatches::new(&pool) });
            let raw = bf.raw();
            let mut bf = bf.exclusive();
            let page = bf.page_bytes_mut();
            let sp = crate::slotted_page::SlottedPage::init(page.try_into().unwrap());
            sp.set_flag(1 << 1);
            page_header::write_page_type(page, PageType::Index);
            (*raw)
                .header
                .core
                .referenced
                .store(false, Ordering::Relaxed);
            drop(bf);

            assert!(
                !pool.try_evict_one(),
                "B-tree root pages must stay resident for eviction parent search"
            );
            assert_eq!(
                pool.try_evict_batch(1),
                0,
                "batch eviction should also skip B-tree roots"
            );
        }
    }

    #[test]
    #[cfg(not(miri))]
    fn try_evict_one_skips_contended_latch_upgrade() {
        let pool = BufferPool::new(1);
        let swip = pool.allocate_page();

        unsafe {
            let bf = pool.fix_stable(&swip, unsafe { NoLatches::new(&pool) });
            let raw = bf.raw();
            drop(bf);
            (*raw)
                .header
                .core
                .referenced
                .store(false, Ordering::Relaxed);

            let _shared = (*raw).latch.lock_shared();
            assert!(
                !pool.try_evict_one(),
                "opportunistic eviction should skip frames with contended latch upgrades"
            );
            assert_eq!(
                (*raw).header.core.state.load(Ordering::Acquire),
                FrameState::Resident,
                "failed opportunistic eviction should leave the frame resident"
            );
        }
    }

    #[test]
    #[cfg(not(miri))]
    fn try_evict_any_batch_replenishes_resident_budget() {
        let pool = BufferPool::new(2);
        let swips: Vec<StableSwip> = (0..2).map(|_| pool.allocate_page()).collect();

        for swip in &swips {
            unsafe {
                let bf = pool.fix_stable(swip, unsafe { NoLatches::new(&pool) });
                let raw = bf.raw();
                drop(bf);
                (*raw)
                    .header
                    .core
                    .referenced
                    .store(false, Ordering::Relaxed);
            }
        }

        assert_eq!(
            pool.approx_available_budget(),
            0,
            "test setup should fully consume resident budget"
        );
        assert!(
            pool.try_evict_any_batch(1) > 0,
            "deterministic batch fallback should evict at least one unreferenced frame"
        );
        assert!(
            pool.approx_available_budget() > 0,
            "batch fallback should return resident budget tokens"
        );
    }

    #[test]
    #[cfg(not(miri))]
    fn eviction_under_pressure() {
        let pool = BufferPool::new(2);
        let swips: Vec<StableSwip> = (0..5).map(|_| pool.allocate_page()).collect();

        for swip in &swips {
            let mut bf = pool
                .fix_stable(swip, unsafe { NoLatches::new(&pool) })
                .exclusive();
            let pid = bf.pid();
            bf.page_mut()[0] = (pid & 0xFF) as u8;
            bf.mark_dirty();
        }

        let hot_count = swips
            .iter()
            .filter(|s| s.load(Ordering::Relaxed).is_hot())
            .count();
        assert!(hot_count <= 2);

        for swip in &swips {
            let bf = pool
                .fix_stable(swip, unsafe { NoLatches::new(&pool) })
                .shared();
            let pid = bf.pid();
            assert_eq!(bf.page()[0], (pid & 0xFF) as u8);
        }
    }

    #[test]
    #[should_panic(expected = "buffer pool exhausted")]
    #[cfg(not(miri))]
    fn panics_when_all_pinned() {
        let pool = BufferPool::new(2);
        let s1 = pool.allocate_page();
        let s2 = pool.allocate_page();
        let s3 = pool.allocate_page();

        let _bf1 = pool.fix_stable(&s1, unsafe { NoLatches::new(&pool) });
        let _bf2 = pool.fix_stable(&s2, unsafe { NoLatches::new(&pool) });
        let _bf3 = pool.fix_stable(&s3, unsafe { NoLatches::new(&pool) });
    }

    #[test]
    #[cfg(not(miri))]
    fn resident_budget_is_separate_from_slot_capacity() {
        let pool = BufferPool::new(4);
        let swips: Vec<StableSwip> = (0..100).map(|_| pool.allocate_page()).collect();

        assert_eq!(pool.num_frames(), 4);

        for swip in &swips {
            let mut bf = pool
                .fix_stable(swip, unsafe { NoLatches::new(&pool) })
                .exclusive();
            let pid = bf.pid();
            bf.page_mut()[0] = (pid & 0xFF) as u8;
            bf.mark_dirty();
        }

        assert!(
            pool.num_occupied() <= pool.num_frames(),
            "resident pages should still respect the configured budget"
        );
    }

    #[test]
    #[cfg(not(miri))]
    fn large_eviction_churn() {
        let pool = BufferPool::new(4);
        let swips: Vec<StableSwip> = (0..100).map(|_| pool.allocate_page()).collect();

        for swip in &swips {
            let mut bf = pool
                .fix_stable(swip, unsafe { NoLatches::new(&pool) })
                .exclusive();
            let pid = bf.pid();
            bf.page_mut()[0] = (pid & 0xFF) as u8;
            bf.page_mut()[1] = ((pid >> 8) & 0xFF) as u8;
            bf.mark_dirty();
        }

        for swip in &swips {
            let bf = pool
                .fix_stable(swip, unsafe { NoLatches::new(&pool) })
                .shared();
            let pid = bf.pid();
            assert_eq!(bf.page()[0], (pid & 0xFF) as u8);
            assert_eq!(bf.page()[1], ((pid >> 8) & 0xFF) as u8);
        }
    }

    // -----------------------------------------------------------------------
    // Concurrent tests
    // -----------------------------------------------------------------------

    #[cfg(not(miri))]
    use std::sync::{Arc, Barrier};
    #[cfg(not(miri))]
    use std::thread;

    #[test]
    #[cfg(not(miri))]
    fn concurrent_fix_same_page() {
        let pool = Arc::new(BufferPool::new(4));
        let swip = pool.allocate_page();
        drop(pool.fix_stable(&swip, unsafe { NoLatches::new(&pool) }));
        let swip = Arc::new(swip);

        let n = 8;
        let barrier = Arc::new(Barrier::new(n));
        let handles: Vec<_> = (0..n)
            .map(|_| {
                let pool = pool.clone();
                let swip = swip.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    let bf = pool.fix_stable(&swip, unsafe { NoLatches::new(&pool) });
                    let _pid = bf.pid();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let s = swip.load(Ordering::Relaxed);
        assert!(s.is_hot());
        let bf = unsafe { s.as_ptr::<BufferFrame>() };
        assert_eq!(
            unsafe { (*bf).header.core.pin_count.load(Ordering::Relaxed) },
            0
        );
    }

    #[test]
    #[cfg(not(miri))]
    fn fix_orphan_waits_through_evicting_without_recursive_retry() {
        let pool = Arc::new(BufferPool::new(4));
        let (pid, bf) = pool.allocate_and_fix(unsafe { NoLatches::new(&pool) });
        let raw = bf.raw();
        drop(bf);

        unsafe {
            let _guard = (*raw).latch.lock_exclusive();
            (*raw).header.core.state.transition(
                FrameState::Resident,
                FrameState::Evicting,
                Ordering::Relaxed,
            );
            (*raw).header.core.pin_count.store(0, Ordering::Relaxed);
        }

        let bf_addr = raw as usize;
        let wake = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            let bf = bf_addr as *mut BufferFrame;
            unsafe {
                let _guard = (*bf).latch.lock_exclusive();
                (*bf).header.core.state.transition(
                    FrameState::Evicting,
                    FrameState::Resident,
                    Ordering::Release,
                );
                (*bf).header.core.referenced.store(false, Ordering::Relaxed);
                (*bf).header.core.pin_count.store(0, Ordering::Relaxed);
            }
        });

        unsafe {
            let fixed = pool.fix_orphan_frame(pid, unsafe { NoLatches::new(&pool) });
            assert_eq!(
                fixed.pid(),
                pid,
                "fix_orphan should eventually return the original resident frame",
            );
        }

        wake.join().unwrap();
    }

    #[test]
    fn try_fix_orphan_rejects_reused_frame_from_stale_page_table_lookup() {
        let pool = BufferPool::new(2);
        let requested_pid = pool.allocate_page().load(Ordering::Acquire).as_page_id();
        let resident_pid = pool.allocate_page().load(Ordering::Acquire).as_page_id();
        let bf = pool.state.free_list.try_pop().unwrap();
        assert!(pool.try_reserve_resident_budget());

        {
            let _guard = unsafe { (*bf).latch.lock_exclusive() };
            unsafe {
                (*bf).header.core.pid = resident_pid;
                (*bf).header.core.state.transition(
                    FrameState::Free,
                    FrameState::Resident,
                    Ordering::Release,
                );
            }
        }
        pool.page_table.insert(requested_pid, bf);

        let fixed = unsafe { pool.try_fix_orphan_frame(requested_pid) };
        let fixed_pid = fixed.as_ref().map(PinnedFrame::pid);
        drop(fixed);

        pool.page_table.remove(requested_pid);
        {
            let _guard = unsafe { (*bf).latch.lock_exclusive() };
            unsafe {
                (*bf)
                    .header
                    .core
                    .state
                    .store(FrameState::Free, Ordering::Release);
            }
        }
        pool.release_resident_budget(bf);
        pool.state.free_list.push(bf);

        assert_eq!(
            fixed_pid, None,
            "a stale page-table pointer must not return a reused frame for the wrong page"
        );
    }

    #[test]
    #[cfg(not(miri))]
    fn concurrent_orphan_load_races_preserve_resident_budget() {
        const THREADS: usize = 8;
        const PAGES: usize = 64;

        let pool = Arc::new(BufferPool::new(8));
        let page_ids = Arc::new(
            (0..PAGES)
                .map(|_| pool.allocate_page().load(Ordering::Acquire).as_page_id())
                .collect::<Vec<_>>(),
        );
        let barrier = Arc::new(std::sync::Barrier::new(THREADS));
        let handles = (0..THREADS)
            .map(|_| {
                let pool = Arc::clone(&pool);
                let page_ids = Arc::clone(&page_ids);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    for &page_id in page_ids.iter() {
                        barrier.wait();
                        let frame = unsafe {
                            pool.fix_orphan_frame(page_id, unsafe { NoLatches::new(&pool) })
                        };
                        assert_eq!(frame.pid(), page_id);
                        drop(frame);
                        barrier.wait();
                    }
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle.join().unwrap();
        }

        let occupied = pool.num_occupied();
        let available = pool.approx_available_budget();
        assert_eq!(
            occupied + available,
            pool.num_frames(),
            "orphan-load races must neither leak nor invent resident-budget tokens"
        );
    }

    #[test]
    #[cfg(not(miri))]
    fn concurrent_try_orphan_load_races_preserve_free_frames() {
        const THREADS: usize = 8;
        const PAGES: usize = 4;

        let pool = Arc::new(BufferPool::new(8));
        let page_ids = Arc::new(
            (0..PAGES)
                .map(|_| pool.allocate_page().load(Ordering::Acquire).as_page_id())
                .collect::<Vec<_>>(),
        );
        let barrier = Arc::new(Barrier::new(THREADS));
        let handles = (0..THREADS)
            .map(|_| {
                let pool = pool.clone();
                let page_ids = page_ids.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    for &page_id in page_ids.iter() {
                        barrier.wait();
                        loop {
                            if let Some(frame) = unsafe { pool.try_fix_orphan_frame(page_id) } {
                                assert_eq!(frame.pid(), page_id);
                                break;
                            }
                            thread::yield_now();
                        }
                        barrier.wait();
                    }
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(
            pool.state.free_list.len(),
            pool.approx_available_budget(),
            "losers of a non-blocking orphan-load race must return their unclaimed frames"
        );
    }

    #[test]
    fn concurrent_free_list_churn_preserves_frame_ownership() {
        const FRAMES: usize = 64;
        const THREADS: usize = 8;
        const ITERATIONS: usize = 50_000;

        let pool = Arc::new(BufferPool::new(FRAMES));
        let barrier = Arc::new(Barrier::new(THREADS));
        let handles = (0..THREADS)
            .map(|_| {
                let pool = pool.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..ITERATIONS {
                        let frame = loop {
                            if let Some(frame) = pool.state.free_list.try_pop() {
                                break frame;
                            }
                            thread::yield_now();
                        };
                        thread::yield_now();
                        pool.state.free_list.push(frame);
                    }
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle.join().unwrap();
        }

        let mut frames = std::collections::HashSet::new();
        while let Some(frame) = pool.state.free_list.try_pop() {
            assert!(
                frames.insert(frame as usize),
                "free list returned the same frame more than once"
            );
            assert!(
                frames.len() <= FRAMES,
                "free list contained more frames than the pool owns"
            );
        }
        assert_eq!(frames.len(), FRAMES, "free-list churn lost frames");
    }

    #[test]
    #[cfg(not(miri))]
    fn concurrent_fix_different_pages() {
        let pool = Arc::new(BufferPool::new(16));
        let n = 8;
        let swips: Vec<Arc<StableSwip>> = (0..n).map(|_| Arc::new(pool.allocate_page())).collect();
        let barrier = Arc::new(Barrier::new(n));

        let handles: Vec<_> = (0..n)
            .map(|i| {
                let pool = pool.clone();
                let swip = swips[i].clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    let mut bf = pool
                        .fix_stable(&swip, unsafe { NoLatches::new(&pool) })
                        .exclusive();
                    let pid = bf.pid();
                    bf.page_mut()[0] = (pid & 0xFF) as u8;
                    bf.mark_dirty();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        for swip in &swips {
            let bf = pool
                .fix_stable(swip, unsafe { NoLatches::new(&pool) })
                .shared();
            let pid = bf.pid();
            assert_eq!(bf.page()[0], (pid & 0xFF) as u8);
        }
    }

    #[test]
    #[cfg(not(miri))]
    fn concurrent_fix_unfix_churn() {
        let num_frames = 8;
        let num_pages = 32;
        let pool = Arc::new(BufferPool::new(num_frames));
        let swips: Arc<Vec<StableSwip>> =
            Arc::new((0..num_pages).map(|_| pool.allocate_page()).collect());

        let n = 4;
        let iterations = 200;
        let barrier = Arc::new(Barrier::new(n));

        let handles: Vec<_> = (0..n)
            .map(|t| {
                let pool = pool.clone();
                let swips = swips.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    for i in 0..iterations {
                        let idx = (t * 7 + i * 13) % num_pages;
                        let mut bf = pool
                            .fix_stable(&swips[idx], unsafe { NoLatches::new(&pool) })
                            .exclusive();
                        let pid = bf.pid();
                        bf.page_mut()[0] = (pid & 0xFF) as u8;
                        bf.mark_dirty();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        for swip in swips.iter() {
            let bf = pool
                .fix_stable(swip, unsafe { NoLatches::new(&pool) })
                .shared();
            let pid = bf.pid();
            assert_eq!(bf.page()[0], (pid & 0xFF) as u8);
        }
    }

    #[test]
    #[cfg(not(miri))]
    fn concurrent_allocate_page() {
        let pool = Arc::new(BufferPool::new(4));
        let n = 8;
        let per_thread = 50;
        let barrier = Arc::new(Barrier::new(n));

        let handles: Vec<_> = (0..n)
            .map(|_| {
                let pool = pool.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    let mut pids = Vec::new();
                    for _ in 0..per_thread {
                        let swip = pool.allocate_page();
                        let s = swip.load(Ordering::Relaxed);
                        pids.push(s.as_page_id());
                    }
                    pids
                })
            })
            .collect();

        let mut all_pids: Vec<u64> = Vec::new();
        for h in handles {
            all_pids.extend(h.join().unwrap());
        }

        all_pids.sort();
        all_pids.dedup();
        assert_eq!(all_pids.len(), n * per_thread);
    }

    #[test]
    #[cfg(not(miri))]
    fn eviction_under_heavy_pressure() {
        let n = 4;
        let num_frames = n + 4;
        let num_pages = 200;
        let pool = Arc::new(BufferPool::new(num_frames));
        let swips: Arc<Vec<StableSwip>> =
            Arc::new((0..num_pages).map(|_| pool.allocate_page()).collect());

        let iterations = 100;
        let barrier = Arc::new(Barrier::new(n));

        let handles: Vec<_> = (0..n)
            .map(|t| {
                let pool = pool.clone();
                let swips = swips.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    for i in 0..iterations {
                        let idx = (t * 31 + i * 37) % num_pages;
                        let mut bf = pool
                            .fix_stable(&swips[idx], unsafe { NoLatches::new(&pool) })
                            .exclusive();
                        let pid = bf.pid();
                        bf.page_mut()[0] = (pid & 0xFF) as u8;
                        bf.page_mut()[1] = ((pid >> 8) & 0xFF) as u8;
                        bf.mark_dirty();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        for swip in swips.iter() {
            let bf = pool
                .fix_stable(swip, unsafe { NoLatches::new(&pool) })
                .shared();
            let pid = bf.pid();
            assert_eq!(bf.page()[0], (pid & 0xFF) as u8);
            assert_eq!(bf.page()[1], ((pid >> 8) & 0xFF) as u8);
        }
    }

    #[test]
    fn mark_dirty_with_lsn_overwrites_without_monotonicity_guard() {
        // Document the current contract: mark_dirty_with_lsn does NOT guard
        // against LSN regression. Callers (via claim_lsn) always pass
        // monotonic LSNs. This test pins the current behaviour so a future
        // monotonicity guard would be a visible change.
        let pool = BufferPool::new(4);
        let swip = pool.allocate_page();
        let bf = pool.fix_stable(&swip, unsafe { NoLatches::new(&pool) });
        let exclusive = bf.exclusive();

        exclusive.mark_dirty_with_lsn(100);
        assert_eq!(
            unsafe {
                (*exclusive.raw())
                    .header
                    .core
                    .page_lsn
                    .load(Ordering::Relaxed)
            },
            100
        );

        exclusive.mark_dirty_with_lsn(50);
        // Current behaviour: overwrites without guard.
        assert_eq!(
            unsafe {
                (*exclusive.raw())
                    .header
                    .core
                    .page_lsn
                    .load(Ordering::Relaxed)
            },
            50,
            "mark_dirty_with_lsn currently overwrites without monotonicity guard; \
             callers must not regress LSNs"
        );
    }

    #[test]
    fn referenced_bit_blocks_first_eviction_attempt() {
        let pool = BufferPool::new(1);
        let swip = pool.allocate_page();

        // Fix and unfix — the referenced bit should be set by fix.
        let raw = {
            let bf = pool.fix_stable(&swip, unsafe { NoLatches::new(&pool) });
            assert!(
                unsafe { (*bf.raw()).header.core.referenced.load(Ordering::Relaxed) },
                "fix should set referenced bit"
            );
            let raw = bf.raw();
            drop(bf);
            raw
        };

        // The frame is now unreferenced but hot with referenced=true.
        // try_evict_one should either skip it (clearing referenced) or
        // evict it. The key invariant: after one attempt, either the
        // frame is evicted or the referenced bit is cleared.
        let evicted = pool.try_evict_one();

        if !evicted {
            // If not evicted, referenced should have been cleared
            // (second-chance clock semantics).
            assert!(
                !unsafe { (*raw).header.core.referenced.load(Ordering::Relaxed) },
                "referenced bit should be cleared after first failed eviction attempt"
            );

            // Second attempt should now succeed.
            assert!(
                pool.try_evict_one(),
                "eviction should succeed after referenced bit is cleared"
            );
        }
        // If evicted on first attempt, that's also valid (referenced may
        // have been cleared between fix and evict).
    }
}
