//! Slotted-page container: a sorted key/value byte layout that fits in one
//! unified compile-time Pagebox page.
//!
//! [`SlottedPage`] is the in-memory page format used by the B+tree. It is
//! `#[repr(transparent)]` over a
//! `[u8; PAGE_SIZE]`: zero allocation, zero indirection — a `SlottedPage`
//! *is* the frame's page bytes reinterpreted. All access is through
//! `unsafe` casting helpers ([`SlottedPage::from_page`] /
//! [`SlottedPage::from_page_mut`]) that the caller asserts are valid (the
//! page must have been `init`-ed).
//!
//! ## Layout
//!
//! ```text
//!  ┌──────────┬───────────────┬──────────────┬────────────────────┐
//!  │PageHeader│ Slot[0] ...   │   <gap>      │ value key  value key│
//!  │ 24 bytes │ slot array    │ (free space) │     ↑ data heap  ↑  │
//!  └──────────┴───────────────┴──────────────┴────────────────────┘
//!   ← slot array grows forward                data heap grows backward ←
//! ```
//!
//! The 24-byte `PageHeader` holds `num_slots`, `data_offset` (start of the
//! live data heap, growing *down* from `PAGE_SIZE`), `space_used`, a packed
//! `flags` word (low byte = page-specific flags; nibble 1 = page-type
//! discriminator; see [`PageType`]), and the page LSN. Each 12-byte `Slot`
//! caches `offset`, `key_len`, `val_len`, and the *first four bytes of the
//! key* as a big-endian, zero-padded `head` for fast binary-search
//! short-circuiting.
//!
//! ## Compaction
//!
//! Removals (and overwrites that change a value's length) leave dead bytes
//! in the data heap and set the `has_garbage` flag.
//! [`SlottedPage::compactify`] rewrites the live data contiguously from the
//! end of the page, eliminating the gap and clearing the flag. **Reserved
//! suffixes** — bytes reserved at the page tail via
//! [`SlottedPage::reserve_suffix`] right after `init` — survive compaction:
//! the B-tree uses this to keep right-sibling / upper-link metadata at the
//! end of the page even when the slot heap moves underneath it.
//!
//! ## Optimistic-read variants
//!
//! Methods named `try_*` are bounds-checked variants intended for the
//! B+tree's optimistic read path: an optimistic reader may observe a torn
//! page (a writer is mid-mutation), so any slot metadata pointing outside
//! the page or to overlapping regions is rejected as `None` rather than
//! panicking. The non-`try_` variants `debug_assert` those invariants.
//!
//! ## Overflow contract
//!
//! `insert` panics with `"slotted page full"` when the page cannot fit a
//! new entry even after compaction (see
//! [`SlottedPage::can_insert`] for a pre-flight check). Overflow is a
//! page-split signal in the B+tree; there is no chained-overflow page.

use std::cmp::Ordering;
use std::sync::OnceLock;

#[cfg(feature = "metrics")]
use fast_telemetry::{Counter, ExportMetrics, MetricVisitor};

#[cfg(not(feature = "metrics"))]
use crate::metrics_stub::{Counter, MetricVisitor};

use crate::buffer_frame::PAGE_SIZE;
pub use crate::page_header::PageType;

const HEADER_SIZE: usize = std::mem::size_of::<PageHeader>();
const SLOT_SIZE: usize = std::mem::size_of::<Slot>();

// Compile-time layout checks.
const _: () = assert!(HEADER_SIZE == 24);
const _: () = assert!(SLOT_SIZE == 12);

#[repr(C)]
#[derive(Clone, Copy)]
struct PageHeader {
    page_lsn: u64,
    data_offset: u32,
    space_used: u32,
    num_slots: u16,
    flags: u16,
}

const FLAG_HAS_GARBAGE: u16 = 1;

/// Per-entry metadata stored in the slot array.
#[repr(C)]
#[derive(Clone, Copy)]
struct Slot {
    /// Byte offset into page where key|value data starts.
    offset: u16,
    /// Key length in bytes.
    key_len: u16,
    /// Value length in bytes.
    val_len: u16,
    _pad: u16,
    /// First 4 bytes of key, big-endian padded, for fast comparison.
    head: u32,
}

/// Read the page LSN from raw page bytes without constructing a SlottedPage.
/// Works on any page type that follows the common header layout.
pub fn read_page_lsn(page: &[u8; PAGE_SIZE]) -> u64 {
    crate::page_header::read_page_lsn(page)
}

/// Write the page LSN into raw page bytes without constructing a SlottedPage.
/// Works on any page type that follows the common header layout.
pub fn write_page_lsn(page: &mut [u8; PAGE_SIZE], lsn: u64) {
    crate::page_header::write_page_lsn(page, lsn);
}

#[cfg_attr(feature = "metrics", derive(ExportMetrics))]
#[cfg_attr(feature = "metrics", metric_prefix = "slotted_page")]
struct SlottedPageStats {
    #[cfg_attr(feature = "metrics", help = "Slotted page compactions")]
    compactions: Counter,
    #[cfg_attr(
        feature = "metrics",
        help = "Bytes moved while compactifying slotted pages"
    )]
    compactify_bytes_moved: Counter,
    #[cfg_attr(
        feature = "metrics",
        help = "Bytes moved while shifting slotted-page slots"
    )]
    slot_shift_bytes: Counter,
}

fn slotted_page_stats() -> &'static SlottedPageStats {
    static STATS: OnceLock<SlottedPageStats> = OnceLock::new();
    STATS.get_or_init(|| {
        let shards = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        SlottedPageStats {
            compactions: Counter::new(shards),
            compactify_bytes_moved: Counter::new(shards),
            slot_shift_bytes: Counter::new(shards),
        }
    })
}

#[cfg(not(feature = "metrics"))]
impl SlottedPageStats {
    fn visit_metrics<V: MetricVisitor + ?Sized>(&self, _visitor: &mut V) {}
}

pub fn visit_stats<V: MetricVisitor + ?Sized>(visitor: &mut V) {
    slotted_page_stats().visit_metrics(visitor);
}

/// Read the page type from raw page bytes via the flags field.
pub fn read_page_type(page: &[u8; PAGE_SIZE]) -> PageType {
    crate::page_header::read_page_type(page)
}

/// Returns true if the page is a B-tree inner (non-leaf) index page.
/// Used by eviction to skip inner nodes — they are kept permanently
/// resident so traversal never encounters an evicted inner node.
pub fn is_inner_index_page(page: &[u8; PAGE_SIZE]) -> bool {
    crate::page_header::is_inner_index_page(page)
}

/// Returns true if the page should remain resident in the buffer pool.
///
/// Table/DB state pages are touched frequently by metadata updates and are
/// small in count relative to data pages, so evicting them tends to create
/// pathological churn under small pool sizes.
pub fn is_resident_meta_page(page: &[u8; PAGE_SIZE]) -> bool {
    crate::page_header::should_remain_resident(page)
}

/// Write the page type into raw page bytes via the flags field.
/// Preserves the low-byte flags (has_garbage, is_leaf, etc.).
pub fn write_page_type(page: &mut [u8; PAGE_SIZE], pt: PageType) {
    crate::page_header::write_page_type(page, pt);
}

/// A sorted key/value container backed by a `[u8; PAGE_SIZE]`.
///
/// `#[repr(transparent)]` over the page bytes — construction via
/// [`from_page`](Self::from_page) / [`from_page_mut`](Self::from_page_mut)
/// is a zero-cost cast the caller asserts is valid (the page must be
/// `init`-ed). See the [module-level docs](self) for the byte layout,
/// compaction semantics, the optimistic-read `try_*` variants, and the
/// overflow contract.
#[repr(transparent)]
pub struct SlottedPage {
    data: [u8; PAGE_SIZE],
}

// ---------------------------------------------------------------------------
// Construction / casting
// ---------------------------------------------------------------------------

impl SlottedPage {
    /// Initialize a page as an empty slotted page.
    pub fn init(page: &mut [u8; PAGE_SIZE]) -> &mut SlottedPage {
        page.fill(0);
        let sp = Self::from_page_mut(page);
        sp.header_mut().data_offset = PAGE_SIZE as u32;
        sp
    }

    /// Reinterpret a mutable page byte array as a SlottedPage.
    ///
    /// # Safety
    /// The page must have been initialized via `init()`.
    pub fn from_page_mut(page: &mut [u8; PAGE_SIZE]) -> &mut SlottedPage {
        // SAFETY: SlottedPage is #[repr(transparent)] over [u8; PAGE_SIZE].
        unsafe { &mut *(page as *mut [u8; PAGE_SIZE] as *mut SlottedPage) }
    }

    /// Reinterpret an immutable page byte array as a SlottedPage.
    ///
    /// # Safety
    /// The page must have been initialized via `init()`.
    pub fn from_page(page: &[u8; PAGE_SIZE]) -> &SlottedPage {
        unsafe { &*(page as *const [u8; PAGE_SIZE] as *const SlottedPage) }
    }

    /// Raw immutable byte access.
    pub fn as_bytes(&self) -> &[u8; PAGE_SIZE] {
        // SAFETY: SlottedPage is #[repr(transparent)] over [u8; PAGE_SIZE].
        unsafe { &*(self as *const SlottedPage as *const [u8; PAGE_SIZE]) }
    }

    /// Raw mutable byte access.
    pub fn as_bytes_mut(&mut self) -> &mut [u8; PAGE_SIZE] {
        unsafe { &mut *(self as *mut SlottedPage as *mut [u8; PAGE_SIZE]) }
    }
}

// ---------------------------------------------------------------------------
// Internal accessors
// ---------------------------------------------------------------------------

impl SlottedPage {
    fn header(&self) -> &PageHeader {
        unsafe { &*(self.data.as_ptr() as *const PageHeader) }
    }

    fn header_mut(&mut self) -> &mut PageHeader {
        unsafe { &mut *(self.data.as_mut_ptr() as *mut PageHeader) }
    }

    fn slot(&self, id: u16) -> &Slot {
        debug_assert!((id as usize) < self.header().num_slots as usize);
        let offset = HEADER_SIZE + (id as usize) * SLOT_SIZE;
        unsafe { &*(self.data.as_ptr().add(offset) as *const Slot) }
    }

    fn slot_mut(&mut self, id: u16) -> &mut Slot {
        debug_assert!((id as usize) < self.header().num_slots as usize);
        let offset = HEADER_SIZE + (id as usize) * SLOT_SIZE;
        unsafe { &mut *(self.data.as_mut_ptr().add(offset) as *mut Slot) }
    }

    fn slot_array_end(&self) -> usize {
        HEADER_SIZE + (self.header().num_slots as usize) * SLOT_SIZE
    }

    fn try_slot_copy(&self, id: u16) -> Option<Slot> {
        let num_slots = self.header().num_slots;
        if id >= num_slots {
            return None;
        }
        let offset = HEADER_SIZE + (id as usize) * SLOT_SIZE;
        let end = offset.checked_add(SLOT_SIZE)?;
        if end > PAGE_SIZE {
            return None;
        }
        Some(unsafe { *(self.data.as_ptr().add(offset) as *const Slot) })
    }
}

// ---------------------------------------------------------------------------
// Key head computation and comparison
// ---------------------------------------------------------------------------

impl SlottedPage {
    /// Extract a 4-byte head from a key for fast comparison.
    /// Big-endian, zero-padded for keys shorter than 4 bytes.
    fn make_head(key: &[u8]) -> u32 {
        let mut buf = [0u8; 4];
        let n = key.len().min(4);
        buf[..n].copy_from_slice(&key[..n]);
        u32::from_be_bytes(buf)
    }

    fn cmp_keys(a: &[u8], b: &[u8]) -> Ordering {
        a.cmp(b)
    }

    fn read_be_u64(bytes: &[u8]) -> u64 {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(bytes);
        u64::from_be_bytes(buf)
    }

    fn slot_key_u64(&self, slot_id: u16) -> Option<u64> {
        let s = self.slot(slot_id);
        if s.key_len != 8 {
            return None;
        }
        let start = s.offset as usize;
        let end = start.checked_add(8)?;
        if end > PAGE_SIZE {
            return None;
        }
        Some(Self::read_be_u64(&self.data[start..end]))
    }

    fn try_slot_key_u64(&self, slot_id: u16) -> Option<u64> {
        let s = self.try_slot_copy(slot_id)?;
        if s.key_len != 8 {
            return None;
        }
        let start = s.offset as usize;
        let end = start.checked_add(8)?;
        if end > PAGE_SIZE {
            return None;
        }
        Some(Self::read_be_u64(&self.data[start..end]))
    }

    fn lower_bound_u64_be(&self, search_key: u64) -> (u16, bool) {
        let count = self.header().num_slots as usize;
        if count == 0 {
            return (0, false);
        }

        let mut lo: usize = 0;
        let mut hi: usize = count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let Some(slot_key) = self.slot_key_u64(mid as u16) else {
                return self.lower_bound_generic(&search_key.to_be_bytes());
            };
            match slot_key.cmp(&search_key) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return (mid as u16, true),
            }
        }

        (lo as u16, false)
    }

    fn try_lower_bound_u64_be(&self, search_key: u64) -> Option<(u16, bool)> {
        let count = self.header().num_slots as usize;
        if count == 0 {
            return Some((0, false));
        }
        if count > PAGE_SIZE / 12 {
            return None;
        }

        let mut lo: usize = 0;
        let mut hi: usize = count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let Some(slot_key) = self.try_slot_key_u64(mid as u16) else {
                return self.try_lower_bound_generic(&search_key.to_be_bytes());
            };
            match slot_key.cmp(&search_key) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Some((mid as u16, true)),
            }
        }

        Some((lo as u16, false))
    }

    fn lower_bound_generic(&self, key: &[u8]) -> (u16, bool) {
        let count = self.header().num_slots as usize;
        if count == 0 {
            return (0, false);
        }

        let search_head = Self::make_head(key);
        let mut lo: usize = 0;
        let mut hi: usize = count;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let s = self.slot(mid as u16);

            match s.head.cmp(&search_head) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => {
                    let slot_key = self.get_key(mid as u16);
                    match Self::cmp_keys(slot_key, key) {
                        Ordering::Less => lo = mid + 1,
                        Ordering::Greater => hi = mid,
                        Ordering::Equal => return (mid as u16, true),
                    }
                }
            }
        }

        (lo as u16, false)
    }

    fn try_lower_bound_generic(&self, key: &[u8]) -> Option<(u16, bool)> {
        let count = self.header().num_slots as usize;
        if count == 0 {
            return Some((0, false));
        }
        // Sanity: max possible slots in a page.
        if count > PAGE_SIZE / 12 {
            return None;
        }

        let search_head = Self::make_head(key);
        let mut lo: usize = 0;
        let mut hi: usize = count;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let s = self.try_slot_copy(mid as u16)?;

            match s.head.cmp(&search_head) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => {
                    let slot_key = self.try_get_key(mid as u16)?;
                    match Self::cmp_keys(slot_key, key) {
                        Ordering::Less => lo = mid + 1,
                        Ordering::Greater => hi = mid,
                        Ordering::Equal => return Some((mid as u16, true)),
                    }
                }
            }
        }

        Some((lo as u16, false))
    }
}

// ---------------------------------------------------------------------------
// Public read accessors
// ---------------------------------------------------------------------------

impl SlottedPage {
    pub fn num_slots(&self) -> u16 {
        self.header().num_slots
    }

    /// Free space in the gap between slot array and data region.
    pub fn free_space(&self) -> usize {
        let slot_end = self.slot_array_end();
        let data_start = self.header().data_offset as usize;
        data_start.saturating_sub(slot_end)
    }

    /// Free space available if we compacted (eliminates garbage).
    pub fn free_space_after_compaction(&self) -> usize {
        let slot_end = self.slot_array_end();
        let live_data = self.header().space_used as usize;
        PAGE_SIZE.saturating_sub(slot_end + live_data)
    }

    /// Total bytes used by live key+value data.
    pub fn space_used(&self) -> usize {
        self.header().space_used as usize
    }

    /// The page's on-disk LSN (most recent WAL record applied).
    /// Part of the common page header contract — usable by recovery
    /// and checkpoint logic regardless of page type.
    pub fn page_lsn(&self) -> u64 {
        self.header().page_lsn
    }

    /// Set the page's on-disk LSN.
    pub fn set_page_lsn(&mut self, lsn: u64) {
        self.header_mut().page_lsn = lsn;
    }

    pub fn has_garbage(&self) -> bool {
        self.header().flags & FLAG_HAS_GARBAGE != 0
    }

    /// Check if a custom flag bit is set.
    pub fn has_custom_flag(&self, flag: u16) -> bool {
        self.header().flags & flag != 0
    }

    /// Set a custom flag bit.
    pub fn set_flag(&mut self, flag: u16) {
        self.header_mut().flags |= flag;
    }

    /// Reserve `n` bytes at the end of the data region by lowering the
    /// initial data_offset. Must be called immediately after `init()`.
    pub fn reserve_suffix(&mut self, n: usize) {
        debug_assert_eq!(
            self.header().num_slots,
            0,
            "reserve_suffix on non-empty page"
        );
        let hdr = self.header_mut();
        hdr.data_offset -= n as u32;
        hdr.space_used += n as u32;
    }

    /// Get the key bytes for the given slot.
    pub fn get_key(&self, slot_id: u16) -> &[u8] {
        let s = self.slot(slot_id);
        let start = s.offset as usize;
        &self.data[start..start + s.key_len as usize]
    }

    /// Get the value bytes for the given slot.
    pub fn get_value(&self, slot_id: u16) -> &[u8] {
        let s = self.slot(slot_id);
        let start = s.offset as usize + s.key_len as usize;
        &self.data[start..start + s.val_len as usize]
    }

    /// Bounds-checked value access for use under optimistic reads.
    /// Returns `None` if the slot metadata points outside the page,
    /// which can happen when reading torn data concurrently.
    pub fn try_get_value(&self, slot_id: u16) -> Option<&[u8]> {
        let s = self.try_slot_copy(slot_id)?;
        let start = (s.offset as usize).checked_add(s.key_len as usize)?;
        let end = start.checked_add(s.val_len as usize)?;
        if end > PAGE_SIZE {
            return None;
        }
        Some(&self.data[start..end])
    }

    /// Bounds-checked key access for use under optimistic reads.
    pub fn try_get_key(&self, slot_id: u16) -> Option<&[u8]> {
        let s = self.try_slot_copy(slot_id)?;
        let start = s.offset as usize;
        let end = start.checked_add(s.key_len as usize)?;
        if end > PAGE_SIZE {
            return None;
        }
        Some(&self.data[start..end])
    }

    /// Bounds-checked lower_bound for use under optimistic reads.
    /// Returns `None` if any slot data looks corrupt (torn read).
    /// On success returns `Some((pos, exact_match))`.
    pub fn try_lower_bound(&self, key: &[u8]) -> Option<(u16, bool)> {
        if key.len() == 8 {
            let search_key = Self::read_be_u64(key);
            return self.try_lower_bound_u64_be(search_key);
        }
        self.try_lower_bound_generic(key)
    }
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

impl SlottedPage {
    /// Check if there is room for a key+value pair (including slot overhead).
    pub fn can_insert(&self, key_len: usize, val_len: usize) -> bool {
        let needed = SLOT_SIZE + key_len + val_len;
        self.free_space() >= needed || self.free_space_after_compaction() >= needed
    }

    /// Insert key+value at the given slot position.
    /// Shifts slots `[slot_id..num_slots)` right by one.
    /// Panics if insufficient space.
    pub fn insert(&mut self, slot_id: u16, key: &[u8], value: &[u8]) {
        let data_len = key.len() + value.len();
        let needed = SLOT_SIZE + data_len;

        // Ensure space, compacting if necessary.
        if self.free_space() < needed {
            assert!(
                self.free_space_after_compaction() >= needed,
                "slotted page full"
            );
            self.compactify();
        }

        // Allocate space in data region (grows downward).
        let new_data_offset = self.header().data_offset as usize - data_len;
        self.data.copy_within(
            // Source: key bytes, then value bytes.
            // We'll write them directly instead.
            0..0, // dummy — we write below
            0,
        );

        // Write key then value into data region.
        self.data[new_data_offset..new_data_offset + key.len()].copy_from_slice(key);
        self.data[new_data_offset + key.len()..new_data_offset + data_len].copy_from_slice(value);

        let count = self.header().num_slots;
        if slot_id < count {
            let base = self.data.as_mut_ptr();
            let src_offset = HEADER_SIZE + (slot_id as usize) * SLOT_SIZE;
            let dst_offset = HEADER_SIZE + (slot_id as usize + 1) * SLOT_SIZE;
            let n = (count - slot_id) as usize * SLOT_SIZE;
            if n > 0 {
                slotted_page_stats().slot_shift_bytes.add(n as isize);
            }
            unsafe { std::ptr::copy(base.add(src_offset), base.add(dst_offset), n) };
        }

        // Write the new slot.
        let new_count = count + 1;
        self.header_mut().num_slots = new_count;
        let s = self.slot_mut(slot_id);
        *s = Slot {
            offset: new_data_offset as u16,
            key_len: key.len() as u16,
            val_len: value.len() as u16,
            _pad: 0,
            head: Self::make_head(key),
        };

        // Update header.
        self.header_mut().data_offset = new_data_offset as u32;
        self.header_mut().space_used += data_len as u32;
    }

    /// Remove entry at `slot_id`. Shifts remaining slots left.
    pub fn remove(&mut self, slot_id: u16) {
        let count = self.header().num_slots;
        assert!((slot_id as usize) < count as usize);

        let data_len = self.slot(slot_id).key_len as usize + self.slot(slot_id).val_len as usize;
        self.header_mut().space_used -= data_len as u32;

        // Shift slots left. Derive both src and dst from one &mut self borrow.
        let remaining = count - slot_id - 1;
        if remaining > 0 {
            let base = self.data.as_mut_ptr();
            let dst_offset = HEADER_SIZE + (slot_id as usize) * SLOT_SIZE;
            let src_offset = HEADER_SIZE + (slot_id as usize + 1) * SLOT_SIZE;
            let shifted = remaining as usize * SLOT_SIZE;
            if shifted > 0 {
                slotted_page_stats().slot_shift_bytes.add(shifted as isize);
            }
            unsafe {
                std::ptr::copy(base.add(src_offset), base.add(dst_offset), shifted);
            }
        }

        self.header_mut().num_slots = count - 1;
        self.header_mut().flags |= FLAG_HAS_GARBAGE;
    }

    /// Update value in-place if new value is the same length as existing.
    /// Returns `false` if lengths differ.
    pub fn update_value_if_same_length(&mut self, slot_id: u16, value: &[u8]) -> bool {
        let s = self.slot(slot_id);
        if value.len() != s.val_len as usize {
            return false;
        }
        let start = s.offset as usize + s.key_len as usize;
        self.data[start..start + value.len()].copy_from_slice(value);
        true
    }

    /// Byte range occupied by a slot's value in the underlying page.
    pub fn value_range(&self, slot_id: u16) -> std::ops::Range<usize> {
        let s = self.slot(slot_id);
        let start = s.offset as usize + s.key_len as usize;
        start..start + s.val_len as usize
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

impl SlottedPage {
    /// Binary search for `key`. Returns `(slot_id, is_exact)`.
    /// `slot_id` is the first slot where `slot_key >= key`.
    /// Uses head comparison to skip full key comparisons when possible.
    pub fn lower_bound(&self, key: &[u8]) -> (u16, bool) {
        if key.len() == 8 {
            let search_key = Self::read_be_u64(key);
            return self.lower_bound_u64_be(search_key);
        }
        self.lower_bound_generic(key)
    }
}

// ---------------------------------------------------------------------------
// Maintenance
// ---------------------------------------------------------------------------

impl SlottedPage {
    /// Defragment the data region: rewrite all live data contiguously
    /// from the end of the page, eliminating gaps from removed entries.
    pub fn compactify(&mut self) {
        slotted_page_stats().compactions.inc();
        let count = self.header().num_slots as usize;
        if count == 0 {
            // Preserve any reserved suffix space when the page is empty.
            // For example, B-tree leaves/inner nodes reserve bytes at the end
            // of the page for right-sibling / upper-link metadata. After the
            // last logical entry is removed, `space_used` still reflects only
            // that reserved suffix, and compactification must not drop it.
            let reserved_suffix = self.header().space_used as usize;
            self.header_mut().data_offset = (PAGE_SIZE - reserved_suffix) as u32;
            self.header_mut().flags &= !FLAG_HAS_GARBAGE;
            return;
        }

        // Collect slot metadata (we'll rewrite data in-place from the end).
        // Use a temporary buffer for the data region only.
        let mut tmp = [0u8; PAGE_SIZE];
        let mut live_data = 0usize;
        for i in 0..count {
            let s = self.slot(i as u16);
            let data_len = s.key_len as usize + s.val_len as usize;
            let end = s.offset as usize + data_len;
            assert!(
                end <= PAGE_SIZE,
                "compactify saw corrupt slot: page_type={:?} slot={} offset={} key_len={} val_len={} end={}",
                read_page_type(&self.data),
                i,
                s.offset,
                s.key_len,
                s.val_len,
                end
            );
            live_data += data_len;
        }
        let reserved_suffix = self.header().space_used as usize - live_data;
        slotted_page_stats()
            .compactify_bytes_moved
            .add((live_data.saturating_mul(2)) as isize);
        let mut write_offset = PAGE_SIZE - reserved_suffix;

        for i in 0..count {
            let s = self.slot(i as u16);
            let data_len = s.key_len as usize + s.val_len as usize;
            let src_start = s.offset as usize;
            write_offset -= data_len;
            tmp[write_offset..write_offset + data_len]
                .copy_from_slice(&self.data[src_start..src_start + data_len]);
        }

        // Collect data lengths from slots before mutating.
        // (Avoids interleaving shared and mutable borrows of self.)
        let mut data_lens = [0u16; (PAGE_SIZE - HEADER_SIZE) / SLOT_SIZE];
        for (i, data_len) in data_lens.iter_mut().enumerate().take(count) {
            let s = self.slot(i as u16);
            *data_len = s.key_len + s.val_len;
        }

        // Write back data and update slot offsets.
        let mut offset = PAGE_SIZE - reserved_suffix;
        for (i, data_len) in data_lens.iter().enumerate().take(count) {
            let data_len = *data_len as usize;
            offset -= data_len;
            self.data[offset..offset + data_len].copy_from_slice(&tmp[offset..offset + data_len]);
            self.slot_mut(i as u16).offset = offset as u16;
        }

        self.header_mut().data_offset = offset as u32;
        self.header_mut().flags &= !FLAG_HAS_GARBAGE;
    }
}

// ---------------------------------------------------------------------------
// Bulk operations
// ---------------------------------------------------------------------------

impl SlottedPage {
    /// Copy slots `[src_slot..src_slot+count)` from `self` into `dst`
    /// starting at `dst_slot`. Used for split operations.
    ///
    /// `dst` must have enough free space and `dst_slot` must equal
    /// `dst.num_slots()` (append-only for simplicity).
    pub fn copy_key_value_range(
        &self,
        dst: &mut SlottedPage,
        dst_slot: u16,
        src_slot: u16,
        count: u16,
    ) {
        debug_assert!(dst_slot == dst.num_slots());
        for i in 0..count {
            let slot = src_slot + i;
            let key = self.get_key(slot);
            let value = self.get_value(slot);
            let pos = dst.num_slots();
            dst.insert(pos, key, value);
        }
    }

    /// Truncate to only keep slots `[0..new_count)`. Compacts afterward.
    pub fn truncate(&mut self, new_count: u16) {
        let old_count = self.header().num_slots;
        assert!(new_count <= old_count);
        if new_count == old_count {
            return;
        }

        // Subtract removed data from space_used.
        for i in new_count..old_count {
            let s = self.slot(i);
            let data_len = s.key_len as usize + s.val_len as usize;
            self.header_mut().space_used -= data_len as u32;
        }

        self.header_mut().num_slots = new_count;
        self.header_mut().flags |= FLAG_HAS_GARBAGE;
        self.compactify();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use proptest::prelude::*;
    use proptest::test_runner::{Config as ProptestConfig, TestRunner};

    use super::*;

    /// Aligned buffer for testing — matches Page alignment.
    #[repr(C, align(4096))]
    struct AlignedBuf([u8; PAGE_SIZE]);

    fn new_page() -> AlignedBuf {
        AlignedBuf([0u8; PAGE_SIZE])
    }

    struct GeneratedCase {
        state: u64,
    }

    impl GeneratedCase {
        fn new(seed: u64) -> Self {
            Self {
                state: seed ^ 0x8ebc_6af0_9c88_c6e3,
            }
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        fn usize(&mut self, min: usize, max: usize) -> usize {
            if min == max {
                return min;
            }
            min + (self.next_u64() as usize % (max - min + 1))
        }

        fn bytes(&mut self, min_size: usize, max_size: usize) -> Vec<u8> {
            let len = self.usize(min_size, max_size);
            (0..len).map(|_| self.next_u64() as u8).collect()
        }

        fn bytes_exact(&mut self, len: usize) -> Vec<u8> {
            (0..len).map(|_| self.next_u64() as u8).collect()
        }
    }

    macro_rules! generated_test {
        ($cases:expr, |$tc:ident| $body:block) => {{
            let mut runner = TestRunner::new(ProptestConfig::with_cases($cases));
            runner
                .run(&any::<u64>(), |seed| {
                    let mut $tc = GeneratedCase::new(seed);
                    $body
                    Ok(())
                })
                .unwrap();
        }};
    }

    #[test]
    fn works_on_unaligned_page_storage() {
        let mut raw = vec![0u8; PAGE_SIZE + 64];
        let base = raw.as_mut_ptr() as usize;
        let aligned_off = (8 - (base % 8)) % 8;
        let off = if (base + aligned_off).is_multiple_of(4096) {
            aligned_off + 8
        } else {
            aligned_off
        };
        let page: &mut [u8; PAGE_SIZE] = (&mut raw[off..off + PAGE_SIZE]).try_into().unwrap();
        assert_ne!(
            (page.as_ptr() as usize) % 4096,
            0,
            "test page unexpectedly remained 4096-byte aligned"
        );
        let sp = SlottedPage::init(page);
        sp.insert(0, b"hello", b"world");

        let sp = SlottedPage::from_page(page);
        assert_eq!(sp.num_slots(), 1);
        assert_eq!(sp.get_key(0), b"hello");
        assert_eq!(sp.get_value(0), b"world");
    }

    #[test]
    fn insert_orders_slots_across_positions() {
        let cases = [
            (
                "sorted_append",
                vec![
                    (0, b"aaa".as_slice(), b"v1".as_slice()),
                    (1, b"bbb", b"v2"),
                    (2, b"ccc", b"v3"),
                ],
                vec![
                    (b"aaa".as_slice(), b"v1".as_slice()),
                    (b"bbb", b"v2"),
                    (b"ccc", b"v3"),
                ],
            ),
            (
                "shift_front",
                vec![(0, b"ccc".as_slice(), b"v3".as_slice()), (0, b"aaa", b"v1")],
                vec![(b"aaa".as_slice(), b"v1".as_slice()), (b"ccc", b"v3")],
            ),
            (
                "insert_middle",
                vec![
                    (0, b"aaa".as_slice(), b"v1".as_slice()),
                    (1, b"ccc", b"v3"),
                    (1, b"bbb", b"v2"),
                ],
                vec![
                    (b"aaa".as_slice(), b"v1".as_slice()),
                    (b"bbb", b"v2"),
                    (b"ccc", b"v3"),
                ],
            ),
        ];

        for (label, ops, expected) in cases {
            let mut buf = new_page();
            let sp = SlottedPage::init(&mut buf.0);
            for (slot, key, value) in ops {
                sp.insert(slot, key, value);
            }

            assert_eq!(
                sp.num_slots() as usize,
                expected.len(),
                "{label}: wrong slot count"
            );
            for (idx, (key, value)) in expected.iter().enumerate() {
                assert_eq!(
                    sp.get_key(idx as u16),
                    *key,
                    "{label}: wrong key at slot {idx}"
                );
                assert_eq!(
                    sp.get_value(idx as u16),
                    *value,
                    "{label}: wrong value at slot {idx}"
                );
            }
        }
    }

    #[test]
    fn lower_bound_cases_cover_empty_heads_and_boundaries() {
        let cases = [
            (
                "empty",
                Vec::<&[u8]>::new(),
                vec![(b"anything".as_slice(), (0, false))],
            ),
            (
                "basic",
                vec![b"bbb".as_slice(), b"ddd", b"fff"],
                vec![
                    (b"bbb".as_slice(), (0, true)),
                    (b"ddd", (1, true)),
                    (b"fff", (2, true)),
                    (b"aaa", (0, false)),
                    (b"ccc", (1, false)),
                    (b"eee", (2, false)),
                    (b"ggg", (3, false)),
                ],
            ),
            (
                "short_keys",
                vec![b"a".as_slice(), b"b", b"c"],
                vec![
                    (b"a".as_slice(), (0, true)),
                    (b"b", (1, true)),
                    (b"c", (2, true)),
                    (b"ab", (1, false)),
                ],
            ),
            (
                "same_head",
                vec![b"abcd1".as_slice(), b"abcd2", b"abcd3"],
                vec![
                    (b"abcd1".as_slice(), (0, true)),
                    (b"abcd2", (1, true)),
                    (b"abcd3", (2, true)),
                    (b"abcd15", (1, false)),
                ],
            ),
        ];

        for (label, keys, probes) in cases {
            let mut buf = new_page();
            let sp = SlottedPage::init(&mut buf.0);
            for (idx, key) in keys.iter().enumerate() {
                sp.insert(idx as u16, key, b"value");
            }
            for (probe, expected) in probes {
                assert_eq!(
                    sp.lower_bound(probe),
                    expected,
                    "{label}: probe {probe:?} mismatch"
                );
            }
        }
    }

    #[test]
    fn remove_and_garbage() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);

        sp.insert(0, b"aaa", b"v1");
        sp.insert(1, b"bbb", b"v2");
        sp.insert(2, b"ccc", b"v3");

        let free_before = sp.free_space();
        sp.remove(1); // remove "bbb"

        assert_eq!(sp.num_slots(), 2);
        assert_eq!(sp.get_key(0), b"aaa");
        assert_eq!(sp.get_key(1), b"ccc");
        assert!(sp.has_garbage());
        // free_space doesn't increase (data region has hole).
        assert_eq!(sp.free_space(), free_before + SLOT_SIZE);
        // But free_space_after_compaction does.
        assert!(sp.free_space_after_compaction() > sp.free_space());
    }

    #[test]
    fn compactify() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);

        sp.insert(0, b"aaa", b"value1");
        sp.insert(1, b"bbb", b"value2");
        sp.insert(2, b"ccc", b"value3");
        sp.insert(3, b"ddd", b"value4");
        sp.insert(4, b"eee", b"value5");

        sp.remove(1); // remove "bbb"
        sp.remove(2); // remove "ddd" (was at 3, now at 2 after first remove)

        assert!(sp.has_garbage());
        let free_after_compaction = sp.free_space_after_compaction();

        sp.compactify();

        assert!(!sp.has_garbage());
        assert_eq!(sp.free_space(), free_after_compaction);
        assert_eq!(sp.num_slots(), 3);
        assert_eq!(sp.get_key(0), b"aaa");
        assert_eq!(sp.get_key(1), b"ccc");
        assert_eq!(sp.get_key(2), b"eee");
        assert_eq!(sp.get_value(0), b"value1");
        assert_eq!(sp.get_value(1), b"value3");
        assert_eq!(sp.get_value(2), b"value5");
    }

    #[test]
    fn compactify_preserves_reserved_suffix_nonempty() {
        let mut buf = new_page();
        {
            let sp = SlottedPage::init(&mut buf.0);
            sp.reserve_suffix(16);
            sp.insert(0, b"aaa", b"value1");
            sp.insert(1, b"bbb", b"value2");
            sp.insert(2, b"ccc", b"value3");
            sp.remove(1);
        }

        let suffix_before = buf.0[PAGE_SIZE - 16..].to_vec();

        let sp = SlottedPage::from_page_mut(&mut buf.0);
        let free_after_compaction = sp.free_space_after_compaction();
        sp.compactify();

        assert_eq!(sp.free_space(), free_after_compaction);
        assert_eq!(&buf.0[PAGE_SIZE - 16..], suffix_before.as_slice());
    }

    #[test]
    fn compactify_preserves_reserved_suffix_empty() {
        let mut buf = new_page();
        {
            let sp = SlottedPage::init(&mut buf.0);
            sp.reserve_suffix(16);
        }
        buf.0[PAGE_SIZE - 16..PAGE_SIZE - 8].copy_from_slice(&0x1122334455667788u64.to_ne_bytes());
        buf.0[PAGE_SIZE - 8..].copy_from_slice(&0x8877665544332211u64.to_ne_bytes());

        let sp = SlottedPage::from_page_mut(&mut buf.0);
        sp.insert(0, b"aaa", b"value1");
        sp.remove(0);
        sp.compactify();

        assert_eq!(sp.num_slots(), 0);
        assert_eq!(sp.space_used(), 16);
        assert_eq!(sp.free_space(), PAGE_SIZE - HEADER_SIZE - 16);
        assert_eq!(
            &buf.0[PAGE_SIZE - 16..PAGE_SIZE - 8],
            &0x1122334455667788u64.to_ne_bytes()
        );
        assert_eq!(
            &buf.0[PAGE_SIZE - 8..],
            &0x8877665544332211u64.to_ne_bytes()
        );
    }

    #[test]
    fn fill_to_capacity() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);

        let key = [0xABu8; 8];
        let val = [0xCDu8; 16];
        let entry_size = SLOT_SIZE + key.len() + val.len(); // 12 + 8 + 16 = 36

        let mut count = 0u16;
        while sp.can_insert(key.len(), val.len()) {
            // Use count as suffix to make keys unique and sorted.
            let mut k = key;
            k[6] = (count >> 8) as u8;
            k[7] = count as u8;
            sp.insert(count, &k, &val);
            count += 1;
        }

        assert!(count > 0);
        assert!(!sp.can_insert(key.len(), val.len()));
        assert_eq!(sp.num_slots(), count);

        // Max theoretical: (PAGE_SIZE - HEADER_SIZE) / entry_size
        let max = (PAGE_SIZE - HEADER_SIZE) / entry_size;
        assert_eq!(count as usize, max);

        // Verify all entries.
        for i in 0..count {
            let k = sp.get_key(i);
            assert_eq!(k.len(), 8);
            assert_eq!(k[6], (i >> 8) as u8);
            assert_eq!(k[7], i as u8);
            assert_eq!(sp.get_value(i), &val);
        }
    }

    #[test]
    fn copy_key_value_range_split() {
        let mut buf1 = new_page();
        let sp1 = SlottedPage::init(&mut buf1.0);

        for i in 0..10u8 {
            let key = [b'a' + i];
            let val = [i; 4];
            sp1.insert(i as u16, &key, &val);
        }

        // Split upper half (slots 5..10) into a new page.
        let mut buf2 = new_page();
        let sp2 = SlottedPage::init(&mut buf2.0);

        sp1.copy_key_value_range(sp2, 0, 5, 5);

        assert_eq!(sp2.num_slots(), 5);
        for i in 0..5u8 {
            assert_eq!(sp2.get_key(i as u16), &[b'a' + 5 + i]);
            assert_eq!(sp2.get_value(i as u16), &[5 + i; 4]);
        }

        // Truncate sp1 to keep only lower half.
        sp1.truncate(5);
        assert_eq!(sp1.num_slots(), 5);
        for i in 0..5u8 {
            assert_eq!(sp1.get_key(i as u16), &[b'a' + i]);
            assert_eq!(sp1.get_value(i as u16), &[i; 4]);
        }
    }

    #[test]
    fn empty_key() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);

        sp.insert(0, b"", b"empty_key_value");
        assert_eq!(sp.get_key(0), b"");
        assert_eq!(sp.get_value(0), b"empty_key_value");
        assert_eq!(sp.lower_bound(b""), (0, true));
    }

    #[test]
    fn empty_value() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);

        sp.insert(0, b"key", b"");
        assert_eq!(sp.get_key(0), b"key");
        assert_eq!(sp.get_value(0), b"");
    }

    #[test]
    fn large_value() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);

        let key = b"k";
        let max_data = PAGE_SIZE - HEADER_SIZE - SLOT_SIZE - key.len();
        let val = vec![0x42u8; max_data];

        assert!(sp.can_insert(key.len(), val.len()));
        sp.insert(0, key, &val);
        assert_eq!(sp.get_key(0), key);
        assert_eq!(sp.get_value(0), &val[..]);
        assert!(!sp.can_insert(1, 0)); // no more room
    }

    #[test]
    fn update_value_same_length() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);

        sp.insert(0, b"key", b"val1");
        assert!(sp.update_value_if_same_length(0, b"val2"));
        assert_eq!(sp.get_value(0), b"val2");

        // Different length should fail.
        assert!(!sp.update_value_if_same_length(0, b"longer_value"));
        assert_eq!(sp.get_value(0), b"val2"); // unchanged
    }

    #[test]
    fn insert_triggers_compaction() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);

        // Fill most of the page.
        let val = [0u8; 100];
        let mut i = 0u16;
        while sp.can_insert(4, val.len()) {
            let key = i.to_be_bytes();
            let mut k = [0u8; 4];
            k[2..4].copy_from_slice(&key);
            sp.insert(i, &k, &val);
            i += 1;
        }
        let full_count = sp.num_slots();

        // Remove half the entries to create garbage.
        for j in (0..full_count).rev().step_by(2) {
            sp.remove(j);
        }
        assert!(sp.has_garbage());

        // Now insert should succeed via compaction.
        let remaining = sp.num_slots();
        assert!(sp.can_insert(4, val.len()));
        let key = [0xFF, 0xFF, 0xFF, 0xFFu8]; // sorts last
        sp.insert(remaining, &key, &val);
        assert_eq!(sp.get_key(remaining), &key);
    }

    #[test]
    fn head_computation() {
        assert_eq!(SlottedPage::make_head(b""), 0x00000000);
        assert_eq!(SlottedPage::make_head(b"\xAB"), 0xAB000000);
        assert_eq!(SlottedPage::make_head(b"\xAB\xCD"), 0xABCD0000);
        assert_eq!(SlottedPage::make_head(b"\xAB\xCD\xEF"), 0xABCDEF00);
        assert_eq!(SlottedPage::make_head(b"\xAB\xCD\xEF\x01"), 0xABCDEF01);
        assert_eq!(SlottedPage::make_head(b"\xAB\xCD\xEF\x01\x99"), 0xABCDEF01);
    }

    #[test]
    fn lower_bound_with_insert() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);

        // Build a page using lower_bound to find insertion positions.
        let keys: &[&[u8]] = &[b"dog", b"cat", b"bird", b"fish", b"ant"];
        for key in keys {
            let (pos, _) = sp.lower_bound(key);
            sp.insert(pos, key, b"");
        }

        // Should be sorted.
        assert_eq!(sp.num_slots(), 5);
        assert_eq!(sp.get_key(0), b"ant");
        assert_eq!(sp.get_key(1), b"bird");
        assert_eq!(sp.get_key(2), b"cat");
        assert_eq!(sp.get_key(3), b"dog");
        assert_eq!(sp.get_key(4), b"fish");
    }

    struct SlottedPageModelMachine {
        buf: AlignedBuf,
        model: BTreeMap<Vec<u8>, Vec<u8>>,
    }

    impl SlottedPageModelMachine {
        fn page(&self) -> &SlottedPage {
            SlottedPage::from_page(&self.buf.0)
        }

        fn page_mut(&mut self) -> &mut SlottedPage {
            SlottedPage::from_page_mut(&mut self.buf.0)
        }

        fn ordered_entry(&self, index: usize) -> (&Vec<u8>, &Vec<u8>) {
            self.model
                .iter()
                .nth(index)
                .expect("state machine chose an out-of-range model slot")
        }

        fn expected_lower_bound(&self, probe: &[u8]) -> (u16, bool) {
            for (idx, key) in self.model.keys().enumerate() {
                match key.as_slice().cmp(probe) {
                    Ordering::Less => continue,
                    Ordering::Equal => return (idx as u16, true),
                    Ordering::Greater => return (idx as u16, false),
                }
            }

            (self.model.len() as u16, false)
        }

        fn insert_unique(&mut self, tc: &mut GeneratedCase) {
            let key = tc.bytes(0, 16);
            let value = tc.bytes(0, 32);

            if self.model.contains_key(&key) || !self.page().can_insert(key.len(), value.len()) {
                return;
            }

            let (slot, found) = self.page().lower_bound(&key);
            assert!(!found, "new key unexpectedly matched an existing slot");

            self.page_mut().insert(slot, &key, &value);
            self.model.insert(key, value);
        }

        fn remove_existing(&mut self, tc: &mut GeneratedCase) {
            if self.model.is_empty() {
                return;
            }

            let index = tc.usize(0, self.model.len() - 1);
            let key = self.ordered_entry(index).0.clone();
            let (slot, found) = self.page().lower_bound(&key);
            assert!(found, "model key must exist in the page before removal");

            self.page_mut().remove(slot);
            let removed = self.model.remove(&key);
            assert!(removed.is_some(), "model removal must succeed");
        }

        fn update_existing_same_length(&mut self, tc: &mut GeneratedCase) {
            if self.model.is_empty() {
                return;
            }

            let index = tc.usize(0, self.model.len() - 1);
            let (key, old_value) = self.ordered_entry(index);
            let key = key.clone();
            let old_len = old_value.len();
            let new_value = tc.bytes_exact(old_len);

            let (slot, found) = self.page().lower_bound(&key);
            assert!(found, "model key must exist in the page before update");
            assert!(
                self.page_mut()
                    .update_value_if_same_length(slot, &new_value),
                "same-length update must succeed"
            );
            self.model.insert(key, new_value);
        }

        fn reject_different_length_update(&mut self, tc: &mut GeneratedCase) {
            if self.model.is_empty() {
                return;
            }

            let index = tc.usize(0, self.model.len() - 1);
            let (key, old_value) = self.ordered_entry(index);
            let key = key.clone();
            let old_value = old_value.clone();
            let old_len = old_value.len();
            let mut new_len = tc.usize(0, 32);
            if new_len == old_len {
                new_len = if old_len == 32 { 31 } else { old_len + 1 };
            }

            let new_value = tc.bytes_exact(new_len);
            let (slot, found) = self.page().lower_bound(&key);
            assert!(
                found,
                "model key must exist in the page before rejected update"
            );
            assert!(
                !self
                    .page_mut()
                    .update_value_if_same_length(slot, &new_value),
                "different-length update must be rejected"
            );
            assert_eq!(
                self.model.get(&key),
                Some(&old_value),
                "rejected update must not change the model"
            );
        }

        fn compactify(&mut self) {
            self.page_mut().compactify();
        }

        fn page_matches_model(&mut self) {
            let page = self.page();

            assert_eq!(
                page.num_slots() as usize,
                self.model.len(),
                "page slot count must match model entry count"
            );
            assert!(
                page.free_space_after_compaction() >= page.free_space(),
                "compaction cannot reduce available free space"
            );

            let expected_space_used: usize = self
                .model
                .iter()
                .map(|(key, value)| key.len() + value.len())
                .sum();
            assert_eq!(
                page.space_used(),
                expected_space_used,
                "page live-data bytes must match the model payload"
            );

            for (idx, (key, value)) in self.model.iter().enumerate() {
                assert_eq!(
                    page.get_key(idx as u16),
                    key.as_slice(),
                    "slot {idx} key mismatch"
                );
                assert_eq!(
                    page.get_value(idx as u16),
                    value.as_slice(),
                    "slot {idx} value mismatch"
                );
                assert_eq!(
                    page.lower_bound(key),
                    (idx as u16, true),
                    "exact lower_bound must find the existing key at slot {idx}"
                );
            }
        }

        fn lower_bound_matches_model(&mut self, tc: &mut GeneratedCase) {
            let probe = tc.bytes(0, 16);
            let expected = self.expected_lower_bound(&probe);
            assert_eq!(
                self.page().lower_bound(&probe),
                expected,
                "lower_bound must agree with the ordered model for probe {probe:?}"
            );
        }
    }

    #[test]
    #[cfg(not(miri))]
    fn generated_inserts_preserve_sorted_lookup_order() {
        generated_test!(256, |tc| {
            let mut buf = new_page();
            let sp = SlottedPage::init(&mut buf.0);

            let mut expected = BTreeMap::new();
            let entry_count = tc.usize(0, 32);
            for _ in 0..entry_count {
                let key = tc.bytes(0, 16);
                let value = tc.bytes(0, 32);
                expected.entry(key).or_insert(value);
            }

            for (key, value) in &expected {
                let (slot, found) = sp.lower_bound(key);
                assert!(
                    !found,
                    "deduplicated generated keys should not already exist in the page"
                );
                sp.insert(slot, key, value);
            }

            assert_eq!(
                sp.num_slots() as usize,
                expected.len(),
                "page slot count must match inserted unique keys"
            );

            for (idx, (key, value)) in expected.iter().enumerate() {
                assert_eq!(
                    sp.get_key(idx as u16),
                    key.as_slice(),
                    "key order must stay sorted"
                );
                assert_eq!(
                    sp.get_value(idx as u16),
                    value.as_slice(),
                    "value must stay attached to its key"
                );
                assert_eq!(
                    sp.lower_bound(key),
                    (idx as u16, true),
                    "exact key lookup must find the inserted slot"
                );
            }
        });
    }

    #[test]
    #[cfg(not(miri))]
    fn stateful_model_matches_slotted_page() {
        generated_test!(256, |tc| {
            let mut buf = new_page();
            SlottedPage::init(&mut buf.0);
            let mut machine = SlottedPageModelMachine {
                buf,
                model: BTreeMap::new(),
            };
            let steps = tc.usize(1, 64);
            for _ in 0..steps {
                match tc.usize(0, 4) {
                    0 => machine.insert_unique(&mut tc),
                    1 => machine.remove_existing(&mut tc),
                    2 => machine.update_existing_same_length(&mut tc),
                    3 => machine.reject_different_length_update(&mut tc),
                    _ => machine.compactify(),
                }
                machine.page_matches_model();
                machine.lower_bound_matches_model(&mut tc);
            }
        });
    }

    #[test]
    fn remove_first_and_last() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);

        sp.insert(0, b"aaa", b"v1");
        sp.insert(1, b"bbb", b"v2");
        sp.insert(2, b"ccc", b"v3");

        sp.remove(2); // remove last
        assert_eq!(sp.num_slots(), 2);
        assert_eq!(sp.get_key(1), b"bbb");

        sp.remove(0); // remove first
        assert_eq!(sp.num_slots(), 1);
        assert_eq!(sp.get_key(0), b"bbb");
    }

    #[test]
    fn truncate_to_zero() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);

        sp.insert(0, b"aaa", b"v1");
        sp.insert(1, b"bbb", b"v2");

        sp.truncate(0);
        assert_eq!(sp.num_slots(), 0);
        assert_eq!(sp.free_space(), PAGE_SIZE - HEADER_SIZE);
    }

    #[test]
    #[cfg(not(miri))]
    fn many_small_entries() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);

        // Insert many 1-byte key, 1-byte value entries.
        // Each needs SLOT_SIZE + 2 = 14 bytes.
        let max = (PAGE_SIZE - HEADER_SIZE) / (SLOT_SIZE + 2);
        for i in 0..max {
            let k = [i as u8];
            let v = [!i as u8];
            let (pos, _) = sp.lower_bound(&k);
            sp.insert(pos, &k, &v);
        }
        assert_eq!(sp.num_slots() as usize, max);

        // Verify all.
        for i in 0..max {
            let k = [i as u8];
            let (pos, found) = sp.lower_bound(&k);
            assert!(found, "key {i} not found");
            assert_eq!(sp.get_value(pos), &[!i as u8]);
        }
    }

    #[test]
    #[should_panic(expected = "slotted page full")]
    fn insert_when_full_panics() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);

        let key = b"k";
        let max_data = PAGE_SIZE - HEADER_SIZE - SLOT_SIZE - key.len();
        let val = vec![0u8; max_data];
        sp.insert(0, key, &val);

        // Page is completely full — this should panic.
        sp.insert(1, b"x", b"y");
    }

    #[test]
    #[should_panic]
    fn remove_out_of_bounds_panics() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);
        sp.insert(0, b"k", b"v");
        sp.remove(1);
    }

    #[test]
    fn compactify_idempotent_on_clean_page() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);
        sp.insert(0, b"aaa", b"v1");
        sp.insert(1, b"bbb", b"v2");

        let free_before = sp.free_space();
        let slots_before = sp.num_slots();

        sp.compactify();

        assert_eq!(sp.free_space(), free_before);
        assert_eq!(sp.num_slots(), slots_before);
        assert_eq!(sp.get_key(0), b"aaa");
        assert_eq!(sp.get_value(0), b"v1");
        assert_eq!(sp.get_key(1), b"bbb");
        assert_eq!(sp.get_value(1), b"v2");
    }

    #[test]
    fn reinsert_after_remove_restores_page_state() {
        let mut buf = new_page();
        let sp = SlottedPage::init(&mut buf.0);
        sp.insert(0, b"key1", b"val1");
        sp.insert(1, b"key2", b"val2");

        let free_before = sp.free_space_after_compaction();

        sp.remove(0);
        sp.compactify();
        sp.insert(0, b"key1", b"val1");

        assert_eq!(sp.num_slots(), 2);
        assert_eq!(sp.get_key(0), b"key1");
        assert_eq!(sp.get_value(0), b"val1");
        assert_eq!(sp.get_key(1), b"key2");
        assert_eq!(sp.get_value(1), b"val2");
        assert_eq!(
            sp.free_space_after_compaction(),
            free_before,
            "page should return to pre-remove free space"
        );
    }

    #[test]
    fn reserve_suffix_survives_copy_key_value_range_split() {
        let mut src_buf = new_page();
        let src = SlottedPage::init(&mut src_buf.0);
        src.reserve_suffix(64);

        src.insert(0, b"aaa", b"v1");
        src.insert(1, b"bbb", b"v2");
        src.insert(2, b"ccc", b"v3");

        let mut dst_buf = new_page();
        let dst = SlottedPage::init(&mut dst_buf.0);
        dst.reserve_suffix(64);

        src.copy_key_value_range(dst, 0, 1, 2);

        assert_eq!(dst.num_slots(), 2);
        assert_eq!(dst.get_key(0), b"bbb");
        assert_eq!(dst.get_value(0), b"v2");
        assert_eq!(dst.get_key(1), b"ccc");
        assert_eq!(dst.get_value(1), b"v3");

        // The reserved suffix should reduce free space on dst.
        // free_space_after_compaction = PAGE_SIZE - HEADER_SIZE - suffix - slots - data
        let expected_free = PAGE_SIZE
            - HEADER_SIZE
            - 64 // reserved suffix
            - 2 * SLOT_SIZE
            - (3 + 2 + 3 + 2); // data: "bbb"+"v2" + "ccc"+"v3"
        assert_eq!(
            dst.free_space_after_compaction(),
            expected_free,
            "dst free space should account for reserved suffix"
        );
    }
}
