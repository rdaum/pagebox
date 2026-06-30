//! Sharded free-page allocator with central best-fit freelist plus per-shard
//! monotonic reservation.
//!
//! The [`BufferPool`](crate::buffer_pool::BufferPool) asks the allocator for
//! fresh page IDs under each `allocate_and_fix*` call. The page IDs come from
//! [`crate::buffer_frame::PageId`]s. Allocation is hot:
//! each thread holds a stable shard hint (see `thread_alloc_shard_hint` in
//! `buffer_pool`) so contention across threads is amortised. The design has
//! three layers, consulted in order:
//!
//! 1. **Hot shard buckets** — each shard owns a small array of power-of-two
//!    buckets (`HOT_BUCKET_LIMITS`). Small allocations (`len <= 256` base
//!    pages) are served lock-free from per-shard `SegQueue`s; allocations
//!    larger than the largest bucket bypass the hot path.
//! 2. **Central best-fit freelist** — a single `Mutex<BTreeMap<u64,
//!    FreeExtent>>` indexed by start page number, holding reusable extents
//!    retired via [`FreePageAllocator::promote_reusable_extent`]. Adjacent
//!    central layer is also what refills hot buckets in batches
//!    (`HOT_REFILL_BATCH = 8`).
//! 3. **Monotonic growth** — each shard reserves an independent range
//!    (`MONOTONIC_REFILL_BASE_PAGES = 256`) carved from a single global cursor.
//!    Allocations in the shard's reserved range touch no global lock; when the
//!    reservation is empty, the shard grabs the next 256-page chunk. This is
//!    what `next_page_number()` reports.
//!
//! ## Allocation policy
//!
//! `allocate_page(shard_hint)` always prefers reusable extents over
//! monotonic growth (so reopened extents from a retired tree are consumed
//! before the data file grows). Within a single shard the order is:
//!
//! 1. The matching hot bucket, falling back to larger buckets.
//! 2. A central refill, which seeds the hot buckets with a batch extent and
//!    splits it for the requested size.
//! 3. Steal from sibling shards (`1..shards.len()`).
//! 4. Central best-fit (for sizes too large for the hot buckets).
//! 5. Monotonic reservation in this shard, then 256-page reservation from the
//!    global cursor.
//!
//! ## Uniqueness and reuse
//!
//! Page IDs returned by `allocate_page` are unique for the lifetime of the
//! allocator. Reuse across reopen is gated by the WAL checkpoint: a retired
//! page extent is held back in `pending_reusable_extents` on
//! `BufferPool::retire_unlinked_exclusive_frame` until `flush` makes the
//! unlink durable, then handed to `promote_reusable_extent`. The allocator
//! itself performs no durability bookkeeping — see `buffer_pool.rs`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crossbeam_queue::SegQueue;
use parking_lot::Mutex;

use crate::buffer_frame::PageId;

const HOT_BUCKET_LIMITS: [u64; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const HOT_REFILL_BATCH: usize = 8;
const MONOTONIC_REFILL_BASE_PAGES: u64 = 256;

/// A contiguous range of free base-page numbers `[start, start + len)`.
///
/// Owned by the central freelist and the per-shard hot buckets. `start` is
/// always `> 0`: page 0 is reserved as the page-store header page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeExtent {
    pub start_page_number: u64,
    pub len: u64,
}

impl FreeExtent {
    /// Construct a free extent. Panics if `start_page_number` is `0` (page 0
    /// is the header page and never reusable) or `len` is zero.
    pub fn new(start_page_number: u64, len: u64) -> Self {
        assert!(start_page_number > 0, "page 0 is reserved");
        assert!(len > 0, "free extents must be non-empty");
        Self {
            start_page_number,
            len,
        }
    }

    fn end_page_number(self) -> u64 {
        self.start_page_number
            .checked_add(self.len)
            .expect("free extent end overflow")
    }

    fn split_head(&mut self, len: u64) -> Self {
        assert!(len > 0 && len <= self.len);
        let allocated = FreeExtent::new(self.start_page_number, len);
        self.start_page_number += len;
        self.len -= len;
        allocated
    }
}

struct CentralInner {
    by_start_page_number: BTreeMap<u64, FreeExtent>,
    free_base_pages: u64,
}

impl CentralInner {
    fn new() -> Self {
        Self {
            by_start_page_number: BTreeMap::new(),
            free_base_pages: 0,
        }
    }

    fn insert(&mut self, mut extent: FreeExtent) {
        let added_len = extent.len;
        let mut start = extent.start_page_number;
        let mut end = extent.end_page_number();

        if let Some((&prev_start, &prev)) = self.by_start_page_number.range(..=start).next_back() {
            let prev_end = prev.end_page_number();
            assert!(
                prev_end <= start,
                "free extent overlaps existing extent: {extent:?} overlaps {prev:?}"
            );
            if prev_end == start {
                self.by_start_page_number.remove(&prev_start);
                start = prev.start_page_number;
                extent.len += prev.len;
            }
        }

        if let Some((&next_start, &next)) = self.by_start_page_number.range(start..).next() {
            assert!(
                end <= next_start,
                "free extent overlaps existing extent: {extent:?} overlaps {next:?}"
            );
            if end == next_start {
                self.by_start_page_number.remove(&next_start);
                end = next.end_page_number();
                extent.len = end - start;
            }
        }

        extent.start_page_number = start;
        self.free_base_pages += added_len;
        self.by_start_page_number.insert(start, extent);
    }

    fn allocate_best_fit(&mut self, len: u64) -> Option<FreeExtent> {
        let (&start, &extent) = self
            .by_start_page_number
            .iter()
            .filter(|(_, extent)| extent.len >= len)
            .min_by_key(|(_, extent)| extent.len)?;
        self.by_start_page_number.remove(&start);
        self.free_base_pages -= len;

        let mut remainder = extent;
        let allocated = remainder.split_head(len);
        if remainder.len > 0 {
            self.by_start_page_number
                .insert(remainder.start_page_number, remainder);
        }
        Some(allocated)
    }

    fn free_base_pages(&self) -> u64 {
        self.free_base_pages
    }

    fn extent_count(&self) -> usize {
        self.by_start_page_number.len()
    }
}

struct CentralExtentAllocator {
    inner: Mutex<CentralInner>,
}

impl CentralExtentAllocator {
    fn new() -> Self {
        Self {
            inner: Mutex::new(CentralInner::new()),
        }
    }

    fn insert(&self, extent: FreeExtent) {
        self.inner.lock().insert(extent);
    }

    fn allocate_best_fit(&self, len: u64) -> Option<FreeExtent> {
        self.inner.lock().allocate_best_fit(len)
    }

    fn free_base_pages(&self) -> u64 {
        self.inner.lock().free_base_pages()
    }

    fn extent_count(&self) -> usize {
        self.inner.lock().extent_count()
    }
}

struct FreePageShard {
    buckets: Box<[SegQueue<FreeExtent>]>,
    monotonic: MonotonicRange,
}

#[repr(align(64))]
struct MonotonicRange {
    next_page_number: AtomicU64,
    end_page_number: AtomicU64,
    refill_lock: Mutex<()>,
}

impl MonotonicRange {
    fn new() -> Self {
        Self {
            next_page_number: AtomicU64::new(0),
            end_page_number: AtomicU64::new(0),
            refill_lock: Mutex::new(()),
        }
    }

    fn try_allocate(&self, len: u64) -> Option<FreeExtent> {
        let next = self.next_page_number.fetch_add(len, Ordering::AcqRel);
        if next == 0 {
            return None;
        }
        let new_next = next.checked_add(len).expect("page number overflow");
        let end = self.end_page_number.load(Ordering::Acquire);
        if new_next <= end {
            return Some(FreeExtent::new(next, len));
        }
        None
    }

    fn install_reserved_tail(&self, next_page_number: u64, end_page_number: u64) {
        self.end_page_number
            .store(end_page_number, Ordering::Release);
        self.next_page_number
            .store(next_page_number, Ordering::Release);
    }
}

impl FreePageShard {
    fn new() -> Self {
        let buckets = (0..HOT_BUCKET_LIMITS.len())
            .map(|_| SegQueue::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            buckets,
            monotonic: MonotonicRange::new(),
        }
    }

    fn push(&self, extent: FreeExtent) -> Result<(), FreeExtent> {
        let Some(bucket) = hot_bucket_for_len(extent.len) else {
            return Err(extent);
        };
        self.buckets[bucket].push(extent);
        Ok(())
    }

    fn pop_at_least(&self, len: u64) -> Option<FreeExtent> {
        let start_bucket = hot_bucket_for_len(len)?;
        for bucket in start_bucket..self.buckets.len() {
            if let Some(extent) = self.buckets[bucket].pop() {
                debug_assert!(extent.len >= len);
                return Some(extent);
            }
        }
        None
    }
}

/// Sharded free-page allocator. See the [module-level docs](self) for the
/// layered allocation policy.
pub struct FreePageAllocator {
    global_next_page_number: AtomicU64,
    shards: Box<[FreePageShard]>,
    central: CentralExtentAllocator,
}

impl FreePageAllocator {
    /// Construct an allocator at `next_page_number` with `shard_count`
    /// per-shard monotonic reservations. `shard_count` is clamped to at least
    /// one. Panics if `next_page_number == 0` (page 0 is reserved).
    pub fn new(next_page_number: u64, shard_count: usize) -> Self {
        assert!(next_page_number > 0, "page 0 is reserved");
        let shard_count = shard_count.max(1);
        let shards = (0..shard_count)
            .map(|_| FreePageShard::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            global_next_page_number: AtomicU64::new(next_page_number),
            shards,
            central: CentralExtentAllocator::new(),
        }
    }

    /// Insert a reusable extent into the central freelist. Adjacent extents
    /// coalesce on insert, so batched retirements of contiguous pages may
    /// later satisfy a larger-class allocation in one piece. Safe to call
    /// from any thread.
    pub fn promote_reusable_extent(&self, extent: FreeExtent) {
        self.central.insert(extent);
    }

    /// Allocate a single page, preferring reusable shards over
    /// monotonic growth. The returned [`PageId`] is the start base-page
    /// number; `shard_hint` selects which shard is tried first (modulo
    /// shard count) — pass the calling thread's alloc-shard hint for
    /// cache locality. See the [module-level docs](self) for the full
    /// fallback order.
    pub fn allocate_page(&self, shard_hint: usize) -> PageId {
        // With a single page class, one allocation consumes one base page.
        let len = 1u64;
        if let Some(extent) = self.allocate_extent(len, shard_hint) {
            return extent.start_page_number;
        }
        self.allocate_monotonic_extent(len, shard_hint)
            .start_page_number
    }

    /// High-water mark of the global monotonic cursor. Each shard reserves
    /// `MONOTONIC_REFILL_BASE_PAGES` ahead of need, so this is an upper bound
    /// on the highest base page number ever handed out (not the exact
    /// boundary of allocated pages). Used by benchmarks and the page store
    /// to size the data file.
    pub fn next_page_number(&self) -> u64 {
        self.global_next_page_number.load(Ordering::Relaxed)
    }

    /// Total base pages currently held in the central freelist. Excludes
    /// pages still cached in per-shard hot buckets and excludes any
    /// unreservable monotonic-range tail.
    pub fn central_free_base_pages(&self) -> u64 {
        self.central.free_base_pages()
    }

    /// Number of distinct extents (post-coalescing) in the central freelist.
    /// Useful for sanity-checking the freelist shape after a workload.
    pub fn central_extent_count(&self) -> usize {
        self.central.extent_count()
    }

    fn allocate_extent(&self, len: u64, shard_hint: usize) -> Option<FreeExtent> {
        let shard_idx = shard_hint % self.shards.len();
        if let Some(extent) = self.allocate_from_shard(shard_idx, len) {
            return Some(extent);
        }

        self.refill_shard(shard_idx, len);
        if let Some(extent) = self.allocate_from_shard(shard_idx, len) {
            return Some(extent);
        }

        for offset in 1..self.shards.len() {
            let steal_idx = (shard_idx + offset) % self.shards.len();
            if let Some(extent) = self.allocate_from_shard(steal_idx, len) {
                return Some(extent);
            }
        }

        self.central.allocate_best_fit(len)
    }

    fn allocate_from_shard(&self, shard_idx: usize, len: u64) -> Option<FreeExtent> {
        let mut extent = self.shards[shard_idx].pop_at_least(len)?;
        let allocated = extent.split_head(len);
        if extent.len > 0 {
            self.return_remainder(shard_idx, extent);
        }
        Some(allocated)
    }

    fn refill_shard(&self, shard_idx: usize, len: u64) {
        if hot_bucket_for_len(len).is_none() {
            return;
        }

        let max_hot_len = *HOT_BUCKET_LIMITS
            .last()
            .expect("hot buckets must be non-empty");
        let refill_len = len.saturating_mul(HOT_REFILL_BATCH as u64).min(max_hot_len);
        if let Some(extent) = self.central.allocate_best_fit(refill_len) {
            self.return_remainder(shard_idx, extent);
            return;
        }

        let Some(extent) = self.central.allocate_best_fit(len) else {
            return;
        };
        if let Err(extent) = self.shards[shard_idx].push(extent) {
            self.central.insert(extent);
        }
    }

    fn return_remainder(&self, shard_idx: usize, extent: FreeExtent) {
        if let Err(extent) = self.shards[shard_idx].push(extent) {
            self.central.insert(extent);
        }
    }

    fn allocate_monotonic_extent(&self, len: u64, shard_hint: usize) -> FreeExtent {
        let shard_idx = shard_hint % self.shards.len();
        let shard = &self.shards[shard_idx];
        if let Some(extent) = shard.monotonic.try_allocate(len) {
            return extent;
        }

        let _refill = shard.monotonic.refill_lock.lock();
        if let Some(extent) = shard.monotonic.try_allocate(len) {
            return extent;
        }

        let reserve_len = MONOTONIC_REFILL_BASE_PAGES.max(len);
        let start_page_number = self
            .global_next_page_number
            .fetch_add(reserve_len, Ordering::Relaxed);
        let allocated = FreeExtent::new(start_page_number, len);
        let next_page_number = start_page_number
            .checked_add(len)
            .expect("page number overflow");
        let end_page_number = start_page_number
            .checked_add(reserve_len)
            .expect("page number overflow");
        shard
            .monotonic
            .install_reserved_tail(next_page_number, end_page_number);
        allocated
    }
}

fn hot_bucket_for_len(len: u64) -> Option<usize> {
    HOT_BUCKET_LIMITS.iter().position(|&limit| len <= limit)
}

#[cfg(test)]
mod tests {
    #[cfg(not(miri))]
    use std::collections::BTreeSet;
    #[cfg(not(miri))]
    use std::sync::Arc;

    use crate::buffer_frame::physical_page_number;

    use super::*;

    #[test]
    fn allocate_falls_back_to_monotonic_page_numbers() {
        let allocator = FreePageAllocator::new(1, 4);

        let first = allocator.allocate_page(0);
        let second = allocator.allocate_page(0);

        assert_eq!(first, 1);
        assert_eq!(second, 2);
        assert_eq!(
            allocator.next_page_number(),
            1 + MONOTONIC_REFILL_BASE_PAGES
        );
    }

    #[test]
    fn monotonic_allocations_reuse_shard_reserved_range() {
        let allocator = FreePageAllocator::new(1, 4);

        let first = allocator.allocate_page(0);
        let reserved_high_water = allocator.next_page_number();
        let second = allocator.allocate_page(0);
        let third = allocator.allocate_page(0);

        assert_eq!(physical_page_number(first), 1);
        assert_eq!(physical_page_number(second), 2);
        assert_eq!(physical_page_number(third), 3);
        assert_eq!(
            allocator.next_page_number(),
            reserved_high_water,
            "same-shard monotonic allocations should not touch the global cursor until refill"
        );
    }

    #[test]
    fn monotonic_shards_reserve_disjoint_ranges() {
        let allocator = FreePageAllocator::new(1, 4);

        let first = allocator.allocate_page(0);
        let second = allocator.allocate_page(1);

        assert_eq!(physical_page_number(first), 1);
        assert_eq!(
            physical_page_number(second),
            1 + MONOTONIC_REFILL_BASE_PAGES,
            "each shard should reserve an independent monotonic range"
        );
        assert_eq!(
            allocator.next_page_number(),
            1 + MONOTONIC_REFILL_BASE_PAGES * 2
        );
    }

    #[test]
    fn reusable_extent_is_used_before_monotonic_growth() {
        let allocator = FreePageAllocator::new(100, 4);
        allocator.promote_reusable_extent(FreeExtent::new(10, 1));

        let pid = allocator.allocate_page(0);

        assert_eq!(pid, 10);
        assert_eq!(allocator.next_page_number(), 100);
    }

    #[test]
    fn larger_extent_splits_for_smaller_allocations() {
        let allocator = FreePageAllocator::new(100, 4);
        allocator.promote_reusable_extent(FreeExtent::new(10, 4));

        let first = allocator.allocate_page(0);
        let second = allocator.allocate_page(0);

        assert_eq!(physical_page_number(first), 10);
        assert_eq!(physical_page_number(second), 11);
        assert_eq!(allocator.next_page_number(), 100);
    }

    #[test]
    fn central_allocator_coalesces_adjacent_extents() {
        let allocator = FreePageAllocator::new(100, 4);
        allocator.promote_reusable_extent(FreeExtent::new(10, 8));
        allocator.promote_reusable_extent(FreeExtent::new(18, 8));

        assert_eq!(allocator.central_extent_count(), 1);
        assert_eq!(allocator.central_free_base_pages(), 16);

        // With a single page class each allocation consumes one base page,
        // so the coalesced 16-page extent satisfies 16 sequential allocations.
        let first = allocator.allocate_page(0);
        assert_eq!(first, 10);
        assert_eq!(allocator.next_page_number(), 100);
    }

    #[test]
    fn retired_large_extent_can_be_reused_as_single_pages() {
        let allocator = FreePageAllocator::new(100, 4);
        allocator.promote_reusable_extent(FreeExtent::new(32, 4));

        let pages = (0..4)
            .map(|_| allocator.allocate_page(0))
            .map(physical_page_number)
            .collect::<Vec<_>>();

        assert_eq!(pages, vec![32, 33, 34, 35]);
        assert_eq!(allocator.next_page_number(), 100);
    }

    #[test]
    fn hot_reuse_refills_shard_with_batch_extent() {
        let allocator = FreePageAllocator::new(100, 4);
        allocator.promote_reusable_extent(FreeExtent::new(10, 16));

        let first = allocator.allocate_page(0);

        assert_eq!(physical_page_number(first), 10);
        assert_eq!(
            allocator.central_free_base_pages(),
            8,
            "first allocation should pull an 8-page refill batch from central"
        );

        for expected in 11..18 {
            let pid = allocator.allocate_page(0);
            assert_eq!(physical_page_number(pid), expected);
            assert_eq!(
                allocator.central_free_base_pages(),
                8,
                "cached shard refill should satisfy remaining batch allocations"
            );
        }

        let next = allocator.allocate_page(0);
        assert_eq!(physical_page_number(next), 18);
        assert_eq!(allocator.central_free_base_pages(), 0);
    }

    #[test]
    #[cfg(not(miri))]
    fn concurrent_allocation_from_reusable_extents_returns_unique_pages() {
        let allocator = Arc::new(FreePageAllocator::new(10_000, 8));
        allocator.promote_reusable_extent(FreeExtent::new(1_000, 512));

        let mut handles = Vec::new();
        for shard in 0..8 {
            let allocator = Arc::clone(&allocator);
            handles.push(std::thread::spawn(move || {
                (0..32)
                    .map(|_| allocator.allocate_page(shard))
                    .map(physical_page_number)
                    .collect::<Vec<_>>()
            }));
        }

        let mut pages = BTreeSet::new();
        for handle in handles {
            for page in handle.join().unwrap() {
                assert!(pages.insert(page), "page {page} was allocated twice");
            }
        }
        assert_eq!(pages.len(), 256);
        assert_eq!(allocator.next_page_number(), 10_000);
    }

    #[test]
    #[cfg(not(miri))]
    fn concurrent_monotonic_allocation_returns_unique_pages() {
        let allocator = Arc::new(FreePageAllocator::new(1, 8));

        let mut handles = Vec::new();
        for shard in 0..8 {
            let allocator = Arc::clone(&allocator);
            handles.push(std::thread::spawn(move || {
                (0..64)
                    .map(|_| allocator.allocate_page(shard))
                    .map(physical_page_number)
                    .collect::<Vec<_>>()
            }));
        }

        let mut pages = BTreeSet::new();
        for handle in handles {
            for page in handle.join().unwrap() {
                assert!(pages.insert(page), "page {page} was allocated twice");
            }
        }
        assert_eq!(pages.len(), 512);
        assert_eq!(
            allocator.next_page_number(),
            1 + MONOTONIC_REFILL_BASE_PAGES * 8
        );
    }

    #[test]
    fn shard_second_refill_advances_next_page_number() {
        let allocator = FreePageAllocator::new(1, 1);

        // Exhaust the first reserved range (256 pages).
        for _ in 0..MONOTONIC_REFILL_BASE_PAGES {
            allocator.allocate_page(0);
        }
        let first_next = allocator.next_page_number();

        // Next allocation should trigger a second refill from central.
        allocator.allocate_page(0);
        let second_next = allocator.next_page_number();

        assert!(
            second_next > first_next,
            "second refill should advance next_page_number: {first_next} → {second_next}"
        );
    }

    #[test]
    #[should_panic(expected = "overlaps existing extent")]
    fn promote_reusable_extent_twice_panics_on_overlap() {
        // The allocator rejects overlapping extents — promoting the same
        // extent twice is a contract violation, not an idempotent operation.
        let allocator = FreePageAllocator::new(1, 1);
        let extent = FreeExtent::new(50, 4);
        allocator.promote_reusable_extent(extent);
        allocator.promote_reusable_extent(extent);
    }
}
