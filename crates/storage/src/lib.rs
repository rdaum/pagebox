//! Pagebox's storage substrate: buffer pool, page store, slotted pages, free-page
//! allocator, and buffer frames.
//!
//! `pagebox-storage` is the layer underneath persistent page-based structures. It owns page
//! *residency* (which pages are memory-resident and how they got there), page
//! *format* (the byte layout shared by every buffer-managed page), and page
//! *allocation* (both on disk via the page store and in-memory via the
//! sharded free-page allocator). Higher-level machinery such as the B+tree
//! composes these primitives into indexes and other structures. This
//! crate intentionally contains no SQL, no MVCC, and no row format of its
//! own; the slotted-page container is a generic sorted key/value byte layout.
//!
//! ## Architecture
//!
//! The design follows the LeanStore / Umbra hybrid in-memory/disk model: the
//! hot working set is memory-resident and accessed through swizzled pointers,
//! but every page is also disk-backed so the dataset can exceed RAM. The pieces
//! are:
//!
//! - [`buffer_frame`]: the per-page in-memory slot — a 4096-aligned frame
//!   combining a [`buffer_frame::BufferFrame`]'s latch (a
//!   `pagebox_hybrid_latch::HybridLatch`), a frame header, and the page bytes
//!   themselves. Page bytes use the workspace's unified compile-time
//!   `pagebox_frame_kernel::PAGE_SIZE`; the frame is the unit of pinning,
//!   latching, and eviction.
//! - [`buffer_pool`]: the [`buffer_pool::BufferPool`] owns an array of frames.
//!   It performs fix/evict (resident-budget second-chance or batch-clock), pin
//!   accounting, prefetch, dirty-page tracking, the recovery-aware dt registry,
//!   and integration with the WAL. Every public method takes `&self`;
//!   concurrency is handled with atomics and the per-frame latch.
//! - [`page_store`]: the disk side. The [`page_store::PageStore`] trait
//!   abstracts read / write / allocate / sync; concrete implementations are
//!   [`page_store::InMemoryPageStore`] (tests) and
//!   [`page_store::FilePageStore`] (single file, positioned `pread`/`pwrite`,
//!   optional `O_DIRECT`). The header page (page 0) carries the magic, page
//!   count, checkpoint LSN, and two user-meta slots used by reopened trees.
//! - [`free_page_allocator`]: a sharded allocator with a central best-fit
//!   freelist plus per-shard monotonic reservation. Reusable (promoted)
//!   extents are consumed before monotonic growth; retired pages return to the
//!   freelist. Page IDs are never reused across reopen until the WAL checkpoint
//!   advances past them.
//! - [`slotted_page`]: the byte-level page format. Slots grow forward from the
//!   header, key/value heap grows backward from the end of the page, and
//!   [`slotted_page::SlottedPage::compactify`] reclaims the gap. Reserved
//!   suffixes (B-tree right-sibling / upper-link bytes) survive compaction. The
//!   page panics on overflow by contract.
//! - [`page_header`]: the small common prefix every page shares — page LSN,
//!   page type discriminator, and the leaf flag. Used by recovery and
//!   residency classification.
//! - [`page_provider`]: optional background thread that proactively evicts
//!   unpinned pages to keep the resident budget from going empty. Enabled by
//!   `PAGEBOX_ENABLE_BACKGROUND_PAGE_PROVIDER=1`.
//!
//! ## Key invariants
//!
//! The following are preserved across changes; violation of any of them is a
//! bug. Tests for the subtle ones live next to the implementation.
//!
//! - **SWIP state machine**: page references live in a 64-bit word encoding
//!   hot/cool/evicted states with a page ID. Atomic CAS drives transitions;
//!   failed CAS returns the live word, which is the contract optimistic
//!   restart in the B+tree relies on. See `pagebox-swip-kernel`.
//! - **Hybrid-latch optimistic restart**: in-flight optimistic readers may
//!   observe a stale page; `validate` reports this as
//!   `pagebox_hybrid_latch::Restart` and the caller must loop. Upgrades from
//!   optimistic to shared/exclusive restart if the version moved. See
//!   `pagebox-hybrid-latch`.
//! - **Resident budget**: a fixed count of base pages is allowed to be
//!   resident at once. Eviction reclaims via referenced-bit second-chance
//!   (`RandomSecondChance`, default) or batch clock (`BatchClock`,
//!   `PAGEBOX_EVICTION_POLICY=batch_clock`). Dirty pages are no-steal:
//!   `try_evict_*` refuses them until [`buffer_pool::BufferPool::flush`] (or
//!   `flush_dirty_batch`) cleans them. Stable parent-link pages (e.g. the
//!   B-tree root wired through [`buffer_frame::ParentLink::Stable`]) must not
//!   be evicted.
//! - **Pin count**: incremented on `fix` and decremented on `unfix`/`Drop`.
//!   When every frame is pinned and a new fix is requested, the pool panics
//!   (`buffer pool exhausted`).
//! - **Slotted-page layout**: slot array grows from the front, heap grows
//!   from the back, [`slotted_page::SlottedPage::compactify`] reclaims the
//!   gap, reserved suffixes survive compaction, overflow is a panic by
//!   contract. The slot `head` field caches the first four bytes of the key
//!   (big-endian, zero-padded) to short-circuit binary search comparisons.
//! - **WAL durability**: dirty pages are logged before being written to the
//!   data file (WAL protocol). When the `metrics` feature is on, dirty page
//!   images and re-logs are counted per page kind.
//!
//! ## Telemetry
//!
//! The `metrics` feature (default) wires `fast-telemetry` counters, gauges,
//! and histograms into the buffer pool, slotted page, and page store. Disable
//! it with `--no-default-features` to select this crate's no-op shim types.
//! Internal dependencies currently retain their own default features, so that
//! flag alone does not guarantee a telemetry-free dependency graph. The
//! `cfg(miri)` build also deploys the local shims so Miri runs against the
//! unmodified hot path without telemetry allocations.
//!
//! ## Miri and loom
//!
//! The buffer pool and slotted page are Miri-clean (the storage tests are
//! tagged `#[cfg(not(miri))]` only where they touch real files or spin up
//! threads). The hybrid-latch and SWIP primitives that this crate builds on
//! have their own `cfg(loom)` models; the buffer pool itself is too large to
//! model-check as a whole, so its concurrency invariants are anchored to
//! those underlying loom tests plus differential stress tests with real
//! invariant assertions (post-hoc scans, uniqueness checks, counts).

pub mod buffer_frame;
pub mod buffer_pool;
pub mod free_page_allocator;
#[cfg(not(feature = "metrics"))]
mod metrics_stub;
pub mod page_header;
pub mod page_provider;
pub mod page_store;
pub mod slotted_page;
