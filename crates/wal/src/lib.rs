//! Append-only write-ahead log with group commit, configurable sync backends,
//! streaming replay, and crash recovery.
//!
//! `pagebox-wal` owns the log records and I/O path used to make storage and
//! tree mutations durable. It is the substrate beneath the buffer pool's
//! no-steal dirty-page writeback: every dirty page is logged here before
//! being written to the data file (WAL protocol).
//!
//! ## Architecture
//!
//! A [`Wal`] instance is one or more *shards*, each running a writer thread
//! and (for `fdatasync` backends) a syncer thread. Appends claim an LSN from a
//! shared atomic, route to the owning shard by LSN, and copy record bytes
//! into a 64 MiB `AlignedBuf` write buffer packed as `[batch meta page] +
//! up to `BATCH_MAX_RECORDS` data pages`. The writer thread drains full
//! (or deadline-elapsed) buffers to disk with `pwrite`; the syncer thread
//! then calls the configured sync backend. Stragglers that miss the current
//! batch wait on the shard's `flush_done` condvar.
//!
//! ### Record layout
//!
//! Each shard file starts with a header page (`WAL_MAGIC`, version, flags,
//! record size) followed by an append-only sequence of *batches*. A batch is
//! one *batch-meta page* followed by up to `BATCH_MAX_RECORDS` *data pages*:
//!
//! ```text
//!   ┌─────────┬───────────────┬───────────────┬─────┬───────────────┐
//!   │ header  │ batch meta[0] │ data page[0]  │ ... │ batch meta[1] │ ...
//!   │PAGE_SIZE│  PAGE_SIZE    │  PAGE_SIZE    │     │
//!   └─────────┴───────────────┴───────────────┴─────┴───────────────┘
//! ```
//!
//! The batch-meta page holds a fixed-size array of `BATCH_ENTRY_SIZE`-byte
//! `BatchEntry` records (LSN, arg, CRC, len, kind, flags); each data page
//! holds one page-image payload or one (possibly multi-chunk) logical-record
//! payload.
//! CRC32 (ISO-HDLC) covers the batch-meta page (excluding the CRC field
//! itself); a missing or wrong-CRC batch terminates replay at the previous
//! batch (so a torn tail is silently truncated at open time).
//!
//! ### Commit modes
//!
//! [`CommitMode::Strict`] (default) — [`Wal::commit`] / [`Wal::flush`] block
//! the caller until the requested LSN is durable. [`CommitMode::Relaxed`] —
//! the same calls return the requested LSN without waiting; a background
//! syncer advances `durable_lsn` at a configurable cadence. Strict and
//! relaxed share one pipeline; mode is set per-process via
//! [`Wal::set_commit_mode`].
//!
//! ### Sync backends
//!
//! Selected by `PAGEBOX_WAL_SYNC_BACKEND`:
//!
//! - `fdatasync` (default) — `pwrite` to the file, then `fdatasync` in a
//!   dedicated syncer thread. Compatible with every Unix.
//! - `pwritev2_dsync` — Linux `pwritev2` with `RWF_DSYNC`; one syscall does
//!   the durable write, so no separate syncer thread is spawned. Falls back
//!   to `fdatasync` on non-Linux.
//! - `io_uring` — Linux io_uring (kernel ≥5.18 with `IORING_SETUP_NO_MMAP`);
//!   writes submitted as `IORING_OP_WRITE` SQEs with a linked
//!   `IORING_OP_FSYNC` SQE so durability arrives as a CQE. One ring-driver
//!   thread per shard drains completions to advance `written_lsn` /
//!   `durable_lsn` and reclaim buffers. v1 uses plain kernel polling (no
//!   `SQPOLL`, no registered buffers/files). Falls back to `fdatasync` on
//!   non-Linux.
//!
//! ### Group commit
//!
//! Both backends apply a leader-follower group-commit policy: the first
//! follower to arrive after a quiet period becomes the leader, waits a short
//! delay (`PAGEBOX_WAL_GROUP_COMMIT_DELAY_MAX_US`, default 1 ms for
//! `fdatasync`, 250 µs for `pwritev2_dsync`), accumulates arriving followers,
//! then releases them. The leader cuts the delay short once
//! `PAGEBOX_WAL_GROUP_COMMIT_TARGET_RECORDS` followers pile on. Relaxed mode
//! also drives background writes and syncs on time / record-count
//! thresholds (`PAGEBOX_WAL_RELAXED_*`).
//!
//! ### Direct I/O
//!
//! `PAGEBOX_WAL_DIRECT_IO=1` opts into Linux `O_DIRECT` (the file header
//! tracks whether direct I/O was in use). All WAL buffers are allocated
//! `DIRECT_IO_ALIGN`-aligned via an internal `AlignedBuf` so the same append
//! path satisfies both buffered and direct I/O.
//!
//! ### Page-image overwrites
//!
//! When a page is dirtied multiple times before its WAL buffer drains, the
//! later image can overwrite the earlier buffered slot in place (via
//! [`Wal::append_or_overwrite_page_image`] /
//! [`Wal::overwrite_buffered_page_image_with_lsn`]). The overwrite path
//! validates shard, buffer epoch, slot index, and page id before touching
//! the slot, so an overwrite never corrupts a different record. Once a
//! buffer has been handed to the writer thread, overwrites are impossible —
//! the buffer's epoch has moved.
//!
//! ## Recovery
//!
//! [`Wal::recover`] drives replay into a [`RecoveryPageStore`]:
//!
//! 1. Records with `lsn <= checkpoint_lsn` are skipped.
//! 2. Page-image records are written to the store if `lsn >
//!    read_page_lsn(page_bytes)`; otherwise skipped (idempotent recovery).
//! 3. Logical records are surfaced to a caller callback that applies the
//!    semantic-level change (page-patch records are decoded and applied here).
//!
//! Multi-shard WALs replay in LSN-merged order (records are collected, sorted,
//! then applied); single-shard WALs stream directly. [`Wal::replay`] /
//! [`Wal::replay_records`] are the read-only inspection variants; both flush
//! pending appends first. [`Wal::reset`] truncates the WAL back to its
//! header — used after a checkpoint makes all logged records obsolete.
//!
//! ## Telemetry
//!
//! `metrics` (default): labelled counters for events (flush fast-path /
//! wait, write, sync, durable advance, page-image overwrite attempts /
//! successes, logical records / bytes) and latency histograms for flush wait,
//! write, and sync — all keyed by internal `WalEvent` / `WalLatency` enums
//! (which `Wal::visit_metrics` exposes outside the crate). With the feature
//! off, the shims take their place so the hot path stays dependency-free and
//! zero-allocation.
//!
//! ## Sync-failure policy
//!
//! A failed `fsync` / `fdatasync` / `pwritev2` is treated as a durability
//! contract violation and panics. There is no fall-back to "PK-buffer and
//! retry" — the caller is informed by process death so the operator can
//! restore from a backup rather than silently continuing with possibly-torn
//! state.
//!
//! ## Miri
//!
//! The Wal test suite is tagged `#[cfg(not(miri))]` because it touches real
//! files and threads; the underlying primitives this crate builds on
//! (`pagebox-frame-kernel`, `pagebox-threading`) are Miri-clean.

mod aligned_buf;
mod backend;
mod format;
mod io;
#[cfg(target_os = "linux")]
mod io_uring;
#[cfg(not(feature = "metrics"))]
mod metrics_stub;
mod wal_impl;

pub use format::WAL_BUF_RECORDS;
pub use wal_impl::{
    BufferedWalRecord, CommitMode, RecoveryPageStore, RecoveryReport, Wal, WalReplayRecord,
    WalStats,
};

#[cfg(test)]
#[cfg(not(miri))]
mod tests;
