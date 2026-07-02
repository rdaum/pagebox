//! [`Wal`] — the append / write / sync pipeline, group-commit coordination,
//! and recovery entry point.
//!
//! TheWal is a thin outer shell over a primary [`WalInner`] plus zero or
//! more extra shards. Each shard owns:
//!
//! - A 64 MiB `AlignedBuf`-backed `WalBuffer` that callers copy record bytes
//!   into under a `parking_lot::Mutex<WalState>`.
//! - A *writer* background thread (always spawned) that drains full or
//!   deadline-elapsed buffers via `pwrite`.
//! - A *syncer* background thread (spawned only for the `fdatasync` backend)
//!   that advances `durable_lsn` once per drained buffer. The
//!   `pwritev2_dsync` backend does the durable write inline in the writer
//!   thread, so no syncer is needed.
//!
//! ## LSN routing
//!
//! LSNS are claimed from a shared atomic (`next_lsn`); the shard is selected
//! by `(lsn - 1) / shard_width` where `shard_width` is computed at open time
//! so the LSN space is split evenly across shards. Per-thread shard
//! affinity is also tracked (via `WAL_THREAD_STATES`) so=
//! [`Wal::commit_current_thread`] can resolve the right shard from the
//! calling thread's local state.
//!
//! ## Append path
//!
//! `append_*_with_lsn` takes a pre-claimed LSN and a record payload,
//! reserves a slot under the shard's state lock, copies the payload into
//! the active buffer, and updates the batch-meta page so the writer can
//! later checksum it. The lock is held only across the slot reservation and
//! the byte copy, not across the durable write. Page-image appends use
//! `append_or_overwrite_page_image` / `try_overwrite_page_image_with_lsn`
//! to coalesce repeated mutations of the same page into a single buffered
//! slot before the writer drains it.
//!
//! ## Flush and commit
//!
//! Three durability entry points share one pipeline:
//!
//! - [`Wal::commit`] — commit every shard's appends so far; under
//!   [`CommitMode::Strict`] this blocks until durable, under
//!   [`CommitMode::Relaxed`] it returns the requested LSN without waiting.
//! - [`Wal::flush`] — like `commit` but always strict (used at checkpoint).
//! - [`Wal::flush_at_least`] — strictly flush only the shard owning
//!   `target_lsn`, blocking until that LSN is durable. Relaxed-mode waits
//!   inside the same function return the target LSN immediately.
//!
//! Group-commit leadership rotates: the first caller to land on an empty
//! shard after a quiet period becomes the leader, accumulates followers for
//! up to `group_commit_delay_max_us`, then releases them all together.
//! Late-arriving callers wait on the shard's `flush_done` condvar until the
//! in-flight write completes.
//!
//! ## Recovery
//!
//! [`Wal::recover`] flushes any buffered appends, then either streams
//! records from the single shard (`recover`) or collects, sorts by LSN, and
//! replays in merged order (`recover_merged`, for multi-shard WALs). Both
//! delegate the actual page-write decision into [`recover_page_image`] /
//! [`recover_logical_payload`], which respect the idempotent-recovery rule:
//! a page-image record at LSN `L` is replayed into the store iff
//! `read_page_lsn(page_bytes) < L` and `L > checkpoint_lsn`.
//!
//! ## Crashing and shutdown
//!
//! [`Wal::crash`] is the deliberate-unsafe shutdown: stops threads without
//! flushing pending buffers and closes the fd, simulating a process crash
//! mid-append. The recovery path exercises this on reopen.
//! [`Drop`] for [`Wal`] does the opposite — `stop_shard(_, _, _, false)`
//! drains all pending appends, fsyncs the fd, then closes — so dropping a
//! [`Wal`] in a benchmark is the same as a clean process exit.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[cfg(feature = "metrics")]
use fast_telemetry::{DeriveLabel, ExportMetrics, LabeledCounter, LabeledHistogram, MetricVisitor};
use pagebox_frame_kernel::{Lsn, PAGE_SIZE, PageId, page_end_base_page, page_size};
use pagebox_threading as threading;

#[cfg(not(feature = "metrics"))]
use crate::metrics_stub::{LabeledCounter, LabeledHistogram, MetricVisitor};
use parking_lot::{Condvar, Mutex};

use crate::aligned_buf::AlignedBuf;
use crate::format::SEGMENT_SIZE;
use crate::format::{
    BATCH_MAX_RECORDS, BatchEntry, LOGICAL_CHUNK_MAX_LEN, LOGICAL_FLAG_FIRST, LOGICAL_FLAG_LAST,
    PACKED_LOGICAL_ENTRY_HEADER_LEN, PACKED_LOGICAL_MAX_PAYLOAD_LEN, RECORD_KIND_LOGICAL,
    RECORD_KIND_LOGICAL_PACKED, RECORD_KIND_PAGE_IMAGE, WAL_BUF_CAPACITY,
    WAL_FDATASYNC_GROUP_COMMIT_DELAY_MAX_US, WAL_FDATASYNC_GROUP_COMMIT_TARGET_RECORDS,
    WAL_GROUP_COMMIT_DELAY_MIN_US, WAL_HEADER_SIZE, WAL_PWRITEV2_DSYNC_GROUP_COMMIT_DELAY_MAX_US,
    WAL_PWRITEV2_DSYNC_GROUP_COMMIT_TARGET_RECORDS, WAL_RECORD_SIZE, WAL_RELAXED_SYNC_INTERVAL_US,
    WAL_RELAXED_SYNC_RECORDS, WAL_RELAXED_WRITE_INTERVAL_US, WAL_RELAXED_WRITE_RECORDS,
    batch_meta_count, batch_meta_count_unchecked, build_wal_header, env_u64_us,
    finalize_batch_meta, init_batch_meta, overwrite_batch_entry_crc, overwrite_batch_entry_lsn,
    page_crc, payload_crc, read_batch_entry, set_batch_meta_count, validate_wal_header,
    write_batch_entry,
};
use crate::io::{
    extend_file, fdatasync_file, fstat_size, pread_all, pwrite_all, pwritev2_dsync_all,
    round_up_u64, sync_wal_fd,
};

/// Statistics from WAL recovery.
#[derive(Debug, Default, Clone)]
pub struct RecoveryReport {
    /// Total valid WAL records scanned.
    pub records_scanned: u64,
    /// Records applied to the page store (page was older than WAL).
    pub records_applied: u64,
    /// Records skipped because LSN <= checkpoint_lsn.
    pub skipped_checkpoint: u64,
    /// Records skipped because on-disk page_lsn >= WAL record LSN.
    pub skipped_page_lsn: u64,
    /// Highest LSN seen in any WAL record.
    pub max_lsn: Lsn,
}

/// Labelled event counter key for [`WalStats`]. Each variant names a
/// discrete WAL pipeline step; the cumulative count is exposed through
/// [`Wal::visit_metrics`].
#[cfg_attr(feature = "metrics", derive(DeriveLabel))]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "metrics", label_name = "event")]
pub enum WalEvent {
    /// `flush_at_least` entered the durable-wait slow path.
    FlushCall,
    /// Fast-path no-wait return (target LSN already durable).
    FlushFastPath,
    /// Slow-path durable-wait condvar wake-up.
    FlushWait,
    /// Background writer completed one `pwrite` syscall.
    WriteCall,
    /// Background writer bytes written (sum across `WriteCall`s).
    WriteBytes,
    /// Background syncer invoked a sync syscall.
    SyncCall,
    /// `durable_lsn` was advanced by the syncer.
    DurableAdvance,
    /// Page-image records appended.
    PageImageRecords,
    /// Page-image bytes appended.
    PageImageBytes,
    /// Attempts to overwrite a buffered page-image in place.
    PageImageOverwriteAttempts,
    /// Successful buffered page-image overwrites (saved one full record slot).
    PageImageOverwriteSuccesses,
    /// Logical records appended.
    LogicalRecords,
    /// Logical-record bytes appended.
    LogicalBytes,
}

/// Labelled latency histogram key for [`WalStats`].
#[cfg_attr(feature = "metrics", derive(DeriveLabel))]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "metrics", label_name = "latency")]
pub enum WalLatency {
    /// Time spent in [`Wal::flush`] / [`Wal::flush_at_least`] waiting on the
    /// syncer, under strict mode.
    FlushWait,
    /// Per-`pwrite` latency from the writer thread.
    Write,
    /// Per-`fdatasync` latency from the syncer thread.
    Sync,
}

/// WAL telemetry: per-event labelled counters and per-latency labelled
/// histograms. Empty struct; access is via the methods on [`Wal`]
/// (`visit_metrics`) or the [`WalStats`] re-exported for embedded callers.
#[cfg_attr(feature = "metrics", derive(ExportMetrics))]
#[cfg_attr(feature = "metrics", metric_prefix = "wal")]
pub struct WalStats {
    #[cfg_attr(feature = "metrics", help = "WAL events")]
    pub(crate) events: LabeledCounter<WalEvent>,
    #[cfg_attr(feature = "metrics", help = "WAL latency in nanoseconds")]
    pub(crate) latencies: LabeledHistogram<WalLatency>,
}

/// One record surfaced by [`Wal::replay`] / [`Wal::replay_records`].
///
/// `PageImage` carries the full page bytes for `page_id` at `lsn`;
/// `Logical` carries a (re-assembled) logical record with caller-defined
/// `kind` and payload. The lifetime is tied to the WAL's internal replay
/// buffer; multi-shard merged-replay copies records into an owned staging
/// vector that lives for the lifetime of the closure invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalReplayRecord<'a> {
    PageImage {
        lsn: Lsn,
        page_id: PageId,
        data: &'a [u8],
    },
    Logical {
        lsn: Lsn,
        kind: u64,
        payload: &'a [u8],
    },
}

enum OwnedWalReplayRecord {
    PageImage {
        lsn: Lsn,
        page_id: PageId,
        data: Vec<u8>,
    },
    Logical {
        lsn: Lsn,
        kind: u64,
        payload: Vec<u8>,
    },
}

impl OwnedWalReplayRecord {
    fn lsn(&self) -> Lsn {
        match self {
            Self::PageImage { lsn, .. } | Self::Logical { lsn, .. } => *lsn,
        }
    }
}

impl Default for WalStats {
    fn default() -> Self {
        let shards = std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1)
            .max(1);
        Self {
            events: LabeledCounter::new(shards),
            latencies: LabeledHistogram::new(&wal_latency_bounds_ns(), shards),
        }
    }
}

#[cfg(not(feature = "metrics"))]
impl WalStats {
    pub fn visit_metrics<V: MetricVisitor + ?Sized>(&self, _visitor: &mut V) {}
}

fn wal_latency_bounds_ns() -> [u64; 13] {
    [
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
        1_000_000_000,
    ]
}

/// Handle to a record buffered in a shard's write buffer.
///
/// Returned by every `append_*` variant. Carries enough state — buffer
/// `epoch`, byte `offset` encoded as `(meta_start, slot_idx)`, and
/// owning `shard_idx` — for the WAL to locate the record for in-place
/// overwrite via `overwrite_buffered_page_image_with_lsn` /
/// `append_or_overwrite_page_image`. Once the writer drains the buffer,
/// the buffer's epoch advances and the handle becomes un-overwritable
/// (calls return `false` / fall back to a fresh append).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferedWalRecord {
    pub epoch: u64,
    pub offset: u32,
    pub shard_idx: u16,
}

/// Per-process commit policy for the WAL.
///
/// - [`CommitMode::Strict`] (default): [`Wal::commit`] / [`Wal::flush`] / \
///   [`Wal::flush_at_least`] block until the requested LSN is durable.
/// - [`CommitMode::Relaxed`]: the same calls return the target LSN immediately
///   without waiting for the syncer; useful for benchmarks / bulk loads where
///   the caller tolerates a small loss of recent commits on crash.
///
/// Strict and relaxed share one pipeline — mode just changes whether the
/// caller blocks. Set process-wide via [`Wal::set_commit_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CommitMode {
    Strict = 0,
    Relaxed = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WalSyncBackend {
    Fdatasync,
    Pwritev2Dsync,
}

impl WalSyncBackend {
    fn group_commit_delay_max_us(self) -> u64 {
        match self {
            Self::Fdatasync => env_u64_us(
                "PAGEBOX_WAL_FDATASYNC_DELAY_MAX_US",
                env_u64_us(
                    "PAGEBOX_WAL_GROUP_COMMIT_DELAY_MAX_US",
                    WAL_FDATASYNC_GROUP_COMMIT_DELAY_MAX_US,
                ),
            ),
            Self::Pwritev2Dsync => env_u64_us(
                "PAGEBOX_WAL_PWRITEV2_DSYNC_DELAY_MAX_US",
                WAL_PWRITEV2_DSYNC_GROUP_COMMIT_DELAY_MAX_US,
            ),
        }
    }

    fn group_commit_target_records(self) -> u64 {
        match self {
            Self::Fdatasync => env_u64_us(
                "PAGEBOX_WAL_FDATASYNC_TARGET_RECORDS",
                env_u64_us(
                    "PAGEBOX_WAL_GROUP_COMMIT_TARGET_RECORDS",
                    WAL_FDATASYNC_GROUP_COMMIT_TARGET_RECORDS,
                ),
            ),
            Self::Pwritev2Dsync => env_u64_us(
                "PAGEBOX_WAL_PWRITEV2_DSYNC_TARGET_RECORDS",
                WAL_PWRITEV2_DSYNC_GROUP_COMMIT_TARGET_RECORDS,
            ),
        }
    }
}

/// Page-store trait the WAL recovery path writes into.
///
/// Proxy for `pagebox_storage::PageStore` so this crate can drive recovery
/// without depending on `pagebox-storage`. Both
/// `pagebox_storage::InMemoryPageStore` and `pagebox_storage::FilePageStore`
/// implement this; embedders may supply their own implementation.
pub trait RecoveryPageStore: Send + Sync {
    fn read_page(&self, pid: PageId, buf: &mut [u8]) -> io::Result<bool>;
    fn write_page(&self, pid: PageId, data: &[u8]) -> io::Result<()>;
    fn allocate(&self, pid: PageId) -> io::Result<()>;
    fn sync(&self) -> io::Result<()>;
    fn next_page_id(&self) -> PageId;
}

const LOGICAL_KIND_PAGE_IMAGE_BYTES: u64 = 0x4258_5049_4D47_0001;
const LOGICAL_KIND_PAGE_PATCH: u64 = 0x4258_5041_5443_0001;
const PAGE_IMAGE_BYTES_HEADER_LEN: usize = 16;
const PAGE_PATCH_HEADER_LEN: usize = 16;
const PAGE_PATCH_RANGE_HEADER_LEN: usize = 8;

struct WalInner {
    state: Mutex<WalState>,
    sync_backend: WalSyncBackend,
    next_lsn: Arc<AtomicU64>,
    appended_lsn: AtomicU64,
    written_lsn: AtomicU64,
    durable_lsn: AtomicU64,
    requested_write_lsn: AtomicU64,
    requested_durable_lsn: AtomicU64,
    commit_mode: AtomicU64,
    flush_done: Condvar,
    flush_requested: Condvar,
    stats: WalStats,
}

/// Handle to a write-ahead log. Owns the primary shard plus zero or more
/// extra shards, each running writer / syncer background threads.
///
/// Construct with [`Wal::open`] / [`Wal::open_opts`] (the latter honours
/// `PAGEBOX_WAL_SYNC_BACKEND` and `PAGEBOX_WAL_SHARDS`). All public methods
/// take `&self` and route to the owning shard by LSN or thread-local state;
/// the type is `Send + Sync` via the inner `Arc<WalInner>`.
///
/// Drop performs clean shutdown: it drains all pending appends, fsyncs each
/// shard's fd, and joins the worker threads. Use [`Wal::crash`] for the
/// deliberate-unsafe variant (no drain, threads stopped in place) used to
/// exercise recovery on the next reopen.
pub struct Wal {
    inner: Arc<WalInner>,
    writer: Option<std::thread::JoinHandle<()>>,
    syncer: Option<std::thread::JoinHandle<()>>,
    extra_shards: Vec<WalShard>,
    next_thread_shard: AtomicUsize,
}

struct WalShard {
    inner: Arc<WalInner>,
    writer: Option<std::thread::JoinHandle<()>>,
    syncer: Option<std::thread::JoinHandle<()>>,
}

struct OpenWalFile {
    fd: std::os::fd::RawFd,
    direct_io: bool,
    file_offset: u64,
    allocated_size: u64,
    max_lsn: Lsn,
}

impl std::fmt::Debug for Wal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wal")
            .field("next_lsn", &self.inner.next_lsn.load(Ordering::Relaxed))
            .field(
                "appended_lsn",
                &self.inner.appended_lsn.load(Ordering::Relaxed),
            )
            .field(
                "written_lsn",
                &self.inner.written_lsn.load(Ordering::Relaxed),
            )
            .field(
                "durable_lsn",
                &self.inner.durable_lsn.load(Ordering::Relaxed),
            )
            .finish()
    }
}

struct WalState {
    fd: std::os::fd::RawFd,
    #[allow(dead_code)]
    direct_io: bool,
    active: WalBuffer,
    spare_buffers: Vec<WalBuffer>,
    pending_writes: VecDeque<PendingWalWrite>,
    writes_in_progress: usize,
    file_offset: u64,
    allocated_size: u64,
    flush_waiters: usize,
    next_buffer_epoch: u64,
    crash_shutdown: bool,
    shutdown: bool,
}

struct WalBuffer {
    buffer: AlignedBuf,
    used: usize,
    records: usize,
    epoch: u64,
    max_lsn: Lsn,
    open_batch_meta_offset: Option<usize>,
    open_batch_count: usize,
    packed_logical: Option<PackedLogicalSlot>,
}

#[derive(Clone, Copy)]
struct PackedLogicalSlot {
    meta_start: usize,
    slot_idx: usize,
    data_start: usize,
    len: usize,
    max_lsn: Lsn,
}

struct PendingWalWrite {
    buffer: WalBuffer,
    fd: std::os::fd::RawFd,
    file_offset: u64,
    len: usize,
    max_lsn: Lsn,
}

impl WalBuffer {
    fn new(epoch: u64) -> Self {
        Self {
            buffer: AlignedBuf::new(WAL_BUF_CAPACITY),
            used: 0,
            records: 0,
            epoch,
            max_lsn: 0,
            open_batch_meta_offset: None,
            open_batch_count: 0,
            packed_logical: None,
        }
    }

    fn reset(&mut self, epoch: u64) {
        self.used = 0;
        self.records = 0;
        self.epoch = epoch;
        self.max_lsn = 0;
        self.open_batch_meta_offset = None;
        self.open_batch_count = 0;
        self.packed_logical = None;
    }
}

fn encode_record_offset(meta_offset: usize, slot_idx: usize) -> u32 {
    debug_assert!(meta_offset.is_multiple_of(WAL_RECORD_SIZE));
    debug_assert!(slot_idx < BATCH_MAX_RECORDS);
    let meta_page = meta_offset / WAL_RECORD_SIZE;
    ((meta_page as u32) << 8) | slot_idx as u32
}

fn decode_record_offset(offset: u32) -> (usize, usize) {
    let slot_idx = (offset & 0xff) as usize;
    let meta_page = (offset >> 8) as usize;
    (meta_page * WAL_RECORD_SIZE, slot_idx)
}

fn chunk_count(payload_len: usize) -> usize {
    payload_len.div_ceil(LOGICAL_CHUNK_MAX_LEN).max(1)
}

fn validate_entry_data(entry: BatchEntry, data: &[u8]) -> Option<&[u8]> {
    match entry.kind {
        RECORD_KIND_PAGE_IMAGE => {
            if entry.flags != 0 || data.len() != PAGE_SIZE {
                return None;
            }
            let page: &[u8; PAGE_SIZE] = data.try_into().ok()?;
            (page_crc(page) == entry.crc).then_some(data)
        }
        RECORD_KIND_LOGICAL => {
            let len = entry.len as usize;
            if len > LOGICAL_CHUNK_MAX_LEN || len > data.len() {
                return None;
            }
            let allowed_flags = LOGICAL_FLAG_FIRST | LOGICAL_FLAG_LAST;
            if entry.flags & !allowed_flags != 0 {
                return None;
            }
            let payload = &data[..len];
            (payload_crc(payload) == entry.crc).then_some(payload)
        }
        RECORD_KIND_LOGICAL_PACKED => {
            let len = entry.len as usize;
            if entry.flags != 0 || len > data.len() {
                return None;
            }
            let payload = &data[..len];
            (payload_crc(payload) == entry.crc).then_some(payload)
        }
        _ => None,
    }
}

fn packed_logical_entry_len(payload_len: usize) -> Option<usize> {
    PACKED_LOGICAL_ENTRY_HEADER_LEN.checked_add(payload_len)
}

fn encode_packed_logical_entry(
    dst: &mut [u8],
    lsn: Lsn,
    kind: u64,
    payload: &[u8],
) -> Option<usize> {
    let len = u32::try_from(payload.len()).ok()?;
    let entry_len = packed_logical_entry_len(payload.len())?;
    let entry = dst.get_mut(..entry_len)?;
    entry[..8].copy_from_slice(&lsn.to_le_bytes());
    entry[8..16].copy_from_slice(&kind.to_le_bytes());
    entry[16..20].copy_from_slice(&len.to_le_bytes());
    entry[PACKED_LOGICAL_ENTRY_HEADER_LEN..].copy_from_slice(payload);
    Some(entry_len)
}

fn for_each_packed_logical_record<F>(payload: &[u8], mut f: F) -> bool
where
    F: FnMut(Lsn, u64, &[u8]),
{
    let mut offset = 0usize;
    while offset < payload.len() {
        let Some(header) = payload.get(offset..offset + PACKED_LOGICAL_ENTRY_HEADER_LEN) else {
            return false;
        };
        let lsn = u64::from_le_bytes(header[..8].try_into().expect("packed LSN header"));
        let kind = u64::from_le_bytes(header[8..16].try_into().expect("packed kind header"));
        let len = u32::from_le_bytes(header[16..20].try_into().expect("packed len header"));
        let payload_start = offset + PACKED_LOGICAL_ENTRY_HEADER_LEN;
        let payload_end = payload_start + len as usize;
        let Some(record_payload) = payload.get(payload_start..payload_end) else {
            return false;
        };
        f(lsn, kind, record_payload);
        offset = payload_end;
    }
    true
}

fn encode_page_image_bytes_payload(
    page_id: PageId,
    page_len: usize,
    fill_page: impl FnOnce(&mut [u8]),
) -> Vec<u8> {
    assert_eq!(
        page_len,
        page_size(page_id),
        "large WAL page image length must match page id class"
    );
    let mut payload = vec![0u8; PAGE_IMAGE_BYTES_HEADER_LEN + page_len];
    payload[..8].copy_from_slice(&page_id.to_le_bytes());
    payload[8..16].copy_from_slice(&(page_len as u64).to_le_bytes());
    fill_page(&mut payload[PAGE_IMAGE_BYTES_HEADER_LEN..]);
    payload
}

fn decode_page_image_bytes_payload(payload: &[u8]) -> Option<(PageId, &[u8])> {
    if payload.len() < PAGE_IMAGE_BYTES_HEADER_LEN {
        return None;
    }
    let page_id = u64::from_le_bytes(payload[..8].try_into().ok()?);
    let page_len = u64::from_le_bytes(payload[8..16].try_into().ok()?) as usize;
    let page = payload.get(PAGE_IMAGE_BYTES_HEADER_LEN..)?;
    if page.len() != page_len || page_len != page_size(page_id) {
        return None;
    }
    Some((page_id, page))
}

fn encode_page_patch_payload(
    page_id: PageId,
    before: &[u8; PAGE_SIZE],
    after: &[u8; PAGE_SIZE],
) -> Option<Vec<u8>> {
    let mut ranges = Vec::<(usize, usize)>::new();
    let mut offset = 0usize;
    while offset < PAGE_SIZE {
        if before[offset] == after[offset] {
            offset += 1;
            continue;
        }
        let start = offset;
        offset += 1;
        while offset < PAGE_SIZE && before[offset] != after[offset] {
            offset += 1;
        }
        ranges.push((start, offset - start));
    }

    if ranges.is_empty() {
        return None;
    }

    let payload_len = PAGE_PATCH_HEADER_LEN
        + ranges
            .iter()
            .map(|(_, len)| PAGE_PATCH_RANGE_HEADER_LEN + *len)
            .sum::<usize>();
    if payload_len >= PAGE_SIZE {
        return None;
    }

    let range_count = u32::try_from(ranges.len()).ok()?;
    let mut payload = Vec::with_capacity(payload_len);
    payload.extend_from_slice(&page_id.to_le_bytes());
    payload.extend_from_slice(&range_count.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    for (start, len) in ranges {
        let start = u32::try_from(start).ok()?;
        let len = u32::try_from(len).ok()?;
        payload.extend_from_slice(&start.to_le_bytes());
        payload.extend_from_slice(&len.to_le_bytes());
        payload.extend_from_slice(&after[start as usize..start as usize + len as usize]);
    }
    Some(payload)
}

type PagePatchRanges<'a> = Vec<(usize, &'a [u8])>;

fn decode_page_patch_payload(payload: &[u8]) -> Option<(PageId, PagePatchRanges<'_>)> {
    if payload.len() < PAGE_PATCH_HEADER_LEN {
        return None;
    }
    let page_id = u64::from_le_bytes(payload[..8].try_into().ok()?);
    if page_size(page_id) != PAGE_SIZE {
        return None;
    }
    let range_count = u32::from_le_bytes(payload[8..12].try_into().ok()?);
    let mut ranges = Vec::with_capacity(range_count as usize);
    let mut offset = PAGE_PATCH_HEADER_LEN;
    for _ in 0..range_count {
        let header = payload.get(offset..offset + PAGE_PATCH_RANGE_HEADER_LEN)?;
        let start = u32::from_le_bytes(header[..4].try_into().ok()?) as usize;
        let len = u32::from_le_bytes(header[4..8].try_into().ok()?) as usize;
        offset += PAGE_PATCH_RANGE_HEADER_LEN;
        let end = start.checked_add(len)?;
        if end > PAGE_SIZE {
            return None;
        }
        let data = payload.get(offset..offset + len)?;
        ranges.push((start, data));
        offset += len;
    }
    (offset == payload.len()).then_some((page_id, ranges))
}

struct PageImageRecovery<'a, S: ?Sized, F> {
    store: &'a S,
    checkpoint_lsn: Lsn,
    read_page_lsn: &'a F,
}

struct PageImage<'a> {
    lsn: Lsn,
    pid: PageId,
    data: &'a [u8],
}

struct PagePatch<'a> {
    lsn: Lsn,
    pid: PageId,
    ranges: PagePatchRanges<'a>,
}

fn recover_page_image<S, F>(
    recovery: &PageImageRecovery<'_, S, F>,
    report: &mut RecoveryReport,
    current_next: &mut PageId,
    page_buf: &mut Vec<u8>,
    image: PageImage<'_>,
) -> io::Result<()>
where
    S: RecoveryPageStore + ?Sized,
    F: Fn(&[u8]) -> Lsn,
{
    let PageImage { lsn, pid, data } = image;

    if lsn <= recovery.checkpoint_lsn {
        report.skipped_checkpoint += 1;
        return Ok(());
    }

    let page_len = page_size(pid);
    if data.len() != page_len {
        return Ok(());
    }

    if page_end_base_page(pid) >= *current_next {
        recovery.store.allocate(pid)?;
        *current_next = recovery.store.next_page_id();
    } else {
        page_buf.resize(page_len, 0);
        if recovery.store.read_page(pid, page_buf)? {
            let on_disk_lsn = (recovery.read_page_lsn)(page_buf);
            if on_disk_lsn >= lsn {
                report.skipped_page_lsn += 1;
                return Ok(());
            }
        }
    }

    recovery.store.write_page(pid, data)?;
    report.records_applied += 1;
    Ok(())
}

fn recover_page_patch<S, F>(
    recovery: &PageImageRecovery<'_, S, F>,
    report: &mut RecoveryReport,
    current_next: &mut PageId,
    page_buf: &mut Vec<u8>,
    patch: PagePatch<'_>,
) -> io::Result<()>
where
    S: RecoveryPageStore + ?Sized,
    F: Fn(&[u8]) -> Lsn,
{
    if patch.lsn <= recovery.checkpoint_lsn {
        report.skipped_checkpoint += 1;
        return Ok(());
    }

    let page_len = page_size(patch.pid);
    if page_len != PAGE_SIZE {
        return Ok(());
    }

    if page_end_base_page(patch.pid) >= *current_next {
        recovery.store.allocate(patch.pid)?;
        *current_next = recovery.store.next_page_id();
    }

    page_buf.resize(PAGE_SIZE, 0);
    if recovery.store.read_page(patch.pid, page_buf)? {
        let on_disk_lsn = (recovery.read_page_lsn)(page_buf);
        if on_disk_lsn >= patch.lsn {
            report.skipped_page_lsn += 1;
            return Ok(());
        }
    }

    for (start, data) in patch.ranges {
        page_buf[start..start + data.len()].copy_from_slice(data);
    }
    recovery.store.write_page(patch.pid, page_buf)?;
    report.records_applied += 1;
    Ok(())
}

fn recover_logical_payload<S, F>(
    recovery: &PageImageRecovery<'_, S, F>,
    report: &mut RecoveryReport,
    current_next: &mut PageId,
    page_buf: &mut Vec<u8>,
    lsn: Lsn,
    kind: u64,
    payload: &[u8],
) -> io::Result<bool>
where
    S: RecoveryPageStore + ?Sized,
    F: Fn(&[u8]) -> Lsn,
{
    if kind == LOGICAL_KIND_PAGE_IMAGE_BYTES {
        let Some((pid, page_data)) = decode_page_image_bytes_payload(payload) else {
            return Ok(false);
        };
        recover_page_image(
            recovery,
            report,
            current_next,
            page_buf,
            PageImage {
                lsn,
                pid,
                data: page_data,
            },
        )?;
        return Ok(true);
    }

    if kind == LOGICAL_KIND_PAGE_PATCH {
        let Some((pid, ranges)) = decode_page_patch_payload(payload) else {
            return Ok(false);
        };
        recover_page_patch(
            recovery,
            report,
            current_next,
            page_buf,
            PagePatch { lsn, pid, ranges },
        )?;
        return Ok(true);
    }

    Ok(true)
}

fn wal_direct_io_enabled() -> bool {
    matches!(
        std::env::var("PAGEBOX_WAL_DIRECT_IO").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

fn wal_sync_backend() -> WalSyncBackend {
    match std::env::var("PAGEBOX_WAL_SYNC_BACKEND").ok().as_deref() {
        Some("pwritev2_dsync") | Some("pwritev2-dsync") | Some("rwf_dsync") => {
            WalSyncBackend::Pwritev2Dsync
        }
        _ => WalSyncBackend::Fdatasync,
    }
}

thread_local! {
    static WAL_THREAD_STATES: RefCell<HashMap<usize, ThreadWalState>> = RefCell::new(HashMap::new());
}

#[derive(Clone, Copy)]
struct ThreadWalState {
    shard_idx: usize,
    last_lsn: Lsn,
}

fn align_lsn_to_shard(lsn: Lsn, shard_idx: usize, shard_count: usize) -> Lsn {
    debug_assert!(shard_idx < shard_count);
    let shard_count = shard_count as Lsn;
    let shard_idx = shard_idx as Lsn;
    let lsn = lsn.max(1);
    let current_shard = (lsn - 1) % shard_count;
    let delta = (shard_idx + shard_count - current_shard) % shard_count;
    lsn + delta
}

fn wal_shard_count() -> usize {
    std::env::var("PAGEBOX_WAL_SHARDS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .clamp(1, 256)
}

fn wal_shard_path(path: &Path, shard_idx: usize) -> std::path::PathBuf {
    if shard_idx == 0 {
        return path.to_path_buf();
    }

    let mut os = path.as_os_str().to_os_string();
    os.push(format!(".shard{shard_idx}"));
    std::path::PathBuf::from(os)
}

fn open_buffered_wal_fd(c_path: *const libc::c_char) -> io::Result<std::os::fd::RawFd> {
    let fd = unsafe { libc::open(c_path, libc::O_RDWR | libc::O_CREAT, 0o644) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(fd)
}

#[cfg(target_os = "linux")]
fn open_direct_wal_fd(c_path: *const libc::c_char) -> io::Result<std::os::fd::RawFd> {
    let fd = unsafe { libc::open(c_path, libc::O_RDWR | libc::O_CREAT | libc::O_DIRECT, 0o644) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(fd)
}

#[cfg(not(target_os = "linux"))]
fn open_direct_wal_fd(_c_path: *const libc::c_char) -> io::Result<std::os::fd::RawFd> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "WAL direct I/O is only available on Linux",
    ))
}

fn saturating_duration_nanos(elapsed: Duration) -> u64 {
    elapsed.as_nanos().min(u64::MAX as u128) as u64
}

impl Wal {
    fn spawn_writer(inner: &Arc<WalInner>) -> io::Result<std::thread::JoinHandle<()>> {
        let inner = inner.clone();
        threading::spawn_efficient("wal-writer", move || inner.writer_loop())
    }

    fn spawn_syncer(inner: &Arc<WalInner>) -> io::Result<std::thread::JoinHandle<()>> {
        let inner = inner.clone();
        threading::spawn_efficient("wal-syncer", move || inner.syncer_loop())
    }

    /// Open or create a WAL at `path` with the process-default sync backend
    /// and shard count. Equivalent to [`Wal::open_opts`]. Honours
    /// `PAGEBOX_WAL_DIRECT_IO`, `PAGEBOX_WAL_SYNC_BACKEND`, and
    /// `PAGEBOX_WAL_SHARDS`. Reopening scans the existing file, truncates any
    /// torn tail, and pre-allocates one `SEGMENT_SIZE` chunk ahead of the
    /// last valid byte.
    pub fn open(path: &Path) -> io::Result<Self> {
        Self::open_opts(path)
    }

    /// Alias for [`Wal::open`] (kept for the historical name).
    pub fn open_opts(path: &Path) -> io::Result<Self> {
        let sync_backend = wal_sync_backend();
        Self::open_opts_with_shards(path, sync_backend, wal_shard_count())
    }

    #[cfg(test)]
    pub(crate) fn open_with_shards_for_test(path: &Path, shard_count: usize) -> io::Result<Self> {
        Self::open_opts_with_shards(path, WalSyncBackend::Fdatasync, shard_count)
    }

    fn open_opts_with_shards(
        path: &Path,
        sync_backend: WalSyncBackend,
        shard_count: usize,
    ) -> io::Result<Self> {
        let shard_count = shard_count.clamp(1, 256);

        let mut opened = Vec::with_capacity(shard_count);
        for shard_idx in 0..shard_count {
            let shard_path = wal_shard_path(path, shard_idx);
            opened.push(Self::open_wal_file(&shard_path, sync_backend)?);
        }

        let max_lsn = opened.iter().map(|file| file.max_lsn).max().unwrap_or(0);
        let next_lsn = Arc::new(AtomicU64::new(max_lsn + 1));
        let mut shards = Vec::with_capacity(shard_count);
        for file in opened {
            shards.push(Self::start_shard(
                file,
                sync_backend,
                Arc::clone(&next_lsn),
            )?);
        }

        let mut shards = shards.into_iter();
        let primary = shards.next().expect("at least one WAL shard");

        Ok(Self {
            inner: primary.inner,
            writer: primary.writer,
            syncer: primary.syncer,
            extra_shards: shards.collect(),
            next_thread_shard: AtomicUsize::new(0),
        })
    }

    fn open_wal_file(path: &Path, sync_backend: WalSyncBackend) -> io::Result<OpenWalFile> {
        let (fd, direct_io) = Self::open_fd(path)?;

        let close_on_err = |e: io::Error| -> io::Error {
            unsafe { libc::close(fd) };
            e
        };

        let file_size = fstat_size(fd).map_err(close_on_err)?;
        if file_size < WAL_HEADER_SIZE as u64 {
            let hdr = build_wal_header(direct_io);
            let mut hdr_buf = AlignedBuf::new(WAL_HEADER_SIZE);
            hdr_buf.as_mut_slice().copy_from_slice(&hdr);
            pwrite_all(fd, hdr_buf.as_slice(), 0).map_err(close_on_err)?;
            extend_file(fd, SEGMENT_SIZE).map_err(close_on_err)?;
            if sync_backend == WalSyncBackend::Pwritev2Dsync {
                fdatasync_file(fd).map_err(close_on_err)?;
            }
            return Ok(OpenWalFile {
                fd,
                direct_io,
                file_offset: WAL_HEADER_SIZE as u64,
                allocated_size: SEGMENT_SIZE,
                max_lsn: 0,
            });
        }

        let mut hdr_buf = AlignedBuf::new(WAL_HEADER_SIZE);
        pread_all(fd, hdr_buf.as_mut_slice(), 0).map_err(close_on_err)?;
        let hdr: &[u8; WAL_HEADER_SIZE] = hdr_buf.as_slice().try_into().unwrap();
        validate_wal_header(hdr).map_err(close_on_err)?;

        let data_size = file_size - WAL_HEADER_SIZE as u64;
        let (max_lsn, data_valid_end) = WalInner::scan_valid_records(fd, data_size);
        let valid_end = WAL_HEADER_SIZE as u64 + data_valid_end;
        if valid_end < file_size && unsafe { libc::ftruncate(fd, valid_end as libc::off_t) } != 0 {
            return Err(close_on_err(io::Error::last_os_error()));
        }
        let allocated_size = {
            let target = round_up_u64(valid_end, SEGMENT_SIZE).max(SEGMENT_SIZE);
            if target > valid_end {
                extend_file(fd, target).map_err(close_on_err)?;
                if sync_backend == WalSyncBackend::Pwritev2Dsync {
                    fdatasync_file(fd).map_err(close_on_err)?;
                }
            }
            target
        };

        Ok(OpenWalFile {
            fd,
            direct_io,
            file_offset: valid_end,
            allocated_size,
            max_lsn,
        })
    }

    fn start_shard(
        file: OpenWalFile,
        sync_backend: WalSyncBackend,
        next_lsn: Arc<AtomicU64>,
    ) -> io::Result<WalShard> {
        let inner = Arc::new(WalInner::new(
            file.fd,
            file.direct_io,
            sync_backend,
            file.file_offset,
            file.allocated_size,
            file.max_lsn,
            next_lsn,
        ));

        let writer = Self::spawn_writer(&inner).inspect_err(|_| {
            let state = inner.state.lock();
            unsafe { libc::close(state.fd) };
        })?;
        let syncer = if sync_backend == WalSyncBackend::Fdatasync {
            match Self::spawn_syncer(&inner) {
                Ok(syncer) => Some(syncer),
                Err(err) => {
                    {
                        let mut state = inner.state.lock();
                        state.shutdown = true;
                    }
                    inner.flush_requested.notify_all();
                    let _ = writer.join();
                    let state = inner.state.lock();
                    unsafe { libc::close(state.fd) };
                    return Err(err);
                }
            }
        } else {
            None
        };

        Ok(WalShard {
            inner,
            writer: Some(writer),
            syncer,
        })
    }

    fn open_fd(path: &Path) -> io::Result<(std::os::fd::RawFd, bool)> {
        let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

        if !wal_direct_io_enabled() {
            return open_buffered_wal_fd(c_path.as_ptr()).map(|fd| (fd, false));
        }

        if let Ok(fd) = open_direct_wal_fd(c_path.as_ptr()) {
            return Ok((fd, true));
        }

        open_buffered_wal_fd(c_path.as_ptr()).map(|fd| (fd, false))
    }

    /// Switch the per-process commit policy. Strict and relaxed share one
    /// pipeline; the policy only changes whether `commit_*` / `flush_*` calls
    /// block on the syncer. Wakes background syncers so any relaxed-mode
    /// appends in flight get advanced.
    pub fn set_commit_mode(&self, mode: CommitMode) {
        self.inner.commit_mode.store(mode as u64, Ordering::Release);
        self.inner.flush_requested.notify_all();
        for shard in &self.extra_shards {
            shard
                .inner
                .commit_mode
                .store(mode as u64, Ordering::Release);
            shard.inner.flush_requested.notify_all();
        }
    }

    /// Read the live commit mode. See [`CommitMode`].
    pub fn commit_mode(&self) -> CommitMode {
        match self.inner.commit_mode.load(Ordering::Acquire) {
            x if x == CommitMode::Relaxed as u64 => CommitMode::Relaxed,
            _ => CommitMode::Strict,
        }
    }

    fn shard_count(&self) -> usize {
        1 + self.extra_shards.len()
    }

    fn current_thread_shard(&self) -> usize {
        let shard_count = self.shard_count();
        if shard_count == 1 {
            return 0;
        }

        let wal_key = Arc::as_ptr(&self.inner) as usize;
        WAL_THREAD_STATES.with(|states| {
            let mut states = states.borrow_mut();
            states
                .entry(wal_key)
                .or_insert_with(|| ThreadWalState {
                    shard_idx: self.next_thread_shard.fetch_add(1, Ordering::Relaxed) % shard_count,
                    last_lsn: 0,
                })
                .shard_idx
        })
    }

    fn current_thread_last_lsn(&self) -> Lsn {
        let wal_key = Arc::as_ptr(&self.inner) as usize;
        WAL_THREAD_STATES.with(|states| {
            states
                .borrow()
                .get(&wal_key)
                .map(|state| state.last_lsn)
                .unwrap_or(0)
        })
    }

    fn set_current_thread_last_lsn(&self, lsn: Lsn) {
        let wal_key = Arc::as_ptr(&self.inner) as usize;
        let shard_count = self.shard_count();
        let shard_idx = self.shard_idx_for_lsn(lsn);
        WAL_THREAD_STATES.with(|states| {
            let mut states = states.borrow_mut();
            let state = states.entry(wal_key).or_insert_with(|| ThreadWalState {
                shard_idx: if shard_count == 1 { 0 } else { shard_idx },
                last_lsn: 0,
            });
            state.shard_idx = shard_idx;
            state.last_lsn = state.last_lsn.max(lsn);
        });
    }

    fn shard_idx_for_lsn(&self, lsn: Lsn) -> usize {
        let shard_count = self.shard_count();
        if shard_count == 1 {
            return 0;
        }
        (lsn.saturating_sub(1) as usize) % shard_count
    }

    fn inner_for_lsn(&self, lsn: Lsn) -> &Arc<WalInner> {
        let shard_idx = self.shard_idx_for_lsn(lsn);
        if shard_idx == 0 {
            return &self.inner;
        }
        &self.extra_shards[shard_idx - 1].inner
    }

    fn for_each_inner(&self, mut f: impl FnMut(&Arc<WalInner>)) {
        f(&self.inner);
        for shard in &self.extra_shards {
            f(&shard.inner);
        }
    }

    fn for_each_inner_with_index(&self, mut f: impl FnMut(usize, &WalInner)) {
        f(0, &self.inner);
        for (idx, shard) in self.extra_shards.iter().enumerate() {
            f(idx + 1, &shard.inner);
        }
    }

    fn flush_targets(&self) -> Vec<(Arc<WalInner>, Lsn)> {
        let mut targets = Vec::with_capacity(self.shard_count());
        self.for_each_inner(|inner| {
            targets.push((Arc::clone(inner), inner.current_flush_target()));
        });
        targets
    }

    fn commit_targets(&self, targets: Vec<(Arc<WalInner>, Lsn)>) -> Lsn {
        let mut committed = 0;
        for (inner, target) in targets {
            if target == 0 {
                continue;
            }
            committed = committed.max(inner.commit_at_least(target));
        }
        committed
    }

    /// Atomically claim the next LSN. Routes to the shard owning the new LSN
    /// and returns the caller a value suitable for `append_*_with_lsn` /
    /// `append_logical_with_lsn`. Use this when you need the LSN before the
    /// bytes (e.g. to stamp it into the page being logged) rather than the
    /// bytes-first [`Wal::append_page_image`] / [`Wal::append_logical`] forms.
    pub fn claim_lsn(&self) -> Lsn {
        let shard_idx = self.current_thread_shard();
        let shard_count = self.shard_count();
        if shard_count == 1 {
            let lsn = self.inner.next_lsn.fetch_add(1, Ordering::Relaxed);
            self.set_current_thread_last_lsn(lsn);
            return lsn;
        }

        let next_lsn = &self.inner.next_lsn;
        loop {
            let current = next_lsn.load(Ordering::Relaxed);
            let lsn = align_lsn_to_shard(current, shard_idx, shard_count);
            let next = lsn + 1;
            if next_lsn
                .compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                self.set_current_thread_last_lsn(lsn);
                return lsn;
            }
        }
    }

    fn reserve_active_record_slot_locked(
        inner: &WalInner,
        state: &mut WalState,
    ) -> io::Result<(usize, usize, usize)> {
        state.active.packed_logical = None;
        let needs_new_batch = state.active.open_batch_meta_offset.is_none()
            || state.active.open_batch_count == BATCH_MAX_RECORDS;
        let needed = if needs_new_batch {
            2 * WAL_RECORD_SIZE
        } else {
            WAL_RECORD_SIZE
        };
        if state.active.used + needed > WAL_BUF_CAPACITY {
            inner.seal_active_buffer_locked(state)?;
            inner.flush_requested.notify_all();
        }
        if state.active.open_batch_meta_offset.is_none()
            || state.active.open_batch_count == BATCH_MAX_RECORDS
        {
            let meta_start = state.active.used;
            let meta_end = meta_start + WAL_RECORD_SIZE;
            init_batch_meta(&mut state.active.buffer.as_mut_slice()[meta_start..meta_end]);
            state.active.used = meta_end;
            state.active.open_batch_meta_offset = Some(meta_start);
            state.active.open_batch_count = 0;
        }

        let meta_start = state
            .active
            .open_batch_meta_offset
            .expect("open batch must exist");
        let slot_idx = state.active.open_batch_count;
        let data_start = state.active.used;
        Ok((meta_start, slot_idx, data_start))
    }

    fn finish_active_record_slot_locked(
        state: &mut WalState,
        meta_start: usize,
        slot_idx: usize,
        entry: BatchEntry,
    ) {
        let data_end = state.active.used + WAL_RECORD_SIZE;
        write_batch_entry(
            &mut state.active.buffer.as_mut_slice()[meta_start..meta_start + WAL_RECORD_SIZE],
            slot_idx,
            entry,
        );
        state.active.open_batch_count += 1;
        let batch_count = state.active.open_batch_count;
        set_batch_meta_count(
            &mut state.active.buffer.as_mut_slice()[meta_start..meta_start + WAL_RECORD_SIZE],
            batch_count,
        );
        state.active.used = data_end;
        state.active.records += 1;
        if entry.kind == RECORD_KIND_PAGE_IMAGE
            || entry.kind == RECORD_KIND_LOGICAL_PACKED
            || entry.flags & LOGICAL_FLAG_LAST == LOGICAL_FLAG_LAST
        {
            state.active.max_lsn = state.active.max_lsn.max(entry.lsn);
        }
        if state.active.open_batch_count == BATCH_MAX_RECORDS {
            state.active.open_batch_meta_offset = None;
            state.active.open_batch_count = 0;
        }
    }

    fn try_append_packed_logical_locked(
        state: &mut WalState,
        lsn: Lsn,
        kind: u64,
        payload: &[u8],
        entry_len: usize,
    ) -> bool {
        let Some(slot) = state.active.packed_logical else {
            return false;
        };
        if state.active.used != slot.data_start + WAL_RECORD_SIZE {
            state.active.packed_logical = None;
            return false;
        }

        let Some(new_len) = slot.len.checked_add(entry_len) else {
            return false;
        };
        if new_len > WAL_RECORD_SIZE {
            return false;
        }

        let data_start = slot.data_start + slot.len;
        let Some(written) = encode_packed_logical_entry(
            &mut state.active.buffer.as_mut_slice()[data_start..slot.data_start + WAL_RECORD_SIZE],
            lsn,
            kind,
            payload,
        ) else {
            return false;
        };
        debug_assert_eq!(written, entry_len);

        let max_lsn = slot.max_lsn.max(lsn);
        write_batch_entry(
            &mut state.active.buffer.as_mut_slice()
                [slot.meta_start..slot.meta_start + WAL_RECORD_SIZE],
            slot.slot_idx,
            BatchEntry::packed_logical(max_lsn, 0, new_len),
        );
        state.active.packed_logical = Some(PackedLogicalSlot {
            len: new_len,
            max_lsn,
            ..slot
        });
        state.active.max_lsn = state.active.max_lsn.max(max_lsn);
        true
    }

    /// Page-image append with in-place overwrite attempt.
    ///
    /// If `prev_record` is `Some(..)` and still overwritable (same shard,
    /// same buffer epoch, same page_id, slot still in the active buffer),
    /// the existing slot's bytes are rewritten with the new page image and
    /// the LSN/CRC fields updated in place — saving one full record slot.
    /// Otherwise (or if `prev_record` is `None`) a fresh append occurs.
    /// Returns the LSN claimed for the append **and** a
    /// [`BufferedWalRecord`] handle the caller can pass back on the next
    /// mutation of the same page for another overwrite attempt.
    ///
    /// `fill_page` runs under the shard state lock; copy only the page bytes
    /// into the supplied buffer — long computation here serialises the
    /// shard.
    pub fn append_or_overwrite_page_image<F>(
        &self,
        prev_record: Option<BufferedWalRecord>,
        page_id: PageId,
        fill_page: F,
    ) -> io::Result<(Lsn, BufferedWalRecord)>
    where
        F: FnOnce(Lsn, &mut [u8; PAGE_SIZE]),
    {
        let lsn = self.claim_lsn();
        let inner = self.inner_for_lsn(lsn);
        let record = if let Some(prev_record) = prev_record {
            inner.stats.events.inc(WalEvent::PageImageOverwriteAttempts);
            if self.can_overwrite_buffered_page_image(prev_record, lsn, page_id)? {
                self.overwrite_buffered_page_image_with_lsn(prev_record, lsn, page_id, fill_page)?;
                inner
                    .stats
                    .events
                    .inc(WalEvent::PageImageOverwriteSuccesses);
                prev_record
            } else {
                self.append_page_image_with_lsn(lsn, page_id, fill_page)?
            }
        } else {
            self.append_page_image_with_lsn(lsn, page_id, fill_page)?
        };

        Ok((lsn, record))
    }

    /// Append a full page image using a pre-claimed `lsn` (see
    /// [`Wal::claim_lsn`]). Used by the buffer pool's eviction / writeback
    /// path which already holds a claimed LSN. `fill_page` copies the page
    /// bytes into the WAL buffer under the shard lock.
    pub fn append_page_image_with_lsn(
        &self,
        lsn: Lsn,
        page_id: PageId,
        fill_page: impl FnOnce(Lsn, &mut [u8; PAGE_SIZE]),
    ) -> io::Result<BufferedWalRecord> {
        let inner = self.inner_for_lsn(lsn);
        inner.stats.events.inc(WalEvent::PageImageRecords);
        inner
            .stats
            .events
            .add(WalEvent::PageImageBytes, PAGE_SIZE as isize);
        let mut state = inner.state.lock();
        let (meta_start, slot_idx, data_start) =
            Self::reserve_active_record_slot_locked(inner, &mut state)?;
        let data_end = data_start + WAL_RECORD_SIZE;
        let page_buf: &mut [u8; PAGE_SIZE] = (&mut state.active.buffer.as_mut_slice()
            [data_start..data_end])
            .try_into()
            .expect("page slot must be page-sized");
        fill_page(lsn, page_buf);
        Self::finish_active_record_slot_locked(
            &mut state,
            meta_start,
            slot_idx,
            BatchEntry::page_image(lsn, page_id, 0),
        );
        inner.appended_lsn.fetch_max(lsn, Ordering::Release);
        if data_start == WAL_RECORD_SIZE || state.active.records >= WAL_RELAXED_WRITE_RECORDS {
            inner.flush_requested.notify_all();
        }
        Ok(BufferedWalRecord {
            epoch: state.active.epoch,
            offset: encode_record_offset(meta_start, slot_idx),
            shard_idx: self.shard_idx_for_lsn(lsn) as u16,
        })
    }

    /// Attempt to overwrite a previously-buffered page image in place using a
    /// pre-claimed `lsn`. Returns `Ok(true)` if the overwrite succeeded and
    /// `Ok(false)` if it failed any precheck (shard moved, epoch advanced,
    /// different page_id, slot already drained) — in the latter case the
    /// caller should fall back to a fresh `append_page_image_with_lsn`.
    pub fn try_overwrite_page_image_with_lsn(
        &self,
        record: BufferedWalRecord,
        lsn: Lsn,
        page_id: PageId,
        fill_page: impl FnOnce(Lsn, &mut [u8; PAGE_SIZE]),
    ) -> io::Result<bool> {
        let inner = self.inner_for_lsn(lsn);
        inner.stats.events.inc(WalEvent::PageImageOverwriteAttempts);
        if !self.overwrite_buffered_page_image_with_lsn(record, lsn, page_id, fill_page)? {
            return Ok(false);
        }
        inner
            .stats
            .events
            .inc(WalEvent::PageImageOverwriteSuccesses);
        Ok(true)
    }

    /// Append a page patch (a delta of changed byte ranges) as a logical
    /// record. Returns `Ok(false)` if `before` and `after` have no
    /// differing ranges that fit the patch encoding; in that case the caller
    /// should fall back to a full page image. Patches are decoded on
    /// replay by [`Wal::recover`] and applied to the page in store.
    pub fn append_page_patch_with_lsn(
        &self,
        lsn: Lsn,
        page_id: PageId,
        before: &[u8; PAGE_SIZE],
        after: &[u8; PAGE_SIZE],
    ) -> io::Result<bool> {
        let Some(payload) = encode_page_patch_payload(page_id, before, after) else {
            return Ok(false);
        };
        let inner = self.inner_for_lsn(lsn);
        inner.stats.events.inc(WalEvent::LogicalRecords);
        inner.stats.events.add(
            WalEvent::LogicalBytes,
            payload.len().min(isize::MAX as usize) as isize,
        );
        self.append_logical_with_lsn(lsn, LOGICAL_KIND_PAGE_PATCH, &payload)?;
        Ok(true)
    }

    /// Append a logical record (caller-defined `kind`, opaque `payload`)
    /// and return its newly-claimed LSN. Payloads are chunked at
    /// `LOGICAL_CHUNK_MAX_LEN`; short payloads are packed into a single
    /// data page alongside others. [`Wal::recover`] surfaces these as
    /// [`WalReplayRecord::Logical`] for the caller's index/table layer to
    /// apply.
    pub fn append_logical(&self, kind: u64, payload: &[u8]) -> io::Result<Lsn> {
        let lsn = self.claim_lsn();
        let inner = self.inner_for_lsn(lsn);
        inner.stats.events.inc(WalEvent::LogicalRecords);
        inner.stats.events.add(
            WalEvent::LogicalBytes,
            payload.len().min(isize::MAX as usize) as isize,
        );
        self.append_logical_with_lsn(lsn, kind, payload)?;
        Ok(lsn)
    }

    /// Append a logical record using a pre-claimed `lsn` (paired with
    /// [`Wal::claim_lsn`]). The LSN is honoured exactly even across shards;
    /// callers must guarantee no other append may have already used it.
    pub fn append_logical_with_lsn(&self, lsn: Lsn, kind: u64, payload: &[u8]) -> io::Result<()> {
        if kind != LOGICAL_KIND_PAGE_IMAGE_BYTES && payload.len() <= PACKED_LOGICAL_MAX_PAYLOAD_LEN
        {
            return self.append_packed_logical_with_lsn(lsn, kind, payload);
        }

        let chunks = chunk_count(payload.len());
        let inner = self.inner_for_lsn(lsn);
        let mut state = inner.state.lock();
        for chunk_idx in 0..chunks {
            let start = chunk_idx * LOGICAL_CHUNK_MAX_LEN;
            let end = (start + LOGICAL_CHUNK_MAX_LEN).min(payload.len());
            let chunk = if start < payload.len() {
                &payload[start..end]
            } else {
                &[]
            };
            let (meta_start, slot_idx, data_start) =
                Self::reserve_active_record_slot_locked(inner, &mut state)?;
            let data_end = data_start + WAL_RECORD_SIZE;
            let slot = &mut state.active.buffer.as_mut_slice()[data_start..data_end];
            slot.fill(0);
            slot[..chunk.len()].copy_from_slice(chunk);
            let mut flags = 0;
            if chunk_idx == 0 {
                flags |= LOGICAL_FLAG_FIRST;
            }
            if chunk_idx + 1 == chunks {
                flags |= LOGICAL_FLAG_LAST;
            }
            Self::finish_active_record_slot_locked(
                &mut state,
                meta_start,
                slot_idx,
                BatchEntry::logical_chunk(lsn, kind, 0, flags, chunk.len()),
            );
            if data_start == WAL_RECORD_SIZE || state.active.records >= WAL_RELAXED_WRITE_RECORDS {
                inner.flush_requested.notify_all();
            }
        }

        inner.appended_lsn.fetch_max(lsn, Ordering::Release);
        Ok(())
    }

    fn append_packed_logical_with_lsn(
        &self,
        lsn: Lsn,
        kind: u64,
        payload: &[u8],
    ) -> io::Result<()> {
        let entry_len = packed_logical_entry_len(payload.len()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "packed logical payload too large",
            )
        })?;
        if entry_len > WAL_RECORD_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "packed logical payload too large",
            ));
        }

        let inner = self.inner_for_lsn(lsn);
        let mut state = inner.state.lock();
        if Self::try_append_packed_logical_locked(&mut state, lsn, kind, payload, entry_len) {
            inner.appended_lsn.fetch_max(lsn, Ordering::Release);
            return Ok(());
        }

        let (meta_start, slot_idx, data_start) =
            Self::reserve_active_record_slot_locked(inner, &mut state)?;
        let data_end = data_start + WAL_RECORD_SIZE;
        let slot = &mut state.active.buffer.as_mut_slice()[data_start..data_end];
        slot.fill(0);
        let written = encode_packed_logical_entry(slot, lsn, kind, payload).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "packed logical payload too large",
            )
        })?;
        debug_assert_eq!(written, entry_len);
        Self::finish_active_record_slot_locked(
            &mut state,
            meta_start,
            slot_idx,
            BatchEntry::packed_logical(lsn, 0, entry_len),
        );
        state.active.packed_logical = Some(PackedLogicalSlot {
            meta_start,
            slot_idx,
            data_start,
            len: entry_len,
            max_lsn: lsn,
        });
        inner.appended_lsn.fetch_max(lsn, Ordering::Release);
        if data_start == WAL_RECORD_SIZE || state.active.records >= WAL_RELAXED_WRITE_RECORDS {
            inner.flush_requested.notify_all();
        }
        Ok(())
    }

    /// Append a full page image using a `&[u8; PAGE_SIZE]` directly. Allocates
    /// a fresh LSN; returns it on success.
    pub fn append_page_image(
        &self,
        page_id: PageId,
        page_data: &[u8; PAGE_SIZE],
    ) -> io::Result<Lsn> {
        let lsn = self.claim_lsn();
        let _ = self.append_page_image_with_lsn(lsn, page_id, |_, dst| {
            dst.copy_from_slice(page_data);
        })?;
        Ok(lsn)
    }

    /// Append a variable-length page image as a logical record (rather than a
    /// fixed `PAGE_SIZE` page slot). `page_len` is the length covered by
    /// `fill_page`; `page_id` carries the page ID.
    pub fn append_page_image_bytes_with_lsn(
        &self,
        lsn: Lsn,
        page_id: PageId,
        page_len: usize,
        fill_page: impl FnOnce(Lsn, &mut [u8]),
    ) -> io::Result<()> {
        let inner = self.inner_for_lsn(lsn);
        inner.stats.events.inc(WalEvent::PageImageRecords);
        inner.stats.events.add(
            WalEvent::PageImageBytes,
            page_len.min(isize::MAX as usize) as isize,
        );
        let payload = encode_page_image_bytes_payload(page_id, page_len, |page| {
            fill_page(lsn, page);
        });
        self.append_logical_with_lsn(lsn, LOGICAL_KIND_PAGE_IMAGE_BYTES, &payload)
    }

    /// Variable-length page-image append using `&[u8]` directly. Allocates a
    /// fresh LSN; returns it on success.
    pub fn append_page_image_bytes(&self, page_id: PageId, page_data: &[u8]) -> io::Result<Lsn> {
        let lsn = self.claim_lsn();
        self.append_page_image_bytes_with_lsn(lsn, page_id, page_data.len(), |_, dst| {
            dst.copy_from_slice(page_data);
        })?;
        Ok(lsn)
    }

    fn can_overwrite_buffered_page_image(
        &self,
        record: BufferedWalRecord,
        lsn: Lsn,
        page_id: PageId,
    ) -> io::Result<bool> {
        if usize::from(record.shard_idx) != self.shard_idx_for_lsn(lsn) {
            return Ok(false);
        }
        let inner = self.inner_for_lsn(lsn);
        let state = inner.state.lock();
        if state.active.epoch != record.epoch {
            return Ok(false);
        }
        let (meta_start, slot_idx) = decode_record_offset(record.offset);
        let meta_end = meta_start + WAL_RECORD_SIZE;
        if meta_end > state.active.used {
            return Ok(false);
        }
        let meta = &state.active.buffer.as_slice()[meta_start..meta_end];
        let Some(count) = batch_meta_count_unchecked(meta) else {
            return Ok(false);
        };
        if slot_idx >= count {
            return Ok(false);
        }
        let entry = read_batch_entry(meta, slot_idx);
        Ok(entry.kind == RECORD_KIND_PAGE_IMAGE && entry.arg == page_id)
    }

    /// Strict overwrite attempt: same as
    /// [`Wal::try_overwrite_page_image_with_lsn`] but takes a pre-claimed
    /// LSN directly. Returns `Ok(true)` on success.
    pub fn overwrite_buffered_page_image_with_lsn(
        &self,
        record: BufferedWalRecord,
        lsn: Lsn,
        page_id: PageId,
        fill_page: impl FnOnce(Lsn, &mut [u8; PAGE_SIZE]),
    ) -> io::Result<bool> {
        if usize::from(record.shard_idx) != self.shard_idx_for_lsn(lsn) {
            return Ok(false);
        }
        let inner = self.inner_for_lsn(lsn);
        let mut state = inner.state.lock();
        if state.active.epoch != record.epoch {
            return Ok(false);
        }
        let (meta_start, slot_idx) = decode_record_offset(record.offset);
        let meta_end = meta_start + WAL_RECORD_SIZE;
        if meta_end > state.active.used {
            return Ok(false);
        }
        let count = {
            let meta = &state.active.buffer.as_slice()[meta_start..meta_end];
            let Some(count) = batch_meta_count_unchecked(meta) else {
                return Ok(false);
            };
            count
        };
        if slot_idx >= count {
            return Ok(false);
        }
        let entry = {
            let meta = &state.active.buffer.as_slice()[meta_start..meta_end];
            read_batch_entry(meta, slot_idx)
        };
        if entry.kind != RECORD_KIND_PAGE_IMAGE || entry.arg != page_id {
            return Ok(false);
        }
        {
            let meta = &mut state.active.buffer.as_mut_slice()[meta_start..meta_end];
            overwrite_batch_entry_lsn(meta, slot_idx, lsn);
            overwrite_batch_entry_crc(meta, slot_idx, 0);
        }
        let data_start = meta_start + WAL_RECORD_SIZE * (slot_idx + 1);
        let data_end = data_start + WAL_RECORD_SIZE;
        let page_buf: &mut [u8; PAGE_SIZE] = (&mut state.active.buffer.as_mut_slice()
            [data_start..data_end])
            .try_into()
            .expect("page slot must be page-sized");
        fill_page(lsn, page_buf);
        state.active.max_lsn = state.active.max_lsn.max(lsn);
        inner.appended_lsn.fetch_max(lsn, Ordering::Release);
        Ok(true)
    }

    /// Commit every shard's appends so far under the current
    /// [`CommitMode`]: strict blocks until durable, relaxed returns the
    /// requested LSN immediately. Returns the LSN highest across shards
    /// (or `0` if no shard had pending appends).
    pub fn commit(&self) -> Lsn {
        let targets = self.flush_targets();
        if targets.iter().all(|(_, target)| *target == 0) {
            return 0;
        }
        self.commit_targets(targets)
    }

    /// Commit only this thread's appends so far — uses the calling thread's
    /// shard affinity to skip scanning every shard. Useful for benchmarks
    /// where each worker is the only producer on its shard.
    pub fn commit_current_thread(&self) -> Lsn {
        let target = self.current_thread_last_lsn();
        if target == 0 {
            return self.commit();
        }
        self.inner_for_lsn(target).commit_at_least(target)
    }

    /// Strictly flush at least `target_lsn`. Under [`CommitMode::Strict`]
    /// blocks until `durable_lsn >= target_lsn`; under [`CommitMode::Relaxed`]
    /// returns the target LSN immediately without waiting. Routes to the
    /// shard owning `target_lsn`.
    pub fn flush_at_least(&self, target_lsn: Lsn) -> Lsn {
        self.inner_for_lsn(target_lsn)
            .strict_flush_at_least(target_lsn)
    }

    /// Force-flush every shard's pending appends and block until all are
    /// durable. Always strict — used at quiescence / checkpoint. Returns the
    /// highest durable LSN across shards.
    pub fn flush(&self) -> Lsn {
        let targets = self.flush_targets();
        if targets.iter().all(|(_, target)| *target == 0) {
            return 0;
        }
        let mut flushed = 0;
        for (inner, target) in targets {
            if target == 0 {
                continue;
            }
            flushed = flushed.max(inner.strict_flush_at_least(target));
        }
        flushed
    }

    /// Deliberate crash shutdown: stop worker threads in place **without**
    /// draining pending appends and close the fd. Used by recovery tests to
    /// simulate a process crash mid-append. After this the WAL file resembles
    /// one torn mid-batch; reopening must call [`Wal::recover`] before any
    /// new appends.
    ///
    /// Consumes `self` via `ManuallyDrop` so no `Drop` cleanup runs.
    pub fn crash(self) {
        let mut this = std::mem::ManuallyDrop::new(self);
        let this: &mut Wal = &mut this;
        Self::stop_shard(&mut this.inner, &mut this.writer, &mut this.syncer, true);
        for shard in &mut this.extra_shards {
            Self::stop_shard(&mut shard.inner, &mut shard.writer, &mut shard.syncer, true);
        }
    }

    fn stop_shard(
        inner: &mut Arc<WalInner>,
        writer: &mut Option<std::thread::JoinHandle<()>>,
        syncer: &mut Option<std::thread::JoinHandle<()>>,
        crash_shutdown: bool,
    ) {
        {
            let mut state = inner.state.lock();
            state.crash_shutdown = crash_shutdown;
            state.shutdown = true;
            if !crash_shutdown {
                let target = inner.appended_lsn.load(Ordering::Relaxed);
                inner
                    .requested_write_lsn
                    .fetch_max(target, Ordering::Release);
                inner
                    .requested_durable_lsn
                    .fetch_max(target, Ordering::Release);
            }
        }
        inner.flush_requested.notify_all();
        if let Some(writer) = writer.take() {
            let _ = writer.join();
        }
        if let Some(syncer) = syncer.take() {
            let _ = syncer.join();
        }
        let inner = Arc::get_mut(inner).expect("WAL inner still shared at close");
        let state = inner.state.get_mut();
        unsafe {
            if !crash_shutdown {
                libc::fsync(state.fd);
            }
            libc::close(state.fd);
        }
    }

    /// The highest LSN currently durable across all shards. Acquire-load on
    /// each shard's `durable_lsn` and return the max. Cheap (no kernel I/O).
    pub fn durable_lsn(&self) -> Lsn {
        self.max_lsn(|inner| inner.durable_lsn.load(Ordering::Acquire))
    }

    fn max_lsn(&self, load: impl Fn(&WalInner) -> Lsn) -> Lsn {
        let mut max_lsn = 0;
        self.for_each_inner_with_index(|_, inner| {
            max_lsn = max_lsn.max(load(inner));
        });
        max_lsn
    }

    /// Forward WAL-side metrics (event counters and latency histograms) to
    /// `visitor`, visiting the primary shard and every extra shard. No-op
    /// under `--no-default-features`.
    pub fn visit_metrics<V: MetricVisitor + ?Sized>(&self, visitor: &mut V) {
        self.inner.stats.visit_metrics(visitor);
        for shard in &self.extra_shards {
            shard.inner.stats.visit_metrics(visitor);
        }
    }

    /// Advance `next_lsn` past `min_lsn` if it currently sits at or below.
    /// Used during recovery to ensure no future append reuses an LSN that a
    /// (possibly missing-tail) shard had already handed out.
    pub fn advance_lsn_past(&self, min_lsn: Lsn) {
        self.inner.advance_lsn_past(min_lsn);
        for shard in &self.extra_shards {
            shard.inner.advance_lsn_past(min_lsn);
        }
    }

    /// Streaming page-image replay. Flushes pending appends, then iterates
    /// page-image records (in single-shard order, or LSN-merged for
    /// multi-shard). The closure receives `(lsn, page_id, &page)` for each
    /// record. Skips logical records. Use [`Wal::replay_records`] for the
    /// variant that yields both record kinds.
    pub fn replay<F>(&self, f: F) -> io::Result<()>
    where
        F: FnMut(Lsn, PageId, &[u8; PAGE_SIZE]),
    {
        if self.shard_count() > 1 {
            let mut f = f;
            return self.replay_records(|record| {
                if let WalReplayRecord::PageImage { lsn, page_id, data } = record
                    && let Ok(page) = <&[u8; PAGE_SIZE]>::try_from(data)
                {
                    f(lsn, page_id, page);
                }
            });
        }
        self.flush();
        self.inner.replay(f)
    }

    /// Streaming replay with both [`WalReplayRecord::PageImage`] and
    /// [`WalReplayRecord::Logical`] surfaced to the closure. Flushes pending
    /// appends first; multi-shard WALs collect into memory and sort by LSN
    /// before invoking the closure. Single-shard streams records directly.
    pub fn replay_records<F>(&self, f: F) -> io::Result<()>
    where
        F: FnMut(WalReplayRecord<'_>),
    {
        self.flush();
        if self.shard_count() > 1 {
            return self.replay_records_merged(f);
        }
        self.inner.replay_records(f)
    }

    fn replay_records_merged<F>(&self, mut f: F) -> io::Result<()>
    where
        F: FnMut(WalReplayRecord<'_>),
    {
        let mut records = Vec::new();
        self.collect_records(&mut records)?;
        records.sort_by_key(OwnedWalReplayRecord::lsn);
        for record in &records {
            match record {
                OwnedWalReplayRecord::PageImage { lsn, page_id, data } => {
                    f(WalReplayRecord::PageImage {
                        lsn: *lsn,
                        page_id: *page_id,
                        data,
                    });
                }
                OwnedWalReplayRecord::Logical { lsn, kind, payload } => {
                    f(WalReplayRecord::Logical {
                        lsn: *lsn,
                        kind: *kind,
                        payload,
                    });
                }
            }
        }
        Ok(())
    }

    fn collect_records(&self, records: &mut Vec<OwnedWalReplayRecord>) -> io::Result<()> {
        let mut result = Ok(());
        self.for_each_inner(|inner| {
            if result.is_err() {
                return;
            }
            result = inner.replay_records(|record| match record {
                WalReplayRecord::PageImage { lsn, page_id, data } => {
                    records.push(OwnedWalReplayRecord::PageImage {
                        lsn,
                        page_id,
                        data: data.to_vec(),
                    });
                }
                WalReplayRecord::Logical { lsn, kind, payload } => {
                    records.push(OwnedWalReplayRecord::Logical {
                        lsn,
                        kind,
                        payload: payload.to_vec(),
                    });
                }
            });
        });
        result
    }

    /// Drive recovery into `store`.
    ///
    /// Flushes pending appends, then replays records in LSN order applying
    /// the idempotent rule: page-image records with `lsn <= checkpoint_lsn`
    /// or `lsn <= read_page_lsn(page_bytes)` are skipped, the rest are
    /// written into `store`. Logical records are forwarded to the caller-
    /// supplied `read_page_lsn` only for the LSN-derivation step; logical
    /// payloads (page-patch included) are applied inline via the recovery
    /// helpers. Returns a [`RecoveryReport`] with the scan/apply counts and
    /// the highest LSN seen.
    pub fn recover<S, F>(
        &self,
        store: &S,
        checkpoint_lsn: Lsn,
        read_page_lsn: F,
    ) -> io::Result<RecoveryReport>
    where
        S: RecoveryPageStore + ?Sized,
        F: Fn(&[u8]) -> Lsn,
    {
        self.flush();
        if self.shard_count() > 1 {
            return self.recover_merged(store, checkpoint_lsn, read_page_lsn);
        }
        self.inner.recover(store, checkpoint_lsn, read_page_lsn)
    }

    fn recover_merged<S, F>(
        &self,
        store: &S,
        checkpoint_lsn: Lsn,
        read_page_lsn: F,
    ) -> io::Result<RecoveryReport>
    where
        S: RecoveryPageStore + ?Sized,
        F: Fn(&[u8]) -> Lsn,
    {
        let mut records = Vec::new();
        self.collect_records(&mut records)?;
        records.sort_by_key(OwnedWalReplayRecord::lsn);

        let mut report = RecoveryReport::default();
        let mut current_next = store.next_page_id();
        let mut page_buf = Vec::new();
        let recovery = PageImageRecovery {
            store,
            checkpoint_lsn,
            read_page_lsn: &read_page_lsn,
        };

        for record in &records {
            report.records_scanned += 1;
            report.max_lsn = report.max_lsn.max(record.lsn());
            match record {
                OwnedWalReplayRecord::PageImage { lsn, page_id, data } => {
                    recover_page_image(
                        &recovery,
                        &mut report,
                        &mut current_next,
                        &mut page_buf,
                        PageImage {
                            lsn: *lsn,
                            pid: *page_id,
                            data,
                        },
                    )?;
                }
                OwnedWalReplayRecord::Logical { lsn, kind, payload } => {
                    if !recover_logical_payload(
                        &recovery,
                        &mut report,
                        &mut current_next,
                        &mut page_buf,
                        *lsn,
                        *kind,
                        payload,
                    )? {
                        return Ok(report);
                    }
                }
            }
        }

        if report.records_applied > 0 {
            store.sync()?;
        }

        Ok(report)
    }

    /// Truncate every shard file back to its header page, discarding all
    /// appended records. Used after a checkpoint makes the existing log
    /// obsolete. Flushes pending appends first so any concurrent appenders
    /// observe the truncation point consistently.
    pub fn reset(&self) -> io::Result<()> {
        self.flush();
        self.inner.reset()?;
        for shard in &self.extra_shards {
            shard.inner.reset()?;
        }
        Ok(())
    }
}

impl Drop for Wal {
    fn drop(&mut self) {
        Self::stop_shard(&mut self.inner, &mut self.writer, &mut self.syncer, false);
        for shard in &mut self.extra_shards {
            Self::stop_shard(
                &mut shard.inner,
                &mut shard.writer,
                &mut shard.syncer,
                false,
            );
        }
    }
}

impl WalInner {
    fn request_durable(&self, target_lsn: Lsn) {
        let _state = self.state.lock();
        self.requested_durable_lsn
            .fetch_max(target_lsn, Ordering::Release);
        self.requested_write_lsn
            .fetch_max(target_lsn, Ordering::Release);
        self.flush_requested.notify_all();
    }

    fn wait_for_durable(&self, target_lsn: Lsn) -> Lsn {
        let mut state = self.state.lock();
        loop {
            let current = self.durable_lsn.load(Ordering::Acquire);
            if current >= target_lsn {
                return current;
            }
            self.requested_durable_lsn
                .fetch_max(target_lsn, Ordering::Release);
            self.requested_write_lsn
                .fetch_max(target_lsn, Ordering::Release);
            self.flush_requested.notify_all();
            self.stats.events.inc(WalEvent::FlushWait);
            state.flush_waiters += 1;
            let wait_start = Instant::now();
            self.flush_done
                .wait_for(&mut state, Duration::from_millis(10));
            state.flush_waiters -= 1;
            self.stats
                .latencies
                .get(WalLatency::FlushWait)
                .record(saturating_duration_nanos(wait_start.elapsed()));
        }
    }

    fn commit_at_least(&self, target_lsn: Lsn) -> Lsn {
        self.stats.events.inc(WalEvent::FlushCall);
        let current = self.durable_lsn.load(Ordering::Acquire);
        if current >= target_lsn {
            self.stats.events.inc(WalEvent::FlushFastPath);
            return current;
        }
        let strict = self.commit_mode.load(Ordering::Acquire) == CommitMode::Strict as u64;
        if strict {
            self.request_durable(target_lsn);
            return self.wait_for_durable(target_lsn);
        }
        target_lsn
    }

    fn strict_flush_at_least(&self, target_lsn: Lsn) -> Lsn {
        self.stats.events.inc(WalEvent::FlushCall);
        let current = self.durable_lsn.load(Ordering::Acquire);
        if current >= target_lsn {
            self.stats.events.inc(WalEvent::FlushFastPath);
            return current;
        }
        self.request_durable(target_lsn);
        self.wait_for_durable(target_lsn)
    }

    fn current_flush_target(&self) -> Lsn {
        self.appended_lsn.load(Ordering::Relaxed)
    }

    fn new(
        fd: std::os::fd::RawFd,
        direct_io: bool,
        sync_backend: WalSyncBackend,
        file_offset: u64,
        allocated_size: u64,
        max_lsn: Lsn,
        next_lsn: Arc<AtomicU64>,
    ) -> Self {
        Self {
            state: Mutex::new(WalState {
                fd,
                direct_io,
                active: WalBuffer::new(1),
                spare_buffers: vec![WalBuffer::new(2)],
                pending_writes: VecDeque::new(),
                writes_in_progress: 0,
                file_offset,
                allocated_size,
                flush_waiters: 0,
                next_buffer_epoch: 3,
                crash_shutdown: false,
                shutdown: false,
            }),
            sync_backend,
            next_lsn,
            appended_lsn: AtomicU64::new(max_lsn),
            written_lsn: AtomicU64::new(max_lsn),
            durable_lsn: AtomicU64::new(max_lsn),
            requested_write_lsn: AtomicU64::new(max_lsn),
            requested_durable_lsn: AtomicU64::new(max_lsn),
            commit_mode: AtomicU64::new(CommitMode::Strict as u64),
            flush_done: Condvar::new(),
            flush_requested: Condvar::new(),
            stats: WalStats::default(),
        }
    }

    fn maybe_group_commit_delay(&self, leader_target_lsn: Lsn) {
        let max_delay_us = self.sync_backend.group_commit_delay_max_us();
        if max_delay_us == 0 {
            return;
        }

        let target_records = self.sync_backend.group_commit_target_records();

        let initial_delay_us = {
            let state = self.state.lock();
            let backlog_lsn = self.next_lsn.load(Ordering::Relaxed).saturating_sub(1);
            let backlog_records = backlog_lsn.saturating_sub(leader_target_lsn);
            let follower_waiters = state.flush_waiters.saturating_sub(1) as u64;
            if follower_waiters == 0 && backlog_records == 0 {
                return;
            }
            let extra_waiter_budget = follower_waiters.saturating_mul(50);
            let extra_backlog_budget = std::cmp::min(backlog_records, 8).saturating_mul(50);
            let delay = env_u64_us(
                "PAGEBOX_WAL_GROUP_COMMIT_DELAY_MIN_US",
                WAL_GROUP_COMMIT_DELAY_MIN_US,
            )
            .saturating_add(extra_waiter_budget)
            .saturating_add(extra_backlog_budget);
            std::cmp::min(delay, max_delay_us)
        };

        if initial_delay_us == 0 {
            return;
        }

        let min_deadline = Instant::now() + Duration::from_micros(initial_delay_us);
        let max_deadline = Instant::now() + Duration::from_micros(max_delay_us);
        let probe = Duration::from_micros(20);
        let mut last_backlog_lsn = leader_target_lsn;
        loop {
            let now = Instant::now();
            if now >= max_deadline {
                break;
            }
            let sleep_until = if now < min_deadline {
                min_deadline
            } else {
                max_deadline
            };
            let sleep_for = std::cmp::min(probe, sleep_until.saturating_duration_since(now));
            std::thread::sleep(sleep_for);
            let woke = Instant::now();

            let state = self.state.lock();
            let backlog_lsn = self.next_lsn.load(Ordering::Relaxed).saturating_sub(1);
            let backlog_records = backlog_lsn.saturating_sub(leader_target_lsn);
            let waiters = state.flush_waiters;
            drop(state);
            if backlog_records >= target_records {
                break;
            }
            if backlog_lsn <= leader_target_lsn {
                break;
            }
            if woke >= min_deadline && (waiters == 0 || backlog_lsn <= last_backlog_lsn) {
                break;
            }
            last_backlog_lsn = backlog_lsn;
        }
    }

    fn scan_valid_records(fd: std::os::fd::RawFd, data_size: u64) -> (Lsn, u64) {
        let mut max_lsn: Lsn = 0;
        let mut valid_data_bytes: u64 = 0;
        let mut meta_buf = AlignedBuf::new(WAL_RECORD_SIZE);
        let mut page_buf = AlignedBuf::new(WAL_RECORD_SIZE);
        let mut batch_offset = WAL_HEADER_SIZE as u64;
        let file_end = WAL_HEADER_SIZE as u64 + data_size;

        while batch_offset + WAL_RECORD_SIZE as u64 <= file_end {
            if pread_all(fd, meta_buf.as_mut_slice(), batch_offset as i64).is_err() {
                break;
            }
            let Some(count) = batch_meta_count(meta_buf.as_slice()) else {
                break;
            };
            let batch_len = WAL_RECORD_SIZE as u64 * (count as u64 + 1);
            if batch_offset + batch_len > file_end {
                break;
            }
            let mut batch_valid = true;
            for idx in 0..count {
                let page_offset = batch_offset + WAL_RECORD_SIZE as u64 * (idx as u64 + 1);
                if pread_all(fd, page_buf.as_mut_slice(), page_offset as i64).is_err() {
                    batch_valid = false;
                    break;
                }
                let entry = read_batch_entry(meta_buf.as_slice(), idx);
                if validate_entry_data(entry, page_buf.as_slice()).is_none() {
                    batch_valid = false;
                    break;
                }
                if entry.kind == RECORD_KIND_PAGE_IMAGE
                    || entry.kind == RECORD_KIND_LOGICAL_PACKED
                    || entry.flags & LOGICAL_FLAG_LAST == LOGICAL_FLAG_LAST
                {
                    max_lsn = max_lsn.max(entry.lsn);
                }
            }
            if !batch_valid {
                break;
            }
            valid_data_bytes = batch_offset + batch_len - WAL_HEADER_SIZE as u64;
            batch_offset += batch_len;
        }

        (max_lsn, valid_data_bytes)
    }

    fn should_sync_relaxed(&self, written_lsn: Lsn, last_sync: Instant) -> bool {
        let durable = self.durable_lsn.load(Ordering::Acquire);
        if written_lsn <= durable {
            return false;
        }
        if written_lsn.saturating_sub(durable) >= WAL_RELAXED_SYNC_RECORDS {
            return true;
        }
        last_sync.elapsed()
            >= Duration::from_micros(env_u64_us(
                "PAGEBOX_WAL_RELAXED_SYNC_INTERVAL_US",
                WAL_RELAXED_SYNC_INTERVAL_US,
            ))
    }

    fn should_write_relaxed(&self, state: &WalState, last_write: Instant) -> bool {
        if state.active.records == 0 {
            return false;
        }
        if state.active.records >= WAL_RELAXED_WRITE_RECORDS {
            return true;
        }
        last_write.elapsed()
            >= Duration::from_micros(env_u64_us(
                "PAGEBOX_WAL_RELAXED_WRITE_INTERVAL_US",
                WAL_RELAXED_WRITE_INTERVAL_US,
            ))
    }

    fn writer_loop(&self) {
        let mut last_write = Instant::now();
        loop {
            let mut state = self.state.lock();
            loop {
                let relaxed_mode =
                    self.commit_mode.load(Ordering::Acquire) == CommitMode::Relaxed as u64;
                let written = self.written_lsn.load(Ordering::Acquire);
                let requested_write = self.requested_write_lsn.load(Ordering::Acquire);
                let pending_write = !state.pending_writes.is_empty()
                    || requested_write > written
                    || (relaxed_mode && self.should_write_relaxed(&state, last_write))
                    || (state.shutdown && state.active.used > 0);
                if state.crash_shutdown {
                    return;
                }
                if state.shutdown
                    && !pending_write
                    && state.active.used == 0
                    && state.writes_in_progress == 0
                {
                    return;
                }
                if pending_write {
                    break;
                }
                if relaxed_mode && state.active.used > 0 {
                    self.flush_requested.wait_for(
                        &mut state,
                        Duration::from_micros(env_u64_us(
                            "PAGEBOX_WAL_RELAXED_WRITE_INTERVAL_US",
                            WAL_RELAXED_WRITE_INTERVAL_US,
                        )),
                    );
                } else {
                    self.flush_requested.wait(&mut state);
                }
            }
            let relaxed_mode =
                self.commit_mode.load(Ordering::Acquire) == CommitMode::Relaxed as u64;
            let requested_write = self.requested_write_lsn.load(Ordering::Acquire);
            let written = self.written_lsn.load(Ordering::Acquire);
            if !relaxed_mode
                && requested_write > written
                && state.pending_writes.is_empty()
                && state.active.used > 0
            {
                // Batch active appends before sealing; otherwise strict commits
                // can degrade into many tiny writes followed by frequent syncs.
                drop(state);
                self.maybe_group_commit_delay(requested_write);
                state = self.state.lock();
            }

            let requested_write = self.requested_write_lsn.load(Ordering::Acquire);
            let written = self.written_lsn.load(Ordering::Acquire);
            let should_write = requested_write > written
                || !state.pending_writes.is_empty()
                || (relaxed_mode && self.should_write_relaxed(&state, last_write));
            if should_write && state.pending_writes.is_empty() && state.active.used > 0 {
                self.seal_active_buffer_locked(&mut state)
                    .expect("WAL buffer seal failed");
            }
            if should_write && let Some(write) = self.pop_pending_write_locked(&mut state) {
                drop(state);
                self.write_sealed_buffer(write).expect("WAL write failed");
                last_write = Instant::now();
            }
        }
    }

    fn syncer_loop(&self) {
        let mut last_sync = Instant::now();
        loop {
            let mut state = self.state.lock();
            loop {
                let relaxed_mode =
                    self.commit_mode.load(Ordering::Acquire) == CommitMode::Relaxed as u64;
                let durable = self.durable_lsn.load(Ordering::Acquire);
                let written = self.written_lsn.load(Ordering::Acquire);
                let requested_durable = self.requested_durable_lsn.load(Ordering::Acquire);
                if requested_durable > written {
                    self.requested_write_lsn
                        .fetch_max(requested_durable, Ordering::Release);
                    self.flush_requested.notify_all();
                }

                let pending_sync = written > durable
                    && (requested_durable > durable
                        || (relaxed_mode && self.should_sync_relaxed(written, last_sync)));
                if state.crash_shutdown {
                    return;
                }
                if state.shutdown && requested_durable <= durable && !pending_sync {
                    return;
                }
                if pending_sync {
                    break;
                }
                if relaxed_mode && written > durable {
                    self.flush_requested.wait_for(
                        &mut state,
                        Duration::from_micros(env_u64_us(
                            "PAGEBOX_WAL_RELAXED_SYNC_INTERVAL_US",
                            WAL_RELAXED_SYNC_INTERVAL_US,
                        )),
                    );
                } else {
                    self.flush_requested.wait(&mut state);
                }
            }

            let leader_target_lsn = self.requested_durable_lsn.load(Ordering::Acquire);
            drop(state);
            self.maybe_group_commit_delay(leader_target_lsn);

            let state = self.state.lock();
            let durable = self.durable_lsn.load(Ordering::Acquire);
            let written = self.written_lsn.load(Ordering::Acquire);
            let requested_durable = self.requested_durable_lsn.load(Ordering::Acquire);
            if requested_durable > written {
                self.requested_write_lsn
                    .fetch_max(requested_durable, Ordering::Release);
                self.flush_requested.notify_all();
                continue;
            }
            let relaxed_mode =
                self.commit_mode.load(Ordering::Acquire) == CommitMode::Relaxed as u64;
            let need_sync = written > durable
                && (requested_durable > durable
                    || (relaxed_mode && self.should_sync_relaxed(written, last_sync)));
            if !need_sync {
                continue;
            }
            // Writes can continue while fdatasync is in flight. Only the LSNs
            // written before this sync starts are reported durable here.
            let synced_lsn = written;
            let fd = state.fd;
            drop(state);

            self.sync_fd(fd);
            last_sync = Instant::now();
            self.advance_durable_lsn(synced_lsn);
            self.flush_done.notify_all();
            self.flush_requested.notify_all();
        }
    }

    fn sync_fd(&self, fd: std::os::fd::RawFd) {
        self.sync_fd_blocking(fd);
    }

    fn sync_fd_blocking(&self, fd: std::os::fd::RawFd) {
        self.stats.events.inc(WalEvent::SyncCall);
        let sync_start = Instant::now();
        let ret = sync_wal_fd(fd);
        self.stats
            .latencies
            .get(WalLatency::Sync)
            .record(saturating_duration_nanos(sync_start.elapsed()));
        if let Err(err) = ret {
            panic!("WAL sync failed — durability compromised: {err}");
        }
    }

    fn advance_lsn_past(&self, min_lsn: Lsn) {
        self.next_lsn.fetch_max(min_lsn + 1, Ordering::Relaxed);
        self.appended_lsn.fetch_max(min_lsn, Ordering::Relaxed);
        self.written_lsn.fetch_max(min_lsn, Ordering::Relaxed);
        self.durable_lsn.fetch_max(min_lsn, Ordering::Relaxed);
        self.requested_write_lsn
            .fetch_max(min_lsn, Ordering::Relaxed);
        self.requested_durable_lsn
            .fetch_max(min_lsn, Ordering::Relaxed);
    }

    fn advance_durable_lsn(&self, lsn: Lsn) {
        let previous = self.durable_lsn.fetch_max(lsn, Ordering::Release);
        if lsn > previous {
            self.stats.events.inc(WalEvent::DurableAdvance);
        }
    }

    fn next_buffer_epoch_locked(state: &mut WalState) -> u64 {
        let epoch = state.next_buffer_epoch;
        state.next_buffer_epoch = state.next_buffer_epoch.wrapping_add(1).max(1);
        epoch
    }

    fn empty_buffer_locked(state: &mut WalState) -> WalBuffer {
        let epoch = Self::next_buffer_epoch_locked(state);
        if let Some(mut buffer) = state.spare_buffers.pop() {
            buffer.reset(epoch);
            return buffer;
        }
        WalBuffer::new(epoch)
    }

    fn seal_active_buffer_locked(&self, state: &mut WalState) -> io::Result<bool> {
        if state.active.used == 0 {
            return Ok(false);
        }

        let len = state.active.used;
        let max_lsn = state.active.max_lsn;
        let file_offset = state.file_offset;
        let write_end = file_offset + len as u64;
        if write_end > state.allocated_size {
            // Pre-extend by multiple segments to amortize the ftruncate
            // syscall across seals. ftruncate creates a sparse file so
            // the extra space costs no disk — it just extends the inode
            // size, avoiding frequent inode RWSEM contention under
            // concurrent writes.
            const WAL_PREALLOCATE_SEGMENTS: u64 = 8;
            let new_size = round_up_u64(write_end, SEGMENT_SIZE)
                + SEGMENT_SIZE * (WAL_PREALLOCATE_SEGMENTS - 1);
            extend_file(state.fd, new_size)?;
            if self.sync_backend == WalSyncBackend::Pwritev2Dsync {
                fdatasync_file(state.fd)?;
            }
            state.allocated_size = new_size;
        }
        state.file_offset = write_end;

        let new_active = Self::empty_buffer_locked(state);
        let buffer = std::mem::replace(&mut state.active, new_active);
        state.pending_writes.push_back(PendingWalWrite {
            buffer,
            fd: state.fd,
            file_offset,
            len,
            max_lsn,
        });
        Ok(true)
    }

    fn pop_pending_write_locked(&self, state: &mut WalState) -> Option<PendingWalWrite> {
        let write = state.pending_writes.pop_front()?;
        state.writes_in_progress += 1;
        Some(write)
    }

    fn write_sealed_buffer(&self, mut write: PendingWalWrite) -> io::Result<()> {
        finalize_buffer_records(&mut write.buffer, write.len);

        self.stats.events.inc(WalEvent::WriteCall);
        self.stats.events.add(
            WalEvent::WriteBytes,
            write.len.min(isize::MAX as usize) as isize,
        );
        let write_start = Instant::now();
        let data = &write.buffer.buffer.as_slice()[..write.len];
        match self.sync_backend {
            WalSyncBackend::Fdatasync => pwrite_all(write.fd, data, write.file_offset as i64)?,
            WalSyncBackend::Pwritev2Dsync => {
                pwritev2_dsync_all(write.fd, data, write.file_offset as i64)?;
            }
        }
        self.stats
            .latencies
            .get(WalLatency::Write)
            .record(saturating_duration_nanos(write_start.elapsed()));

        self.written_lsn.fetch_max(write.max_lsn, Ordering::Release);
        if self.sync_backend == WalSyncBackend::Pwritev2Dsync {
            self.advance_durable_lsn(write.max_lsn);
        }

        let mut state = self.state.lock();
        write
            .buffer
            .reset(Self::next_buffer_epoch_locked(&mut state));
        state.spare_buffers.push(write.buffer);
        state.writes_in_progress -= 1;
        self.flush_done.notify_all();
        self.flush_requested.notify_all();

        Ok(())
    }

    fn replay<F>(&self, mut f: F) -> io::Result<()>
    where
        F: FnMut(Lsn, PageId, &[u8; PAGE_SIZE]),
    {
        self.replay_records(|record| {
            if let WalReplayRecord::PageImage { lsn, page_id, data } = record
                && let Ok(page) = <&[u8; PAGE_SIZE]>::try_from(data)
            {
                f(lsn, page_id, page);
            }
        })
    }

    fn replay_records<F>(&self, mut f: F) -> io::Result<()>
    where
        F: FnMut(WalReplayRecord<'_>),
    {
        let state = self.state.lock();
        let end = state.file_offset;
        let fd = state.fd;
        let mut meta_buf = AlignedBuf::new(WAL_RECORD_SIZE);
        let mut page_buf = AlignedBuf::new(WAL_RECORD_SIZE);
        let mut logical_lsn = 0;
        let mut logical_kind = 0;
        let mut logical_payload = Vec::new();
        let mut batch_offset = WAL_HEADER_SIZE as u64;

        while batch_offset + WAL_RECORD_SIZE as u64 <= end {
            pread_all(fd, meta_buf.as_mut_slice(), batch_offset as i64)?;
            let Some(count) = batch_meta_count(meta_buf.as_slice()) else {
                break;
            };
            let batch_len = WAL_RECORD_SIZE as u64 * (count as u64 + 1);
            if batch_offset + batch_len > end {
                break;
            }
            for idx in 0..count {
                let offset = batch_offset + WAL_RECORD_SIZE as u64 * (idx as u64 + 1);
                pread_all(fd, page_buf.as_mut_slice(), offset as i64)?;
                let entry = read_batch_entry(meta_buf.as_slice(), idx);
                let Some(payload) = validate_entry_data(entry, page_buf.as_slice()) else {
                    return Ok(());
                };
                match entry.kind {
                    RECORD_KIND_PAGE_IMAGE => {
                        if !logical_payload.is_empty() {
                            logical_payload.clear();
                            logical_lsn = 0;
                            logical_kind = 0;
                        }
                        f(WalReplayRecord::PageImage {
                            lsn: entry.lsn,
                            page_id: entry.arg,
                            data: payload,
                        });
                    }
                    RECORD_KIND_LOGICAL => {
                        let first = entry.flags & LOGICAL_FLAG_FIRST == LOGICAL_FLAG_FIRST;
                        let last = entry.flags & LOGICAL_FLAG_LAST == LOGICAL_FLAG_LAST;
                        if first {
                            if !logical_payload.is_empty() {
                                logical_payload.clear();
                            }
                            logical_lsn = entry.lsn;
                            logical_kind = entry.arg;
                        } else if logical_payload.is_empty()
                            || entry.lsn != logical_lsn
                            || entry.arg != logical_kind
                        {
                            return Ok(());
                        }
                        logical_payload.extend_from_slice(payload);
                        if last {
                            if logical_kind == LOGICAL_KIND_PAGE_IMAGE_BYTES {
                                let Some((page_id, page)) =
                                    decode_page_image_bytes_payload(&logical_payload)
                                else {
                                    return Ok(());
                                };
                                f(WalReplayRecord::PageImage {
                                    lsn: logical_lsn,
                                    page_id,
                                    data: page,
                                });
                            } else {
                                f(WalReplayRecord::Logical {
                                    lsn: logical_lsn,
                                    kind: logical_kind,
                                    payload: &logical_payload,
                                });
                            }
                            logical_payload.clear();
                            logical_lsn = 0;
                            logical_kind = 0;
                        }
                    }
                    RECORD_KIND_LOGICAL_PACKED => {
                        if !logical_payload.is_empty() {
                            logical_payload.clear();
                            logical_lsn = 0;
                            logical_kind = 0;
                        }
                        if !for_each_packed_logical_record(payload, |lsn, kind, payload| {
                            f(WalReplayRecord::Logical { lsn, kind, payload });
                        }) {
                            return Ok(());
                        }
                    }
                    _ => return Ok(()),
                }
            }
            batch_offset += batch_len;
        }

        Ok(())
    }

    fn recover<S, F>(
        &self,
        store: &S,
        checkpoint_lsn: Lsn,
        read_page_lsn: F,
    ) -> io::Result<RecoveryReport>
    where
        S: RecoveryPageStore + ?Sized,
        F: Fn(&[u8]) -> Lsn,
    {
        let state = self.state.lock();
        let end = state.file_offset;
        let fd = state.fd;
        let mut report = RecoveryReport::default();
        let mut meta_buf = AlignedBuf::new(WAL_RECORD_SIZE);
        let mut read_buf = AlignedBuf::new(WAL_RECORD_SIZE);
        let mut page_buf = Vec::new();
        let mut logical_lsn = 0;
        let mut logical_kind = 0;
        let mut logical_payload = Vec::new();
        let mut current_next = store.next_page_id();
        let mut batch_offset = WAL_HEADER_SIZE as u64;
        let recovery = PageImageRecovery {
            store,
            checkpoint_lsn,
            read_page_lsn: &read_page_lsn,
        };

        while batch_offset + WAL_RECORD_SIZE as u64 <= end {
            pread_all(fd, meta_buf.as_mut_slice(), batch_offset as i64)?;
            let Some(count) = batch_meta_count(meta_buf.as_slice()) else {
                break;
            };
            let batch_len = WAL_RECORD_SIZE as u64 * (count as u64 + 1);
            if batch_offset + batch_len > end {
                break;
            }
            for idx in 0..count {
                let offset = batch_offset + WAL_RECORD_SIZE as u64 * (idx as u64 + 1);
                pread_all(fd, read_buf.as_mut_slice(), offset as i64)?;
                let entry = read_batch_entry(meta_buf.as_slice(), idx);
                let Some(payload) = validate_entry_data(entry, read_buf.as_slice()) else {
                    return Ok(report);
                };
                report.records_scanned += 1;
                if entry.kind == RECORD_KIND_PAGE_IMAGE
                    || entry.kind == RECORD_KIND_LOGICAL_PACKED
                    || entry.flags & LOGICAL_FLAG_LAST == LOGICAL_FLAG_LAST
                {
                    report.max_lsn = report.max_lsn.max(entry.lsn);
                }

                match entry.kind {
                    RECORD_KIND_PAGE_IMAGE => {
                        if !logical_payload.is_empty() {
                            logical_payload.clear();
                            logical_lsn = 0;
                            logical_kind = 0;
                        }
                        recover_page_image(
                            &recovery,
                            &mut report,
                            &mut current_next,
                            &mut page_buf,
                            PageImage {
                                lsn: entry.lsn,
                                pid: entry.arg,
                                data: payload,
                            },
                        )?;
                    }
                    RECORD_KIND_LOGICAL => {
                        let first = entry.flags & LOGICAL_FLAG_FIRST == LOGICAL_FLAG_FIRST;
                        let last = entry.flags & LOGICAL_FLAG_LAST == LOGICAL_FLAG_LAST;
                        if first {
                            if !logical_payload.is_empty() {
                                logical_payload.clear();
                            }
                            logical_lsn = entry.lsn;
                            logical_kind = entry.arg;
                        } else if logical_payload.is_empty()
                            || entry.lsn != logical_lsn
                            || entry.arg != logical_kind
                        {
                            return Ok(report);
                        }
                        logical_payload.extend_from_slice(payload);
                        if last {
                            if !recover_logical_payload(
                                &recovery,
                                &mut report,
                                &mut current_next,
                                &mut page_buf,
                                logical_lsn,
                                logical_kind,
                                &logical_payload,
                            )? {
                                return Ok(report);
                            }
                            logical_payload.clear();
                            logical_lsn = 0;
                            logical_kind = 0;
                        }
                    }
                    RECORD_KIND_LOGICAL_PACKED => {
                        if !logical_payload.is_empty() {
                            logical_payload.clear();
                            logical_lsn = 0;
                            logical_kind = 0;
                        }
                        let mut records = Vec::new();
                        if !for_each_packed_logical_record(payload, |lsn, kind, payload| {
                            records.push((lsn, kind, payload.to_vec()));
                        }) {
                            return Ok(report);
                        }
                        for (lsn, kind, payload) in records {
                            if !recover_logical_payload(
                                &recovery,
                                &mut report,
                                &mut current_next,
                                &mut page_buf,
                                lsn,
                                kind,
                                &payload,
                            )? {
                                return Ok(report);
                            }
                        }
                    }
                    _ => return Ok(report),
                }
            }
            batch_offset += batch_len;
        }

        if report.records_applied > 0 {
            store.sync()?;
        }

        Ok(report)
    }

    fn reset(&self) -> io::Result<()> {
        let mut state = self.state.lock();
        if unsafe { libc::ftruncate(state.fd, 0) } != 0 {
            return Err(io::Error::last_os_error());
        }

        let hdr = build_wal_header(state.direct_io);
        let mut hdr_buf = AlignedBuf::new(WAL_HEADER_SIZE);
        hdr_buf.as_mut_slice().copy_from_slice(&hdr);
        pwrite_all(state.fd, hdr_buf.as_slice(), 0)?;

        state.file_offset = WAL_HEADER_SIZE as u64;
        let epoch = Self::next_buffer_epoch_locked(&mut state);
        state.active.reset(epoch);
        state.pending_writes.clear();
        state.spare_buffers.clear();
        let spare_epoch = Self::next_buffer_epoch_locked(&mut state);
        state.spare_buffers.push(WalBuffer::new(spare_epoch));

        extend_file(state.fd, SEGMENT_SIZE)?;
        state.allocated_size = SEGMENT_SIZE;

        Ok(())
    }
}

fn finalize_buffer_records(buffer: &mut WalBuffer, len: usize) {
    let mut offset = 0usize;
    while offset < len {
        let meta_end = offset + WAL_RECORD_SIZE;
        let count = batch_meta_count_unchecked(&buffer.buffer.as_slice()[offset..meta_end])
            .expect("buffered batch metadata must be valid");
        for idx in 0..count {
            let page_offset = offset + WAL_RECORD_SIZE * (idx + 1);
            let page_end = page_offset + WAL_RECORD_SIZE;
            let crc = {
                let entry = read_batch_entry(&buffer.buffer.as_slice()[offset..meta_end], idx);
                let data = &buffer.buffer.as_slice()[page_offset..page_end];
                match entry.kind {
                    RECORD_KIND_PAGE_IMAGE => {
                        let page: &[u8; PAGE_SIZE] = data
                            .try_into()
                            .expect("buffered page slot must be page-sized");
                        page_crc(page)
                    }
                    RECORD_KIND_LOGICAL | RECORD_KIND_LOGICAL_PACKED => {
                        payload_crc(&data[..entry.len as usize])
                    }
                    _ => panic!("unknown WAL record kind {}", entry.kind),
                }
            };
            let meta = &mut buffer.buffer.as_mut_slice()[offset..meta_end];
            overwrite_batch_entry_crc(meta, idx, crc);
        }
        finalize_batch_meta(&mut buffer.buffer.as_mut_slice()[offset..meta_end]);
        offset = meta_end + count * WAL_RECORD_SIZE;
    }
    debug_assert_eq!(offset, len);
}
