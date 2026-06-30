//! On-disk WAL record format, batch layout, checksums, and runtime knobs.
//!
//! This module is the single source of truth for the byte layout of a WAL
//! shard file. The higher-level [`Wal`](crate::Wal) lives in
//! [`crate::wal_impl`] and calls into these helpers to encode and decode
//! records; nothing here is stateful.
//!
//! ## File layout
//!
//! ```text
//!   ┌─────────┬───────────────────────────┬───────────────────────────┬─────┐
//!   │ header  │ batch-meta page           │ data page × N            │ ... │
//!   │PAGE_SIZE│  (BATCH_META_HEADER_SIZE) │  (one per BatchEntry)    │     │
//!   └─────────┴───────────────────────────┴───────────────────────────┴─────┘
//! ```
//!
//! The file header (one `PAGE_SIZE` block) carries `WAL_MAGIC` (`"BWAL"`),
//! a `WAL_VERSION`, a `WAL_FLAG_DIRECT_IO` bit, and the on-disk
//! `WAL_RECORD_SIZE` so a future tool can refuse an incompatible file. Each
//! *batch* is `1 + N` pages: a *batch-meta page* holding an array of
//! `BATCH_ENTRY_SIZE`-byte `BatchEntry` records (LSN, arg, CRC, len, kind,
//! flags), followed by up to `BATCH_MAX_RECORDS` *data pages* — one per
//! entry. The number of data pages actually used is stored in
//! `BATCH_META_COUNT_OFF`; the meta page's own CRC32 covers everything except
//! the CRC field itself.
//!
//! ## Record kinds
//!
//! `BatchEntry::kind` discriminates three record kinds:
//!
//! - `RECORD_KIND_PAGE_IMAGE` — a full page image. `arg` is the
//!   [`PageId`](pagebox_frame_kernel::PageId); the data page is the page's
//!   bytes verbatim.
//! - `RECORD_KIND_LOGICAL` — one chunk of a (possibly multi-chunk) logical
//!   record. `arg` is the caller's logical kind; `flags` carries
//!   `LOGICAL_FLAG_FIRST` / `LOGICAL_FLAG_LAST`; `len` is the chunk payload
//!   length (≤ `LOGICAL_CHUNK_MAX_LEN`). A logical record may have 1..N
//!   chunks depending on payload size.
//! - `RECORD_KIND_LOGICAL_PACKED` — a short logical record packed alongside
//!   others into a single data page (header + payload ≤ one page). Used for
//!   the common case where many small logical records fit together.
//!
//! ## Checksums and truncated tails
//!
//! Each batch's meta page is CRC32 (ISO-HDLC, via `crc-fast`); a torn tail
//! at the end of a shard file (e.g. a crash mid-write) is detected by either
//! a bad CRC, a missing magic, a zero count, or a stored count exceeding
//! `BATCH_MAX_RECORDS`. On open, the WAL truncates such a tail in place so
//! subsequent appends start from the last good batch.
//!
//! ## Buffer sizing
//!
//! `WAL_BUF_CAPACITY` (64 MiB) is the in-memory append buffer.
//! `wal_buf_records` derives the maximum number of records that fit:
//! `(BATCH_MAX_RECORDS + 1) × PAGE_SIZE` per full batch, plus a partial final
//! batch. The re-exported [`WAL_BUF_RECORDS`] constant is the public cap
//! callers use to size structures that mirror the WAL's record index.
//!
//! ## Runtime knobs
//!
//! `env_u64_us` parses a fixed allow-list of `PAGEBOX_WAL_*` environment
//! variables (group-commit delay / target, relaxed-mode intervals and
//! thresholds) once on first access; the table is cached for the process
//! lifetime. The defaults are summarised in the [crate-level docs](crate)
//! alongside the sync-backend semantics.

use std::collections::HashMap;
use std::io;
use std::sync::OnceLock;

use crc_fast::{CrcAlgorithm::Crc32IsoHdlc, Digest};
use pagebox_frame_kernel::{Lsn, PAGE_SIZE, PageId};

/// On-disk WAL block size.
pub(crate) const WAL_RECORD_SIZE: usize = PAGE_SIZE;

/// File header occupies the first WAL block.
pub(crate) const WAL_HEADER_SIZE: usize = WAL_RECORD_SIZE;

pub(crate) const WAL_MAGIC: u32 = 0x4257414C; // "BWAL"
pub(crate) const WAL_VERSION: u16 = 4;
pub(crate) const WAL_FLAG_DIRECT_IO: u16 = 1 << 0;

pub(crate) const HDR_MAGIC_OFF: usize = 0;
pub(crate) const HDR_VERSION_OFF: usize = 4;
pub(crate) const HDR_FLAGS_OFF: usize = 6;
pub(crate) const HDR_RECORD_SIZE_OFF: usize = 8;

pub(crate) const BATCH_META_MAGIC: u32 = 0x424D4554; // "BMET"
pub(crate) const BATCH_META_MAGIC_OFF: usize = 0;
pub(crate) const BATCH_META_VERSION_OFF: usize = 4;
pub(crate) const BATCH_META_COUNT_OFF: usize = 6;
pub(crate) const BATCH_META_CRC_OFF: usize = 8;
pub(crate) const BATCH_META_HEADER_SIZE: usize = 16;

pub(crate) const BATCH_ENTRY_LSN_OFF: usize = 0;
pub(crate) const BATCH_ENTRY_ARG_OFF: usize = 8;
pub(crate) const BATCH_ENTRY_CRC_OFF: usize = 16;
pub(crate) const BATCH_ENTRY_LEN_OFF: usize = 20;
pub(crate) const BATCH_ENTRY_KIND_OFF: usize = 24;
pub(crate) const BATCH_ENTRY_FLAGS_OFF: usize = 25;
pub(crate) const BATCH_ENTRY_SIZE: usize = 28;
pub(crate) const BATCH_MAX_RECORDS: usize = (PAGE_SIZE - BATCH_META_HEADER_SIZE) / BATCH_ENTRY_SIZE;
const _: () = assert!(BATCH_META_HEADER_SIZE + BATCH_MAX_RECORDS * BATCH_ENTRY_SIZE <= PAGE_SIZE);

pub(crate) const RECORD_KIND_PAGE_IMAGE: u8 = 1;
pub(crate) const RECORD_KIND_LOGICAL: u8 = 2;
pub(crate) const RECORD_KIND_LOGICAL_PACKED: u8 = 3;
pub(crate) const LOGICAL_FLAG_FIRST: u8 = 1 << 0;
pub(crate) const LOGICAL_FLAG_LAST: u8 = 1 << 1;
pub(crate) const LOGICAL_CHUNK_MAX_LEN: usize = PAGE_SIZE;
pub(crate) const PACKED_LOGICAL_ENTRY_HEADER_LEN: usize = 20;
pub(crate) const PACKED_LOGICAL_MAX_PAYLOAD_LEN: usize =
    PAGE_SIZE - PACKED_LOGICAL_ENTRY_HEADER_LEN;

const WAL_BATCH_BYTES: usize = (BATCH_MAX_RECORDS + 1) * PAGE_SIZE;

pub(crate) const WAL_BUF_CAPACITY: usize = {
    let cap = WAL_BATCH_BYTES.next_power_of_two();
    assert!(cap.is_multiple_of(WAL_RECORD_SIZE));
    cap
};
const _: () = assert!(WAL_BUF_CAPACITY >= WAL_BATCH_BYTES);
const _: () = assert!(WAL_BUF_CAPACITY.is_multiple_of(WAL_RECORD_SIZE));

const fn wal_buf_records(capacity: usize) -> usize {
    let full_batch_bytes = (BATCH_MAX_RECORDS + 1) * PAGE_SIZE;
    let full_batches = capacity / full_batch_bytes;
    let mut records = full_batches * BATCH_MAX_RECORDS;
    let mut remaining = capacity - full_batches * full_batch_bytes;
    if remaining < 2 * PAGE_SIZE {
        return records;
    }
    remaining -= PAGE_SIZE;
    let extra = remaining / PAGE_SIZE;
    records += if extra > BATCH_MAX_RECORDS {
        BATCH_MAX_RECORDS
    } else {
        extra
    };
    records
}

/// Maximum number of page-image records that fit in the write buffer.
pub const WAL_BUF_RECORDS: usize = wal_buf_records(WAL_BUF_CAPACITY);

/// Pre-allocation segment size.
pub(crate) const SEGMENT_SIZE: u64 = 64 * 1024 * 1024;

/// Minimum alignment for O_DIRECT buffers and I/O.
pub(crate) const DIRECT_IO_ALIGN: usize = 4096;

/// Minimum leader delay to let obvious followers pile onto the same flush.
pub(crate) const WAL_GROUP_COMMIT_DELAY_MIN_US: u64 = 100;
/// Maximum leader delay under clear contention/backlog.
pub(crate) const WAL_GROUP_COMMIT_DELAY_MAX_US: u64 = 1000;
/// Target follower records to accumulate before ending group-commit delay.
pub(crate) const WAL_GROUP_COMMIT_TARGET_RECORDS: u64 = 256;
/// Fdatasync relies on natural concurrent batching by default; explicit delay
/// can be restored with PAGEBOX_WAL_FDATASYNC_DELAY_MAX_US.
pub(crate) const WAL_FDATASYNC_GROUP_COMMIT_DELAY_MAX_US: u64 = 0;
/// Fdatasync-specific follower-record target.
pub(crate) const WAL_FDATASYNC_GROUP_COMMIT_TARGET_RECORDS: u64 = 256;
/// Durable-write backends should not wait as aggressively as fdatasync.
pub(crate) const WAL_PWRITEV2_DSYNC_GROUP_COMMIT_DELAY_MAX_US: u64 = 250;
/// Durable-write backends use a smaller target to avoid delaying every write.
pub(crate) const WAL_PWRITEV2_DSYNC_GROUP_COMMIT_TARGET_RECORDS: u64 = 32;
/// Relaxed mode background sync interval.
pub(crate) const WAL_RELAXED_SYNC_INTERVAL_US: u64 = 50_000;
/// Relaxed mode background sync threshold in records.
pub(crate) const WAL_RELAXED_SYNC_RECORDS: u64 = 2_048;
/// Relaxed mode background write interval.
pub(crate) const WAL_RELAXED_WRITE_INTERVAL_US: u64 = 50_000;
/// Relaxed mode background write threshold in records.
pub(crate) const WAL_RELAXED_WRITE_RECORDS: usize = 4_096;

pub(crate) fn env_u64_us(name: &'static str, default: u64) -> u64 {
    static OVERRIDES: OnceLock<HashMap<&'static str, u64>> = OnceLock::new();
    let overrides = OVERRIDES.get_or_init(|| {
        let mut values = HashMap::new();
        for (key, fallback) in [
            (
                "PAGEBOX_WAL_RELAXED_SYNC_INTERVAL_US",
                WAL_RELAXED_SYNC_INTERVAL_US,
            ),
            (
                "PAGEBOX_WAL_RELAXED_WRITE_INTERVAL_US",
                WAL_RELAXED_WRITE_INTERVAL_US,
            ),
            (
                "PAGEBOX_WAL_GROUP_COMMIT_DELAY_MIN_US",
                WAL_GROUP_COMMIT_DELAY_MIN_US,
            ),
            (
                "PAGEBOX_WAL_GROUP_COMMIT_DELAY_MAX_US",
                WAL_GROUP_COMMIT_DELAY_MAX_US,
            ),
            (
                "PAGEBOX_WAL_GROUP_COMMIT_TARGET_RECORDS",
                WAL_GROUP_COMMIT_TARGET_RECORDS,
            ),
            (
                "PAGEBOX_WAL_FDATASYNC_DELAY_MAX_US",
                WAL_FDATASYNC_GROUP_COMMIT_DELAY_MAX_US,
            ),
            (
                "PAGEBOX_WAL_FDATASYNC_TARGET_RECORDS",
                WAL_FDATASYNC_GROUP_COMMIT_TARGET_RECORDS,
            ),
            (
                "PAGEBOX_WAL_PWRITEV2_DSYNC_DELAY_MAX_US",
                WAL_PWRITEV2_DSYNC_GROUP_COMMIT_DELAY_MAX_US,
            ),
            (
                "PAGEBOX_WAL_PWRITEV2_DSYNC_TARGET_RECORDS",
                WAL_PWRITEV2_DSYNC_GROUP_COMMIT_TARGET_RECORDS,
            ),
        ] {
            let value = std::env::var(key)
                .ok()
                .and_then(|raw| raw.parse::<u64>().ok())
                .unwrap_or(fallback);
            values.insert(key, value);
        }
        values
    });
    overrides.get(name).copied().unwrap_or(default)
}

pub(crate) fn payload_crc(payload: &[u8]) -> u32 {
    let mut digest = Digest::new(Crc32IsoHdlc);
    digest.update(payload);
    digest.finalize() as u32
}

pub(crate) fn page_crc(page: &[u8; PAGE_SIZE]) -> u32 {
    payload_crc(page)
}

fn batch_meta_crc(buf: &[u8; PAGE_SIZE]) -> u32 {
    let mut digest = Digest::new(Crc32IsoHdlc);
    digest.update(&buf[..BATCH_META_CRC_OFF]);
    digest.update(&[0u8; 4]);
    digest.update(&buf[BATCH_META_CRC_OFF + 4..]);
    digest.finalize() as u32
}

pub(crate) fn init_batch_meta(buf: &mut [u8]) {
    debug_assert!(buf.len() == PAGE_SIZE);
    buf.fill(0);
    buf[BATCH_META_MAGIC_OFF..BATCH_META_MAGIC_OFF + 4]
        .copy_from_slice(&BATCH_META_MAGIC.to_le_bytes());
    buf[BATCH_META_VERSION_OFF..BATCH_META_VERSION_OFF + 2]
        .copy_from_slice(&WAL_VERSION.to_le_bytes());
}

pub(crate) fn batch_meta_count_unchecked(buf: &[u8]) -> Option<usize> {
    if buf.len() != PAGE_SIZE {
        return None;
    }
    let magic = u32::from_le_bytes(
        buf[BATCH_META_MAGIC_OFF..BATCH_META_MAGIC_OFF + 4]
            .try_into()
            .ok()?,
    );
    if magic != BATCH_META_MAGIC {
        return None;
    }
    let version = u16::from_le_bytes(
        buf[BATCH_META_VERSION_OFF..BATCH_META_VERSION_OFF + 2]
            .try_into()
            .ok()?,
    );
    if version != WAL_VERSION {
        return None;
    }
    let count = u16::from_le_bytes(
        buf[BATCH_META_COUNT_OFF..BATCH_META_COUNT_OFF + 2]
            .try_into()
            .ok()?,
    ) as usize;
    if count == 0 || count > BATCH_MAX_RECORDS {
        return None;
    }
    Some(count)
}

pub(crate) fn batch_meta_count(buf: &[u8]) -> Option<usize> {
    let count = batch_meta_count_unchecked(buf)?;
    let stored = u32::from_le_bytes(
        buf[BATCH_META_CRC_OFF..BATCH_META_CRC_OFF + 4]
            .try_into()
            .ok()?,
    );
    if stored != batch_meta_crc(buf.try_into().ok()?) {
        return None;
    }
    Some(count)
}

pub(crate) fn set_batch_meta_count(buf: &mut [u8], count: usize) {
    debug_assert!(buf.len() == PAGE_SIZE);
    debug_assert!(count <= BATCH_MAX_RECORDS);
    buf[BATCH_META_COUNT_OFF..BATCH_META_COUNT_OFF + 2]
        .copy_from_slice(&(count as u16).to_le_bytes());
}

pub(crate) fn finalize_batch_meta(buf: &mut [u8]) {
    debug_assert!(buf.len() == PAGE_SIZE);
    let page: &[u8; PAGE_SIZE] = (&*buf).try_into().expect("batch meta must be page-sized");
    let crc = batch_meta_crc(page);
    buf[BATCH_META_CRC_OFF..BATCH_META_CRC_OFF + 4].copy_from_slice(&crc.to_le_bytes());
}

fn batch_entry_offset(idx: usize) -> usize {
    debug_assert!(idx < BATCH_MAX_RECORDS);
    BATCH_META_HEADER_SIZE + idx * BATCH_ENTRY_SIZE
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BatchEntry {
    pub(crate) lsn: Lsn,
    pub(crate) arg: u64,
    pub(crate) crc: u32,
    pub(crate) len: u32,
    pub(crate) kind: u8,
    pub(crate) flags: u8,
}

impl BatchEntry {
    pub(crate) fn page_image(lsn: Lsn, pid: PageId, crc: u32) -> Self {
        Self {
            lsn,
            arg: pid,
            crc,
            kind: RECORD_KIND_PAGE_IMAGE,
            flags: 0,
            len: PAGE_SIZE as u32,
        }
    }

    pub(crate) fn logical_chunk(
        lsn: Lsn,
        logical_kind: u64,
        crc: u32,
        flags: u8,
        len: usize,
    ) -> Self {
        Self {
            lsn,
            arg: logical_kind,
            crc,
            kind: RECORD_KIND_LOGICAL,
            flags,
            len: len as u32,
        }
    }

    pub(crate) fn packed_logical(max_lsn: Lsn, crc: u32, len: usize) -> Self {
        Self {
            lsn: max_lsn,
            arg: 0,
            crc,
            kind: RECORD_KIND_LOGICAL_PACKED,
            flags: 0,
            len: len as u32,
        }
    }
}

pub(crate) fn write_batch_entry(buf: &mut [u8], idx: usize, entry: BatchEntry) {
    debug_assert!(buf.len() == PAGE_SIZE);
    let off = batch_entry_offset(idx);
    buf[off + BATCH_ENTRY_LSN_OFF..off + BATCH_ENTRY_LSN_OFF + 8]
        .copy_from_slice(&entry.lsn.to_le_bytes());
    buf[off + BATCH_ENTRY_ARG_OFF..off + BATCH_ENTRY_ARG_OFF + 8]
        .copy_from_slice(&entry.arg.to_le_bytes());
    buf[off + BATCH_ENTRY_CRC_OFF..off + BATCH_ENTRY_CRC_OFF + 4]
        .copy_from_slice(&entry.crc.to_le_bytes());
    buf[off + BATCH_ENTRY_LEN_OFF..off + BATCH_ENTRY_LEN_OFF + 4]
        .copy_from_slice(&entry.len.to_le_bytes());
    buf[off + BATCH_ENTRY_KIND_OFF] = entry.kind;
    buf[off + BATCH_ENTRY_FLAGS_OFF] = entry.flags;
}

pub(crate) fn overwrite_batch_entry_lsn(buf: &mut [u8], idx: usize, lsn: Lsn) {
    debug_assert!(buf.len() == PAGE_SIZE);
    let off = batch_entry_offset(idx);
    buf[off + BATCH_ENTRY_LSN_OFF..off + BATCH_ENTRY_LSN_OFF + 8]
        .copy_from_slice(&lsn.to_le_bytes());
}

pub(crate) fn overwrite_batch_entry_crc(buf: &mut [u8], idx: usize, crc: u32) {
    debug_assert!(buf.len() == PAGE_SIZE);
    let off = batch_entry_offset(idx);
    buf[off + BATCH_ENTRY_CRC_OFF..off + BATCH_ENTRY_CRC_OFF + 4]
        .copy_from_slice(&crc.to_le_bytes());
}

pub(crate) fn read_batch_entry(buf: &[u8], idx: usize) -> BatchEntry {
    debug_assert!(buf.len() == PAGE_SIZE);
    let off = batch_entry_offset(idx);
    let lsn = u64::from_le_bytes(
        buf[off + BATCH_ENTRY_LSN_OFF..off + BATCH_ENTRY_LSN_OFF + 8]
            .try_into()
            .unwrap(),
    );
    let arg = u64::from_le_bytes(
        buf[off + BATCH_ENTRY_ARG_OFF..off + BATCH_ENTRY_ARG_OFF + 8]
            .try_into()
            .unwrap(),
    );
    let crc = u32::from_le_bytes(
        buf[off + BATCH_ENTRY_CRC_OFF..off + BATCH_ENTRY_CRC_OFF + 4]
            .try_into()
            .unwrap(),
    );
    let len = u32::from_le_bytes(
        buf[off + BATCH_ENTRY_LEN_OFF..off + BATCH_ENTRY_LEN_OFF + 4]
            .try_into()
            .unwrap(),
    );
    let kind = buf[off + BATCH_ENTRY_KIND_OFF];
    let flags = buf[off + BATCH_ENTRY_FLAGS_OFF];
    BatchEntry {
        lsn,
        arg,
        crc,
        len,
        kind,
        flags,
    }
}

/// Build a WAL file header.
pub(crate) fn build_wal_header(direct_io: bool) -> [u8; WAL_HEADER_SIZE] {
    let mut hdr = [0u8; WAL_HEADER_SIZE];
    hdr[HDR_MAGIC_OFF..HDR_MAGIC_OFF + 4].copy_from_slice(&WAL_MAGIC.to_le_bytes());
    hdr[HDR_VERSION_OFF..HDR_VERSION_OFF + 2].copy_from_slice(&WAL_VERSION.to_le_bytes());
    let flags: u16 = if direct_io { WAL_FLAG_DIRECT_IO } else { 0 };
    hdr[HDR_FLAGS_OFF..HDR_FLAGS_OFF + 2].copy_from_slice(&flags.to_le_bytes());
    hdr[HDR_RECORD_SIZE_OFF..HDR_RECORD_SIZE_OFF + 8]
        .copy_from_slice(&(WAL_RECORD_SIZE as u64).to_le_bytes());
    hdr
}

/// Validate a WAL file header. Returns an error describing what's wrong.
pub(crate) fn validate_wal_header(hdr: &[u8; WAL_HEADER_SIZE]) -> io::Result<()> {
    let magic = u32::from_le_bytes(hdr[HDR_MAGIC_OFF..HDR_MAGIC_OFF + 4].try_into().unwrap());
    if magic != WAL_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad WAL magic: expected 0x{WAL_MAGIC:08X}, got 0x{magic:08X}"),
        ));
    }
    let version = u16::from_le_bytes(
        hdr[HDR_VERSION_OFF..HDR_VERSION_OFF + 2]
            .try_into()
            .unwrap(),
    );
    if version != WAL_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported WAL version: expected {WAL_VERSION}, got {version}"),
        ));
    }
    let rec_size = u64::from_le_bytes(
        hdr[HDR_RECORD_SIZE_OFF..HDR_RECORD_SIZE_OFF + 8]
            .try_into()
            .unwrap(),
    );
    if rec_size != WAL_RECORD_SIZE as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("incompatible WAL record size: expected {WAL_RECORD_SIZE}, got {rec_size}"),
        ));
    }
    Ok(())
}
