//! Linux io_uring backend for the WAL write/sync path.
//!
//! `libc` 0.2 exposes the `io_uring_setup` / `io_uring_enter` /
//! `io_uring_register` syscall numbers but *not* the ring layout structs, so
//! this module hand-defines the kernel uapi (`linux/io_uring.h`) types and
//! constants and drives the ring via raw syscalls + `mmap`. This keeps the
//! crate's dependency set closed (no `io-uring` / `rustix` crate): only
//! `libc`, which the WAL already depends on, is used.
//!
//! ## Model
//!
//! [`IoUring`] owns the ring fd and two user-allocated regions: an SQE array
//! and a combined SQ/CQ ring structure supplied through
//! `IORING_SETUP_NO_MMAP`. Writes are submitted as `IORING_OP_WRITE` SQEs; an
//! `IORING_OP_FSYNC` SQE is linked to each write with `IOSQE_IO_LINK`, so the
//! fsync only fires after its write completes. SQE `user_data` encodes whether
//! the completion is a write or fsync and the slab slot holding the in-flight
//! `WalBuffer`.
//!
//! A dedicated reaper thread calls `io_uring_enter` to flush submissions and
//! wait for CQEs. Reaping a ready CQE is then an atomic CQ tail/head update;
//! the reaper dispatches it to the correct `WalInner` through an fd registry.
//!
//! v1 deliberately uses plain kernel polling (no `SQPOLL`, no registered
//! buffers/files); those are future optimisations requiring their own bench
//! evidence.

#![cfg(target_os = "linux")]

use std::collections::HashMap;
use std::io;
use std::os::fd::RawFd;
use std::sync::Arc;
use std::sync::Weak;
use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::Mutex;

use crate::wal_impl::{PendingWalWrite, WalBuffer};

// ---------------------------------------------------------------------------
// uapi definitions (linux/io_uring.h). `libc` gives us the syscall numbers
// but not these structs/consts, so they are hand-defined to match the kernel
// ABI exactly.
// ---------------------------------------------------------------------------

const IORING_ENTER_GETEVENTS: u32 = 1 << 0;

/// `IORING_SETUP_NO_MMAP` — the ring memory is user-allocated rather than
/// mapped from the ring fd. Available since Linux 6.5.
const IORING_SETUP_NO_MMAP: u32 = 1 << 14;

#[allow(dead_code)]
const IORING_OP_NOP: u8 = 0;
const IORING_OP_FSYNC: u8 = 3;
const IORING_OP_WRITE: u8 = 23;

const IOSQE_IO_LINK: u8 = 1 << 2;

const IORING_FSYNC_DATASYNC: u32 = 1 << 0;

/// Sentinel `user_data` for NOP SQEs used to wake the reaper thread from
/// `io_uring_enter(GETEVENTS)` during shutdown.
const NOP_USER_DATA: u64 = u64::MAX;

/// `struct io_uring_sqe` (64 bytes). Matches the kernel layout exactly.
#[repr(C)]
#[derive(Clone, Copy)]
struct IoUringSqe {
    opcode: u8,
    flags: u8,
    ioprio: u16,
    fd: i32,
    union_off: u64,
    addr: u64,
    len: u32,
    union_flags: u32,
    user_data: u64,
    buf_index: u16,
    personality: u16,
    splice_fd_in: i32,
    addr3: u64,
    __pad2: [u64; 1],
}

impl IoUringSqe {
    const fn zero() -> Self {
        Self {
            opcode: 0,
            flags: 0,
            ioprio: 0,
            fd: 0,
            union_off: 0,
            addr: 0,
            len: 0,
            union_flags: 0,
            user_data: 0,
            buf_index: 0,
            personality: 0,
            splice_fd_in: 0,
            addr3: 0,
            __pad2: [0; 1],
        }
    }
}

/// `struct io_uring_cqe` (16 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct IoUringCqe {
    user_data: u64,
    res: i32,
    flags: u32,
}

/// `struct io_sqring_offsets` (40 bytes). The `user_addr` field (at the
/// position of `resv2` in older kernels) is used as both input (address to
/// the user-allocated SQ ring memory when `IORING_SETUP_NO_MMAP` is set) and
/// output.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IoSqringOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    flags: u32,
    dropped: u32,
    array: u32,
    resv1: u32,
    user_addr: u64,
}

/// `struct io_cqring_offsets` (40 bytes). `user_addr` is the
/// user-allocated CQ ring memory when `IORING_SETUP_NO_MMAP` is set.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IoCqringOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    overflow: u32,
    cqes: u32,
    flags: u32,
    resv1: u32,
    user_addr: u64,
}

/// `struct io_uring_params` (120 bytes). Only the fields we read are named
/// precisely; the rest are reserved zeroed padding matching the kernel.
#[repr(C)]
#[derive(Clone, Copy)]
struct IoUringParams {
    sq_entries: u32,
    cq_entries: u32,
    flags: u32,
    sq_thread_cpu: u32,
    sq_thread_idle: u32,
    features: u32,
    wq_fd: u32,
    resv: [u32; 3],
    sq_off: IoSqringOffsets,
    cq_off: IoCqringOffsets,
}

impl IoUringParams {
    fn zero() -> Self {
        Self {
            sq_entries: 0,
            cq_entries: 0,
            flags: 0,
            sq_thread_cpu: 0,
            sq_thread_idle: 0,
            features: 0,
            wq_fd: 0,
            resv: [0; 3],
            sq_off: IoSqringOffsets::default(),
            cq_off: IoCqringOffsets::default(),
        }
    }
}

// `IORING_FEAT` bits are not needed for v1; only `params.sq_entries` and the
// ring offsets are consumed.
//
// ---------------------------------------------------------------------------
// IoUring ring handle
// ---------------------------------------------------------------------------

/// Packed `user_data`: bit 63 set ⇒ fsync completion; otherwise a write whose
/// slab slot index is in the low bits.
const USER_DATA_FSYNC: u64 = 1 << 63;
const USER_DATA_SLOT_MASK: u64 = !(1 << 63);

fn pack_write(slot: u32) -> u64 {
    slot as u64
}
fn pack_fsync(slot: u32) -> u64 {
    (slot as u64) | USER_DATA_FSYNC
}
fn is_fsync(user_data: u64) -> bool {
    (user_data & USER_DATA_FSYNC) != 0
}
fn slot_of(user_data: u64) -> u32 {
    (user_data & USER_DATA_SLOT_MASK) as u32
}

/// A submitted io_uring ring. Owns the ring fd and the user-allocated ring
/// memory (via `IORING_SETUP_NO_MMAP`).
pub(crate) struct IoUring {
    ring_fd: RawFd,
    // Anonymous mappings supplied to the kernel for the ring lifetime. These
    // must not come from the process allocator: closing an io_uring fd starts
    // asynchronous kernel teardown, so heap pages could otherwise be reused
    // while the kernel still holds them pinned.
    sq_region: *mut u8,
    sq_region_sz: usize,
    cq_region: *mut u8,
    cq_region_sz: usize,
    // Cached field addresses inside the ring memory.
    sqes_ptr: *mut IoUringSqe,
    sq_head: *mut AtomicU32,
    sq_tail: *mut AtomicU32,
    sq_mask: u32,
    sq_array: *mut AtomicU32,
    cq_head: *mut AtomicU32,
    cq_tail: *mut AtomicU32,
    cq_mask: u32,
    cq_cqes: *mut IoUringCqe,
    // Number of SQEs submitted (sq_tail advanced) but not yet flushed to the
    // kernel via io_uring_enter.
    pending_submissions: u32,
    sq_entries: u32,
}

unsafe impl Send for IoUring {}
unsafe impl Sync for IoUring {}

impl IoUring {
    fn page_size() -> io::Result<usize> {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return Err(io::Error::last_os_error());
        }
        usize::try_from(page_size)
            .ok()
            .filter(|size| size.is_power_of_two())
            .ok_or_else(|| io::Error::other("invalid system page size"))
    }

    fn page_align(size: usize, page_size: usize) -> io::Result<usize> {
        size.checked_add(page_size - 1)
            .map(|size| size & !(page_size - 1))
            .ok_or_else(|| io::Error::other("io_uring region size overflow"))
    }

    fn map_region(size: usize) -> io::Result<*mut u8> {
        let region = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if region == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(region.cast())
    }

    unsafe fn unmap_region(region: *mut u8, size: usize) {
        if !region.is_null() {
            unsafe { libc::munmap(region.cast(), size) };
        }
    }

    /// Create a ring with `entries` SQ slots. On a kernel without io_uring
    /// (`ENOSYS` from `io_uring_setup`) the error is surfaced so `Wal::open`
    /// fails clearly rather than silently degrading.
    ///
    /// Uses `IORING_SETUP_NO_MMAP` to avoid mapping the ring fd, which is
    /// broken on some nvidia-variant kernels. Anonymous mappings are handed to
    /// the kernel via the `sq_off.user_addr` / `cq_off.user_addr` fields. A
    /// preview `io_uring_setup` call determines the required geometry first.
    pub(crate) fn new(entries: u32) -> io::Result<Self> {
        // --- Preview call: discover the kernel's actual entry counts and
        // ring field offsets. We don't mmap this ring — just read params. ---
        let mut preview = IoUringParams::zero();
        let preview_fd = unsafe {
            libc::syscall(
                libc::SYS_io_uring_setup,
                entries,
                &mut preview as *mut IoUringParams,
            )
        };
        if preview_fd < 0 {
            return Err(io::Error::last_os_error());
        }
        unsafe { libc::close(preview_fd as RawFd) };

        let sq_entries = preview.sq_entries;
        let cq_entries = preview.cq_entries;
        let sq_ring_sz =
            (preview.sq_off.array as usize) + (sq_entries as usize) * std::mem::size_of::<u32>();
        let sqe_array_sz = (sq_entries as usize) * std::mem::size_of::<IoUringSqe>();
        let cq_ring_sz = (preview.cq_off.cqes as usize)
            + (cq_entries as usize) * std::mem::size_of::<IoUringCqe>();
        // The SQ array lives within the rings struct (at cq_off.user_addr),
        // so the CQ region must be large enough for both the CQ entries AND
        // the SQ array — whichever extends further.
        let rings_struct_sz = sq_ring_sz.max(cq_ring_sz);

        let page = Self::page_size()?;
        // SQ region: SQE array only (sq_off.user_addr points to SQEs).
        let sq_region_sz = Self::page_align(sqe_array_sz, page)?.max(page * 16);
        // CQ region: the rings struct (SQ ring fields + CQ ring fields + SQ
        // array). Generously over-allocated to absorb any kernel-version
        // variation in field offsets.
        let cq_region_sz = Self::page_align(rings_struct_sz, page)?.max(page * 32);

        // Anonymous mappings remain distinct from allocator-managed memory
        // while the kernel asynchronously tears down its pinned-page view.
        let sq_region = Self::map_region(sq_region_sz)?;
        let cq_region = match Self::map_region(cq_region_sz) {
            Ok(region) => region,
            Err(err) => {
                unsafe { Self::unmap_region(sq_region, sq_region_sz) };
                return Err(err);
            }
        };

        // --- Real call: IORING_SETUP_NO_MMAP. ---
        let mut params = IoUringParams::zero();
        params.flags = IORING_SETUP_NO_MMAP;
        params.sq_off.user_addr = sq_region as u64;
        params.cq_off.user_addr = cq_region as u64;
        let fd = unsafe {
            libc::syscall(
                libc::SYS_io_uring_setup,
                entries,
                &mut params as *mut IoUringParams,
            )
        };
        if fd < 0 {
            let e = io::Error::last_os_error();
            unsafe { Self::unmap_region(sq_region, sq_region_sz) };
            unsafe { Self::unmap_region(cq_region, cq_region_sz) };
            return Err(e);
        }
        let ring_fd = fd as RawFd;

        // The real setup must fit the preview-sized mappings. Keep this a
        // release-mode check because dereferencing mismatched geometry would
        // be memory-unsafe.
        let real_sqe_size = params.sq_entries as usize * std::mem::size_of::<IoUringSqe>();
        let real_sq_ring_size =
            params.sq_off.array as usize + params.sq_entries as usize * std::mem::size_of::<u32>();
        let real_cq_ring_size = params.cq_off.cqes as usize
            + params.cq_entries as usize * std::mem::size_of::<IoUringCqe>();
        let geometry_matches = params.sq_entries == sq_entries
            && params.cq_entries == cq_entries
            && real_sqe_size <= sq_region_sz
            && real_sq_ring_size.max(real_cq_ring_size) <= cq_region_sz;
        if !geometry_matches {
            unsafe { libc::close(ring_fd) };
            unsafe { Self::unmap_region(sq_region, sq_region_sz) };
            unsafe { Self::unmap_region(cq_region, cq_region_sz) };
            return Err(io::Error::other(
                "io_uring geometry changed between preview and real setup",
            ));
        }

        // --- Resolve field pointers within user-allocated memory. ---
        // With NO_MMAP: sq_off.user_addr = SQE array; cq_off.user_addr =
        // the combined ring struct (SQ ring fields + CQ ring fields together).
        let ring_base = cq_region;
        let sq_head = unsafe { ring_base.add(params.sq_off.head as usize) as *mut AtomicU32 };
        let sq_tail = unsafe { ring_base.add(params.sq_off.tail as usize) as *mut AtomicU32 };
        let sq_array = unsafe { ring_base.add(params.sq_off.array as usize) as *mut AtomicU32 };
        let sq_mask = unsafe { *(ring_base.add(params.sq_off.ring_mask as usize) as *const u32) };

        // SQEs are at sq_off.user_addr (the separate SQ region allocation).
        let sqes_ptr = sq_region as *mut IoUringSqe;

        let cq_head = unsafe { ring_base.add(params.cq_off.head as usize) as *mut AtomicU32 };
        let cq_tail = unsafe { ring_base.add(params.cq_off.tail as usize) as *mut AtomicU32 };
        let cq_mask = unsafe { *(ring_base.add(params.cq_off.ring_mask as usize) as *const u32) };
        let cq_cqes = unsafe { ring_base.add(params.cq_off.cqes as usize) as *mut IoUringCqe };

        Ok(Self {
            ring_fd,
            sq_region,
            sq_region_sz,
            cq_region,
            cq_region_sz,
            sqes_ptr,
            sq_head,
            sq_tail,
            sq_mask,
            sq_array,
            cq_head,
            cq_tail,
            cq_mask,
            cq_cqes,
            pending_submissions: 0,
            sq_entries: params.sq_entries,
        })
    }

    /// Number of SQ slots the ring was created with.
    #[allow(dead_code)]
    pub(crate) fn sq_entries(&self) -> u32 {
        self.sq_entries
    }

    fn available_submissions(&self) -> u32 {
        let tail = unsafe { (*self.sq_tail).load(Ordering::Relaxed) };
        let head = unsafe { (*self.sq_head).load(Ordering::Acquire) };
        self.sq_entries.saturating_sub(tail.wrapping_sub(head))
    }

    fn ensure_submission_space(&mut self, needed: u32) -> io::Result<()> {
        if needed > self.sq_entries {
            return Err(io::Error::other("io_uring submission exceeds ring size"));
        }
        if self.available_submissions() < needed {
            self.flush_and_enter(0)?;
        }
        if self.available_submissions() < needed {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "io_uring submission queue remained full",
            ));
        }
        Ok(())
    }

    /// Submit a `WRITE` SQE (single-buffer pwrite). The buffer must remain
    /// valid until the write's CQE is reaped. Returns `true` if the SQE was
    /// enqueued, `false` if the SQ ring is full (caller should `flush_and_enter`
    /// first).
    pub(crate) fn submit_write(
        &mut self,
        fd: RawFd,
        buf_ptr: *const u8,
        len: u32,
        offset: u64,
        user_data: u64,
        link: bool,
    ) -> io::Result<bool> {
        let tail = unsafe { (*self.sq_tail).load(Ordering::Relaxed) };
        let head = unsafe { (*self.sq_head).load(Ordering::Acquire) };
        if tail.wrapping_sub(head) >= self.sq_entries {
            return Ok(false);
        }
        let idx = (tail & self.sq_mask) as usize;
        let sqe = unsafe { &mut *self.sqes_ptr.add(idx) };
        *sqe = IoUringSqe::zero();
        sqe.opcode = IORING_OP_WRITE;
        sqe.fd = fd;
        sqe.addr = buf_ptr as u64;
        sqe.len = len;
        sqe.union_off = offset;
        sqe.user_data = user_data;
        if link {
            sqe.flags |= IOSQE_IO_LINK;
        }
        unsafe {
            (*self.sq_array.add(idx)).store(idx as u32, Ordering::Release);
        }
        unsafe {
            (*self.sq_tail).store(tail.wrapping_add(1), Ordering::Release);
        }
        self.pending_submissions = self.pending_submissions.wrapping_add(1);
        Ok(true)
    }

    /// Submit an `FSYNC` SQE. With `IORING_FSYNC_DATASYNC` it matches the
    /// `fdatasync` semantics the other backends use. Set `link=false` for a
    /// standalone fsync; the linked-write variant links the *preceding* write
    /// SQE instead (see `submit_writev` with `link=true`).
    pub(crate) fn submit_fsync(
        &mut self,
        fd: RawFd,
        user_data: u64,
        datasync: bool,
    ) -> io::Result<bool> {
        let tail = unsafe { (*self.sq_tail).load(Ordering::Relaxed) };
        let head = unsafe { (*self.sq_head).load(Ordering::Acquire) };
        if tail.wrapping_sub(head) >= self.sq_entries {
            return Ok(false);
        }
        let idx = (tail & self.sq_mask) as usize;
        let sqe = unsafe { &mut *self.sqes_ptr.add(idx) };
        *sqe = IoUringSqe::zero();
        sqe.opcode = IORING_OP_FSYNC;
        sqe.fd = fd;
        sqe.user_data = user_data;
        if datasync {
            sqe.union_flags = IORING_FSYNC_DATASYNC;
        }
        unsafe {
            (*self.sq_array.add(idx)).store(idx as u32, Ordering::Release);
        }
        unsafe {
            (*self.sq_tail).store(tail.wrapping_add(1), Ordering::Release);
        }
        self.pending_submissions = self.pending_submissions.wrapping_add(1);
        Ok(true)
    }

    /// Submit a `NOP` SQE. Generates a CQE immediately (no I/O), used to
    /// wake the reaper thread from `io_uring_enter(GETEVENTS)` during
    /// shutdown.
    pub(crate) fn submit_nop(&mut self, user_data: u64) -> io::Result<bool> {
        let tail = unsafe { (*self.sq_tail).load(Ordering::Relaxed) };
        let head = unsafe { (*self.sq_head).load(Ordering::Acquire) };
        if tail.wrapping_sub(head) >= self.sq_entries {
            return Ok(false);
        }
        let idx = (tail & self.sq_mask) as usize;
        let sqe = unsafe { &mut *self.sqes_ptr.add(idx) };
        *sqe = IoUringSqe::zero();
        sqe.opcode = IORING_OP_NOP;
        sqe.user_data = user_data;
        unsafe {
            (*self.sq_array.add(idx)).store(idx as u32, Ordering::Release);
        }
        unsafe {
            (*self.sq_tail).store(tail.wrapping_add(1), Ordering::Release);
        }
        self.pending_submissions = self.pending_submissions.wrapping_add(1);
        Ok(true)
    }
    /// completions. Returns the number of CQEs that became ready (best-effort;
    /// the caller reaps via [`IoUring::reap_one`]).
    pub(crate) fn flush_and_enter(&mut self, min_complete: u32) -> io::Result<()> {
        if self.pending_submissions == 0 && min_complete == 0 {
            return Ok(());
        }
        let to_submit = self.pending_submissions;
        let mut flags = 0u32;
        if min_complete > 0 {
            flags |= IORING_ENTER_GETEVENTS;
        }
        let ret = unsafe {
            libc::syscall(
                libc::SYS_io_uring_enter,
                self.ring_fd,
                to_submit,
                min_complete,
                flags,
                std::ptr::null::<libc::sigset_t>(),
                0usize,
            )
        };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }
        let consumed = u32::try_from(ret)
            .unwrap_or(u32::MAX)
            .min(self.pending_submissions);
        self.pending_submissions -= consumed;
        Ok(())
    }

    /// Reap one ready CQE. Returns `Some((user_data, res))` or `None` if the
    /// CQ is empty. Advances the CQ head so the kernel can reuse the slot.
    pub(crate) fn reap_one(&mut self) -> Option<(u64, i32)> {
        let head = unsafe { (*self.cq_head).load(Ordering::Acquire) };
        let tail = unsafe { (*self.cq_tail).load(Ordering::Acquire) };
        if head == tail {
            return None;
        }
        let idx = (head & self.cq_mask) as usize;
        let cqe = unsafe { *self.cq_cqes.add(idx) };
        unsafe {
            (*self.cq_head).store(head.wrapping_add(1), Ordering::Release);
        }
        Some((cqe.user_data, cqe.res))
    }

    /// Drain all ready CQEs.
    pub(crate) fn drain_cqes(&mut self, sink: &mut dyn FnMut(u64, i32)) {
        while let Some((user_data, res)) = self.reap_one() {
            sink(user_data, res);
        }
    }

    /// How many SQEs are submitted-but-not-flushed.
    pub(crate) fn pending_submissions(&self) -> u32 {
        self.pending_submissions
    }

    /// Atomically take and reset the pending-submission count so the reaper
    /// (or the blocking fallback hook) can enter the kernel without holding
    /// the ring mutex.
    pub(crate) fn take_pending_submissions(&mut self) -> u32 {
        let pending = self.pending_submissions;
        self.pending_submissions = 0;
        pending
    }

    pub(crate) fn restore_pending_submissions(&mut self, pending: u32) {
        self.pending_submissions = self.pending_submissions.saturating_add(pending);
    }
}

impl Drop for IoUring {
    fn drop(&mut self) {
        unsafe {
            if self.ring_fd >= 0 {
                libc::close(self.ring_fd);
            }
            Self::unmap_region(self.sq_region, self.sq_region_sz);
            Self::unmap_region(self.cq_region, self.cq_region_sz);
        }
    }
}

// ---------------------------------------------------------------------------
// IoUringShared — ring state shared across all WAL shards
// ---------------------------------------------------------------------------

use crate::backend::{Completion, CompletionKind, SubmitResult, WalIoBackend};
use crate::format::{
    WAL_IO_URING_GROUP_COMMIT_DELAY_MAX_US, WAL_IO_URING_GROUP_COMMIT_TARGET_RECORDS, env_u64_us,
};
use crate::wal_impl::{WalEvent, WalInner, WalStats, finalize_buffer_records};
use pagebox_frame_kernel::Lsn;
use pagebox_threading as threading;

/// Shared ring state: one io_uring ring + slab + fd→WalInner registry.
///
/// All shards in a multi-shard WAL share one `IoUringShared` so that only one
/// fsync is in flight at a time (no disk I/O contention). Each shard's
/// `IoUringBackend` holds an `Arc<IoUringShared>` and submits SQEs to the
/// shared ring. CQE completions are dispatched to the correct `WalInner` via
/// the `fd_to_inner` registry (keyed by `RawFd`, registered in `start`).
pub(crate) struct IoUringShared {
    /// The io_uring ring fd, stored separately from `ring` so
    /// `poll_completions_blocking` can call `io_uring_enter` without holding
    /// the ring mutex.
    ring_fd: RawFd,
    ring: Mutex<IoUring>,
    /// Slab of in-flight writes keyed by slot index (== SQE user_data low
    /// bits). Shared across all shards so any driver thread can reap CQEs
    /// for any shard.
    in_flight: Mutex<Vec<Option<InFlightWrite>>>,
    free_slots: Mutex<Vec<u32>>,
    /// Maps each shard's WAL file fd → `Weak<WalInner>`. Used to dispatch
    /// CQE completions to the correct shard. Registered in `start`.
    fd_to_inner: Mutex<HashMap<RawFd, Weak<WalInner>>>,
    /// The dedicated reaper thread handle. One reaper serves all shards,
    /// blocking in `io_uring_enter(GETEVENTS)` and dispatching CQEs to the
    /// correct `WalInner`. This avoids N driver threads each blocking in
    /// `io_uring_enter` (thundering herd).
    reaper: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Shutdown signal for the reaper thread.
    shutdown: std::sync::atomic::AtomicBool,
}

impl IoUringShared {
    pub(crate) fn new() -> io::Result<Self> {
        let ring = IoUring::new(1024)?;
        let ring_fd = ring.ring_fd;
        Ok(Self {
            ring_fd,
            ring: Mutex::new(ring),
            in_flight: Mutex::new(Vec::with_capacity(64)),
            free_slots: Mutex::new(Vec::new()),
            fd_to_inner: Mutex::new(HashMap::new()),
            reaper: Mutex::new(None),
            shutdown: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn alloc_slot(&self, write: InFlightWrite) -> u32 {
        let mut in_flight = self.in_flight.lock();
        let mut free = self.free_slots.lock();
        if let Some(slot) = free.pop() {
            in_flight[slot as usize] = Some(write);
            return slot;
        }
        let slot = in_flight.len() as u32;
        in_flight.push(Some(write));
        slot
    }

    fn take_slot(&self, slot: u32) -> Option<InFlightWrite> {
        let mut in_flight = self.in_flight.lock();
        if (slot as usize) < in_flight.len()
            && let Some(write) = in_flight[slot as usize].take()
        {
            self.free_slots.lock().push(slot);
            Some(write)
        } else {
            None
        }
    }

    fn lookup_inner(&self, fd: RawFd) -> Option<Arc<WalInner>> {
        self.fd_to_inner.lock().get(&fd).and_then(|w| w.upgrade())
    }

    /// The dedicated reaper thread. Blocks in `io_uring_enter(GETEVENTS)` and
    /// dispatches CQEs to the correct `WalInner` via `fd_to_inner`. One reaper
    /// serves all shards, avoiding the thundering herd of N driver threads
    /// each blocking in `io_uring_enter`.
    fn run_reaper(self: Arc<Self>) {
        use std::sync::atomic::Ordering;
        loop {
            if self.shutdown.load(Ordering::Acquire) {
                return;
            }
            // Take pending submissions (flushes SQEs to kernel) and block for
            // ≥1 CQE. The ring mutex is released during the blocking syscall
            // so submit_write on any driver thread can enqueue SQEs
            // concurrently.
            let to_submit = {
                let mut ring = self.ring.lock();
                ring.take_pending_submissions()
            };
            let ret = unsafe {
                libc::syscall(
                    libc::SYS_io_uring_enter,
                    self.ring_fd,
                    to_submit,
                    1u32,
                    IORING_ENTER_GETEVENTS,
                    std::ptr::null::<libc::sigset_t>(),
                    0usize,
                )
            };
            if ret < 0 {
                let err = io::Error::last_os_error();
                let mut ring = self.ring.lock();
                ring.restore_pending_submissions(to_submit);
                drop(ring);
                if err.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                if self.shutdown.load(Ordering::Acquire) {
                    return;
                }
                eprintln!("WAL io_uring reaper error: {err}");
                std::thread::sleep(std::time::Duration::from_millis(1));
                continue;
            }
            let consumed = u32::try_from(ret).unwrap_or(u32::MAX).min(to_submit);
            if consumed < to_submit {
                self.ring
                    .lock()
                    .restore_pending_submissions(to_submit - consumed);
            }
            // Reap and dispatch all ready CQEs.
            let mut ring = self.ring.lock();
            ring.drain_cqes(&mut |user_data, res| {
                if user_data == NOP_USER_DATA {
                    return;
                }
                self.dispatch_cqe(user_data, res);
            });
        }
    }

    /// Dispatch a CQE to the correct `WalInner`, looked up by fd.
    fn dispatch_cqe(&self, user_data: u64, res: i32) {
        if res < 0 {
            let err = io::Error::from_raw_os_error(-res);
            panic!("WAL io_uring op failed — durability compromised: {err}");
        }
        if is_fsync(user_data)
            && let Some(write) = self.take_slot(slot_of(user_data))
        {
            let write_latency_ns =
                write.write_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            let completion = Completion {
                kind: CompletionKind::Written {
                    buffer: write.buffer,
                    max_lsn: write.max_lsn,
                    durable_lsn: Some(write.max_lsn),
                    write_latency_ns,
                },
            };
            match self.lookup_inner(write.fd) {
                Some(inner) => inner.handle_completion(completion),
                None => {
                    // The WalInner has been dropped (shutdown). The buffer
                    // is dropped with the completion — no leak.
                }
            }
        }
    }
}

struct InFlightWrite {
    buffer: WalBuffer,
    max_lsn: Lsn,
    write_start: std::time::Instant,
    /// The WAL file fd this write targets. Used to dispatch the CQE
    /// completion to the correct `WalInner` via `fd_to_inner`.
    fd: RawFd,
}

/// io_uring backend. One `IoUringShared` is shared across all WAL shards so
/// only one fsync is in flight at a time. Writes are submitted as `WRITE` SQEs
/// with a linked `FSYNC` SQE so durability arrives as a separate CQE. CQE
/// completions are dispatched to the correct `WalInner` via the `fd_to_inner`
/// registry.
pub(crate) struct IoUringBackend {
    shared: Arc<IoUringShared>,
    #[allow(dead_code)]
    fd: RawFd,
}

impl IoUringBackend {
    /// Create a standalone backend with its own ring (single-shard or test
    /// path).
    pub(crate) fn new(fd: RawFd) -> io::Result<Self> {
        Ok(Self {
            shared: Arc::new(IoUringShared::new()?),
            fd,
        })
    }

    /// Create a backend from a pre-existing shared ring (multi-shard path).
    pub(crate) fn from_shared(shared: Arc<IoUringShared>, fd: RawFd) -> Self {
        Self { shared, fd }
    }
}

impl WalIoBackend for IoUringBackend {
    fn submit_write(
        &self,
        mut write: PendingWalWrite,
        stats: &WalStats,
    ) -> io::Result<SubmitResult> {
        finalize_buffer_records(&mut write.buffer, write.len);
        stats.events.inc(WalEvent::WriteCall);
        stats.events.add(
            WalEvent::WriteBytes,
            write.len.min(isize::MAX as usize) as isize,
        );
        let write_start = std::time::Instant::now();
        let max_lsn = write.max_lsn;
        let fd = write.fd;
        let offset = write.file_offset;
        let len = write.len;
        let buffer = write.buffer;
        let buf_ptr = buffer.buffer.as_slice().as_ptr();
        let len_u32 = len as u32;
        let mut ring = self.shared.ring.lock();
        ring.ensure_submission_space(2)?;
        let slot = self.shared.alloc_slot(InFlightWrite {
            buffer,
            max_lsn,
            write_start,
            fd,
        });
        let user_data = pack_write(slot);
        let enqueued = ring.submit_write(fd, buf_ptr, len_u32, offset, user_data, true)?;
        debug_assert!(enqueued, "reserved io_uring WRITE slot disappeared");
        let fsync_user_data = pack_fsync(slot);
        let fsync_enqueued = ring.submit_fsync(fd, fsync_user_data, true)?;
        debug_assert!(fsync_enqueued, "reserved io_uring FSYNC slot disappeared");
        Ok(SubmitResult::Submitted)
    }

    fn submit_sync(&self, barrier_lsn: Lsn) {
        let _ = barrier_lsn;
    }

    fn poll_completions(&self, _sink: &mut dyn FnMut(Completion)) {
        let mut ring = self.shared.ring.lock();
        if ring.pending_submissions() > 0 {
            let _ = ring.flush_and_enter(0);
        }
        ring.drain_cqes(&mut |user_data, res| self.shared.dispatch_cqe(user_data, res));
    }

    fn poll_completions_blocking(&self, _sink: &mut dyn FnMut(Completion)) {
        let to_submit = {
            let mut ring = self.shared.ring.lock();
            ring.take_pending_submissions()
        };
        let ret = unsafe {
            libc::syscall(
                libc::SYS_io_uring_enter,
                self.shared.ring_fd,
                to_submit,
                1u32,
                IORING_ENTER_GETEVENTS,
                std::ptr::null::<libc::sigset_t>(),
                0usize,
            )
        };
        if ret < 0 {
            let err = io::Error::last_os_error();
            if err.kind() != io::ErrorKind::Interrupted {
                eprintln!("WAL io_uring_enter warning: {err}");
            }
        }
        let mut ring = self.shared.ring.lock();
        ring.drain_cqes(&mut |user_data, res| self.shared.dispatch_cqe(user_data, res));
    }

    fn has_in_flight(&self) -> bool {
        self.shared.in_flight.lock().iter().any(|s| s.is_some())
    }

    fn needs_syncer_thread(&self) -> bool {
        false
    }

    fn pre_extend_needs_fsync(&self) -> bool {
        false
    }

    fn group_commit_delay_max_us(&self) -> u64 {
        env_u64_us(
            "PAGEBOX_WAL_IO_URING_DELAY_MAX_US",
            WAL_IO_URING_GROUP_COMMIT_DELAY_MAX_US,
        )
    }

    fn group_commit_target_records(&self) -> u64 {
        env_u64_us(
            "PAGEBOX_WAL_IO_URING_TARGET_RECORDS",
            WAL_IO_URING_GROUP_COMMIT_TARGET_RECORDS,
        )
    }

    fn start(&self, inner: &Arc<WalInner>) -> io::Result<()> {
        let fd = inner.state.lock().fd;
        self.shared
            .fd_to_inner
            .lock()
            .insert(fd, Arc::downgrade(inner));
        // Spawn the reaper thread if not already running. Only the first
        // shard to call start() spawns it; subsequent shards just register
        // their fd.
        let mut reaper_guard = self.shared.reaper.lock();
        if reaper_guard.is_none() {
            let shared = Arc::clone(&self.shared);
            let handle = threading::spawn_efficient("wal-iouring-reaper", move || {
                shared.run_reaper();
            })?;
            *reaper_guard = Some(handle);
        }
        drop(reaper_guard);
        Ok(())
    }

    fn drain_for_shutdown(&self) {
        // Signal the reaper to shut down, then wake it with a NOP SQE
        // (it's blocked in io_uring_enter(GETEVENTS) and needs a CQE to
        // return).
        self.shared
            .shutdown
            .store(true, std::sync::atomic::Ordering::Release);
        {
            let mut ring = self.shared.ring.lock();
            let _ = ring.submit_nop(NOP_USER_DATA);
            let _ = ring.flush_and_enter(0);
        }
        // Join the reaper. Only one shard will successfully take the handle.
        if let Some(reaper) = self.shared.reaper.lock().take() {
            let _ = reaper.join();
        }
        // Reap any remaining CQEs.
        let mut ring = self.shared.ring.lock();
        let _ = ring.flush_and_enter(0);
        ring.drain_cqes(&mut |user_data, _res| {
            if is_fsync(user_data) {
                let _ = self.shared.take_slot(slot_of(user_data));
            }
        });
        drop(ring);
        self.shared.fd_to_inner.lock().remove(&self.fd);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_sizes() {
        assert_eq!(
            std::mem::size_of::<IoUringSqe>(),
            64,
            "SQE must be 64 bytes"
        );
        assert_eq!(
            std::mem::size_of::<IoUringCqe>(),
            16,
            "CQE must be 16 bytes"
        );
    }
}
