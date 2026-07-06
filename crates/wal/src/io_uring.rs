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
//! The [`IoUring`] owns the ring fd and three `mmap`'d regions (SQ ring, SQE
//! array, CQ ring). Writes are submitted as `IORING_OP_WRITEV` SQEs; an fsync
//! is submitted as an `IORING_OP_FSYNC` SQE linked to the preceding write via
//! `IOSQE_IO_LINK` so the fsync only fires once the write completes. SQE
//! `user_data` carries a packed id encoding whether the completion is a write
//! or fsync and (for writes) the slab slot holding the in-flight `WalBuffer`.
//!
//! Completion reaping is a single atomic CQ tail/head dance; no syscalls are
//! needed to reap. `io_uring_enter` is called to flush submitted SQEs to the
//! kernel and (optionally) to wait for completions.
//!
//! v1 deliberately uses plain kernel polling (no `SQPOLL`, no registered
//! buffers/files); those are future optimisations requiring their own bench
//! evidence.

#![cfg(target_os = "linux")]

use std::io;
use std::os::fd::RawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::Mutex;

use crate::wal_impl::{PendingWalWrite, WalBuffer};

// ---------------------------------------------------------------------------
// uapi definitions (linux/io_uring.h). `libc` gives us the syscall numbers
// but not these structs/consts, so they are hand-defined to match the kernel
// ABI exactly.
// ---------------------------------------------------------------------------

const IORING_ENTER_GETEVENTS: u32 = 1 << 0;

/// `IORING_SETUP_NO_MMAP` — the ring memory is user-allocated, not mmap'd
/// from the ring fd. Avoids broken CQ mmap on some kernels (≥5.18).
const IORING_SETUP_NO_MMAP: u32 = 1 << 14;

#[allow(dead_code)]
const IORING_OP_NOP: u8 = 0;
const IORING_OP_FSYNC: u8 = 3;
const IORING_OP_WRITE: u8 = 23;

const IOSQE_IO_LINK: u8 = 1 << 2;

const IORING_FSYNC_DATASYNC: u32 = 1 << 0;

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
    // User-allocated ring memory, kept for dealloc in Drop.
    sq_region: *mut u8,
    #[allow(dead_code)]
    sq_region_sz: usize,
    sq_layout: std::alloc::Layout,
    cq_region: *mut u8,
    #[allow(dead_code)]
    cq_region_sz: usize,
    cq_layout: std::alloc::Layout,
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
    /// Create a ring with `entries` SQ slots. On a kernel without io_uring
    /// (`ENOSYS` from `io_uring_setup`) the error is surfaced so `Wal::open`
    /// fails clearly rather than silently degrading.
    ///
    /// Uses `IORING_SETUP_NO_MMAP` (kernel ≥5.18) to avoid `mmap` of the
    /// ring regions, which is broken on some nvidia-variant kernels. The
    /// ring memory is user-allocated and handed to the kernel via the
    /// `sq_off.user_addr` / `cq_off.user_addr` params fields. A preview
    /// `io_uring_setup` call (without NO_MMAP) determines the exact ring
    /// sizes first, then a second call creates the real ring.
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

        let page = 4096usize;
        // SQ region: SQE array only (sq_off.user_addr points to SQEs).
        let sq_region_sz = ((sqe_array_sz + page - 1) & !(page - 1)).max(page * 16);
        // CQ region: the rings struct (SQ ring fields + CQ ring fields + SQ
        // array). Generously over-allocated to absorb any kernel-version
        // variation in field offsets.
        let cq_region_sz = ((rings_struct_sz + page - 1) & !(page - 1)).max(page * 32);

        // --- Allocations: user-provided ring memory. ---
        let sq_layout = std::alloc::Layout::from_size_align(sq_region_sz, page).unwrap();
        let cq_layout = std::alloc::Layout::from_size_align(cq_region_sz, page).unwrap();
        let sq_region = unsafe { std::alloc::alloc_zeroed(sq_layout) };
        let cq_region = unsafe { std::alloc::alloc_zeroed(cq_layout) };
        if sq_region.is_null() || cq_region.is_null() {
            if !sq_region.is_null() {
                unsafe { std::alloc::dealloc(sq_region, sq_layout) };
            }
            if !cq_region.is_null() {
                unsafe { std::alloc::dealloc(cq_region, cq_layout) };
            }
            return Err(io::Error::new(io::ErrorKind::OutOfMemory, "ring alloc"));
        }

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
            unsafe { std::alloc::dealloc(sq_region, sq_layout) };
            unsafe { std::alloc::dealloc(cq_region, cq_layout) };
            return Err(e);
        }
        let ring_fd = fd as RawFd;

        // Verify the real call matches the preview (same entries → same
        // ring geometry). If these differ, our allocation sizes are wrong.
        debug_assert_eq!(
            params.sq_entries, sq_entries,
            "io_uring sq_entries mismatch between preview and real call"
        );
        debug_assert_eq!(
            params.cq_entries, cq_entries,
            "io_uring cq_entries mismatch between preview and real call"
        );
        debug_assert_eq!(
            params.sq_off.array, preview.sq_off.array,
            "io_uring sq_off.array mismatch"
        );
        debug_assert_eq!(
            params.cq_off.cqes, preview.cq_off.cqes,
            "io_uring cq_off.cqes mismatch"
        );

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
            sq_layout,
            cq_region,
            cq_region_sz,
            cq_layout,
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

    /// Flush pending SQEs to the kernel and optionally wait for `min_complete`
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
        self.pending_submissions = 0;
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
}

impl Drop for IoUring {
    fn drop(&mut self) {
        unsafe {
            if self.ring_fd >= 0 {
                libc::close(self.ring_fd);
            }
            if !self.sq_region.is_null() {
                std::alloc::dealloc(self.sq_region, self.sq_layout);
            }
            if !self.cq_region.is_null() {
                std::alloc::dealloc(self.cq_region, self.cq_layout);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// IoUringBackend (WalIoBackend impl)
// ---------------------------------------------------------------------------

use crate::backend::{Completion, CompletionKind, SubmitResult, WalIoBackend};
use crate::format::{
    WAL_IO_URING_GROUP_COMMIT_DELAY_MAX_US, WAL_IO_URING_GROUP_COMMIT_TARGET_RECORDS, env_u64_us,
};
use crate::wal_impl::{WalEvent, WalInner, WalStats, finalize_buffer_records};
use pagebox_frame_kernel::Lsn;

/// io_uring backend. One ring per shard. Writes are submitted as `WRITEV`
/// SQEs with a linked `FSYNC` SQE so durability arrives as a separate CQE.
/// In-flight `WalBuffer`s live in a slab keyed by SQE `user_data` until their
/// write CQE reaps; the slab reuses slots once reaped.
pub(crate) struct IoUringBackend {
    #[allow(dead_code)]
    fd: RawFd,
    /// The ring is guarded by a mutex because the driver loop calls
    /// `submit_write` / `submit_sync` / `poll_completions` sequentially but
    /// `drain_for_shutdown` runs on the shutdown path; a single producer
    /// (the driver) is the steady-state case.
    ring: Mutex<IoUring>,
    /// Slab of in-flight writes keyed by slot index (== SQE user_data low
    /// bits). `None` slots are free. Guarded by its own mutex so reaping
    /// (which releases a slot) need not take the ring mutex.
    in_flight: Mutex<Vec<Option<InFlightWrite>>>,
    /// Free slot indices for reuse.
    free_slots: Mutex<Vec<u32>>,
}

struct InFlightWrite {
    buffer: WalBuffer,
    max_lsn: Lsn,
    write_start: std::time::Instant,
}

impl IoUringBackend {
    pub(crate) fn new(fd: RawFd) -> io::Result<Self> {
        // 1024 SQ entries: generous enough that the SQ ring never fills
        // under normal load (the driver reaps CQEs every iteration). Each
        // write+fsync chain uses 2 SQEs; 512 in-flight chains would be
        // needed to fill the ring, which is far beyond the WAL's
        // single-buffer-at-a-time drain model.
        let ring = IoUring::new(1024)?;
        Ok(Self {
            fd,
            ring: Mutex::new(ring),
            in_flight: Mutex::new(Vec::with_capacity(64)),
            free_slots: Mutex::new(Vec::new()),
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
        // The SQE reads from the buffer asynchronously; ownership of the
        // `WalBuffer` (and its `AlignedBuf`) transfers to the slab until the
        // write CQE reaps. Extract the raw iovec from the buffer before moving
        // it into the slab.
        let buffer = write.buffer;
        let buf_ptr = buffer.buffer.as_slice().as_ptr();
        let len_u32 = len as u32;
        let slot = self.alloc_slot(InFlightWrite {
            buffer,
            max_lsn,
            write_start,
        });
        let user_data = pack_write(slot);
        let mut ring = self.ring.lock();
        // Link the write SQE so a following fsync SQE only fires after it.
        let enqueued = ring.submit_write(fd, buf_ptr, len_u32, offset, user_data, true)?;
        if !enqueued {
            ring.flush_and_enter(0)?;
            ring.submit_write(fd, buf_ptr, len_u32, offset, user_data, true)?;
        }
        // Submit a linked fsync SQE so durability arrives as a CQE. If the
        // SQ ring is full, flush and retry — the FSYNC must be enqueued or
        // the slab slot would leak (the FSYNC CQE is what reclaims it).
        let fsync_user_data = pack_fsync(slot);
        let fsync_enqueued = ring.submit_fsync(fd, fsync_user_data, true)?;
        if !fsync_enqueued {
            ring.flush_and_enter(0)?;
            ring.submit_fsync(fd, fsync_user_data, true)?;
        }
        Ok(SubmitResult::Submitted)
    }

    fn submit_sync(&self, barrier_lsn: Lsn) {
        // Durability is driven by the linked fsync SQE submitted with each
        // write; the barrier LSN is tracked via `requested_durable_lsn` which
        // the driver checks against `durable_lsn`. Flushing the ring is the
        // driver's responsibility (it calls poll_completions, which flushes).
        let _ = barrier_lsn;
    }

    fn poll_completions(&self, sink: &mut dyn FnMut(Completion)) {
        let mut ring = self.ring.lock();
        if ring.pending_submissions() > 0 {
            let _ = ring.flush_and_enter(0);
        }
        ring.drain_cqes(&mut |user_data, res| {
            if res < 0 {
                let err = io::Error::from_raw_os_error(-res);
                panic!("WAL io_uring op failed — durability compromised: {err}");
            }
            if is_fsync(user_data)
                && let Some(write) = self.take_slot(slot_of(user_data))
            {
                let write_latency_ns =
                    write.write_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                sink(Completion {
                    kind: CompletionKind::Written {
                        buffer: write.buffer,
                        max_lsn: write.max_lsn,
                        durable_lsn: Some(write.max_lsn),
                        write_latency_ns,
                    },
                });
            }
        });
    }

    fn has_in_flight(&self) -> bool {
        self.in_flight.lock().iter().any(|s| s.is_some())
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

    fn start(&self, _inner: &Arc<WalInner>) -> io::Result<()> {
        Ok(())
    }

    fn drain_for_shutdown(&self) {
        // Reap any outstanding CQEs so in-flight buffers are reclaimed. On
        // crash shutdown the driver returns without polling further, so
        // outstanding buffers leak with the (dropped) slab — matching the
        // crash contract that pending appends are not flushed.
        let mut ring = self.ring.lock();
        let _ = ring.flush_and_enter(0);
        ring.drain_cqes(&mut |_user_data, _res| {
            // Discard completions during shutdown drain; the slab slots will
            // be dropped with the backend.
        });
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
