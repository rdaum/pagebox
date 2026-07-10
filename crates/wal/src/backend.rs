//! Sync-write backend trait and the synchronous backends.
//!
//! [`WalIoBackend`] is the seam between the WAL driver loop and the underlying
//! I/O path. It expresses write and durability as a completion-based
//! interface so that a synchronous backend (`fdatasync`, `pwritev2_dsync`) and
//! an asynchronous, completion-driven backend (io_uring) can sit behind one
//! trait.
//!
//! ## Completion model
//!
//! [`WalIoBackend::submit_write`] either:
//! - completes the write inline and returns the buffer to the caller (sync
//!   backends — [`SubmitResult::WrittenPendingSync`] /
//!   [`SubmitResult::WrittenAndDurable`]), or
//! - takes ownership of the buffer until a `Written` completion is reaped
//!   ([`SubmitResult::Submitted`] — io_uring).
//!
//! Durability arrives either inline with the write (`pwritev2_dsync`), via a
//! backend-internal syncer thread (`fdatasync`, which advances `durable_lsn`
//! directly and wakes flush waiters), or when the dedicated io_uring reaper
//! dispatches the linked fsync CQE.
//!
//! The driver loop calls [`WalInner::handle_completion`] for inline and polled
//! write completions; the io_uring reaper calls the same method after fd-based
//! dispatch. The fdatasync syncer advances durability directly after the
//! syscall completes.
//!
//! [`WalInner::handle_completion`]: crate::wal_impl::WalInner::handle_completion

use std::io;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use pagebox_frame_kernel::Lsn;
use pagebox_threading as threading;

use crate::format::{
    WAL_FDATASYNC_GROUP_COMMIT_DELAY_MAX_US, WAL_FDATASYNC_GROUP_COMMIT_TARGET_RECORDS,
    WAL_PWRITEV2_DSYNC_GROUP_COMMIT_DELAY_MAX_US, WAL_PWRITEV2_DSYNC_GROUP_COMMIT_TARGET_RECORDS,
    WAL_RELAXED_SYNC_INTERVAL_US, env_u64_us,
};
use crate::io::{pwrite_all, pwritev2_dsync_all, sync_wal_fd};
use crate::wal_impl::{
    PendingWalWrite, WalBuffer, WalEvent, WalInner, WalLatency, WalStats, finalize_buffer_records,
};

use super::wal_impl::WalSyncBackend;

/// Result of [`WalIoBackend::submit_write`].
pub(crate) enum SubmitResult {
    /// `pwrite` completed inline; the write is *not* yet durable. The buffer
    /// is handed back to the caller for reclamation. A separate durability
    /// advance (backend-internal syncer thread) will follow.
    WrittenPendingSync {
        buffer: WalBuffer,
        max_lsn: Lsn,
        write_latency_ns: u64,
    },
    /// `pwritev2(RWF_DSYNC)` completed inline and is already durable. The
    /// buffer is handed back; the caller advances both `written_lsn` and
    /// `durable_lsn`.
    WrittenAndDurable {
        buffer: WalBuffer,
        max_lsn: Lsn,
        write_latency_ns: u64,
    },
    /// The buffer is now in flight (io_uring). Ownership has transferred to
    /// the backend until the dedicated reaper dispatches the linked fsync CQE.
    #[allow(dead_code)]
    Submitted,
}

pub(crate) enum SyncSubmission {
    Submitted,
    Coalesced,
    Unchanged,
    BackendManaged,
}

/// One I/O completion drained from a backend.
pub(crate) struct Completion {
    pub(crate) kind: CompletionKind,
}

pub(crate) enum CompletionKind {
    /// A write finished. The buffer is returned for reclamation; `written_lsn`
    /// should advance to `max_lsn`. If `durable_lsn` is `Some`, the write was
    /// also durable (pwritev2_dsync) and `durable_lsn` should advance too.
    Written {
        buffer: WalBuffer,
        max_lsn: Lsn,
        durable_lsn: Option<Lsn>,
        write_latency_ns: u64,
    },
    /// An fsync finished; `durable_lsn` should advance to `lsn`. Retained for
    /// backends that report durability separately from write completion.
    #[allow(dead_code)]
    Durable {
        lsn: Lsn,
        sync_latency_ns: u64,
        drain_wait_ns: u64,
        fsync_latency_ns: u64,
    },
}

#[allow(dead_code)]
pub(crate) trait WalIoBackend: Send + Sync {
    /// Submit a sealed buffer for writing. Records `WriteCall` / `WriteBytes`
    /// telemetry (the backend knows the byte count); the per-write latency is
    /// measured by the backend and carried in the returned completion so
    /// [`WalInner::handle_completion`] records the `Write` histogram.
    fn submit_write(&self, write: PendingWalWrite, stats: &WalStats) -> io::Result<SubmitResult>;

    /// Request durability up to `barrier_lsn`. For `fdatasync` this is a
    /// no-op (the backend-internal syncer thread is self-driven via the shared
    /// condvar); io_uring submits one drained fsync barrier; and
    /// `pwritev2_dsync` writes are durable inline.
    fn submit_sync(&self, barrier_lsn: Lsn) -> io::Result<SyncSubmission>;

    /// Whether a separately completed durability barrier is in flight.
    /// The driver uses this to avoid repeatedly submitting the same target.
    fn sync_in_flight(&self) -> bool {
        false
    }

    /// Highest written LSN already recorded for a durability barrier.
    fn sync_target(&self) -> Lsn {
        0
    }

    /// Drain ready completions into `sink`. Non-blocking: flushes pending
    /// SQEs and reaps whatever CQEs are ready. The driver loop handles
    /// waiting when no CQEs are ready (via [`WalIoBackend::has_in_flight`]).
    fn poll_completions(&self, sink: &mut dyn FnMut(Completion));

    /// Optional blocking completion hook. Synchronous backends use the no-op
    /// default; the current io_uring path normally relies on its dedicated
    /// reaper thread instead.
    fn poll_completions_blocking(&self, _sink: &mut dyn FnMut(Completion)) {}

    /// Whether the backend has in-flight writes whose completions haven't
    /// been reaped. The driver loop uses this to decide whether to use a
    /// timed wait (to poll again soon) or an indefinite wait.
    fn has_in_flight(&self) -> bool {
        false
    }

    /// Whether the backend spawns its own syncer thread (only `fdatasync`).
    fn needs_syncer_thread(&self) -> bool;

    /// Whether `ftruncate` pre-extends must be `fdatasync`'d before a durable
    /// write lands (only `pwritev2_dsync`).
    fn pre_extend_needs_fsync(&self) -> bool;

    fn group_commit_delay_max_us(&self) -> u64;
    fn group_commit_target_records(&self) -> u64;

    /// Spawn backend-specific threads: the fdatasync syncer or the shared
    /// io_uring reaper. Called after the writer thread is started and
    /// `WalInner` is fully constructed.
    fn start(&self, inner: &Arc<WalInner>) -> io::Result<()>;

    /// Stop and join backend-specific threads and drain outstanding work on
    /// clean shutdown. Called after the writer thread has joined. On crash
    /// shutdown the backend should stop without draining.
    fn drain_for_shutdown(&self);
}

/// Construct the backend for a given selector. The file descriptor is passed
/// so a future io_uring backend can register it up front. For io_uring, the
/// `shared_ring` parameter allows multiple shards to share one ring; when
/// `None`, a standalone ring is created.
pub(crate) fn make_backend(
    selector: WalSyncBackend,
    fd: std::os::fd::RawFd,
    #[allow(unused_variables)] shared_ring: Option<&std::sync::Arc<crate::io_uring::IoUringShared>>,
) -> io::Result<Box<dyn WalIoBackend>> {
    match selector {
        WalSyncBackend::Fdatasync => Ok(Box::new(FdatasyncBackend::new(fd))),
        WalSyncBackend::Pwritev2Dsync => Ok(Box::new(Pwritev2DsyncBackend)),
        #[cfg(target_os = "linux")]
        WalSyncBackend::IoUring => {
            let backend = match shared_ring {
                Some(ring) => crate::io_uring::IoUringBackend::from_shared(ring.clone(), fd),
                None => crate::io_uring::IoUringBackend::new(fd).map_err(|e| {
                    io::Error::new(e.kind(), format!("io_uring backend unavailable: {e}"))
                })?,
            };
            Ok(Box::new(backend))
        }
        #[cfg(not(target_os = "linux"))]
        WalSyncBackend::IoUring => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "io_uring WAL backend is only available on Linux",
        )),
    }
}

// ---------------------------------------------------------------------------
// fdatasync backend
// ---------------------------------------------------------------------------

/// `fdatasync` backend: `pwrite` the buffer inline, then a dedicated syncer
/// thread calls `fdatasync` and pushes a `Durable` completion onto a
/// [`SegQueue`]. The driver loop drains the queue via
/// [`WalIoBackend::poll_completions`] and advances `durable_lsn` in
/// [`WalInner::handle_completion`], so durable advancement and condvar
/// wake-ups happen in the single, shared completion path. The syncer only
/// notifies `flush_requested` (to wake the driver to poll); the driver
/// notifies `flush_done` (when waiters should re-check `durable_lsn`).
struct FdatasyncBackend {
    fd: std::os::fd::RawFd,
    syncer: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl FdatasyncBackend {
    fn new(fd: std::os::fd::RawFd) -> Self {
        Self {
            fd,
            syncer: Mutex::new(None),
        }
    }
}

impl WalIoBackend for FdatasyncBackend {
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
        let write_start = Instant::now();
        let data = &write.buffer.buffer.as_slice()[..write.len];
        pwrite_all(write.fd, data, write.file_offset as i64)?;
        let write_latency_ns = write_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        Ok(SubmitResult::WrittenPendingSync {
            buffer: write.buffer,
            max_lsn: write.max_lsn,
            write_latency_ns,
        })
    }

    fn submit_sync(&self, _barrier_lsn: Lsn) -> io::Result<SyncSubmission> {
        // The syncer thread is self-driven via the shared `flush_requested`
        // condvar (notified by `request_durable` / `wait_for_durable` /
        // `set_commit_mode` / the driver). Nothing to do here.
        Ok(SyncSubmission::BackendManaged)
    }

    fn poll_completions(&self, _sink: &mut dyn FnMut(Completion)) {
        // The syncer advances durable_lsn and notifies flush_done directly
        // (see run_syncer). Nothing to poll here.
    }

    fn needs_syncer_thread(&self) -> bool {
        true
    }

    fn pre_extend_needs_fsync(&self) -> bool {
        false
    }

    fn group_commit_delay_max_us(&self) -> u64 {
        env_u64_us(
            "PAGEBOX_WAL_FDATASYNC_DELAY_MAX_US",
            env_u64_us(
                "PAGEBOX_WAL_GROUP_COMMIT_DELAY_MAX_US",
                WAL_FDATASYNC_GROUP_COMMIT_DELAY_MAX_US,
            ),
        )
    }

    fn group_commit_target_records(&self) -> u64 {
        env_u64_us(
            "PAGEBOX_WAL_FDATASYNC_TARGET_RECORDS",
            env_u64_us(
                "PAGEBOX_WAL_GROUP_COMMIT_TARGET_RECORDS",
                WAL_FDATASYNC_GROUP_COMMIT_TARGET_RECORDS,
            ),
        )
    }

    fn start(&self, inner: &Arc<WalInner>) -> io::Result<()> {
        let inner = inner.clone();
        let handle = threading::spawn_efficient("wal-syncer", move || Self::run_syncer(&inner))?;
        *self.syncer.lock() = Some(handle);
        Ok(())
    }

    fn drain_for_shutdown(&self) {
        if let Some(syncer) = self.syncer.lock().take() {
            let _ = syncer.join();
        }
    }
}

impl FdatasyncBackend {
    /// The syncer thread: self-wakes on `flush_requested`, runs `fdatasync`,
    /// and pushes a `Durable` completion onto the queue for the driver loop
    /// to drain. It does *not* advance `durable_lsn` or notify `flush_done`
    /// itself — that is the driver loop's job (via `handle_completion`) so the
    /// completion path is uniform across backends. The syncer notifies
    /// `flush_requested` after pushing so an idle driver wakes to poll.
    fn run_syncer(inner: &Arc<WalInner>) {
        use crate::wal_impl::CommitMode;

        let mut last_sync = Instant::now();
        loop {
            if inner.has_backend_failure() {
                return;
            }
            let mut state = inner.state.lock();
            loop {
                if inner.has_backend_failure() {
                    return;
                }
                let relaxed_mode =
                    inner.commit_mode.load(Ordering::Acquire) == CommitMode::Relaxed as u64;
                let durable = inner.durable_lsn.load(Ordering::Acquire);
                let written = inner.written_lsn.load(Ordering::Acquire);
                let requested_durable = inner.requested_durable_lsn.load(Ordering::Acquire);
                if requested_durable > written {
                    inner
                        .requested_write_lsn
                        .fetch_max(requested_durable, Ordering::Release);
                    inner.flush_requested.notify_all();
                }

                let pending_sync = written > durable
                    && (requested_durable > durable
                        || (relaxed_mode && inner.should_sync_relaxed(written, last_sync)));
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
                    inner.flush_requested.wait_for(
                        &mut state,
                        Duration::from_micros(env_u64_us(
                            "PAGEBOX_WAL_RELAXED_SYNC_INTERVAL_US",
                            WAL_RELAXED_SYNC_INTERVAL_US,
                        )),
                    );
                } else {
                    inner.flush_requested.wait(&mut state);
                }
            }

            let leader_target_lsn = inner.requested_durable_lsn.load(Ordering::Acquire);
            drop(state);
            inner.maybe_group_commit_delay(leader_target_lsn);

            let state = inner.state.lock();
            let durable = inner.durable_lsn.load(Ordering::Acquire);
            let written = inner.written_lsn.load(Ordering::Acquire);
            let requested_durable = inner.requested_durable_lsn.load(Ordering::Acquire);
            if requested_durable > written {
                inner
                    .requested_write_lsn
                    .fetch_max(requested_durable, Ordering::Release);
                inner.flush_requested.notify_all();
                continue;
            }
            let relaxed_mode =
                inner.commit_mode.load(Ordering::Acquire) == CommitMode::Relaxed as u64;
            let need_sync = written > durable
                && (requested_durable > durable
                    || (relaxed_mode && inner.should_sync_relaxed(written, last_sync)));
            if !need_sync {
                continue;
            }
            // Writes can continue while fdatasync is in flight. Only the LSNs
            // written before this sync starts are reported durable here.
            let synced_lsn = written;
            let fd = state.fd;
            drop(state);

            let sync_latency_ns = match Self::sync_fd(fd) {
                Ok(latency) => latency,
                Err(err) => {
                    inner.record_backend_failure(err);
                    return;
                }
            };
            last_sync = Instant::now();
            // Advance durable_lsn + notify directly. This
            // avoids the round-trip through the SegQueue → driver poll,
            // which introduced a phase-locked timeout race under group-commit
            // contention.
            inner.stats.events.inc(WalEvent::SyncCall);
            inner
                .stats
                .latencies
                .get(WalLatency::Sync)
                .record(sync_latency_ns);
            inner.advance_durable_lsn(synced_lsn);
            inner.flush_done.notify_all();
            inner.flush_requested.notify_all();
        }
    }

    /// Run `fdatasync` and return its latency in nanoseconds.
    fn sync_fd(fd: std::os::fd::RawFd) -> io::Result<u64> {
        let sync_start = Instant::now();
        sync_wal_fd(fd)?;
        let latency = sync_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        Ok(latency)
    }
}

// ---------------------------------------------------------------------------
// pwritev2(RWF_DSYNC) backend
// ---------------------------------------------------------------------------

/// `pwritev2` with `RWF_DSYNC`: one syscall does the durable write, so no
/// separate syncer thread is needed. The write is durable the moment
/// `submit_write` returns.
struct Pwritev2DsyncBackend;

impl WalIoBackend for Pwritev2DsyncBackend {
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
        let write_start = Instant::now();
        let data = &write.buffer.buffer.as_slice()[..write.len];
        pwritev2_dsync_all(write.fd, data, write.file_offset as i64)?;
        let write_latency_ns = write_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        Ok(SubmitResult::WrittenAndDurable {
            buffer: write.buffer,
            max_lsn: write.max_lsn,
            write_latency_ns,
        })
    }

    fn submit_sync(&self, _barrier_lsn: Lsn) -> io::Result<SyncSubmission> {
        Ok(SyncSubmission::BackendManaged)
    }

    fn poll_completions(&self, _sink: &mut dyn FnMut(Completion)) {}

    fn needs_syncer_thread(&self) -> bool {
        false
    }

    fn pre_extend_needs_fsync(&self) -> bool {
        true
    }

    fn group_commit_delay_max_us(&self) -> u64 {
        env_u64_us(
            "PAGEBOX_WAL_PWRITEV2_DSYNC_DELAY_MAX_US",
            WAL_PWRITEV2_DSYNC_GROUP_COMMIT_DELAY_MAX_US,
        )
    }

    fn group_commit_target_records(&self) -> u64 {
        env_u64_us(
            "PAGEBOX_WAL_PWRITEV2_DSYNC_TARGET_RECORDS",
            WAL_PWRITEV2_DSYNC_GROUP_COMMIT_TARGET_RECORDS,
        )
    }

    fn start(&self, _inner: &Arc<WalInner>) -> io::Result<()> {
        Ok(())
    }

    fn drain_for_shutdown(&self) {}
}

// Suppress unused-fd warnings: the fdatasync backend keeps the fd for future
// fixed-file registration / direct use; the syncer currently reads it from
// `WalState` to match the pre-trait code exactly.
#[allow(dead_code)]
fn _keep_fd(b: &FdatasyncBackend) -> std::os::fd::RawFd {
    b.fd
}
