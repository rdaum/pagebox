//! Positional I/O helpers for the WAL: `pread`/`pwrite` retry loops, file
//! extension, and the `pwritev2` + `RWF_DSYNC` backend.
//!
//! Everything in this module is `pub(crate)` — the public surface is on
//! [`Wal`](crate::Wal) itself. The I/O loops here are EINTR-safe: a syscall
//! interrupted by a signal is retried rather than surfaced as `Interrupted`,
//! so callers can treat short reads / writes as terminal errors. Positional
//! `pread` / `pwrite` (rather than `read` / `write`) are used so the kernel
//! handles concurrency on the file descriptor and no per-fd mutex is needed
//! in the WAL state.
//!
//! ## Sync backends
//!
//! - [`fdatasync_file`] — `fdatasync` on Linux / Android, `fsync` elsewhere.
//!   Compatible with every Unix.
//! - [`pwritev2_dsync_all`] — `pwritev2(2)` with `RWF_DSYNC`. Linux-only;
//!   on other platforms it returns `Unsupported` so the WAL falls back to the
//!   `fdatasync` path. Selected at runtime via
//!   `PAGEBOX_WAL_SYNC_BACKEND=pwritev2_dsync`.

use std::io;
use std::os::fd::RawFd;

pub(crate) fn fstat_size(fd: RawFd) -> io::Result<u64> {
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut stat) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(stat.st_size as u64)
}

pub(crate) fn extend_file(fd: RawFd, new_size: u64) -> io::Result<()> {
    if unsafe { libc::ftruncate(fd, new_size as libc::off_t) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

pub(crate) fn fdatasync_file(fd: RawFd) -> io::Result<()> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let ret = unsafe { libc::fdatasync(fd) };

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    let ret = unsafe { libc::fsync(fd) };

    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

pub(crate) fn sync_wal_fd(fd: RawFd) -> io::Result<()> {
    fdatasync_file(fd)
}

pub(crate) fn round_up_u64(n: u64, align: u64) -> u64 {
    (n + align - 1) & !(align - 1)
}

/// Read exactly `buf.len()` bytes at `offset`. EINTR-safe.
pub(crate) fn pread_all(fd: RawFd, buf: &mut [u8], offset: i64) -> io::Result<()> {
    let mut done = 0usize;
    let len = buf.len();
    while done < len {
        let n = unsafe {
            libc::pread(
                fd,
                buf[done..].as_mut_ptr() as *mut libc::c_void,
                len - done,
                offset + done as i64,
            )
        };
        if n < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("pread: eof at offset {}", offset + done as i64),
            ));
        }
        done += n as usize;
    }
    Ok(())
}

/// Write exactly `data.len()` bytes at `offset`. EINTR-safe.
pub(crate) fn pwrite_all(fd: RawFd, data: &[u8], offset: i64) -> io::Result<()> {
    let mut done = 0usize;
    let len = data.len();
    while done < len {
        let n = unsafe {
            libc::pwrite(
                fd,
                data[done..].as_ptr() as *const libc::c_void,
                len - done,
                offset + done as i64,
            )
        };
        if n < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                format!("pwrite: zero bytes at offset {}", offset + done as i64),
            ));
        }
        done += n as usize;
    }
    Ok(())
}

/// Write exactly `data.len()` bytes at `offset` with Linux RWF_DSYNC.
#[cfg(target_os = "linux")]
pub(crate) fn pwritev2_dsync_all(fd: RawFd, data: &[u8], offset: i64) -> io::Result<()> {
    const RWF_DSYNC: libc::c_int = 0x0000_0002;

    let mut done = 0usize;
    let len = data.len();
    while done < len {
        let write_offset = offset + done as i64;
        let write_offset_u = write_offset as u64;
        let iov = libc::iovec {
            iov_base: data[done..].as_ptr() as *mut libc::c_void,
            iov_len: len - done,
        };
        let n = unsafe {
            libc::syscall(
                libc::SYS_pwritev2,
                fd,
                &iov as *const libc::iovec,
                1,
                write_offset_u as usize,
                (write_offset_u >> 32) as usize,
                RWF_DSYNC,
            )
        };
        if n < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                format!("pwritev2: zero bytes at offset {write_offset}"),
            ));
        }
        done += n as usize;
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn pwritev2_dsync_all(_fd: RawFd, _data: &[u8], _offset: i64) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "pwritev2 RWF_DSYNC is only available on Linux",
    ))
}
