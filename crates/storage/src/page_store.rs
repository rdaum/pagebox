//! File- and memory-backed page stores: the disk half of the buffer pool.
//!
//! The [`PageStore`] trait abstracts the four operations the buffer pool needs
//! — `read_page`, `write_page`, `allocate`, `sync` — plus a few optional
//! batching / advisory extensions ([`BatchPageStore`]). Concrete
//! implementations:
//!
//! - [`InMemoryPageStore`] — `HashMap<PageId, Vec<u8>>`; used by tests and by
//!   the `kvstore` binary's recovery dry-runs.
//! - [`FilePageStore`] — single-file, positioned `pread`/`pwrite` backend with
//!   `O_DIRECT` support and a header page at page 0 carrying the magic, page
//!   count, checkpoint LSN, and two user-meta slots used by reopened trees.
//!
//! ## Header page layout (page 0)
//!
//! ```text
//!   bytes  0..4:   magic              (u32 LE) — 0x424F5854 ("BOXT")
//!   bytes  4..6:   version            (u16 LE) — format version
//!   bytes  6..8:   reserved
//!   bytes  8..16:  page_count         (u64 LE) — highest allocated page id
//!   bytes 16..24:  checkpoint_lsn     (u64 LE) — LSN of last completed checkpoint
//!   bytes 24..32:  user_meta_0        (u64 LE) — user meta slot 0 (0 = unset)
//!   bytes 32..40:  user_meta_1        (u64 LE) — user meta slot 1 (0 = unset)
//!   bytes 40..48:  user_meta_2        (u64 LE) — user meta slot 2 (0 = unset)
//! ```
//!
//! The `user_meta_*` slots are how reopened trees find their root: the
//! B+tree writes its root page ID to `user_meta_0` and its height to
//! `user_meta_1` on close, and the same is read back on reopen via
//! `validate_header`. The constants and helpers live in this module under
//! `pub(crate)`; the public face is on [`FilePageStore`] itself
//! (`checkpoint_lsn` / `set_checkpoint_lsn` and the three `user_meta_*`
//! getter/setter pairs).
//!
//! ## Direct I/O
//!
//! `FilePageStore` defaults to buffered I/O. Set
//! `PAGEBOX_PAGE_STORE_DIRECT_IO=1` to try `O_DIRECT` (Linux only); the
//! backend falls back to buffered I/O if `O_DIRECT` is unsupported by the
//! filesystem. Header pages and write buffers are 4 KiB-aligned via
//! `AlignedPage` to satisfy the `O_DIRECT` alignment requirement.
//!
//! `RecoveryPageStore` (from `pagebox-wal`) is implemented for both backends
//! so the WAL replay path can read / write / allocate / sync without a
//! separate page-store trait.

use std::collections::HashMap;
use std::io;
use std::sync::Mutex;
#[cfg(not(miri))]
use std::sync::atomic::{AtomicU64, Ordering};

use pagebox_wal::RecoveryPageStore;

use crate::buffer_frame::{PAGE_SIZE, PageId, page_end_base_page, page_size, physical_page_number};

pub(crate) fn validate_page_buf_len(pid: PageId, len: usize) -> io::Result<()> {
    let expected = page_size(pid);
    if len == expected {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("page {pid} expects {expected} bytes, got {len}"),
    ))
}

pub trait PageStore: Send + Sync {
    /// Read a page into `buf`. Returns `Ok(true)` if the page existed,
    /// `Ok(false)` if the page has not been allocated, or `Err` on I/O failure.
    fn read_page(&self, pid: PageId, buf: &mut [u8]) -> io::Result<bool>;

    /// Write a page from `buf`.
    fn write_page(&self, pid: PageId, data: &[u8]) -> io::Result<()>;

    /// Allocate a slot for a new page (zeroed).
    fn allocate(&self, pid: PageId) -> io::Result<()>;

    /// Flush all pending writes to stable storage.
    fn sync(&self) -> io::Result<()>;

    /// Number of pages currently stored.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The next page id that should be used for allocation.
    /// For a fresh store this is 1 (pid 0 is reserved).
    /// For a reopened store this is one past the highest allocated pid.
    fn next_page_id(&self) -> PageId;

    /// Return the raw file descriptor for direct I/O, if this store is
    /// backed by a single file. Returns `None` for in-memory stores.
    fn raw_fd(&self) -> Option<std::os::fd::RawFd> {
        None
    }

    /// Advise the kernel to drop cached pages for this store's file.
    /// Used by benchmarks to force cache-cold reads.  No-op for
    /// in-memory stores or if the OS doesn't support it.
    fn drop_cache(&self) {
        // Default: no-op.
    }

    /// Write multiple pages and sync to stable storage in one operation.
    ///
    /// The default implementation loops `write_page` then calls `sync`.
    /// Backends with batching support can override this to submit all writes
    /// plus sync in fewer kernel transitions.
    fn write_pages_and_sync(&self, pages: &[(PageId, &[u8])]) -> io::Result<()> {
        for &(pid, data) in pages {
            self.write_page(pid, data)?;
        }
        self.sync()
    }

    /// Write multiple pages without forcing them durable.
    ///
    /// This is appropriate for background eviction/writeback when WAL
    /// already protects durability and the page store only needs to be
    /// brought up to date lazily.
    fn write_pages(&self, pages: &[(PageId, &[u8])]) -> io::Result<()> {
        for &(pid, data) in pages {
            self.write_page(pid, data)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BatchPageStore — submit/complete I/O for batching backends
// ---------------------------------------------------------------------------

/// Extension trait for page stores that support submit/complete I/O.
///
/// The submit methods enqueue an operation and return a token. The wait
/// methods block until tokens (or all in-flight operations) have completed.
///
/// For synchronous backends like `FilePageStore`, the submit methods
/// perform the operation inline and return a dummy token; `wait` is a
/// no-op.
pub trait BatchPageStore: PageStore {
    /// Opaque handle returned by submit methods.
    type Token: Copy;

    /// Enqueue a page read.  The buffer must remain valid until `wait` or
    /// `wait_all` returns.
    ///
    /// # Safety
    ///
    /// The caller must ensure `buf` is not accessed between the call to
    /// `submit_read` and the corresponding `wait`/`wait_all`.
    unsafe fn submit_read(&self, pid: PageId, buf: &mut [u8]) -> io::Result<Self::Token>;

    /// Enqueue a page write.  The data must remain valid until `wait` or
    /// `wait_all` returns.
    ///
    /// # Safety
    ///
    /// The caller must ensure `data` is not modified between the call to
    /// `submit_write` and the corresponding `wait`/`wait_all`.
    unsafe fn submit_write(&self, pid: PageId, data: &[u8]) -> io::Result<Self::Token>;

    /// Enqueue an fsync.
    fn submit_sync(&self) -> io::Result<Self::Token>;

    /// Block until every in-flight operation has completed.
    fn wait_all(&self) -> io::Result<()>;

    /// Block until the specified operations have completed.
    fn wait(&self, tokens: &[Self::Token]) -> io::Result<()>;
}

// ---------------------------------------------------------------------------
// Blanket impls
// ---------------------------------------------------------------------------

/// Blanket impl so `Arc<T: PageStore>` can be used as a `Box<dyn PageStore>`.
impl<T: PageStore> PageStore for std::sync::Arc<T> {
    fn read_page(&self, pid: PageId, buf: &mut [u8]) -> io::Result<bool> {
        (**self).read_page(pid, buf)
    }
    fn write_page(&self, pid: PageId, data: &[u8]) -> io::Result<()> {
        (**self).write_page(pid, data)
    }
    fn allocate(&self, pid: PageId) -> io::Result<()> {
        (**self).allocate(pid)
    }
    fn sync(&self) -> io::Result<()> {
        (**self).sync()
    }
    fn len(&self) -> usize {
        (**self).len()
    }
    fn is_empty(&self) -> bool {
        (**self).is_empty()
    }
    fn next_page_id(&self) -> PageId {
        (**self).next_page_id()
    }
    fn raw_fd(&self) -> Option<std::os::fd::RawFd> {
        (**self).raw_fd()
    }
    fn drop_cache(&self) {
        (**self).drop_cache()
    }
    fn write_pages_and_sync(&self, pages: &[(PageId, &[u8])]) -> io::Result<()> {
        (**self).write_pages_and_sync(pages)
    }
    fn write_pages(&self, pages: &[(PageId, &[u8])]) -> io::Result<()> {
        (**self).write_pages(pages)
    }
}

/// In-memory page store backed by a HashMap. Used for testing and as
/// a placeholder until real disk I/O is implemented.
pub struct InMemoryPageStore {
    pages: Mutex<HashMap<PageId, Vec<u8>>>,
}

impl Default for InMemoryPageStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryPageStore {
    pub fn new() -> Self {
        InMemoryPageStore {
            pages: Mutex::new(HashMap::new()),
        }
    }
}

impl PageStore for InMemoryPageStore {
    fn read_page(&self, pid: PageId, buf: &mut [u8]) -> io::Result<bool> {
        validate_page_buf_len(pid, buf.len())?;
        let pages = self.pages.lock().unwrap();
        match pages.get(&pid) {
            Some(data) => {
                buf.copy_from_slice(data);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn write_page(&self, pid: PageId, data: &[u8]) -> io::Result<()> {
        validate_page_buf_len(pid, data.len())?;
        self.pages.lock().unwrap().insert(pid, data.to_vec());
        Ok(())
    }

    fn allocate(&self, pid: PageId) -> io::Result<()> {
        self.pages
            .lock()
            .unwrap()
            .insert(pid, vec![0u8; page_size(pid)]);
        Ok(())
    }

    fn sync(&self) -> io::Result<()> {
        Ok(()) // In-memory — nothing to flush.
    }

    fn len(&self) -> usize {
        self.pages
            .lock()
            .unwrap()
            .keys()
            .map(|&pid| page_end_base_page(pid) as usize)
            .max()
            .unwrap_or(0)
    }

    fn next_page_id(&self) -> PageId {
        let pages = self.pages.lock().unwrap();
        pages
            .keys()
            .map(|&pid| page_end_base_page(pid))
            .max()
            .map_or(1, |max| max + 1)
    }
}

/// Trivial `BatchPageStore` for `InMemoryPageStore`: operations are inline,
/// tokens are unit, wait is a no-op.
impl BatchPageStore for InMemoryPageStore {
    type Token = ();

    unsafe fn submit_read(&self, pid: PageId, buf: &mut [u8]) -> io::Result<Self::Token> {
        PageStore::read_page(self, pid, buf)?;
        Ok(())
    }

    unsafe fn submit_write(&self, pid: PageId, data: &[u8]) -> io::Result<Self::Token> {
        PageStore::write_page(self, pid, data)?;
        Ok(())
    }

    fn submit_sync(&self) -> io::Result<Self::Token> {
        PageStore::sync(self)?;
        Ok(())
    }

    fn wait_all(&self) -> io::Result<()> {
        Ok(())
    }

    fn wait(&self, _tokens: &[Self::Token]) -> io::Result<()> {
        Ok(())
    }
}

impl RecoveryPageStore for InMemoryPageStore {
    fn read_page(&self, pid: PageId, buf: &mut [u8]) -> io::Result<bool> {
        PageStore::read_page(self, pid, buf)
    }

    fn write_page(&self, pid: PageId, data: &[u8]) -> io::Result<()> {
        PageStore::write_page(self, pid, data)
    }

    fn allocate(&self, pid: PageId) -> io::Result<()> {
        PageStore::allocate(self, pid)
    }

    fn sync(&self) -> io::Result<()> {
        PageStore::sync(self)
    }

    fn next_page_id(&self) -> PageId {
        PageStore::next_page_id(self)
    }
}

// ---------------------------------------------------------------------------
// FilePageStore — single-file, pread/pwrite backed
// ---------------------------------------------------------------------------

// Header page layout (page 0):
//   bytes  0..4:   magic              (u32 LE) — 0x424F5854 ("BOXT")
//   bytes  4..6:   version            (u16 LE) — format version, currently 1
//   bytes  6..8:   reserved
//   bytes  8..16:  page_count         (u64 LE) — highest allocated page id
//   bytes 16..24:  checkpoint_lsn     (u64 LE) — LSN of last completed checkpoint
//   bytes 24..32:  user_meta_0        (u64 LE) — user meta slot 0 (0 = unset)
//   bytes 32..40:  user_meta_1        (u64 LE) — user meta slot 1 (0 = unset)
//   bytes 40..48:  user_meta_2        (u64 LE) — user meta slot 2 (0 = unset)

pub(crate) const HEADER_MAGIC: u32 = 0x424F5854; // "BOXT"
pub(crate) const HEADER_VERSION: u16 = 1;
pub(crate) const HEADER_MAGIC_OFF: usize = 0;
pub(crate) const HEADER_VERSION_OFF: usize = 4;
pub(crate) const HEADER_PAGE_COUNT_OFF: usize = 8;
pub(crate) const HEADER_CHECKPOINT_LSN_OFF: usize = 16;
pub(crate) const HEADER_USER_META_0_OFF: usize = 24;
pub(crate) const HEADER_USER_META_1_OFF: usize = 32;
pub(crate) const HEADER_USER_META_2_OFF: usize = 40;

pub(crate) fn build_header(
    page_count: u64,
    checkpoint_lsn: u64,
    user_meta_0: u64,
    user_meta_1: u64,
    user_meta_2: u64,
) -> [u8; PAGE_SIZE] {
    let mut hdr = [0u8; PAGE_SIZE];
    hdr[HEADER_MAGIC_OFF..4].copy_from_slice(&HEADER_MAGIC.to_le_bytes());
    hdr[HEADER_VERSION_OFF..6].copy_from_slice(&HEADER_VERSION.to_le_bytes());
    hdr[HEADER_PAGE_COUNT_OFF..16].copy_from_slice(&page_count.to_le_bytes());
    hdr[HEADER_CHECKPOINT_LSN_OFF..24].copy_from_slice(&checkpoint_lsn.to_le_bytes());
    hdr[HEADER_USER_META_0_OFF..32].copy_from_slice(&user_meta_0.to_le_bytes());
    hdr[HEADER_USER_META_1_OFF..40].copy_from_slice(&user_meta_1.to_le_bytes());
    hdr[HEADER_USER_META_2_OFF..48].copy_from_slice(&user_meta_2.to_le_bytes());
    hdr
}

pub(crate) fn validate_header(hdr: &[u8; PAGE_SIZE]) -> io::Result<(u64, u64, u64, u64, u64)> {
    let magic = u32::from_le_bytes(hdr[HEADER_MAGIC_OFF..4].try_into().unwrap());
    let version = u16::from_le_bytes(hdr[HEADER_VERSION_OFF..6].try_into().unwrap());
    if magic != HEADER_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad magic: expected 0x{HEADER_MAGIC:08X}, got 0x{magic:08X}"),
        ));
    }
    if version != HEADER_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported version: expected {HEADER_VERSION}, got {version}"),
        ));
    }
    let pc = u64::from_le_bytes(hdr[HEADER_PAGE_COUNT_OFF..16].try_into().unwrap());
    let cl = u64::from_le_bytes(hdr[HEADER_CHECKPOINT_LSN_OFF..24].try_into().unwrap());
    let tm = u64::from_le_bytes(hdr[HEADER_USER_META_0_OFF..32].try_into().unwrap());
    let cp = u64::from_le_bytes(hdr[HEADER_USER_META_1_OFF..40].try_into().unwrap());
    let extra = u64::from_le_bytes(hdr[HEADER_USER_META_2_OFF..48].try_into().unwrap());
    Ok((pc, cl, tm, cp, extra))
}

pub(crate) fn page_offset(pid: PageId) -> i64 {
    (physical_page_number(pid) as i64) * (PAGE_SIZE as i64)
}

/// 4096-aligned page buffer for O_DIRECT I/O.
#[repr(C, align(4096))]
struct AlignedPage([u8; PAGE_SIZE]);

#[cfg(not(miri))]
fn page_store_direct_io_enabled() -> bool {
    matches!(
        std::env::var("PAGEBOX_PAGE_STORE_DIRECT_IO")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

#[cfg(not(miri))]
fn open_buffered_page_store_fd(c_path: *const libc::c_char) -> io::Result<std::os::fd::RawFd> {
    let fd = unsafe { libc::open(c_path, libc::O_RDWR | libc::O_CREAT, 0o644) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(fd)
}

#[cfg(all(not(miri), target_os = "linux"))]
fn open_direct_page_store_fd(c_path: *const libc::c_char) -> io::Result<std::os::fd::RawFd> {
    let fd = unsafe { libc::open(c_path, libc::O_RDWR | libc::O_CREAT | libc::O_DIRECT, 0o644) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(fd)
}

#[cfg(all(not(miri), not(target_os = "linux")))]
fn open_direct_page_store_fd(_c_path: *const libc::c_char) -> io::Result<std::os::fd::RawFd> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "page-store direct I/O is only available on Linux",
    ))
}

#[cfg(all(not(miri), target_os = "linux"))]
fn advise_drop_cache(fd: std::os::fd::RawFd) {
    unsafe {
        libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

#[cfg(all(not(miri), not(target_os = "linux")))]
fn advise_drop_cache(_fd: std::os::fd::RawFd) {}

#[cfg(not(miri))]
fn open_page_store_fd(c_path: *const libc::c_char) -> io::Result<(std::os::fd::RawFd, bool)> {
    if !page_store_direct_io_enabled() {
        return open_buffered_page_store_fd(c_path).map(|fd| (fd, false));
    }

    if let Ok(fd) = open_direct_page_store_fd(c_path) {
        return Ok((fd, true));
    }

    open_buffered_page_store_fd(c_path).map(|fd| (fd, false))
}

/// File-backed page store. Pages are stored at offset `page_offset(pid)` in a
/// single file. Uses `pread`/`pwrite` for positioned I/O (no fd-level mutex
/// needed — the kernel handles concurrent positioned reads/writes).
///
/// Uses buffered I/O by default. Set `PAGEBOX_PAGE_STORE_DIRECT_IO=1` to try
/// O_DIRECT and fall back to buffered I/O if O_DIRECT is not supported.
///
/// Page 0 is reserved as a header page. User pages start at pid 1 (which is
/// what BufferPool already does via `next_page_id` starting at 1).
#[cfg(not(miri))]
#[derive(Debug)]
pub struct FilePageStore {
    fd: std::os::fd::RawFd,
    #[allow(dead_code)]
    direct_io: bool,
    page_count: AtomicU64,
    checkpoint_lsn: AtomicU64,
    user_meta_0: AtomicU64,
    user_meta_1: AtomicU64,
    user_meta_2: AtomicU64,
    alloc_lock: Mutex<()>,
}

#[cfg(not(miri))]
impl FilePageStore {
    /// Whether this store successfully opened its data file with `O_DIRECT`.
    pub fn direct_io_enabled(&self) -> bool {
        self.direct_io
    }

    /// Create or open a page store file at `path`.
    ///
    /// Uses buffered I/O by default. Set `PAGEBOX_PAGE_STORE_DIRECT_IO=1`
    /// to try O_DIRECT and fall back to buffered I/O if unsupported.
    pub fn open(path: &std::path::Path) -> io::Result<Self> {
        let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

        let (fd, direct_io) = open_page_store_fd(c_path.as_ptr())?;

        let file_size = {
            let mut stat: libc::stat = unsafe { std::mem::zeroed() };
            if unsafe { libc::fstat(fd, &mut stat) } != 0 {
                unsafe { libc::close(fd) };
                return Err(io::Error::last_os_error());
            }
            stat.st_size as u64
        };

        let (page_count, checkpoint_lsn, user_meta_0, user_meta_1, user_meta_2) =
            if file_size >= PAGE_SIZE as u64 {
                // Read and validate header page (aligned for O_DIRECT).
                let mut hdr_buf = AlignedPage([0u8; PAGE_SIZE]);
                pread_exact(fd, &mut hdr_buf.0, 0).inspect_err(|_| {
                    unsafe { libc::close(fd) };
                })?;

                validate_header(&hdr_buf.0).inspect_err(|_| {
                    unsafe { libc::close(fd) };
                })?
            } else {
                // Fresh file — write initial header (aligned for O_DIRECT).
                let hdr_data = Self::build_header(0, 0, 0, 0, 0);
                let mut hdr = AlignedPage([0u8; PAGE_SIZE]);
                hdr.0.copy_from_slice(&hdr_data);
                pwrite_exact(fd, &hdr.0, 0).inspect_err(|_| {
                    unsafe { libc::close(fd) };
                })?;
                (0, 0, 0, 0, 0)
            };

        Ok(FilePageStore {
            fd,
            direct_io,
            page_count: AtomicU64::new(page_count),
            checkpoint_lsn: AtomicU64::new(checkpoint_lsn),
            user_meta_0: AtomicU64::new(user_meta_0),
            user_meta_1: AtomicU64::new(user_meta_1),
            user_meta_2: AtomicU64::new(user_meta_2),
            alloc_lock: Mutex::new(()),
        })
    }

    fn build_header(
        page_count: u64,
        checkpoint_lsn: u64,
        user_meta_0: u64,
        user_meta_1: u64,
        user_meta_2: u64,
    ) -> [u8; PAGE_SIZE] {
        build_header(
            page_count,
            checkpoint_lsn,
            user_meta_0,
            user_meta_1,
            user_meta_2,
        )
    }

    pub fn crash(self) {
        let this = std::mem::ManuallyDrop::new(self);
        unsafe {
            libc::close(this.fd);
        }
    }

    fn page_offset(pid: PageId) -> i64 {
        page_offset(pid)
    }

    /// Flush the header page with the current page count and checkpoint LSN.
    fn sync_header(&self) -> io::Result<()> {
        let count = self.page_count.load(Ordering::Relaxed);
        let ckpt = self.checkpoint_lsn.load(Ordering::Relaxed);
        let tmeta = self.user_meta_0.load(Ordering::Relaxed);
        let cmeta = self.user_meta_1.load(Ordering::Relaxed);
        let extra = self.user_meta_2.load(Ordering::Relaxed);
        let hdr_data = Self::build_header(count, ckpt, tmeta, cmeta, extra);
        let mut hdr = AlignedPage([0u8; PAGE_SIZE]);
        hdr.0.copy_from_slice(&hdr_data);
        pwrite_exact(self.fd, &hdr.0, 0)
    }

    /// Get the persisted checkpoint LSN.
    pub fn checkpoint_lsn(&self) -> u64 {
        self.checkpoint_lsn.load(Ordering::Relaxed)
    }

    /// Set the checkpoint LSN (will be persisted on next sync_header/sync).
    pub fn set_checkpoint_lsn(&self, lsn: u64) {
        self.checkpoint_lsn.store(lsn, Ordering::Relaxed);
    }

    /// Get the persisted table metadata page ID (0 = no table).
    pub fn user_meta_0(&self) -> u64 {
        self.user_meta_0.load(Ordering::Relaxed)
    }

    /// Set the table metadata page ID.
    pub fn set_user_meta_0(&self, pid: u64) {
        self.user_meta_0.store(pid, Ordering::Relaxed);
    }

    /// Get the persisted catalog page ID (0 = no catalog).
    pub fn user_meta_1(&self) -> u64 {
        self.user_meta_1.load(Ordering::Relaxed)
    }

    /// Set the catalog page ID.
    pub fn set_user_meta_1(&self, pid: u64) {
        self.user_meta_1.store(pid, Ordering::Relaxed);
    }

    pub fn user_meta_2(&self) -> u64 {
        self.user_meta_2.load(Ordering::Relaxed)
    }

    pub fn set_user_meta_2(&self, value: u64) {
        self.user_meta_2.store(value, Ordering::Relaxed);
    }
}

#[cfg(not(miri))]
impl PageStore for FilePageStore {
    fn read_page(&self, pid: PageId, buf: &mut [u8]) -> io::Result<bool> {
        validate_page_buf_len(pid, buf.len())?;
        let off = Self::page_offset(pid);
        match pread_exact(self.fd, buf, off) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
            Err(e) => Err(e),
        }
    }

    fn write_page(&self, pid: PageId, data: &[u8]) -> io::Result<()> {
        validate_page_buf_len(pid, data.len())?;
        let off = Self::page_offset(pid);
        pwrite_exact(self.fd, data, off)
    }

    fn allocate(&self, pid: PageId) -> io::Result<()> {
        let _guard = self.alloc_lock.lock().unwrap();
        let current = self.page_count.load(Ordering::Relaxed);
        let new_count = page_end_base_page(pid);
        if new_count > current {
            let new_end = Self::page_offset(pid) + page_size(pid) as i64;
            if unsafe { libc::ftruncate(self.fd, new_end as libc::off_t) } != 0 {
                return Err(io::Error::last_os_error());
            }
            self.page_count.store(new_count, Ordering::Relaxed);
            self.sync_header()?;
        }

        Ok(())
    }

    fn sync(&self) -> io::Result<()> {
        self.sync_header()?;
        if unsafe { libc::fsync(self.fd) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.page_count.load(Ordering::Relaxed) as usize
    }

    fn next_page_id(&self) -> PageId {
        self.page_count.load(Ordering::Relaxed) + 1
    }

    fn raw_fd(&self) -> Option<std::os::fd::RawFd> {
        Some(self.fd)
    }

    fn drop_cache(&self) {
        advise_drop_cache(self.fd);
    }
}

/// Trivial `BatchPageStore` for `FilePageStore`: operations are inline
/// pread/pwrite calls, tokens are unit, wait is a no-op.
#[cfg(not(miri))]
impl BatchPageStore for FilePageStore {
    type Token = ();

    unsafe fn submit_read(&self, pid: PageId, buf: &mut [u8]) -> io::Result<Self::Token> {
        PageStore::read_page(self, pid, buf)?;
        Ok(())
    }

    unsafe fn submit_write(&self, pid: PageId, data: &[u8]) -> io::Result<Self::Token> {
        PageStore::write_page(self, pid, data)?;
        Ok(())
    }

    fn submit_sync(&self) -> io::Result<Self::Token> {
        PageStore::sync(self)?;
        Ok(())
    }

    fn wait_all(&self) -> io::Result<()> {
        Ok(())
    }

    fn wait(&self, _tokens: &[Self::Token]) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(not(miri))]
impl Drop for FilePageStore {
    fn drop(&mut self) {
        // Best-effort: ignore errors during drop.
        let _ = self.sync_header();
        unsafe {
            libc::fsync(self.fd);
            libc::close(self.fd);
        }
    }
}

#[cfg(not(miri))]
impl RecoveryPageStore for FilePageStore {
    fn read_page(&self, pid: PageId, buf: &mut [u8]) -> io::Result<bool> {
        PageStore::read_page(self, pid, buf)
    }

    fn write_page(&self, pid: PageId, data: &[u8]) -> io::Result<()> {
        PageStore::write_page(self, pid, data)
    }

    fn allocate(&self, pid: PageId) -> io::Result<()> {
        PageStore::allocate(self, pid)
    }

    fn sync(&self) -> io::Result<()> {
        PageStore::sync(self)
    }

    fn next_page_id(&self) -> PageId {
        PageStore::next_page_id(self)
    }
}

// ---------------------------------------------------------------------------
// Helpers: retry-loop wrappers for partial pread/pwrite
// ---------------------------------------------------------------------------

#[cfg(not(miri))]
fn pread_exact(fd: std::os::fd::RawFd, buf: &mut [u8], offset: i64) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        let n = unsafe {
            libc::pread(
                fd,
                buf[done..].as_mut_ptr() as *mut libc::c_void,
                buf.len() - done,
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
                format!("pread: eof at offset {} (got {done}/{})", offset, buf.len()),
            ));
        }
        done += n as usize;
    }
    Ok(())
}

#[cfg(not(miri))]
fn pwrite_exact(fd: std::os::fd::RawFd, data: &[u8], offset: i64) -> io::Result<()> {
    let mut done = 0usize;
    while done < data.len() {
        let n = unsafe {
            libc::pwrite(
                fd,
                data[done..].as_ptr() as *const libc::c_void,
                data.len() - done,
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
                format!(
                    "pwrite: zero bytes at offset {} (wrote {done}/{})",
                    offset,
                    data.len()
                ),
            ));
        }
        done += n as usize;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(not(miri))]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("pagebox_test_{name}_{}", std::process::id()));
        p
    }

    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn page_roundtrips_through_file_store() {
        let path = tmp_path("page_roundtrip");
        let _c = Cleanup(path.clone());
        // With a single page class, pid 1 maps directly to page number 1.
        let pid: PageId = 1;

        {
            let store = FilePageStore::open(&path).unwrap();
            PageStore::allocate(&store, pid).unwrap();

            let mut data = vec![0u8; PAGE_SIZE];
            data[0] = 7;
            data[PAGE_SIZE - 1] = 9;
            PageStore::write_page(&store, pid, &data).unwrap();
            store.set_user_meta_2(37);
            PageStore::sync(&store).unwrap();

            assert_eq!(
                store.len(),
                1,
                "page allocation should consume one base page"
            );
            assert_eq!(
                PageStore::next_page_id(&store),
                2,
                "next page id should advance past the allocated page"
            );
        }

        {
            let store = FilePageStore::open(&path).unwrap();
            let mut buf = vec![0u8; PAGE_SIZE];

            assert!(
                PageStore::read_page(&store, pid, &mut buf).unwrap(),
                "page should be readable after reopen"
            );
            assert_eq!(buf[0], 7, "first byte should roundtrip");
            assert_eq!(
                store.user_meta_2(),
                37,
                "third user-meta slot should roundtrip"
            );
            assert_eq!(
                buf[PAGE_SIZE - 1],
                9,
                "last byte of the page should roundtrip"
            );
        }
    }

    #[test]
    fn bad_magic_rejected() {
        let path = tmp_path("bad_magic");
        let _c = Cleanup(path.clone());

        // Write a file with wrong magic.
        std::fs::write(&path, [0u8; PAGE_SIZE]).unwrap();
        let err = FilePageStore::open(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("bad magic"));
    }

    #[test]
    fn concurrent_writes() {
        let path = tmp_path("concurrent_writes");
        let _c = Cleanup(path.clone());

        let store = Arc::new(FilePageStore::open(&path).unwrap());
        let n_threads = 4usize;
        let per_thread = 20usize;

        for pid in 1..=(n_threads * per_thread) as u64 {
            store.allocate(pid).unwrap();
        }

        let barrier = Arc::new(Barrier::new(n_threads));
        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let store = store.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    for i in 0..per_thread {
                        let pid = (t * per_thread + i + 1) as u64;
                        let mut data = [0u8; PAGE_SIZE];
                        data[0] = (pid & 0xFF) as u8;
                        data[1] = ((pid >> 8) & 0xFF) as u8;
                        store.write_page(pid, &data).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        for pid in 1..=(n_threads * per_thread) as u64 {
            let mut buf = [0u8; PAGE_SIZE];
            assert!(store.read_page(pid, &mut buf).unwrap());
            assert_eq!(buf[0], (pid & 0xFF) as u8);
            assert_eq!(buf[1], ((pid >> 8) & 0xFF) as u8);
        }
    }

    #[test]
    fn buffer_pool_integration() {
        use crate::buffer_pool::{BufferPool, NoLatches};

        let path = tmp_path("bp_integration");
        let _c = Cleanup(path.clone());

        let store = FilePageStore::open(&path).unwrap();
        let pool = BufferPool::with_store(4, Box::new(store));

        let swips: Vec<_> = (0..20).map(|_| pool.allocate_page()).collect();

        for swip in &swips {
            let mut bf = unsafe { pool.fix_frame(swip, NoLatches::new(&pool)) }.exclusive();
            let pid = bf.pid();
            bf.page_mut()[0] = (pid & 0xFF) as u8;
            bf.page_mut()[1] = ((pid >> 8) & 0xFF) as u8;
            bf.mark_dirty();
        }

        for swip in &swips {
            let bf = unsafe { pool.fix_frame(swip, NoLatches::new(&pool)) };
            let pid = bf.pid();
            assert_eq!(bf.page()[0], (pid & 0xFF) as u8);
            assert_eq!(bf.page()[1], ((pid >> 8) & 0xFF) as u8);
        }
    }

    #[test]
    fn reopen_buffer_pool_no_pid_reuse() {
        use crate::buffer_pool::{BufferPool, NoLatches};
        use std::sync::atomic::Ordering;

        let path = tmp_path("bp_reopen_pid");
        let _c = Cleanup(path.clone());

        // First session: write 5 pages with known data.
        let last_pid;
        {
            let store = FilePageStore::open(&path).unwrap();
            let pool = BufferPool::with_store(8, Box::new(store));

            for _ in 0..5u64 {
                let (pid, bf) = pool.allocate_and_fix(NoLatches::new(&pool));
                let mut bf = bf.exclusive();
                bf.page_mut()[0] = (pid & 0xFF) as u8;
                bf.page_mut()[1] = 0xAA;
                bf.mark_dirty();
            }
            // Record the last pid allocated in this session.
            last_pid = pool.allocate_page().load(Ordering::Relaxed).as_page_id();
            // That allocate created pid 6; we don't need to write it.
        } // drop → fsync + close

        // Second session: reopen and allocate more pages.
        {
            let store = FilePageStore::open(&path).unwrap();
            let pool = BufferPool::with_store(8, Box::new(store));

            let (new_pid, bf) = pool.allocate_and_fix(NoLatches::new(&pool));

            // The new pid must be strictly after everything from session 1.
            assert!(
                new_pid > last_pid,
                "reopened pool allocated pid {new_pid}, expected > {last_pid}"
            );

            // Write to the new page and verify it doesn't clobber old data.
            let mut bf = bf.exclusive();
            bf.page_mut()[0] = 0xFF;
            bf.page_mut()[1] = 0xBB;
            bf.mark_dirty();

            // Verify an old page is still intact by reading from the store.
            // Page 1 should still have its original data.
            let bf = unsafe { pool.fix_orphan_frame(1, NoLatches::new(&pool)) };
            assert_eq!(bf.page()[0], 1);
            assert_eq!(bf.page()[1], 0xAA);
        }
    }

    #[test]
    fn truncated_file_reopen_preserves_intact_pages() {
        let path = tmp_path("truncated_file");
        let _c = Cleanup(path.clone());

        // Write a valid header + 5 pages.
        {
            let store = FilePageStore::open(&path).unwrap();
            for pid in 1..=5u64 {
                PageStore::allocate(&store, pid).unwrap();
                let mut data = [0u8; PAGE_SIZE];
                data[0] = pid as u8;
                PageStore::write_page(&store, pid, &data).unwrap();
            }
            PageStore::sync(&store).unwrap();
        }

        // Truncate to header + 2.5 pages (torn last page).
        let truncated_len = PAGE_SIZE + PAGE_SIZE * 2 + PAGE_SIZE / 2;
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(truncated_len as u64)
            .unwrap();

        // Reopen — the store opens but pages past the truncation point
        // are unreadable; pages before it should still be intact.
        let store = FilePageStore::open(&path).unwrap();

        // Page 1 and 2 should be readable (before truncation).
        let mut buf = [0u8; PAGE_SIZE];
        assert!(
            PageStore::read_page(&store, 1, &mut buf).unwrap(),
            "page 1 should survive truncation"
        );
        assert_eq!(buf[0], 1);
        assert!(
            PageStore::read_page(&store, 2, &mut buf).unwrap(),
            "page 2 should survive truncation"
        );
        assert_eq!(buf[0], 2);

        // Page 4 and 5 are past the truncation point — read should fail
        // or return false.
        let result4 = PageStore::read_page(&store, 4, &mut buf);
        match result4 {
            Ok(false) | Err(_) => { /* expected — page was truncated */ }
            Ok(true) => {
                // If the read succeeded, the data should not match
                // the original (pages are zeroed or torn).
                assert_ne!(buf[0], 4, "truncated page should not have original data");
            }
        }
    }

    #[test]
    fn write_page_to_unallocated_pid_extends_file() {
        let path = tmp_path("write_unallocated");
        let _c = Cleanup(path.clone());

        let store = FilePageStore::open(&path).unwrap();
        // With a single page class, pid 99 maps directly to page number 99.
        let pid: PageId = 99;

        // Writing without allocating first — the store should extend or accept.
        let data = [0xAB; PAGE_SIZE];
        let result = PageStore::write_page(&store, pid, &data);

        match result {
            Ok(()) => {
                let mut buf = [0u8; PAGE_SIZE];
                assert!(
                    PageStore::read_page(&store, pid, &mut buf).unwrap(),
                    "written page should be readable"
                );
                assert_eq!(buf[0], 0xAB);
            }
            Err(e) => {
                assert!(
                    e.kind() == io::ErrorKind::InvalidInput || e.kind() == io::ErrorKind::NotFound,
                    "unexpected error kind for unallocated write: {e}"
                );
            }
        }
    }
}
