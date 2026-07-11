//! Standalone durable key-value store built on the pagebox substrate.
//!
//! Composes four substrate crates (`pagebox-btree`, `pagebox-storage`,
//! `pagebox-wal`, `pagebox-frame-kernel`) into a crash-safe KV engine with
//! swizzled-pointer buffer-pool, file-backed page store, WAL group commit,
//! and a concurrent B+tree. Exposed as a library so that tooling (e.g. the
//! `kvbench` harness) can drive it alongside competitor engines through a
//! uniform adapter trait.

use std::ops::Bound;
use std::path::Path;

use pagebox_btree::BTree;
use pagebox_storage::buffer_pool::{BufferPool, BufferPoolHandle};
use pagebox_storage::page_header::read_page_lsn;
use pagebox_storage::page_store::{FilePageStore, PageStore};
use pagebox_wal::Wal;

/// Default buffer-pool frame count.
pub const DEFAULT_POOL_FRAMES: usize = 1024;

/// Default domain ID for the B+tree data-structure registry.
pub const DEFAULT_DOMAIN_ID: u16 = 1;

/// Write durability mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncMode {
    /// Writes return after the page is modified; WAL flush is asynchronous.
    /// Mirrors the WAL's relaxed commit mode and RocksDB's default buffering.
    #[default]
    Relaxed,
    /// Every write blocks until the WAL has flushed it. Mirrors the WAL's
    /// strict commit mode and the former `Put --sync` CLI flag.
    Strict,
}

/// Tree backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TreeBackend {
    #[default]
    BPlusTree,
}

/// Tunable knobs for [`KvStore::open_with`].
#[derive(Debug, Clone)]
pub struct KvStoreOptions {
    /// Number of buffer-pool frames (resident-page budget).
    pub pool_frames: usize,
    /// B+tree data-structure ID for the parent-finder registry.
    pub domain_id: u16,
    /// Write durability mode.
    pub sync_mode: SyncMode,
    /// Tree backend.
    pub tree_backend: TreeBackend,
}

impl Default for KvStoreOptions {
    fn default() -> Self {
        Self {
            pool_frames: DEFAULT_POOL_FRAMES,
            domain_id: DEFAULT_DOMAIN_ID,
            sync_mode: SyncMode::default(),
            tree_backend: TreeBackend::default(),
        }
    }
}

impl KvStoreOptions {
    /// Create a builder with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the buffer-pool frame count.
    #[must_use]
    pub fn pool_frames(mut self, n: usize) -> Self {
        self.pool_frames = n;
        self
    }

    /// Set the B+tree domain ID.
    #[must_use]
    pub fn domain_id(mut self, id: u16) -> Self {
        self.domain_id = id;
        self
    }

    /// Set the write durability mode.
    #[must_use]
    pub fn sync_mode(mut self, mode: SyncMode) -> Self {
        self.sync_mode = mode;
        self
    }

    /// Set the tree backend.
    #[must_use]
    pub fn tree_backend(mut self, backend: TreeBackend) -> Self {
        self.tree_backend = backend;
        self
    }
}

/// Durable key-value store backed by the pagebox substrate.
pub struct KvStore {
    pool: BufferPoolHandle,
    tree: std::sync::Arc<BTree>,
    wal: std::sync::Arc<Wal>,
    store: std::sync::Arc<FilePageStore>,
    sync_mode: SyncMode,
}

impl KvStore {
    /// Open with default options ([`KvStoreOptions::default`]).
    pub fn open<P: AsRef<Path>>(dir: P) -> std::io::Result<Self> {
        Self::open_with(dir, &KvStoreOptions::default())
    }

    /// Open with explicit options. Runs WAL recovery, then either creates a
    /// fresh B+tree or reopens an existing one from the page-store user-meta
    /// slots.
    pub fn open_with<P: AsRef<Path>>(dir: P, opts: &KvStoreOptions) -> std::io::Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        let data_path = dir.join("kvstore.data");
        let wal_path = dir.join("kvstore.wal");

        let store = std::sync::Arc::new(FilePageStore::open(&data_path)?);
        let wal = Wal::open_opts(&wal_path)?;

        let checkpoint_lsn = store.checkpoint_lsn();
        let report = wal.recover(&*store, checkpoint_lsn, read_page_lsn)?;
        if report.max_lsn > checkpoint_lsn {
            store.sync()?;
            store.set_checkpoint_lsn(report.max_lsn);
            store.sync()?;
            wal.reset()?;
        }
        let effective_checkpoint = store.checkpoint_lsn();
        wal.advance_lsn_past(effective_checkpoint);

        let wal = std::sync::Arc::new(wal);
        let mut pool = BufferPool::with_store(opts.pool_frames, Box::new(store.clone()));
        pool.set_wal(wal.clone());
        let pool: BufferPoolHandle = std::sync::Arc::new(pool).into();

        let root = store.user_meta_0();
        let height = store.user_meta_1() as u32;
        let reachable_pages = store.user_meta_2();
        let tree = std::sync::Arc::new(if root == 0 {
            let t = BTree::new(pool.clone(), opts.domain_id);
            store.set_user_meta_0(t.root_page_id());
            store.set_user_meta_1(0);
            store.set_user_meta_2(t.reachable_page_count().unwrap());
            store.sync()?;
            t
        } else {
            BTree::open_with_page_count(pool.clone(), root, height, reachable_pages, opts.domain_id)
        });
        pool.register_dt(opts.domain_id, tree.clone());

        Ok(Self {
            pool,
            tree,
            wal,
            store,
            sync_mode: opts.sync_mode,
        })
    }

    /// Insert or update a key-value pair. Returns `true` if the key was
    /// newly inserted, `false` if an existing value was updated.
    pub fn put(&self, key: &[u8], value: &[u8]) -> bool {
        let inserted = self.tree.upsert(key, value);
        if self.sync_mode == SyncMode::Strict {
            self.wal.flush();
        }
        inserted
    }

    /// Look up a key, returning an owned copy of the value.
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.tree.lookup(key)
    }

    /// Delete a key. Returns `true` if the key was present.
    pub fn del(&self, key: &[u8]) -> bool {
        self.tree.remove(key)
    }

    /// Ordered scan over all entries. Calls `f` per `(key, value)` pair.
    pub fn scan_all<F: FnMut(&[u8], &[u8])>(&self, f: F) {
        self.tree.scan(f);
    }

    /// Ordered range scan over `[start, end)` (half-open). Calls `f` per
    /// `(key, value)` pair whose key falls within the range.
    pub fn scan_range<F: FnMut(&[u8], &[u8])>(&self, start: &[u8], end: &[u8], mut f: F) {
        self.tree
            .scan_range(Bound::Included(start), Bound::Excluded(end), &mut f);
    }

    /// Flush the WAL (durable fsync) without flushing dirty pages. This is
    /// the minimum flush for strict-durability writes.
    pub fn flush_wal(&self) {
        self.wal.flush();
    }

    /// Explicit durable flush: WAL fsync + dirty-page writeback. Does not
    /// advance the checkpoint or reset the WAL.
    pub fn sync(&self) -> std::io::Result<()> {
        self.wal.flush();
        self.pool.flush()?;
        Ok(())
    }

    /// Checkpoint: flush WAL + dirty pages, advance user-meta slots, reset
    /// the WAL. This is the full recovery-point establishment.
    pub fn checkpoint(&self) -> std::io::Result<()> {
        let checkpoint_lsn = self.wal.flush();
        self.pool.flush()?;
        self.store.set_user_meta_0(self.tree.root_page_id());
        self.store.set_user_meta_1(self.tree.height() as u64);
        self.store
            .set_user_meta_2(self.tree.reachable_page_count().unwrap());
        self.store.set_checkpoint_lsn(checkpoint_lsn);
        self.store.sync()?;
        self.wal.reset()?;
        Ok(())
    }

    /// Return the B+tree's root page ID (for persistence / tooling).
    pub fn root_page_id(&self) -> u64 {
        self.tree.root_page_id()
    }

    /// Return the B+tree height (for persistence / tooling).
    pub fn height(&self) -> u32 {
        self.tree.height()
    }

    /// Configured buffer-pool capacity in 64 KiB pages.
    pub fn cache_capacity_pages(&self) -> usize {
        self.pool.num_frames()
    }

    /// Approximate number of occupied buffer-pool pages.
    pub fn cache_used_pages(&self) -> usize {
        self.pool.num_occupied_estimate()
    }

    /// Buffer-pool evictions since this store was opened.
    pub fn cache_evictions(&self) -> u64 {
        self.pool.eviction_count()
    }

    /// Page-store loads since this store was opened.
    pub fn cache_misses(&self) -> u64 {
        self.pool.page_load_count()
    }

    /// Number of allocated pages in the persistent page store.
    pub fn persisted_pages(&self) -> usize {
        self.pool.num_pages_on_disk()
    }

    /// Number of pages currently reachable from the B+tree root.
    pub fn live_tree_pages(&self) -> usize {
        self.tree.reachable_page_count().unwrap() as usize
    }

    pub fn buffer_pool_diagnostic_stats(
        &self,
    ) -> pagebox_storage::buffer_pool::BufferPoolDiagnosticStats {
        self.pool.diagnostic_stats()
    }

    pub fn btree_diagnostic_stats(&self) -> pagebox_btree::BTreeDiagnosticStats {
        self.tree.diagnostic_stats()
    }

    /// Whether the page-store data file is using direct I/O.
    pub fn direct_io_enabled(&self) -> bool {
        self.store.direct_io_enabled()
    }
}

impl Drop for KvStore {
    fn drop(&mut self) {
        self.pool.unregister_dt(self.tree.domain_id());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use super::*;

    #[test]
    fn parent_finder_registration_follows_store_lifetime() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = KvStore::open(dir.path()).unwrap();
        let pool = store.pool.clone();
        let domain_id = store.tree.domain_id();

        assert!(
            pool.has_registered_dt(domain_id),
            "open store must register its B-tree for eviction parent lookup"
        );
        drop(store);
        assert!(
            !pool.has_registered_dt(domain_id),
            "dropping the store must break the pool/tree registration cycle"
        );
    }

    #[test]
    #[ignore = "expensive file-backed eviction-pressure regression"]
    fn concurrent_file_backed_growth_completes_under_eviction_pressure() {
        const KEYS: u64 = 65_536;
        let threads = std::env::var("PAGEBOX_KVSTORE_PRESSURE_THREADS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(8usize);

        let dir = tempfile::TempDir::new().unwrap();
        let store = Arc::new(
            KvStore::open_with(
                dir.path(),
                &KvStoreOptions::default()
                    .pool_frames(1_024)
                    .sync_mode(SyncMode::Relaxed),
            )
            .unwrap(),
        );
        let next = Arc::new(AtomicU64::new(0));
        let in_flight = Arc::new(
            (0..threads)
                .map(|_| AtomicU64::new(u64::MAX))
                .collect::<Vec<_>>(),
        );
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let handles: Vec<_> = (0..threads)
            .map(|worker_idx| {
                let store = store.clone();
                let next = next.clone();
                let in_flight = in_flight.clone();
                let done_tx = done_tx.clone();
                std::thread::spawn(move || {
                    let value = [0xa5; 2_048];
                    loop {
                        let key = next.fetch_add(1, Ordering::Relaxed);
                        if key >= KEYS {
                            break;
                        }
                        in_flight[worker_idx].store(key, Ordering::Relaxed);
                        assert!(store.put(&key.to_be_bytes(), &value));
                        in_flight[worker_idx].store(u64::MAX, Ordering::Relaxed);
                    }
                    done_tx.send(()).unwrap();
                })
            })
            .collect();
        drop(done_tx);

        let timeout_secs = std::env::var("PAGEBOX_KVSTORE_PRESSURE_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(30);
        for _ in 0..threads {
            if done_rx
                .recv_timeout(Duration::from_secs(timeout_secs))
                .is_err()
            {
                eprintln!(
                    "file-backed growth did not complete: next_key={} in_flight={:?} \
                     tree={:?} pool={:?} evictions={} occupied={} unlinked={}",
                    next.load(Ordering::Relaxed),
                    in_flight
                        .iter()
                        .map(|key| key.load(Ordering::Relaxed))
                        .collect::<Vec<_>>(),
                    store.btree_diagnostic_stats(),
                    store.buffer_pool_diagnostic_stats(),
                    store.cache_evictions(),
                    store.cache_used_pages(),
                    store.pool.num_unlinked_resident_frames(),
                );
                std::process::abort();
            }
        }
        for handle in handles {
            handle.join().unwrap();
        }

        let mut count = 0usize;
        store.scan_all(|_, _| count += 1);
        assert_eq!(count, KEYS as usize);
    }
}
