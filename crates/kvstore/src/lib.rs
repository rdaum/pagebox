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

use pagebox_betree::CowBeTree;
use pagebox_btree::BTree;
use pagebox_storage::buffer_pool::{BufferPool, BufferPoolHandle};
use pagebox_storage::page_header::read_page_lsn;
use pagebox_storage::page_store::{FilePageStore, PageStore};
use pagebox_wal::Wal;

pub const DEFAULT_POOL_FRAMES: usize = 1024;

pub const DEFAULT_DOMAIN_ID: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncMode {
    #[default]
    Relaxed,
    Strict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TreeBackend {
    #[default]
    BPlusTree,
    BeTree,
    BeTreeNoWal,
}

#[derive(Debug, Clone)]
pub struct KvStoreOptions {
    pub pool_frames: usize,
    pub domain_id: u16,
    pub sync_mode: SyncMode,
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
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn pool_frames(mut self, n: usize) -> Self {
        self.pool_frames = n;
        self
    }

    #[must_use]
    pub fn domain_id(mut self, id: u16) -> Self {
        self.domain_id = id;
        self
    }

    #[must_use]
    pub fn sync_mode(mut self, mode: SyncMode) -> Self {
        self.sync_mode = mode;
        self
    }

    #[must_use]
    pub fn tree_backend(mut self, backend: TreeBackend) -> Self {
        self.tree_backend = backend;
        self
    }
}

enum TreeHandle {
    BPlusTree(BTree),
    BeTree(CowBeTree),
    BeTreeNoWal(CowBeTree),
}

impl TreeHandle {
    fn root_page_id(&self) -> u64 {
        match self {
            TreeHandle::BPlusTree(t) => t.root_page_id(),
            TreeHandle::BeTree(t) | TreeHandle::BeTreeNoWal(t) => t.root_page_id(),
        }
    }

    fn height(&self) -> u32 {
        match self {
            TreeHandle::BPlusTree(t) => t.height(),
            TreeHandle::BeTree(t) | TreeHandle::BeTreeNoWal(t) => t.height(),
        }
    }

    #[allow(dead_code)]
    fn needs_wal(&self) -> bool {
        !matches!(self, TreeHandle::BeTreeNoWal(_))
    }

    fn flush_for_sync(
        &self,
        pool: &BufferPoolHandle,
        store: &FilePageStore,
    ) -> std::io::Result<()> {
        match self {
            TreeHandle::BeTreeNoWal(t) => {
                t.flush_all()
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                pool.flush()?;
                store.sync()?;
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

pub struct KvStore {
    pool: BufferPoolHandle,
    tree: TreeHandle,
    wal: Option<std::sync::Arc<Wal>>,
    store: std::sync::Arc<FilePageStore>,
    sync_mode: SyncMode,
}

impl KvStore {
    /// Open with default options ([`KvStoreOptions::default`]).
    pub fn open<P: AsRef<Path>>(dir: P) -> std::io::Result<Self> {
        Self::open_with(dir, &KvStoreOptions::default())
    }

    pub fn open_with<P: AsRef<Path>>(dir: P, opts: &KvStoreOptions) -> std::io::Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        let data_path = dir.join("kvstore.data");
        let store = std::sync::Arc::new(FilePageStore::open(&data_path)?);

        let no_wal = matches!(opts.tree_backend, TreeBackend::BeTreeNoWal);
        let wal_path = dir.join("kvstore.wal");

        let wal = if no_wal {
            None
        } else {
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
            Some(std::sync::Arc::new(wal))
        };

        let mut pool = BufferPool::with_store(opts.pool_frames, Box::new(store.clone()));
        if let Some(ref wal) = wal {
            pool.set_wal(wal.clone());
        }
        let pool: BufferPoolHandle = std::sync::Arc::new(pool).into();

        let root = store.user_meta_0();
        let tree = match opts.tree_backend {
            TreeBackend::BPlusTree => {
                let height = store.user_meta_1() as u32;
                if root == 0 {
                    let t = BTree::new(pool.clone(), opts.domain_id);
                    store.set_user_meta_0(t.root_page_id());
                    store.set_user_meta_1(0);
                    store.sync()?;
                    TreeHandle::BPlusTree(t)
                } else {
                    TreeHandle::BPlusTree(BTree::open(pool.clone(), root, height, opts.domain_id))
                }
            }
            TreeBackend::BeTree | TreeBackend::BeTreeNoWal => {
                if root == 0 {
                    let t = CowBeTree::new(pool.clone());
                    store.set_user_meta_0(t.root_page_id());
                    store.set_user_meta_1(0);
                    store.sync()?;
                    if no_wal {
                        TreeHandle::BeTreeNoWal(t)
                    } else {
                        TreeHandle::BeTree(t)
                    }
                } else {
                    let t = CowBeTree::open(pool.clone(), root);
                    if no_wal {
                        TreeHandle::BeTreeNoWal(t)
                    } else {
                        TreeHandle::BeTree(t)
                    }
                }
            }
        };

        Ok(Self {
            pool,
            tree,
            wal,
            store,
            sync_mode: opts.sync_mode,
        })
    }

    pub fn put(&self, key: &[u8], value: &[u8]) -> bool {
        let inserted = match &self.tree {
            TreeHandle::BPlusTree(t) => t.upsert(key, value),
            TreeHandle::BeTree(t) | TreeHandle::BeTreeNoWal(t) => {
                t.put(key, u64::MAX, value).expect("betree put failed");
                true
            }
        };
        if self.sync_mode == SyncMode::Strict {
            if let Some(ref wal) = self.wal {
                wal.flush();
            } else {
                let _ = self.tree.flush_for_sync(&self.pool, &self.store);
            }
        }
        inserted
    }

    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        match &self.tree {
            TreeHandle::BPlusTree(t) => t.lookup(key),
            TreeHandle::BeTree(t) | TreeHandle::BeTreeNoWal(t) => t.lookup(key),
        }
    }

    pub fn del(&self, key: &[u8]) -> bool {
        match &self.tree {
            TreeHandle::BPlusTree(t) => t.remove(key),
            TreeHandle::BeTree(t) | TreeHandle::BeTreeNoWal(t) => t.remove(key).unwrap_or(false),
        }
    }

    pub fn scan_all<F: FnMut(&[u8], &[u8])>(&self, f: F) {
        match &self.tree {
            TreeHandle::BPlusTree(t) => t.scan(f),
            TreeHandle::BeTree(t) | TreeHandle::BeTreeNoWal(t) => t.scan_prefix(&[], f),
        }
    }

    pub fn scan_range<F: FnMut(&[u8], &[u8])>(&self, start: &[u8], end: &[u8], mut f: F) {
        match &self.tree {
            TreeHandle::BPlusTree(t) => {
                t.scan_range(Bound::Included(start), Bound::Excluded(end), &mut f);
            }
            TreeHandle::BeTree(t) | TreeHandle::BeTreeNoWal(t) => {
                t.scan_prefix(start, |key, value| {
                    if key < end {
                        f(key, value);
                    }
                });
            }
        }
    }

    pub fn flush_wal(&self) {
        if let Some(ref wal) = self.wal {
            wal.flush();
        }
    }

    pub fn sync(&self) -> std::io::Result<()> {
        if let Some(ref wal) = self.wal {
            wal.flush();
            self.pool.flush()?;
        } else {
            self.tree.flush_for_sync(&self.pool, &self.store)?;
        }
        Ok(())
    }

    pub fn checkpoint(&self) -> std::io::Result<()> {
        if let Some(ref wal) = self.wal {
            let checkpoint_lsn = wal.flush();
            self.pool.flush()?;
            self.store.set_user_meta_0(self.tree.root_page_id());
            self.store.set_user_meta_1(self.tree.height() as u64);
            self.store.set_checkpoint_lsn(checkpoint_lsn);
            self.store.sync()?;
            wal.reset()?;
        } else {
            self.tree.flush_for_sync(&self.pool, &self.store)?;
            self.store.set_user_meta_0(self.tree.root_page_id());
            self.store.set_user_meta_1(self.tree.height() as u64);
            self.store.sync()?;
        }
        Ok(())
    }

    pub fn root_page_id(&self) -> u64 {
        self.tree.root_page_id()
    }

    pub fn height(&self) -> u32 {
        self.tree.height()
    }
}
