//! Engine adapter trait and shared types.
//!
//! Every engine (kvstore, fjall, redb, rocksdb, LMDB) implements [`KvEngine`].
//! The driver calls through the trait, so all engines are measured
//! identically.

use std::collections::HashMap;
use std::path::Path;

/// Write durability mode. Mirrors `kvstore::SyncMode` but lives in the
/// harness so adapters for external engines map it to their own semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SyncMode {
    /// Writes are visible when the call returns; crash durability may lag.
    #[default]
    Relaxed,
    /// Every write blocks until durable (fsync).
    Strict,
}

/// Tuning knobs passed to every engine's [`KvEngine::open`].
///
/// Engines ignore fields that don't apply to them. `cache_budget_bytes` always
/// denotes application-managed cache memory, never total RSS or OS page cache.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EngineOpts {
    /// Write durability mode.
    pub sync_mode: SyncMode,
    /// Application-managed buffer-pool / cache budget in bytes.
    #[serde(default = "default_cache_budget_bytes")]
    pub cache_budget_bytes: usize,
    /// Bypass the operating-system page cache for engine data-file reads.
    /// Only engines declaring [`KvEngine::SUPPORTS_DIRECT_IO`] accept this.
    #[serde(default)]
    pub direct_io: bool,
    /// WAL sync backend for kvstore ("fdatasync", "pwritev2_dsync", "io_uring").
    /// Other engines ignore this. Sets the `PAGEBOX_WAL_SYNC_BACKEND` env var
    /// before opening.
    #[serde(default)]
    pub wal_backend: Option<String>,
    /// Engine-specific key=value overrides (opaque to the driver).
    #[serde(default)]
    pub engine_specific: HashMap<String, String>,
}

impl Default for EngineOpts {
    fn default() -> Self {
        Self {
            sync_mode: SyncMode::default(),
            cache_budget_bytes: default_cache_budget_bytes(),
            direct_io: false,
            wal_backend: None,
            engine_specific: HashMap::new(),
        }
    }
}

fn default_cache_budget_bytes() -> usize {
    64 * 1024 * 1024
}

/// Whether an engine exposes an application-managed cache budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheControl {
    /// `cache_budget_bytes` controls an engine-owned cache or buffer pool.
    Application,
    /// Residency is managed by the operating system and cannot be bounded by
    /// kvbench independently of the rest of the host.
    OsManaged,
}

/// Engine-reported evidence about the cache and persisted working set.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EngineStats {
    pub direct_io: Option<bool>,
    pub cache_capacity_bytes: Option<u64>,
    pub cache_used_bytes: Option<u64>,
    pub cache_hits: Option<u64>,
    pub cache_misses: Option<u64>,
    pub cache_evictions: Option<u64>,
    /// Cumulative bytes inserted into or loaded through the bounded cache.
    pub cache_insert_bytes: Option<u64>,
    pub storage_read_bytes: Option<u64>,
    /// Bytes reachable from the engine's live logical structures, when the
    /// engine can distinguish them from allocated file high-water space.
    pub live_data_bytes: Option<u64>,
    pub persisted_data_bytes: Option<u64>,
    pub extra: HashMap<String, String>,
}

/// The adapter contract every engine implements.
pub trait KvEngine: Send + Sync {
    /// Engine name (e.g. `"kvstore"`, `"fjall"`).
    const NAME: &'static str;
    /// Kind of cache control exposed by the adapter.
    const CACHE_CONTROL: CacheControl;
    /// Whether the adapter can bypass the OS page cache for data-file reads.
    const SUPPORTS_DIRECT_IO: bool = false;

    /// Open a fresh instance rooted at `dir`. Each run uses a fresh dir.
    fn open(dir: &Path, opts: &EngineOpts) -> std::io::Result<Self>
    where
        Self: Sized;

    /// Insert or update. The mutation is visible when this call returns.
    fn put(&self, key: &[u8], value: &[u8]);
    /// Point lookup. Returns `None` if the key is absent.
    fn get(&self, key: &[u8]) -> Option<Vec<u8>>;
    /// Remove a key. The absence is visible when this call returns.
    fn del(&self, key: &[u8]);
    /// Ordered scan over `[start, end)`. Calls `f` per `(key, value)`.
    fn scan_range(&self, start: &[u8], end: &[u8], f: &mut dyn FnMut(&[u8], &[u8]));

    /// Make every preceding mutation crash-durable before returning.
    fn sync(&self) -> std::io::Result<()>;

    /// Materialize the loaded data set so dropping and reopening the engine
    /// cannot leave measured records in a MemTable or WAL-only representation.
    fn prepare_for_reopen(&self) -> std::io::Result<()> {
        self.sync()
    }

    /// Engine-reported stats for side-channel output. Default: empty.
    fn stats(&self) -> EngineStats {
        EngineStats::default()
    }
}
