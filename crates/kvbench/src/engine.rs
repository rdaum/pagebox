//! Engine adapter trait and shared types.
//!
//! Every engine (kvstore, fjall, redb, sled, rocksdb) implements [`KvEngine`].
//! The driver calls through the trait, so all engines are measured
//! identically.

use std::collections::HashMap;
use std::path::Path;

/// Write durability mode. Mirrors `kvstore::SyncMode` but lives in the
/// harness so adapters for external engines map it to their own semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SyncMode {
    /// Writes return after buffering; WAL/fsync is asynchronous.
    #[default]
    Relaxed,
    /// Every write blocks until durable (fsync).
    Strict,
}

/// Tuning knobs passed to every engine's [`KvEngine::open`].
///
/// Engines ignore fields that don't apply to them (e.g. `buffer_budget_frames`
/// is kvstore-specific; LSM-tree engines use their own block-cache settings).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EngineOpts {
    /// Value size in bytes for generated values.
    pub value_size: usize,
    /// Write durability mode.
    pub sync_mode: SyncMode,
    /// Buffer-pool / cache frame budget (kvstore maps this to pool_frames).
    #[serde(default = "default_buffer_budget")]
    pub buffer_budget_frames: usize,
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
            value_size: 100,
            sync_mode: SyncMode::default(),
            buffer_budget_frames: default_buffer_budget(),
            wal_backend: None,
            engine_specific: HashMap::new(),
        }
    }
}

fn default_buffer_budget() -> usize {
    1024
}

/// Engine-reported side-channel stats (adapter-defined, opaque to the driver).
#[allow(dead_code)]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EngineStats {
    pub extra: HashMap<String, String>,
}

/// The adapter contract every engine implements.
pub trait KvEngine: Send + Sync {
    /// Engine name (e.g. `"kvstore"`, `"fjall"`).
    const NAME: &'static str;

    /// Open a fresh instance rooted at `dir`. Each run uses a fresh dir.
    fn open(dir: &Path, opts: &EngineOpts) -> std::io::Result<Self>
    where
        Self: Sized;

    /// Insert or update. Returns `true` if the key was absent before.
    fn put(&self, key: &[u8], value: &[u8]) -> bool;
    /// Point lookup. Returns `None` if the key is absent.
    fn get(&self, key: &[u8]) -> Option<Vec<u8>>;
    /// Remove a key. Returns `true` if the key was present.
    fn del(&self, key: &[u8]) -> bool;
    /// Ordered scan over `[start, end)`. Calls `f` per `(key, value)`.
    fn scan_range(&self, start: &[u8], end: &[u8], f: &mut dyn FnMut(&[u8], &[u8]));

    /// Durable flush (engine-defined). Called at end of load phase and per
    /// `--sync-interval` if configured.
    fn sync(&self) -> std::io::Result<()>;

    /// Engine-reported stats for side-channel output. Default: empty.
    #[allow(dead_code)]
    fn stats(&self) -> EngineStats {
        EngineStats::default()
    }
}
