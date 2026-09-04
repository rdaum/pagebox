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

/// Engine data-file I/O observed during the measured phase.
///
/// Requested bytes and completed bytes are separate because short reads and
/// failed operations need not complete the full request. Fields remain
/// `None` when an engine exposes no equivalent counter.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StorageIoStats {
    pub read_calls: Option<u64>,
    pub read_requested_bytes: Option<u64>,
    pub read_completed_bytes: Option<u64>,
    pub write_calls: Option<u64>,
    pub write_requested_bytes: Option<u64>,
    pub write_completed_bytes: Option<u64>,
    pub direct_read_completed_bytes: Option<u64>,
    pub direct_write_completed_bytes: Option<u64>,
    pub buffered_read_completed_bytes: Option<u64>,
    pub buffered_write_completed_bytes: Option<u64>,
    pub foreground_read_completed_bytes: Option<u64>,
    pub prefetch_read_completed_bytes: Option<u64>,
    pub recovery_read_completed_bytes: Option<u64>,
    pub foreground_write_completed_bytes: Option<u64>,
    pub background_write_completed_bytes: Option<u64>,
    pub checkpoint_write_completed_bytes: Option<u64>,
    pub recovery_write_completed_bytes: Option<u64>,
    pub metadata_write_completed_bytes: Option<u64>,
    pub batched_read_calls: Option<u64>,
    pub batched_read_pages: Option<u64>,
    pub batched_write_calls: Option<u64>,
    pub batched_write_pages: Option<u64>,
    pub sync_calls: Option<u64>,
    pub foreground_sync_calls: Option<u64>,
    pub checkpoint_sync_calls: Option<u64>,
    pub recovery_sync_calls: Option<u64>,
}

impl StorageIoStats {
    fn phase_delta_since(self, earlier: &Self) -> Self {
        Self {
            read_calls: option_delta(self.read_calls, earlier.read_calls),
            read_requested_bytes: option_delta(
                self.read_requested_bytes,
                earlier.read_requested_bytes,
            ),
            read_completed_bytes: option_delta(
                self.read_completed_bytes,
                earlier.read_completed_bytes,
            ),
            write_calls: option_delta(self.write_calls, earlier.write_calls),
            write_requested_bytes: option_delta(
                self.write_requested_bytes,
                earlier.write_requested_bytes,
            ),
            write_completed_bytes: option_delta(
                self.write_completed_bytes,
                earlier.write_completed_bytes,
            ),
            direct_read_completed_bytes: option_delta(
                self.direct_read_completed_bytes,
                earlier.direct_read_completed_bytes,
            ),
            direct_write_completed_bytes: option_delta(
                self.direct_write_completed_bytes,
                earlier.direct_write_completed_bytes,
            ),
            buffered_read_completed_bytes: option_delta(
                self.buffered_read_completed_bytes,
                earlier.buffered_read_completed_bytes,
            ),
            buffered_write_completed_bytes: option_delta(
                self.buffered_write_completed_bytes,
                earlier.buffered_write_completed_bytes,
            ),
            foreground_read_completed_bytes: option_delta(
                self.foreground_read_completed_bytes,
                earlier.foreground_read_completed_bytes,
            ),
            prefetch_read_completed_bytes: option_delta(
                self.prefetch_read_completed_bytes,
                earlier.prefetch_read_completed_bytes,
            ),
            recovery_read_completed_bytes: option_delta(
                self.recovery_read_completed_bytes,
                earlier.recovery_read_completed_bytes,
            ),
            foreground_write_completed_bytes: option_delta(
                self.foreground_write_completed_bytes,
                earlier.foreground_write_completed_bytes,
            ),
            background_write_completed_bytes: option_delta(
                self.background_write_completed_bytes,
                earlier.background_write_completed_bytes,
            ),
            checkpoint_write_completed_bytes: option_delta(
                self.checkpoint_write_completed_bytes,
                earlier.checkpoint_write_completed_bytes,
            ),
            recovery_write_completed_bytes: option_delta(
                self.recovery_write_completed_bytes,
                earlier.recovery_write_completed_bytes,
            ),
            metadata_write_completed_bytes: option_delta(
                self.metadata_write_completed_bytes,
                earlier.metadata_write_completed_bytes,
            ),
            batched_read_calls: option_delta(self.batched_read_calls, earlier.batched_read_calls),
            batched_read_pages: option_delta(self.batched_read_pages, earlier.batched_read_pages),
            batched_write_calls: option_delta(
                self.batched_write_calls,
                earlier.batched_write_calls,
            ),
            batched_write_pages: option_delta(
                self.batched_write_pages,
                earlier.batched_write_pages,
            ),
            sync_calls: option_delta(self.sync_calls, earlier.sync_calls),
            foreground_sync_calls: option_delta(
                self.foreground_sync_calls,
                earlier.foreground_sync_calls,
            ),
            checkpoint_sync_calls: option_delta(
                self.checkpoint_sync_calls,
                earlier.checkpoint_sync_calls,
            ),
            recovery_sync_calls: option_delta(
                self.recovery_sync_calls,
                earlier.recovery_sync_calls,
            ),
        }
    }
}

/// WAL buffer evidence for one shard.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct WalShardMemoryStats {
    pub configured_buffer_capacity_bytes: u64,
    pub active_buffer_count: u64,
    pub spare_buffer_count: u64,
    pub pending_buffer_count: u64,
    pub in_flight_buffer_count: u64,
    pub allocated_buffer_count: u64,
    pub virtual_buffer_capacity_bytes: u64,
    pub known_touched_buffer_bytes: u64,
    pub active_used_bytes: u64,
    pub active_used_high_water_bytes: u64,
    pub max_submitted_buffer_records: u64,
    pub max_submitted_batch_records: u64,
    pub page_image_bytes_appended: Option<u64>,
    pub logical_bytes_appended: Option<u64>,
}

/// WAL buffer evidence. Virtual reservation and known-touched bytes are kept
/// separate and neither is labelled as resident memory.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct WalMemoryStats {
    pub shard_count: u64,
    pub configured_buffer_capacity_bytes: u64,
    pub active_buffer_count: u64,
    pub spare_buffer_count: u64,
    pub pending_buffer_count: u64,
    pub in_flight_buffer_count: u64,
    pub allocated_buffer_count: u64,
    pub virtual_buffer_capacity_bytes: u64,
    pub known_touched_buffer_bytes: u64,
    pub active_used_bytes: u64,
    pub active_used_high_water_bytes: u64,
    pub max_submitted_buffer_records: u64,
    pub max_submitted_batch_records: u64,
    pub page_image_bytes_appended: Option<u64>,
    pub logical_bytes_appended: Option<u64>,
    pub shards: Vec<WalShardMemoryStats>,
}

impl WalMemoryStats {
    fn phase_delta_since(mut self, earlier: &Self) -> Self {
        self.page_image_bytes_appended = option_delta(
            self.page_image_bytes_appended,
            earlier.page_image_bytes_appended,
        );
        self.logical_bytes_appended =
            option_delta(self.logical_bytes_appended, earlier.logical_bytes_appended);
        for (shard, earlier_shard) in self.shards.iter_mut().zip(&earlier.shards) {
            shard.page_image_bytes_appended = option_delta(
                shard.page_image_bytes_appended,
                earlier_shard.page_image_bytes_appended,
            );
            shard.logical_bytes_appended = option_delta(
                shard.logical_bytes_appended,
                earlier_shard.logical_bytes_appended,
            );
        }
        self
    }
}

/// Engine-reported evidence about the cache and persisted working set.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EngineStats {
    pub direct_io: Option<bool>,
    pub cache_capacity_bytes: Option<u64>,
    pub cache_used_bytes: Option<u64>,
    pub cache_hits: Option<u64>,
    pub cache_misses: Option<u64>,
    /// Engine-native event unit for cache hits/misses. Values with different
    /// units are descriptive only and must not be ranked across engines.
    pub cache_access_unit: Option<String>,
    pub cache_evictions: Option<u64>,
    /// Cumulative bytes inserted into or loaded through the bounded cache.
    pub cache_insert_bytes: Option<u64>,
    pub storage_io: Option<StorageIoStats>,
    pub wal_memory: Option<WalMemoryStats>,
    /// Bytes reachable from the engine's live logical structures, when the
    /// engine can distinguish them from allocated file high-water space.
    pub live_data_bytes: Option<u64>,
    pub persisted_data_bytes: Option<u64>,
    pub extra: HashMap<String, String>,
}

impl EngineStats {
    /// Convert a cumulative snapshot into measured-phase counters while
    /// retaining after-phase gauges such as capacity and current use.
    pub fn phase_delta_since(mut self, earlier: &Self) -> Self {
        self.cache_hits = option_delta(self.cache_hits, earlier.cache_hits);
        self.cache_misses = option_delta(self.cache_misses, earlier.cache_misses);
        self.cache_evictions = option_delta(self.cache_evictions, earlier.cache_evictions);
        self.cache_insert_bytes = option_delta(self.cache_insert_bytes, earlier.cache_insert_bytes);
        self.storage_io = match (self.storage_io, &earlier.storage_io) {
            (Some(current), Some(earlier)) => Some(current.phase_delta_since(earlier)),
            (current, _) => current,
        };
        self.wal_memory = match (self.wal_memory, &earlier.wal_memory) {
            (Some(current), Some(earlier)) => Some(current.phase_delta_since(earlier)),
            (current, _) => current,
        };
        self
    }
}

fn option_delta(current: Option<u64>, earlier: Option<u64>) -> Option<u64> {
    match (current, earlier) {
        (Some(current), Some(earlier)) => Some(current.saturating_sub(earlier)),
        (current, _) => current,
    }
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

    /// Reset engine-owned phase-local high-water marks before measurement.
    fn begin_measurement(&self) {}

    /// Capture stats and convert cumulative counters to measured-phase
    /// deltas. Adapters can override this for engine-specific side channels.
    fn phase_stats_since(&self, earlier: &EngineStats) -> EngineStats {
        self.stats().phase_delta_since(earlier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_delta_separates_counters_from_capacity_and_use_gauges() {
        let earlier = EngineStats {
            cache_capacity_bytes: Some(64),
            cache_used_bytes: Some(40),
            cache_misses: Some(10),
            storage_io: Some(StorageIoStats {
                read_calls: Some(10),
                read_requested_bytes: Some(40_960),
                read_completed_bytes: Some(36_864),
                ..StorageIoStats::default()
            }),
            wal_memory: Some(WalMemoryStats {
                virtual_buffer_capacity_bytes: 1_024,
                known_touched_buffer_bytes: 128,
                page_image_bytes_appended: Some(100),
                shards: vec![WalShardMemoryStats {
                    page_image_bytes_appended: Some(100),
                    ..WalShardMemoryStats::default()
                }],
                ..WalMemoryStats::default()
            }),
            ..EngineStats::default()
        };
        let current = EngineStats {
            cache_capacity_bytes: Some(64),
            cache_used_bytes: Some(52),
            cache_misses: Some(14),
            storage_io: Some(StorageIoStats {
                read_calls: Some(16),
                read_requested_bytes: Some(65_536),
                read_completed_bytes: Some(61_440),
                ..StorageIoStats::default()
            }),
            wal_memory: Some(WalMemoryStats {
                virtual_buffer_capacity_bytes: 2_048,
                known_touched_buffer_bytes: 512,
                active_used_high_water_bytes: 256,
                page_image_bytes_appended: Some(260),
                shards: vec![WalShardMemoryStats {
                    active_used_high_water_bytes: 256,
                    page_image_bytes_appended: Some(260),
                    ..WalShardMemoryStats::default()
                }],
                ..WalMemoryStats::default()
            }),
            ..EngineStats::default()
        };

        let phase = current.phase_delta_since(&earlier);
        assert_eq!(phase.cache_capacity_bytes, Some(64));
        assert_eq!(phase.cache_used_bytes, Some(52));
        assert_eq!(phase.cache_misses, Some(4));
        let io = phase.storage_io.unwrap();
        assert_eq!(io.read_calls, Some(6));
        assert_eq!(io.read_requested_bytes, Some(24_576));
        assert_eq!(io.read_completed_bytes, Some(24_576));
        let wal = phase.wal_memory.unwrap();
        assert_eq!(wal.virtual_buffer_capacity_bytes, 2_048);
        assert_eq!(wal.known_touched_buffer_bytes, 512);
        assert_eq!(wal.active_used_high_water_bytes, 256);
        assert_eq!(wal.page_image_bytes_appended, Some(160));
        assert_eq!(wal.shards[0].page_image_bytes_appended, Some(160));
    }
}
