//! Adapter wrapping `fjall::Database` (LSM-tree comparison engine).
//!
//! ## Memory budget equity
//!
//! Fjall's default block cache is 32 MiB. This adapter divides the configured
//! application-cache byte budget between its block cache and bounded
//! MemTables unless overridden via `engine_specific["cache_size_bytes"]`.
//!
//! Additional overrides:
//! - `max_journaling_size_bytes`: fjall journal size (default 512 MiB).

use std::path::Path;

use fjall::{Database, KeyspaceCreateOptions, PersistMode};

use crate::engine::{CacheControl, EngineOpts, EngineStats, KvEngine, SyncMode};

/// Adapter wrapping `fjall::Database` with a single keyspace.
pub struct FjallAdapter {
    db: Database,
    keyspace: fjall::Keyspace,
    sync_mode: SyncMode,
    cache_budget_bytes: u64,
}

impl KvEngine for FjallAdapter {
    const NAME: &'static str = "fjall";
    const CACHE_CONTROL: CacheControl = CacheControl::Application;

    fn open(dir: &Path, opts: &EngineOpts) -> std::io::Result<Self> {
        let mut builder = Database::builder(dir);

        // Default: match kvstore's buffer-pool byte budget so both engines
        // have the same memory budget for cached pages/blocks.
        let application_bytes = opts
            .engine_specific
            .get("cache_size_bytes")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(opts.cache_budget_bytes as u64);
        if application_bytes < 8 * 1024 * 1024 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "fjall cache budget must be at least 8 MiB",
            ));
        }
        // Fjall has separate block-cache and per-keyspace MemTable controls,
        // but no shared budget manager. Reserve half for blocks and size one
        // MemTable to one eighth: Fjall stalls at four sealed MemTables, so
        // the configured data structures remain within the common budget.
        let block_cache_bytes = application_bytes / 2;
        let memtable_bytes = (application_bytes / 8).max(1024 * 1024);
        builder = builder.cache_size(block_cache_bytes);

        // Optional journal size override.
        if let Some(journal_bytes) = opts
            .engine_specific
            .get("max_journaling_size_bytes")
            .and_then(|s| s.parse::<u64>().ok())
        {
            builder = builder.max_journaling_size(journal_bytes);
        }

        let db = builder.open().map_err(map_fjall_err)?;
        let keyspace = db
            .keyspace("default", || {
                KeyspaceCreateOptions::default().max_memtable_size(memtable_bytes)
            })
            .map_err(map_fjall_err)?;
        Ok(Self {
            db,
            keyspace,
            sync_mode: opts.sync_mode,
            cache_budget_bytes: application_bytes,
        })
    }

    fn put(&self, key: &[u8], value: &[u8]) {
        self.keyspace
            .insert(key, value)
            .map_err(map_fjall_err)
            .unwrap();
        if self.sync_mode == SyncMode::Strict {
            self.db
                .persist(PersistMode::SyncAll)
                .map_err(map_fjall_err)
                .unwrap();
        }
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.keyspace
            .get(key)
            .ok()
            .flatten()
            .map(|slice| slice.to_vec())
    }

    fn del(&self, key: &[u8]) {
        self.keyspace.remove(key).map_err(map_fjall_err).unwrap();
        if self.sync_mode == SyncMode::Strict {
            self.db
                .persist(PersistMode::SyncAll)
                .map_err(map_fjall_err)
                .unwrap();
        }
    }

    fn scan_range(&self, start: &[u8], end: &[u8], f: &mut dyn FnMut(&[u8], &[u8])) {
        use std::ops::Bound;
        let iter = self.keyspace.range::<Vec<u8>, _>((
            Bound::Included(start.to_vec()),
            Bound::Excluded(end.to_vec()),
        ));
        for item in iter {
            if let Ok((k, v)) = item.into_inner() {
                f(k.as_slice(), v.as_slice());
            } else {
                break;
            }
        }
    }

    fn sync(&self) -> std::io::Result<()> {
        self.db.persist(PersistMode::SyncAll).map_err(map_fjall_err)
    }

    fn prepare_for_reopen(&self) -> std::io::Result<()> {
        self.keyspace
            .rotate_memtable_and_wait()
            .map_err(map_fjall_err)?;
        self.sync()
    }

    fn stats(&self) -> EngineStats {
        let metrics = self.keyspace.metrics();
        let hits = metrics.block_load_cached_count() as u64;
        let misses = metrics.block_load_io_count() as u64;
        let mut extra = std::collections::HashMap::new();
        extra.insert(
            "block_cache_capacity_bytes".to_string(),
            self.db.cache_capacity().to_string(),
        );
        extra.insert(
            "write_buffer_bytes".to_string(),
            self.db.write_buffer_size().to_string(),
        );
        EngineStats {
            direct_io: Some(false),
            cache_capacity_bytes: Some(self.cache_budget_bytes),
            cache_used_bytes: Some(self.db.cache_size() + self.db.write_buffer_size()),
            cache_hits: Some(hits),
            cache_misses: Some(misses),
            cache_insert_bytes: Some(metrics.block_io()),
            storage_read_bytes: Some(metrics.block_io()),
            persisted_data_bytes: Some(self.keyspace.disk_space()),
            extra,
            ..EngineStats::default()
        }
    }
}

fn map_fjall_err(e: fjall::Error) -> std::io::Error {
    std::io::Error::other(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_opts() -> EngineOpts {
        EngineOpts {
            sync_mode: SyncMode::Relaxed,
            cache_budget_bytes: 64 * 1024 * 1024,
            direct_io: false,
            wal_backend: None,
            engine_specific: Default::default(),
        }
    }

    #[test]
    fn adapter_contract() {
        let dir = TempDir::new().unwrap();
        let engine = FjallAdapter::open(dir.path(), &test_opts()).unwrap();

        engine.put(b"A", b"val_a");
        engine.put(b"A", b"val_a2");
        engine.put(b"B", b"val_b");
        assert_eq!(engine.get(b"A"), Some(b"val_a2".to_vec()));
        engine.del(b"A");
        assert_eq!(engine.get(b"A"), None);

        let mut results = Vec::new();
        engine.scan_range(b"A", b"C", &mut |k, v| {
            results.push((k.to_vec(), v.to_vec()));
        });
        assert_eq!(results, vec![(b"B".to_vec(), b"val_b".to_vec())]);
    }
}
