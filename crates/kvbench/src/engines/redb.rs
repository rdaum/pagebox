//! Adapter wrapping `redb::Database` (COW B+tree comparison engine).
//!
//! redb is transactional, so each operation opens a write/read transaction.
//! `SyncMode::Strict` maps to `Durability::Immediate`; relaxed maps to
//! `Durability::None`.
//!
//! Each measured mutation is one transaction. The adapter does not acknowledge
//! an operation into an unmeasured batch.
//!
//! ## Memory budget equity
//!
//! redb's default cache is 1 GiB. This adapter sets the cache to match
//! kvstore's configured application-cache byte budget
//! unless overridden via `engine_specific["cache_size_bytes"]`.

use std::path::Path;

use redb::{Database, Durability, ReadableDatabase, TableDefinition};

use crate::engine::{CacheControl, EngineOpts, EngineStats, KvEngine, SyncMode};

/// redb table definition: `&[u8]` keys, `&[u8]` values.
const TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("kv");

/// Adapter wrapping `redb::Database`.
pub struct RedbAdapter {
    db: Database,
    sync_mode: SyncMode,
    path: std::path::PathBuf,
    cache_budget_bytes: usize,
}

impl RedbAdapter {
    fn read_committed(&self, key: &[u8]) -> Option<Vec<u8>> {
        let txn = self.db.begin_read().ok()?;
        let table = txn.open_table(TABLE).ok()?;
        table
            .get(key)
            .ok()
            .flatten()
            .map(|guard| guard.value().to_vec())
    }

    fn durability(&self) -> Durability {
        if self.sync_mode == SyncMode::Strict {
            Durability::Immediate
        } else {
            Durability::None
        }
    }
}

impl KvEngine for RedbAdapter {
    const NAME: &'static str = "redb";
    const CACHE_CONTROL: CacheControl = CacheControl::Application;

    fn open(dir: &Path, opts: &EngineOpts) -> std::io::Result<Self> {
        let path = dir.join("redb.data");

        let cache_bytes = opts
            .engine_specific
            .get("cache_size_bytes")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(opts.cache_budget_bytes);

        let db = Database::builder()
            .set_cache_size(cache_bytes)
            .create(&path)
            .map_err(map_redb_err)?;

        let mut txn = db.begin_write().map_err(map_redb_err)?;
        {
            let _table = txn.open_table(TABLE).map_err(map_redb_err)?;
        }
        let _ = txn.set_durability(Durability::Immediate);
        txn.commit().map_err(map_redb_err)?;

        Ok(Self {
            db,
            sync_mode: opts.sync_mode,
            path,
            cache_budget_bytes: cache_bytes,
        })
    }

    fn put(&self, key: &[u8], value: &[u8]) {
        let mut txn = self.db.begin_write().expect("redb begin_write failed");
        {
            let mut table = txn.open_table(TABLE).expect("redb open_table failed");
            table.insert(key, value).expect("redb insert failed");
        }
        let _ = txn.set_durability(self.durability());
        txn.commit().expect("redb commit failed");
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.read_committed(key)
    }

    fn del(&self, key: &[u8]) {
        let mut txn = self.db.begin_write().map_err(map_redb_err).unwrap();
        {
            let mut table = txn.open_table(TABLE).map_err(map_redb_err).unwrap();
            let _ = table.remove(key);
        }
        let _ = txn.set_durability(self.durability());
        txn.commit().map_err(map_redb_err).unwrap();
    }

    fn scan_range(&self, start: &[u8], end: &[u8], f: &mut dyn FnMut(&[u8], &[u8])) {
        let Ok(txn) = self.db.begin_read() else {
            return;
        };
        let Ok(table) = txn.open_table(TABLE) else {
            return;
        };
        let Ok(iter) = table.range(start..end) else {
            return;
        };
        for item in iter {
            match item {
                Ok((k, v)) => f(k.value(), v.value()),
                Err(_) => break,
            }
        }
    }

    fn sync(&self) -> std::io::Result<()> {
        let mut txn = self.db.begin_write().map_err(map_redb_err)?;
        let _ = txn.set_durability(Durability::Immediate);
        txn.commit().map_err(map_redb_err)?;
        Ok(())
    }

    fn stats(&self) -> EngineStats {
        let stats = self.db.cache_stats();
        EngineStats {
            direct_io: Some(false),
            cache_capacity_bytes: Some(self.cache_budget_bytes as u64),
            cache_used_bytes: Some(stats.used_bytes() as u64),
            cache_hits: Some(stats.read_hits() + stats.write_hits()),
            cache_misses: Some(stats.read_misses() + stats.write_misses()),
            cache_evictions: Some(stats.evictions()),
            persisted_data_bytes: std::fs::metadata(&self.path).ok().map(|meta| meta.len()),
            ..EngineStats::default()
        }
    }
}

fn map_redb_err(e: impl std::fmt::Display) -> std::io::Error {
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
        let engine = RedbAdapter::open(dir.path(), &test_opts()).unwrap();

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
