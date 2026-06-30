//! Adapter wrapping `redb::Database` (COW B+tree comparison engine).
//!
//! redb is transactional, so each operation opens a write/read transaction.
//! `SyncMode::Strict` maps to `Durability::Immediate`; relaxed maps to
//! `Durability::None`.
//!
//! ## Memory budget equity
//!
//! redb's default cache is 1 GiB. This adapter sets the cache to match
//! kvstore's buffer-pool byte budget (`buffer_budget_frames * PAGE_SIZE`)
//! unless overridden via `engine_specific["cache_size_bytes"]`.

use std::path::Path;

use redb::{Database, Durability, ReadableDatabase, ReadableTable, TableDefinition};

use crate::engine::{EngineOpts, EngineStats, KvEngine, SyncMode};

/// Page size used by the pagebox substrate (4 KiB).
const PAGE_SIZE: usize = 4096;

/// redb table definition: `&[u8]` keys, `&[u8]` values.
const TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("kv");

/// Adapter wrapping `redb::Database`.
pub struct RedbAdapter {
    db: Database,
    sync_mode: SyncMode,
}

impl KvEngine for RedbAdapter {
    const NAME: &'static str = "redb";

    fn open(dir: &Path, opts: &EngineOpts) -> std::io::Result<Self> {
        let path = dir.join("redb.data");

        // Default: match kvstore's buffer-pool byte budget.
        let default_cache_bytes = opts.buffer_budget_frames * PAGE_SIZE;
        let cache_bytes = opts
            .engine_specific
            .get("cache_size_bytes")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(default_cache_bytes);

        let db = Database::builder()
            .set_cache_size(cache_bytes)
            .create(&path)
            .map_err(map_redb_err)?;

        // Create the table upfront.
        let mut txn = db.begin_write().map_err(map_redb_err)?;
        {
            let _table = txn.open_table(TABLE).map_err(map_redb_err)?;
        }
        let _ = txn.set_durability(Durability::Immediate);
        txn.commit().map_err(map_redb_err)?;

        Ok(Self {
            db,
            sync_mode: opts.sync_mode,
        })
    }

    fn put(&self, key: &[u8], value: &[u8]) -> bool {
        let mut txn = self.db.begin_write().map_err(map_redb_err).unwrap();
        let was_present;
        {
            let mut table = txn.open_table(TABLE).map_err(map_redb_err).unwrap();
            was_present = table.get(key).map_err(map_redb_err).unwrap().is_some();
            table.insert(key, value).map_err(map_redb_err).unwrap();
        }
        let _ = txn.set_durability(self.durability());
        txn.commit().map_err(map_redb_err).unwrap();
        !was_present
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let txn = self.db.begin_read().map_err(map_redb_err).ok()?;
        let table = txn.open_table(TABLE).map_err(map_redb_err).ok()?;
        table
            .get(key)
            .ok()
            .flatten()
            .map(|guard| guard.value().to_vec())
    }

    fn del(&self, key: &[u8]) -> bool {
        let mut txn = self.db.begin_write().map_err(map_redb_err).unwrap();
        let was_present;
        {
            let mut table = txn.open_table(TABLE).map_err(map_redb_err).unwrap();
            was_present = table.get(key).map_err(map_redb_err).unwrap().is_some();
            table.remove(key).map_err(map_redb_err).unwrap();
        }
        let _ = txn.set_durability(self.durability());
        txn.commit().map_err(map_redb_err).unwrap();
        was_present
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
        // redb commits are durable (or not) based on Durability. A no-op
        // write transaction with Immediate durability forces an fsync.
        let mut txn = self.db.begin_write().map_err(map_redb_err)?;
        let _ = txn.set_durability(Durability::Immediate);
        txn.commit().map_err(map_redb_err)?;
        Ok(())
    }

    fn stats(&self) -> EngineStats {
        EngineStats::default()
    }
}

impl RedbAdapter {
    fn durability(&self) -> Durability {
        if self.sync_mode == SyncMode::Strict {
            Durability::Immediate
        } else {
            Durability::None
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
            value_size: 100,
            sync_mode: SyncMode::Relaxed,
            buffer_budget_frames: 1024,
            engine_specific: Default::default(),
        }
    }

    #[test]
    fn adapter_contract() {
        let dir = TempDir::new().unwrap();
        let engine = RedbAdapter::open(dir.path(), &test_opts()).unwrap();

        assert!(engine.put(b"A", b"val_a"));
        assert!(!engine.put(b"A", b"val_a2"));
        assert!(engine.put(b"B", b"val_b"));
        assert_eq!(engine.get(b"A"), Some(b"val_a2".to_vec()));
        assert!(engine.del(b"A"));
        assert_eq!(engine.get(b"A"), None);

        let mut results = Vec::new();
        engine.scan_range(b"A", b"C", &mut |k, v| {
            results.push((k.to_vec(), v.to_vec()));
        });
        assert_eq!(results, vec![(b"B".to_vec(), b"val_b".to_vec())]);
    }
}
