//! Adapter wrapping `lmdb-rkv` — the Lightning Memory-Mapped Database.
//!
//! LMDB is a memory-mapped B+tree. It uses the OS page cache directly
//! (no separate buffer pool), which makes it an interesting comparison
//! against pagebox's swizzled-pointer buffer pool.
//!
//! LMDB's `map_size` caps the database's virtual address range, not its
//! resident memory. It is deliberately independent of kvbench's cache budget.
//!
//! Each measured mutation is one LMDB transaction. This matches kvbench's
//! point-operation completion contract; the adapter does not hide work in an
//! unmeasured pending-write buffer. Relaxed mode opens the environment with
//! `NO_SYNC`, while strict mode retains LMDB's default durable commit.

use std::path::Path;

use lmdb::{Cursor, Database, Environment, EnvironmentFlags, Transaction, WriteFlags};

use crate::engine::{CacheControl, EngineOpts, EngineStats, KvEngine, SyncMode};

pub struct LmdbAdapter {
    env: Environment,
    db: Database,
}

impl KvEngine for LmdbAdapter {
    const NAME: &'static str = "lmdb";
    const CACHE_CONTROL: CacheControl = CacheControl::OsManaged;

    fn open(dir: &Path, opts: &EngineOpts) -> std::io::Result<Self> {
        let map_size = opts
            .engine_specific
            .get("map_size_bytes")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(8 * 1024 * 1024 * 1024);

        std::fs::create_dir_all(dir)?;
        let mut builder = Environment::new();
        builder.set_map_size(map_size).set_max_dbs(1);
        if opts.sync_mode == SyncMode::Relaxed {
            builder.set_flags(EnvironmentFlags::NO_SYNC);
        }
        let env = builder
            .open(dir)
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        let db = env
            .create_db(Some("default"), lmdb::DatabaseFlags::default())
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        Ok(Self { env, db })
    }

    fn put(&self, key: &[u8], value: &[u8]) {
        let mut txn = self
            .env
            .begin_rw_txn()
            .expect("lmdb begin_rw_txn failed (env closed?)");
        txn.put(self.db, &key, &value, WriteFlags::default())
            .expect("lmdb txn.put failed (map full?)");
        txn.commit().expect("lmdb txn.commit failed (map full?)");
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.read_committed(key)
    }

    fn del(&self, key: &[u8]) {
        let mut txn = self
            .env
            .begin_rw_txn()
            .map_err(|e| std::io::Error::other(e.to_string()))
            .unwrap();
        match txn.del(self.db, &key, None) {
            Ok(()) | Err(lmdb::Error::NotFound) => {}
            Err(error) => panic!("lmdb txn.del failed: {error}"),
        }
        txn.commit()
            .map_err(|e| std::io::Error::other(e.to_string()))
            .unwrap();
    }

    fn scan_range(&self, start: &[u8], end: &[u8], f: &mut dyn FnMut(&[u8], &[u8])) {
        let txn = self
            .env
            .begin_ro_txn()
            .map_err(|e| std::io::Error::other(e.to_string()))
            .unwrap();
        {
            let mut cursor = txn
                .open_ro_cursor(self.db)
                .map_err(|e| std::io::Error::other(e.to_string()))
                .unwrap();
            for result in cursor.iter_from(start) {
                let (k, v) = match result {
                    Ok((k, v)) => (k, v),
                    Err(_) => break,
                };
                if k >= end {
                    break;
                }
                f(k, v);
            }
        }
        txn.abort();
    }

    fn sync(&self) -> std::io::Result<()> {
        self.env
            .sync(true)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    fn stats(&self) -> EngineStats {
        EngineStats::default()
    }
}

impl LmdbAdapter {
    /// Point read against the last-committed snapshot (no pending overlay).
    fn read_committed(&self, key: &[u8]) -> Option<Vec<u8>> {
        let txn = self
            .env
            .begin_ro_txn()
            .map_err(|e| std::io::Error::other(e.to_string()))
            .unwrap();
        let result = txn.get(self.db, &key).ok().map(|v: &[u8]| v.to_vec());
        txn.abort();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn adapter_contract_has_no_pending_write_layer() {
        let dir = TempDir::new().unwrap();
        let engine = LmdbAdapter::open(dir.path(), &EngineOpts::default()).unwrap();

        engine.put(b"A", b"val_a");
        engine.put(b"A", b"val_a2");
        engine.put(b"B", b"val_b");
        assert_eq!(engine.get(b"A"), Some(b"val_a2".to_vec()));
        engine.del(b"A");
        assert_eq!(engine.get(b"A"), None);

        let mut results = Vec::new();
        engine.scan_range(b"A", b"C", &mut |key, value| {
            results.push((key.to_vec(), value.to_vec()));
        });
        assert_eq!(results, vec![(b"B".to_vec(), b"val_b".to_vec())]);
    }
}
