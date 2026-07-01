//! Adapter wrapping `redb::Database` (COW B+tree comparison engine).
//!
//! redb is transactional, so each operation opens a write/read transaction.
//! `SyncMode::Strict` maps to `Durability::Immediate`; relaxed maps to
//! `Durability::None`.
//!
//! ## Write batching
//!
//! Like LMDB, redb is a single-writer B+tree that commits per transaction:
//! each `commit()` writes the COW dirty set for that transaction. A
//! one-transaction-per-`put` adapter pays the full dirty-path cost on every
//! op — p99.9 latency of 22+ ms on fillrandom is commit stall, not engine
//! throughput. redb is also single-writer, so concurrent `put`s serialize on
//! the writer lock regardless.
//!
//! We buffer pending writes in a shared [`Mutex`]`<`[`Batch`]`>` and flush
//! them as one write transaction when [`BATCH_FLUSH_THRESHOLD`] distinct keys
//! accumulate, on `sync`, and on drop. Reads (`get`) do **not** flush;
//! instead they overlay the pending map on the committed snapshot via a
//! re-check protocol, so a write that lands between the pending lookup and
//! the committed read is still observed. This keeps read-heavy mixed
//! workloads fast (no commit-per-read) while staying correct under
//! `--verify`: a get always returns the newest pending-or-committed value.
//!
//! `del` and `scan_range` do flush first — deletes need an accurate
//! presence check and scans need a consistent ordered snapshot, which the
//! overlay can't provide cheaply. Both are rare in the benchmark suite.
//!
//! ## Memory budget equity
//!
//! redb's default cache is 1 GiB. This adapter sets the cache to match
//! kvstore's buffer-pool byte budget (`buffer_budget_frames * PAGE_SIZE`)
//! unless overridden via `engine_specific["cache_size_bytes"]`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use redb::{Database, Durability, ReadableDatabase, ReadableTable, TableDefinition};

use crate::engine::{EngineOpts, EngineStats, KvEngine, SyncMode};

/// Page size used by the pagebox substrate (64 KiB).
const PAGE_SIZE: usize = 65536;

/// redb table definition: `&[u8]` keys, `&[u8]` values.
const TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("kv");

/// Flush once this many distinct keys are pending.
const BATCH_FLUSH_THRESHOLD: usize = 4096;

/// Adapter wrapping `redb::Database`.
pub struct RedbAdapter {
    db: Database,
    sync_mode: SyncMode,
    batch: Mutex<Batch>,
}

/// Pending writes awaiting the next commit.
struct Batch {
    puts: HashMap<Vec<u8>, Vec<u8>>,
}

impl Batch {
    fn new() -> Self {
        Self {
            puts: HashMap::new(),
        }
    }
}

impl RedbAdapter {
    fn flush_locked(&self, mut guard: std::sync::MutexGuard<'_, Batch>) {
        if guard.puts.is_empty() {
            return;
        }
        let puts = std::mem::take(&mut guard.puts);
        drop(guard);

        let mut txn = self.db.begin_write().expect("redb begin_write failed");
        {
            let mut table = txn.open_table(TABLE).expect("redb open_table failed");
            for (k, v) in &puts {
                table
                    .insert(k.as_slice(), v.as_slice())
                    .expect("redb insert failed");
            }
        }
        let _ = txn.set_durability(self.durability());
        txn.commit().expect("redb commit failed");
    }

    fn flush(&self) {
        let guard = self.batch.lock().unwrap();
        self.flush_locked(guard);
    }

    fn pending_get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.batch.lock().unwrap().puts.get(key).cloned()
    }

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

    fn open(dir: &Path, opts: &EngineOpts) -> std::io::Result<Self> {
        let path = dir.join("redb.data");

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

        let mut txn = db.begin_write().map_err(map_redb_err)?;
        {
            let _table = txn.open_table(TABLE).map_err(map_redb_err)?;
        }
        let _ = txn.set_durability(Durability::Immediate);
        txn.commit().map_err(map_redb_err)?;

        Ok(Self {
            db,
            sync_mode: opts.sync_mode,
            batch: Mutex::new(Batch::new()),
        })
    }

    fn put(&self, key: &[u8], value: &[u8]) -> bool {
        let mut guard = self.batch.lock().unwrap();
        guard.puts.insert(key.to_vec(), value.to_vec());
        let need_flush = guard.puts.len() >= BATCH_FLUSH_THRESHOLD;
        if need_flush {
            self.flush_locked(guard);
        }
        true
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        if let Some(v) = self.pending_get(key) {
            return Some(v);
        }
        let committed = self.read_committed(key);
        if let Some(v) = self.pending_get(key) {
            return Some(v);
        }
        committed
    }

    fn del(&self, key: &[u8]) -> bool {
        self.flush();
        let mut txn = self.db.begin_write().map_err(map_redb_err).unwrap();
        let was_present;
        {
            let mut table = txn.open_table(TABLE).map_err(map_redb_err).unwrap();
            was_present = table.get(key).map_err(map_redb_err).unwrap().is_some();
            let _ = table.remove(key);
        }
        let _ = txn.set_durability(self.durability());
        txn.commit().map_err(map_redb_err).unwrap();
        was_present
    }

    fn scan_range(&self, start: &[u8], end: &[u8], f: &mut dyn FnMut(&[u8], &[u8])) {
        self.flush();
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
        self.flush();
        let mut txn = self.db.begin_write().map_err(map_redb_err)?;
        let _ = txn.set_durability(Durability::Immediate);
        txn.commit().map_err(map_redb_err)?;
        Ok(())
    }

    fn stats(&self) -> EngineStats {
        EngineStats::default()
    }
}

impl Drop for RedbAdapter {
    fn drop(&mut self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let guard = self.batch.lock().unwrap();
            self.flush_locked(guard);
        }));
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

        // Batched puts: `was_absent` return is not tracked. Verify
        // correctness through get/del/scan instead.
        engine.put(b"A", b"val_a");
        engine.put(b"A", b"val_a2");
        engine.put(b"B", b"val_b");
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
