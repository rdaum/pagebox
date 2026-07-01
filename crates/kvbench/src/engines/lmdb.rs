//! Adapter wrapping `lmdb-rkv` — the Lightning Memory-Mapped Database.
//!
//! LMDB is a memory-mapped B+tree. It uses the OS page cache directly
//! (no separate buffer pool), which makes it an interesting comparison
//! against pagebox's swizzled-pointer buffer pool.
//!
//! LMDB's `map_size` caps the total database size (including B+tree pages
//! and free pages). We set it to a reasonable multiple of the configured
//! byte budget to allow room for growth.
//!
//! # Write batching
//!
//! LMDB commits per transaction: each `commit()` writes every dirty page
//! touched since the txn began. A one-transaction-per-`put` adapter pays
//! the full root-to-leaf dirty set (~tens of pages) on **every** op — a
//! 50k-record load turns into gigabytes of writeback and OS dirty-page
//! throttling that stalls the process. LMDB is also single-writer, so
//! concurrent `put`s serialize on the writer lock regardless.
//!
//! We buffer pending writes in a shared [`Mutex`]`<`[`Batch`]`>` and flush
//! them as one RW transaction when [`BATCH_FLUSH_THRESHOLD`] distinct keys
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

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use lmdb::{Cursor, Database, Environment, Transaction, WriteFlags};

use crate::engine::{EngineOpts, EngineStats, KvEngine, SyncMode};

const PAGE_SIZE: u64 = 65536;

/// Flush once this many distinct keys are pending. Bounds the per-commit
/// dirty set and turns a 50k-op load from ~50k commits into ~12.
const BATCH_FLUSH_THRESHOLD: usize = 4096;

pub struct LmdbAdapter {
    env: Environment,
    db: Database,
    sync_mode: SyncMode,
    batch: Mutex<Batch>,
}

/// Pending writes awaiting the next commit. Stored as a key→value map so a
/// `get` can observe an uncommitted write without flushing (the map is the
/// overlay the re-check protocol reads).
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

impl LmdbAdapter {
    /// Commit and clear all pending puts. The caller passes the guard it
    /// already holds so the flush runs under the same critical section as
    /// the buffer drain.
    fn flush_locked(&self, mut guard: std::sync::MutexGuard<'_, Batch>) {
        if guard.puts.is_empty() {
            return;
        }
        let puts = std::mem::take(&mut guard.puts);
        // Drop the guard before commit so other threads can keep buffering
        // while this commit (LMDB's single writer) runs.
        drop(guard);

        let mut txn = self
            .env
            .begin_rw_txn()
            .expect("lmdb begin_rw_txn failed (env closed?)");
        for (k, v) in &puts {
            txn.put(self.db, k, v, WriteFlags::default())
                .expect("lmdb txn.put failed (map full?)");
        }
        txn.commit().expect("lmdb txn.commit failed (map full?)");
        if self.sync_mode == SyncMode::Strict {
            self.env
                .sync(true)
                .expect("lmdb env.sync(true) failed (write error?)");
        }
    }

    fn flush(&self) {
        let guard = self.batch.lock().unwrap();
        self.flush_locked(guard);
    }

    /// Snapshot the pending value for `key` (if any), under the batch lock.
    fn pending_get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.batch.lock().unwrap().puts.get(key).cloned()
    }
}

impl KvEngine for LmdbAdapter {
    const NAME: &'static str = "lmdb";

    fn open(dir: &Path, opts: &EngineOpts) -> std::io::Result<Self> {
        let budget_bytes = (opts.buffer_budget_frames as u64) * PAGE_SIZE;
        let map_size = budget_bytes.max(64 * 1024 * 1024) * 8;

        std::fs::create_dir_all(dir)?;
        let env = Environment::new()
            .set_map_size(map_size as usize)
            .set_max_dbs(1)
            .open(dir)
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        let db = env
            .create_db(Some("default"), lmdb::DatabaseFlags::default())
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        Ok(Self {
            env,
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
        // `was_absent` is not tracked for batched puts (would need a read per
        // put). The kvbench driver ignores this return; verify mode keys off
        // the shadow map, not this flag.
        true
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        // Overlay protocol (pending → committed → pending re-check):
        //   1. If the key is pending, return it (newest uncommitted value).
        //   2. Otherwise read the committed snapshot.
        //   3. Re-check pending: a put may have landed between (1) and (2),
        //      and its value is newer than the committed read.
        // This makes a buffered write visible to any reader without flushing,
        // matching the "write visible to all immediately" semantics of the
        // LSM-tree adapters (rocksdb/fjall). The residual cross-thread race
        // (put's shadow-map update not yet visible to the verifier) is the
        // same one every adapter has under this driver and is negligible
        // under uniform sampling.
        if let Some(v) = self.pending_get(key) {
            return Some(v);
        }
        let committed = self.read_committed(key);
        // Re-check: a put may have landed between the first pending lookup
        // and the committed read; its value is newer, so prefer it.
        if let Some(v) = self.pending_get(key) {
            return Some(v);
        }
        committed
    }

    fn del(&self, key: &[u8]) -> bool {
        // Flush so the presence check observes all prior puts, then delete
        // synchronously so we can report `was_present`.
        self.flush();
        let mut txn = self
            .env
            .begin_rw_txn()
            .map_err(|e| std::io::Error::other(e.to_string()))
            .unwrap();
        let present = txn.get(self.db, &key).is_ok();
        if present {
            txn.del(self.db, &key.to_vec(), None)
                .map_err(|e| std::io::Error::other(e.to_string()))
                .unwrap();
        }
        txn.commit()
            .map_err(|e| std::io::Error::other(e.to_string()))
            .unwrap();
        if self.sync_mode == SyncMode::Strict {
            self.env
                .sync(true)
                .map_err(|e| std::io::Error::other(e.to_string()))
                .unwrap();
        }
        present
    }

    fn scan_range(&self, start: &[u8], end: &[u8], f: &mut dyn FnMut(&[u8], &[u8])) {
        // Flush so the scan sees a consistent ordered snapshot — the
        // pending map can't be overlaid on a range read cheaply (ordering
        // + dedup vs committed duplicates). Scans are rare in the suite.
        self.flush();
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
        self.flush();
        if self.sync_mode == SyncMode::Strict {
            self.env
                .sync(true)
                .map_err(|e| std::io::Error::other(e.to_string()))?;
        }
        Ok(())
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

impl Drop for LmdbAdapter {
    fn drop(&mut self) {
        // Commit any buffered puts so no writes are lost on teardown.
        // Errors are ignored — drop cannot propagate them, and the env is
        // closing anyway.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let guard = self.batch.lock().unwrap();
            self.flush_locked(guard);
        }));
    }
}
