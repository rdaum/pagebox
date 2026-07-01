//! Adapter wrapping `fjall::Database` (LSM-tree comparison engine).
//!
//! ## Memory budget equity
//!
//! Fjall's default block cache is 32 MiB. To make the comparison fair
//! against kvstore (which uses `buffer_budget_frames * PAGE_SIZE` bytes),
//! this adapter sets fjall's `cache_size` to match the same byte budget
//! unless overridden via `engine_specific["cache_size_bytes"]`.
//!
//! Additional overrides:
//! - `max_journaling_size_bytes`: fjall journal size (default 512 MiB).

use std::path::Path;

use fjall::{Database, KeyspaceCreateOptions, PersistMode};

use crate::engine::{EngineOpts, EngineStats, KvEngine, SyncMode};

/// Page size used by the pagebox substrate (64 KiB).
const PAGE_SIZE: u64 = 65536;

/// Adapter wrapping `fjall::Database` with a single keyspace.
pub struct FjallAdapter {
    db: Database,
    keyspace: fjall::Keyspace,
    sync_mode: SyncMode,
}

impl KvEngine for FjallAdapter {
    const NAME: &'static str = "fjall";

    fn open(dir: &Path, opts: &EngineOpts) -> std::io::Result<Self> {
        let mut builder = Database::builder(dir);

        // Default: match kvstore's buffer-pool byte budget so both engines
        // have the same memory budget for cached pages/blocks.
        let default_cache_bytes = (opts.buffer_budget_frames as u64) * PAGE_SIZE;
        let cache_bytes = opts
            .engine_specific
            .get("cache_size_bytes")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(default_cache_bytes);
        builder = builder.cache_size(cache_bytes);

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
            .keyspace("default", KeyspaceCreateOptions::default)
            .map_err(map_fjall_err)?;
        Ok(Self {
            db,
            keyspace,
            sync_mode: opts.sync_mode,
        })
    }

    fn put(&self, key: &[u8], value: &[u8]) -> bool {
        let was_present = self.keyspace.get(key).ok().flatten().is_some();
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
        !was_present
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.keyspace
            .get(key)
            .ok()
            .flatten()
            .map(|slice| slice.to_vec())
    }

    fn del(&self, key: &[u8]) -> bool {
        let was_present = self.keyspace.get(key).ok().flatten().is_some();
        self.keyspace.remove(key).map_err(map_fjall_err).unwrap();
        if self.sync_mode == SyncMode::Strict {
            self.db
                .persist(PersistMode::SyncAll)
                .map_err(map_fjall_err)
                .unwrap();
        }
        was_present
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

    fn stats(&self) -> EngineStats {
        EngineStats::default()
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
            value_size: 100,
            sync_mode: SyncMode::Relaxed,
            buffer_budget_frames: 1024,
            engine_specific: Default::default(),
        }
    }

    #[test]
    fn adapter_contract() {
        let dir = TempDir::new().unwrap();
        let engine = FjallAdapter::open(dir.path(), &test_opts()).unwrap();

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
