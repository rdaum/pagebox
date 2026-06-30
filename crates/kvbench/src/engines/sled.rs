//! Adapter wrapping `sled::Db` (LSM-tree comparison engine).
//!
//! sled's API is `BTreeMap`-like: `insert`, `get`, `remove`, `range` directly
//! on `sled::Db`. `SyncMode::Strict` maps to `sled::Mode::HighThroughput` with
//! explicit `flush()`; relaxed uses `LowSpace`.
//!
//! ## Memory budget equity
//!
//! sled's default `cache_capacity` is 1 GiB. This adapter sets the cache to
//! match kvstore's buffer-pool byte budget
//! (`buffer_budget_frames * PAGE_SIZE`) unless overridden via
//! `engine_specific["cache_size_bytes"]`.

use std::path::Path;

use crate::engine::{EngineOpts, EngineStats, KvEngine, SyncMode};

/// Page size used by the pagebox substrate (4 KiB).
const PAGE_SIZE: u64 = 4096;

/// Adapter wrapping `sled::Db` (the default tree).
pub struct SledAdapter {
    db: sled::Db,
    sync_mode: SyncMode,
}

impl KvEngine for SledAdapter {
    const NAME: &'static str = "sled";

    fn open(dir: &Path, opts: &EngineOpts) -> std::io::Result<Self> {
        let default_cache_bytes = (opts.buffer_budget_frames as u64) * PAGE_SIZE;
        let cache_bytes = opts
            .engine_specific
            .get("cache_size_bytes")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(default_cache_bytes);

        let db = sled::Config::new()
            .path(dir)
            .cache_capacity(cache_bytes)
            .mode(if opts.sync_mode == SyncMode::Strict {
                sled::Mode::HighThroughput
            } else {
                sled::Mode::LowSpace
            })
            .open()
            .map_err(map_sled_err)?;

        Ok(Self {
            db,
            sync_mode: opts.sync_mode,
        })
    }

    fn put(&self, key: &[u8], value: &[u8]) -> bool {
        // sled::insert returns the old value if the key existed.
        let old = self.db.insert(key, value).map_err(map_sled_err).unwrap();
        old.is_none()
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.db.get(key).ok().flatten().map(|ivec| ivec.to_vec())
    }

    fn del(&self, key: &[u8]) -> bool {
        // sled::remove returns the old value if the key existed.
        let old = self.db.remove(key).map_err(map_sled_err).unwrap();
        old.is_some()
    }

    fn scan_range(&self, start: &[u8], end: &[u8], f: &mut dyn FnMut(&[u8], &[u8])) {
        for item in self.db.range::<Vec<u8>, _>(start.to_vec()..end.to_vec()) {
            match item {
                Ok((k, v)) => f(k.as_ref(), v.as_ref()),
                Err(_) => break,
            }
        }
    }

    fn sync(&self) -> std::io::Result<()> {
        self.db.flush().map_err(map_sled_err)?;
        Ok(())
    }

    fn stats(&self) -> EngineStats {
        EngineStats::default()
    }
}

impl SledAdapter {
    #[allow(dead_code)]
    fn is_strict(&self) -> bool {
        self.sync_mode == SyncMode::Strict
    }
}

fn map_sled_err(e: impl std::fmt::Display) -> std::io::Error {
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
        let engine = SledAdapter::open(dir.path(), &test_opts()).unwrap();

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
