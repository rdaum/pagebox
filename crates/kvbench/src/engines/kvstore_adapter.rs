//! Adapter wrapping `kvstore::KvStore` (the pagebox substrate composition).

use std::path::Path;

use kvstore::{KvStore, KvStoreOptions, SyncMode as KvSyncMode, TreeBackend};

use crate::engine::{EngineOpts, EngineStats, KvEngine, SyncMode};

pub struct KvstoreAdapter {
    inner: KvStore,
}

impl KvEngine for KvstoreAdapter {
    const NAME: &'static str = "kvstore";

    fn open(dir: &Path, opts: &EngineOpts) -> std::io::Result<Self> {
        let _ = opts.engine_specific.get("tree_backend");
        let tree_backend = TreeBackend::BPlusTree;
        let kv_opts = KvStoreOptions::default()
            .pool_frames(opts.buffer_budget_frames)
            .sync_mode(match opts.sync_mode {
                SyncMode::Relaxed => KvSyncMode::Relaxed,
                SyncMode::Strict => KvSyncMode::Strict,
            })
            .tree_backend(tree_backend);
        let inner = KvStore::open_with(dir, &kv_opts)?;
        Ok(Self { inner })
    }

    fn put(&self, key: &[u8], value: &[u8]) -> bool {
        self.inner.put(key, value)
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.inner.get(key)
    }

    fn del(&self, key: &[u8]) -> bool {
        self.inner.del(key)
    }

    fn scan_range(&self, start: &[u8], end: &[u8], f: &mut dyn FnMut(&[u8], &[u8])) {
        self.inner.scan_range(start, end, f);
    }

    fn sync(&self) -> std::io::Result<()> {
        self.inner.sync()
    }

    fn stats(&self) -> EngineStats {
        EngineStats::default()
    }
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
        let engine = KvstoreAdapter::open(dir.path(), &test_opts()).unwrap();

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

    #[test]
    fn adapter_strict_sync() {
        let dir = TempDir::new().unwrap();
        let opts = EngineOpts {
            sync_mode: SyncMode::Strict,
            ..test_opts()
        };
        let engine = KvstoreAdapter::open(dir.path(), &opts).unwrap();
        assert!(engine.put(b"k1", b"v1"));
        // Strict mode should have flushed the WAL; reopen and verify.
        drop(engine);
        let engine2 = KvstoreAdapter::open(dir.path(), &test_opts()).unwrap();
        assert_eq!(engine2.get(b"k1"), Some(b"v1".to_vec()));
    }
}
