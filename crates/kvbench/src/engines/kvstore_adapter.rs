//! Adapter wrapping `kvstore::KvStore` (the pagebox substrate composition).

use std::path::Path;

use kvstore::{KvStore, KvStoreOptions, SyncMode as KvSyncMode, TreeBackend};

use crate::engine::{CacheControl, EngineOpts, EngineStats, KvEngine, SyncMode};

const PAGE_SIZE: usize = 65_536;

pub struct KvstoreAdapter {
    inner: KvStore,
    sync_mode: SyncMode,
}

impl KvEngine for KvstoreAdapter {
    const NAME: &'static str = "kvstore";
    const CACHE_CONTROL: CacheControl = CacheControl::Application;
    const SUPPORTS_DIRECT_IO: bool = true;

    fn open(dir: &Path, opts: &EngineOpts) -> std::io::Result<Self> {
        // SAFETY: kvbench runs one engine per process.
        unsafe {
            std::env::set_var(
                "PAGEBOX_PAGE_STORE_DIRECT_IO",
                if opts.direct_io { "1" } else { "0" },
            )
        };
        // The WAL backend is selected via env var (read by Wal::open_opts).
        // Set it before opening so the spec / CLI override takes effect.
        if let Some(ref backend) = opts.wal_backend {
            // SAFETY: setting an env var is process-global; the benchmark
            // runs one engine per process so there is no interference.
            unsafe { std::env::set_var("PAGEBOX_WAL_SYNC_BACKEND", backend) };
            eprintln!("  WAL backend: {backend}");
        }

        let _ = opts.engine_specific.get("tree_backend");
        let tree_backend = TreeBackend::BPlusTree;
        if !opts.cache_budget_bytes.is_multiple_of(PAGE_SIZE) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "kvstore cache budget {} must be a multiple of its {}-byte page size",
                    opts.cache_budget_bytes, PAGE_SIZE
                ),
            ));
        }
        let pool_frames = opts.cache_budget_bytes / PAGE_SIZE;
        if pool_frames == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "kvstore cache budget must hold at least one page",
            ));
        }
        let kv_opts = KvStoreOptions::default()
            .pool_frames(pool_frames)
            .sync_mode(match opts.sync_mode {
                SyncMode::Relaxed => KvSyncMode::Relaxed,
                SyncMode::Strict => KvSyncMode::Strict,
            })
            .tree_backend(tree_backend);
        let inner = KvStore::open_with(dir, &kv_opts)?;
        Ok(Self {
            inner,
            sync_mode: opts.sync_mode,
        })
    }

    fn put(&self, key: &[u8], value: &[u8]) {
        let _ = self.inner.put(key, value);
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.inner.get(key)
    }

    fn del(&self, key: &[u8]) {
        let _ = self.inner.del(key);
        if self.sync_mode == SyncMode::Strict {
            self.inner.flush_wal();
        }
    }

    fn scan_range(&self, start: &[u8], end: &[u8], f: &mut dyn FnMut(&[u8], &[u8])) {
        self.inner.scan_range(start, end, f);
    }

    fn sync(&self) -> std::io::Result<()> {
        self.inner.flush_wal();
        Ok(())
    }

    fn prepare_for_reopen(&self) -> std::io::Result<()> {
        self.inner.checkpoint()
    }

    fn stats(&self) -> EngineStats {
        let mut extra = std::collections::HashMap::new();
        if let Ok(backend) = std::env::var("PAGEBOX_WAL_SYNC_BACKEND") {
            extra.insert("wal_backend".to_string(), backend);
        }
        let cache_misses = self.inner.cache_misses();
        let cache_evictions = self.inner.cache_evictions();
        let buffer = self.inner.buffer_pool_diagnostic_stats();
        let btree = self.inner.btree_diagnostic_stats();
        let diagnostics = [
            ("load_inner", buffer.inner_index_loads),
            ("load_leaf", buffer.leaf_index_loads),
            ("load_tuple", buffer.tuple_loads),
            ("load_delta", buffer.delta_loads),
            ("load_meta", buffer.resident_meta_loads),
            ("load_unknown", buffer.unknown_loads),
            ("parent_hint_hits", buffer.parent_hint_hits),
            ("parent_hint_latch_misses", buffer.parent_hint_latch_misses),
            ("parent_dfs_fallbacks", buffer.parent_dfs_fallbacks),
            ("parent_dfs_failures", buffer.parent_dfs_failures),
            ("second_chance_skips", buffer.second_chance_skips),
            ("resident_frames", buffer.resident_frames),
            ("pinned_frames", buffer.pinned_frames),
            ("dirty_frames", buffer.dirty_frames),
            ("referenced_frames", buffer.referenced_frames),
            ("eviction_allowed_frames", buffer.eviction_allowed_frames),
            ("free_list_frames", buffer.free_list_frames),
            (
                "resident_budget_available",
                buffer.resident_budget_available,
            ),
            ("eviction_in_flight", buffer.eviction_in_flight),
            ("insert_restarts", btree.insert_restarts),
            ("leaf_descent_restarts", btree.leaf_descent_restarts),
            ("leaf_upgrade_restarts", btree.leaf_upgrade_restarts),
            ("split_path_restarts", btree.split_path_restarts),
            ("parent_publish_restarts", btree.parent_publish_restarts),
            ("parent_fallbacks", btree.parent_fallbacks),
            ("resolve_cold", btree.resolve_cold),
            ("unswizzle_calls", btree.eviction_unswizzle_calls),
            ("unswizzle_restarts", btree.eviction_unswizzle_restarts),
            (
                "unswizzle_parent_hits",
                btree.eviction_unswizzle_parent_hits,
            ),
            (
                "unswizzle_upgrade_failures",
                btree.eviction_unswizzle_upgrade_failures,
            ),
            (
                "unswizzle_nodes_visited",
                btree.eviction_unswizzle_nodes_visited,
            ),
        ];
        extra.extend(
            diagnostics
                .into_iter()
                .map(|(name, value)| (name.to_string(), value.to_string())),
        );
        let live_data_bytes = (self.inner.live_tree_pages() * PAGE_SIZE) as u64;
        EngineStats {
            direct_io: Some(self.inner.direct_io_enabled()),
            cache_capacity_bytes: Some((self.inner.cache_capacity_pages() * PAGE_SIZE) as u64),
            cache_used_bytes: Some((self.inner.cache_used_pages() * PAGE_SIZE) as u64),
            cache_misses: Some(cache_misses),
            cache_evictions: Some(cache_evictions),
            cache_insert_bytes: Some(cache_misses.saturating_mul(PAGE_SIZE as u64)),
            live_data_bytes: Some(live_data_bytes),
            persisted_data_bytes: Some((self.inner.persisted_pages() * PAGE_SIZE) as u64),
            extra,
            ..EngineStats::default()
        }
    }
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
        let engine = KvstoreAdapter::open(dir.path(), &test_opts()).unwrap();

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

    #[test]
    fn adapter_strict_sync() {
        let dir = TempDir::new().unwrap();
        let opts = EngineOpts {
            sync_mode: SyncMode::Strict,
            ..test_opts()
        };
        let engine = KvstoreAdapter::open(dir.path(), &opts).unwrap();
        engine.put(b"k1", b"v1");
        // Strict mode should have flushed the WAL; reopen and verify.
        drop(engine);
        let engine2 = KvstoreAdapter::open(dir.path(), &test_opts()).unwrap();
        assert_eq!(engine2.get(b"k1"), Some(b"v1".to_vec()));
    }

    #[test]
    fn adapter_persists_live_tree_bytes_across_reopen() {
        let dir = TempDir::new().unwrap();
        let engine = KvstoreAdapter::open(dir.path(), &test_opts()).unwrap();
        let value = [0x6d; 2_048];
        for key in 0..2_000u64 {
            engine.put(&key.to_be_bytes(), &value);
        }
        let before = engine.stats().live_data_bytes.unwrap();
        assert!(before > PAGE_SIZE as u64, "test must split the root leaf");
        engine.prepare_for_reopen().unwrap();
        drop(engine);

        let reopened = KvstoreAdapter::open(dir.path(), &test_opts()).unwrap();
        assert_eq!(
            reopened.stats().live_data_bytes,
            Some(before),
            "reachable-page accounting must survive checkpoint and reopen"
        );
    }
}
