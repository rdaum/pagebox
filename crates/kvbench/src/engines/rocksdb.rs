//! Adapter wrapping `rocksdb::DB` — the mature LSM-tree engine.
//!
//! RocksDB exposes explicit MemTable and block-cache controls. This adapter
//! charges both to one configured cache and write-buffer manager for
//! application-cache comparisons. The operating-system page cache remains
//! outside that budget, as it does for the other file-backed adapters.

use std::path::Path;

use rocksdb::{
    BlockBasedOptions, Cache, DB, Options, ReadOptions, WriteBufferManager, WriteOptions,
    properties, statistics::Ticker,
};

use crate::engine::{CacheControl, EngineOpts, EngineStats, KvEngine, SyncMode};

pub struct RocksdbAdapter {
    db: DB,
    sync_mode: SyncMode,
    statistics: Options,
    cache: Cache,
    write_buffer_manager: WriteBufferManager,
    cache_budget_bytes: usize,
    direct_io: bool,
}

impl KvEngine for RocksdbAdapter {
    const NAME: &'static str = "rocksdb";
    const CACHE_CONTROL: CacheControl = CacheControl::Application;
    const SUPPORTS_DIRECT_IO: bool = true;

    fn open(dir: &Path, opts: &EngineOpts) -> std::io::Result<Self> {
        let cache_bytes = opts.cache_budget_bytes;
        let cache = Cache::new_lru_cache(cache_bytes);
        let write_buffer_manager = WriteBufferManager::new_write_buffer_manager_with_cache(
            cache_bytes,
            true,
            cache.clone(),
        );

        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);
        db_opts.enable_statistics();
        db_opts.set_use_direct_reads(opts.direct_io);
        db_opts.set_use_direct_io_for_flush_and_compaction(opts.direct_io);

        db_opts.set_write_buffer_manager(&write_buffer_manager);

        db_opts.set_max_background_jobs(8);
        db_opts.set_max_open_files(-1);

        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_block_cache(&cache);

        let mut cf_opts = Options::default();
        cf_opts.set_block_based_table_factory(&block_opts);
        cf_opts.set_write_buffer_size((cache_bytes / 4).max(1024 * 1024));
        cf_opts.set_max_write_buffer_number(4);
        cf_opts.set_min_write_buffer_number_to_merge(2);
        let cf_descriptors = vec![rocksdb::ColumnFamilyDescriptor::new("default", cf_opts)];

        let db = DB::open_cf_descriptors(&db_opts, dir, cf_descriptors)
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        Ok(Self {
            db,
            sync_mode: opts.sync_mode,
            statistics: db_opts,
            cache,
            write_buffer_manager,
            cache_budget_bytes: cache_bytes,
            direct_io: opts.direct_io,
        })
    }

    fn put(&self, key: &[u8], value: &[u8]) {
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.sync_mode == SyncMode::Strict);
        let cf = self.db.cf_handle("default").unwrap();
        self.db
            .put_cf_opt(&cf, key, value, &write_opts)
            .map_err(|e| std::io::Error::other(e.to_string()))
            .unwrap();
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let cf = self.db.cf_handle("default").unwrap();
        let read_opts = ReadOptions::default();
        self.db.get_cf_opt(&cf, key, &read_opts).ok().flatten()
    }

    fn del(&self, key: &[u8]) {
        let cf = self.db.cf_handle("default").unwrap();
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.sync_mode == SyncMode::Strict);
        self.db
            .delete_cf_opt(&cf, key, &write_opts)
            .map_err(|e| std::io::Error::other(e.to_string()))
            .unwrap();
    }

    fn scan_range(&self, start: &[u8], end: &[u8], f: &mut dyn FnMut(&[u8], &[u8])) {
        let cf = self.db.cf_handle("default").unwrap();
        let read_opts = ReadOptions::default();
        let iter = self.db.iterator_cf_opt(
            &cf,
            read_opts,
            rocksdb::IteratorMode::From(start, rocksdb::Direction::Forward),
        );
        for item in iter {
            let Ok((k, v)) = item else {
                break;
            };
            if k.as_ref() >= end {
                break;
            }
            f(k.as_ref(), v.as_ref());
        }
    }

    fn sync(&self) -> std::io::Result<()> {
        self.db
            .flush_wal(true)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    fn prepare_for_reopen(&self) -> std::io::Result<()> {
        let cf = self.db.cf_handle("default").unwrap();
        self.db
            .flush_cf(&cf)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        self.sync()
    }

    fn stats(&self) -> EngineStats {
        let cf = self.db.cf_handle("default").unwrap();
        let persisted_data_bytes = self
            .db
            .property_int_value_cf(&cf, properties::LIVE_SST_FILES_SIZE)
            .ok()
            .flatten();
        let mut extra = std::collections::HashMap::new();
        extra.insert(
            "write_buffer_bytes".to_string(),
            self.write_buffer_manager.get_usage().to_string(),
        );
        EngineStats {
            direct_io: Some(self.direct_io),
            cache_capacity_bytes: Some(self.cache_budget_bytes as u64),
            cache_used_bytes: Some(self.cache.get_usage() as u64),
            cache_hits: Some(self.statistics.get_ticker_count(Ticker::BlockCacheHit)),
            cache_misses: Some(self.statistics.get_ticker_count(Ticker::BlockCacheMiss)),
            cache_insert_bytes: Some(
                self.statistics
                    .get_ticker_count(Ticker::BlockCacheBytesWrite),
            ),
            storage_read_bytes: Some(self.statistics.get_ticker_count(Ticker::BytesRead)),
            persisted_data_bytes,
            extra,
            ..EngineStats::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn adapter_contract_does_not_require_presence_reads() {
        let dir = TempDir::new().unwrap();
        let engine = RocksdbAdapter::open(dir.path(), &EngineOpts::default()).unwrap();

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
