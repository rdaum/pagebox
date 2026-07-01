//! Adapter wrapping `rocksdb::DB` — the mature LSM-tree engine.
//!
//! RocksDB has bounded memory: `write_buffer_size` caps each MemTable,
//! `max_write_buffer_number` caps queued MemTables, and L0 file count
//! triggers write slowdown/halt. This makes it a fair comparison against
//! kvstore's bounded buffer pool, unlike sled which has unbounded MemTable.

use std::path::Path;

use rocksdb::{DB, Options, ReadOptions, WriteOptions};

use crate::engine::{EngineOpts, EngineStats, KvEngine, SyncMode};

const PAGE_SIZE: u64 = 65536;

pub struct RocksdbAdapter {
    db: DB,
    sync_mode: SyncMode,
}

impl KvEngine for RocksdbAdapter {
    const NAME: &'static str = "rocksdb";

    fn open(dir: &Path, opts: &EngineOpts) -> std::io::Result<Self> {
        let cache_bytes = (opts.buffer_budget_frames as u64) * PAGE_SIZE;

        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);

        db_opts.set_write_buffer_size(cache_bytes as usize);
        db_opts.set_max_write_buffer_number(4);
        db_opts.set_min_write_buffer_number_to_merge(2);

        db_opts.set_max_background_jobs(8);
        db_opts.set_max_open_files(-1);

        let cf_opts = Options::default();
        let cf_descriptors = vec![rocksdb::ColumnFamilyDescriptor::new("default", cf_opts)];

        let db = DB::open_cf_descriptors(&db_opts, dir, cf_descriptors)
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        Ok(Self {
            db,
            sync_mode: opts.sync_mode,
        })
    }

    fn put(&self, key: &[u8], value: &[u8]) -> bool {
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.sync_mode == SyncMode::Strict);
        let cf = self.db.cf_handle("default").unwrap();
        let old = self.db.get_cf(&cf, key).ok().flatten();
        self.db
            .put_cf_opt(&cf, key, value, &write_opts)
            .map_err(|e| std::io::Error::other(e.to_string()))
            .unwrap();
        old.is_none()
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let cf = self.db.cf_handle("default").unwrap();
        let read_opts = ReadOptions::default();
        self.db.get_cf_opt(&cf, key, &read_opts).ok().flatten()
    }

    fn del(&self, key: &[u8]) -> bool {
        let cf = self.db.cf_handle("default").unwrap();
        let old = self.db.get_cf(&cf, key).ok().flatten();
        let write_opts = WriteOptions::default();
        let _ = self.db.delete_cf_opt(&cf, key, &write_opts);
        old.is_some()
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
        let cf = self.db.cf_handle("default").unwrap();
        self.db
            .flush_cf(&cf)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    fn stats(&self) -> EngineStats {
        EngineStats::default()
    }
}
