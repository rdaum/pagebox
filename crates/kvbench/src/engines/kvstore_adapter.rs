//! Adapter wrapping `kvstore::KvStore` (the pagebox substrate composition).

use std::path::Path;

use kvstore::{KvStore, KvStoreOptions, PAGE_SIZE, SyncMode as KvSyncMode, TreeBackend};

use crate::engine::{
    CacheControl, EngineOpts, EngineStats, KvEngine, StorageIoStats, SyncMode,
    WalBufferRecordHistogram, WalMemoryStats, WalShardMemoryStats,
};

pub struct KvstoreAdapter {
    inner: KvStore,
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
        Ok(Self { inner })
    }

    fn put(&self, key: &[u8], value: &[u8]) {
        let _ = self.inner.put(key, value);
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.inner.get(key)
    }

    fn del(&self, key: &[u8]) {
        let _ = self.inner.del(key);
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

    fn begin_measurement(&self) {
        self.inner.reset_wal_memory_high_water_marks();
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
        let io = self.inner.page_store_io_stats();
        let wal = self.inner.wal_memory_stats();
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
            (
                "page_table_lock_contentions",
                buffer.page_table_lock_contentions,
            ),
            ("page_table_lock_wait_ns", buffer.page_table_lock_wait_ns),
            ("loading_frame_waits", buffer.loading_frame_waits),
            ("loading_frame_wait_ns", buffer.loading_frame_wait_ns),
            (
                "eviction_final_lock_waits",
                buffer.eviction_final_lock_waits,
            ),
            (
                "eviction_final_lock_wait_ns",
                buffer.eviction_final_lock_wait_ns,
            ),
            ("dirty_flush_batches", buffer.dirty_flush_batches),
            (
                "foreground_dirty_flush_batches",
                buffer.foreground_dirty_flush_batches,
            ),
            (
                "background_dirty_flush_batches",
                buffer.background_dirty_flush_batches,
            ),
            ("dirty_flush_pages", buffer.dirty_flush_pages),
            (
                "foreground_dirty_flush_pages",
                buffer.foreground_dirty_flush_pages,
            ),
            (
                "background_dirty_flush_pages",
                buffer.background_dirty_flush_pages,
            ),
            ("dirty_flush_wal_wait_ns", buffer.dirty_flush_wal_wait_ns),
            (
                "dirty_flush_data_write_ns",
                buffer.dirty_flush_data_write_ns,
            ),
            (
                "foreground_dirty_flush_data_write_ns",
                buffer.foreground_dirty_flush_data_write_ns,
            ),
            (
                "background_dirty_flush_data_write_ns",
                buffer.background_dirty_flush_data_write_ns,
            ),
            (
                "dirty_flush_cleaned_pages",
                buffer.dirty_flush_cleaned_pages,
            ),
            (
                "foreground_dirty_flush_cleaned_pages",
                buffer.foreground_dirty_flush_cleaned_pages,
            ),
            (
                "background_dirty_flush_cleaned_pages",
                buffer.background_dirty_flush_cleaned_pages,
            ),
            ("dirty_flush_stale_pages", buffer.dirty_flush_stale_pages),
            (
                "foreground_dirty_flush_stale_pages",
                buffer.foreground_dirty_flush_stale_pages,
            ),
            (
                "background_dirty_flush_stale_pages",
                buffer.background_dirty_flush_stale_pages,
            ),
            (
                "dirty_wal_page_patch_records",
                buffer.dirty_wal_page_patch_records,
            ),
            (
                "dirty_wal_page_patch_bytes",
                buffer.dirty_wal_page_patch_bytes,
            ),
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
            cache_access_unit: Some("buffer_pool_page_loads".to_string()),
            cache_evictions: Some(cache_evictions),
            cache_insert_bytes: Some(cache_misses.saturating_mul(PAGE_SIZE as u64)),
            storage_io: Some(StorageIoStats {
                read_calls: Some(io.reads.calls),
                read_requested_bytes: Some(io.reads.requested_bytes),
                read_completed_bytes: Some(io.reads.completed_bytes),
                write_calls: Some(io.writes.calls),
                write_requested_bytes: Some(io.writes.requested_bytes),
                write_completed_bytes: Some(io.writes.completed_bytes),
                direct_read_completed_bytes: Some(io.direct_reads.completed_bytes),
                direct_write_completed_bytes: Some(io.direct_writes.completed_bytes),
                buffered_read_completed_bytes: Some(io.buffered_reads.completed_bytes),
                buffered_write_completed_bytes: Some(io.buffered_writes.completed_bytes),
                foreground_read_completed_bytes: Some(io.foreground_reads.completed_bytes),
                prefetch_read_completed_bytes: Some(io.prefetch_reads.completed_bytes),
                recovery_read_completed_bytes: Some(io.recovery_reads.completed_bytes),
                foreground_write_completed_bytes: Some(io.foreground_writes.completed_bytes),
                background_write_completed_bytes: Some(io.background_writes.completed_bytes),
                checkpoint_write_completed_bytes: Some(io.checkpoint_writes.completed_bytes),
                recovery_write_completed_bytes: Some(io.recovery_writes.completed_bytes),
                metadata_write_completed_bytes: Some(io.metadata_writes.completed_bytes),
                batched_read_calls: Some(io.batched_reads.calls),
                batched_read_pages: Some(io.batched_reads.pages),
                batched_write_calls: Some(io.batched_writes.calls),
                batched_write_pages: Some(io.batched_writes.pages),
                sync_calls: Some(io.sync_calls),
                foreground_sync_calls: Some(io.foreground_sync_calls),
                checkpoint_sync_calls: Some(io.checkpoint_sync_calls),
                recovery_sync_calls: Some(io.recovery_sync_calls),
            }),
            wal_memory: Some(WalMemoryStats {
                shard_count: wal.shard_count,
                configured_buffer_record_capacity: wal.configured_buffer_record_capacity,
                configured_buffer_capacity_bytes: wal.configured_buffer_capacity_bytes,
                active_buffer_count: wal.active_buffer_count,
                spare_buffer_count: wal.spare_buffer_count,
                pending_buffer_count: wal.pending_buffer_count,
                in_flight_buffer_count: wal.in_flight_buffer_count,
                allocated_buffer_count: wal.allocated_buffer_count,
                virtual_buffer_capacity_bytes: wal.virtual_buffer_capacity_bytes,
                known_touched_buffer_bytes: wal.known_touched_buffer_bytes,
                active_used_bytes: wal.active_used_bytes,
                active_used_high_water_bytes: wal.active_used_high_water_bytes,
                max_submitted_buffer_records: wal.max_submitted_buffer_records,
                max_submitted_batch_records: wal.max_submitted_batch_records,
                submitted_buffer_count: wal.submitted_buffer_count,
                submitted_buffer_records: wal.submitted_buffer_records,
                submitted_buffer_record_histogram: WalBufferRecordHistogram {
                    one: wal.submitted_buffer_record_histogram.one,
                    two_to_seven: wal.submitted_buffer_record_histogram.two_to_seven,
                    eight_to_thirty_one: wal.submitted_buffer_record_histogram.eight_to_thirty_one,
                    thirty_two_to_sixty_three: wal
                        .submitted_buffer_record_histogram
                        .thirty_two_to_sixty_three,
                    sixty_four_to_two_fifty_five: wal
                        .submitted_buffer_record_histogram
                        .sixty_four_to_two_fifty_five,
                    two_fifty_six_to_one_thousand_twenty_three: wal
                        .submitted_buffer_record_histogram
                        .two_fifty_six_to_one_thousand_twenty_three,
                    one_thousand_twenty_four_or_more: wal
                        .submitted_buffer_record_histogram
                        .one_thousand_twenty_four_or_more,
                },
                flush_calls: wal.flush_calls,
                flush_waits: wal.flush_waits,
                buffer_backpressure_waits: wal.buffer_backpressure_waits,
                write_calls: wal.write_calls,
                write_bytes: wal.write_bytes,
                sync_calls: wal.sync_calls,
                durable_advances: wal.durable_advances,
                page_image_records_appended: wal.page_image_records_appended,
                page_image_bytes_appended: wal.page_image_bytes_appended,
                logical_bytes_appended: wal.logical_bytes_appended,
                shards: wal
                    .shards
                    .iter()
                    .map(|shard| WalShardMemoryStats {
                        configured_buffer_record_capacity: shard.configured_buffer_record_capacity,
                        configured_buffer_capacity_bytes: shard.configured_buffer_capacity_bytes,
                        active_buffer_count: shard.active_buffer_count,
                        spare_buffer_count: shard.spare_buffer_count,
                        pending_buffer_count: shard.pending_buffer_count,
                        in_flight_buffer_count: shard.in_flight_buffer_count,
                        allocated_buffer_count: shard.allocated_buffer_count,
                        virtual_buffer_capacity_bytes: shard.virtual_buffer_capacity_bytes,
                        known_touched_buffer_bytes: shard.known_touched_buffer_bytes,
                        active_used_bytes: shard.active_used_bytes,
                        active_used_high_water_bytes: shard.active_used_high_water_bytes,
                        max_submitted_buffer_records: shard.max_submitted_buffer_records,
                        max_submitted_batch_records: shard.max_submitted_batch_records,
                        submitted_buffer_count: shard.submitted_buffer_count,
                        submitted_buffer_records: shard.submitted_buffer_records,
                        submitted_buffer_record_histogram: WalBufferRecordHistogram {
                            one: shard.submitted_buffer_record_histogram.one,
                            two_to_seven: shard.submitted_buffer_record_histogram.two_to_seven,
                            eight_to_thirty_one: shard
                                .submitted_buffer_record_histogram
                                .eight_to_thirty_one,
                            thirty_two_to_sixty_three: shard
                                .submitted_buffer_record_histogram
                                .thirty_two_to_sixty_three,
                            sixty_four_to_two_fifty_five: shard
                                .submitted_buffer_record_histogram
                                .sixty_four_to_two_fifty_five,
                            two_fifty_six_to_one_thousand_twenty_three: shard
                                .submitted_buffer_record_histogram
                                .two_fifty_six_to_one_thousand_twenty_three,
                            one_thousand_twenty_four_or_more: shard
                                .submitted_buffer_record_histogram
                                .one_thousand_twenty_four_or_more,
                        },
                        flush_calls: shard.flush_calls,
                        flush_waits: shard.flush_waits,
                        buffer_backpressure_waits: shard.buffer_backpressure_waits,
                        write_calls: shard.write_calls,
                        write_bytes: shard.write_bytes,
                        sync_calls: shard.sync_calls,
                        durable_advances: shard.durable_advances,
                        page_image_records_appended: shard.page_image_records_appended,
                        page_image_bytes_appended: shard.page_image_bytes_appended,
                        logical_bytes_appended: shard.logical_bytes_appended,
                    })
                    .collect(),
            }),
            live_data_bytes: Some(live_data_bytes),
            persisted_data_bytes: Some((self.inner.persisted_pages() * PAGE_SIZE) as u64),
            extra,
            ..EngineStats::default()
        }
    }

    fn phase_stats_since(&self, earlier: &EngineStats) -> EngineStats {
        const EXTRA_GAUGES: &[&str] = &[
            "resident_frames",
            "pinned_frames",
            "dirty_frames",
            "referenced_frames",
            "eviction_allowed_frames",
            "free_list_frames",
            "resident_budget_available",
            "eviction_in_flight",
        ];

        let mut stats = self.stats().phase_delta_since(earlier);
        for (name, value) in &mut stats.extra {
            if EXTRA_GAUGES.contains(&name.as_str()) {
                continue;
            }
            let (Ok(current), Some(Ok(before))) = (
                value.parse::<u64>(),
                earlier.extra.get(name).map(|value| value.parse::<u64>()),
            ) else {
                continue;
            };
            *value = current.saturating_sub(before).to_string();
        }
        stats
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

    #[test]
    fn adapter_reports_measured_phase_page_store_io() {
        let dir = TempDir::new().unwrap();
        let engine = KvstoreAdapter::open(dir.path(), &test_opts()).unwrap();
        engine.begin_measurement();
        let before = engine.stats();
        engine.put(b"key", b"value");
        engine.put(b"key", b"other");
        engine.inner.checkpoint().unwrap();
        let phase = engine.phase_stats_since(&before);
        let io = phase
            .storage_io
            .expect("kvstore must expose page-store I/O");
        assert!(
            io.write_calls.unwrap() > 0,
            "checkpoint must issue page-store writes"
        );
        assert_eq!(
            io.write_requested_bytes, io.write_completed_bytes,
            "successful full-page writes must complete every requested byte"
        );
        assert_eq!(
            phase
                .extra
                .get("dirty_wal_page_patch_records")
                .map(String::as_str),
            Some("1"),
            "buffer-pool diagnostic counters must also be phase deltas"
        );
    }
}
