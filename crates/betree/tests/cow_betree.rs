#![cfg(feature = "metrics")]

use std::collections::BTreeMap;
use std::ops::Bound;
use std::sync::{Arc, Barrier};

use fast_telemetry::{HistogramSnapshot, MetricLabels, MetricMeta, MetricVisitor};
use pagebox_betree::{
    CowBeTree, CowBeTreeConfig, CowBeTreeGcCursor, CowBeTreeMessage, PageKindDebug,
};
use pagebox_storage::buffer_frame::PAGE_SIZE;
use pagebox_storage::buffer_pool::BufferPool;

type ModelVersions = Vec<(u64, Option<Vec<u8>>)>;
type VersionModel = BTreeMap<Vec<u8>, ModelVersions>;

#[derive(Default)]
struct TestCowBeTreeMetrics {
    root_buffer_appends: u64,
    buffer_flushes: u64,
    messages_flushed: u64,
    fork_page_copies: u64,
    in_place_page_rewrites: u64,
    internal_buffer_sorted_rewrites: u64,
    leaf_batch_rewrites: u64,
    direct_leaf_flushes: u64,
    direct_leaf_flush_messages: u64,
    raw_buffer_appends: u64,
    raw_leaf_appends: u64,
    raw_leaf_append_batches: u64,
    leaf_splits: u64,
    internal_splits: u64,
    leaf_merges: u64,
    root_replacements: u64,
    root_collapses: u64,
    gc_runs: u64,
    gc_versions_pruned: u64,
    gc_leaf_pages_rewritten: u64,
    gc_cursor_wraps: u64,
    leaf_rebuild_bytes: u64,
    page_image_rewrite_bytes: u64,
    leaf_page_image_rewrite_bytes: u64,
    internal_page_image_rewrite_bytes: u64,
}

impl MetricVisitor for TestCowBeTreeMetrics {
    fn counter(&mut self, meta: MetricMeta<'_>, labels: MetricLabels<'_>, value: i64) {
        let value = value.max(0) as u64;
        if meta.name == "cow_betree_events" {
            let Some(label) = labels.iter().next() else {
                return;
            };
            match label.value {
                "root_buffer_appends" => self.root_buffer_appends += value,
                "buffer_flushes" => self.buffer_flushes += value,
                "messages_flushed" => self.messages_flushed += value,
                "fork_page_copies" => self.fork_page_copies += value,
                "in_place_page_rewrites" => self.in_place_page_rewrites += value,
                "internal_buffer_sorted_rewrites" => self.internal_buffer_sorted_rewrites += value,
                "leaf_batch_rewrites" => self.leaf_batch_rewrites += value,
                "direct_leaf_flushes" => self.direct_leaf_flushes += value,
                "direct_leaf_flush_messages" => self.direct_leaf_flush_messages += value,
                "raw_buffer_appends" => self.raw_buffer_appends += value,
                "raw_leaf_appends" => self.raw_leaf_appends += value,
                "raw_leaf_append_batches" => self.raw_leaf_append_batches += value,
                "leaf_splits" => self.leaf_splits += value,
                "internal_splits" => self.internal_splits += value,
                "leaf_merges" => self.leaf_merges += value,
                "root_replacements" => self.root_replacements += value,
                "root_collapses" => self.root_collapses += value,
                "gc_runs" => self.gc_runs += value,
                "gc_versions_pruned" => self.gc_versions_pruned += value,
                "gc_leaf_pages_rewritten" => self.gc_leaf_pages_rewritten += value,
                "gc_cursor_wraps" => self.gc_cursor_wraps += value,
                _ => {}
            }
            return;
        }
        match meta.name {
            "cow_betree_leaf_rebuild_bytes" => self.leaf_rebuild_bytes += value,
            "cow_betree_leaf_page_image_rewrite_bytes" => {
                self.leaf_page_image_rewrite_bytes += value
            }
            "cow_betree_internal_page_image_rewrite_bytes" => {
                self.internal_page_image_rewrite_bytes += value
            }
            _ => {}
        }
    }

    fn gauge_i64(&mut self, _meta: MetricMeta<'_>, _labels: MetricLabels<'_>, _value: i64) {}

    fn gauge_f64(&mut self, _meta: MetricMeta<'_>, _labels: MetricLabels<'_>, _value: f64) {}

    fn histogram(
        &mut self,
        _meta: MetricMeta<'_>,
        _labels: MetricLabels<'_>,
        _histogram: &dyn HistogramSnapshot,
    ) {
    }
}

fn tree_metrics(tree: &CowBeTree) -> TestCowBeTreeMetrics {
    let mut metrics = TestCowBeTreeMetrics::default();
    tree.visit_metrics(&mut metrics);
    metrics.page_image_rewrite_bytes =
        metrics.leaf_page_image_rewrite_bytes + metrics.internal_page_image_rewrite_bytes;
    metrics
}

fn tiny_config() -> CowBeTreeConfig {
    CowBeTreeConfig {
        flush_threshold_messages: 8,
        flush_threshold_bytes: 512,
        max_leaf_entries: 4,
        max_internal_children: 4,
        merge_leaf_entries: 16,
        merge_internal_children: 8,
    }
}

#[test]
fn default_config_derives_fanout_and_flush_budget_from_page_size() {
    let config = CowBeTreeConfig::default();

    assert_eq!(
        config.max_internal_children, 8,
        "64 KiB pages should derive a bounded epsilon-style fanout"
    );
    assert_eq!(
        config.flush_threshold_bytes,
        16 * 1024,
        "64 KiB pages should use a bounded batch flush budget"
    );
    assert_eq!(
        config.flush_threshold_messages, 256,
        "message threshold should be derived from the byte flush budget"
    );
}

#[test]
fn page_image_rewrite_bytes_track_full_page_size_not_encoded_payload() {
    let pool = std::sync::Arc::new(BufferPool::new(2));
    let tree = CowBeTree::with_config(&pool, CowBeTreeConfig::default());

    tree.put(b"hot", 1, b"old").unwrap();
    let before = tree_metrics(&tree);

    tree.put(b"hot", 2, b"new").unwrap();

    let after = tree_metrics(&tree);
    let page_image_delta = after.page_image_rewrite_bytes - before.page_image_rewrite_bytes;
    let leaf_page_image_delta =
        after.leaf_page_image_rewrite_bytes - before.leaf_page_image_rewrite_bytes;
    let internal_page_image_delta =
        after.internal_page_image_rewrite_bytes - before.internal_page_image_rewrite_bytes;
    let encoded_leaf_delta = after.leaf_rebuild_bytes - before.leaf_rebuild_bytes;

    assert_eq!(
        after.in_place_page_rewrites - before.in_place_page_rewrites,
        1,
        "same-key update in a leaf should require one full page rewrite"
    );
    assert_eq!(
        page_image_delta, PAGE_SIZE as u64,
        "page-image telemetry should count the full page image copied into the frame"
    );
    assert_eq!(
        leaf_page_image_delta, page_image_delta,
        "leaf page-image counter should account for the full rewrite"
    );
    assert_eq!(
        internal_page_image_delta, 0,
        "leaf-only update should not charge internal page-image bytes"
    );
    assert!(
        page_image_delta > encoded_leaf_delta,
        "page-image telemetry should expose the copy cost hidden by encoded payload bytes"
    );
}

fn be_key(n: u16) -> [u8; 2] {
    n.to_be_bytes()
}

fn visible_model(model: &VersionModel, key: &[u8], read_ts: u64) -> Option<Vec<u8>> {
    model.get(key).and_then(|versions| {
        versions
            .iter()
            .find(|(commit_ts, _)| *commit_ts <= read_ts)
            .and_then(|(_, value)| value.clone())
    })
}

#[test]
fn root_buffered_writes_do_not_rebuild_leaves_before_flush_threshold() {
    let pool = std::sync::Arc::new(BufferPool::new(256));
    let tree = CowBeTree::with_config(&pool, tiny_config());

    for i in 0u16..5 {
        tree.put(&be_key(i), 1, &be_key(i)).unwrap();
    }
    assert_eq!(
        tree.debug_snapshot().root_kind,
        Some(PageKindDebug::Internal),
        "five inserts with max_leaf_entries=4 should split the root leaf"
    );
    let _fork = tree.fork().unwrap();
    let before = tree_metrics(&tree);

    for i in 20u16..24 {
        tree.put(&be_key(i), 2, &be_key(i)).unwrap();
    }

    let after = tree_metrics(&tree);
    let debug = tree.debug_snapshot();
    assert_eq!(
        after.leaf_rebuild_bytes, before.leaf_rebuild_bytes,
        "below-threshold writes must stay in the internal message buffer"
    );
    assert_eq!(
        after.root_buffer_appends - before.root_buffer_appends,
        4,
        "writes to an internal root should append logical messages"
    );
    assert_eq!(
        after.raw_buffer_appends - before.raw_buffer_appends,
        4,
        "below-threshold root messages should use the raw in-place append path"
    );
    assert_eq!(
        debug.buffered_messages, 4,
        "debug snapshot should expose buffered path messages"
    );
    for i in 20u16..24 {
        assert_eq!(
            tree.lookup(&be_key(i)).as_deref(),
            Some(&be_key(i)[..]),
            "lookup must merge root-buffer messages before they reach leaves"
        );
    }
}

#[test]
fn page_local_leaf_rewrites_existing_keys_without_root_buffering() {
    let pool = std::sync::Arc::new(BufferPool::new(256));
    let tree = CowBeTree::with_config(&pool, tiny_config());

    for i in 0u16..5 {
        tree.put(&be_key(i), 10, &be_key(i)).unwrap();
    }
    assert_eq!(
        tree.debug_snapshot().root_kind,
        Some(PageKindDebug::Internal),
        "test setup should create an internal root before page-local updates"
    );
    let before = tree_metrics(&tree);

    for commit_ts in 20u64..24 {
        tree.put(&be_key(1), commit_ts, &[commit_ts as u8]).unwrap();
    }

    let after = tree_metrics(&tree);
    assert_eq!(
        after.root_buffer_appends, before.root_buffer_appends,
        "existing-key updates should rewrite the target leaf without appending to the root buffer"
    );
    assert_eq!(
        after.raw_buffer_appends, before.raw_buffer_appends,
        "leaf-local rewrites should not create buffered path messages"
    );
    assert_eq!(
        after.in_place_page_rewrites - before.in_place_page_rewrites,
        4,
        "each same-key update should rewrite only its target leaf page"
    );
    assert_eq!(
        tree.lookup(&be_key(1)),
        Some(vec![23]),
        "latest page-local leaf rewrite should be visible"
    );
    assert_eq!(
        tree.lookup_visible(&be_key(1), 10).as_deref(),
        Some(&be_key(1)[..]),
        "page-local leaf rewrite should retain older MVCC versions"
    );
}

#[test]
fn page_local_leaf_path_buffers_new_non_append_keys() {
    let pool = std::sync::Arc::new(BufferPool::new(256));
    let tree = CowBeTree::with_config(&pool, tiny_config());

    for key in [b"a", b"c", b"e", b"g", b"i"] {
        tree.put(key, 10, key).unwrap();
    }
    assert_eq!(
        tree.debug_snapshot().root_kind,
        Some(PageKindDebug::Internal),
        "test setup should create an internal root before the non-append insert"
    );
    let before = tree_metrics(&tree);

    tree.put(b"b", 20, b"buffered").unwrap();

    let after = tree_metrics(&tree);
    assert_eq!(
        after.in_place_page_rewrites, before.in_place_page_rewrites,
        "new non-append keys should stay on the B-epsilon buffered path instead of rewriting a leaf"
    );
    assert_eq!(
        after.root_buffer_appends - before.root_buffer_appends,
        1,
        "height-one inserts that cannot append to a leaf should buffer at the root"
    );
    assert_eq!(
        after.raw_buffer_appends - before.raw_buffer_appends,
        1,
        "the root-buffer insert should use the raw page append path"
    );
    assert_eq!(
        tree.lookup(b"b").as_deref(),
        Some(&b"buffered"[..]),
        "lookup should merge the buffered non-append insert"
    );
}

#[test]
fn insert_new_message_buffers_new_non_append_keys() {
    let pool = std::sync::Arc::new(BufferPool::new(256));
    let tree = CowBeTree::with_config(&pool, tiny_config());

    for key in [b"a", b"c", b"e", b"g", b"i"] {
        tree.insert_new_message(CowBeTreeMessage::put_version(*key, 10, *key))
            .unwrap();
    }
    assert_eq!(
        tree.debug_snapshot().root_kind,
        Some(PageKindDebug::Internal),
        "test setup should create an internal root before the non-append insert"
    );
    let before = tree_metrics(&tree);

    tree.insert_new_message(CowBeTreeMessage::put_version(b"b", 20, b"buffered"))
        .unwrap();

    let after = tree_metrics(&tree);
    assert_eq!(
        after.in_place_page_rewrites, before.in_place_page_rewrites,
        "insert-new writes should not probe and rewrite leaves for absent non-append keys"
    );
    assert_eq!(
        after.root_buffer_appends - before.root_buffer_appends,
        1,
        "height-one insert-new writes that cannot leaf-append should buffer at the root"
    );
    assert_eq!(
        tree.lookup(b"b").as_deref(),
        Some(&b"buffered"[..]),
        "lookup should merge the insert-new buffered message"
    );
}

#[test]
fn concurrent_page_local_leaf_rewrites_preserve_versions_without_root_buffering() {
    let pool = Arc::new(BufferPool::new(512));
    let tree = Arc::new(CowBeTree::with_config(&pool, tiny_config()));

    for i in 0u16..5 {
        tree.put(&be_key(i), 10, &be_key(i)).unwrap();
    }
    assert_eq!(
        tree.debug_snapshot().root_kind,
        Some(PageKindDebug::Internal),
        "test setup should create an internal root before concurrent updates"
    );
    let before = tree_metrics(&tree);
    let barrier = Arc::new(Barrier::new(4));
    let handles = [0u16, 1, 3, 4]
        .into_iter()
        .enumerate()
        .map(|(worker, key)| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for iter in 0u64..25 {
                    let commit_ts = 100 + worker as u64 * 100 + iter;
                    tree.put(&be_key(key), commit_ts, &[worker as u8, iter as u8])
                        .unwrap();
                }
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle.join().unwrap();
    }

    let after = tree_metrics(&tree);
    assert_eq!(
        after.root_buffer_appends, before.root_buffer_appends,
        "concurrent existing-key updates should not serialize through the root buffer"
    );
    assert_eq!(
        after.raw_buffer_appends, before.raw_buffer_appends,
        "concurrent leaf-local updates should not create path-buffer messages"
    );
    for (worker, key) in [0u16, 1, 3, 4].into_iter().enumerate() {
        assert_eq!(
            tree.lookup(&be_key(key)),
            Some(vec![worker as u8, 24]),
            "latest concurrent update should be visible for key {key}"
        );
        assert_eq!(
            tree.lookup_visible(&be_key(key), 10).as_deref(),
            Some(&be_key(key)[..]),
            "concurrent page-local rewrites should retain the original version for key {key}"
        );
    }
}

#[test]
fn page_local_child_internal_buffer_bypasses_root_buffer() {
    let pool = std::sync::Arc::new(BufferPool::new(2048));
    let config = CowBeTreeConfig {
        flush_threshold_messages: 8,
        flush_threshold_bytes: 512,
        max_leaf_entries: 3,
        max_internal_children: 3,
        merge_leaf_entries: 16,
        merge_internal_children: 8,
    };
    let tree = CowBeTree::with_config(&pool, config);

    for i in 0u16..96 {
        tree.put(&be_key(i), i as u64, &be_key(i)).unwrap();
    }
    tree.flush_all().unwrap();
    assert!(
        tree.debug_snapshot().height >= 2,
        "test setup should create an internal child below the root"
    );
    let before = tree_metrics(&tree);

    tree.put(&be_key(42), 100, b"buffered").unwrap();

    let after = tree_metrics(&tree);
    assert_eq!(
        after.root_buffer_appends, before.root_buffer_appends,
        "page-local routing should not append to the root buffer when a child internal page can buffer"
    );
    assert_eq!(
        after.raw_buffer_appends - before.raw_buffer_appends,
        1,
        "the routed child internal page should receive the buffered update"
    );
    assert_eq!(
        after.leaf_rebuild_bytes, before.leaf_rebuild_bytes,
        "buffering in the child internal page should avoid a leaf rebuild"
    );
    assert_eq!(
        tree.lookup(&be_key(42)).as_deref(),
        Some(&b"buffered"[..]),
        "lookup should merge the child internal buffer update"
    );
}

#[test]
fn ordered_leaf_writes_use_raw_append_before_split() {
    let pool = std::sync::Arc::new(BufferPool::new(256));
    let tree = CowBeTree::with_config(&pool, tiny_config());

    for i in 0u16..3 {
        tree.put(&be_key(i), 1, &be_key(i)).unwrap();
    }

    let stats = tree_metrics(&tree);
    assert!(
        stats.raw_leaf_appends >= 3,
        "ordered leaf writes should append entries without full leaf rewrites"
    );
    assert_eq!(
        tree.debug_snapshot().root_kind,
        Some(PageKindDebug::Leaf),
        "test setup should remain below the leaf split threshold"
    );
    for i in 0u16..3 {
        assert_eq!(
            tree.lookup(&be_key(i)).as_deref(),
            Some(&be_key(i)[..]),
            "raw leaf append should preserve lookup for key {i}"
        );
    }
}

#[test]
fn ordinary_writes_keep_root_page_until_explicit_fork_divergence() {
    let pool = std::sync::Arc::new(BufferPool::new(256));
    let tree = CowBeTree::with_config(&pool, tiny_config());
    let root_pid = tree.root_page_id();

    for i in 0u16..24 {
        tree.put(&be_key(i), 1, &be_key(i)).unwrap();
    }

    let stats = tree_metrics(&tree);
    assert_eq!(
        tree.root_page_id(),
        root_pid,
        "normal writes and root splits should rewrite the root page in place"
    );
    assert_eq!(
        stats.root_replacements, 0,
        "normal writes must not publish a replacement root"
    );
    assert!(
        stats.in_place_page_rewrites > 0,
        "normal writes should be accounted as page-local rewrites"
    );
}

#[test]
fn explicit_fork_preserves_snapshot_until_each_tree_mutates() {
    let pool = std::sync::Arc::new(BufferPool::new(512));
    let config = CowBeTreeConfig {
        flush_threshold_messages: 3,
        ..tiny_config()
    };
    let tree = CowBeTree::with_config(&pool, config);

    for i in 0u16..8 {
        tree.put(&be_key(i), 10, &be_key(i)).unwrap();
    }

    let fork = tree.fork().unwrap();
    tree.put(&be_key(1), 20, b"primary").unwrap();
    for i in 30u16..35 {
        tree.put(&be_key(i), 20, &be_key(i)).unwrap();
    }

    assert_eq!(
        fork.lookup(&be_key(1)).as_deref(),
        Some(&be_key(1)[..]),
        "fork should keep the pre-divergence version of an updated key"
    );
    assert_eq!(
        fork.lookup(&be_key(30)),
        None,
        "fork should not see keys inserted after it was created"
    );
    assert_eq!(
        tree.lookup(&be_key(1)).as_deref(),
        Some(&b"primary"[..]),
        "mutating the original should install its own visible version"
    );

    fork.put(&be_key(1), 30, b"fork").unwrap();
    assert_eq!(
        fork.lookup(&be_key(1)).as_deref(),
        Some(&b"fork"[..]),
        "fork should be independently mutable after divergence"
    );
    assert_eq!(
        tree.lookup(&be_key(1)).as_deref(),
        Some(&b"primary"[..]),
        "fork mutation should not alter the original tree"
    );
    assert!(
        tree_metrics(&tree).fork_page_copies > 0,
        "first post-fork write should copy at least the shared root"
    );
}

#[test]
fn flush_threshold_flushes_largest_child_batch_and_keeps_other_messages_buffered() {
    let pool = std::sync::Arc::new(BufferPool::new(512));
    let config = CowBeTreeConfig {
        flush_threshold_messages: 4,
        ..tiny_config()
    };
    let tree = CowBeTree::with_config(&pool, config);

    for i in 0u16..5 {
        tree.put(&be_key(i), 1, &be_key(i)).unwrap();
    }
    let _fork = tree.fork().unwrap();
    let before = tree_metrics(&tree);
    for i in [30u16, 1, 31, 32] {
        tree.put(&be_key(i), 2, &be_key(i)).unwrap();
    }

    let stats = tree_metrics(&tree);
    let debug = tree.debug_snapshot();
    assert!(
        stats.buffer_flushes > before.buffer_flushes,
        "threshold-crossing root buffer should flush"
    );
    assert_eq!(
        stats.messages_flushed - before.messages_flushed,
        3,
        "flush should choose the child with the largest pending batch across append order"
    );
    assert_eq!(
        debug.buffered_messages, 1,
        "non-selected child messages should remain buffered on the parent"
    );
    for i in [1u16, 30, 31, 32] {
        assert_eq!(
            tree.lookup(&be_key(i)).as_deref(),
            Some(&be_key(i)[..]),
            "largest-child flush should preserve message visibility for key {i}"
        );
    }
}

#[test]
fn threshold_flush_batches_ordered_leaf_appends_by_child() {
    let pool = std::sync::Arc::new(BufferPool::new(512));
    let config = CowBeTreeConfig {
        flush_threshold_messages: 3,
        flush_threshold_bytes: 512,
        max_leaf_entries: 8,
        max_internal_children: 8,
        merge_leaf_entries: 16,
        merge_internal_children: 8,
    };
    let tree = CowBeTree::with_config(&pool, config);

    for i in 0u16..9 {
        tree.put(&be_key(i), 1, &be_key(i)).unwrap();
    }
    assert_eq!(
        tree.debug_snapshot().root_kind,
        Some(PageKindDebug::Internal),
        "test setup should create an internal root before buffering"
    );
    let _fork = tree.fork().unwrap();
    let before = tree_metrics(&tree);

    for i in 9u16..12 {
        tree.put(&be_key(i), 2, &be_key(i)).unwrap();
    }

    let after = tree_metrics(&tree);
    assert!(
        after.buffer_flushes > before.buffer_flushes,
        "third buffered write should force a root flush"
    );
    assert_eq!(
        after.raw_leaf_appends - before.raw_leaf_appends,
        3,
        "flush should append all same-child ordered messages to the leaf"
    );
    assert_eq!(
        after.raw_leaf_append_batches - before.raw_leaf_append_batches,
        1,
        "flush should append the same-child messages in one leaf batch"
    );
    assert_eq!(
        after.leaf_batch_rewrites, before.leaf_batch_rewrites,
        "batched leaf append should avoid a logical leaf batch rewrite"
    );
    for i in 9u16..12 {
        assert_eq!(
            tree.lookup(&be_key(i)).as_deref(),
            Some(&be_key(i)[..]),
            "batched flush should preserve lookup for key {i}"
        );
    }
}

#[test]
fn same_path_flush_batch_rewrites_target_leaf_without_buffering_each_level() {
    let pool = std::sync::Arc::new(BufferPool::new(2048));
    let config = CowBeTreeConfig {
        flush_threshold_messages: 4,
        flush_threshold_bytes: 512,
        max_leaf_entries: 3,
        max_internal_children: 3,
        merge_leaf_entries: 16,
        merge_internal_children: 8,
    };
    let tree = CowBeTree::with_config(&pool, config);

    for i in 0u16..96 {
        tree.put(&be_key(i), i as u64, &be_key(i)).unwrap();
    }
    tree.flush_all().unwrap();
    assert!(
        tree.debug_snapshot().height >= 2,
        "test setup should create an internal child below the root"
    );

    let _fork = tree.fork().unwrap();
    let before = tree_metrics(&tree);
    for commit_ts in 100u64..104 {
        tree.put(&be_key(42), commit_ts, &[commit_ts as u8])
            .unwrap();
    }

    let after = tree_metrics(&tree);
    assert!(
        after.buffer_flushes > before.buffer_flushes,
        "threshold-crossing hot-key updates should flush the root buffer"
    );
    assert_eq!(
        after.direct_leaf_flushes - before.direct_leaf_flushes,
        1,
        "same-path batch should be applied at the target leaf in one flush mode"
    );
    assert_eq!(
        after.direct_leaf_flush_messages - before.direct_leaf_flush_messages,
        4,
        "direct leaf flush should account for the whole selected batch"
    );
    assert_eq!(
        after.leaf_batch_rewrites - before.leaf_batch_rewrites,
        1,
        "leaf should be rebuilt once for the hot-key batch"
    );
    assert_eq!(
        after.page_image_rewrite_bytes - before.page_image_rewrite_bytes,
        (after.in_place_page_rewrites - before.in_place_page_rewrites) * PAGE_SIZE as u64,
        "page-image rewrite bytes should count every full rewritten page image"
    );
    assert_eq!(
        after.page_image_rewrite_bytes - before.page_image_rewrite_bytes,
        (after.leaf_page_image_rewrite_bytes - before.leaf_page_image_rewrite_bytes)
            + (after.internal_page_image_rewrite_bytes - before.internal_page_image_rewrite_bytes),
        "total page-image rewrite bytes should be the leaf/internal sum"
    );
    assert!(
        after.internal_page_image_rewrite_bytes > before.internal_page_image_rewrite_bytes,
        "same-path direct flush should expose parent-path internal rewrite traffic"
    );
    assert_eq!(
        after.raw_buffer_appends - before.raw_buffer_appends,
        3,
        "root buffer should append hot-key messages until the flush threshold is reached"
    );
    assert_eq!(
        after.internal_buffer_sorted_rewrites - before.internal_buffer_sorted_rewrites,
        0,
        "append-ordered internal buffers should not require sorted buffer rewrites"
    );
    assert_eq!(
        tree.lookup(&be_key(42)),
        Some(vec![103]),
        "latest hot-key version should be visible after direct leaf flush"
    );
    assert_eq!(
        tree.lookup_visible(&be_key(42), 99).as_deref(),
        Some(&be_key(42)[..]),
        "direct leaf flush must preserve older MVCC snapshots"
    );
}

#[test]
fn path_buffer_versions_participate_in_mvcc_visibility() {
    let pool = std::sync::Arc::new(BufferPool::new(256));
    let tree = CowBeTree::with_config(&pool, tiny_config());

    for i in 0u16..5 {
        tree.put(&be_key(i), 10, &be_key(i)).unwrap();
    }
    tree.put(&be_key(1), 20, b"new").unwrap();
    tree.remove_at(&be_key(2), 30).unwrap();

    assert_eq!(
        tree.lookup_visible(&be_key(1), 19).as_deref(),
        Some(&be_key(1)[..]),
        "old leaf version remains visible before the buffered update"
    );
    assert_eq!(
        tree.lookup_visible(&be_key(1), 20).as_deref(),
        Some(&b"new"[..]),
        "buffered update should win at its commit timestamp"
    );
    assert_eq!(
        tree.lookup_visible(&be_key(2), 29).as_deref(),
        Some(&be_key(2)[..]),
        "delete tombstone should not hide earlier snapshots"
    );
    assert_eq!(
        tree.lookup_visible(&be_key(2), 30),
        None,
        "buffered delete should hide the row at its commit timestamp"
    );
}

#[test]
fn prune_versions_keeps_watermark_floor_and_latest_versions() {
    let pool = std::sync::Arc::new(BufferPool::new(256));
    let tree = CowBeTree::with_config(&pool, tiny_config());

    tree.put(b"hot", 10, b"v10").unwrap();
    tree.put(b"hot", 20, b"v20").unwrap();
    tree.put(b"hot", 30, b"v30").unwrap();

    let first = tree.prune_versions(25).unwrap();
    assert_eq!(
        first.versions_pruned, 1,
        "GC should remove versions older than the retained watermark floor"
    );
    assert_eq!(
        tree.lookup_visible(b"hot", 10),
        None,
        "snapshots older than the GC watermark are no longer retained"
    );
    assert_eq!(
        tree.lookup_visible(b"hot", 25).as_deref(),
        Some(&b"v20"[..]),
        "the newest version at or below the watermark must be retained"
    );
    assert_eq!(
        tree.lookup_visible(b"hot", 30).as_deref(),
        Some(&b"v30"[..]),
        "versions newer than the watermark must be retained"
    );

    let second = tree.prune_versions(40).unwrap();
    assert_eq!(
        second.versions_pruned, 1,
        "a later watermark should prune the previously retained floor"
    );
    assert_eq!(
        tree.lookup_visible(b"hot", 25),
        None,
        "the old floor should be gone once the watermark advances"
    );
    assert_eq!(
        tree.lookup_visible(b"hot", 40).as_deref(),
        Some(&b"v30"[..]),
        "GC must keep the latest committed version for current readers"
    );

    let stats = tree_metrics(&tree);
    assert_eq!(stats.gc_runs, 2, "GC run counter should track both passes");
    assert_eq!(
        stats.gc_versions_pruned, 2,
        "GC telemetry should account for pruned versions"
    );
    assert!(
        stats.gc_leaf_pages_rewritten >= 2,
        "each pruning pass should rewrite the affected leaf"
    );
}

#[test]
fn prune_versions_flushes_path_buffers_before_reclaiming() {
    let pool = std::sync::Arc::new(BufferPool::new(512));
    let tree = CowBeTree::with_config(&pool, tiny_config());

    for i in 0u16..8 {
        tree.put(&be_key(i), 10, &be_key(i)).unwrap();
    }
    tree.flush_all().unwrap();
    assert_eq!(
        tree.debug_snapshot().root_kind,
        Some(PageKindDebug::Internal),
        "test setup should have an internal root with child leaves"
    );

    let _fork = tree.fork().unwrap();
    let buffered_key = be_key(100);
    tree.put(&buffered_key, 10, b"old").unwrap();
    tree.put(&buffered_key, 20, b"new").unwrap();
    assert!(
        tree.debug_snapshot().buffered_messages >= 2,
        "test setup should retain updates in an internal path buffer"
    );
    assert_eq!(
        tree.lookup_visible(&buffered_key, 10).as_deref(),
        Some(&b"old"[..]),
        "path-buffered versions should be visible before GC"
    );

    let result = tree.prune_versions(25).unwrap();
    assert_eq!(
        result.versions_pruned, 1,
        "GC should flush path buffers and prune obsolete buffered history"
    );
    assert_eq!(
        tree.debug_snapshot().buffered_messages,
        0,
        "GC should leave no path buffers behind after its full flush"
    );
    assert_eq!(
        tree.lookup_visible(&buffered_key, 10),
        None,
        "obsolete buffered history should be gone after GC"
    );
    assert_eq!(
        tree.lookup_visible(&buffered_key, 25).as_deref(),
        Some(&b"new"[..]),
        "the retained floor version should remain visible at the watermark"
    );
}

#[test]
fn prune_versions_incremental_advances_leaf_cursor_with_budget() {
    let pool = std::sync::Arc::new(BufferPool::new(512));
    let tree = CowBeTree::with_config(&pool, tiny_config());

    for i in 0u16..16 {
        tree.put(&be_key(i), 10, b"old").unwrap();
    }
    for i in 0u16..16 {
        tree.put(&be_key(i), 20, b"new").unwrap();
    }
    tree.flush_all().unwrap();
    assert!(
        tree.debug_snapshot().leaf_pages > 1,
        "test setup should spread rows across multiple leaves"
    );

    let mut cursor = CowBeTreeGcCursor::default();
    let first = tree.prune_versions_incremental(25, &mut cursor, 1).unwrap();
    assert_eq!(
        first.leaf_pages_visited, 1,
        "one-leaf budget should visit exactly one leaf"
    );
    assert!(
        first.budget_exhausted,
        "first pass should stop because the leaf budget was consumed"
    );
    assert!(
        cursor.next_key().is_some(),
        "incremental GC should publish a resume key after a partial pass"
    );
    assert!(
        first.versions_pruned > 0,
        "visited leaf should prune obsolete row versions"
    );

    let mut total_pruned = first.versions_pruned;
    let mut passes = 1usize;
    while cursor.next_key().is_some() {
        let pass = tree.prune_versions_incremental(25, &mut cursor, 1).unwrap();
        assert!(
            pass.leaf_pages_visited <= 1,
            "incremental GC must honour the one-leaf budget"
        );
        total_pruned += pass.versions_pruned;
        passes += 1;
        assert!(passes < 16, "cursor should wrap after walking the leaf set");
    }

    assert!(passes > 1, "GC work should be split across multiple passes");
    assert_eq!(
        total_pruned, 16,
        "one obsolete version per key should be reclaimed across the full cycle"
    );
    assert_eq!(
        tree_metrics(&tree).gc_cursor_wraps,
        1,
        "telemetry should count the completed cursor cycle"
    );
    for i in 0u16..16 {
        assert_eq!(
            tree.lookup_visible(&be_key(i), 10),
            None,
            "obsolete snapshot should be gone for key {i}"
        );
        assert_eq!(
            tree.lookup_visible(&be_key(i), 25).as_deref(),
            Some(&b"new"[..]),
            "latest retained version should remain visible for key {i}"
        );
    }
}

#[test]
fn prune_versions_incremental_drains_path_buffers_before_reclaiming() {
    let pool = std::sync::Arc::new(BufferPool::new(512));
    let tree = CowBeTree::with_config(&pool, tiny_config());

    for i in 0u16..8 {
        tree.put(&be_key(i), 10, &be_key(i)).unwrap();
    }
    tree.flush_all().unwrap();
    assert_eq!(
        tree.debug_snapshot().root_kind,
        Some(PageKindDebug::Internal),
        "test setup should have an internal root"
    );

    let _fork = tree.fork().unwrap();
    let buffered_key = be_key(100);
    tree.put(&buffered_key, 10, b"old").unwrap();
    tree.put(&buffered_key, 20, b"new").unwrap();
    assert!(
        tree.debug_snapshot().buffered_messages >= 2,
        "test setup should retain versions in a path buffer"
    );

    let mut cursor = CowBeTreeGcCursor::default();
    let mut total_pruned = 0usize;
    for _ in 0..16 {
        let pass = tree.prune_versions_incremental(25, &mut cursor, 1).unwrap();
        total_pruned += pass.versions_pruned;
        if total_pruned > 0 && cursor.next_key().is_none() {
            break;
        }
    }

    assert_eq!(
        total_pruned, 1,
        "incremental GC should flush the buffered hot key and reclaim obsolete history"
    );
    assert_eq!(
        tree.debug_snapshot().buffered_messages,
        0,
        "incremental GC should drain the relevant path buffer"
    );
    assert_eq!(
        tree.lookup_visible(&buffered_key, 10),
        None,
        "old buffered version should be reclaimed after the watermark"
    );
    assert_eq!(
        tree.lookup_visible(&buffered_key, 25).as_deref(),
        Some(&b"new"[..]),
        "retained watermark floor should remain visible"
    );
}

#[test]
fn leaf_and_internal_splits_publish_separators_and_keep_scan_order() {
    let pool = std::sync::Arc::new(BufferPool::new(1024));
    let config = CowBeTreeConfig {
        flush_threshold_messages: 3,
        max_leaf_entries: 3,
        max_internal_children: 3,
        merge_leaf_entries: 16,
        merge_internal_children: 8,
        ..CowBeTreeConfig::default()
    };
    let tree = CowBeTree::with_config(&pool, config);

    for i in 0u16..80 {
        tree.put(&be_key(i), i as u64, &be_key(i)).unwrap();
    }
    tree.flush_all().unwrap();

    let stats = tree_metrics(&tree);
    let debug = tree.debug_snapshot();
    assert!(stats.leaf_splits > 0, "leaf split counter should move");
    assert!(
        stats.internal_splits > 0,
        "internal split counter should move with max_internal_children=3"
    );
    assert!(
        debug.height >= 2,
        "internal splits should create a multi-level tree"
    );

    let mut scanned = Vec::new();
    tree.scan_range(
        Bound::Included(&be_key(10)[..]),
        Bound::Excluded(&be_key(21)[..]),
        |key, value| scanned.push((key.to_vec(), value.to_vec())),
    );
    let expected = (10u16..21)
        .map(|i| (be_key(i).to_vec(), be_key(i).to_vec()))
        .collect::<Vec<_>>();
    assert_eq!(
        scanned, expected,
        "range scan must remain sorted across splits"
    );
}

#[test]
fn compact_merges_leaf_siblings_and_collapses_single_child_root() {
    let pool = std::sync::Arc::new(BufferPool::new(512));
    let tree = CowBeTree::with_config(&pool, tiny_config());

    for i in 0u16..8 {
        tree.put(&be_key(i), 1, &be_key(i)).unwrap();
    }
    tree.flush_all().unwrap();
    let before = tree.debug_snapshot();
    assert!(
        before.leaf_pages > 1,
        "test setup should create sibling leaves before compaction"
    );

    tree.compact().unwrap();
    let after = tree.debug_snapshot();
    let stats = tree_metrics(&tree);
    assert!(
        stats.leaf_merges > 0,
        "compaction should execute leaf merge logic"
    );
    assert!(
        stats.root_collapses > 0,
        "merged root with one child should collapse to that child"
    );
    assert_eq!(
        after.leaf_pages, 1,
        "all sparse leaves should fit into one merged leaf"
    );
    assert_eq!(after.height, 0, "root collapse should leave a leaf root");
}

#[test]
fn deterministic_model_matches_btreemap_across_flushes_and_compaction() {
    let pool = std::sync::Arc::new(BufferPool::new(1024));
    let tree = CowBeTree::with_config(&pool, tiny_config());
    let mut model: VersionModel = BTreeMap::new();

    for ts in 1u64..160 {
        let key = vec![(ts % 23) as u8];
        if ts % 7 == 0 {
            tree.insert_message(CowBeTreeMessage::delete(key.clone(), ts))
                .unwrap();
            model.entry(key).or_default().insert(0, (ts, None));
        } else {
            let value = vec![ts as u8, (ts * 3) as u8];
            tree.insert_message(CowBeTreeMessage::put(key.clone(), value.clone(), ts))
                .unwrap();
            model.entry(key).or_default().insert(0, (ts, Some(value)));
        }

        if ts % 31 == 0 {
            tree.flush_all().unwrap();
        }
        if ts % 53 == 0 {
            tree.compact().unwrap();
        }
    }

    for read_ts in [1, 13, 37, 79, 159] {
        for key in 0u8..23 {
            let key = vec![key];
            assert_eq!(
                tree.lookup_visible(&key, read_ts),
                visible_model(&model, &key, read_ts),
                "model mismatch for key {key:?} at read timestamp {read_ts}"
            );
        }
    }
}

#[test]
fn evicted_pages_remain_reachable_after_buffer_flushes_and_splits() {
    let pool = std::sync::Arc::new(BufferPool::new(24));
    let tree = CowBeTree::with_config(&pool, tiny_config());

    for i in 0u16..96 {
        tree.put(&be_key(i), 1, &be_key(i)).unwrap();
    }
    tree.flush_all().unwrap();

    for _ in 0..256 {
        let _ = pool.try_evict_one();
    }

    for i in [0u16, 7, 31, 64, 95] {
        assert_eq!(
            tree.lookup(&be_key(i)).as_deref(),
            Some(&be_key(i)[..]),
            "lookup should survive buffer-pool eviction for key {i}"
        );
    }
}
