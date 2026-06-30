#[cfg(feature = "metrics")]
use fast_telemetry::{Counter, DeriveLabel, ExportMetrics, LabeledCounter};

#[cfg(not(feature = "metrics"))]
use crate::metrics_stub::{Counter, LabeledCounter};

#[cfg_attr(feature = "metrics", derive(DeriveLabel))]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "metrics", label_name = "cow_betree_event")]
#[cfg_attr(not(feature = "metrics"), allow(dead_code))]
pub(crate) enum CowBeTreeEvent {
    RootBufferAppends,
    BufferFlushes,
    MessagesFlushed,
    CowPagesAllocated,
    Forks,
    ForkPageCopies,
    InPlacePageRewrites,
    InternalBufferSortedRewrites,
    LeafBatchRewrites,
    DirectLeafFlushes,
    DirectLeafFlushMessages,
    RawBufferAppends,
    RawLeafAppends,
    RawLeafAppendBatches,
    LeafSplits,
    InternalSplits,
    LeafMerges,
    InternalMerges,
    RootReplacements,
    RootCollapses,
    Restarts,
    ColdResolves,
    PageFaults,
    PathBufferMerges,
    SecondaryVerifications,
    GcRuns,
    GcVersionsPruned,
    GcLeafPagesVisited,
    GcLeafPagesRewritten,
    GcVersionBytesPruned,
    GcCursorWraps,
    GcBudgetExhausted,
}

#[cfg_attr(feature = "metrics", derive(ExportMetrics))]
#[cfg_attr(feature = "metrics", metric_prefix = "cow_betree")]
#[cfg_attr(not(feature = "metrics"), allow(dead_code))]
pub(crate) struct CowBeTreeStats {
    #[cfg_attr(feature = "metrics", help = "B-e tree events")]
    events: LabeledCounter<CowBeTreeEvent>,
    #[cfg_attr(feature = "metrics", help = "Bytes rebuilt in B-e tree leaves")]
    leaf_rebuild_bytes: Counter,
    #[cfg_attr(feature = "metrics", help = "Bytes rebuilt in B-e tree internals")]
    internal_rebuild_bytes: Counter,
    #[cfg_attr(
        feature = "metrics",
        help = "Bytes rewritten in B-e tree leaf page images"
    )]
    leaf_page_image_rewrite_bytes: Counter,
    #[cfg_attr(
        feature = "metrics",
        help = "Bytes rewritten in B-e tree internal page images"
    )]
    internal_page_image_rewrite_bytes: Counter,
    #[cfg_attr(
        feature = "metrics",
        help = "Bytes emitted to WAL by B-e tree row storage"
    )]
    wal_bytes: Counter,
}

impl CowBeTreeStats {
    pub(crate) fn new(shards: usize) -> Self {
        Self {
            events: LabeledCounter::new(shards),
            leaf_rebuild_bytes: Counter::new(shards),
            internal_rebuild_bytes: Counter::new(shards),
            leaf_page_image_rewrite_bytes: Counter::new(shards),
            internal_page_image_rewrite_bytes: Counter::new(shards),
            wal_bytes: Counter::new(shards),
        }
    }

    pub(crate) fn inc(&self, event: CowBeTreeEvent) {
        self.events.inc(event);
    }

    pub(crate) fn add(&self, event: CowBeTreeEvent, value: usize) {
        self.events.add(event, value as isize);
    }

    pub(crate) fn add_leaf_bytes(&self, value: usize) {
        self.leaf_rebuild_bytes.add(value as isize);
    }

    pub(crate) fn add_internal_bytes(&self, value: usize) {
        self.internal_rebuild_bytes.add(value as isize);
    }

    pub(crate) fn add_leaf_page_image_rewrite_bytes(&self, value: usize) {
        self.leaf_page_image_rewrite_bytes.add(value as isize);
    }

    pub(crate) fn add_internal_page_image_rewrite_bytes(&self, value: usize) {
        self.internal_page_image_rewrite_bytes.add(value as isize);
    }
}

#[cfg(not(feature = "metrics"))]
impl CowBeTreeStats {
    pub(crate) fn visit_metrics<V: ?Sized>(&self, _visitor: &mut V) {}
}
