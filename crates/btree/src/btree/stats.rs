#[cfg(feature = "metrics")]
use fast_telemetry::{Counter, DeriveLabel, ExportMetrics, LabeledCounter};

#[cfg(not(feature = "metrics"))]
use crate::metrics_stub::{Counter, LabeledCounter};

#[cfg_attr(feature = "metrics", derive(DeriveLabel))]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "metrics", label_name = "event")]
#[cfg_attr(not(feature = "metrics"), allow(dead_code))]
pub(crate) enum BTreeEvent {
    InsertRestarts,
    LeafDescentRestarts,
    LeafUpgradeRestarts,
    SplitPathRestarts,
    ParentPublishRestarts,
    LeafSplits,
    InnerSplits,
    ParentFallbacks,
    LeftChases,
    ResolveCold,
    EvictionUnswizzleCalls,
    EvictionUnswizzleRestarts,
    EvictionUnswizzleParentHits,
    EvictionUnswizzleUpgradeFailures,
}

#[cfg_attr(feature = "metrics", derive(ExportMetrics))]
#[cfg_attr(feature = "metrics", metric_prefix = "btree")]
pub(crate) struct BTreeStats {
    #[cfg_attr(feature = "metrics", help = "B-tree events")]
    events: LabeledCounter<BTreeEvent>,
    #[cfg_attr(feature = "metrics", help = "B-tree eviction unswizzle nodes visited")]
    pub(crate) eviction_unswizzle_nodes_visited: Counter,
}

impl BTreeStats {
    pub(crate) fn inc(&self, event: BTreeEvent) {
        self.events.inc(event);
    }

    pub(crate) fn new(shards: usize) -> Self {
        Self {
            events: LabeledCounter::new(shards),
            eviction_unswizzle_nodes_visited: Counter::new(shards),
        }
    }
}

#[cfg(not(feature = "metrics"))]
impl BTreeStats {
    pub(crate) fn visit_metrics<V: ?Sized>(&self, _visitor: &mut V) {}
}
