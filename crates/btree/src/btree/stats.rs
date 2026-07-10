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

/// Snapshot of traversal and eviction-routing activity since tree creation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BTreeDiagnosticStats {
    pub insert_restarts: u64,
    pub leaf_descent_restarts: u64,
    pub leaf_upgrade_restarts: u64,
    pub split_path_restarts: u64,
    pub parent_publish_restarts: u64,
    pub parent_fallbacks: u64,
    pub resolve_cold: u64,
    pub eviction_unswizzle_calls: u64,
    pub eviction_unswizzle_restarts: u64,
    pub eviction_unswizzle_parent_hits: u64,
    pub eviction_unswizzle_upgrade_failures: u64,
    pub eviction_unswizzle_nodes_visited: u64,
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

    pub(crate) fn diagnostic_stats(&self) -> BTreeDiagnosticStats {
        BTreeDiagnosticStats {
            insert_restarts: self.events.get(BTreeEvent::InsertRestarts) as u64,
            leaf_descent_restarts: self.events.get(BTreeEvent::LeafDescentRestarts) as u64,
            leaf_upgrade_restarts: self.events.get(BTreeEvent::LeafUpgradeRestarts) as u64,
            split_path_restarts: self.events.get(BTreeEvent::SplitPathRestarts) as u64,
            parent_publish_restarts: self.events.get(BTreeEvent::ParentPublishRestarts) as u64,
            parent_fallbacks: self.events.get(BTreeEvent::ParentFallbacks) as u64,
            resolve_cold: self.events.get(BTreeEvent::ResolveCold) as u64,
            eviction_unswizzle_calls: self.events.get(BTreeEvent::EvictionUnswizzleCalls) as u64,
            eviction_unswizzle_restarts: self.events.get(BTreeEvent::EvictionUnswizzleRestarts)
                as u64,
            eviction_unswizzle_parent_hits: self.events.get(BTreeEvent::EvictionUnswizzleParentHits)
                as u64,
            eviction_unswizzle_upgrade_failures: self
                .events
                .get(BTreeEvent::EvictionUnswizzleUpgradeFailures)
                as u64,
            eviction_unswizzle_nodes_visited: 0,
        }
    }
}

#[cfg(not(feature = "metrics"))]
impl BTreeStats {
    pub(crate) fn visit_metrics<V: ?Sized>(&self, _visitor: &mut V) {}
}
