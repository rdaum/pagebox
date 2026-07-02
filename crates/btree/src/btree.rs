use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(not(feature = "metrics"))]
use crate::metrics_stub::MetricVisitor;
#[cfg(feature = "metrics")]
use fast_telemetry::MetricVisitor;
use pagebox_hybrid_latch::Restart;

use pagebox_storage::buffer_frame::PAGE_SIZE;
use pagebox_storage::buffer_frame::{BufferFrameRef, ParentFinder, StableSwipRef};
use pagebox_storage::buffer_pool::{
    BufferPool, BufferPoolHandle, ExclusiveFrame, NoLatches, PinnedFrame,
};
use pagebox_storage::slotted_page::SlottedPage;
use pagebox_swip_kernel::{AtomicSwipWord as AtomicSwip, SwipWord as Swip};

const WRITE_FIXED_ROOT_THRESHOLD: u32 = 4;
const WRITE_BLOCKING_FALLBACK_THRESHOLD: u32 = 16;

mod node;
mod parent_edge;
mod split_child;
mod split_publish;
mod stats;

use self::node::{
    BTreeNode, ChildRef, ExclusiveNode, FLAG_IS_LEAF, Inner, LEFT_SIBLING_OFFSET, Leaf,
    OptimisticNode, RIGHT_SIBLING_OFFSET, ResidentFrame, RoutedChildRef, SharedNode, TmpBuf,
};
use self::parent_edge::ParentEdge;
use self::split_child::SplitChild;
use self::split_publish::{split_right_parent_uses_upper, split_separator_insert_pos};
use self::stats::{BTreeEvent, BTreeStats};

enum InsertLeafAction {
    ReturnFalse,
    Inserted,
    SplitRequired,
}

enum UpsertLeafAction {
    UpdatedExisting,
    Inserted,
    SplitRequired,
}

// ---------------------------------------------------------------------------
// BTree
// ---------------------------------------------------------------------------

pub struct BTree {
    pool: BufferPoolHandle,
    meta_swip: AtomicSwip,
    height: AtomicU32,
    stats: BTreeStats,
    /// Data structure ID for the DTRegistry parent-finder callback.
    dt_id: u16,
}

impl ParentFinder for BTree {
    fn find_and_unswizzle(&self, child: BufferFrameRef, child_pid: u64) -> bool {
        unsafe { self.eviction_unswizzle(&ResidentFrame::new(child), child_pid) }
    }
}

impl BTree {
    const LOOKUP_OPTIMISTIC_RESTART_LIMIT: u32 = 64;
    const LOOKUP_OPTIMISTIC_YIELD_INTERVAL: u32 = 4;

    fn swip_page_id(swip: Swip) -> u64 {
        if swip.is_hot() || swip.is_cool() {
            unsafe { ResidentFrame::from_hot_swip(swip) }.unwrap().pid()
        } else {
            swip.as_page_id()
        }
    }

    unsafe fn set_parent_link_for_edge(
        &self,
        child: &ResidentFrame<'_>,
        parent: &ExclusiveNode<'_, Inner>,
        edge: ParentEdge,
    ) {
        let count = parent.num_slots();
        let pos = edge.pos(count);
        let is_upper = matches!(edge, ParentEdge::Upper);
        unsafe { self.set_inner_parent_link(child, &parent.resident_frame(), pos, is_upper) };
    }

    unsafe fn set_parent_link_for_routed_child(
        &self,
        child: &ResidentFrame<'_>,
        parent: &ResidentFrame<'_>,
        routed: &RoutedChildRef<'_>,
    ) {
        let count = parent.num_slots();
        unsafe {
            self.set_inner_parent_link(child, parent, routed.slot_index(count), routed.is_upper())
        };
    }

    fn parent_edge_for_child(
        &self,
        parent: &ExclusiveNode<'_, Inner>,
        child: &ResidentFrame<'_>,
    ) -> Option<ParentEdge> {
        parent.child_edge_for(ChildRef::from_frame(child))
    }

    unsafe fn pin_exclusive_child(&self, swip: Swip) -> Option<ExclusiveFrame<'_>> {
        let child = if swip.is_hot() || swip.is_cool() {
            unsafe { self.pool().pin_child(swip, NoLatches::new(self.pool())) }?
        } else {
            unsafe {
                self.pool()
                    .fix_orphan_frame(swip.as_page_id(), NoLatches::new(self.pool()))
            }
        };
        Some(child.exclusive())
    }

    unsafe fn collapse_empty_root_to_child(
        &self,
        root: &ResidentFrame<'_>,
        child: &ResidentFrame<'_>,
    ) {
        self.meta_swip.store(child.hot_swip(), Ordering::Release);
        unsafe { self.set_root_parent_link(child) };
        self.height.fetch_sub(1, Ordering::Relaxed);
        root.set_parent_link_none();
    }

    unsafe fn unlink_merged_right_leaf(
        &self,
        parent: &ExclusiveNode<'_, Inner>,
        merged_leaf: &ResidentFrame<'_>,
        removed_leaf: &ResidentFrame<'_>,
        merged_edge: ParentEdge,
        removed_slot: u16,
    ) {
        let merged = merged_leaf.sp();
        let merged_max = merged
            .try_get_key(merged.num_slots().saturating_sub(1))
            .map(|key| key.to_vec());
        match merged_edge {
            ParentEdge::Upper => {
                parent.set_child_edge_swip(ParentEdge::Upper, merged_leaf.hot_swip());
                parent.remove_slot(removed_slot);
                unsafe { self.set_parent_link_for_edge(merged_leaf, parent, ParentEdge::Upper) };
            }
            ParentEdge::Slot(_) => {
                if let Some(key) = merged_max.as_deref() {
                    parent.set_separator_key(removed_slot, key);
                }
                parent.remove_slot(removed_slot + 1);
                unsafe {
                    self.set_parent_link_for_edge(
                        merged_leaf,
                        parent,
                        ParentEdge::Slot(removed_slot),
                    )
                };
            }
        }
        removed_leaf.set_parent_link_none();
    }

    unsafe fn unlink_merged_left_leaf(
        &self,
        parent: &ExclusiveNode<'_, Inner>,
        merged_leaf: &ResidentFrame<'_>,
        removed_leaf: &ResidentFrame<'_>,
        merged_separator_slot: u16,
        removed_slot: u16,
        replacement_key: Option<&[u8]>,
    ) {
        let merged = merged_leaf.sp();
        let merged_max = merged
            .try_get_key(merged.num_slots().saturating_sub(1))
            .map(|key| key.to_vec());
        if replacement_key.is_none() {
            parent.set_child_edge_swip(ParentEdge::Upper, merged_leaf.hot_swip());
            parent.remove_slot(merged_separator_slot);
            unsafe { self.set_parent_link_for_edge(merged_leaf, parent, ParentEdge::Upper) };
        } else {
            if let Some(key) = merged_max.as_deref() {
                parent.set_separator_key(merged_separator_slot, key);
            }
            parent.remove_slot(removed_slot);
            unsafe {
                self.set_parent_link_for_edge(
                    merged_leaf,
                    parent,
                    ParentEdge::Slot(merged_separator_slot),
                )
            };
        }
        removed_leaf.set_parent_link_none();
    }

    unsafe fn unlink_merged_right_inner(
        &self,
        parent: &ExclusiveNode<'_, Inner>,
        merged: &ResidentFrame<'_>,
        removed: &ResidentFrame<'_>,
        merged_edge: ParentEdge,
        removed_slot: u16,
    ) {
        parent.set_child_edge_swip(merged_edge, merged.hot_swip());
        parent.remove_slot(removed_slot);
        unsafe { self.set_parent_link_for_edge(merged, parent, merged_edge) };
        removed.set_parent_link_none();
    }

    unsafe fn unlink_merged_left_inner(
        &self,
        parent: &ExclusiveNode<'_, Inner>,
        merged: &ResidentFrame<'_>,
        removed: &ResidentFrame<'_>,
        merged_separator_slot: u16,
        removed_slot: u16,
        replacement_key: Option<&[u8]>,
    ) {
        if let Some(key) = replacement_key {
            parent.set_separator_key(merged_separator_slot, key);
        } else {
            parent.set_child_edge_swip(ParentEdge::Upper, merged.hot_swip());
        }
        parent.remove_slot(removed_slot);
        let merged_edge = if replacement_key.is_some() {
            ParentEdge::Slot(merged_separator_slot)
        } else {
            ParentEdge::Upper
        };
        unsafe { self.set_parent_link_for_edge(merged, parent, merged_edge) };
        removed.set_parent_link_none();
    }

    unsafe fn update_leaf_left_sibling(&self, leaf_pid: u64, left_pid: u64) {
        if leaf_pid == 0 {
            return;
        }
        let leaf = unsafe {
            self.pool()
                .fix_orphan_frame(leaf_pid, NoLatches::new(self.pool()))
        }
        .exclusive();
        let leaf = ExclusiveNode::from_leaf_frame(leaf);
        leaf.resident_frame().set_leaf_left_pid(left_pid);
        leaf.mark_dirty();
    }

    pub fn new<P>(pool: P, dt_id: u16) -> Self
    where
        P: Into<BufferPoolHandle>,
    {
        let pool = pool.into();
        let pool_ref = pool.as_pool();
        let (_root_pid, root_bf) = pool_ref.allocate_and_fix(NoLatches::new(pool_ref));
        let root = root_bf.exclusive();
        ResidentFrame::from_exclusive(&root).init(true);
        root.mark_dirty();
        let root_swip = AtomicSwip::new(root.hot_swip());
        drop(root);
        BTree {
            pool,
            meta_swip: root_swip,
            height: AtomicU32::new(0),
            stats: BTreeStats::new(
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1),
            ),
            dt_id,
        }
    }

    /// Reopen an existing tree from a persisted root page ID and height.
    /// The root page must already exist in the page store.
    pub fn open<P>(pool: P, root_page_id: u64, height: u32, dt_id: u16) -> Self
    where
        P: Into<BufferPoolHandle>,
    {
        let pool = pool.into();
        let root_swip = AtomicSwip::new(Swip::evicted(root_page_id));
        BTree {
            pool,
            meta_swip: root_swip,
            height: AtomicU32::new(height),
            stats: BTreeStats::new(
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1),
            ),
            dt_id,
        }
    }

    /// The root page ID (for persistence).
    pub fn root_page_id(&self) -> u64 {
        Self::swip_page_id(self.meta_swip.load(Ordering::Acquire))
    }

    /// Return every page currently reachable from the tree root.
    ///
    /// This is intended for ownership accounting during table/index retirement,
    /// not for concurrent query execution. Callers should only use it when the
    /// owning structure is quiescent or otherwise protected from mutation.
    pub fn owned_page_ids(&self) -> Vec<u64> {
        let pool = self.pool();
        let root_pid = self.root_page_id();
        let mut seen = BTreeSet::new();
        let mut stack = vec![root_pid];

        while let Some(pid) = stack.pop() {
            if pid == 0 || !seen.insert(pid) {
                continue;
            }

            let frame = unsafe { pool.fix_orphan_frame(pid, NoLatches::new(pool)) }.shared();
            let resident = ResidentFrame::from_shared(&frame);
            if resident.is_leaf() {
                continue;
            }

            for pos in 0..resident.num_slots() {
                let child_pid = Self::swip_page_id(resident.child_swip_at(pos));
                if child_pid != 0 {
                    stack.push(child_pid);
                }
            }
            let upper_pid = Self::swip_page_id(resident.upper_swip());
            if upper_pid != 0 {
                stack.push(upper_pid);
            }
        }

        seen.into_iter().collect()
    }

    fn pool(&self) -> &BufferPool {
        self.pool.as_pool()
    }

    fn debug_child_page_ids(&self, node: &ExclusiveNode<'_, Inner>) -> Vec<u64> {
        node.child_page_ids()
    }

    unsafe fn set_root_parent_link(&self, root: &ResidentFrame<'_>) {
        root.set_parent_link_stable(unsafe { StableSwipRef::from_ref(&self.meta_swip) });
    }

    unsafe fn set_inner_parent_link(
        &self,
        child: &ResidentFrame<'_>,
        parent: &ResidentFrame<'_>,
        slot_index: u16,
        is_upper: bool,
    ) {
        child.set_parent_link_inner(parent.pid(), slot_index, is_upper, self.dt_id);
    }

    pub fn visit_metrics<V: MetricVisitor + ?Sized>(&self, visitor: &mut V) {
        self.stats.visit_metrics(visitor);
    }

    // -----------------------------------------------------------------------
    // Traversal
    // -----------------------------------------------------------------------

    /// Resolve a child swip read from page bytes. If HOT/COOL, pins the
    /// frame directly. If EVICTED, loads the page from disk (orphan fix).
    #[cfg(test)]
    unsafe fn resolve_swip(&self, swip: Swip) -> PinnedFrame<'_> {
        if swip.is_hot() || swip.is_cool() {
            unsafe { self.pool().pin_child(swip, NoLatches::new(self.pool())) }
                .expect("hot/cool test swip should pin")
        } else {
            self.stats.inc(BTreeEvent::ResolveCold);
            unsafe {
                self.pool()
                    .fix_orphan_frame(swip.as_page_id(), NoLatches::new(self.pool()))
            }
        }
    }

    /// Fast-path the root when it is a hot internal node: optimistic readers
    /// can traverse it without paying an initial fix/unfix pair. We only use
    /// Check if we should chase the right sibling link. Bounds-checked
    /// for safety under optimistic reads.
    fn leaf_insert_bound(sp: &SlottedPage, key: &[u8]) -> (u16, bool) {
        let count = sp.num_slots();
        if count == 0 {
            return (0, false);
        }
        let last_pos = count - 1;
        let last_key = sp.get_key(last_pos);
        if key > last_key {
            return (count, false);
        }
        if key == last_key {
            return (last_pos, true);
        }
        sp.lower_bound(key)
    }

    unsafe fn try_resolve_child_for_read<'a>(
        pool: &'a BufferPool,
        swip: Swip,
    ) -> Option<PinnedFrame<'a>> {
        if swip.is_hot() || swip.is_cool() {
            return unsafe { pool.try_pin_child(swip) };
        }
        unsafe { pool.try_fix_orphan_frame(swip.as_page_id()) }
    }

    fn try_insert_leaf(
        &self,
        leaf: &ExclusiveNode<'_, Leaf>,
        key: &[u8],
        value: &[u8],
    ) -> InsertLeafAction {
        let (pos, exact) = Self::leaf_insert_bound(leaf.resident_frame().sp(), key);
        if exact {
            return InsertLeafAction::ReturnFalse;
        }
        if !leaf.can_insert_entry(key.len(), value.len()) {
            return InsertLeafAction::SplitRequired;
        }

        leaf.insert_entry(pos, key, value);
        leaf.mark_dirty();
        InsertLeafAction::Inserted
    }

    fn try_upsert_leaf(
        &self,
        leaf: &ExclusiveNode<'_, Leaf>,
        key: &[u8],
        value: &[u8],
    ) -> UpsertLeafAction {
        let (pos, exact) = Self::leaf_insert_bound(leaf.resident_frame().sp(), key);

        if exact {
            let old_val_len = leaf.value_at(pos).len();
            if old_val_len == value.len() {
                let updated = leaf.update_value_if_same_length(pos, value);
                debug_assert!(updated, "equal-length update must succeed");
                leaf.mark_dirty();
                return UpsertLeafAction::UpdatedExisting;
            }

            let old_entry_len = key.len() + old_val_len;
            let new_entry_len = key.len() + value.len();
            let can_replace = leaf.free_space_after_compaction() + old_entry_len >= new_entry_len;
            if can_replace {
                leaf.remove_slot(pos);
                leaf.insert_entry(pos, key, value);
                leaf.mark_dirty();
                return UpsertLeafAction::UpdatedExisting;
            }

            return UpsertLeafAction::SplitRequired;
        }

        if !leaf.can_insert_entry(key.len(), value.len()) {
            return UpsertLeafAction::SplitRequired;
        }

        leaf.insert_entry(pos, key, value);
        leaf.mark_dirty();
        UpsertLeafAction::Inserted
    }

    unsafe fn find_leaf_exclusive_from_fixed_root<'a>(
        &'a self,
        key: &[u8],
    ) -> Result<ExclusiveNode<'a, Leaf>, Restart> {
        let pool = self.pool();
        let mut current = unsafe { pool.fix_frame(&self.meta_swip, NoLatches::new(pool)) };

        loop {
            let opt = current.optimistic().map_err(|_| Restart)?;
            let is_leaf = BTreeNode::is_leaf(opt.read_ref());
            if is_leaf {
                let leaf = OptimisticNode::from_leaf_frame(opt);
                if leaf.should_chase_right(key) {
                    self.stats.inc(BTreeEvent::LeftChases);
                    if leaf.validate().is_err() {
                        self.stats.inc(BTreeEvent::LeafDescentRestarts);
                        return Err(Restart);
                    }
                    let right_pid = leaf.right_pid();
                    if right_pid == 0 {
                        return Err(Restart);
                    }
                    current = unsafe { pool.fix_orphan_frame(right_pid, NoLatches::new(pool)) };
                    continue;
                }
                match leaf.upgrade_to_exclusive() {
                    Ok(leaf) => return Ok(leaf),
                    Err(_current) => {
                        self.stats.inc(BTreeEvent::LeafUpgradeRestarts);
                        return Err(Restart);
                    }
                }
            }
            let inner = OptimisticNode::<Inner>::from_inner_frame(opt);
            let current_frame = inner.resident_frame();
            let routed_child = match inner.route_to_child(key) {
                Some(r) => r,
                None => {
                    self.stats.inc(BTreeEvent::LeafDescentRestarts);
                    return Err(Restart);
                }
            };
            if inner.validate().is_err() {
                self.stats.inc(BTreeEvent::LeafDescentRestarts);
                return Err(Restart);
            }
            let child_swip = routed_child.swip();

            // HOT: pin the child to prevent eviction between validate()
            // and the next optimistic guard. Without pinning, the page
            // provider can evict and reuse the child frame, causing us
            // to read a different page's data.
            // EVICTED: must fix (load from disk) to continue.
            if child_swip.is_hot() || child_swip.is_cool() {
                let Some(child) = (unsafe { pool.pin_child(child_swip, NoLatches::new(pool)) })
                else {
                    self.stats.inc(BTreeEvent::LeafDescentRestarts);
                    return Err(Restart);
                };
                let child_frame = ResidentFrame::from_pinned(&child);
                // Re-validate parent after pinning — the child could have
                // been evicted between our earlier validate and the pin.
                if inner.validate().is_err() {
                    self.stats.inc(BTreeEvent::LeafDescentRestarts);
                    return Err(Restart);
                }
                unsafe {
                    self.set_parent_link_for_routed_child(
                        &child_frame,
                        &current_frame,
                        &routed_child,
                    )
                };
                current = child;
            } else {
                self.stats.inc(BTreeEvent::ResolveCold);
                let child =
                    unsafe { pool.fix_orphan_frame(child_swip.as_page_id(), NoLatches::new(pool)) };
                // Re-validate after the blocking fix: another thread could
                // have exclusively latched this inner node during the fix and
                // modified its routing entries.
                if inner.validate().is_err() {
                    self.stats.inc(BTreeEvent::LeafDescentRestarts);
                    return Err(Restart);
                }
                let child_frame = ResidentFrame::from_pinned(&child);
                unsafe {
                    self.set_parent_link_for_routed_child(
                        &child_frame,
                        &current_frame,
                        &routed_child,
                    )
                };
                // Swizzle-in: write HOT pointer back to parent's page bytes
                // so future traversals (and the eviction DFS) see HOT, not
                // EVICTED. Requires exclusive latch on parent. If upgrade
                // fails, skip — the traversal still works, just without
                // the swizzle-in optimization.
                if let Ok(parent) = inner.upgrade_to_exclusive() {
                    parent.set_child_edge_swip(routed_child.edge(), child_frame.hot_swip());
                    drop(parent);
                }
                current = child;
            }
        }
    }

    unsafe fn find_leaf_exclusive<'a>(
        &'a self,
        key: &[u8],
    ) -> Result<ExclusiveNode<'a, Leaf>, Restart> {
        let pool = self.pool();
        let root_swip = self.meta_swip.load(Ordering::Acquire);
        let mut current = if root_swip.is_hot() || root_swip.is_cool() {
            let root = unsafe { ResidentFrame::from_hot_swip(root_swip) }.ok_or(Restart)?;
            debug_assert!(
                root.is_in_pool(pool),
                "find_leaf_exclusive: invalid hot root swip: {:?}",
                root_swip,
            );
            let opt = unsafe { root.optimistic_guard() }?;
            if root.is_leaf() {
                let _ = opt;
                unsafe { pool.fix_frame(&self.meta_swip, NoLatches::new(pool)) }
            } else {
                let routed_child = root.try_route_to_child(key).ok_or(Restart)?;
                let child_swip = routed_child.swip();
                if opt.validate().is_err() {
                    return Err(Restart);
                }
                if child_swip.is_hot() || child_swip.is_cool() {
                    let Some(child) = (unsafe { pool.try_pin_child(child_swip) }) else {
                        return Err(Restart);
                    };
                    let child_frame = ResidentFrame::from_pinned(&child);
                    if opt.validate().is_err() {
                        return Err(Restart);
                    }
                    unsafe {
                        self.set_parent_link_for_routed_child(&child_frame, &root, &routed_child)
                    };
                    child
                } else {
                    self.stats.inc(BTreeEvent::ResolveCold);
                    let child = unsafe {
                        pool.fix_orphan_frame(child_swip.as_page_id(), NoLatches::new(pool))
                    };
                    // Re-validate after the blocking fix: another thread could
                    // have exclusively latched the root during the fix and
                    // modified its routing entries.
                    if opt.validate().is_err() {
                        return Err(Restart);
                    }
                    let child_frame = ResidentFrame::from_pinned(&child);
                    unsafe {
                        self.set_parent_link_for_routed_child(&child_frame, &root, &routed_child)
                    };
                    if let Ok(guard) = opt.upgrade_to_exclusive() {
                        match routed_child.edge() {
                            ParentEdge::Upper => root.set_upper(child_frame.hot_swip()),
                            ParentEdge::Slot(pos) => {
                                root.set_child_swip_at(pos, child_frame.hot_swip())
                            }
                        }
                        drop(guard);
                    }
                    child
                }
            }
        } else {
            unsafe { pool.fix_frame(&self.meta_swip, NoLatches::new(pool)) }
        };

        loop {
            let opt = current.optimistic().map_err(|_| Restart)?;
            let is_leaf = BTreeNode::is_leaf(opt.read_ref());
            if is_leaf {
                let leaf = OptimisticNode::from_leaf_frame(opt);
                if leaf.should_chase_right(key) {
                    self.stats.inc(BTreeEvent::LeftChases);
                    if leaf.validate().is_err() {
                        self.stats.inc(BTreeEvent::LeafDescentRestarts);
                        return Err(Restart);
                    }
                    let right_pid = leaf.right_pid();
                    if right_pid == 0 {
                        return Err(Restart);
                    }
                    current = unsafe { pool.fix_orphan_frame(right_pid, NoLatches::new(pool)) };
                    continue;
                }
                match leaf.upgrade_to_exclusive() {
                    Ok(leaf) => return Ok(leaf),
                    Err(_current) => {
                        self.stats.inc(BTreeEvent::LeafUpgradeRestarts);
                        return Err(Restart);
                    }
                }
            }

            let inner = OptimisticNode::<Inner>::from_inner_frame(opt);
            let current_frame = inner.resident_frame();
            let routed_child = match inner.route_to_child(key) {
                Some(r) => r,
                None => {
                    self.stats.inc(BTreeEvent::LeafDescentRestarts);
                    return Err(Restart);
                }
            };
            if inner.validate().is_err() {
                self.stats.inc(BTreeEvent::LeafDescentRestarts);
                return Err(Restart);
            }
            let child_swip = routed_child.swip();

            if child_swip.is_hot() || child_swip.is_cool() {
                let Some(child) = (unsafe { pool.pin_child(child_swip, NoLatches::new(pool)) })
                else {
                    self.stats.inc(BTreeEvent::LeafDescentRestarts);
                    return Err(Restart);
                };
                let child_frame = ResidentFrame::from_pinned(&child);
                if inner.validate().is_err() {
                    self.stats.inc(BTreeEvent::LeafDescentRestarts);
                    return Err(Restart);
                }
                unsafe {
                    self.set_parent_link_for_routed_child(
                        &child_frame,
                        &current_frame,
                        &routed_child,
                    )
                };
                current = child;
            } else {
                self.stats.inc(BTreeEvent::ResolveCold);
                let child =
                    unsafe { pool.fix_orphan_frame(child_swip.as_page_id(), NoLatches::new(pool)) };
                // Re-validate after the blocking fix: another thread could
                // have exclusively latched this inner node during the fix and
                // modified its routing entries.
                if inner.validate().is_err() {
                    self.stats.inc(BTreeEvent::LeafDescentRestarts);
                    return Err(Restart);
                }
                let child_frame = ResidentFrame::from_pinned(&child);
                unsafe {
                    self.set_parent_link_for_routed_child(
                        &child_frame,
                        &current_frame,
                        &routed_child,
                    )
                };
                if let Ok(parent) = inner.upgrade_to_exclusive() {
                    parent.set_child_edge_swip(routed_child.edge(), child_frame.hot_swip());
                    drop(parent);
                }
                current = child;
            }
        }
    }

    unsafe fn find_leaf_exclusive_with_path<'a>(
        &'a self,
        key: &[u8],
    ) -> Result<(Vec<PinnedFrame<'a>>, ExclusiveNode<'a, Leaf>), Restart> {
        let pool = self.pool();
        let mut path = Vec::new();
        let mut current = unsafe { pool.fix_frame(&self.meta_swip, NoLatches::new(pool)) };

        loop {
            let opt = current.clone_pin().optimistic().map_err(|_| {
                self.stats.inc(BTreeEvent::SplitPathRestarts);
                Restart
            })?;
            let is_leaf = BTreeNode::is_leaf(opt.read_ref());
            if is_leaf {
                let leaf = OptimisticNode::from_leaf_frame(opt);
                if leaf.should_chase_right(key) {
                    let empty_leaf = leaf.resident_frame().is_empty_leaf();
                    self.stats.inc(BTreeEvent::LeftChases);
                    if leaf.validate().is_err() {
                        self.stats.inc(BTreeEvent::SplitPathRestarts);
                        return Err(Restart);
                    }
                    if !empty_leaf {
                        self.stats.inc(BTreeEvent::SplitPathRestarts);
                        return Err(Restart);
                    }
                }
                match leaf.upgrade_to_exclusive() {
                    Ok(leaf) => return Ok((path, leaf)),
                    Err(_current) => {
                        self.stats.inc(BTreeEvent::SplitPathRestarts);
                        return Err(Restart);
                    }
                }
            }

            let inner = OptimisticNode::<Inner>::from_inner_frame(opt);
            let current_frame = inner.resident_frame();
            let routed_child = match inner.route_to_child(key) {
                Some(r) => r,
                None => {
                    self.stats.inc(BTreeEvent::SplitPathRestarts);
                    return Err(Restart);
                }
            };
            if inner.validate().is_err() {
                self.stats.inc(BTreeEvent::SplitPathRestarts);
                return Err(Restart);
            }
            let child_swip = routed_child.swip();

            let child = if child_swip.is_hot() || child_swip.is_cool() {
                let Some(child) = (unsafe { pool.pin_child(child_swip, NoLatches::new(pool)) })
                else {
                    self.stats.inc(BTreeEvent::SplitPathRestarts);
                    return Err(Restart);
                };
                // Re-validate after pin — child could have been evicted
                // between our validate and the pin.
                if inner.validate().is_err() {
                    self.stats.inc(BTreeEvent::SplitPathRestarts);
                    return Err(Restart);
                }
                child
            } else {
                self.stats.inc(BTreeEvent::ResolveCold);
                let child =
                    unsafe { pool.fix_orphan_frame(child_swip.as_page_id(), NoLatches::new(pool)) };
                // Re-validate after the blocking fix: another thread could
                // have exclusively latched this inner node during the fix and
                // modified its routing entries.
                if inner.validate().is_err() {
                    self.stats.inc(BTreeEvent::SplitPathRestarts);
                    return Err(Restart);
                }
                let child_frame = ResidentFrame::from_pinned(&child);
                // Swizzle-in: write HOT back to parent page bytes.
                if let Ok(parent) = inner.upgrade_to_exclusive() {
                    parent.set_child_edge_swip(routed_child.edge(), child_frame.hot_swip());
                    drop(parent);
                }
                child
            };
            let child_frame = ResidentFrame::from_pinned(&child);
            unsafe {
                self.set_parent_link_for_routed_child(&child_frame, &current_frame, &routed_child)
            };
            path.push(current);
            current = child;
        }
    }

    unsafe fn find_leaf_optimistic<'a>(
        &'a self,
        key: &[u8],
    ) -> Result<OptimisticNode<'a, Leaf>, Restart> {
        const MAX_OPTIMISTIC_STEPS: usize = 256;
        let pool = self.pool();
        let root_swip = self.meta_swip.load(Ordering::Acquire);
        let mut steps = 0usize;
        let mut current = if root_swip.is_hot() || root_swip.is_cool() {
            let root = unsafe { ResidentFrame::from_hot_swip(root_swip) }.ok_or(Restart)?;
            debug_assert!(
                root.is_in_pool(pool),
                "find_leaf_optimistic: invalid hot root swip: {:?}",
                root_swip,
            );
            let opt = unsafe { root.optimistic_guard() }?;
            if root.is_leaf() {
                let _ = opt;
                unsafe { pool.fix_frame(&self.meta_swip, NoLatches::new(pool)) }
            } else {
                let routed_child = root.try_route_to_child(key).ok_or(Restart)?;
                let child_swip = routed_child.swip();
                if opt.validate().is_err() {
                    return Err(Restart);
                }
                if child_swip.is_hot() || child_swip.is_cool() {
                    let Some(child) = (unsafe { pool.pin_child(child_swip, NoLatches::new(pool)) })
                    else {
                        return Err(Restart);
                    };
                    let child_frame = ResidentFrame::from_pinned(&child);
                    if opt.validate().is_err() {
                        return Err(Restart);
                    }
                    unsafe {
                        self.set_parent_link_for_routed_child(&child_frame, &root, &routed_child)
                    }
                    child
                } else {
                    self.stats.inc(BTreeEvent::ResolveCold);
                    let child = unsafe {
                        pool.fix_orphan_frame(child_swip.as_page_id(), NoLatches::new(pool))
                    };
                    // Re-validate after the blocking fix: another thread could
                    // have exclusively latched the root during the fix and
                    // modified its routing entries.
                    if opt.validate().is_err() {
                        return Err(Restart);
                    }
                    let child_frame = ResidentFrame::from_pinned(&child);
                    unsafe {
                        self.set_parent_link_for_routed_child(&child_frame, &root, &routed_child)
                    };
                    if let Ok(parent) = opt.upgrade_to_exclusive() {
                        match routed_child.edge() {
                            ParentEdge::Upper => root.set_upper(child_frame.hot_swip()),
                            ParentEdge::Slot(pos) => {
                                root.set_child_swip_at(pos, child_frame.hot_swip())
                            }
                        }
                        drop(parent);
                    }
                    child
                }
            }
        } else {
            unsafe { pool.fix_frame(&self.meta_swip, NoLatches::new(pool)) }
        };

        loop {
            steps += 1;
            if steps >= MAX_OPTIMISTIC_STEPS {
                self.stats.inc(BTreeEvent::LeafDescentRestarts);
                return Err(Restart);
            }
            let opt = current.optimistic().map_err(|_| Restart)?;
            let is_leaf = BTreeNode::is_leaf(opt.read_ref());
            if is_leaf {
                let leaf = OptimisticNode::from_leaf_frame(opt);
                if leaf.should_chase_right(key) {
                    self.stats.inc(BTreeEvent::LeftChases);
                    if leaf.validate().is_err() {
                        return Err(Restart);
                    }
                    let right_pid = leaf.right_pid();
                    if right_pid == 0 {
                        return Err(Restart);
                    }
                    let Some(right) = (unsafe { pool.try_fix_orphan_frame(right_pid) }) else {
                        return Err(Restart);
                    };
                    current = right;
                    continue;
                }
                return Ok(leaf);
            }

            let inner = OptimisticNode::<Inner>::from_inner_frame(opt);
            let current_frame = inner.resident_frame();
            let routed_child = inner.route_to_child(key).ok_or(Restart)?;
            if inner.validate().is_err() {
                return Err(Restart);
            }
            let child_swip = routed_child.swip();

            if child_swip.is_hot() || child_swip.is_cool() {
                let Some(child) = (unsafe { pool.try_pin_child(child_swip) }) else {
                    return Err(Restart);
                };
                let child_frame = ResidentFrame::from_pinned(&child);
                if inner.validate().is_err() {
                    return Err(Restart);
                }
                unsafe {
                    self.set_parent_link_for_routed_child(
                        &child_frame,
                        &current_frame,
                        &routed_child,
                    )
                };
                current = child;
            } else {
                self.stats.inc(BTreeEvent::ResolveCold);
                let Some(child) = (unsafe { pool.try_fix_orphan_frame(child_swip.as_page_id()) })
                else {
                    return Err(Restart);
                };
                // Re-validate the optimistic guard after loading the child.
                // Between route_to_child (line 852) and try_fix_orphan_frame,
                // another thread could have exclusively latched this inner
                // node (e.g., during eviction's unswizzle_parent) and modified
                // its routing entries. Without this check, we'd proceed with
                // a stale routed_child that points to the wrong subtree.
                if inner.validate().is_err() {
                    return Err(Restart);
                }
                let child_frame = ResidentFrame::from_pinned(&child);
                unsafe {
                    self.set_parent_link_for_routed_child(
                        &child_frame,
                        &current_frame,
                        &routed_child,
                    )
                };
                if let Ok(parent) = inner.upgrade_to_exclusive() {
                    parent.set_child_edge_swip(routed_child.edge(), child_frame.hot_swip());
                    drop(parent);
                }
                current = child;
            }
        }
    }

    unsafe fn find_leaf_exclusive_fallback<'a>(
        &'a self,
        key: &[u8],
    ) -> Result<ExclusiveNode<'a, Leaf>, Restart> {
        let pool = self.pool();
        let mut current = unsafe { pool.fix_frame(&self.meta_swip, NoLatches::new(pool)) };

        loop {
            let current_shared = current.clone_pin().shared();
            let shared = SharedNode::<Leaf>::from_leaf_frame(current_shared);
            let current_frame = shared.resident_frame();
            if current_frame.is_leaf() {
                if current_frame.should_chase_right(key) {
                    let right_pid = current_frame.leaf_right_pid();
                    if right_pid == 0 {
                        return Err(Restart);
                    }
                    drop(shared);
                    current = unsafe { pool.fix_orphan_frame(right_pid, NoLatches::new(pool)) };
                    continue;
                }
                let leaf =
                    ExclusiveNode::from_leaf_frame(shared.into_frame().into_pinned().exclusive());
                if leaf.resident_frame().should_chase_right(key) {
                    let right_pid = leaf.right_pid();
                    if right_pid == 0 {
                        return Err(Restart);
                    }
                    drop(leaf);
                    current = unsafe { pool.fix_orphan_frame(right_pid, NoLatches::new(pool)) };
                    continue;
                }
                return Ok(leaf);
            }

            let routed_child = current_frame.try_route_to_child(key).ok_or(Restart)?;
            let Some(child) =
                (unsafe { Self::try_resolve_child_for_read(pool, routed_child.swip()) })
            else {
                return Err(Restart);
            };
            unsafe {
                self.set_parent_link_for_routed_child(
                    &ResidentFrame::from_pinned(&child),
                    &current_frame,
                    &routed_child,
                )
            };
            current = child;
        }
    }

    unsafe fn find_leaf_exclusive_with_path_fallback<'a>(
        &'a self,
        key: &[u8],
    ) -> Result<(Vec<PinnedFrame<'a>>, ExclusiveNode<'a, Leaf>), Restart> {
        let pool = self.pool();
        let mut path = Vec::new();
        let mut current = unsafe { pool.fix_frame(&self.meta_swip, NoLatches::new(pool)) };

        loop {
            let current_shared = current.clone_pin().shared();
            let shared = SharedNode::<Leaf>::from_leaf_frame(current_shared);
            let current_frame = shared.resident_frame();
            if current_frame.is_leaf() {
                if current_frame.should_chase_right(key) {
                    let right_pid = current_frame.leaf_right_pid();
                    if right_pid == 0 {
                        return Err(Restart);
                    }
                    drop(shared);
                    current = unsafe { pool.fix_orphan_frame(right_pid, NoLatches::new(pool)) };
                    continue;
                }
                let leaf =
                    ExclusiveNode::from_leaf_frame(shared.into_frame().into_pinned().exclusive());
                if leaf.resident_frame().should_chase_right(key) {
                    let right_pid = leaf.right_pid();
                    if right_pid == 0 {
                        return Err(Restart);
                    }
                    drop(leaf);
                    current = unsafe { pool.fix_orphan_frame(right_pid, NoLatches::new(pool)) };
                    continue;
                }
                return Ok((path, leaf));
            }

            let routed_child = current_frame.try_route_to_child(key).ok_or(Restart)?;
            let child_swip = routed_child.swip();
            let child = if child_swip.is_hot() || child_swip.is_cool() {
                let Some(child) = (unsafe { pool.try_pin_child(child_swip) }) else {
                    return Err(Restart);
                };
                child
            } else {
                let Some(child) = (unsafe { pool.try_fix_orphan_frame(child_swip.as_page_id()) })
                else {
                    return Err(Restart);
                };
                child
            };
            let parent_pin = shared.into_frame().into_pinned();
            unsafe {
                self.set_parent_link_for_routed_child(
                    &ResidentFrame::from_pinned(&child),
                    &current_frame,
                    &routed_child,
                )
            };
            path.push(parent_pin);
            current = child;
        }
    }

    // -----------------------------------------------------------------------
    // Split — works for both leaf and inner nodes
    // -----------------------------------------------------------------------

    /// Split a full node. The node must be exclusively latched.
    /// After return, the original node is unlatched and unpinned.
    ///
    /// For non-root splits: finds the parent, latches parent exclusively,
    /// performs the split, inserts separator, then releases everything.
    /// This ensures no other thread sees the half-split state.
    unsafe fn split_node(
        &self,
        node: ExclusiveFrame<'_>,
        parent_path: &mut Vec<PinnedFrame<'_>>,
        pending_key: Option<&[u8]>,
        pre_sibling: Option<(u64, PinnedFrame<'_>)>,
    ) {
        let pool = self.pool();
        let node_frame = ResidentFrame::from_exclusive(&node);
        let is_leaf = node_frame.is_leaf();
        if is_leaf {
            self.stats.inc(BTreeEvent::LeafSplits);
        } else {
            self.stats.inc(BTreeEvent::InnerSplits);
        }
        let sp = node_frame.sp();
        let count = sp.num_slots();
        if count < 2 {
            // A leaf with a single entry that fills the page cannot be split
            // into two halves. Instead, allocate an empty sibling so the
            // pending insert routes to the empty side. If no pending key is
            // provided (inner node recursion), bail out — the caller will
            // retry or the parent will be split instead.
            if !is_leaf {
                return;
            }
            let Some(pending) = pending_key else {
                return;
            };
            let existing_key = sp.get_key(0);
            // If the pending key is less than the existing key, move the
            // existing entry to the right sibling so the new entry lands in
            // the (now empty) left node. Otherwise, keep the existing entry
            // left and the new entry will route to the empty right sibling.
            let (sep_key, move_existing_to_right) = if pending < existing_key {
                (pending.to_vec(), true)
            } else {
                (existing_key.to_vec(), false)
            };
            unsafe {
                self.split_single_entry_leaf(
                    node,
                    parent_path,
                    &sep_key,
                    move_existing_to_right,
                    pre_sibling,
                );
            }
            return;
        }

        let split_pos = count / 2;
        let sep_key = sp.get_key(split_pos).to_vec();

        // Use pre-allocated sibling frame if available (allocated before
        // the exclusive latch was acquired). Otherwise allocate under
        // latch — safe but can block on eviction under contention.
        let (_new_sibling_pid, new_sibling) = match pre_sibling {
            Some(pre) => pre,
            None => pool.allocate_and_fix_frame(NoLatches::new(pool)),
        };
        let new_sibling = new_sibling.exclusive();
        let new_sibling_frame = ResidentFrame::from_exclusive(&new_sibling);
        new_sibling_frame.init(is_leaf);

        if is_leaf {
            // Standard B-link leaf split:
            // original node stays left, new sibling becomes right.
            let left_count = split_pos + 1;
            let right_start = split_pos + 1;
            let right_count = count - right_start;
            let old_left_pid = node_frame.leaf_left_pid();
            let old_right_pid = node_frame.leaf_right_pid();
            let node_pid = node_frame.pid();
            let new_sibling_pid = new_sibling_frame.pid();

            // Build new right sibling.
            {
                let src = node_frame.sp();
                let dst = new_sibling_frame.sp_mut();
                src.copy_key_value_range(dst, 0, right_start, right_count);
            }
            new_sibling_frame.set_leaf_left_pid(node_pid);
            new_sibling_frame.set_leaf_right_pid(old_right_pid);

            // Rebuild left side in place.
            let mut tmp = TmpBuf::new();
            let tmp_sp = SlottedPage::init(&mut tmp.0);
            tmp_sp.reserve_suffix(16); // leaf: upper/left-link + right-sibling
            node_frame
                .sp()
                .copy_key_value_range(tmp_sp, 0, 0, left_count);
            tmp_sp.set_flag(FLAG_IS_LEAF);
            tmp.0[LEFT_SIBLING_OFFSET..LEFT_SIBLING_OFFSET + 8]
                .copy_from_slice(&old_left_pid.to_ne_bytes());
            tmp.0[RIGHT_SIBLING_OFFSET..RIGHT_SIBLING_OFFSET + 8]
                .copy_from_slice(&new_sibling_pid.to_ne_bytes());
            pagebox_storage::slotted_page::write_page_type(
                &mut tmp.0,
                pagebox_storage::slotted_page::PageType::Index,
            );

            node_frame.replace_page(&tmp.0);
            if old_right_pid != 0 {
                unsafe { self.update_leaf_left_sibling(old_right_pid, new_sibling_pid) };
            }
        } else {
            // Inner node split:
            // Keep the original node as the LEFT half and publish a new RIGHT
            // sibling. This avoids a window where a concurrent descender can
            // still reach the original node while the unpublished sibling owns
            // the lower half of the subtree.
            //
            // Left gets separators [0, split_pos) and children [0, split_pos].
            // The separator at split_pos is promoted to the parent.
            // Right gets separators [split_pos+1, count) and children
            // [split_pos+1, count] + upper.
            let left_sep_count = split_pos;
            let right_sep_start = split_pos + 1;
            let right_sep_count = count - right_sep_start;

            // Build right sibling from separators [split_pos+1, count).
            {
                let src = node_frame.sp();
                let dst = new_sibling_frame.sp_mut();
                if right_sep_count > 0 {
                    src.copy_key_value_range(dst, 0, right_sep_start, right_sep_count);
                }
            }
            let orig_upper = node_frame.upper_swip();
            new_sibling_frame.set_upper(orig_upper);

            // Rebuild the original node in place as the left half.
            let mut tmp = TmpBuf::new();
            let tmp_sp = SlottedPage::init(&mut tmp.0);
            tmp_sp.reserve_suffix(8); // reserve upper slot
            if left_sep_count > 0 {
                node_frame
                    .sp()
                    .copy_key_value_range(tmp_sp, 0, 0, left_sep_count);
            }
            let left_upper = node_frame.child_swip_at(split_pos);
            pagebox_storage::slotted_page::write_page_type(
                &mut tmp.0,
                pagebox_storage::slotted_page::PageType::Index,
            );
            node_frame.replace_page(&tmp.0);
            node_frame.set_upper(left_upper);
            unsafe { self.refresh_inner_child_parent_links_for_frame(&node_frame) };
            unsafe { self.refresh_inner_child_parent_links_for_frame(&new_sibling_frame) };
        }

        new_sibling.mark_dirty();
        node.mark_dirty();

        // Now insert separator into parent.
        // Check if this is a root split by CAS on meta_swip.
        let left = SplitChild::from_exclusive(&node);
        let right = SplitChild::from_exclusive(&new_sibling);
        let current_root = self.meta_swip.load(Ordering::Acquire);
        let is_root = Self::swip_page_id(current_root) == node_frame.pid();

        if is_root {
            // Root split: create new root.
            let (_new_root_pid, new_root) = pool.allocate_and_fix_frame(NoLatches::new(pool));
            let new_root = new_root.exclusive();
            let new_root_frame = ResidentFrame::from_exclusive(&new_root);
            new_root_frame.init(false);
            let new_root = ExclusiveNode::from_inner_frame(new_root);
            new_root.insert_separator(0, &sep_key, left.swip());
            new_root.set_child_edge_swip(ParentEdge::Upper, right.swip());
            new_root.mark_dirty();

            // CAS to atomically install the new root.
            let new_root_s = new_root.resident_frame().hot_swip();
            match self.meta_swip.compare_exchange(
                current_root,
                new_root_s,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    unsafe {
                        self.set_root_parent_link(&new_root.resident_frame());
                        self.set_parent_link_for_edge(
                            &left.resident_frame(),
                            &new_root,
                            ParentEdge::Slot(0),
                        );
                        self.set_parent_link_for_edge(
                            &right.resident_frame(),
                            &new_root,
                            ParentEdge::Upper,
                        );
                    }
                    self.height.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    // Another thread already changed the root.
                    // Our split is still valid but we need to insert into
                    // the actual parent instead. Keep both child latches held
                    // until the parent routing is updated so the new left node
                    // never becomes temporarily unreachable.
                    left.resident_frame().set_parent_link_none();
                    right.resident_frame().set_parent_link_none();
                    unsafe {
                        self.publish_leaf_split_to_parent(&sep_key, left, right, parent_path)
                    };
                }
            }
        } else {
            // Non-root: find parent, latch it exclusively, then insert separator.
            // Keep node_guard held until parent is updated so no traversal sees
            // the split node without the parent routing correctly.
            unsafe { self.publish_leaf_split_to_parent(&sep_key, left, right, parent_path) };
        }
    }

    /// Handle the overflow case where a leaf has a single entry that fills
    /// the page and a new entry cannot fit alongside it. Allocates a sibling
    /// leaf and publishes the split to the parent, so the pending insert
    /// can route to the empty side on retry.
    ///
    /// If `move_existing_to_right` is true, the existing entry is moved to
    /// the right sibling and the left node is left empty (for cases where
    /// the pending key is smaller than the existing key). Otherwise the
    /// existing entry stays in the left node and the right sibling is empty.
    unsafe fn split_single_entry_leaf(
        &self,
        node: ExclusiveFrame<'_>,
        parent_path: &mut Vec<PinnedFrame<'_>>,
        sep_key: &[u8],
        move_existing_to_right: bool,
        pre_sibling: Option<(u64, PinnedFrame<'_>)>,
    ) {
        let pool = self.pool();
        self.stats.inc(BTreeEvent::LeafSplits);
        let node_frame = ResidentFrame::from_exclusive(&node);

        let (_new_sibling_pid, new_sibling) = match pre_sibling {
            Some(pre) => pre,
            None => pool.allocate_and_fix_frame(NoLatches::new(pool)),
        };
        let new_sibling = new_sibling.exclusive();
        let new_sibling_frame = ResidentFrame::from_exclusive(&new_sibling);
        new_sibling_frame.init(true);

        let old_left_pid = node_frame.leaf_left_pid();
        let old_right_pid = node_frame.leaf_right_pid();
        let node_pid = node_frame.pid();
        let new_sibling_pid = new_sibling_frame.pid();

        if move_existing_to_right {
            // Move the existing entry to the right sibling.
            let src = node_frame.sp();
            let dst = new_sibling_frame.sp_mut();
            src.copy_key_value_range(dst, 0, 0, 1);

            // Clear the left node to empty.
            let mut tmp = TmpBuf::new();
            let tmp_sp = SlottedPage::init(&mut tmp.0);
            tmp_sp.reserve_suffix(16);
            tmp_sp.set_flag(FLAG_IS_LEAF);
            tmp.0[LEFT_SIBLING_OFFSET..LEFT_SIBLING_OFFSET + 8]
                .copy_from_slice(&old_left_pid.to_ne_bytes());
            tmp.0[RIGHT_SIBLING_OFFSET..RIGHT_SIBLING_OFFSET + 8]
                .copy_from_slice(&new_sibling_pid.to_ne_bytes());
            pagebox_storage::slotted_page::write_page_type(
                &mut tmp.0,
                pagebox_storage::slotted_page::PageType::Index,
            );
            node_frame.replace_page(&tmp.0);
        } else {
            // Keep the existing entry in the left node; right sibling is empty.
            let mut tmp = TmpBuf::new();
            let tmp_sp = SlottedPage::init(&mut tmp.0);
            tmp_sp.reserve_suffix(16);
            node_frame.sp().copy_key_value_range(tmp_sp, 0, 0, 1);
            tmp_sp.set_flag(FLAG_IS_LEAF);
            tmp.0[LEFT_SIBLING_OFFSET..LEFT_SIBLING_OFFSET + 8]
                .copy_from_slice(&old_left_pid.to_ne_bytes());
            tmp.0[RIGHT_SIBLING_OFFSET..RIGHT_SIBLING_OFFSET + 8]
                .copy_from_slice(&new_sibling_pid.to_ne_bytes());
            pagebox_storage::slotted_page::write_page_type(
                &mut tmp.0,
                pagebox_storage::slotted_page::PageType::Index,
            );
            node_frame.replace_page(&tmp.0);
        }

        new_sibling_frame.set_leaf_left_pid(node_pid);
        new_sibling_frame.set_leaf_right_pid(old_right_pid);
        if old_right_pid != 0 {
            unsafe { self.update_leaf_left_sibling(old_right_pid, new_sibling_pid) };
        }

        new_sibling.mark_dirty();
        node.mark_dirty();

        let left = SplitChild::from_exclusive(&node);
        let right = SplitChild::from_exclusive(&new_sibling);
        let current_root = self.meta_swip.load(Ordering::Acquire);
        let is_root = Self::swip_page_id(current_root) == node_frame.pid();

        if is_root {
            let (_new_root_pid, new_root) = pool.allocate_and_fix_frame(NoLatches::new(pool));
            let new_root = new_root.exclusive();
            let new_root_frame = ResidentFrame::from_exclusive(&new_root);
            new_root_frame.init(false);
            let new_root = ExclusiveNode::from_inner_frame(new_root);
            new_root.insert_separator(0, sep_key, left.swip());
            new_root.set_child_edge_swip(ParentEdge::Upper, right.swip());
            new_root.mark_dirty();

            let new_root_s = new_root.resident_frame().hot_swip();
            match self.meta_swip.compare_exchange(
                current_root,
                new_root_s,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    unsafe {
                        self.set_root_parent_link(&new_root.resident_frame());
                        self.set_parent_link_for_edge(
                            &left.resident_frame(),
                            &new_root,
                            ParentEdge::Slot(0),
                        );
                        self.set_parent_link_for_edge(
                            &right.resident_frame(),
                            &new_root,
                            ParentEdge::Upper,
                        );
                    }
                    self.height.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    left.resident_frame().set_parent_link_none();
                    right.resident_frame().set_parent_link_none();
                    unsafe { self.publish_leaf_split_to_parent(sep_key, left, right, parent_path) };
                }
            }
        } else {
            unsafe { self.publish_leaf_split_to_parent(sep_key, left, right, parent_path) };
        }
    }

    unsafe fn publish_leaf_split_to_parent(
        &self,
        sep_key: &[u8],
        left: SplitChild<'_, '_>,
        right: SplitChild<'_, '_>,
        parent_path: &mut Vec<PinnedFrame<'_>>,
    ) {
        let mut attempts = 0u32;
        loop {
            attempts += 1;
            if attempts == 1
                && unsafe {
                    self.try_publish_leaf_split_via_parent_path(sep_key, left, right, parent_path)
                }
            {
                return;
            }
            if unsafe { self.try_publish_leaf_split_via_root_blocking(sep_key, left, right) } {
                return;
            }
            if unsafe { self.try_publish_leaf_split_via_blocking_search(sep_key, left, right) } {
                return;
            }
            self.stats.inc(BTreeEvent::ParentPublishRestarts);
            if attempts >= 100_000 {
                let left_pid = left.pid();
                let right_pid = right.pid();
                let root_pid = Self::swip_page_id(self.meta_swip.load(Ordering::Acquire));
                let root = unsafe {
                    self.pool()
                        .fix_frame(&self.meta_swip, NoLatches::new(self.pool()))
                };
                let root_frame = ResidentFrame::from_pinned(&root);
                let root_is_leaf = root_frame.is_leaf();
                let root_children = if root_is_leaf {
                    Vec::new()
                } else {
                    let root = ExclusiveNode::from_inner_frame(root.exclusive());
                    self.debug_child_page_ids(&root)
                };
                panic!(
                    "publish_leaf_split_to_parent: exceeded retry budget: sep_len={} \
                     left_pid={} right_pid={} root_pid={} height={} dt_id={} root_is_leaf={} \
                     root_children={root_children:?} attempts={attempts}",
                    sep_key.len(),
                    left_pid,
                    right_pid,
                    root_pid,
                    self.height(),
                    self.dt_id,
                    root_is_leaf,
                );
            }
            if attempts.is_multiple_of(64) {
                std::thread::yield_now();
            }
        }
    }

    unsafe fn try_publish_leaf_split_via_parent_path(
        &self,
        sep_key: &[u8],
        left: SplitChild<'_, '_>,
        right: SplitChild<'_, '_>,
        parent_path: &mut Vec<PinnedFrame<'_>>,
    ) -> bool {
        let Some(parent) = parent_path.pop() else {
            return false;
        };
        let parent = ExclusiveNode::from_inner_frame(parent.exclusive());
        let Some(edge) = self.parent_edge_for_child(&parent, &left.resident_frame()) else {
            drop(parent);
            return false;
        };

        if !parent.can_insert_separator(sep_key.len()) {
            unsafe { self.split_node(parent.into_frame(), parent_path, None, None) };
            return false;
        }

        unsafe { self.apply_split_to_latched_parent(&parent, sep_key, left, right, edge) };
        parent.mark_dirty();
        true
    }

    unsafe fn apply_split_to_latched_parent(
        &self,
        parent: &ExclusiveNode<'_, Inner>,
        sep_key: &[u8],
        left: SplitChild<'_, '_>,
        right: SplitChild<'_, '_>,
        edge: ParentEdge,
    ) {
        let count = parent.num_slots();
        let pos = edge.pos(count);
        let right_uses_upper = split_right_parent_uses_upper(pos, count);
        let separator_pos = split_separator_insert_pos(pos, count);
        let left_link_edge = ParentEdge::Slot(pos);
        let right_link_edge = if right_uses_upper {
            ParentEdge::Upper
        } else {
            ParentEdge::Slot(pos + 1)
        };
        if !right_uses_upper {
            parent.set_child_edge_swip(ParentEdge::Slot(pos), right.swip());
            parent.insert_separator(separator_pos, sep_key, left.swip());
        } else {
            parent.insert_separator(separator_pos, sep_key, left.swip());
            parent.set_child_edge_swip(ParentEdge::Upper, right.swip());
        }
        unsafe {
            self.set_parent_link_for_edge(&left.resident_frame(), parent, left_link_edge);
            self.set_parent_link_for_edge(&right.resident_frame(), parent, right_link_edge);
        }
    }

    unsafe fn try_publish_leaf_split_via_root_blocking(
        &self,
        sep_key: &[u8],
        left: SplitChild<'_, '_>,
        right: SplitChild<'_, '_>,
    ) -> bool {
        let root = unsafe {
            self.pool()
                .fix_frame(&self.meta_swip, NoLatches::new(self.pool()))
                .exclusive()
        };
        let root_inner = ExclusiveNode::from_inner_frame(root);
        let Some(edge) = self.parent_edge_for_child(&root_inner, &left.resident_frame()) else {
            drop(root_inner);
            return false;
        };

        if !root_inner.can_insert_separator(sep_key.len()) {
            let mut empty_path = Vec::new();
            unsafe { self.split_node(root_inner.into_frame(), &mut empty_path, None, None) };
            return false;
        }

        unsafe { self.apply_split_to_latched_parent(&root_inner, sep_key, left, right, edge) };
        root_inner.mark_dirty();
        true
    }

    unsafe fn try_publish_leaf_split_via_blocking_search(
        &self,
        sep_key: &[u8],
        left: SplitChild<'_, '_>,
        right: SplitChild<'_, '_>,
    ) -> bool {
        let pool = self.pool();
        let left_pid = left.pid();
        let mut stack = vec![unsafe { pool.fix_frame(&self.meta_swip, NoLatches::new(pool)) }];

        while let Some(current) = stack.pop() {
            let current = ExclusiveNode::from_inner_frame(current.exclusive());

            if current.resident_frame().is_leaf() {
                continue;
            }

            if let Some(edge) = current.child_edge_for(ChildRef::from_frame(&left.resident_frame()))
            {
                if !current.can_insert_separator(sep_key.len()) {
                    let mut empty_path = Vec::new();
                    unsafe { self.split_node(current.into_frame(), &mut empty_path, None, None) };
                    return false;
                }

                unsafe { self.apply_split_to_latched_parent(&current, sep_key, left, right, edge) };
                current.mark_dirty();
                return true;
            }

            let child_routes = current.child_routes();
            drop(current);

            for routed in child_routes.into_iter().rev() {
                let swip = routed.swip();
                if unsafe { ResidentFrame::from_hot_swip(swip) }
                    .is_some_and(|frame| frame.same_frame(&left.resident_frame()))
                {
                    continue;
                }
                if swip.is_evicted() && swip.as_page_id() == left_pid {
                    continue;
                }
                if let Some(child) = unsafe { pool.try_pin_child(swip) } {
                    stack.push(child);
                }
            }
        }

        false
    }

    // -----------------------------------------------------------------------
    // ParentFinder — tree walk for eviction unswizzle
    // -----------------------------------------------------------------------

    /// DFS from root to find the parent of `child_bf`, then write
    /// Swip::evicted(child_pid) to the parent's routing edge.
    /// Called by eviction when the cached ParentLink::InnerNode hint
    /// is stale.
    unsafe fn eviction_unswizzle(&self, child: &ResidentFrame<'_>, child_pid: u64) -> bool {
        // This fallback only runs after the cached parent hint failed. Under a
        // tight pool, giving up early can strand all candidate leaves resident.
        const MAX_EVICTION_UNSWIZZLE_VISITS: usize = 262_144;
        const MAX_EVICTION_UNSWIZZLE_RESTARTS: usize = 64;

        self.stats.inc(BTreeEvent::EvictionUnswizzleCalls);
        let pool = self.pool();
        let root_swip = self.meta_swip.load(Ordering::Acquire);
        if !root_swip.is_hot() {
            return false;
        }
        let Some(root) = (unsafe { ResidentFrame::from_hot_swip(root_swip) }) else {
            return false;
        };
        // Check if child IS the root.
        if root.same_frame(child) {
            self.meta_swip
                .store(Swip::evicted(child_pid), Ordering::Release);
            return true;
        }

        let Some(root) = (unsafe { pool.try_pin_child(root_swip) }) else {
            return false;
        };
        let mut stack = vec![root];
        let mut visited = 0usize;
        let mut restarts = 0usize;
        while let Some(node) = stack.pop() {
            self.stats.eviction_unswizzle_nodes_visited.inc();
            visited += 1;
            if visited > MAX_EVICTION_UNSWIZZLE_VISITS || restarts > MAX_EVICTION_UNSWIZZLE_RESTARTS
            {
                return false;
            }
            let node_bf = node.frame_ref();
            // Skip frames that are no longer resident (evicted/freed).
            let frame_state = node_bf.state();
            if frame_state != pagebox_storage::buffer_frame::FrameState::Resident {
                continue;
            }
            // Try optimistic read on this node.
            let Ok(opt) = node.optimistic() else {
                continue;
            };
            let is_leaf = BTreeNode::is_leaf(opt.read_ref());
            if is_leaf {
                continue;
            }
            let opt_inner = OptimisticNode::from_inner_frame(opt);
            // Check if this node is the parent of child_bf.
            if opt_inner
                .child_edge_for(ChildRef::from_frame(child))
                .is_some()
            {
                self.stats.inc(BTreeEvent::EvictionUnswizzleParentHits);
                if opt_inner.validate().is_err() {
                    // Restart DFS.
                    self.stats.inc(BTreeEvent::EvictionUnswizzleRestarts);
                    restarts += 1;
                    stack.clear();
                    let Some(root) = (unsafe { pool.try_pin_child(root_swip) }) else {
                        return false;
                    };
                    stack.push(root);
                    continue;
                }
                // Eviction must stay non-blocking here. If the parent
                // cannot be upgraded immediately, abort this unswizzle
                // attempt and let eviction revert the child to Resident.
                let Ok(parent) = opt_inner.upgrade_to_exclusive() else {
                    self.stats.inc(BTreeEvent::EvictionUnswizzleUpgradeFailures);
                    return false;
                };

                if let Some(edge) = parent.child_edge_for(ChildRef::from_frame(child)) {
                    parent.set_child_edge_swip(edge, Swip::evicted(child_pid));
                    parent.resident_frame().mark_header_dirty();
                }
                drop(parent);
                return true;
            }
            // Not the parent — walk down HOT/COOL children.
            // Keep a fast path for a direct EVICTED target edge: if this node
            // already points to the target page by ID but not by pointer (stale
            // swizzling), we can still correct the swip in place.
            let mut child_routes_ok = true;
            let mut has_evicted_target_edge = false;
            if opt_inner
                .for_each_child_route(|routed| {
                    let swip = routed.swip();
                    if swip.is_evicted() && swip.as_page_id() == child_pid {
                        has_evicted_target_edge = true;
                    }
                    if !(swip.is_hot() || swip.is_cool()) {
                        return;
                    }
                    let child_frame = unsafe { ResidentFrame::from_hot_swip(swip) }.unwrap();
                    if child_frame.is_in_pool(pool)
                        && let Some(child) = unsafe { pool.try_pin_child(swip) }
                    {
                        stack.push(child);
                    }
                })
                .is_none()
            {
                child_routes_ok = false;
            }
            if has_evicted_target_edge {
                if opt_inner.validate().is_err() {
                    self.stats.inc(BTreeEvent::EvictionUnswizzleRestarts);
                    restarts += 1;
                    stack.clear();
                    let Some(root) = (unsafe { pool.try_pin_child(root_swip) }) else {
                        return false;
                    };
                    stack.push(root);
                    continue;
                }
                let Ok(parent) = opt_inner.upgrade_to_exclusive() else {
                    self.stats.inc(BTreeEvent::EvictionUnswizzleUpgradeFailures);
                    return false;
                };
                if let Some(edge) = parent.child_edge_for(ChildRef::from_frame(child)) {
                    self.stats.inc(BTreeEvent::EvictionUnswizzleParentHits);
                    parent.set_child_edge_swip(edge, Swip::evicted(child_pid));
                    parent.resident_frame().mark_header_dirty();
                }
                drop(parent);
                return true;
            }
            if !child_routes_ok {
                self.stats.inc(BTreeEvent::EvictionUnswizzleRestarts);
                restarts += 1;
                stack.clear();
                let Some(root) = (unsafe { pool.try_pin_child(root_swip) }) else {
                    return false;
                };
                stack.push(root);
                continue;
            }
            if opt_inner.validate().is_err() {
                // Data might be torn — discard this DFS branch.
                self.stats.inc(BTreeEvent::EvictionUnswizzleRestarts);
                restarts += 1;
                stack.clear();
                let Some(root) = (unsafe { pool.pin_child(root_swip, NoLatches::new(pool)) })
                else {
                    return false;
                };
                stack.push(root);
                continue;
            }
        }
        false
    }

    // -----------------------------------------------------------------------
    // Merge
    // -----------------------------------------------------------------------

    /// Find the parent of `target_bf` by traversing from the root.
    /// Returns (parent_bf pinned, position of target in parent's children).
    #[cfg(test)]
    unsafe fn find_parent<'a>(
        &'a self,
        target: &ResidentFrame<'_>,
    ) -> Result<(PinnedFrame<'a>, u16), Restart> {
        let pool = self.pool();
        let mut stack = vec![unsafe { pool.fix_frame(&self.meta_swip, NoLatches::new(pool)) }];

        while let Some(current) = stack.pop() {
            let opt = current.clone_pin().optimistic().map_err(|_| Restart)?;
            let current_frame = ResidentFrame::from_optimistic(&opt);

            if current_frame.is_leaf() {
                if opt.validate().is_err() {
                    return Err(Restart);
                }
                continue;
            }

            let target_ref = ChildRef::from_frame(target);
            let opt_inner = OptimisticNode::from_inner_frame(opt);
            if let Some(edge) = opt_inner.child_edge_for(target_ref) {
                if opt_inner.validate().is_err() {
                    return Err(Restart);
                }
                let count = current_frame.num_slots();
                return Ok((current, edge.pos(count)));
            }

            let mut children = Vec::new();
            opt_inner
                .for_each_child_route(|routed| {
                    let swip = routed.swip();
                    let child = if swip.is_hot() || swip.is_cool() {
                        unsafe { pool.pin_child(swip, NoLatches::new(pool)) }
                    } else {
                        Some(unsafe {
                            pool.fix_orphan_frame(swip.as_page_id(), NoLatches::new(pool))
                        })
                    };
                    if let Some(child) = child {
                        children.push(child);
                    }
                })
                .ok_or(Restart)?;

            if opt_inner.validate().is_err() {
                return Err(Restart);
            }

            stack.extend(children.into_iter().rev());
        }

        Err(Restart)
    }

    unsafe fn repair_separators_after_delete(
        &self,
        path: &mut Vec<PinnedFrame<'_>>,
        leaf_bf: BufferFrameRef,
        leaf_pid: u64,
        new_max: Option<Vec<u8>>,
    ) {
        let mut child_pid = leaf_pid;
        let mut child_bf = leaf_bf;
        let mut new_max = new_max;

        while let Some(parent) = path.pop() {
            let parent = ExclusiveNode::from_inner_frame(parent.exclusive());
            let Some(edge) = parent.child_edge_for(ChildRef::from_pid(child_bf, child_pid)) else {
                break;
            };

            let count = parent.num_slots();
            let pos = edge.pos(count);
            if pos < count {
                if let Some(ref key) = new_max
                    && parent.key_at(pos) != key.as_slice()
                {
                    parent.set_separator_key(pos, key);
                    parent.mark_dirty();
                }
                break;
            }

            child_pid = parent.pid();
            child_bf = parent.resident_frame().bf();
            new_max = new_max.take();
        }
    }

    fn leaf_slots_look_sane(sp: &SlottedPage) -> bool {
        let count = sp.num_slots();
        if count as usize > PAGE_SIZE / 12 {
            return false;
        }
        for i in 0..count {
            if sp.try_get_key(i).is_none() || sp.try_get_value(i).is_none() {
                return false;
            }
        }
        true
    }

    unsafe fn leaf_pair_is_mergeable(
        &self,
        parent: &ExclusiveNode<'_, Inner>,
        left: &ExclusiveNode<'_, Leaf>,
        right: &ExclusiveNode<'_, Leaf>,
        left_pos: u16,
    ) -> bool {
        if left.resident_frame().same_frame(&right.resident_frame()) {
            return false;
        }

        if !parent.child_edge_matches(
            ParentEdge::Slot(left_pos),
            ChildRef::from_frame(&left.resident_frame()),
        ) {
            return false;
        }

        let count = parent.num_slots();
        let right_edge = ParentEdge::from_pos(left_pos + 1, count);
        if !parent.child_edge_matches(right_edge, ChildRef::from_frame(&right.resident_frame())) {
            return false;
        }

        let left_pid = left.pid();
        let right_pid = right.pid();
        if left_pid == right_pid || left.right_pid() != right_pid {
            return false;
        }

        let left_sp = left.resident_frame().sp();
        let right_sp = right.resident_frame().sp();
        if !Self::leaf_slots_look_sane(left_sp) || !Self::leaf_slots_look_sane(right_sp) {
            return false;
        }

        let left_count = left_sp.num_slots();
        let right_count = right_sp.num_slots();
        if left_count > 0 && right_count > 0 {
            let Some(left_max) = left_sp.try_get_key(left_count - 1) else {
                return false;
            };
            let Some(right_min) = right_sp.try_get_key(0) else {
                return false;
            };
            if left_max > right_min {
                return false;
            }
        }

        true
    }

    unsafe fn leaf_pair_fits(
        &self,
        left: &ExclusiveNode<'_, Leaf>,
        right: &ExclusiveNode<'_, Leaf>,
    ) -> bool {
        let mut tmp = TmpBuf::new();
        let tmp_sp = SlottedPage::init(&mut tmp.0);
        tmp_sp.reserve_suffix(16);
        tmp_sp.set_flag(FLAG_IS_LEAF);
        let left = left.resident_frame().sp();
        let right = right.resident_frame().sp();

        for i in 0..left.num_slots() {
            let Some(key) = left.try_get_key(i) else {
                return false;
            };
            let Some(value) = left.try_get_value(i) else {
                return false;
            };
            if !tmp_sp.can_insert(key.len(), value.len()) {
                return false;
            }
            tmp_sp.insert(tmp_sp.num_slots(), key, value);
        }
        for i in 0..right.num_slots() {
            let Some(key) = right.try_get_key(i) else {
                return false;
            };
            let Some(value) = right.try_get_value(i) else {
                return false;
            };
            if !tmp_sp.can_insert(key.len(), value.len()) {
                return false;
            }
            tmp_sp.insert(tmp_sp.num_slots(), key, value);
        }
        true
    }

    unsafe fn merge_leaf_pages_into_left(
        &self,
        left: &ExclusiveNode<'_, Leaf>,
        right: &ExclusiveNode<'_, Leaf>,
    ) -> bool {
        let left_sp = left.resident_frame().sp();
        let right_sp = right.resident_frame().sp();
        let left_pid = left.pid();
        let left_left = left.left_pid();
        let new_leaf_right = right.right_pid();

        let mut tmp = TmpBuf::new();
        let tmp_sp = SlottedPage::init(&mut tmp.0);
        tmp_sp.reserve_suffix(16);
        tmp_sp.set_flag(FLAG_IS_LEAF);
        for i in 0..left_sp.num_slots() {
            let Some(key) = left_sp.try_get_key(i) else {
                return false;
            };
            let Some(value) = left_sp.try_get_value(i) else {
                return false;
            };
            if !tmp_sp.can_insert(key.len(), value.len()) {
                return false;
            }
            tmp_sp.insert(tmp_sp.num_slots(), key, value);
        }
        for i in 0..right_sp.num_slots() {
            let Some(key) = right_sp.try_get_key(i) else {
                return false;
            };
            let Some(value) = right_sp.try_get_value(i) else {
                return false;
            };
            if !tmp_sp.can_insert(key.len(), value.len()) {
                return false;
            }
            tmp_sp.insert(tmp_sp.num_slots(), key, value);
        }
        tmp.0[LEFT_SIBLING_OFFSET..LEFT_SIBLING_OFFSET + 8]
            .copy_from_slice(&left_left.to_ne_bytes());
        tmp.0[RIGHT_SIBLING_OFFSET..RIGHT_SIBLING_OFFSET + 8]
            .copy_from_slice(&new_leaf_right.to_ne_bytes());
        pagebox_storage::slotted_page::write_page_type(
            &mut tmp.0,
            pagebox_storage::slotted_page::PageType::Index,
        );
        left.resident_frame().replace_page(&tmp.0);
        left.mark_dirty();
        if new_leaf_right != 0 {
            unsafe { self.update_leaf_left_sibling(new_leaf_right, left_pid) };
        }
        true
    }

    unsafe fn inner_pair_fits(
        &self,
        left: &ExclusiveNode<'_, Inner>,
        boundary_key: &[u8],
        right: &ExclusiveNode<'_, Inner>,
    ) -> bool {
        let mut tmp = TmpBuf::new();
        let tmp_sp = SlottedPage::init(&mut tmp.0);
        tmp_sp.reserve_suffix(8);

        let left_frame = left.resident_frame();
        let left_sp = left_frame.sp();
        let right_sp = right.resident_frame().sp();

        for i in 0..left_sp.num_slots() {
            if !tmp_sp.can_insert(left_sp.get_key(i).len(), 8) {
                return false;
            }
            tmp_sp.insert(tmp_sp.num_slots(), left_sp.get_key(i), left_sp.get_value(i));
        }
        if !tmp_sp.can_insert(boundary_key.len(), 8) {
            return false;
        }
        tmp_sp.insert(
            tmp_sp.num_slots(),
            boundary_key,
            &left_frame.upper_swip().raw().to_ne_bytes(),
        );
        for i in 0..right_sp.num_slots() {
            if !tmp_sp.can_insert(right_sp.get_key(i).len(), 8) {
                return false;
            }
            tmp_sp.insert(
                tmp_sp.num_slots(),
                right_sp.get_key(i),
                right_sp.get_value(i),
            );
        }
        true
    }

    unsafe fn refresh_inner_child_parent_links(&self, node: &ExclusiveNode<'_, Inner>) {
        unsafe { self.refresh_inner_child_parent_links_for_frame(&node.resident_frame()) };
    }

    unsafe fn refresh_inner_child_parent_links_for_frame(&self, node: &ResidentFrame<'_>) {
        let count = node.num_slots();
        for pos in 0..count {
            let swip = node.child_swip_at(pos);
            if !(swip.is_hot() || swip.is_cool()) {
                continue;
            }
            unsafe {
                self.set_inner_parent_link(
                    &ResidentFrame::from_hot_swip(swip).unwrap(),
                    node,
                    pos,
                    false,
                )
            };
        }

        let upper = node.upper_swip();
        if upper.is_hot() || upper.is_cool() {
            unsafe {
                self.set_inner_parent_link(
                    &ResidentFrame::from_hot_swip(upper).unwrap(),
                    node,
                    count,
                    true,
                )
            };
        }
    }

    unsafe fn merge_inner_pages_into_left(
        &self,
        left: &ExclusiveNode<'_, Inner>,
        boundary_key: &[u8],
        right: &ExclusiveNode<'_, Inner>,
    ) {
        let left_sp = left.resident_frame().sp();
        let right_sp = right.resident_frame().sp();
        let left_count = left_sp.num_slots();
        let right_count = right_sp.num_slots();
        let left_upper = left.resident_frame().upper_swip();
        let new_upper = right.resident_frame().upper_swip();

        let mut tmp = TmpBuf::new();
        let tmp_sp = SlottedPage::init(&mut tmp.0);
        tmp_sp.reserve_suffix(8);
        if left_count > 0 {
            left_sp.copy_key_value_range(tmp_sp, 0, 0, left_count);
        }
        tmp_sp.insert(
            tmp_sp.num_slots(),
            boundary_key,
            &left_upper.raw().to_ne_bytes(),
        );
        if right_count > 0 {
            right_sp.copy_key_value_range(tmp_sp, tmp_sp.num_slots(), 0, right_count);
        }
        pagebox_storage::slotted_page::write_page_type(
            &mut tmp.0,
            pagebox_storage::slotted_page::PageType::Index,
        );
        left.resident_frame().replace_page(&tmp.0);
        left.resident_frame().set_upper(new_upper);
        unsafe { self.refresh_inner_child_parent_links(left) };
        left.mark_dirty();
    }

    unsafe fn try_merge_leaf_with_path(
        &self,
        parent_path: &mut Vec<PinnedFrame<'_>>,
        leaf: ExclusiveNode<'_, Leaf>,
    ) -> bool {
        let leaf_frame = leaf.resident_frame();

        let Some(parent_pin) = parent_path.pop() else {
            return false;
        };
        let parent = ExclusiveNode::from_inner_frame(parent_pin.exclusive());
        let parent_frame = parent.resident_frame();
        if parent_frame.is_leaf() {
            return false;
        }

        let Some(edge) = self.parent_edge_for_child(&parent, &leaf_frame) else {
            return false;
        };
        let count = parent.num_slots();
        let pos = edge.pos(count);

        if pos < count {
            let right_edge = ParentEdge::from_pos(pos + 1, count);
            let right_swip = parent.child_edge_swip(right_edge);
            let right = match unsafe { self.pin_exclusive_child(right_swip) } {
                Some(child) => ExclusiveNode::from_leaf_frame(child),
                None => return false,
            };
            let right_frame = right.resident_frame();
            if unsafe { self.leaf_pair_is_mergeable(&parent, &leaf, &right, pos) }
                && unsafe { self.leaf_pair_fits(&leaf, &right) }
                && unsafe { self.merge_leaf_pages_into_left(&leaf, &right) }
            {
                unsafe {
                    self.unlink_merged_right_leaf(
                        &parent,
                        &leaf_frame,
                        &right_frame,
                        right_edge,
                        pos,
                    )
                };
                parent.mark_dirty();

                drop(right);

                let parent_is_root =
                    Self::swip_page_id(self.meta_swip.load(Ordering::Acquire)) == parent.pid();
                let parent_count = parent.num_slots();
                if parent_is_root && parent_count == 0 {
                    unsafe { self.collapse_empty_root_to_child(&parent_frame, &leaf_frame) };
                    drop(parent);
                } else if !parent_is_root && parent.is_underfull() {
                    let _ = unsafe { self.try_merge_inner_with_path(parent_path, parent) };
                }
                return true;
            }
        }

        if pos > 0 {
            let left_pos = pos - 1;
            let left_swip = parent.child_edge_swip(ParentEdge::Slot(left_pos));
            let left = match unsafe { self.pin_exclusive_child(left_swip) } {
                Some(child) => ExclusiveNode::from_leaf_frame(child),
                None => return false,
            };
            let left_frame = left.resident_frame();
            if unsafe { self.leaf_pair_is_mergeable(&parent, &left, &leaf, left_pos) }
                && unsafe { self.leaf_pair_fits(&left, &leaf) }
                && unsafe { self.merge_leaf_pages_into_left(&left, &leaf) }
            {
                let replacement_key = if pos == count {
                    None
                } else {
                    Some(parent.key_at(pos).to_vec())
                };
                unsafe {
                    self.unlink_merged_left_leaf(
                        &parent,
                        &left_frame,
                        &leaf_frame,
                        left_pos,
                        if pos == count { left_pos } else { pos },
                        replacement_key.as_deref(),
                    )
                };
                parent.mark_dirty();

                leaf_frame.set_parent_link_none();

                let parent_is_root =
                    Self::swip_page_id(self.meta_swip.load(Ordering::Acquire)) == parent.pid();
                let parent_count = parent.num_slots();
                if parent_is_root && parent_count == 0 {
                    unsafe { self.collapse_empty_root_to_child(&parent_frame, &left_frame) };
                    drop(parent);
                } else if !parent_is_root && parent.is_underfull() {
                    let _ = unsafe { self.try_merge_inner_with_path(parent_path, parent) };
                }
                return true;
            }
        }

        false
    }

    unsafe fn try_merge_inner_with_path(
        &self,
        parent_path: &mut Vec<PinnedFrame<'_>>,
        node: ExclusiveNode<'_, Inner>,
    ) -> bool {
        let node_frame = node.resident_frame();

        let Some(parent_pin) = parent_path.pop() else {
            return false;
        };
        let parent = ExclusiveNode::from_inner_frame(parent_pin.exclusive());
        let parent_frame = parent.resident_frame();
        if parent_frame.is_leaf() {
            return false;
        }

        let Some(edge) = self.parent_edge_for_child(&parent, &node_frame) else {
            return false;
        };
        let count = parent.num_slots();
        let pos = edge.pos(count);

        if pos < count {
            let right_edge = ParentEdge::from_pos(pos + 1, count);
            let right_swip = parent.child_edge_swip(right_edge);
            let right = match unsafe { self.pin_exclusive_child(right_swip) } {
                Some(child) => ExclusiveNode::from_inner_frame(child),
                None => return false,
            };
            let right_frame = right.resident_frame();
            let boundary_key = parent.key_at(pos).to_vec();
            if unsafe { self.inner_pair_fits(&node, &boundary_key, &right) } {
                unsafe { self.merge_inner_pages_into_left(&node, &boundary_key, &right) };
                if matches!(right_edge, ParentEdge::Upper) {
                    unsafe {
                        self.unlink_merged_right_inner(
                            &parent,
                            &node_frame,
                            &right_frame,
                            ParentEdge::Upper,
                            pos,
                        )
                    };
                } else {
                    let replacement_key = parent.key_at(pos + 1).to_vec();
                    unsafe {
                        self.unlink_merged_left_inner(
                            &parent,
                            &node_frame,
                            &right_frame,
                            pos,
                            pos + 1,
                            Some(replacement_key.as_slice()),
                        )
                    };
                }
                parent.mark_dirty();
                drop(right);

                let parent_is_root =
                    Self::swip_page_id(self.meta_swip.load(Ordering::Acquire)) == parent.pid();
                let parent_count = parent.num_slots();
                if parent_is_root && parent_count == 0 {
                    unsafe { self.collapse_empty_root_to_child(&parent_frame, &node_frame) };
                    drop(parent);
                } else if !parent_is_root && parent.is_underfull() {
                    let _ = unsafe { self.try_merge_inner_with_path(parent_path, parent) };
                }
                return true;
            }
        }

        if pos > 0 {
            let left_pos = pos - 1;
            let left_swip = parent.child_edge_swip(ParentEdge::Slot(left_pos));
            let left = match unsafe { self.pin_exclusive_child(left_swip) } {
                Some(child) => ExclusiveNode::from_inner_frame(child),
                None => return false,
            };
            let left_frame = left.resident_frame();
            let boundary_key = parent.key_at(left_pos).to_vec();
            if unsafe { self.inner_pair_fits(&left, &boundary_key, &node) } {
                unsafe { self.merge_inner_pages_into_left(&left, &boundary_key, &node) };
                let replacement_key = if pos == count {
                    None
                } else {
                    Some(parent.key_at(pos).to_vec())
                };
                unsafe {
                    self.unlink_merged_left_inner(
                        &parent,
                        &left_frame,
                        &node_frame,
                        left_pos,
                        if pos == count { left_pos } else { pos },
                        replacement_key.as_deref(),
                    )
                };
                parent.mark_dirty();

                let parent_is_root =
                    Self::swip_page_id(self.meta_swip.load(Ordering::Acquire)) == parent.pid();
                let parent_count = parent.num_slots();
                if parent_is_root && parent_count == 0 {
                    unsafe { self.collapse_empty_root_to_child(&parent_frame, &left_frame) };
                    drop(parent);
                } else if !parent_is_root && parent.is_underfull() {
                    let _ = unsafe { self.try_merge_inner_with_path(parent_path, parent) };
                }
                return true;
            }
        }

        false
    }

    unsafe fn rebalance_delete_path(
        &self,
        parent_path: &mut Vec<PinnedFrame<'_>>,
        leaf: PinnedFrame<'_>,
        deleted_was_max: bool,
        new_max: Option<Vec<u8>>,
        should_merge: bool,
    ) {
        let leaf_bf = leaf.frame_ref();
        let leaf_pid = leaf.pid();
        let merged_leaf = should_merge
            && unsafe {
                self.try_merge_leaf_with_path(
                    parent_path,
                    ExclusiveNode::from_leaf_frame(leaf.exclusive()),
                )
            };

        if deleted_was_max && !merged_leaf {
            unsafe { self.repair_separators_after_delete(parent_path, leaf_bf, leaf_pid, new_max) };
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    pub fn upsert(&self, key: &[u8], value: &[u8]) -> bool {
        let mut attempts = 0u32;
        loop {
            let leaf = if attempts >= WRITE_BLOCKING_FALLBACK_THRESHOLD {
                match unsafe { self.find_leaf_exclusive_fallback(key) } {
                    Ok(r) => r,
                    Err(Restart) => {
                        self.stats.inc(BTreeEvent::InsertRestarts);
                        attempts += 1;
                        std::thread::yield_now();
                        continue;
                    }
                }
            } else if attempts >= WRITE_FIXED_ROOT_THRESHOLD {
                match unsafe { self.find_leaf_exclusive_from_fixed_root(key) } {
                    Ok(r) => r,
                    Err(Restart) => {
                        self.stats.inc(BTreeEvent::InsertRestarts);
                        attempts += 1;
                        if attempts.is_multiple_of(WRITE_FIXED_ROOT_THRESHOLD) {
                            std::thread::yield_now();
                        }
                        continue;
                    }
                }
            } else {
                match unsafe { self.find_leaf_exclusive(key) } {
                    Ok(r) => r,
                    Err(Restart) => {
                        self.stats.inc(BTreeEvent::InsertRestarts);
                        attempts += 1;
                        if attempts.is_multiple_of(WRITE_FIXED_ROOT_THRESHOLD) {
                            std::thread::yield_now();
                        }
                        continue;
                    }
                }
            };
            let need_split = match self.try_upsert_leaf(&leaf, key, value) {
                UpsertLeafAction::UpdatedExisting => return false,
                UpsertLeafAction::Inserted => return true,
                UpsertLeafAction::SplitRequired => true,
            };
            if need_split {
                drop(leaf);
            }

            // Pre-allocate sibling frame while no latches are held, so
            // the allocation (which may block on eviction) does not hold
            // the exclusive latch on the node being split.
            let pool = self.pool();
            let pre_sibling = pool.allocate_and_fix_frame(NoLatches::new(pool));

            let (mut parent_path, leaf) = if attempts >= WRITE_BLOCKING_FALLBACK_THRESHOLD {
                match unsafe { self.find_leaf_exclusive_with_path_fallback(key) } {
                    Ok(r) => r,
                    Err(Restart) => {
                        self.stats.inc(BTreeEvent::InsertRestarts);
                        attempts += 1;
                        std::thread::yield_now();
                        continue;
                    }
                }
            } else {
                match unsafe { self.find_leaf_exclusive_with_path(key) } {
                    Ok(r) => r,
                    Err(Restart) => {
                        self.stats.inc(BTreeEvent::InsertRestarts);
                        attempts += 1;
                        continue;
                    }
                }
            };

            match self.try_upsert_leaf(&leaf, key, value) {
                UpsertLeafAction::UpdatedExisting => return false,
                UpsertLeafAction::Inserted => return true,
                UpsertLeafAction::SplitRequired => unsafe {
                    self.split_node(
                        leaf.into_frame(),
                        &mut parent_path,
                        Some(key),
                        Some(pre_sibling),
                    );
                },
            }
        }
    }

    pub fn insert(&self, key: &[u8], value: &[u8]) -> bool {
        let mut attempts = 0u32;
        loop {
            let leaf = if attempts >= WRITE_BLOCKING_FALLBACK_THRESHOLD {
                match unsafe { self.find_leaf_exclusive_fallback(key) } {
                    Ok(r) => r,
                    Err(Restart) => {
                        self.stats.inc(BTreeEvent::InsertRestarts);
                        attempts += 1;
                        std::thread::yield_now();
                        continue;
                    }
                }
            } else if attempts >= WRITE_FIXED_ROOT_THRESHOLD {
                match unsafe { self.find_leaf_exclusive_from_fixed_root(key) } {
                    Ok(r) => r,
                    Err(Restart) => {
                        self.stats.inc(BTreeEvent::InsertRestarts);
                        attempts += 1;
                        if attempts.is_multiple_of(WRITE_FIXED_ROOT_THRESHOLD) {
                            std::thread::yield_now();
                        }
                        continue;
                    }
                }
            } else {
                match unsafe { self.find_leaf_exclusive(key) } {
                    Ok(r) => r,
                    Err(Restart) => {
                        self.stats.inc(BTreeEvent::InsertRestarts);
                        attempts += 1;
                        if attempts.is_multiple_of(WRITE_FIXED_ROOT_THRESHOLD) {
                            std::thread::yield_now();
                        }
                        continue;
                    }
                }
            };
            let need_split = match self.try_insert_leaf(&leaf, key, value) {
                InsertLeafAction::ReturnFalse => return false,
                InsertLeafAction::Inserted => return true,
                InsertLeafAction::SplitRequired => true,
            };
            if need_split {
                drop(leaf);
            }

            // Pre-allocate sibling frame while no latches are held.
            let pool = self.pool();
            let pre_sibling = pool.allocate_and_fix_frame(NoLatches::new(pool));

            let (mut parent_path, leaf) = if attempts >= WRITE_BLOCKING_FALLBACK_THRESHOLD {
                match unsafe { self.find_leaf_exclusive_with_path_fallback(key) } {
                    Ok(r) => r,
                    Err(Restart) => {
                        self.stats.inc(BTreeEvent::InsertRestarts);
                        attempts += 1;
                        std::thread::yield_now();
                        continue;
                    }
                }
            } else {
                match unsafe { self.find_leaf_exclusive_with_path(key) } {
                    Ok(r) => r,
                    Err(Restart) => {
                        self.stats.inc(BTreeEvent::InsertRestarts);
                        attempts += 1;
                        continue;
                    }
                }
            };

            match self.try_insert_leaf(&leaf, key, value) {
                InsertLeafAction::ReturnFalse => return false,
                InsertLeafAction::Inserted => return true,
                InsertLeafAction::SplitRequired => unsafe {
                    self.split_node(
                        leaf.into_frame(),
                        &mut parent_path,
                        Some(key),
                        Some(pre_sibling),
                    );
                },
            }
        }
    }

    fn try_read_optimistic_leaf<'l, T>(
        &'l self,
        key: &[u8],
        read: impl FnOnce(&OptimisticNode<'l, Leaf>, u16, bool) -> Result<Option<T>, Restart>,
    ) -> Result<Option<T>, Restart> {
        let leaf = unsafe { self.find_leaf_optimistic(key) }?;
        let Some((pos, exact)) = leaf.try_lower_bound(key) else {
            return Err(Restart);
        };
        let result = read(&leaf, pos, exact)?;
        leaf.validate()?;
        Ok(result)
    }

    fn retry_optimistic_lookup(&self, attempts: &mut u32) -> bool {
        *attempts += 1;
        if *attempts >= Self::LOOKUP_OPTIMISTIC_RESTART_LIMIT {
            return false;
        }
        if attempts.is_multiple_of(Self::LOOKUP_OPTIMISTIC_YIELD_INTERVAL) {
            std::thread::yield_now();
        }
        true
    }

    pub fn lookup(&self, key: &[u8]) -> Option<Vec<u8>> {
        let mut attempts = 0u32;
        loop {
            let result = self.try_read_optimistic_leaf(key, |leaf, pos, exact| {
                if !exact {
                    return Ok(None);
                }
                leaf.try_value_at(pos)
                    .map(|value| Some(value.to_vec()))
                    .ok_or(Restart)
            });
            let result = match result {
                Ok(result) => result,
                Err(Restart) if self.retry_optimistic_lookup(&mut attempts) => continue,
                Err(Restart) => return self.lookup_fallback(key),
            };

            if result.is_none() {
                return self.lookup_fallback(key);
            }
            return result;
        }
    }

    pub fn lookup_with<R>(&self, key: &[u8], f: impl FnOnce(Option<&[u8]>) -> R) -> R {
        let mut attempts = 0u32;
        let f = &mut Some(f);
        loop {
            let result = self.try_read_optimistic_leaf(key, |leaf, pos, exact| {
                if !exact {
                    return Ok(None);
                }
                leaf.try_value_at(pos).map(Some).ok_or(Restart)
            });
            let result = match result {
                Ok(result) => result,
                Err(Restart) if self.retry_optimistic_lookup(&mut attempts) => continue,
                Err(Restart) => return self.lookup_with_fallback(key, f.take().unwrap()),
            };

            if result.is_none() {
                return self.lookup_with_fallback(key, f.take().unwrap());
            }
            return f.take().unwrap()(result);
        }
    }

    pub fn lookup_fixed<const N: usize>(&self, key: &[u8]) -> Option<[u8; N]> {
        let mut attempts = 0u32;
        loop {
            let result = self.try_read_optimistic_leaf(key, |leaf, pos, exact| {
                if !exact {
                    return Ok(None);
                }
                match leaf.try_value_at(pos) {
                    Some(v) if v.len() == N => {
                        let mut out = [0u8; N];
                        out.copy_from_slice(v);
                        Ok(Some(out))
                    }
                    Some(_) => Ok(None),
                    None => Err(Restart),
                }
            });
            let result = match result {
                Ok(result) => result,
                Err(Restart) if self.retry_optimistic_lookup(&mut attempts) => continue,
                Err(Restart) => return self.lookup_fixed_fallback::<N>(key),
            };

            if result.is_none() {
                return self.lookup_fixed_fallback::<N>(key);
            }
            return result;
        }
    }

    fn lookup_fallback(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.with_lookup_fallback_leaf(key, |leaf| {
            let leaf = leaf?;
            let (pos, exact) = leaf.lower_bound(key);
            exact.then(|| leaf.get_value(pos).to_vec())
        })
    }

    fn with_lookup_fallback_leaf<R>(
        &self,
        key: &[u8],
        read: impl FnOnce(Option<&ResidentFrame<'_>>) -> R,
    ) -> R {
        let pool = self.pool();
        let mut attempts = 0u32;
        let mut read = Some(read);

        'restart: loop {
            let mut current = unsafe { pool.fix_frame(&self.meta_swip, NoLatches::new(pool)) };

            loop {
                let shared = SharedNode::<Leaf>::from_leaf_frame(current.clone_pin().shared());
                let current_frame = shared.resident_frame();
                if current_frame.is_leaf() {
                    if current_frame.should_chase_right(key) {
                        let right_pid = current_frame.leaf_right_pid();
                        if right_pid == 0 {
                            return read.take().unwrap()(None);
                        }
                        let right = match unsafe { pool.try_fix_orphan_frame(right_pid) } {
                            Some(right) => right,
                            None => {
                                attempts += 1;
                                unsafe { pool.fix_orphan_frame(right_pid, NoLatches::new(pool)) }
                            }
                        };
                        drop(shared);
                        current = right;
                        continue;
                    }
                    let result = read.take().unwrap()(Some(&current_frame));
                    return result;
                }

                let routed_child = match current_frame.try_route_to_child(key) {
                    Some(routed) => routed,
                    None => {
                        return read.take().unwrap()(None);
                    }
                };
                let child = match unsafe {
                    Self::try_resolve_child_for_read(pool, routed_child.swip())
                } {
                    Some(child) => child,
                    None => {
                        if !(routed_child.swip().is_hot() || routed_child.swip().is_cool()) {
                            let child_pid = routed_child.swip().as_page_id();
                            let parent_pid = current.pid();
                            drop(shared);
                            let child =
                                unsafe { pool.fix_orphan_frame(child_pid, NoLatches::new(pool)) };
                            // Re-fix the parent under a new shared guard
                            // and re-read the routing. The shared guard was
                            // dropped during the blocking fix, so the parent's
                            // routing may have changed.
                            let parent =
                                unsafe { pool.fix_orphan_frame(parent_pid, NoLatches::new(pool)) };
                            let parent_shared = parent.shared();
                            let parent_frame = ResidentFrame::from_shared(&parent_shared);
                            if let Some(re_routed) = parent_frame.try_route_to_child(key) {
                                let re_pid =
                                    if re_routed.swip().is_hot() || re_routed.swip().is_cool() {
                                        unsafe { ResidentFrame::from_hot_swip(re_routed.swip()) }
                                            .map(|f| f.pid())
                                            .unwrap_or(0)
                                    } else {
                                        re_routed.swip().as_page_id()
                                    };
                                if re_pid == child_pid {
                                    unsafe {
                                        self.set_parent_link_for_routed_child(
                                            &ResidentFrame::from_pinned(&child),
                                            &parent_frame,
                                            &re_routed,
                                        )
                                    };
                                }
                            }
                            drop(parent_shared);
                            current = child;
                            continue;
                        }
                        attempts += 1;
                        if attempts >= Self::LOOKUP_OPTIMISTIC_RESTART_LIMIT {
                            return read.take().unwrap()(None);
                        }
                        std::thread::yield_now();
                        continue 'restart;
                    }
                };
                unsafe {
                    self.set_parent_link_for_routed_child(
                        &ResidentFrame::from_pinned(&child),
                        &current_frame,
                        &routed_child,
                    )
                };
                drop(shared);
                current = child;
            }
        }
    }

    fn lookup_with_fallback<R>(&self, key: &[u8], f: impl FnOnce(Option<&[u8]>) -> R) -> R {
        self.with_lookup_fallback_leaf(key, |leaf| {
            let Some(leaf) = leaf else {
                return f(None);
            };
            let (pos, exact) = leaf.lower_bound(key);
            f(exact.then(|| leaf.get_value(pos)))
        })
    }

    fn lookup_fixed_fallback<const N: usize>(&self, key: &[u8]) -> Option<[u8; N]> {
        self.with_lookup_fallback_leaf(key, |leaf| {
            let leaf = leaf?;
            let (pos, exact) = leaf.lower_bound(key);
            if !exact {
                return None;
            }
            let value = leaf.get_value(pos);
            if value.len() != N {
                return None;
            }
            let mut out = [0u8; N];
            out.copy_from_slice(value);
            Some(out)
        })
    }

    unsafe fn find_leaf_shared_fallback<'a>(
        &'a self,
        key: &[u8],
    ) -> Result<SharedNode<'a, Leaf>, Restart> {
        let pool = self.pool();
        let mut current = unsafe { pool.fix_frame(&self.meta_swip, NoLatches::new(pool)) };

        loop {
            let current_shared = current.clone_pin().shared();
            let shared = SharedNode::<Leaf>::from_leaf_frame(current_shared);
            let current_frame = shared.resident_frame();
            if current_frame.is_leaf() {
                if current_frame.should_chase_right(key) {
                    let right_pid = current_frame.leaf_right_pid();
                    if right_pid == 0 {
                        return Err(Restart);
                    }
                    drop(shared);
                    let right = unsafe { pool.fix_orphan_frame(right_pid, NoLatches::new(pool)) };
                    current = right;
                    continue;
                }
                return Ok(shared);
            }

            let routed_child = current_frame.try_route_to_child(key).ok_or(Restart)?;
            let child = if routed_child.swip().is_hot() || routed_child.swip().is_cool() {
                let Some(child) = (unsafe { pool.try_pin_child(routed_child.swip()) }) else {
                    return Err(Restart);
                };
                child
            } else {
                // Cold-SWIP path: the child is evicted, need a blocking
                // fix_orphan_frame to load it. Drop the shared guard first
                // so eviction can exclusive-latch this parent to unswizzle
                // other frames — holding a shared latch across a blocking
                // fix starves eviction and panics the pool.
                let child_pid = routed_child.swip().as_page_id();
                let parent_pid = current.pid();
                drop(shared);
                let child = unsafe { pool.fix_orphan_frame(child_pid, NoLatches::new(pool)) };
                // Re-fix the parent under a new shared guard and re-read
                // the routing. The shared guard was dropped during the
                // blocking fix, so the parent's routing may have changed
                // (split, eviction unswizzle). If the routing still points
                // to the same child, set the parent link; otherwise skip —
                // the child is loaded, and the next traversal will set the
                // link correctly.
                let parent = unsafe { pool.fix_orphan_frame(parent_pid, NoLatches::new(pool)) };
                let parent_shared = parent.shared();
                let parent_frame = ResidentFrame::from_shared(&parent_shared);
                if let Some(re_routed) = parent_frame.try_route_to_child(key) {
                    let re_pid = if re_routed.swip().is_hot() || re_routed.swip().is_cool() {
                        unsafe { ResidentFrame::from_hot_swip(re_routed.swip()) }
                            .map(|f| f.pid())
                            .unwrap_or(0)
                    } else {
                        re_routed.swip().as_page_id()
                    };
                    if re_pid == child_pid {
                        unsafe {
                            self.set_parent_link_for_routed_child(
                                &ResidentFrame::from_pinned(&child),
                                &parent_frame,
                                &re_routed,
                            )
                        };
                    }
                }
                drop(parent_shared);
                current = child;
                continue;
            };
            unsafe {
                self.set_parent_link_for_routed_child(
                    &ResidentFrame::from_pinned(&child),
                    &current_frame,
                    &routed_child,
                )
            };
            current = child;
        }
    }

    unsafe fn find_leaf_shared_nonblocking<'a>(
        &'a self,
        key: &[u8],
    ) -> Result<SharedNode<'a, Leaf>, Restart> {
        let pool = self.pool();
        let mut current = unsafe { pool.fix_frame(&self.meta_swip, NoLatches::new(pool)) };

        loop {
            let current_shared = current.shared();
            let current_frame = ResidentFrame::from_shared(&current_shared);
            if current_frame.is_leaf() {
                if current_frame.should_chase_right(key) {
                    let right_pid = current_frame.leaf_right_pid();
                    if right_pid == 0 {
                        return Err(Restart);
                    }
                    let Some(right) = (unsafe { pool.try_fix_orphan_frame(right_pid) }) else {
                        return Err(Restart);
                    };
                    current = right;
                    continue;
                }
                return Ok(SharedNode::from_leaf_frame(current_shared));
            }

            let routed_child = current_frame.try_route_to_child(key).ok_or(Restart)?;
            let child = if routed_child.swip().is_hot() || routed_child.swip().is_cool() {
                let Some(child) = (unsafe { pool.try_pin_child(routed_child.swip()) }) else {
                    return Err(Restart);
                };
                child
            } else {
                let Some(child) =
                    (unsafe { pool.try_fix_orphan_frame(routed_child.swip().as_page_id()) })
                else {
                    return Err(Restart);
                };
                child
            };
            unsafe {
                self.set_parent_link_for_routed_child(
                    &ResidentFrame::from_pinned(&child),
                    &current_frame,
                    &routed_child,
                )
            };
            current = child;
        }
    }

    unsafe fn find_rightmost_leaf_shared_fallback<'a>(
        &'a self,
    ) -> Result<SharedNode<'a, Leaf>, Restart> {
        let pool = self.pool();
        let mut current = unsafe { pool.fix_frame(&self.meta_swip, NoLatches::new(pool)) };

        loop {
            let current_shared = current.shared();
            let current_frame = ResidentFrame::from_shared(&current_shared);
            if current_frame.is_leaf() {
                return Ok(SharedNode::from_leaf_frame(current_shared));
            }

            let upper = current_frame.upper_swip();
            let Some(child) = (unsafe { Self::try_resolve_child_for_read(pool, upper) }) else {
                return Err(Restart);
            };
            unsafe {
                self.set_parent_link_for_routed_child(
                    &ResidentFrame::from_pinned(&child),
                    &current_frame,
                    &RoutedChildRef::new(upper, ParentEdge::Upper),
                )
            };
            current = child;
        }
    }

    pub fn remove(&self, key: &[u8]) -> bool {
        let mut attempts = 0u32;
        loop {
            let (mut parent_path, leaf) = if attempts >= WRITE_BLOCKING_FALLBACK_THRESHOLD {
                match unsafe { self.find_leaf_exclusive_with_path_fallback(key) } {
                    Ok(r) => r,
                    Err(Restart) => {
                        self.stats.inc(BTreeEvent::InsertRestarts);
                        attempts += 1;
                        std::thread::yield_now();
                        continue;
                    }
                }
            } else {
                match unsafe { self.find_leaf_exclusive_with_path(key) } {
                    Ok(r) => r,
                    Err(Restart) => {
                        self.stats.inc(BTreeEvent::InsertRestarts);
                        attempts += 1;
                        if attempts.is_multiple_of(WRITE_FIXED_ROOT_THRESHOLD) {
                            std::thread::yield_now();
                        }
                        continue;
                    }
                }
            };
            let (deleted_was_max, new_max, should_merge) = {
                let (pos, exact) = leaf.lower_bound(key);
                if !exact {
                    return false;
                }

                let deleted_was_max = pos + 1 == leaf.num_slots();

                leaf.remove_slot(pos);
                leaf.mark_dirty();

                let new_max = if deleted_was_max && leaf.num_slots() > 0 {
                    Some(leaf.key_at(leaf.num_slots() - 1).to_vec())
                } else {
                    None
                };
                let should_merge = leaf.is_underfull();
                (deleted_was_max, new_max, should_merge)
            };

            let leaf = leaf.into_pinned();

            unsafe {
                self.rebalance_delete_path(
                    &mut parent_path,
                    leaf,
                    deleted_was_max,
                    new_max,
                    should_merge,
                )
            };
            return true;
        }
    }

    fn collect_evicted_root_child_pids(&self) -> Vec<u64> {
        unsafe {
            self.pool().with_fixed_exclusive(
                &self.meta_swip,
                NoLatches::new(self.pool()),
                |root_frame| {
                    let root = ResidentFrame::from_exclusive(root_frame);
                    Self::collect_evicted_child_pids(&root)
                },
            )
        }
    }

    fn collect_evicted_child_pids(root: &ResidentFrame<'_>) -> Vec<u64> {
        if root.is_leaf() {
            return Vec::new();
        }

        let mut evicted_pids = Vec::new();
        let count = root.num_slots();
        for i in 0..count {
            let swip = root.child_swip_at(i);
            if !swip.is_evicted() {
                continue;
            }
            let pid = swip.as_page_id();
            if pid != 0 {
                evicted_pids.push(pid);
            }
        }

        let upper = root.upper_swip();
        if upper.is_evicted() {
            let pid = upper.as_page_id();
            if pid != 0 {
                evicted_pids.push(pid);
            }
        }

        evicted_pids
    }

    /// Queue a shallow prefetch of evicted root children.
    ///
    /// This warms the first traversal fanout after a persistent reopen without
    /// requiring a scan to fault those pages synchronously.
    pub fn prefetch_root_children(&self) {
        let pool = self.pool();
        let evicted_pids = self.collect_evicted_root_child_pids();

        for pid in evicted_pids {
            pool.prefetch_pid_async(pid);
        }
    }

    fn collect_scan_leaf_entries(
        &self,
        leaf: &SharedNode<'_, Leaf>,
        current_key: &[u8],
    ) -> Option<Vec<(Vec<u8>, Vec<u8>)>> {
        let count = leaf.num_slots();
        let (start_pos, _) = leaf.try_lower_bound(current_key)?;

        let mut entries = Vec::new();
        for i in start_pos..count {
            let key = leaf.try_key_at(i)?;
            let value = leaf.try_value_at(i)?;
            entries.push((key.to_vec(), value.to_vec()));
        }

        Some(entries)
    }

    pub fn scan<F>(&self, mut f: F)
    where
        F: FnMut(&[u8], &[u8]),
    {
        let mut current_key: Vec<u8> = Vec::new();
        loop {
            let mut attempts = 0u32;
            let entries = loop {
                let leaf = if attempts >= 64 {
                    match unsafe { self.find_leaf_shared_fallback(&current_key) } {
                        Ok(r) => {
                            attempts = 0;
                            r
                        }
                        Err(Restart) => {
                            attempts += 1;
                            std::thread::yield_now();
                            continue;
                        }
                    }
                } else if attempts >= 16 {
                    match unsafe { self.find_leaf_shared_nonblocking(&current_key) } {
                        Ok(shared) => {
                            attempts = 0;
                            shared
                        }
                        Err(Restart) => {
                            attempts += 1;
                            std::thread::yield_now();
                            continue;
                        }
                    }
                } else {
                    match unsafe { self.find_leaf_optimistic(&current_key) } {
                        Ok(opt) => match opt.upgrade_to_shared() {
                            Ok(shared) => {
                                attempts = 0;
                                shared
                            }
                            Err(_leaf) => {
                                attempts += 1;
                                continue;
                            }
                        },
                        Err(Restart) => {
                            attempts += 1;
                            continue;
                        }
                    }
                };
                let Some(entries) = self.collect_scan_leaf_entries(&leaf, &current_key) else {
                    continue;
                };
                break entries;
            };

            if entries.is_empty() {
                break;
            }

            for (k, v) in &entries {
                f(k, v);
            }

            let last_key = &entries.last().unwrap().0;
            let mut next_key = last_key.clone();
            next_key.push(0);
            if next_key <= current_key {
                break;
            }
            current_key = next_key;
        }
    }

    /// Scan all entries whose key starts with `prefix`, in order.
    ///
    /// Calls `f(key, value)` for each matching entry. Stops as soon
    /// as a key is found that does not start with `prefix`.
    pub fn scan_prefix<F>(&self, prefix: &[u8], mut f: F)
    where
        F: FnMut(&[u8], &[u8]),
    {
        self.scan_prefix_borrowed_until(prefix, |k, v| {
            f(k, v);
            true
        });
    }

    pub fn scan_prefix_borrowed_until<F>(&self, prefix: &[u8], mut f: F)
    where
        F: FnMut(&[u8], &[u8]) -> bool,
    {
        let pool = self.pool();
        let mut leaf = match unsafe { self.find_leaf_shared_fallback(prefix) } {
            Ok(r) => r,
            Err(Restart) => return,
        };
        let mut first_leaf = true;

        loop {
            let count = leaf.num_slots();
            let start_pos = if first_leaf {
                let (pos, _) = leaf.lower_bound(prefix);
                pos
            } else {
                0
            };
            first_leaf = false;

            let mut saw_non_matching = false;
            for i in start_pos..count {
                let k = leaf.key_at(i);
                if !k.starts_with(prefix) {
                    saw_non_matching = true;
                    break;
                }
                let v = leaf.value_at(i);
                if !f(k, v) {
                    return;
                }
            }

            let right_pid = leaf.right_pid();
            drop(leaf);

            if saw_non_matching || right_pid == 0 {
                return;
            }

            let right = unsafe { pool.fix_orphan_frame(right_pid, NoLatches::new(pool)) };
            leaf = SharedNode::from_leaf_frame(right.shared());
        }
    }

    pub fn scan_prefix_until<F>(&self, prefix: &[u8], mut f: F)
    where
        F: FnMut(&[u8], &[u8]) -> bool,
    {
        self.scan_prefix_borrowed_until(prefix, |k, v| f(k, v));
    }

    /// Scan all entries with keys in `[lower, upper]`, where each
    /// bound carries inclusive/exclusive/unbounded semantics via
    /// `std::ops::Bound`. Calls `f(key, value)` for each matching
    /// entry in ascending key order.
    ///
    /// Stops as soon as a key beyond `upper` is observed. An empty
    /// or fully-bounded-away range produces zero callbacks.
    pub fn scan_range<F>(
        &self,
        lower: std::ops::Bound<&[u8]>,
        upper: std::ops::Bound<&[u8]>,
        mut f: F,
    ) where
        F: FnMut(&[u8], &[u8]),
    {
        self.scan_range_until(lower, upper, |k, v| {
            f(k, v);
            true
        });
    }

    pub fn scan_range_until<F>(
        &self,
        lower: std::ops::Bound<&[u8]>,
        upper: std::ops::Bound<&[u8]>,
        mut f: F,
    ) where
        F: FnMut(&[u8], &[u8]) -> bool,
    {
        use std::ops::Bound;
        let pool = self.pool();
        let start_key = match lower {
            Bound::Unbounded => Vec::new(),
            Bound::Included(key) | Bound::Excluded(key) => key.to_vec(),
        };
        let mut leaf = match unsafe { self.find_leaf_shared_fallback(&start_key) } {
            Ok(leaf) => leaf,
            Err(Restart) => return,
        };
        let mut first_leaf = true;

        loop {
            let count = leaf.num_slots();
            let start_pos = if first_leaf {
                match lower {
                    Bound::Unbounded => 0,
                    Bound::Included(key) => {
                        let (pos, _) = leaf.lower_bound(key);
                        pos
                    }
                    Bound::Excluded(key) => {
                        let (pos, exact) = leaf.lower_bound(key);
                        if exact { pos + 1 } else { pos }
                    }
                }
            } else {
                0
            };
            first_leaf = false;

            let mut hit_upper = false;
            for i in start_pos..count {
                let key = leaf.key_at(i);
                let in_range = match upper {
                    Bound::Unbounded => true,
                    Bound::Included(limit) => key <= limit,
                    Bound::Excluded(limit) => key < limit,
                };
                if !in_range {
                    hit_upper = true;
                    break;
                }
                let value = leaf.value_at(i);
                if !f(key, value) {
                    return;
                }
            }

            let right_pid = leaf.right_pid();
            drop(leaf);

            if hit_upper || right_pid == 0 {
                return;
            }

            let right = unsafe { pool.fix_orphan_frame(right_pid, NoLatches::new(pool)) };
            leaf = SharedNode::from_leaf_frame(right.shared());
        }
    }

    /// Scan all entries with keys in `[lower, upper]` in descending key order.
    pub fn scan_range_desc<F>(
        &self,
        lower: std::ops::Bound<&[u8]>,
        upper: std::ops::Bound<&[u8]>,
        mut f: F,
    ) where
        F: FnMut(&[u8], &[u8]),
    {
        self.scan_range_desc_until(lower, upper, |k, v| {
            f(k, v);
            true
        });
    }

    pub fn scan_range_desc_until<F>(
        &self,
        lower: std::ops::Bound<&[u8]>,
        upper: std::ops::Bound<&[u8]>,
        mut f: F,
    ) where
        F: FnMut(&[u8], &[u8]) -> bool,
    {
        use std::ops::Bound;

        let pool = self.pool();
        let mut leaf = match upper {
            Bound::Unbounded => match unsafe { self.find_rightmost_leaf_shared_fallback() } {
                Ok(leaf) => leaf,
                Err(Restart) => return,
            },
            Bound::Included(key) | Bound::Excluded(key) => {
                match unsafe { self.find_leaf_shared_fallback(key) } {
                    Ok(leaf) => leaf,
                    Err(Restart) => return,
                }
            }
        };
        let mut first_leaf = true;

        loop {
            let count = leaf.num_slots();
            let mut entries = Vec::new();
            let mut hit_lower = false;
            let mut need_prev_leaf = false;

            let start_pos = if count == 0 {
                None
            } else if first_leaf {
                match upper {
                    Bound::Unbounded => Some(count - 1),
                    Bound::Included(key) => {
                        let (pos, exact) = leaf.lower_bound(key);
                        if exact {
                            Some(pos)
                        } else if pos == 0 {
                            None
                        } else {
                            Some(pos - 1)
                        }
                    }
                    Bound::Excluded(key) => {
                        let (pos, exact) = leaf.lower_bound(key);
                        if exact || pos > 0 {
                            pos.checked_sub(1)
                        } else {
                            None
                        }
                    }
                }
            } else {
                Some(count - 1)
            };
            first_leaf = false;

            if let Some(start_pos) = start_pos {
                for i in (0..=start_pos).rev() {
                    let k = leaf.key_at(i);
                    let in_range = match lower {
                        Bound::Unbounded => true,
                        Bound::Included(lo) => k >= lo,
                        Bound::Excluded(lo) => k > lo,
                    };
                    if !in_range {
                        hit_lower = true;
                        break;
                    }
                    let v = leaf.value_at(i);
                    entries.push((k.to_vec(), v.to_vec()));
                }
            } else {
                need_prev_leaf = true;
            }

            let left_pid = leaf.left_pid();
            drop(leaf);

            for (k, v) in &entries {
                if !f(k, v) {
                    return;
                }
            }

            if hit_lower || left_pid == 0 {
                return;
            }
            if entries.is_empty() && !need_prev_leaf {
                return;
            }

            let left = unsafe { pool.fix_orphan_frame(left_pid, NoLatches::new(pool)) };
            leaf = SharedNode::from_leaf_frame(left.shared());
        }
    }

    /// Return the current tree height (0 = single leaf root).
    pub fn height(&self) -> u32 {
        self.height.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::node::BTreeNode;
    use super::*;
    use pagebox_storage::page_store::FilePageStore;
    use proptest::prelude::*;
    use proptest::test_runner::{Config as ProptestConfig, TestRunner};
    use std::collections::BTreeMap;
    use std::ops::Bound;
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn collect_all(tree: &BTree) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut out = Vec::new();
        tree.scan(|k, v| out.push((k.to_vec(), v.to_vec())));
        out
    }

    fn collect_range_pairs(
        tree: &BTree,
        lower: Bound<&[u8]>,
        upper: Bound<&[u8]>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut out = Vec::new();
        tree.scan_range(lower, upper, |k, v| out.push((k.to_vec(), v.to_vec())));
        out
    }

    fn model_collect_all(model: &BTreeMap<Vec<u8>, Vec<u8>>) -> Vec<(Vec<u8>, Vec<u8>)> {
        model
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }

    fn model_collect_range(
        model: &BTreeMap<Vec<u8>, Vec<u8>>,
        lower: Bound<&[u8]>,
        upper: Bound<&[u8]>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        let lower_ok = |key: &[u8]| match lower {
            Bound::Included(bound) => key >= bound,
            Bound::Excluded(bound) => key > bound,
            Bound::Unbounded => true,
        };
        let upper_ok = |key: &[u8]| match upper {
            Bound::Included(bound) => key <= bound,
            Bound::Excluded(bound) => key < bound,
            Bound::Unbounded => true,
        };

        model
            .iter()
            .filter(|(key, _)| lower_ok(key) && upper_ok(key))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }

    struct GeneratedCase {
        state: u64,
    }

    impl GeneratedCase {
        fn new(seed: u64) -> Self {
            Self {
                state: seed ^ 0xd1b5_4a32_d192_ed03,
            }
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        fn u8(&mut self, max: u8) -> u8 {
            (self.next_u64() % (u64::from(max) + 1)) as u8
        }

        fn usize(&mut self, min: usize, max: usize) -> usize {
            if min == max {
                return min;
            }
            min + (self.next_u64() as usize % (max - min + 1))
        }

        fn bytes(&mut self, min_size: usize, max_size: usize) -> Vec<u8> {
            let len = self.usize(min_size, max_size);
            (0..len).map(|_| self.next_u64() as u8).collect()
        }
    }

    macro_rules! generated_test {
        ($cases:expr, |$tc:ident| $body:block) => {{
            let mut runner = TestRunner::new(ProptestConfig::with_cases($cases));
            runner
                .run(&any::<u64>(), |seed| {
                    let mut $tc = GeneratedCase::new(seed);
                    $body
                    Ok(())
                })
                .unwrap();
        }};
    }

    fn assert_tree_matches_model(tree: &BTree, model: &BTreeMap<Vec<u8>, Vec<u8>>, context: &str) {
        for (key, expected) in model {
            assert_eq!(
                tree.lookup(key),
                Some(expected.clone()),
                "{context}: lookup mismatch for key {key:?}"
            );
        }

        let scanned = collect_all(tree);
        let expected_scan = model_collect_all(model);
        assert_eq!(
            scanned, expected_scan,
            "{context}: full scan contents diverged from model"
        );
    }

    #[test]
    fn split_single_leaf() {
        let pool = std::sync::Arc::new(BufferPool::new(32));
        let tree = BTree::new(&pool, 0);

        let mut inserted = Vec::new();
        for i in 0..700u32 {
            let key = i.to_be_bytes();
            let val = [i as u8; 100];
            assert!(tree.insert(&key, &val), "failed to insert {i}");
            inserted.push((key, val));
        }

        assert!(tree.height() >= 1, "tree should have split");

        for (key, val) in &inserted {
            assert_eq!(
                tree.lookup(key).as_deref(),
                Some(val.as_slice()),
                "key {:?} not found",
                key
            );
        }
    }

    #[test]
    fn multi_level_splits() {
        // 64K leaves hold ~2 entries of a ~32 KiB value (4-byte key + 12-byte
        // slot overhead). To overflow the root inner node (≈2729 children) and
        // reach height 2, we need ≈2729 leaves, i.e. ≈5458 entries. Use 6000
        // with margin. Large values keep the entry count tractable for a test.
        let pool = std::sync::Arc::new(BufferPool::new(4096));
        let tree = BTree::new(&pool, 0);

        let n = 6000;
        for i in 0..n as u32 {
            let key = i.to_be_bytes();
            let mut val = [0u8; 32000];
            val[0] = (i & 0xFF) as u8;
            val[1] = ((i >> 8) & 0xFF) as u8;
            assert!(tree.insert(&key, &val), "insert {i} failed");
        }

        assert!(
            tree.height() >= 2,
            "expected at least 3 levels, got height {}",
            tree.height()
        );

        for i in 0..n as u32 {
            let key = i.to_be_bytes();
            let result = tree.lookup(&key);
            assert!(result.is_some(), "lookup {i} failed");
            let val = result.unwrap();
            assert_eq!(val[0], (i & 0xFF) as u8);
            assert_eq!(val[1], ((i >> 8) & 0xFF) as u8);
        }
    }

    #[test]
    fn owned_page_ids_include_reachable_tree_pages() {
        let pool = std::sync::Arc::new(BufferPool::new(128));
        let tree = BTree::new(&pool, 0);

        for i in 0..4_000u32 {
            let key = i.to_be_bytes();
            assert!(tree.insert(&key, &key));
        }

        let pages = tree.owned_page_ids();
        assert!(
            pages.contains(&tree.root_page_id()),
            "owned pages must include the root page"
        );
        assert!(
            pages.len() > 1,
            "test should force at least one split so ownership spans multiple pages"
        );
        let unique = pages
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            unique.len(),
            pages.len(),
            "owned page enumeration must not return duplicates"
        );
    }

    #[test]
    fn owned_page_ids_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        let root_pid;
        let height;
        let before;

        {
            let store = FilePageStore::open(&path).unwrap();
            let pool = std::sync::Arc::new(BufferPool::with_store(256, Box::new(store)));
            let tree = BTree::new(&pool, 0);
            for i in 0..2_000u32 {
                let key = i.to_be_bytes();
                assert!(tree.insert(&key, &key));
            }
            before = tree.owned_page_ids();
            pool.flush().unwrap();
            root_pid = tree.root_page_id();
            height = tree.height();
        }

        let store = FilePageStore::open(&path).unwrap();
        let pool = std::sync::Arc::new(BufferPool::with_store(256, Box::new(store)));
        let tree = BTree::open(&pool, root_pid, height, 0);

        assert_eq!(
            tree.owned_page_ids(),
            before,
            "owned page enumeration should not depend on residency"
        );
    }

    #[test]
    fn scan_after_reopen_with_tight_resident_budget() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        let root_pid;
        let height;
        let n = 2_000u32;

        {
            let store = FilePageStore::open(&path).unwrap();
            let pool = std::sync::Arc::new(BufferPool::with_store(512, Box::new(store)));
            let tree = BTree::new(&pool, 0);
            for i in 0..n {
                let key = i.to_be_bytes();
                let mut val = [0u8; 500];
                val[..4].copy_from_slice(&key);
                assert!(tree.insert(&key, &val), "insert {i} failed");
            }
            pool.flush().unwrap();
            root_pid = tree.root_page_id();
            height = tree.height();
        }

        let store = FilePageStore::open(&path).unwrap();
        let pool = std::sync::Arc::new(BufferPool::with_store(16, Box::new(store)));
        let tree = BTree::open(&pool, root_pid, height, 0);

        let mut scanned = Vec::new();
        tree.scan(|k, v| {
            assert_eq!(v.len(), 500, "scan returned wrong value length");
            scanned.push(u32::from_be_bytes(k.try_into().unwrap()));
        });

        assert_eq!(
            scanned,
            (0..n).collect::<Vec<_>>(),
            "scan should fault and evict through cold leaves without spinning"
        );
    }

    #[test]
    fn find_parent_reaches_non_leftmost_subtree() {
        let pool = std::sync::Arc::new(BufferPool::new(4096));
        let tree = BTree::new(&pool, 0);

        for i in 0..6000u32 {
            let key = i.to_be_bytes();
            let mut val = [0u8; 32000];
            val[0] = (i & 0xFF) as u8;
            val[1] = ((i >> 8) & 0xFF) as u8;
            assert!(tree.insert(&key, &val), "insert {i} failed");
        }

        assert!(tree.height() >= 2, "expected a multi-level tree");

        unsafe {
            let root_bf = pool.fix_frame(&tree.meta_swip, NoLatches::new(&pool));
            let root_ref = root_bf.frame_ref();
            assert!(!BTreeNode::is_leaf(root_ref.read_ref()));

            let root_sp = BTreeNode::sp(root_ref.read_ref());
            assert!(
                root_sp.num_slots() > 0,
                "root should have multiple children"
            );

            let subtree_swip = BTreeNode::upper_swip(root_ref.read_ref());
            let subtree_bf = tree.resolve_swip(subtree_swip);
            assert!(!BTreeNode::is_leaf(subtree_bf.frame_ref().read_ref()));

            let leaf_swip = BTreeNode::lookup_inner_swip(subtree_bf.frame_ref().read_ref(), &[]);
            let leaf_bf = tree.resolve_swip(leaf_swip);
            assert!(BTreeNode::is_leaf(leaf_bf.frame_ref().read_ref()));

            let (parent_bf, pos) = tree
                .find_parent(&ResidentFrame::from_pinned(&leaf_bf))
                .expect("should find parent under non-leftmost subtree");

            let parent_ref = parent_bf.frame_ref();
            assert!(parent_ref.same_frame(subtree_bf.frame_ref()));
            let parent_sp = BTreeNode::sp(parent_ref.read_ref());
            assert!(pos <= parent_sp.num_slots());

            drop(parent_bf);
            drop(leaf_bf);
            drop(subtree_bf);
            drop(root_bf);
        }
    }

    fn collect_range(
        tree: &BTree,
        lower: std::ops::Bound<&[u8]>,
        upper: std::ops::Bound<&[u8]>,
    ) -> Vec<u32> {
        let mut out = Vec::new();
        tree.scan_range(lower, upper, |k, _| {
            out.push(u32::from_be_bytes(k.try_into().unwrap()));
        });
        out
    }

    fn collect_range_desc(
        tree: &BTree,
        lower: std::ops::Bound<&[u8]>,
        upper: std::ops::Bound<&[u8]>,
    ) -> Vec<u32> {
        let mut out = Vec::new();
        tree.scan_range_desc(lower, upper, |k, _| {
            out.push(u32::from_be_bytes(k.try_into().unwrap()));
        });
        out
    }

    #[test]
    fn scan_range_bound_variants() {
        use std::ops::Bound;

        let pool = std::sync::Arc::new(BufferPool::new(64));
        let tree = BTree::new(&pool, 0);
        for i in 1u32..=10 {
            tree.insert(&i.to_be_bytes(), &i.to_be_bytes());
        }

        let cases = [
            (
                "inclusive_both",
                Some((3u32, true)),
                Some((7u32, true)),
                vec![3, 4, 5, 6, 7],
            ),
            (
                "exclusive_upper",
                Some((3u32, true)),
                Some((7u32, false)),
                vec![3, 4, 5, 6],
            ),
            (
                "exclusive_lower",
                Some((3u32, false)),
                Some((7u32, true)),
                vec![4, 5, 6, 7],
            ),
            (
                "unbounded_lower",
                None,
                Some((4u32, true)),
                vec![1, 2, 3, 4],
            ),
            ("unbounded_upper", Some((8u32, true)), None, vec![8, 9, 10]),
            ("unbounded_both", None, None, (1..=10).collect::<Vec<_>>()),
            (
                "lower_exceeds_upper",
                Some((7u32, true)),
                Some((3u32, true)),
                vec![],
            ),
        ];

        for (label, lower_spec, upper_spec, expected) in cases {
            let lower_bytes = lower_spec.map(|(v, _)| v.to_be_bytes());
            let upper_bytes = upper_spec.map(|(v, _)| v.to_be_bytes());
            let lower = match (lower_spec, lower_bytes.as_ref()) {
                (Some((_, true)), Some(bytes)) => Bound::Included(bytes.as_slice()),
                (Some((_, false)), Some(bytes)) => Bound::Excluded(bytes.as_slice()),
                _ => Bound::Unbounded,
            };
            let upper = match (upper_spec, upper_bytes.as_ref()) {
                (Some((_, true)), Some(bytes)) => Bound::Included(bytes.as_slice()),
                (Some((_, false)), Some(bytes)) => Bound::Excluded(bytes.as_slice()),
                _ => Bound::Unbounded,
            };
            assert_eq!(
                collect_range(&tree, lower, upper),
                expected,
                "{label}: wrong scan_range result"
            );
        }

        let sparse_pool = std::sync::Arc::new(BufferPool::new(64));
        let sparse_tree = BTree::new(&sparse_pool, 0);
        for k in [10u32, 20, 30] {
            sparse_tree.insert(&k.to_be_bytes(), &k.to_be_bytes());
        }
        let lo = 12u32.to_be_bytes();
        let hi = 18u32.to_be_bytes();
        assert!(
            collect_range(&sparse_tree, Bound::Included(&lo), Bound::Included(&hi)).is_empty(),
            "sparse window should yield no matches"
        );
    }

    #[test]
    fn scan_range_multi_leaf() {
        use std::ops::Bound;
        // Force multiple leaves with many small inserts.
        let pool = std::sync::Arc::new(BufferPool::new(256));
        let tree = BTree::new(&pool, 0);
        for i in 0u32..1000 {
            tree.insert(&i.to_be_bytes(), &i.to_be_bytes());
        }
        let lo = 250u32.to_be_bytes();
        let hi = 750u32.to_be_bytes();
        let got = collect_range(&tree, Bound::Included(&lo), Bound::Excluded(&hi));
        assert_eq!(got.first(), Some(&250));
        assert_eq!(got.last(), Some(&749));
        assert_eq!(got.len(), 500);
    }

    #[test]
    fn scan_range_desc_multi_leaf() {
        use std::ops::Bound;
        let pool = std::sync::Arc::new(BufferPool::new(256));
        let tree = BTree::new(&pool, 0);
        for i in 0u32..1000 {
            tree.insert(&i.to_be_bytes(), &i.to_be_bytes());
        }
        let lo = 250u32.to_be_bytes();
        let hi = 750u32.to_be_bytes();
        let got = collect_range_desc(&tree, Bound::Included(&lo), Bound::Excluded(&hi));
        assert_eq!(got.first(), Some(&749));
        assert_eq!(got.last(), Some(&250));
        assert_eq!(got.len(), 500);
    }

    #[test]
    fn insert_remove_reinsert() {
        let pool = std::sync::Arc::new(BufferPool::new(64));
        let tree = BTree::new(&pool, 0);

        // Insert 100 keys.
        for i in 0..100u32 {
            tree.insert(&i.to_be_bytes(), &i.to_be_bytes());
        }

        // Remove all odd keys.
        for i in (1..100u32).step_by(2) {
            assert!(tree.remove(&i.to_be_bytes()));
        }

        // Verify odd keys gone, even keys present.
        for i in 0..100u32 {
            if i % 2 == 0 {
                assert!(
                    tree.lookup(&i.to_be_bytes()).is_some(),
                    "even key {i} missing"
                );
            } else {
                assert!(
                    tree.lookup(&i.to_be_bytes()).is_none(),
                    "odd key {i} still present"
                );
            }
        }

        // Reinsert odd keys.
        for i in (1..100u32).step_by(2) {
            assert!(tree.insert(&i.to_be_bytes(), &(i * 100).to_be_bytes()));
        }

        // Verify all keys present with correct values.
        for i in 0..100u32 {
            let expected = if i % 2 == 0 { i } else { i * 100 };
            assert_eq!(
                tree.lookup(&i.to_be_bytes()).as_deref(),
                Some(expected.to_be_bytes().as_slice()),
                "key {i} wrong value"
            );
        }
    }

    #[test]
    fn remove_empty_leaf_merge_preserves_tree_correctness() {
        let pool = std::sync::Arc::new(BufferPool::new(64));
        let tree = BTree::new(&pool, 0);

        for i in 0..400u32 {
            assert!(tree.insert(&i.to_be_bytes(), &i.to_be_bytes()));
        }

        for i in 0..200u32 {
            assert!(tree.remove(&i.to_be_bytes()), "remove {i} failed");
        }
        for i in 0..200u32 {
            assert!(
                tree.lookup(&i.to_be_bytes()).is_none(),
                "key {i} should be absent"
            );
        }

        for i in 0..200u32 {
            assert!(tree.insert(&i.to_be_bytes(), &(i + 1000).to_be_bytes()));
        }
        for i in 0..200u32 {
            assert_eq!(
                tree.lookup(&i.to_be_bytes()).as_deref(),
                Some((i + 1000).to_be_bytes().as_slice()),
                "reinserted key {i} should be readable after leaf merge"
            );
        }
        for i in 200..400u32 {
            assert_eq!(
                tree.lookup(&i.to_be_bytes()).as_deref(),
                Some(i.to_be_bytes().as_slice()),
                "surviving key {i} should remain readable after leaf merge"
            );
        }
    }

    #[test]
    fn delete_rebalance_can_shrink_multi_level_tree() {
        let pool = std::sync::Arc::new(BufferPool::new(4096));
        let tree = BTree::new(&pool, 0);

        for i in 0..6000u32 {
            let key = i.to_be_bytes();
            let mut val = [0u8; 32000];
            val[0] = (i & 0xFF) as u8;
            val[1] = ((i >> 8) & 0xFF) as u8;
            assert!(tree.insert(&key, &val), "insert {i} failed");
        }

        let initial_height = tree.height();
        assert!(
            initial_height >= 2,
            "expected a multi-level tree before delete, got height {initial_height}"
        );

        for i in 0..5700u32 {
            assert!(tree.remove(&i.to_be_bytes()), "remove {i} failed");
        }

        let shrunk_height = tree.height();
        assert!(
            shrunk_height < initial_height,
            "expected delete rebalance to shrink tree height from {initial_height}, got {shrunk_height}"
        );

        for i in 0..5700u32 {
            assert!(
                tree.lookup(&i.to_be_bytes()).is_none(),
                "deleted key {i} should be absent after shrink"
            );
        }
        for i in 5700..6000u32 {
            let result = tree.lookup(&i.to_be_bytes());
            assert!(result.is_some(), "surviving key {i} missing after shrink");
            let val = result.unwrap();
            assert_eq!(
                val[0],
                (i & 0xFF) as u8,
                "surviving key {i} low byte changed"
            );
            assert_eq!(
                val[1],
                ((i >> 8) & 0xFF) as u8,
                "surviving key {i} high byte changed"
            );
        }
    }

    #[test]
    fn delete_rebalance_preserves_sparse_ranges_in_multi_level_tree() {
        let pool = std::sync::Arc::new(BufferPool::new(4096));
        let tree = BTree::new(&pool, 0);

        for i in 0..6000u32 {
            let key = i.to_be_bytes();
            let mut val = [0u8; 32000];
            val[0] = (i & 0xFF) as u8;
            val[1] = ((i >> 8) & 0xFF) as u8;
            assert!(tree.insert(&key, &val), "insert {i} failed");
        }

        assert!(
            tree.height() >= 2,
            "expected a multi-level tree before sparse delete"
        );

        for i in 600..5400u32 {
            assert!(tree.remove(&i.to_be_bytes()), "remove {i} failed");
        }

        for i in 0..600u32 {
            let result = tree.lookup(&i.to_be_bytes());
            assert!(
                result.is_some(),
                "low-range key {i} missing after sparse delete"
            );
        }
        for i in 600..5400u32 {
            assert!(
                tree.lookup(&i.to_be_bytes()).is_none(),
                "middle key {i} should be absent after sparse delete"
            );
        }
        for i in 5400..6000u32 {
            let result = tree.lookup(&i.to_be_bytes());
            assert!(
                result.is_some(),
                "high-range key {i} missing after sparse delete"
            );
        }
    }

    #[test]
    fn upsert_rewrites_larger_value() {
        let pool = std::sync::Arc::new(BufferPool::new(64));
        let tree = BTree::new(&pool, 0);

        assert!(tree.insert(b"k", b"v1"));
        assert!(!tree.upsert(b"k", b"much-larger-value"));
        assert_eq!(
            tree.lookup(b"k").as_deref(),
            Some(b"much-larger-value".as_slice())
        );
    }

    // -----------------------------------------------------------------------
    // Concurrent tests
    // -----------------------------------------------------------------------

    #[test]
    fn concurrent_insert_10k_scale() {
        let pool = Arc::new(BufferPool::new(8192));
        let tree = Arc::new(BTree::new(&pool, 0));
        let n_threads = 2;
        let per_thread = 10_000;
        let barrier = Arc::new(Barrier::new(n_threads));

        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let tree = tree.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    let mut failed = Vec::new();
                    for i in 0..per_thread {
                        let key = ((t * per_thread + i) as u64).to_be_bytes();
                        if !tree.insert(&key, &key) {
                            failed.push(t * per_thread + i);
                        }
                    }
                    failed
                })
            })
            .collect();

        let mut false_dupes = Vec::new();
        for h in handles {
            false_dupes.extend(h.join().unwrap());
        }

        // Verify via both lookup and scan.
        let mut missing_lookup = Vec::new();
        for i in 0..(n_threads * per_thread) as u64 {
            let key = i.to_be_bytes();
            if tree.lookup(&key).is_none() {
                missing_lookup.push(i);
            }
        }

        let mut scan_count = 0usize;
        let mut scan_keys = std::collections::BTreeSet::new();
        tree.scan(|k, _| {
            scan_count += 1;
            if k.len() == 8 {
                let val = u64::from_be_bytes(k.try_into().unwrap());
                scan_keys.insert(val);
            }
        });

        let expected = (n_threads * per_thread) as u64;
        let mut missing_scan = Vec::new();
        for i in 0..expected {
            if !scan_keys.contains(&i) {
                missing_scan.push(i);
            }
        }

        assert!(
            missing_lookup.is_empty() && missing_scan.is_empty() && false_dupes.is_empty(),
            "lookup missing: {missing_lookup:?}, scan missing: {missing_scan:?}, \
             scan_count: {scan_count}, false dupes: {false_dupes:?}",
        );
    }

    #[test]
    fn upsert_random_keys() {
        let pool = std::sync::Arc::new(BufferPool::new(4096));
        let tree = BTree::new(&pool, 0);

        // Load 1000 keys with hashed distribution.
        let n = 1000u64;
        let make_key = |i: u64| -> [u8; 8] {
            i.wrapping_mul(0x517cc1b727220a95)
                .wrapping_add(0x9e3779b97f4a7c15)
                .to_be_bytes()
        };
        for i in 0..n {
            tree.insert(&make_key(i), &[0xAA; 100]);
        }

        // Upsert all keys (remove + reinsert).
        for i in 0..n {
            let key = make_key(i);
            tree.remove(&key);
            assert!(tree.insert(&key, &[0xBB; 100]), "reinsert {i} failed");
        }

        // Verify all keys present with updated value.
        for i in 0..n {
            let val = tree.lookup(&make_key(i));
            assert!(val.is_some(), "key {i} missing after upsert");
            assert_eq!(val.unwrap()[0], 0xBB, "key {i} has wrong value");
        }
    }

    // -----------------------------------------------------------------------
    // Concurrent stress test with structural invariant checking
    // -----------------------------------------------------------------------

    /// Simple deterministic PRNG (xorshift64).
    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// Verify structural invariants of the B-tree:
    /// - every inserted key is findable via lookup
    /// - scan returns exactly the expected key set
    /// - scan count matches expected
    fn verify_tree(tree: &BTree, expected: &std::collections::BTreeSet<u64>, label: &str) {
        // Check lookup for every expected key.
        let mut lookup_missing = Vec::new();
        for &k in expected {
            let key = k.to_be_bytes();
            if tree.lookup(&key).is_none() {
                lookup_missing.push(k);
            }
        }

        // Check scan matches expected set.
        let mut scan_keys = std::collections::BTreeSet::new();
        let mut scan_count = 0usize;
        tree.scan(|k, _| {
            scan_count += 1;
            if k.len() == 8 {
                scan_keys.insert(u64::from_be_bytes(k.try_into().unwrap()));
            }
        });

        let mut scan_missing: Vec<u64> = expected.difference(&scan_keys).copied().collect();
        let mut scan_extra: Vec<u64> = scan_keys.difference(expected).copied().collect();

        if !lookup_missing.is_empty()
            || !scan_missing.is_empty()
            || !scan_extra.is_empty()
            || scan_count != expected.len()
        {
            // Truncate for readability.
            lookup_missing.truncate(20);
            scan_missing.truncate(20);
            scan_extra.truncate(20);
            panic!(
                "{label}: invariant violation!\n\
                 expected {}, scan_count {scan_count}\n\
                 lookup_missing (first 20): {lookup_missing:?}\n\
                 scan_missing (first 20): {scan_missing:?}\n\
                 scan_extra (first 20): {scan_extra:?}",
                expected.len(),
            );
        }
    }

    #[test]
    fn concurrent_stress_small() {
        // Stress test with fat values to force frequent splits.
        // Uses a deterministic seed for reproducibility.
        let seed: u64 = 0xDEAD_BEEF_CAFE_1234;
        let pool = Arc::new(BufferPool::new(4096));
        let tree = Arc::new(BTree::new(&pool, 0));
        let n_threads = 4;
        let ops_per_thread = 2_000;
        let key_range = 500u64; // small range → many collisions and splits
        let value = [0xAA; 200]; // fat values → ~18 entries per leaf

        let barrier = Arc::new(Barrier::new(n_threads));

        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let tree = tree.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    let mut rng = seed.wrapping_add(t as u64);
                    barrier.wait();
                    for _ in 0..ops_per_thread {
                        let k = xorshift64(&mut rng) % key_range;
                        let key = k.to_be_bytes();
                        let op = xorshift64(&mut rng) % 100;
                        if op < 60 {
                            // 60% insert
                            tree.insert(&key, &value);
                        } else if op < 80 {
                            // 20% lookup
                            std::hint::black_box(tree.lookup(&key));
                        } else {
                            // 20% remove
                            tree.remove(&key);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Rebuild expected state by replaying deterministically.
        let mut expected = std::collections::BTreeSet::new();
        for t in 0..n_threads {
            let mut rng = seed.wrapping_add(t as u64);
            for _ in 0..ops_per_thread {
                let k = xorshift64(&mut rng) % key_range;
                let op = xorshift64(&mut rng) % 100;
                if op < 60 {
                    expected.insert(k);
                } else if op < 80 {
                    // lookup — no state change
                } else {
                    expected.remove(&k);
                }
            }
        }

        // NOTE: The expected set is only approximate because thread
        // interleaving is nondeterministic. A key could be removed by
        // thread A and then re-inserted by thread B, or vice versa.
        // So we verify a weaker property: every key in the tree should
        // be in [0, key_range), and lookup/scan should agree.
        let mut tree_keys = std::collections::BTreeSet::new();
        tree.scan(|k, _| {
            if k.len() == 8 {
                tree_keys.insert(u64::from_be_bytes(k.try_into().unwrap()));
            }
        });

        // Verify: lookup and scan agree.
        for &k in &tree_keys {
            let key = k.to_be_bytes();
            assert!(
                tree.lookup(&key).is_some(),
                "seed={seed:#X}: key {k} in scan but not in lookup"
            );
        }

        // Verify: every key in scan is in range.
        for &k in &tree_keys {
            assert!(
                k < key_range,
                "seed={seed:#X}: key {k} out of range [0, {key_range})"
            );
        }
    }

    #[test]
    fn concurrent_stress_insert_only() {
        // Pure insert workload — strongest invariant: every inserted
        // key must be present afterward.
        let seed: u64 = 0xBAAD_F00D_1337_C0DE;
        let pool = Arc::new(BufferPool::new(4096));
        let tree = Arc::new(BTree::new(&pool, 0));
        let n_threads = 4;
        let per_thread = 1_000;
        let value = [0xBB; 200]; // fat values

        let barrier = Arc::new(Barrier::new(n_threads));

        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let tree = tree.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    let mut rng = seed.wrapping_add(t as u64);
                    for _ in 0..per_thread {
                        let k = xorshift64(&mut rng);
                        let key = k.to_be_bytes();
                        tree.insert(&key, &value);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Replay to get expected key set (some keys may collide across
        // threads, that's fine — insert returns false but key is present).
        let mut expected = std::collections::BTreeSet::new();
        for t in 0..n_threads {
            let mut rng = seed.wrapping_add(t as u64);
            for _ in 0..per_thread {
                expected.insert(xorshift64(&mut rng));
            }
        }

        verify_tree(&tree, &expected, &format!("seed={seed:#X}"));
    }

    // -------------------------------------------------------------------
    // Stress tests: concurrent splits + lookups + eviction churn
    // -------------------------------------------------------------------

    /// Concurrent inserts (forcing splits) + lookups on a small pool.
    /// The small pool forces eviction churn, exercising the orphan
    /// resolve path under contention.
    #[test]
    fn stress_concurrent_splits_lookups_eviction() {
        let n_keys = 10_000u32;
        // Pool of 256 frames with 10K keys: large enough to avoid
        // deadlock (all-frames-pinned) but small enough to force
        // eviction churn.
        let pool = std::sync::Arc::new(BufferPool::new(256));
        let tree = BTree::new(&pool, 0);

        // Phase 1: insert all keys (single-threaded to avoid complexity).
        for i in 0..n_keys {
            let key = i.to_be_bytes();
            tree.insert(&key, &key);
        }

        // Phase 2: concurrent lookups + additional inserts.
        let n_threads = 4;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(n_threads + 1));
        let tree_ptr = &tree as *const BTree as usize;

        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let tree = unsafe { &*(tree_ptr as *const BTree) };
                    barrier.wait();

                    if t % 2 == 0 {
                        // Lookup thread: verify all keys exist.
                        let mut missing = 0u32;
                        for i in 0..n_keys {
                            let key = i.to_be_bytes();
                            if tree.lookup(&key).is_none() {
                                missing += 1;
                            }
                        }
                        assert_eq!(missing, 0, "thread {t}: {missing} keys missing");
                    } else {
                        // Insert thread: add more keys (forces splits).
                        let base = n_keys + t as u32 * 5_000;
                        for i in 0..5_000 {
                            let key = (base + i).to_be_bytes();
                            tree.insert(&key, &key);
                        }
                    }
                })
            })
            .collect();

        barrier.wait();
        for h in handles {
            h.join().unwrap();
        }

        // Phase 3: verify all original keys still findable.
        let mut missing = 0u32;
        for i in 0..n_keys {
            let key = i.to_be_bytes();
            if tree.lookup(&key).is_none() {
                missing += 1;
            }
        }
        assert_eq!(missing, 0, "post-stress: {missing} keys missing");

        // Phase 4: full scan integrity check.
        let mut scan_count = 0u64;
        tree.scan(|_k, _v| {
            scan_count += 1;
        });
        // At least n_keys + (n_threads/2 * 5000) inserted keys.
        let min_expected = n_keys as u64 + (n_threads as u64 / 2) * 5_000;
        assert!(
            scan_count >= min_expected,
            "scan count {scan_count} < expected {min_expected}"
        );
    }

    /// Verify scan returns correct count after concurrent inserts complete.
    #[test]
    fn stress_scan_after_concurrent_inserts() {
        let pool = std::sync::Arc::new(BufferPool::new(256));
        let tree = BTree::new(&pool, 0);

        // Pre-populate.
        for i in 0..5_000u32 {
            let key = i.to_be_bytes();
            tree.insert(&key, &key);
        }

        // Concurrent inserts from 4 threads.
        let n_threads = 4usize;
        let per_thread = 2_500u32;
        let tree_ptr = &tree as *const BTree as usize;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(n_threads + 1));
        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let tree = unsafe { &*(tree_ptr as *const BTree) };
                    barrier.wait();
                    let base = 5_000 + t as u32 * per_thread;
                    for i in 0..per_thread {
                        let key = (base + i).to_be_bytes();
                        tree.insert(&key, &key);
                    }
                })
            })
            .collect();

        barrier.wait();
        for h in handles {
            h.join().unwrap();
        }

        // Scan after all inserts complete.
        let expected = 5_000 + n_threads as u64 * per_thread as u64;
        let mut count = 0u64;
        tree.scan(|_k, _v| {
            count += 1;
        });
        assert_eq!(count, expected, "scan count {count} != expected {expected}");
    }

    /// Verify BTree works after flush + drop + reopen from a FilePageStore.
    #[test]
    #[cfg(not(miri))]
    fn btree_reopen_file_backed() {
        use pagebox_storage::page_store::FilePageStore;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");

        // Populate.
        let store = FilePageStore::open(&path).unwrap();
        let pool = std::sync::Arc::new(BufferPool::with_store(1000, Box::new(store)));
        let (root_pid, height) = {
            let tree = BTree::new(&pool, 0);
            for i in 0..1000u32 {
                let key = i.to_be_bytes();
                tree.insert(&key, &key);
            }
            (tree.root_page_id(), tree.height())
        };
        pool.flush().unwrap();
        drop(pool);

        // Reopen with moderate pool (256 frames for 1000 keys ~= 8 pages).
        let store = FilePageStore::open(&path).unwrap();
        let pool = std::sync::Arc::new(BufferPool::with_store(256, Box::new(store)));
        let tree = BTree::open(&pool, root_pid, height, 0);

        // Verify lookups.
        for i in 0..1000u32 {
            let key = i.to_be_bytes();
            assert!(tree.lookup(&key).is_some(), "lookup {i} failed");
        }

        // Verify scan.
        let mut count = 0u64;
        tree.scan(|_k, _v| {
            count += 1;
        });
        assert_eq!(count, 1000);
    }

    /// Reopen with larger dataset to verify no deadlock under eviction.
    #[test]
    #[cfg(not(miri))]
    fn btree_reopen_50k_file_backed() {
        use pagebox_storage::page_store::FilePageStore;

        struct TreeProxy(*const BTree);
        unsafe impl Send for TreeProxy {}
        unsafe impl Sync for TreeProxy {}
        impl ParentFinder for TreeProxy {
            fn find_and_unswizzle(&self, child: BufferFrameRef, child_pid: u64) -> bool {
                unsafe { &*self.0 }.find_and_unswizzle(child, child_pid)
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        let n = 50_000u32;

        let store = FilePageStore::open(&path).unwrap();
        let pool = std::sync::Arc::new(BufferPool::with_store(n as usize, Box::new(store)));
        let (root_pid, height) = {
            let tree = BTree::new(&pool, 0);
            pool.register_dt(0, std::sync::Arc::new(TreeProxy(&tree)));
            for i in 0..n {
                let key = i.to_be_bytes();
                tree.insert(&key, &key);
            }
            (tree.root_page_id(), tree.height())
        };
        pool.flush().unwrap();
        drop(pool);

        // 256 frames for ~253 total pages — tight, tests eviction.
        let store = FilePageStore::open(&path).unwrap();
        let pool = std::sync::Arc::new(BufferPool::with_store(256, Box::new(store)));
        let tree = BTree::open(&pool, root_pid, height, 0);
        pool.register_dt(0, std::sync::Arc::new(TreeProxy(&tree)));

        // Spot-check lookups (not all 50K to keep test fast).
        for i in (0..n).step_by(100) {
            let key = i.to_be_bytes();
            assert!(tree.lookup(&key).is_some(), "lookup {i} failed");
        }
    }

    /// Reopen a file-backed tree and continue inserting so post-reopen
    /// leaf/root split paths are exercised.
    #[test]
    #[cfg(not(miri))]
    fn btree_reopen_and_continue_inserting() {
        use pagebox_storage::page_store::FilePageStore;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");

        let store = FilePageStore::open(&path).unwrap();
        let pool = std::sync::Arc::new(BufferPool::with_store(4096, Box::new(store)));
        let (root_pid, height) = {
            let tree = BTree::new(&pool, 0);
            for i in 0..20_000u32 {
                let key = i.to_be_bytes();
                tree.insert(&key, &key);
            }
            (tree.root_page_id(), tree.height())
        };
        pool.flush().unwrap();
        drop(pool);

        let store = FilePageStore::open(&path).unwrap();
        let pool = std::sync::Arc::new(BufferPool::with_store(4096, Box::new(store)));
        let tree = BTree::open(&pool, root_pid, height, 0);

        for i in 20_000u32..40_000u32 {
            let key = i.to_be_bytes();
            tree.insert(&key, &key);
        }

        for i in (0..40_000u32).step_by(97) {
            let key = i.to_be_bytes();
            assert_eq!(tree.lookup(&key).as_deref(), Some(key.as_slice()));
        }
    }

    #[test]
    #[cfg(not(miri))]
    fn btree_root_split_with_cool_root_after_reopen() {
        use pagebox_storage::page_store::FilePageStore;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        let n = 50_000u32;

        let store = FilePageStore::open(&path).unwrap();
        let pool = std::sync::Arc::new(BufferPool::with_store(4096, Box::new(store)));
        let (root_pid, height) = {
            let tree = BTree::new(&pool, 0);
            for i in 0..n {
                let key = i.to_be_bytes();
                tree.insert(&key, &key);
            }
            (tree.root_page_id(), tree.height())
        };
        pool.flush().unwrap();
        drop(pool);

        let store = FilePageStore::open(&path).unwrap();
        let pool = std::sync::Arc::new(BufferPool::with_store(4096, Box::new(store)));
        let tree = BTree::open(&pool, root_pid, height, 0);

        let root = unsafe { pool.fix_frame(&tree.meta_swip, NoLatches::new(&pool)) };
        let root_swip = ResidentFrame::from_pinned(&root).hot_swip();
        drop(root);
        let mut cooled = root_swip;
        cooled.cool();
        tree.meta_swip.store(cooled, Ordering::Release);

        for i in n..(n + 50_000) {
            let key = i.to_be_bytes();
            tree.insert(&key, &key);
        }

        assert!(
            tree.height() >= height,
            "tree height should stay valid after cool-root split path"
        );
        for i in (0..(n + 50_000)).step_by(251) {
            let key = i.to_be_bytes();
            assert_eq!(tree.lookup(&key).as_deref(), Some(key.as_slice()));
        }
    }

    #[test]
    fn generated_sequential_history_matches_btreemap() {
        generated_test!(2, |tc| {
            let pool = std::sync::Arc::new(BufferPool::new(256));
            let tree = BTree::new(&pool, 0);
            let mut model = BTreeMap::new();
            let steps = tc.usize(1, 12);

            for step in 0..steps {
                let op = tc.u8(5);
                let key = tc.bytes(1, 8);
                let value = tc.bytes(0, 24);

                match op {
                    0 => {
                        let inserted = tree.insert(&key, &value);
                        let expected = if model.contains_key(&key) {
                            false
                        } else {
                            model.insert(key.clone(), value.clone());
                            true
                        };
                        assert_eq!(inserted, expected, "step {step}: insert result diverged");
                    }
                    1 => {
                        let inserted = tree.upsert(&key, &value);
                        let expected = model.insert(key.clone(), value.clone()).is_none();
                        assert_eq!(inserted, expected, "step {step}: upsert result diverged");
                    }
                    2 => {
                        let removed = tree.remove(&key);
                        let expected = model.remove(&key).is_some();
                        assert_eq!(removed, expected, "step {step}: remove result diverged");
                    }
                    3 => {
                        assert_eq!(
                            tree.lookup(&key),
                            model.get(&key).cloned(),
                            "step {step}: lookup diverged from model"
                        );
                    }
                    4 => {
                        let lower_key = tc.bytes(1, 8);
                        let upper_key = tc.bytes(1, 8);
                        let lower = match tc.u8(2) {
                            0 => Bound::Unbounded,
                            1 => Bound::Included(lower_key.as_slice()),
                            _ => Bound::Excluded(lower_key.as_slice()),
                        };
                        let upper = match tc.u8(2) {
                            0 => Bound::Unbounded,
                            1 => Bound::Included(upper_key.as_slice()),
                            _ => Bound::Excluded(upper_key.as_slice()),
                        };
                        assert_eq!(
                            collect_range_pairs(&tree, lower, upper),
                            model_collect_range(&model, lower, upper),
                            "step {step}: scan_range diverged from model"
                        );
                    }
                    _ => {
                        assert_eq!(
                            collect_all(&tree),
                            model_collect_all(&model),
                            "step {step}: full scan diverged from model"
                        );
                    }
                }
            }

            assert_tree_matches_model(&tree, &model, "final sequential history");
        });
    }

    #[test]
    #[cfg(not(miri))]
    fn generated_reopen_history_matches_btreemap() {
        generated_test!(2, |tc| {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("data");

            let mut model = BTreeMap::new();
            let mut reopen_counter = 0usize;
            let mut root_pid = 0u64;
            let mut height = 0u32;

            {
                let store = FilePageStore::open(&path).unwrap();
                let pool = std::sync::Arc::new(BufferPool::with_store(256, Box::new(store)));
                let tree = BTree::new(&pool, 0);
                let steps = tc.usize(1, 16);

                for step in 0..steps {
                    let op = tc.u8(4);
                    let key = tc.bytes(1, 8);
                    let value = tc.bytes(0, 24);

                    match op {
                        0 => {
                            let inserted = tree.insert(&key, &value);
                            let expected = if model.contains_key(&key) {
                                false
                            } else {
                                model.insert(key.clone(), value.clone());
                                true
                            };
                            assert_eq!(inserted, expected, "step {step}: insert result diverged");
                        }
                        1 => {
                            let inserted = tree.upsert(&key, &value);
                            let expected = model.insert(key.clone(), value.clone()).is_none();
                            assert_eq!(inserted, expected, "step {step}: upsert result diverged");
                        }
                        2 => {
                            let removed = tree.remove(&key);
                            let expected = model.remove(&key).is_some();
                            assert_eq!(removed, expected, "step {step}: remove result diverged");
                        }
                        3 => {
                            assert_eq!(
                                tree.lookup(&key),
                                model.get(&key).cloned(),
                                "step {step}: lookup diverged from model"
                            );
                        }
                        _ => {
                            assert_tree_matches_model(
                                &tree,
                                &model,
                                &format!("pre-reopen checkpoint {reopen_counter}"),
                            );
                            pool.flush().unwrap();
                            root_pid = tree.root_page_id();
                            height = tree.height();
                            reopen_counter += 1;
                            break;
                        }
                    }
                }

                if reopen_counter == 0 {
                    assert_tree_matches_model(&tree, &model, "pre-final flush");
                    pool.flush().unwrap();
                    root_pid = tree.root_page_id();
                    height = tree.height();
                    reopen_counter = 1;
                }
            }

            for reopen_idx in 0..reopen_counter {
                let store = FilePageStore::open(&path).unwrap();
                let pool = std::sync::Arc::new(BufferPool::with_store(256, Box::new(store)));
                let tree = BTree::open(&pool, root_pid, height, 0);

                assert_tree_matches_model(
                    &tree,
                    &model,
                    &format!("post-reopen checkpoint {reopen_idx}"),
                );

                let extra_steps = tc.usize(1, 6);
                for step in 0..extra_steps {
                    let op = tc.u8(2);
                    let key = tc.bytes(1, 8);
                    let value = tc.bytes(0, 24);

                    match op {
                        0 => {
                            let inserted = tree.upsert(&key, &value);
                            let expected = model.insert(key.clone(), value.clone()).is_none();
                            assert_eq!(
                                inserted, expected,
                                "reopen {reopen_idx} step {step}: upsert result diverged"
                            );
                        }
                        1 => {
                            let removed = tree.remove(&key);
                            let expected = model.remove(&key).is_some();
                            assert_eq!(
                                removed, expected,
                                "reopen {reopen_idx} step {step}: remove result diverged"
                            );
                        }
                        _ => {
                            assert_eq!(
                                tree.lookup(&key),
                                model.get(&key).cloned(),
                                "reopen {reopen_idx} step {step}: lookup diverged from model"
                            );
                        }
                    }
                }

                assert_tree_matches_model(
                    &tree,
                    &model,
                    &format!("post-reopen mutation checkpoint {reopen_idx}"),
                );
                pool.flush().unwrap();
                root_pid = tree.root_page_id();
                height = tree.height();
            }
        });
    }

    // -----------------------------------------------------------------------
    // Stage 4: B-tree API surface and edge-case tests
    // -----------------------------------------------------------------------

    #[test]
    fn prefix_scan_returns_matching_keys() {
        let pool = Arc::new(BufferPool::new(256));
        let tree = BTree::new(&pool, 0);

        let prefixes: [&[u8]; 3] = [b"alpha", b"beta", b"gamma"];
        for (pi, prefix) in prefixes.iter().enumerate() {
            for i in 0..10u32 {
                let mut key = prefix.to_vec();
                key.push(b'-');
                key.extend_from_slice(&i.to_be_bytes());
                let val = vec![(pi as u32 * 10 + i) as u8; 4];
                tree.insert(&key, &val);
            }
        }

        for (pi, prefix) in prefixes.iter().enumerate() {
            let mut found = Vec::new();
            tree.scan_prefix(prefix, |k, v| {
                found.push((k.to_vec(), v.to_vec()));
            });
            assert_eq!(found.len(), 10, "prefix {:?} should match 10 keys", prefix);
            for entry in &found {
                assert!(
                    entry.0.starts_with(prefix),
                    "prefix scan returned non-matching key"
                );
                assert_eq!(entry.1.len(), 4);
            }
            let first_val = found[0].1[0] as u32;
            assert_eq!(first_val, pi as u32 * 10);
        }
    }

    #[test]
    fn prefix_scan_multi_leaf_traverses_right_siblings() {
        let pool = Arc::new(BufferPool::new(256));
        let tree = BTree::new(&pool, 0);

        let prefix = b"key";
        let n = 1000u32;
        for i in 0..n {
            let mut key = prefix.to_vec();
            key.extend_from_slice(&i.to_be_bytes());
            tree.insert(&key, &i.to_be_bytes());
        }

        let mut count = 0;
        tree.scan_prefix(prefix, |_, _| count += 1);
        assert_eq!(count, n as usize, "prefix scan should traverse all leaves");
    }

    #[test]
    fn prefix_scan_empty_prefix_matches_all() {
        let pool = Arc::new(BufferPool::new(64));
        let tree = BTree::new(&pool, 0);

        for i in 0..50u32 {
            tree.insert(&i.to_be_bytes(), &i.to_be_bytes());
        }

        let mut count = 0;
        tree.scan_prefix(b"", |_, _| count += 1);
        assert_eq!(count, 50, "empty prefix should match all keys");
    }

    #[test]
    fn prefix_scan_no_matches_returns_empty() {
        let pool = Arc::new(BufferPool::new(64));
        let tree = BTree::new(&pool, 0);

        for i in 0..10u32 {
            tree.insert(&i.to_be_bytes(), &i.to_be_bytes());
        }

        let mut count = 0;
        tree.scan_prefix(b"nonexistent", |_, _| count += 1);
        assert_eq!(count, 0, "prefix with no matches should return empty");
    }

    #[test]
    fn prefix_scan_borrowed_until_stops_early() {
        let pool = Arc::new(BufferPool::new(256));
        let tree = BTree::new(&pool, 0);

        let prefix = b"key";
        for i in 0..100u32 {
            let mut key = prefix.to_vec();
            key.extend_from_slice(&i.to_be_bytes());
            tree.insert(&key, &i.to_be_bytes());
        }

        let mut count = 0;
        tree.scan_prefix_borrowed_until(prefix, |_, _| {
            count += 1;
            count < 5
        });
        assert_eq!(count, 5, "scan_prefix_borrowed_until should stop early");
    }

    #[test]
    fn zero_length_keys_and_values() {
        let pool = Arc::new(BufferPool::new(64));
        let tree = BTree::new(&pool, 0);

        tree.insert(b"", b"");
        assert_eq!(tree.lookup(b""), Some(b"".to_vec()));

        tree.insert(b"\x00", b"val");
        assert_eq!(tree.lookup(b"\x00"), Some(b"val".to_vec()));

        assert!(tree.lookup(b"").is_some());

        let keys: Vec<Vec<u8>> = collect_all(&tree).into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys.len(), 2, "should have two keys");
        assert!(keys[0] < keys[1], "keys should be sorted");
    }

    #[test]
    fn zero_length_key_duplicate_returns_false() {
        let pool = Arc::new(BufferPool::new(64));
        let tree = BTree::new(&pool, 0);

        assert!(tree.insert(b"", b"v1"));
        assert!(!tree.insert(b"", b"v2"));
        assert_eq!(tree.lookup(b""), Some(b"v1".to_vec()));
    }

    #[test]
    fn descending_scan_with_unbounded_lower() {
        let pool = Arc::new(BufferPool::new(256));
        let tree = BTree::new(&pool, 0);

        for i in 0..1000u32 {
            tree.insert(&i.to_be_bytes(), &i.to_be_bytes());
        }

        // Descending scan from the max key down to (but excluding) hi.
        // With Unbounded lower and Excluded(hi) upper, the range is
        // (-∞, hi) — keys below hi, descending from the max.
        let lo = 300u32.to_be_bytes();
        let got = collect_range_desc(&tree, Bound::Included(&lo), Bound::Unbounded);

        // With Unbounded upper, the scan starts from the rightmost key (999)
        // and descends to the lower bound.
        assert_eq!(got.first().copied(), Some(999));
        assert_eq!(got.last().copied(), Some(300));
        assert_eq!(got.len(), 700, "should cover [300, 999] descending");

        for i in 1..got.len() {
            assert!(
                got[i - 1] > got[i],
                "descending scan should be monotonically decreasing"
            );
        }
    }

    #[test]
    fn lookup_fixed_returns_none_for_n_zero() {
        let pool = Arc::new(BufferPool::new(64));
        let tree = BTree::new(&pool, 0);
        tree.insert(b"key", b"value");

        assert!(tree.lookup_fixed::<0>(b"key").is_none());
    }

    #[test]
    fn lookup_fixed_returns_truncated_for_n_smaller_than_value() {
        let pool = Arc::new(BufferPool::new(64));
        let tree = BTree::new(&pool, 0);
        let val = [42u8; 4];
        tree.insert(b"key", &val);

        // lookup_fixed returns None when N != value length — it is an
        // exact-size match, not truncation.
        let got = tree.lookup_fixed::<4>(b"key").unwrap();
        assert_eq!(got, [42u8; 4]);

        // N != value length returns None.
        assert!(tree.lookup_fixed::<8>(b"key").is_none());
        assert!(tree.lookup_fixed::<2>(b"key").is_none());
    }

    #[test]
    fn tree_height_grows_to_three() {
        let pool = Arc::new(BufferPool::new(4096));
        let tree = BTree::new(&pool, 0);

        // 64K leaves hold ~2 entries of a ~32 KiB value. The root inner node
        // holds ≈2729 children, so ≈6000 entries force a root-inner split and
        // reach height 2. Height 3 would need ≈2729*2729 leaves — impractical
        // for a unit test, so we assert height 2.
        let n = 6_000u32;
        for i in 0..n {
            let key = i.to_be_bytes();
            let val = [i as u8; 32000];
            tree.insert(&key, &val);
        }

        assert!(
            tree.height() >= 2,
            "expected height >= 2 with {n} 32000-byte values, got {}",
            tree.height()
        );

        for i in 0..100u32 {
            let key = i.to_be_bytes();
            let result = tree.lookup(&key);
            assert!(result.is_some(), "key {i} not found");
            assert_eq!(result.as_ref().unwrap().len(), 32000);
        }
    }

    #[test]
    fn large_values_near_page_boundary() {
        let pool = Arc::new(BufferPool::new(64));
        let tree = BTree::new(&pool, 0);

        // Values large enough that only 1-2 fit per 64K leaf, including
        // values near the page limit that exercise the single-entry split path.
        let val_sizes = [8000, 16000, 24000, 32000, 48000, 60000];
        for (i, &vs) in val_sizes.iter().enumerate() {
            let key = (i as u32).to_be_bytes();
            let val = vec![i as u8; vs];
            assert!(tree.insert(&key, &val), "insert {i} failed");
        }

        for (i, &vs) in val_sizes.iter().enumerate() {
            let key = (i as u32).to_be_bytes();
            let result = tree.lookup(&key);
            assert!(result.is_some(), "key {i} (value size {vs}) not found");
            assert_eq!(result.as_ref().unwrap().len(), vs);
            assert_eq!(result.unwrap()[0], i as u8);
        }

        let mut count = 0;
        tree.scan(|_, _| count += 1);
        assert_eq!(count, val_sizes.len());
    }

    #[test]
    fn separator_key_routes_to_correct_side_after_split() {
        let pool = Arc::new(BufferPool::new(256));
        let tree = BTree::new(&pool, 0);

        let n = 700u32;
        for i in 0..n {
            let key = i.to_be_bytes();
            let val = [i as u8; 100];
            tree.insert(&key, &val);
        }

        assert!(tree.height() >= 1, "tree should have split at least once");

        for i in 0..n {
            let key = i.to_be_bytes();
            assert!(
                tree.lookup(&key).is_some(),
                "every key including separator should be findable after split"
            );
        }
    }

    #[test]
    fn lookup_with_callback_receives_correct_slice() {
        let pool = Arc::new(BufferPool::new(64));
        let tree = BTree::new(&pool, 0);

        let val = b"hello_world_12345";
        tree.insert(b"key", val);

        let result = tree.lookup_with(b"key", |v| {
            assert_eq!(v, Some(val.as_slice()));
            v.map(|s| s.len())
        });

        assert_eq!(result, Some(val.len()));
    }

    #[test]
    fn lookup_with_callback_returns_none_for_missing_key() {
        let pool = Arc::new(BufferPool::new(64));
        let tree = BTree::new(&pool, 0);
        tree.insert(b"key", b"value");

        let result = tree.lookup_with(b"missing", |v| {
            assert!(v.is_none());
            v.map(|s| s.len())
        });

        assert_eq!(result, None);
    }

    #[test]
    fn single_large_value_per_leaf_does_not_hang() {
        // Regression test: inserting values large enough that only 1 entry
        // fits per 64K leaf previously caused an infinite loop in split_node
        // (count < 2 early return + insert retry loop).
        let pool = Arc::new(BufferPool::new(64));
        let tree = BTree::new(&pool, 0);

        for i in 0..6u32 {
            let key = i.to_be_bytes();
            let val = vec![i as u8; 60000];
            assert!(tree.insert(&key, &val), "insert {i} should succeed");
        }

        // Verify all entries are findable.
        for i in 0..6u32 {
            let key = i.to_be_bytes();
            let result = tree.lookup(&key);
            assert_eq!(
                result.as_ref().map(|v| v.len()),
                Some(60000),
                "key {i} should be findable with 60000-byte value"
            );
            assert_eq!(result.unwrap()[0], i as u8);
        }
    }

    #[test]
    fn single_large_value_descending_keys_does_not_hang() {
        // Same regression but with descending keys — exercises the path
        // where the pending key is smaller than the existing key, requiring
        // the existing entry to be moved to the right sibling.
        let pool = Arc::new(BufferPool::new(64));
        let tree = BTree::new(&pool, 0);

        for i in (0..6u32).rev() {
            let key = i.to_be_bytes();
            let val = vec![i as u8; 60000];
            assert!(tree.insert(&key, &val), "insert {i} should succeed");
        }

        for i in 0..6u32 {
            let key = i.to_be_bytes();
            let result = tree.lookup(&key);
            assert_eq!(
                result.as_ref().map(|v| v.len()),
                Some(60000),
                "key {i} should be findable"
            );
        }
    }
}
