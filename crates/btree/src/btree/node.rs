use std::marker::PhantomData;

use pagebox_hybrid_latch::{OptimisticGuard, Restart};

use pagebox_storage::buffer_frame::{
    BufferFrameReadRef, BufferFrameRef, BufferFrameWriteRef, FrameState, PAGE_SIZE, ParentLink,
    StableSwipRef,
};
use pagebox_storage::buffer_pool::{
    BufferPool, ExclusiveFrame, OptimisticFrame, PinnedFrame, SharedFrame,
};
use pagebox_storage::slotted_page::SlottedPage;
use pagebox_swip_kernel::SwipWord as Swip;

use super::parent_edge::{ParentEdge, parent_edge_round_trip_pos};

pub(crate) const FLAG_IS_LEAF: u16 = 1 << 1;
const UPPER_OFFSET: usize = PAGE_SIZE - 8;
/// Left-sibling page ID for leaf nodes, stored in the suffix.
/// Leaves reserve 16 bytes of suffix: 8 for left-sibling + 8 for right-sibling.
pub(crate) const LEFT_SIBLING_OFFSET: usize = UPPER_OFFSET;
/// Right-sibling page ID for leaf nodes, stored in the suffix.
pub(crate) const RIGHT_SIBLING_OFFSET: usize = PAGE_SIZE - 16;

/// Threshold: a node is "underfull" if free_space_after_compaction > this.
/// ~60% of usable space free -> candidate for merge.
const UNDERFULL_THRESHOLD: usize = (PAGE_SIZE * 60) / 100;

pub(crate) struct BTreeNode;

impl BTreeNode {
    pub(crate) fn init(bf: BufferFrameWriteRef<'_>, is_leaf: bool) {
        let page = bf.page_mut();
        let sp = SlottedPage::init(page);
        if is_leaf {
            sp.reserve_suffix(16);
            sp.set_flag(FLAG_IS_LEAF);
        } else {
            sp.reserve_suffix(8);
        }
        let page = bf.page_mut();
        pagebox_storage::slotted_page::write_page_type(
            page,
            pagebox_storage::slotted_page::PageType::Index,
        );
        page[UPPER_OFFSET..].fill(0);
        page[RIGHT_SIBLING_OFFSET..RIGHT_SIBLING_OFFSET + 8].fill(0);
    }

    pub(crate) fn is_leaf(bf: BufferFrameReadRef<'_>) -> bool {
        Self::sp(bf).has_custom_flag(FLAG_IS_LEAF)
    }

    pub(crate) fn sp<'a>(bf: BufferFrameReadRef<'a>) -> &'a SlottedPage {
        SlottedPage::from_page(bf.page())
    }

    pub(crate) fn sp_mut<'a>(bf: BufferFrameWriteRef<'a>) -> &'a mut SlottedPage {
        SlottedPage::from_page_mut(bf.page_mut())
    }

    pub(crate) fn set_upper(bf: BufferFrameWriteRef<'_>, swip: Swip) {
        let page = bf.page_mut();
        page[UPPER_OFFSET..].copy_from_slice(&swip.raw().to_ne_bytes());
    }

    pub(crate) fn upper_swip(bf: BufferFrameReadRef<'_>) -> Swip {
        let page = bf.page();
        let raw = u64::from_ne_bytes(page[UPPER_OFFSET..].try_into().unwrap());
        Swip::from_raw(raw)
    }

    pub(crate) fn set_leaf_right_pid(bf: BufferFrameWriteRef<'_>, pid: u64) {
        let page = bf.page_mut();
        page[RIGHT_SIBLING_OFFSET..RIGHT_SIBLING_OFFSET + 8].copy_from_slice(&pid.to_ne_bytes());
    }

    pub(crate) fn set_leaf_left_pid(bf: BufferFrameWriteRef<'_>, pid: u64) {
        let page = bf.page_mut();
        page[LEFT_SIBLING_OFFSET..LEFT_SIBLING_OFFSET + 8].copy_from_slice(&pid.to_ne_bytes());
    }

    pub(crate) fn leaf_right_pid(bf: BufferFrameReadRef<'_>) -> u64 {
        let page = bf.page();
        u64::from_ne_bytes(
            page[RIGHT_SIBLING_OFFSET..RIGHT_SIBLING_OFFSET + 8]
                .try_into()
                .unwrap(),
        )
    }

    pub(crate) fn leaf_left_pid(bf: BufferFrameReadRef<'_>) -> u64 {
        let page = bf.page();
        u64::from_ne_bytes(
            page[LEFT_SIBLING_OFFSET..LEFT_SIBLING_OFFSET + 8]
                .try_into()
                .unwrap(),
        )
    }

    #[cfg(test)]
    pub(crate) fn lookup_inner_swip(bf: BufferFrameReadRef<'_>, key: &[u8]) -> Swip {
        let sp = Self::sp(bf);
        let (pos, _) = sp.lower_bound(key);
        if pos == sp.num_slots() {
            Self::upper_swip(bf)
        } else {
            let val = sp.get_value(pos);
            Swip::from_raw(u64::from_ne_bytes(val[..8].try_into().unwrap()))
        }
    }

    pub(crate) fn child_swip_at(bf: BufferFrameReadRef<'_>, pos: u16) -> Swip {
        let sp = Self::sp(bf);
        let val = sp.get_value(pos);
        Swip::from_raw(u64::from_ne_bytes(val[..8].try_into().unwrap()))
    }

    pub(crate) fn insert_inner_at(
        bf: BufferFrameWriteRef<'_>,
        pos: u16,
        key: &[u8],
        child_swip: Swip,
    ) {
        let sp = Self::sp_mut(bf);
        sp.insert(pos, key, &child_swip.raw().to_ne_bytes());
    }

    pub(crate) fn set_child_swip_at(bf: BufferFrameWriteRef<'_>, pos: u16, child_swip: Swip) {
        let sp = Self::sp_mut(bf);
        let ok = sp.update_value_if_same_length(pos, &child_swip.raw().to_ne_bytes());
        debug_assert!(ok, "child swip value must remain 8 bytes");
    }

    pub(crate) fn replace_inner_key(bf: BufferFrameWriteRef<'_>, pos: u16, key: &[u8]) {
        let child_swip = Self::child_swip_at(bf.read_ref(), pos);
        let sp = Self::sp_mut(bf);
        sp.remove(pos);
        sp.insert(pos, key, &child_swip.raw().to_ne_bytes());
    }

    pub(crate) fn can_insert_inner(bf: BufferFrameReadRef<'_>, key_len: usize) -> bool {
        Self::sp(bf).can_insert(key_len, 8)
    }

    pub(crate) fn is_underfull(bf: BufferFrameReadRef<'_>) -> bool {
        Self::sp(bf).free_space_after_compaction() > UNDERFULL_THRESHOLD
    }
}

#[repr(C, align(4096))]
pub(crate) struct TmpBuf(pub(crate) [u8; PAGE_SIZE]);

impl TmpBuf {
    pub(crate) fn new() -> Self {
        TmpBuf([0u8; PAGE_SIZE])
    }
}

pub(crate) struct Leaf;
pub(crate) struct Inner;

#[derive(Clone, Copy)]
pub(crate) struct RoutedChild {
    swip: Swip,
    edge: ParentEdge,
}

impl RoutedChild {
    pub(crate) fn new(swip: Swip, edge: ParentEdge) -> Self {
        Self { swip, edge }
    }

    pub(crate) fn swip(self) -> Swip {
        self.swip
    }

    pub(crate) fn edge(self) -> ParentEdge {
        self.edge
    }

    pub(crate) fn slot_index(self, count: u16) -> u16 {
        self.edge.pos(count)
    }

    pub(crate) fn is_upper(self) -> bool {
        self.edge.is_upper()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ResidentFrame {
    bf: BufferFrameRef,
}

impl ResidentFrame {
    pub(crate) fn new(bf: BufferFrameRef) -> Self {
        Self { bf }
    }

    pub(crate) fn from_hot_swip(swip: Swip) -> Option<Self> {
        BufferFrameRef::from_hot_swip(swip).map(|bf| Self { bf })
    }

    pub(crate) fn from_pinned(frame: &PinnedFrame<'_>) -> Self {
        Self {
            bf: frame.frame_ref(),
        }
    }

    pub(crate) fn from_optimistic(frame: &OptimisticFrame<'_>) -> Self {
        Self {
            bf: frame.frame_ref(),
        }
    }

    pub(crate) fn from_shared(frame: &SharedFrame<'_>) -> Self {
        Self {
            bf: frame.frame_ref(),
        }
    }

    pub(crate) fn from_exclusive(frame: &ExclusiveFrame<'_>) -> Self {
        Self {
            bf: frame.frame_ref(),
        }
    }

    pub(crate) fn same_frame(self, other: ResidentFrame) -> bool {
        self.bf.same_frame(other.bf)
    }

    pub(crate) fn is_in_pool(self, pool: &BufferPool) -> bool {
        pool.contains_frame(self.bf)
    }

    pub(crate) fn optimistic_guard<'a>(self) -> Result<OptimisticGuard<'a>, Restart> {
        unsafe { self.bf.optimistic_guard() }
    }

    fn read_ref<'a>(self) -> BufferFrameReadRef<'a> {
        unsafe { self.bf.read_ref() }
    }

    fn write_ref<'a>(self) -> BufferFrameWriteRef<'a> {
        unsafe { self.bf.write_ref() }
    }

    pub(crate) fn pid(self) -> u64 {
        self.bf.pid()
    }

    pub(crate) fn state(self) -> FrameState {
        self.bf.state()
    }

    pub(crate) fn init(self, is_leaf: bool) {
        BTreeNode::init(self.write_ref(), is_leaf)
    }

    pub(crate) fn hot_swip(self) -> Swip {
        self.bf.hot_swip()
    }

    pub(crate) fn is_leaf(self) -> bool {
        BTreeNode::is_leaf(self.read_ref())
    }

    pub(crate) fn sp<'a>(self) -> &'a SlottedPage {
        BTreeNode::sp(self.read_ref())
    }

    pub(crate) fn sp_mut<'a>(self) -> &'a mut SlottedPage {
        BTreeNode::sp_mut(self.write_ref())
    }

    pub(crate) fn num_slots(self) -> u16 {
        self.sp().num_slots()
    }

    pub(crate) fn lower_bound(self, key: &[u8]) -> (u16, bool) {
        self.sp().lower_bound(key)
    }

    pub(crate) fn try_lower_bound(self, key: &[u8]) -> Option<(u16, bool)> {
        self.sp().try_lower_bound(key)
    }

    pub(crate) fn get_key<'a>(self, pos: u16) -> &'a [u8] {
        self.sp().get_key(pos)
    }

    pub(crate) fn try_get_key<'a>(self, pos: u16) -> Option<&'a [u8]> {
        self.sp().try_get_key(pos)
    }

    pub(crate) fn get_value<'a>(self, pos: u16) -> &'a [u8] {
        self.sp().get_value(pos)
    }

    pub(crate) fn try_get_value<'a>(self, pos: u16) -> Option<&'a [u8]> {
        self.sp().try_get_value(pos)
    }

    pub(crate) fn can_insert(self, key_len: usize, value_len: usize) -> bool {
        self.sp().can_insert(key_len, value_len)
    }

    pub(crate) fn insert(self, pos: u16, key: &[u8], value: &[u8]) {
        self.sp_mut().insert(pos, key, value);
    }

    pub(crate) fn remove_slot(self, pos: u16) {
        self.sp_mut().remove(pos);
    }

    pub(crate) fn update_value_if_same_length(self, pos: u16, value: &[u8]) -> bool {
        self.sp_mut().update_value_if_same_length(pos, value)
    }

    pub(crate) fn free_space_after_compaction(self) -> usize {
        self.sp().free_space_after_compaction()
    }

    pub(crate) fn leaf_right_pid(self) -> u64 {
        BTreeNode::leaf_right_pid(self.read_ref())
    }

    pub(crate) fn leaf_left_pid(self) -> u64 {
        BTreeNode::leaf_left_pid(self.read_ref())
    }

    pub(crate) fn set_leaf_right_pid(self, pid: u64) {
        BTreeNode::set_leaf_right_pid(self.write_ref(), pid)
    }

    pub(crate) fn set_leaf_left_pid(self, pid: u64) {
        BTreeNode::set_leaf_left_pid(self.write_ref(), pid)
    }

    pub(crate) fn upper_swip(self) -> Swip {
        BTreeNode::upper_swip(self.read_ref())
    }

    pub(crate) fn set_upper(self, swip: Swip) {
        BTreeNode::set_upper(self.write_ref(), swip)
    }

    pub(crate) fn child_swip_at(self, pos: u16) -> Swip {
        BTreeNode::child_swip_at(self.read_ref(), pos)
    }

    pub(crate) fn set_child_swip_at(self, pos: u16, swip: Swip) {
        BTreeNode::set_child_swip_at(self.write_ref(), pos, swip)
    }

    pub(crate) fn replace_inner_key(self, pos: u16, key: &[u8]) {
        BTreeNode::replace_inner_key(self.write_ref(), pos, key)
    }

    pub(crate) fn insert_inner_at(self, pos: u16, key: &[u8], child_swip: Swip) {
        BTreeNode::insert_inner_at(self.write_ref(), pos, key, child_swip)
    }

    pub(crate) fn can_insert_inner(self, key_len: usize) -> bool {
        BTreeNode::can_insert_inner(self.read_ref(), key_len)
    }

    pub(crate) fn is_underfull_inner(self) -> bool {
        BTreeNode::is_underfull(self.read_ref())
    }

    pub(crate) fn try_route_to_child(self, key: &[u8]) -> Option<RoutedChild> {
        let sp = self.sp();
        let count = sp.num_slots();
        let (pos, _) = sp.try_lower_bound(key)?;
        let edge = ParentEdge::from_pos(pos, count);
        let slot = parent_edge_round_trip_pos(pos, count);
        let swip = match edge {
            ParentEdge::Upper => self.upper_swip(),
            ParentEdge::Slot(_) => {
                let val = sp.try_get_value(slot)?;
                if val.len() < 8 {
                    return None;
                }
                Swip::from_raw(u64::from_ne_bytes(val[..8].try_into().unwrap()))
            }
        };
        Some(RoutedChild::new(swip, edge))
    }

    pub(crate) fn is_empty_leaf(self) -> bool {
        self.sp().num_slots() == 0
    }

    pub(crate) fn is_underfull(self) -> bool {
        BTreeNode::is_underfull(self.read_ref())
    }

    pub(crate) fn should_chase_right(self, key: &[u8]) -> bool {
        let sp = self.sp();
        let right_pid = self.leaf_right_pid();
        if right_pid == 0 {
            return false;
        }
        if sp.num_slots() == 0 {
            return true;
        }
        let last_pos = sp.num_slots() - 1;
        match sp.try_get_key(last_pos) {
            Some(last_key) => key > last_key,
            None => false,
        }
    }

    pub(crate) fn set_parent_link_none(self) {
        self.write_ref().set_parent_link_none();
    }

    pub(crate) fn set_parent_link_stable(self, meta_swip: StableSwipRef) {
        self.write_ref().set_parent_link_stable(meta_swip);
    }

    pub(crate) fn set_parent_link_inner(
        self,
        parent_pid: u64,
        slot_index: u16,
        is_upper: bool,
        dt_id: u16,
    ) {
        let current = self.read_ref().parent_link();
        if let ParentLink::InnerNode(link) = current
            && link.parent_pid == parent_pid
            && link.slot_index == slot_index
            && link.is_upper == is_upper
            && link.dt_id == dt_id
        {
            return;
        }
        self.write_ref()
            .set_parent_link_inner(parent_pid, slot_index, is_upper, dt_id);
    }

    pub(crate) fn mark_header_dirty(self) {
        self.write_ref().mark_header_dirty();
    }

    pub(crate) fn replace_page(self, page: &[u8; PAGE_SIZE]) {
        self.write_ref().page_mut().copy_from_slice(page);
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ChildRef {
    frame: ResidentFrame,
    pid: u64,
}

impl ChildRef {
    pub(crate) fn from_frame(frame: ResidentFrame) -> Self {
        Self {
            frame,
            pid: frame.pid(),
        }
    }

    fn matches_swip(self, swip: Swip) -> bool {
        (ResidentFrame::from_hot_swip(swip).is_some_and(|frame| frame.same_frame(self.frame)))
            || (swip.is_evicted() && swip.as_page_id() == self.pid)
    }
}

pub(crate) struct OptimisticNode<'a, Kind> {
    frame: OptimisticFrame<'a>,
    _kind: PhantomData<Kind>,
}

pub(crate) struct SharedNode<'a, Kind> {
    frame: SharedFrame<'a>,
    _kind: PhantomData<Kind>,
}

pub(crate) struct ExclusiveNode<'a, Kind> {
    frame: ExclusiveFrame<'a>,
    _kind: PhantomData<Kind>,
}

impl<'a> OptimisticNode<'a, Leaf> {
    pub(crate) fn from_leaf_frame(frame: OptimisticFrame<'a>) -> Self {
        Self {
            frame,
            _kind: PhantomData,
        }
    }

    pub(crate) fn resident_frame(&self) -> ResidentFrame {
        ResidentFrame::from_optimistic(&self.frame)
    }

    pub(crate) fn validate(&self) -> Result<(), Restart> {
        self.frame.validate()
    }

    pub(crate) fn should_chase_right(&self, key: &[u8]) -> bool {
        self.resident_frame().should_chase_right(key)
    }

    pub(crate) fn right_pid(&self) -> u64 {
        self.resident_frame().leaf_right_pid()
    }

    pub(crate) fn try_lower_bound(&self, key: &[u8]) -> Option<(u16, bool)> {
        self.resident_frame().try_lower_bound(key)
    }

    pub(crate) fn try_value_at(&self, pos: u16) -> Option<&'a [u8]> {
        self.resident_frame().try_get_value(pos)
    }

    pub(crate) fn upgrade_to_exclusive(self) -> Result<ExclusiveNode<'a, Leaf>, PinnedFrame<'a>> {
        let frame = self.frame.upgrade_to_exclusive()?;
        Ok(ExclusiveNode {
            frame,
            _kind: PhantomData,
        })
    }

    pub(crate) fn upgrade_to_shared(self) -> Result<SharedNode<'a, Leaf>, PinnedFrame<'a>> {
        let frame = self.frame.upgrade_to_shared()?;
        Ok(SharedNode {
            frame,
            _kind: PhantomData,
        })
    }
}

impl<'a> OptimisticNode<'a, Inner> {
    pub(crate) fn from_inner_frame(frame: OptimisticFrame<'a>) -> Self {
        Self {
            frame,
            _kind: PhantomData,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), Restart> {
        self.frame.validate()
    }

    pub(crate) fn route_to_child(&self, key: &[u8]) -> Option<RoutedChild> {
        ResidentFrame::from_optimistic(&self.frame).try_route_to_child(key)
    }

    pub(crate) fn child_edge_for(&self, child: ChildRef) -> Option<ParentEdge> {
        let frame = ResidentFrame::from_optimistic(&self.frame);
        let sp = frame.sp();
        let count = sp.num_slots();
        if count as usize > PAGE_SIZE / 12 {
            return None;
        }
        for pos in 0..count {
            let val = match sp.try_get_value(pos) {
                Some(val) if val.len() >= 8 => val,
                _ => return None,
            };
            let swip = Swip::from_raw(u64::from_ne_bytes(val[..8].try_into().unwrap()));
            if child.matches_swip(swip) {
                return Some(ParentEdge::Slot(pos));
            }
        }

        let upper = frame.upper_swip();
        if child.matches_swip(upper) {
            return Some(ParentEdge::Upper);
        }
        None
    }

    pub(crate) fn for_each_child_route(&self, mut f: impl FnMut(RoutedChild)) -> Option<()> {
        let frame = ResidentFrame::from_optimistic(&self.frame);
        let sp = frame.sp();
        let count = sp.num_slots();
        if count as usize > PAGE_SIZE / 12 {
            return None;
        }

        for pos in 0..count {
            let val = match sp.try_get_value(pos) {
                Some(val) if val.len() >= 8 => val,
                _ => return None,
            };
            let swip = Swip::from_raw(u64::from_ne_bytes(val[..8].try_into().unwrap()));
            f(RoutedChild::new(swip, ParentEdge::Slot(pos)));
        }
        f(RoutedChild::new(frame.upper_swip(), ParentEdge::Upper));
        Some(())
    }

    pub(crate) fn upgrade_to_exclusive(self) -> Result<ExclusiveNode<'a, Inner>, PinnedFrame<'a>> {
        let frame = self.frame.upgrade_to_exclusive()?;
        Ok(ExclusiveNode {
            frame,
            _kind: PhantomData,
        })
    }
}

impl<'a> SharedNode<'a, Leaf> {
    pub(crate) fn from_leaf_frame(frame: SharedFrame<'a>) -> Self {
        Self {
            frame,
            _kind: PhantomData,
        }
    }

    pub(crate) fn resident_frame(&self) -> ResidentFrame {
        ResidentFrame::from_shared(&self.frame)
    }

    pub(crate) fn lower_bound(&self, key: &[u8]) -> (u16, bool) {
        self.resident_frame().lower_bound(key)
    }

    pub(crate) fn try_lower_bound(&self, key: &[u8]) -> Option<(u16, bool)> {
        self.resident_frame().try_lower_bound(key)
    }

    pub(crate) fn num_slots(&self) -> u16 {
        self.resident_frame().num_slots()
    }

    pub(crate) fn key_at(&self, pos: u16) -> &'a [u8] {
        self.resident_frame().get_key(pos)
    }

    pub(crate) fn try_key_at(&self, pos: u16) -> Option<&'a [u8]> {
        self.resident_frame().try_get_key(pos)
    }

    pub(crate) fn value_at(&self, pos: u16) -> &'a [u8] {
        self.resident_frame().get_value(pos)
    }

    pub(crate) fn try_value_at(&self, pos: u16) -> Option<&'a [u8]> {
        self.resident_frame().try_get_value(pos)
    }

    pub(crate) fn right_pid(&self) -> u64 {
        self.resident_frame().leaf_right_pid()
    }

    pub(crate) fn left_pid(&self) -> u64 {
        self.resident_frame().leaf_left_pid()
    }
}

impl<'a, Kind> ExclusiveNode<'a, Kind> {
    pub(crate) fn resident_frame(&self) -> ResidentFrame {
        ResidentFrame::from_exclusive(&self.frame)
    }

    pub(crate) fn mark_dirty(&self) {
        self.frame.mark_dirty();
    }

    pub(crate) fn into_frame(self) -> ExclusiveFrame<'a> {
        let this = std::mem::ManuallyDrop::new(self);
        unsafe { std::ptr::read(&this.frame) }
    }

    pub(crate) fn into_pinned(self) -> PinnedFrame<'a> {
        self.into_frame().into_pinned()
    }
}

impl<'a> ExclusiveNode<'a, Leaf> {
    pub(crate) fn from_leaf_frame(frame: ExclusiveFrame<'a>) -> Self {
        Self {
            frame,
            _kind: PhantomData,
        }
    }

    pub(crate) fn pid(&self) -> u64 {
        self.resident_frame().pid()
    }

    pub(crate) fn right_pid(&self) -> u64 {
        self.resident_frame().leaf_right_pid()
    }

    pub(crate) fn left_pid(&self) -> u64 {
        self.resident_frame().leaf_left_pid()
    }

    pub(crate) fn num_slots(&self) -> u16 {
        self.resident_frame().num_slots()
    }

    pub(crate) fn lower_bound(&self, key: &[u8]) -> (u16, bool) {
        self.resident_frame().lower_bound(key)
    }

    pub(crate) fn key_at(&self, pos: u16) -> &'a [u8] {
        self.resident_frame().get_key(pos)
    }

    pub(crate) fn value_at(&self, pos: u16) -> &'a [u8] {
        self.resident_frame().get_value(pos)
    }

    pub(crate) fn can_insert_entry(&self, key_len: usize, value_len: usize) -> bool {
        self.resident_frame().can_insert(key_len, value_len)
    }

    pub(crate) fn insert_entry(&self, pos: u16, key: &[u8], value: &[u8]) {
        self.resident_frame().insert(pos, key, value);
    }

    pub(crate) fn remove_slot(&self, pos: u16) {
        self.resident_frame().remove_slot(pos);
    }

    pub(crate) fn update_value_if_same_length(&self, pos: u16, value: &[u8]) -> bool {
        self.resident_frame()
            .update_value_if_same_length(pos, value)
    }

    pub(crate) fn free_space_after_compaction(&self) -> usize {
        self.resident_frame().free_space_after_compaction()
    }

    pub(crate) fn is_underfull(&self) -> bool {
        self.resident_frame().is_underfull()
    }
}

impl<'a> ExclusiveNode<'a, Inner> {
    pub(crate) fn from_inner_frame(frame: ExclusiveFrame<'a>) -> Self {
        Self {
            frame,
            _kind: PhantomData,
        }
    }

    pub(crate) fn num_slots(&self) -> u16 {
        self.resident_frame().sp().num_slots()
    }

    pub(crate) fn pid(&self) -> u64 {
        self.resident_frame().pid()
    }

    pub(crate) fn is_underfull(&self) -> bool {
        self.resident_frame().is_underfull_inner()
    }

    pub(crate) fn child_edge_swip(&self, edge: ParentEdge) -> Swip {
        match edge {
            ParentEdge::Slot(pos) => self.resident_frame().child_swip_at(pos),
            ParentEdge::Upper => self.resident_frame().upper_swip(),
        }
    }

    pub(crate) fn child_edge_for(&self, child: ChildRef) -> Option<ParentEdge> {
        let count = self.num_slots();
        for pos in 0..count {
            if child.matches_swip(self.resident_frame().child_swip_at(pos)) {
                return Some(ParentEdge::Slot(pos));
            }
        }

        if child.matches_swip(self.resident_frame().upper_swip()) {
            return Some(ParentEdge::Upper);
        }
        None
    }

    pub(crate) fn child_routes(&self) -> Vec<RoutedChild> {
        let count = self.num_slots();
        let mut routes = Vec::with_capacity(count as usize + 1);
        for pos in 0..count {
            routes.push(RoutedChild::new(
                self.resident_frame().child_swip_at(pos),
                ParentEdge::Slot(pos),
            ));
        }
        routes.push(RoutedChild::new(
            self.resident_frame().upper_swip(),
            ParentEdge::Upper,
        ));
        routes
    }

    pub(crate) fn child_page_ids(&self) -> Vec<u64> {
        self.child_routes()
            .into_iter()
            .map(|route| {
                let swip = route.swip();
                if swip.is_hot() || swip.is_cool() {
                    ResidentFrame::from_hot_swip(swip).unwrap().pid()
                } else {
                    swip.as_page_id()
                }
            })
            .collect()
    }

    pub(crate) fn child_edge_matches(&self, edge: ParentEdge, child: ChildRef) -> bool {
        child.matches_swip(self.child_edge_swip(edge))
    }

    pub(crate) fn key_at(&self, pos: u16) -> &'a [u8] {
        self.resident_frame().sp().get_key(pos)
    }

    pub(crate) fn set_child_edge_swip(&self, edge: ParentEdge, swip: Swip) {
        match edge {
            ParentEdge::Slot(pos) => self.resident_frame().set_child_swip_at(pos, swip),
            ParentEdge::Upper => self.resident_frame().set_upper(swip),
        }
    }

    pub(crate) fn can_insert_separator(&self, key_len: usize) -> bool {
        self.resident_frame().can_insert_inner(key_len)
    }

    pub(crate) fn insert_separator(&self, pos: u16, key: &[u8], child_swip: Swip) {
        self.resident_frame().insert_inner_at(pos, key, child_swip);
    }

    pub(crate) fn set_separator_key(&self, pos: u16, key: &[u8]) {
        self.resident_frame().replace_inner_key(pos, key);
    }

    pub(crate) fn remove_slot(&self, pos: u16) {
        self.resident_frame().sp_mut().remove(pos);
    }
}
