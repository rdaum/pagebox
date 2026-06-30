use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Bound;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

#[cfg(not(feature = "metrics"))]
use crate::metrics_stub::MetricVisitor;
#[cfg(feature = "metrics")]
use fast_telemetry::MetricVisitor;
use pagebox_hybrid_latch::HybridLatch;
use pagebox_storage::buffer_frame::{
    BufferFrame, BufferFrameRef, Lsn, PAGE_SIZE, PageWritebackPreparer, ParentFinder,
    StableSwipRef, page_size,
};
use pagebox_storage::buffer_pool::{BufferPool, BufferPoolHandle, ExclusiveFrame, PinnedFrame};
use pagebox_storage::page_header::{self, PageType};
use pagebox_swip_kernel::{AtomicSwipWord as AtomicSwip, SwipWord as Swip};

use crate::message::{
    BufferedMessage, CowBeTreeMessage, Timestamp, VersionRecord, sort_buffer_messages,
};
use crate::page::{
    CowBeTreeError, Fence, LeafEntry, LeafPageReader, LookupStep, NodePage, PageKindDebug,
    RawVisibleVersion, append_internal_buffer_kv, append_internal_buffer_message,
    append_leaf_entry_message, append_leaf_entry_prefix, append_leaf_kv, apply_message_to_entries,
    buffer_encoded_len, decode_page, encode_internal_page, encode_leaf_page, encoded_page_len,
    internal_child_array_range, leaf_should_chase_right, lookup_child_slot, lookup_step,
    lower_bound_entries, read_child_swip_at, read_right_sibling, route_child,
    split_leaf_into_pages, write_child_swip_at, write_right_sibling,
};
use crate::stats::{CowBeTreeEvent, CowBeTreeStats};

#[derive(Clone, Copy, Debug)]
pub struct CowBeTreeConfig {
    pub flush_threshold_messages: usize,
    pub flush_threshold_bytes: usize,
    pub max_leaf_entries: usize,
    pub max_internal_children: usize,
    pub merge_leaf_entries: usize,
    pub merge_internal_children: usize,
}

impl Default for CowBeTreeConfig {
    fn default() -> Self {
        let flush_threshold_bytes = default_flush_threshold_bytes(PAGE_SIZE);
        Self {
            flush_threshold_messages: default_flush_threshold_messages(flush_threshold_bytes),
            flush_threshold_bytes,
            max_leaf_entries: usize::MAX,
            max_internal_children: default_internal_children(PAGE_SIZE),
            merge_leaf_entries: usize::MAX,
            merge_internal_children: default_internal_children(PAGE_SIZE) * 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CowBeTreeDebugState {
    pub root_kind: Option<PageKindDebug>,
    pub height: u32,
    pub leaf_pages: usize,
    pub internal_pages: usize,
    pub leaf_entries: usize,
    pub buffered_messages: usize,
    pub max_buffered_messages_on_page: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CowBeTreeVisibleVersion {
    pub commit_ts: Timestamp,
    pub deleted: bool,
    pub value: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CowBeTreeGcResult {
    pub versions_pruned: usize,
    pub leaf_pages_visited: usize,
    pub leaf_pages_rewritten: usize,
    pub version_bytes_pruned: usize,
    pub cursor_wrapped: bool,
    pub budget_exhausted: bool,
}

impl CowBeTreeGcResult {
    fn add_assign(&mut self, other: Self) {
        self.versions_pruned += other.versions_pruned;
        self.leaf_pages_visited += other.leaf_pages_visited;
        self.leaf_pages_rewritten += other.leaf_pages_rewritten;
        self.version_bytes_pruned += other.version_bytes_pruned;
        self.cursor_wrapped |= other.cursor_wrapped;
        self.budget_exhausted |= other.budget_exhausted;
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CowBeTreeGcCursor {
    next_key: Option<Vec<u8>>,
}

impl CowBeTreeGcCursor {
    pub fn next_key(&self) -> Option<&[u8]> {
        self.next_key.as_deref()
    }

    pub fn reset(&mut self) {
        self.next_key = None;
    }
}

#[derive(Debug)]
enum Rewrite {
    One {
        pid: u64,
    },
    Split {
        left_pid: u64,
        right_pid: u64,
        separator: Vec<u8>,
    },
}

#[derive(Clone, Debug)]
struct DebugWalk {
    snapshot: CowBeTreeDebugState,
}

#[derive(Clone, Debug)]
struct VisibleCandidate {
    commit_ts: Timestamp,
    deleted: bool,
    value: Vec<u8>,
}

impl From<VisibleCandidate> for CowBeTreeVisibleVersion {
    fn from(candidate: VisibleCandidate) -> Self {
        Self {
            commit_ts: candidate.commit_ts,
            deleted: candidate.deleted,
            value: candidate.value,
        }
    }
}

enum VisibleLookupStep {
    Leaf {
        visible: Option<VisibleCandidate>,
    },
    Internal {
        parent_pid: u64,
        child_swip: Swip,
        child_slot: u16,
        visible_buffer: Option<VisibleCandidate>,
        buffer_count: usize,
    },
}

enum WriteRouteStep {
    Leaf,
    Internal { child_swip: Swip },
}

struct IncrementalGcState<'a> {
    watermark: Timestamp,
    lower: Option<&'a [u8]>,
    remaining_leaf_pages: usize,
    result: CowBeTreeGcResult,
    next_key: Option<Vec<u8>>,
    reached_end: bool,
}

struct ForkRegistry {
    active_roots: AtomicUsize,
    shared_pages: RwLock<HashMap<u64, usize>>,
}

impl ForkRegistry {
    fn new() -> Self {
        Self {
            active_roots: AtomicUsize::new(1),
            shared_pages: RwLock::new(HashMap::new()),
        }
    }
}

const BETREE_DT_ID: u16 = 2;

pub struct CowBeTree {
    pool: BufferPoolHandle,
    root: Box<AtomicSwip>,
    write_latch: HybridLatch,
    append_hint: AtomicU64,
    forks: Arc<ForkRegistry>,
    stats: CowBeTreeStats,
    config: CowBeTreeConfig,
    dt_id: u16,
}

impl CowBeTree {
    pub fn new<P>(pool: P) -> Self
    where
        P: Into<BufferPoolHandle>,
    {
        Self::with_config(pool, CowBeTreeConfig::default())
    }

    pub fn with_config<P>(pool: P, config: CowBeTreeConfig) -> Self
    where
        P: Into<BufferPoolHandle>,
    {
        let pool = pool.into();
        register_internal_writeback_preparer(pool.as_pool());
        let stats = CowBeTreeStats::new(counter_shards());
        let mut image = vec![0u8; PAGE_SIZE];
        let bytes = encode_leaf_page(&mut image, &Fence::root(), &[])
            .expect("empty CoW B-epsilon root leaf must fit");
        let (pid, frame) = pool.allocate_and_fix();
        let mut frame = frame.exclusive();
        frame.page_bytes_mut().copy_from_slice(&image);
        frame.mark_dirty();
        stats.inc(CowBeTreeEvent::CowPagesAllocated);
        stats.add_leaf_bytes(bytes);

        let root = Box::new(AtomicSwip::new(frame.hot_swip()));
        frame.set_parent_link_stable(unsafe { StableSwipRef::from_ref(root.as_ref()) });
        drop(frame);

        let proxy = Arc::new(DtProxy { pool: pool.clone() });
        pool.as_pool().register_dt(BETREE_DT_ID, proxy);
        Self {
            pool,
            root,
            write_latch: HybridLatch::new(),
            append_hint: AtomicU64::new(pid),
            forks: Arc::new(ForkRegistry::new()),
            stats,
            config,
            dt_id: BETREE_DT_ID,
        }
    }

    pub fn open<P>(pool: P, root_page_id: u64) -> Self
    where
        P: Into<BufferPoolHandle>,
    {
        Self::open_with_config(pool, root_page_id, CowBeTreeConfig::default())
    }

    pub fn open_with_config<P>(pool: P, root_page_id: u64, config: CowBeTreeConfig) -> Self
    where
        P: Into<BufferPoolHandle>,
    {
        let pool = pool.into();
        register_internal_writeback_preparer(pool.as_pool());
        let proxy = Arc::new(DtProxy { pool: pool.clone() });
        pool.as_pool().register_dt(BETREE_DT_ID, proxy);
        Self {
            pool,
            root: Box::new(AtomicSwip::new(Swip::evicted(root_page_id))),
            write_latch: HybridLatch::new(),
            append_hint: AtomicU64::new(root_page_id),
            forks: Arc::new(ForkRegistry::new()),
            stats: CowBeTreeStats::new(counter_shards()),
            config,
            dt_id: BETREE_DT_ID,
        }
    }

    pub fn fork(&self) -> Result<Self, CowBeTreeError> {
        let _write_guard = self.write_latch.lock_exclusive();
        let root_page_id = self.root_page_id();
        self.mark_reachable_pages_shared(root_page_id)?;
        self.forks.active_roots.fetch_add(1, Ordering::AcqRel);
        self.stats.inc(CowBeTreeEvent::Forks);

        Ok(Self {
            pool: self.pool.clone(),
            root: Box::new(AtomicSwip::new(Swip::evicted(root_page_id))),
            write_latch: HybridLatch::new(),
            append_hint: AtomicU64::new(root_page_id),
            forks: Arc::clone(&self.forks),
            stats: CowBeTreeStats::new(counter_shards()),
            config: self.config,
            dt_id: self.dt_id,
        })
    }

    pub fn root_page_id(&self) -> u64 {
        swip_page_id(self.root.load(Ordering::Acquire))
    }

    /// Return every page currently reachable from the tree root.
    ///
    /// This is intended for ownership accounting during table retirement,
    /// not for concurrent query execution. Callers should only use it when the
    /// tree is quiescent or otherwise protected from mutation.
    pub fn owned_page_ids(&self) -> Result<Vec<u64>, CowBeTreeError> {
        let mut pages = Vec::new();
        let mut visited = HashSet::new();
        self.collect_reachable_pages(self.root_page_id(), &mut visited, &mut pages)?;
        pages.sort_unstable();
        Ok(pages)
    }

    pub fn visit_metrics<V: MetricVisitor + ?Sized>(&self, visitor: &mut V) {
        self.stats.visit_metrics(visitor);
    }

    pub fn height(&self) -> u32 {
        self.debug_snapshot().height
    }

    pub fn debug_snapshot(&self) -> CowBeTreeDebugState {
        let Ok(root) = self.load_root() else {
            return CowBeTreeDebugState::default();
        };
        let root_kind = root.kind().debug();
        let mut walk = DebugWalk {
            snapshot: CowBeTreeDebugState {
                root_kind: Some(root_kind),
                ..CowBeTreeDebugState::default()
            },
        };
        self.debug_walk(root, 0, &mut walk);
        walk.snapshot
    }

    pub fn insert_message(&self, message: CowBeTreeMessage) -> Result<(), CowBeTreeError> {
        self.write_message(message.into_buffered(), None)
    }

    pub fn insert_message_with_lsn(
        &self,
        message: CowBeTreeMessage,
        lsn: Lsn,
    ) -> Result<(), CowBeTreeError> {
        self.write_message(message.into_buffered(), Some(lsn))
    }

    pub fn insert_new_message(&self, message: CowBeTreeMessage) -> Result<(), CowBeTreeError> {
        self.write_new_message(message.into_buffered(), None)
    }

    pub fn insert_new_message_with_lsn(
        &self,
        message: CowBeTreeMessage,
        lsn: Lsn,
    ) -> Result<(), CowBeTreeError> {
        self.write_new_message(message.into_buffered(), Some(lsn))
    }

    pub fn put(
        &self,
        key: &[u8],
        commit_ts: Timestamp,
        value: &[u8],
    ) -> Result<(), CowBeTreeError> {
        if self.try_put_page_local(key, commit_ts, value)? {
            return Ok(());
        }
        let message = BufferedMessage::put(key, commit_ts, value);
        self.write_message_structural(message, None)
    }

    fn try_put_page_local(
        &self,
        key: &[u8],
        commit_ts: Timestamp,
        value: &[u8],
    ) -> Result<bool, CowBeTreeError> {
        if self.forks.active_roots.load(Ordering::Acquire) > 1 {
            return Ok(false);
        }

        let mut page_id = self.root_page_id();
        loop {
            match self.write_route_step(page_id, key)? {
                WriteRouteStep::Leaf => {
                    return self.try_append_leaf_kv_page_local(page_id, key, value, commit_ts);
                }
                WriteRouteStep::Internal { child_swip } => {
                    page_id = swip_page_id(child_swip);
                    match self.write_route_step(page_id, key)? {
                        WriteRouteStep::Leaf => {
                            return self
                                .try_append_leaf_kv_page_local(page_id, key, value, commit_ts);
                        }
                        WriteRouteStep::Internal { .. } => {
                            if self.try_append_buffer_kv(page_id, key, value, commit_ts)? {
                                return Ok(true);
                            }
                        }
                    }
                }
            }
        }
    }

    fn try_append_leaf_kv_page_local(
        &self,
        page_id: u64,
        key: &[u8],
        value: &[u8],
        commit_ts: Timestamp,
    ) -> Result<bool, CowBeTreeError> {
        let right = {
            let frame = unsafe { self.pool().fix_orphan_frame(page_id) }.shared();
            if leaf_should_chase_right(frame.page_bytes(), key) {
                read_right_sibling(frame.page_bytes())
            } else {
                0
            }
        };
        if right != 0 {
            return self.try_append_leaf_kv_page_local(right, key, value, commit_ts);
        }

        let mut frame = unsafe { self.pool().fix_orphan_frame(page_id) }.exclusive();
        if leaf_should_chase_right(frame.page_bytes(), key) {
            let right = read_right_sibling(frame.page_bytes());
            if right != 0 {
                drop(frame);
                return self.try_append_leaf_kv_page_local(right, key, value, commit_ts);
            }
        }
        if let Some(appended) = append_leaf_kv(
            frame.page_bytes_mut(),
            key,
            value,
            commit_ts,
            self.config.max_leaf_entries,
        )? {
            mark_frame_dirty(&frame, None);
            self.append_hint.store(page_id, Ordering::Release);
            self.stats
                .add(CowBeTreeEvent::RawLeafAppends, appended.message_count);
            self.stats.inc(CowBeTreeEvent::RawLeafAppendBatches);
            return Ok(true);
        }

        let step = lookup_step(frame.page_bytes(), key, Timestamp::MAX)?;
        match step {
            LookupStep::Leaf { visible: None } => return Ok(false),
            LookupStep::Internal { .. } => return Ok(false),
            LookupStep::Leaf { visible: Some(_) } => {}
        }

        let mut node = decode_page(frame.page_bytes())?;
        let NodePage::Leaf { fence, entries } = &mut node else {
            return Ok(false);
        };
        let message = BufferedMessage::put(key, commit_ts, value);
        apply_message_to_entries(entries, &message);
        if self.leaf_should_split(page_id, fence, entries) {
            return Ok(false);
        }

        let mut image = vec![0u8; PAGE_SIZE];
        let bytes = encode_leaf_page(&mut image, fence, entries)?;
        let page_image_bytes = image.len();
        frame.page_bytes_mut().copy_from_slice(&image);
        mark_frame_dirty(&frame, None);
        self.stats.inc(CowBeTreeEvent::InPlacePageRewrites);
        self.stats.add_leaf_bytes(bytes);
        self.stats
            .add_leaf_page_image_rewrite_bytes(page_image_bytes);
        Ok(true)
    }

    pub fn remove(&self, key: &[u8]) -> Result<bool, CowBeTreeError> {
        self.remove_at(key, Timestamp::MAX)
    }

    pub fn remove_at(&self, key: &[u8], commit_ts: Timestamp) -> Result<bool, CowBeTreeError> {
        let existed = self.lookup_visible(key, commit_ts).is_some();
        self.write_message(BufferedMessage::delete(key, commit_ts), None)?;
        Ok(existed)
    }

    pub fn flush_all(&self) -> Result<(), CowBeTreeError> {
        let _write_guard = self.write_latch.lock_exclusive();
        self.flush_all_root()?;
        self.append_hint
            .store(self.root_page_id(), Ordering::Release);
        Ok(())
    }

    pub fn compact(&self) -> Result<(), CowBeTreeError> {
        let _write_guard = self.write_latch.lock_exclusive();
        self.flush_all_root()?;
        let root_pid = self.root_page_id();
        let compact_root = self.compact_page(root_pid)?;
        self.install_root_rewrite(root_pid, compact_root, None)?;

        let root_pid = self.root_page_id();
        if let NodePage::Internal {
            children, buffer, ..
        } = self.load_orphan(root_pid)?
            && children.len() == 1
            && buffer.is_empty()
        {
            let child = self.load_orphan(children[0])?;
            self.write_node_at(root_pid, &child, None)?;
            self.stats.inc(CowBeTreeEvent::RootCollapses);
        }

        self.append_hint
            .store(self.root_page_id(), Ordering::Release);
        Ok(())
    }

    pub fn prune_versions(
        &self,
        watermark: Timestamp,
    ) -> Result<CowBeTreeGcResult, CowBeTreeError> {
        let _write_guard = self.write_latch.lock_exclusive();
        self.stats.inc(CowBeTreeEvent::GcRuns);

        self.flush_all_root()?;

        let root_pid = self.root_page_id();
        let mut result = CowBeTreeGcResult::default();
        let (rewrite, changed) = self.prune_versions_page(root_pid, watermark, &mut result)?;
        if changed {
            self.install_root_rewrite(root_pid, rewrite, None)?;
        }

        if result.versions_pruned > 0 {
            let root_pid = self.root_page_id();
            if let NodePage::Internal {
                children, buffer, ..
            } = self.load_orphan(root_pid)?
                && children.len() == 1
                && buffer.is_empty()
            {
                let child = self.load_orphan(children[0])?;
                self.write_node_at(root_pid, &child, None)?;
                self.stats.inc(CowBeTreeEvent::RootCollapses);
            }
        }

        self.record_gc_result(result);
        self.append_hint
            .store(self.root_page_id(), Ordering::Release);
        Ok(result)
    }

    pub fn prune_versions_incremental(
        &self,
        watermark: Timestamp,
        cursor: &mut CowBeTreeGcCursor,
        max_leaf_pages: usize,
    ) -> Result<CowBeTreeGcResult, CowBeTreeError> {
        if max_leaf_pages == 0 {
            let result = CowBeTreeGcResult {
                budget_exhausted: true,
                ..CowBeTreeGcResult::default()
            };
            self.stats.inc(CowBeTreeEvent::GcBudgetExhausted);
            return Ok(result);
        }

        let _write_guard = self.write_latch.lock_exclusive();
        self.stats.inc(CowBeTreeEvent::GcRuns);

        let lower = cursor.next_key.clone();
        let root_pid = self.root_page_id();
        let mut state = IncrementalGcState {
            watermark,
            lower: lower.as_deref(),
            remaining_leaf_pages: max_leaf_pages,
            result: CowBeTreeGcResult::default(),
            next_key: lower.clone(),
            reached_end: false,
        };
        let (rewrite, changed) = self.prune_versions_incremental_page(root_pid, &mut state)?;
        if changed {
            self.install_root_rewrite(root_pid, rewrite, None)?;
        }

        if state.reached_end {
            state.result.cursor_wrapped = true;
            cursor.reset();
            self.stats.inc(CowBeTreeEvent::GcCursorWraps);
        } else {
            state.result.budget_exhausted = state.remaining_leaf_pages == 0;
            cursor.next_key = state.next_key;
            if state.result.budget_exhausted {
                self.stats.inc(CowBeTreeEvent::GcBudgetExhausted);
            }
        }

        self.record_gc_result(state.result);
        self.append_hint
            .store(self.root_page_id(), Ordering::Release);
        Ok(state.result)
    }

    pub fn lookup(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.lookup_visible(key, Timestamp::MAX)
    }

    pub fn lookup_visible(&self, key: &[u8], read_ts: Timestamp) -> Option<Vec<u8>> {
        self.lookup_visible_version(key, read_ts)
            .and_then(|version| (!version.deleted).then_some(version.value))
    }

    pub fn lookup_visible_version(
        &self,
        key: &[u8],
        read_ts: Timestamp,
    ) -> Option<CowBeTreeVisibleVersion> {
        let mut root = true;
        let mut next_swip = Swip::evicted(0);
        let mut next_parent = 0u64;
        let mut visible = None;
        let mut saw_path_buffer = false;
        loop {
            let Ok(step) = (if root {
                self.lookup_root_step(key, read_ts)
            } else {
                self.lookup_orphan_step(next_parent, next_swip, key, read_ts)
            }) else {
                return self
                    .lookup_visible_candidate_owned(key, read_ts)
                    .map(CowBeTreeVisibleVersion::from);
            };
            root = false;

            match step {
                VisibleLookupStep::Leaf { visible: leaf } => {
                    if let Some(leaf) = leaf {
                        merge_owned_visible_candidate(&mut visible, leaf);
                    }
                    if saw_path_buffer {
                        self.stats.inc(CowBeTreeEvent::PathBufferMerges);
                    }
                    return visible.map(CowBeTreeVisibleVersion::from);
                }
                VisibleLookupStep::Internal {
                    parent_pid,
                    child_swip,
                    child_slot,
                    visible_buffer,
                    buffer_count,
                } => {
                    if buffer_count > 0 {
                        saw_path_buffer = true;
                    }
                    if let Some(buffer) = visible_buffer {
                        merge_owned_visible_candidate(&mut visible, buffer);
                    }
                    if child_swip.is_evicted() {
                        self.swizzle_child_read_only(
                            parent_pid,
                            child_swip.as_page_id(),
                            child_slot,
                        );
                    }
                    next_swip = child_swip;
                    next_parent = parent_pid;
                }
            }
        }
    }

    fn lookup_visible_candidate_owned(
        &self,
        key: &[u8],
        read_ts: Timestamp,
    ) -> Option<VisibleCandidate> {
        let mut node = self.load_root().ok()?;
        let mut path_messages: Vec<BufferedMessage> = Vec::new();
        loop {
            match node {
                NodePage::Leaf { entries, .. } => {
                    let pos = crate::page::lower_bound_entries(&entries, key);
                    let mut versions = if entries.get(pos).is_some_and(|entry| entry.key == key) {
                        entries[pos].versions.clone()
                    } else {
                        Vec::new()
                    };
                    for message in path_messages.iter().filter(|message| message.key == key) {
                        insert_version(&mut versions, message.version.clone());
                    }
                    if !path_messages.is_empty() {
                        self.stats.inc(CowBeTreeEvent::PathBufferMerges);
                    }
                    return visible_candidate_from_versions(&versions, read_ts);
                }
                NodePage::Internal {
                    children,
                    separators,
                    buffer,
                    ..
                } => {
                    path_messages.extend(buffer);
                    let idx = route_child(&separators, key);
                    let child_pid = *children.get(idx)?;
                    node = self.load_orphan(child_pid).ok()?;
                }
            }
        }
    }

    pub fn lookup_with<R>(&self, key: &[u8], f: impl FnOnce(Option<&[u8]>) -> R) -> R {
        let value = self.lookup(key);
        f(value.as_deref())
    }

    pub fn lookup_fixed<const N: usize>(&self, key: &[u8]) -> Option<[u8; N]> {
        let value = self.lookup(key)?;
        value.as_slice().try_into().ok()
    }

    pub fn scan_prefix<F>(&self, prefix: &[u8], mut f: F)
    where
        F: FnMut(&[u8], &[u8]),
    {
        self.scan_prefix_visible(prefix, Timestamp::MAX, |key, value| {
            f(key, value);
            true
        });
    }

    pub fn scan_prefix_visible<F>(&self, prefix: &[u8], read_ts: Timestamp, mut f: F)
    where
        F: FnMut(&[u8], &[u8]) -> bool,
    {
        self.scan_range_visible(Bound::Unbounded, Bound::Unbounded, read_ts, |key, value| {
            if key.starts_with(prefix) {
                return f(key, value);
            }
            true
        });
    }

    pub fn scan_range<F>(&self, lower: Bound<&[u8]>, upper: Bound<&[u8]>, mut f: F)
    where
        F: FnMut(&[u8], &[u8]),
    {
        self.scan_range_visible(lower, upper, Timestamp::MAX, |key, value| {
            f(key, value);
            true
        });
    }

    pub fn scan_range_visible<F>(
        &self,
        lower: Bound<&[u8]>,
        upper: Bound<&[u8]>,
        read_ts: Timestamp,
        mut f: F,
    ) where
        F: FnMut(&[u8], &[u8]) -> bool,
    {
        let Ok(root) = self.load_root() else {
            return;
        };
        let mut rows = BTreeMap::new();
        self.materialize_node(root, &[], &mut rows);

        for (key, versions) in rows {
            if !range_contains(&key, lower, upper) {
                continue;
            }
            let Some(value) = visible_from_versions(&versions, read_ts) else {
                continue;
            };
            if !f(&key, value) {
                return;
            }
        }
    }

    pub fn record_secondary_verification(&self) {
        self.stats.inc(CowBeTreeEvent::SecondaryVerifications);
    }

    fn write_message(
        &self,
        message: BufferedMessage,
        dirty_lsn: Option<Lsn>,
    ) -> Result<(), CowBeTreeError> {
        if self.try_write_message_page_local(&message, dirty_lsn)? {
            return Ok(());
        }
        self.write_message_structural(message, dirty_lsn)
    }

    fn write_new_message(
        &self,
        message: BufferedMessage,
        dirty_lsn: Option<Lsn>,
    ) -> Result<(), CowBeTreeError> {
        if self.try_write_new_message_page_local(&message, dirty_lsn)? {
            return Ok(());
        }
        self.write_message_structural(message, dirty_lsn)
    }

    fn write_message_structural(
        &self,
        message: BufferedMessage,
        dirty_lsn: Option<Lsn>,
    ) -> Result<(), CowBeTreeError> {
        let _write_guard = self.write_latch.lock_exclusive();
        let root_pid = self.ensure_root_mutable(dirty_lsn)?;
        if self.try_append_leaf_message(root_pid, &message, dirty_lsn)? {
            return Ok(());
        }
        if self.try_append_buffer_message(root_pid, &message, true, dirty_lsn)? {
            return Ok(());
        }

        if let Some(rewrite) = self.try_split_root_leaf_direct(root_pid, &message, dirty_lsn)? {
            return self.install_root_rewrite(root_pid, rewrite, dirty_lsn);
        }

        let root = self.load_orphan(root_pid)?;
        let rewrite = match root {
            NodePage::Leaf { fence, entries } => {
                self.rewrite_leaf_batch(root_pid, &fence, entries, &[message], dirty_lsn)?
            }
            NodePage::Internal {
                fence,
                children,
                separators,
                mut buffer,
            } => {
                buffer.push(message);
                self.stats.inc(CowBeTreeEvent::RootBufferAppends);
                if self.should_flush_buffer(&buffer)
                    || self.internal_should_split(root_pid, &fence, &children, &separators, &buffer)
                {
                    self.flush_internal(root_pid, fence, children, separators, buffer, dirty_lsn)?
                } else {
                    self.write_internal_at(
                        root_pid,
                        &fence,
                        &children,
                        &separators,
                        &buffer,
                        dirty_lsn,
                    )?;
                    let pid = root_pid;
                    Rewrite::One { pid }
                }
            }
        };
        self.install_root_rewrite(root_pid, rewrite, dirty_lsn)
    }

    /// Zero-allocation fast path for splitting a full root leaf when the
    /// message is a new-key put whose key sorts after the last entry.
    ///
    /// Reads the source page directly (no `decode_page`), copies entries into
    /// two new destination pages using `split_leaf_into_pages`, then appends
    /// the message to the right page using `append_leaf_kv`.  Returns
    /// `None` when the fast path conditions are not met (root is not a leaf,
    /// message is a delete or an update, key falls in the left page's range,
    /// etc.) so the caller falls back to the decode path.
    fn try_split_root_leaf_direct(
        &self,
        root_pid: u64,
        message: &BufferedMessage,
        dirty_lsn: Option<Lsn>,
    ) -> Result<Option<Rewrite>, CowBeTreeError> {
        let src_frame = unsafe { self.pool().fix_orphan_frame(root_pid) }.shared();
        let src = src_frame.page_bytes();

        let reader = match LeafPageReader::new(src) {
            Ok(r) => r,
            Err(_) => return Ok(None),
        };
        let entry_count = reader.entry_count();
        if entry_count < 2 {
            return Ok(None);
        }

        if message.version.deleted {
            return Ok(None);
        }
        let last_key = reader.entry_key(entry_count - 1)?;
        if message.key.as_slice() <= last_key {
            return Ok(None);
        }

        let mid = entry_count.div_ceil(2);
        if mid >= entry_count {
            return Ok(None);
        }

        let separator_key = reader.entry_key(mid)?;
        let root_fence = reader.fence()?;
        let left_fence = root_fence.left_of(separator_key.to_vec());
        let right_fence = root_fence.right_of(separator_key.to_vec());

        let (left_pid, left_frame_handle) = self.allocate_frame();
        let (right_pid, right_frame_handle) = self.allocate_frame();
        let mut left_frame = left_frame_handle.exclusive();
        let mut right_frame = right_frame_handle.exclusive();

        let split_result = split_leaf_into_pages(
            src,
            left_frame.page_bytes_mut(),
            right_frame.page_bytes_mut(),
            &left_fence,
            &right_fence,
            mid,
        )?;

        let appended = append_leaf_kv(
            right_frame.page_bytes_mut(),
            &message.key,
            &message.version.value,
            message.version.commit_ts,
            self.config.max_leaf_entries,
        )?;
        if appended.is_none() {
            return Ok(None);
        }

        left_frame.set_parent_link_none();
        mark_frame_dirty(&left_frame, dirty_lsn);
        right_frame.set_parent_link_none();
        mark_frame_dirty(&right_frame, dirty_lsn);

        drop(src_frame);
        drop(left_frame);
        drop(right_frame);

        self.append_hint.store(right_pid, Ordering::Release);
        self.stats.inc(CowBeTreeEvent::LeafSplits);
        self.stats.inc(CowBeTreeEvent::CowPagesAllocated);
        self.stats.inc(CowBeTreeEvent::CowPagesAllocated);
        self.stats
            .add_leaf_bytes(split_result.left_count + split_result.right_count + 1);

        Ok(Some(Rewrite::Split {
            left_pid,
            right_pid,
            separator: split_result.separator,
        }))
    }

    fn try_write_new_message_page_local(
        &self,
        message: &BufferedMessage,
        dirty_lsn: Option<Lsn>,
    ) -> Result<bool, CowBeTreeError> {
        if self.forks.active_roots.load(Ordering::Acquire) > 1 {
            return Ok(false);
        }

        let hint_page_id = self.append_hint.load(Ordering::Acquire);
        if hint_page_id != 0 && self.try_append_leaf_message(hint_page_id, message, dirty_lsn)? {
            return Ok(true);
        }

        let mut page_id = self.root_page_id();
        loop {
            match self.write_route_step(page_id, &message.key)? {
                WriteRouteStep::Leaf => {
                    return self.try_append_leaf_message(page_id, message, dirty_lsn);
                }
                WriteRouteStep::Internal { child_swip } => {
                    page_id = swip_page_id(child_swip);
                    match self.write_route_step(page_id, &message.key)? {
                        WriteRouteStep::Leaf => {
                            return self.try_append_leaf_message(page_id, message, dirty_lsn);
                        }
                        WriteRouteStep::Internal { .. } => {
                            if self.try_append_buffer_message(page_id, message, false, dirty_lsn)? {
                                return Ok(true);
                            }
                        }
                    }
                }
            }
        }
    }

    fn try_write_message_page_local(
        &self,
        message: &BufferedMessage,
        dirty_lsn: Option<Lsn>,
    ) -> Result<bool, CowBeTreeError> {
        if self.forks.active_roots.load(Ordering::Acquire) > 1 {
            return Ok(false);
        }

        let mut page_id = self.root_page_id();
        loop {
            match self.write_route_step(page_id, &message.key)? {
                WriteRouteStep::Leaf => {
                    return self.try_rewrite_leaf_page_local(page_id, message, dirty_lsn);
                }
                WriteRouteStep::Internal { child_swip } => {
                    page_id = swip_page_id(child_swip);
                    match self.write_route_step(page_id, &message.key)? {
                        WriteRouteStep::Leaf => {
                            return self.try_rewrite_leaf_page_local(page_id, message, dirty_lsn);
                        }
                        WriteRouteStep::Internal { .. } => {
                            if self.try_append_buffer_message(page_id, message, false, dirty_lsn)? {
                                return Ok(true);
                            }
                        }
                    }
                }
            }
        }
    }

    fn write_route_step(&self, page_id: u64, key: &[u8]) -> Result<WriteRouteStep, CowBeTreeError> {
        let frame = unsafe { self.pool().fix_orphan_frame(page_id) }.shared();
        Ok(match lookup_child_slot(frame.page_bytes(), key)? {
            None => WriteRouteStep::Leaf,
            Some((child_swip, child_slot)) => {
                if child_swip.is_evicted() {
                    self.swizzle_child_in_parent(page_id, child_swip.as_page_id(), child_slot);
                }
                WriteRouteStep::Internal { child_swip }
            }
        })
    }

    fn swizzle_child_in_parent(&self, parent_pid: u64, child_pid: u64, child_slot: u16) {
        let child = unsafe { self.pool().fix_orphan_frame(child_pid) }.exclusive();
        child
            .write_ref()
            .set_parent_link_inner(parent_pid, child_slot, false, self.dt_id);
        let hot_swip = child.write_ref().frame().hot_swip();
        let Some(parent_pin) = (unsafe { self.pool().try_fix_resident_page_frame(parent_pid) })
        else {
            return;
        };
        let Ok(mut parent) = parent_pin.try_exclusive() else {
            return;
        };
        if parent.pid() != parent_pid {
            return;
        }
        let page_bytes = parent.page_bytes();
        if page_header::read_page_type(page_bytes) != PageType::BeTreeInternal {
            return;
        }
        write_child_swip_at(parent.page_bytes_mut(), child_slot, hot_swip);
    }

    fn swizzle_child_read_only(&self, parent_pid: u64, child_pid: u64, child_slot: u16) {
        let child = unsafe { self.pool().fix_orphan_frame(child_pid) };
        let child_frame = child.frame_ref();
        let Some(parent_pin) = (unsafe { self.pool().try_fix_resident_page_frame(parent_pid) })
        else {
            return;
        };
        let Ok(mut parent) = parent_pin.try_exclusive() else {
            return;
        };
        if parent.pid() != parent_pid {
            return;
        }
        let page_bytes = parent.page_bytes();
        if page_header::read_page_type(page_bytes) != PageType::BeTreeInternal {
            return;
        }
        unsafe {
            child_frame
                .write_ref()
                .set_parent_link_inner(parent_pid, child_slot, false, self.dt_id);
        }
        let hot_swip = child_frame.hot_swip();
        write_child_swip_at(parent.page_bytes_mut(), child_slot, hot_swip);
    }

    fn try_rewrite_leaf_page_local(
        &self,
        page_id: u64,
        message: &BufferedMessage,
        dirty_lsn: Option<Lsn>,
    ) -> Result<bool, CowBeTreeError> {
        if self.try_append_leaf_message(page_id, message, dirty_lsn)? {
            return Ok(true);
        }

        let mut frame = unsafe { self.pool().fix_orphan_frame(page_id) }.exclusive();
        let mut node = decode_page(frame.page_bytes())?;
        let NodePage::Leaf { fence, entries } = &mut node else {
            return Ok(false);
        };
        let pos = lower_bound_entries(entries, &message.key);
        if entries
            .get(pos)
            .is_none_or(|entry| entry.key != message.key)
        {
            return Ok(false);
        }

        apply_message_to_entries(entries, message);
        if self.leaf_should_split(page_id, fence, entries) {
            return Ok(false);
        }

        let mut image = vec![0u8; page_size(page_id)];
        let bytes = encode_leaf_page(&mut image, fence, entries)?;
        let page_image_bytes = image.len();
        frame.page_bytes_mut().copy_from_slice(&image);
        mark_frame_dirty(&frame, dirty_lsn);
        self.stats.inc(CowBeTreeEvent::InPlacePageRewrites);
        self.stats.add_leaf_bytes(bytes);
        self.stats
            .add_leaf_page_image_rewrite_bytes(page_image_bytes);
        Ok(true)
    }

    fn root_pid_from_rewrite(
        &self,
        root_pid: u64,
        rewrite: Rewrite,
        dirty_lsn: Option<Lsn>,
    ) -> Result<u64, CowBeTreeError> {
        match rewrite {
            Rewrite::One { pid } => Ok(pid),
            Rewrite::Split {
                left_pid,
                right_pid,
                separator,
            } => {
                let left_pid = if left_pid == root_pid {
                    let left = self.load_orphan(left_pid)?;
                    self.allocate_node(&left, dirty_lsn)?
                } else {
                    left_pid
                };
                self.write_internal_at(
                    root_pid,
                    &Fence::root(),
                    &[left_pid, right_pid],
                    &[separator],
                    &[],
                    dirty_lsn,
                )?;
                Ok(root_pid)
            }
        }
    }

    fn install_root_rewrite(
        &self,
        root_pid: u64,
        rewrite: Rewrite,
        dirty_lsn: Option<Lsn>,
    ) -> Result<(), CowBeTreeError> {
        let new_root_pid = self.root_pid_from_rewrite(root_pid, rewrite, dirty_lsn)?;
        if new_root_pid == root_pid {
            return Ok(());
        }
        self.install_root_page(new_root_pid)
    }

    fn apply_batch_to_page_with_raw_leaf(
        &self,
        page_id: u64,
        batch: &[BufferedMessage],
        allow_raw_leaf: bool,
        dirty_lsn: Option<Lsn>,
    ) -> Result<Rewrite, CowBeTreeError> {
        let page_id = self.ensure_mutable(page_id, dirty_lsn)?;
        if let [message] = batch
            && ((allow_raw_leaf && self.try_append_leaf_message(page_id, message, dirty_lsn)?)
                || self.try_append_buffer_message(page_id, message, false, dirty_lsn)?)
        {
            return Ok(Rewrite::One { pid: page_id });
        }

        match self.load_orphan(page_id)? {
            NodePage::Leaf { fence, entries } => {
                self.rewrite_leaf_batch(page_id, &fence, entries, batch, dirty_lsn)
            }
            NodePage::Internal {
                fence,
                children,
                separators,
                mut buffer,
            } => {
                buffer.extend_from_slice(batch);
                if self.should_flush_buffer(&buffer)
                    || self.internal_should_split(page_id, &fence, &children, &separators, &buffer)
                {
                    self.flush_internal(page_id, fence, children, separators, buffer, dirty_lsn)
                } else {
                    self.write_internal_at(
                        page_id,
                        &fence,
                        &children,
                        &separators,
                        &buffer,
                        dirty_lsn,
                    )?;
                    let pid = page_id;
                    Ok(Rewrite::One { pid })
                }
            }
        }
    }

    fn rewrite_leaf_batch(
        &self,
        page_id: u64,
        fence: &Fence,
        mut entries: Vec<LeafEntry>,
        batch: &[BufferedMessage],
        dirty_lsn: Option<Lsn>,
    ) -> Result<Rewrite, CowBeTreeError> {
        for message in batch {
            apply_message_to_entries(&mut entries, message);
        }

        let batch_rewrite = batch.len() > 1;
        if !self.leaf_should_split(page_id, fence, &entries) {
            self.write_leaf_at(page_id, fence, &entries, dirty_lsn)?;
            if batch_rewrite {
                self.stats.inc(CowBeTreeEvent::LeafBatchRewrites);
            }
            let pid = page_id;
            return Ok(Rewrite::One { pid });
        }

        let rewrite = self.split_leaf(page_id, fence, entries, dirty_lsn)?;
        if batch_rewrite {
            self.stats.inc(CowBeTreeEvent::LeafBatchRewrites);
        }
        Ok(rewrite)
    }

    fn flush_internal(
        &self,
        page_id: u64,
        fence: Fence,
        mut children: Vec<u64>,
        mut separators: Vec<Vec<u8>>,
        buffer: Vec<BufferedMessage>,
        dirty_lsn: Option<Lsn>,
    ) -> Result<Rewrite, CowBeTreeError> {
        if buffer.is_empty() {
            return self
                .rewrite_internal_empty_buffer(page_id, fence, children, separators, dirty_lsn);
        }

        let (child_idx, batch, remaining) = largest_child_batch(&children, &separators, &buffer)?;

        self.stats.inc(CowBeTreeEvent::BufferFlushes);
        self.stats.add(CowBeTreeEvent::MessagesFlushed, batch.len());
        self.apply_child_flush_batch(&mut children, &mut separators, child_idx, batch, dirty_lsn)?;

        if children.len() > self.config.max_internal_children {
            return self
                .split_internal(page_id, &fence, children, separators, remaining, dirty_lsn);
        }
        if !self.internal_fits_page(&fence, &children, &separators, &remaining) {
            return self.flush_internal(page_id, fence, children, separators, remaining, dirty_lsn);
        }
        self.write_internal_at(
            page_id,
            &fence,
            &children,
            &separators,
            &remaining,
            dirty_lsn,
        )?;
        let pid = page_id;
        Ok(Rewrite::One { pid })
    }

    fn apply_child_flush_batch(
        &self,
        children: &mut Vec<u64>,
        separators: &mut Vec<Vec<u8>>,
        child_idx: usize,
        batch: Vec<BufferedMessage>,
        dirty_lsn: Option<Lsn>,
    ) -> Result<usize, CowBeTreeError> {
        let child_pid = *children
            .get(child_idx)
            .ok_or(CowBeTreeError::CorruptPage("child index out of bounds"))?;
        let child_pid = self.ensure_mutable(child_pid, dirty_lsn)?;
        children[child_idx] = child_pid;
        let appended = self.try_append_leaf_prefix(child_pid, &batch, dirty_lsn)?;
        if appended == batch.len() {
            return Ok(0);
        }

        let remaining_batch = &batch[appended..];
        if remaining_batch.len() > 1
            && let Some(rewrite) =
                self.try_flush_batch_direct_to_leaf(child_pid, remaining_batch, dirty_lsn)?
        {
            return self.install_child_rewrite(children, separators, child_idx, rewrite);
        }

        if let [message] = batch.as_slice() {
            let rewrite = self.apply_batch_to_page_with_raw_leaf(
                child_pid,
                std::slice::from_ref(message),
                false,
                dirty_lsn,
            )?;
            return self.install_child_rewrite(children, separators, child_idx, rewrite);
        }

        let mut inserted_children = 0usize;
        let mut skip_first_raw_leaf = appended < batch.len();
        for message in batch.into_iter().skip(appended) {
            let child_idx = route_child(separators, &message.key);
            let child_pid = *children
                .get(child_idx)
                .ok_or(CowBeTreeError::CorruptPage("child index out of bounds"))?;
            let allow_raw_leaf = !skip_first_raw_leaf;
            skip_first_raw_leaf = false;
            let rewrite = self.apply_batch_to_page_with_raw_leaf(
                child_pid,
                std::slice::from_ref(&message),
                allow_raw_leaf,
                dirty_lsn,
            )?;
            inserted_children +=
                self.install_child_rewrite(children, separators, child_idx, rewrite)?;
        }
        Ok(inserted_children)
    }

    fn try_flush_batch_direct_to_leaf(
        &self,
        page_id: u64,
        batch: &[BufferedMessage],
        dirty_lsn: Option<Lsn>,
    ) -> Result<Option<Rewrite>, CowBeTreeError> {
        let Some(first) = batch.first() else {
            return Ok(Some(Rewrite::One { pid: page_id }));
        };
        let last = batch
            .last()
            .expect("non-empty batch should have a last message");

        match self.load_orphan(page_id)? {
            NodePage::Leaf { fence, entries } => {
                let page_id = self.ensure_mutable(page_id, dirty_lsn)?;
                let rewrite =
                    self.rewrite_leaf_batch(page_id, &fence, entries, batch, dirty_lsn)?;
                self.stats.inc(CowBeTreeEvent::DirectLeafFlushes);
                self.stats
                    .add(CowBeTreeEvent::DirectLeafFlushMessages, batch.len());
                Ok(Some(rewrite))
            }
            NodePage::Internal {
                fence,
                mut children,
                mut separators,
                buffer,
            } => {
                let first_idx = route_child(&separators, &first.key);
                let last_idx = route_child(&separators, &last.key);
                if first_idx != last_idx {
                    return Ok(None);
                }
                if first_idx >= children.len() {
                    return Err(CowBeTreeError::CorruptPage(
                        "direct flush routed outside child array",
                    ));
                }

                let Some(child_rewrite) =
                    self.try_flush_batch_direct_to_leaf(children[first_idx], batch, dirty_lsn)?
                else {
                    return Ok(None);
                };

                let page_id = self.ensure_mutable(page_id, dirty_lsn)?;
                self.install_child_rewrite(
                    &mut children,
                    &mut separators,
                    first_idx,
                    child_rewrite,
                )?;
                if children.len() > self.config.max_internal_children
                    || self.internal_should_split(page_id, &fence, &children, &separators, &buffer)
                {
                    return self
                        .split_internal(page_id, &fence, children, separators, buffer, dirty_lsn)
                        .map(Some);
                }

                self.write_internal_at(
                    page_id,
                    &fence,
                    &children,
                    &separators,
                    &buffer,
                    dirty_lsn,
                )?;
                Ok(Some(Rewrite::One { pid: page_id }))
            }
        }
    }

    fn install_child_rewrite(
        &self,
        children: &mut Vec<u64>,
        separators: &mut Vec<Vec<u8>>,
        child_idx: usize,
        rewrite: Rewrite,
    ) -> Result<usize, CowBeTreeError> {
        match rewrite {
            Rewrite::One { pid } => {
                let child = children
                    .get_mut(child_idx)
                    .ok_or(CowBeTreeError::CorruptPage("child index out of bounds"))?;
                *child = pid;
                Ok(0)
            }
            Rewrite::Split {
                left_pid,
                right_pid,
                separator,
            } => {
                let child = children
                    .get_mut(child_idx)
                    .ok_or(CowBeTreeError::CorruptPage("child index out of bounds"))?;
                *child = left_pid;
                children.insert(child_idx + 1, right_pid);
                separators.insert(child_idx, separator);
                Ok(1)
            }
        }
    }

    fn rewrite_internal_empty_buffer(
        &self,
        page_id: u64,
        fence: Fence,
        children: Vec<u64>,
        separators: Vec<Vec<u8>>,
        dirty_lsn: Option<Lsn>,
    ) -> Result<Rewrite, CowBeTreeError> {
        if self.internal_should_split(page_id, &fence, &children, &separators, &[]) {
            return self.split_internal(
                page_id,
                &fence,
                children,
                separators,
                Vec::new(),
                dirty_lsn,
            );
        }
        self.write_internal_at(page_id, &fence, &children, &separators, &[], dirty_lsn)?;
        let pid = page_id;
        Ok(Rewrite::One { pid })
    }

    fn flush_all_page(&self, page_id: u64) -> Result<Rewrite, CowBeTreeError> {
        let page_id = self.ensure_mutable(page_id, None)?;
        match self.load_orphan(page_id)? {
            NodePage::Leaf { .. } => Ok(Rewrite::One { pid: page_id }),
            NodePage::Internal {
                fence,
                children,
                separators,
                buffer,
            } => {
                let mut rewrite =
                    self.flush_internal(page_id, fence, children, separators, buffer, None)?;
                let mut pid = match rewrite {
                    Rewrite::One { pid } => pid,
                    split @ Rewrite::Split { .. } => return Ok(split),
                };
                let (fence, mut children, mut separators) = loop {
                    let NodePage::Internal {
                        fence,
                        children,
                        separators,
                        buffer,
                    } = self.load_orphan(pid)?
                    else {
                        return Ok(Rewrite::One { pid });
                    };
                    if buffer.is_empty() {
                        break (fence, children, separators);
                    }
                    rewrite =
                        self.flush_internal(pid, fence, children, separators, buffer, None)?;
                    pid = match rewrite {
                        Rewrite::One { pid } => pid,
                        split @ Rewrite::Split { .. } => return Ok(split),
                    };
                };

                let mut idx = 0usize;
                while idx < children.len() {
                    let child_rewrite = self.flush_all_page(children[idx])?;
                    match child_rewrite {
                        Rewrite::One { pid } => children[idx] = pid,
                        Rewrite::Split {
                            left_pid,
                            right_pid,
                            separator,
                        } => {
                            children[idx] = left_pid;
                            children.insert(idx + 1, right_pid);
                            separators.insert(idx, separator);
                            idx += 1;
                        }
                    }
                    idx += 1;
                }
                self.rewrite_internal_empty_buffer(pid, fence, children, separators, None)
            }
        }
    }

    fn flush_all_root(&self) -> Result<(), CowBeTreeError> {
        loop {
            let root_pid = self.ensure_root_mutable(None)?;
            let rewrite = self.flush_all_page(root_pid)?;
            let split = matches!(rewrite, Rewrite::Split { .. });
            self.install_root_rewrite(root_pid, rewrite, None)?;
            if !split {
                return Ok(());
            }
        }
    }

    fn compact_page(&self, page_id: u64) -> Result<Rewrite, CowBeTreeError> {
        let page_id = self.ensure_mutable(page_id, None)?;
        match self.load_orphan(page_id)? {
            NodePage::Leaf { .. } => Ok(Rewrite::One { pid: page_id }),
            NodePage::Internal {
                fence,
                mut children,
                mut separators,
                buffer,
            } => {
                if !buffer.is_empty() {
                    return self.flush_internal(page_id, fence, children, separators, buffer, None);
                }

                let mut idx = 0usize;
                while idx < children.len() {
                    let rewrite = self.compact_page(children[idx])?;
                    match rewrite {
                        Rewrite::One { pid } => children[idx] = pid,
                        Rewrite::Split {
                            left_pid,
                            right_pid,
                            separator,
                        } => {
                            children[idx] = left_pid;
                            children.insert(idx + 1, right_pid);
                            separators.insert(idx, separator);
                            idx += 1;
                        }
                    }
                    idx += 1;
                }

                self.merge_siblings(&mut children, &mut separators)?;
                self.rewrite_internal_empty_buffer(page_id, fence, children, separators, None)
            }
        }
    }

    fn prune_versions_page(
        &self,
        page_id: u64,
        watermark: Timestamp,
        result: &mut CowBeTreeGcResult,
    ) -> Result<(Rewrite, bool), CowBeTreeError> {
        match self.load_orphan(page_id)? {
            NodePage::Leaf { fence, mut entries } => {
                let mut page_result = CowBeTreeGcResult {
                    leaf_pages_visited: 1,
                    ..CowBeTreeGcResult::default()
                };
                for entry in &mut entries {
                    page_result.add_assign(prune_version_records(&mut entry.versions, watermark));
                }
                if page_result.versions_pruned == 0 {
                    result.add_assign(page_result);
                    return Ok((Rewrite::One { pid: page_id }, false));
                }

                let page_id = self.ensure_mutable(page_id, None)?;
                self.write_leaf_at(page_id, &fence, &entries, None)?;
                page_result.leaf_pages_rewritten = 1;
                result.add_assign(page_result);
                Ok((Rewrite::One { pid: page_id }, true))
            }
            NodePage::Internal {
                fence,
                mut children,
                mut separators,
                buffer,
            } => {
                if !buffer.is_empty() {
                    let page_id = self.ensure_mutable(page_id, None)?;
                    let rewrite =
                        self.flush_internal(page_id, fence, children, separators, buffer, None)?;
                    return Ok((rewrite, true));
                }

                let mut changed = false;
                let mut idx = 0usize;
                while idx < children.len() {
                    let old_pid = children[idx];
                    let (rewrite, child_changed) =
                        self.prune_versions_page(old_pid, watermark, result)?;
                    if child_changed {
                        changed = true;
                    }
                    match rewrite {
                        Rewrite::One { pid } => {
                            if pid != old_pid {
                                changed = true;
                            }
                            children[idx] = pid;
                        }
                        Rewrite::Split {
                            left_pid,
                            right_pid,
                            separator,
                        } => {
                            children[idx] = left_pid;
                            children.insert(idx + 1, right_pid);
                            separators.insert(idx, separator);
                            idx += 1;
                            changed = true;
                        }
                    }
                    idx += 1;
                }

                if !changed {
                    return Ok((Rewrite::One { pid: page_id }, false));
                }

                self.merge_siblings(&mut children, &mut separators)?;
                let page_id = self.ensure_mutable(page_id, None)?;
                let rewrite =
                    self.rewrite_internal_empty_buffer(page_id, fence, children, separators, None)?;
                Ok((rewrite, true))
            }
        }
    }

    fn prune_versions_incremental_page(
        &self,
        page_id: u64,
        state: &mut IncrementalGcState<'_>,
    ) -> Result<(Rewrite, bool), CowBeTreeError> {
        if state.remaining_leaf_pages == 0 || state.reached_end {
            return Ok((Rewrite::One { pid: page_id }, false));
        }

        match self.load_orphan(page_id)? {
            NodePage::Leaf { fence, mut entries } => {
                if !leaf_may_contain_gc_cursor_or_later(&fence, &entries, state.lower) {
                    if fence.upper.is_none() {
                        state.reached_end = true;
                    }
                    return Ok((Rewrite::One { pid: page_id }, false));
                }

                state.remaining_leaf_pages -= 1;
                state.result.leaf_pages_visited += 1;

                let mut page_result = CowBeTreeGcResult::default();
                for entry in &mut entries {
                    page_result
                        .add_assign(prune_version_records(&mut entry.versions, state.watermark));
                }

                state.next_key = fence.upper.clone();
                if state.next_key.is_none() {
                    state.reached_end = true;
                }

                if page_result.versions_pruned == 0 {
                    return Ok((Rewrite::One { pid: page_id }, false));
                }

                let page_id = self.ensure_mutable(page_id, None)?;
                self.write_leaf_at(page_id, &fence, &entries, None)?;
                page_result.leaf_pages_rewritten = 1;
                state.result.add_assign(page_result);
                Ok((Rewrite::One { pid: page_id }, true))
            }
            NodePage::Internal {
                fence,
                mut children,
                mut separators,
                buffer,
            } => {
                if !buffer.is_empty() {
                    let page_id = self.ensure_mutable(page_id, None)?;
                    let rewrite =
                        self.flush_internal(page_id, fence, children, separators, buffer, None)?;
                    return Ok((rewrite, true));
                }

                let start_idx = state
                    .lower
                    .map(|lower| route_child(&separators, lower))
                    .unwrap_or(0);
                if start_idx >= children.len() {
                    state.reached_end = true;
                    return Ok((Rewrite::One { pid: page_id }, false));
                }

                let mut changed = false;
                let mut idx = start_idx;
                while idx < children.len() && state.remaining_leaf_pages > 0 && !state.reached_end {
                    let old_pid = children[idx];
                    let previous_lower = state.lower;
                    if idx != start_idx {
                        state.lower = None;
                    }
                    let (rewrite, child_changed) =
                        self.prune_versions_incremental_page(old_pid, state)?;
                    state.lower = previous_lower;

                    if child_changed {
                        changed = true;
                    }
                    match rewrite {
                        Rewrite::One { pid } => {
                            if pid != old_pid {
                                changed = true;
                            }
                            children[idx] = pid;
                        }
                        Rewrite::Split {
                            left_pid,
                            right_pid,
                            separator,
                        } => {
                            children[idx] = left_pid;
                            children.insert(idx + 1, right_pid);
                            separators.insert(idx, separator);
                            idx += 1;
                            changed = true;
                        }
                    }
                    idx += 1;
                }

                if !changed {
                    return Ok((Rewrite::One { pid: page_id }, false));
                }

                let page_id = self.ensure_mutable(page_id, None)?;
                let rewrite = self.rewrite_internal_preserving_buffer(
                    page_id, fence, children, separators, buffer, None,
                )?;
                Ok((rewrite, true))
            }
        }
    }

    fn record_gc_result(&self, result: CowBeTreeGcResult) {
        self.stats
            .add(CowBeTreeEvent::GcVersionsPruned, result.versions_pruned);
        self.stats.add(
            CowBeTreeEvent::GcLeafPagesVisited,
            result.leaf_pages_visited,
        );
        self.stats.add(
            CowBeTreeEvent::GcLeafPagesRewritten,
            result.leaf_pages_rewritten,
        );
        self.stats.add(
            CowBeTreeEvent::GcVersionBytesPruned,
            result.version_bytes_pruned,
        );
    }

    fn rewrite_internal_preserving_buffer(
        &self,
        page_id: u64,
        fence: Fence,
        children: Vec<u64>,
        separators: Vec<Vec<u8>>,
        buffer: Vec<BufferedMessage>,
        dirty_lsn: Option<Lsn>,
    ) -> Result<Rewrite, CowBeTreeError> {
        if self.internal_should_split(page_id, &fence, &children, &separators, &buffer) {
            return self.split_internal(page_id, &fence, children, separators, buffer, dirty_lsn);
        }
        self.write_internal_at(page_id, &fence, &children, &separators, &buffer, dirty_lsn)?;
        Ok(Rewrite::One { pid: page_id })
    }

    fn merge_siblings(
        &self,
        children: &mut Vec<u64>,
        separators: &mut Vec<Vec<u8>>,
    ) -> Result<(), CowBeTreeError> {
        let mut idx = 0usize;
        while idx + 1 < children.len() {
            let left = self.load_orphan(children[idx])?;
            let right = self.load_orphan(children[idx + 1])?;
            let Some(merged_pid) = self.try_merge_pair(&left, &right, &separators[idx])? else {
                idx += 1;
                continue;
            };

            children[idx] = merged_pid;
            children.remove(idx + 1);
            separators.remove(idx);
        }
        Ok(())
    }

    fn try_merge_pair(
        &self,
        left: &NodePage,
        right: &NodePage,
        separator: &[u8],
    ) -> Result<Option<u64>, CowBeTreeError> {
        match (left, right) {
            (
                NodePage::Leaf {
                    fence: left_fence,
                    entries: left_entries,
                },
                NodePage::Leaf {
                    fence: right_fence,
                    entries: right_entries,
                },
            ) => {
                let total_entries = left_entries.len() + right_entries.len();
                if total_entries > self.config.merge_leaf_entries {
                    return Ok(None);
                }
                let fence = Fence::span(left_fence, right_fence);
                let mut entries = left_entries.clone();
                entries.extend_from_slice(right_entries);
                if !self.leaf_fits_page(&fence, &entries) {
                    return Ok(None);
                }
                let pid = self.allocate_leaf(&fence, &entries, None)?;
                self.stats.inc(CowBeTreeEvent::LeafMerges);
                Ok(Some(pid))
            }
            (
                NodePage::Internal {
                    fence: left_fence,
                    children: left_children,
                    separators: left_separators,
                    buffer: left_buffer,
                },
                NodePage::Internal {
                    fence: right_fence,
                    children: right_children,
                    separators: right_separators,
                    buffer: right_buffer,
                },
            ) if left_buffer.is_empty() && right_buffer.is_empty() => {
                let total_children = left_children.len() + right_children.len();
                if total_children > self.config.merge_internal_children {
                    return Ok(None);
                }
                let fence = Fence::span(left_fence, right_fence);
                let mut children = left_children.clone();
                children.extend_from_slice(right_children);
                let mut separators = left_separators.clone();
                separators.push(separator.to_vec());
                separators.extend_from_slice(right_separators);
                if !self.internal_fits_page(&fence, &children, &separators, &[]) {
                    return Ok(None);
                }
                let pid = self.allocate_internal(&fence, &children, &separators, &[], None)?;
                self.stats.inc(CowBeTreeEvent::InternalMerges);
                Ok(Some(pid))
            }
            _ => Ok(None),
        }
    }

    fn split_leaf(
        &self,
        page_id: u64,
        fence: &Fence,
        entries: Vec<LeafEntry>,
        dirty_lsn: Option<Lsn>,
    ) -> Result<Rewrite, CowBeTreeError> {
        if entries.len() < 2 {
            let pid = self.allocate_leaf(fence, &entries, dirty_lsn)?;
            return Ok(Rewrite::One { pid });
        }

        let mid = entries.len() / 2;
        let separator = entries[mid].key.clone();
        let left_fence = fence.left_of(separator.clone());
        let right_fence = fence.right_of(separator.clone());
        let left_fits_existing_page =
            self.leaf_fits_capacity(&left_fence, &entries[..mid], page_size(page_id));
        let right_pid = self.allocate_leaf(&right_fence, &entries[mid..], dirty_lsn)?;
        let left_pid = if left_fits_existing_page {
            self.write_leaf_at_with_right(
                page_id,
                &left_fence,
                &entries[..mid],
                right_pid,
                dirty_lsn,
            )?;
            page_id
        } else {
            self.allocate_leaf_with_right(&left_fence, &entries[..mid], right_pid, dirty_lsn)?
        };
        self.append_hint.store(right_pid, Ordering::Release);
        self.stats.inc(CowBeTreeEvent::LeafSplits);
        Ok(Rewrite::Split {
            left_pid,
            right_pid,
            separator,
        })
    }

    fn split_internal(
        &self,
        page_id: u64,
        fence: &Fence,
        children: Vec<u64>,
        separators: Vec<Vec<u8>>,
        buffer: Vec<BufferedMessage>,
        dirty_lsn: Option<Lsn>,
    ) -> Result<Rewrite, CowBeTreeError> {
        if children.len() < 2 {
            return Err(CowBeTreeError::EmptyInternalPage);
        }

        let mid = children.len() / 2;
        if mid == 0 || mid >= children.len() {
            return Err(CowBeTreeError::EmptyInternalPage);
        }

        let separator = separators[mid - 1].clone();
        let left_fence = fence.left_of(separator.clone());
        let right_fence = fence.right_of(separator.clone());
        let mut left_buffer = Vec::new();
        let mut right_buffer = Vec::new();
        for message in buffer {
            if message.key.as_slice() < separator.as_slice() {
                left_buffer.push(message);
                continue;
            }
            right_buffer.push(message);
        }
        self.write_internal_at(
            page_id,
            &left_fence,
            &children[..mid],
            &separators[..mid - 1],
            &left_buffer,
            dirty_lsn,
        )?;
        let left_pid = page_id;
        let right_pid = self.allocate_internal(
            &right_fence,
            &children[mid..],
            &separators[mid..],
            &right_buffer,
            dirty_lsn,
        )?;
        self.stats.inc(CowBeTreeEvent::InternalSplits);
        Ok(Rewrite::Split {
            left_pid,
            right_pid,
            separator,
        })
    }

    fn leaf_should_split(&self, page_id: u64, fence: &Fence, entries: &[LeafEntry]) -> bool {
        if entries.len() > self.config.max_leaf_entries {
            return true;
        }
        encoded_page_len(
            &NodePage::Leaf {
                fence: fence.clone(),
                entries: entries.to_vec(),
            },
            page_size(page_id),
        )
        .is_err()
    }

    fn leaf_fits_page(&self, fence: &Fence, entries: &[LeafEntry]) -> bool {
        self.leaf_fits_capacity(fence, entries, PAGE_SIZE)
    }

    fn leaf_fits_capacity(&self, fence: &Fence, entries: &[LeafEntry], capacity: usize) -> bool {
        encoded_page_len(
            &NodePage::Leaf {
                fence: fence.clone(),
                entries: entries.to_vec(),
            },
            capacity,
        )
        .is_ok()
    }

    fn internal_should_split(
        &self,
        page_id: u64,
        fence: &Fence,
        children: &[u64],
        separators: &[Vec<u8>],
        buffer: &[BufferedMessage],
    ) -> bool {
        if children.len() > self.config.max_internal_children {
            return true;
        }
        encoded_page_len(
            &NodePage::Internal {
                fence: fence.clone(),
                children: children.to_vec(),
                separators: separators.to_vec(),
                buffer: buffer.to_vec(),
            },
            page_size(page_id),
        )
        .is_err()
    }

    fn internal_fits_page(
        &self,
        fence: &Fence,
        children: &[u64],
        separators: &[Vec<u8>],
        buffer: &[BufferedMessage],
    ) -> bool {
        encoded_page_len(
            &NodePage::Internal {
                fence: fence.clone(),
                children: children.to_vec(),
                separators: separators.to_vec(),
                buffer: buffer.to_vec(),
            },
            PAGE_SIZE,
        )
        .is_ok()
    }

    fn should_flush_buffer(&self, buffer: &[BufferedMessage]) -> bool {
        buffer.len() >= self.config.flush_threshold_messages
            || buffer_encoded_len(buffer) >= self.config.flush_threshold_bytes
    }

    fn allocate_leaf(
        &self,
        fence: &Fence,
        entries: &[LeafEntry],
        dirty_lsn: Option<Lsn>,
    ) -> Result<u64, CowBeTreeError> {
        self.allocate_leaf_with_right(fence, entries, 0, dirty_lsn)
    }

    fn allocate_leaf_with_right(
        &self,
        fence: &Fence,
        entries: &[LeafEntry],
        right_sibling: u64,
        dirty_lsn: Option<Lsn>,
    ) -> Result<u64, CowBeTreeError> {
        let mut image = vec![0u8; PAGE_SIZE];
        let bytes = encode_leaf_page(&mut image, fence, entries)?;
        if right_sibling != 0 {
            write_right_sibling(&mut image, right_sibling);
        }
        let (pid, frame) = self.allocate_frame();
        let mut frame = frame.exclusive();
        frame.page_bytes_mut().copy_from_slice(&image);
        frame.set_parent_link_none();
        mark_frame_dirty(&frame, dirty_lsn);
        self.stats.inc(CowBeTreeEvent::CowPagesAllocated);
        self.stats.add_leaf_bytes(bytes);
        Ok(pid)
    }

    fn allocate_internal(
        &self,
        fence: &Fence,
        children: &[u64],
        separators: &[Vec<u8>],
        buffer: &[BufferedMessage],
        dirty_lsn: Option<Lsn>,
    ) -> Result<u64, CowBeTreeError> {
        let mut image = vec![0u8; PAGE_SIZE];
        let bytes = encode_internal_page(&mut image, fence, children, separators, buffer)?;
        let (pid, frame) = self.allocate_frame();
        let mut frame = frame.exclusive();
        frame.page_bytes_mut().copy_from_slice(&image);
        frame.set_parent_link_none();
        mark_frame_dirty(&frame, dirty_lsn);
        self.stats.inc(CowBeTreeEvent::CowPagesAllocated);
        self.stats.add_internal_bytes(bytes);
        Ok(pid)
    }

    fn allocate_node(
        &self,
        node: &NodePage,
        dirty_lsn: Option<Lsn>,
    ) -> Result<u64, CowBeTreeError> {
        match node {
            NodePage::Leaf { fence, entries } => self.allocate_leaf(fence, entries, dirty_lsn),
            NodePage::Internal {
                fence,
                children,
                separators,
                buffer,
            } => self.allocate_internal(fence, children, separators, buffer, dirty_lsn),
        }
    }

    fn write_node_at(
        &self,
        page_id: u64,
        node: &NodePage,
        dirty_lsn: Option<Lsn>,
    ) -> Result<(), CowBeTreeError> {
        match node {
            NodePage::Leaf { fence, entries } => {
                self.write_leaf_at(page_id, fence, entries, dirty_lsn)
            }
            NodePage::Internal {
                fence,
                children,
                separators,
                buffer,
            } => self.write_internal_at(page_id, fence, children, separators, buffer, dirty_lsn),
        }
    }

    fn write_leaf_at(
        &self,
        page_id: u64,
        fence: &Fence,
        entries: &[LeafEntry],
        dirty_lsn: Option<Lsn>,
    ) -> Result<(), CowBeTreeError> {
        self.write_leaf_at_with_right(page_id, fence, entries, 0, dirty_lsn)
    }

    fn write_leaf_at_with_right(
        &self,
        page_id: u64,
        fence: &Fence,
        entries: &[LeafEntry],
        right_sibling: u64,
        dirty_lsn: Option<Lsn>,
    ) -> Result<(), CowBeTreeError> {
        let mut image = vec![0u8; page_size(page_id)];
        let bytes = encode_leaf_page(&mut image, fence, entries)?;
        if right_sibling != 0 {
            write_right_sibling(&mut image, right_sibling);
        }
        let page_image_bytes = image.len();
        self.write_page_image(page_id, image, dirty_lsn);
        self.stats.inc(CowBeTreeEvent::InPlacePageRewrites);
        self.stats.add_leaf_bytes(bytes);
        self.stats
            .add_leaf_page_image_rewrite_bytes(page_image_bytes);
        Ok(())
    }

    fn write_internal_at(
        &self,
        page_id: u64,
        fence: &Fence,
        children: &[u64],
        separators: &[Vec<u8>],
        buffer: &[BufferedMessage],
        dirty_lsn: Option<Lsn>,
    ) -> Result<(), CowBeTreeError> {
        let mut image = vec![0u8; page_size(page_id)];
        let bytes = encode_internal_page(&mut image, fence, children, separators, buffer)?;
        let page_image_bytes = image.len();
        self.write_page_image(page_id, image, dirty_lsn);
        self.stats.inc(CowBeTreeEvent::InPlacePageRewrites);
        self.stats.add_internal_bytes(bytes);
        self.stats
            .add_internal_page_image_rewrite_bytes(page_image_bytes);
        Ok(())
    }

    fn write_page_image(&self, page_id: u64, image: Vec<u8>, dirty_lsn: Option<Lsn>) {
        let mut frame = unsafe { self.pool().fix_orphan_frame(page_id) }.exclusive();
        let page = frame.page_bytes_mut();
        debug_assert_eq!(page.len(), image.len());
        page.copy_from_slice(&image);
        mark_frame_dirty(&frame, dirty_lsn);
    }

    fn allocate_frame(&self) -> (u64, PinnedFrame<'_>) {
        self.pool().allocate_and_fix()
    }

    fn try_append_buffer_message(
        &self,
        page_id: u64,
        message: &BufferedMessage,
        root_append: bool,
        dirty_lsn: Option<Lsn>,
    ) -> Result<bool, CowBeTreeError> {
        let mut frame = unsafe { self.pool().fix_orphan_frame(page_id) }.exclusive();
        let appended = append_internal_buffer_message(
            frame.page_bytes_mut(),
            message,
            self.config.flush_threshold_messages,
            self.config.flush_threshold_bytes,
            self.config.max_internal_children,
        )?;
        let Some(_appended) = appended else {
            return Ok(false);
        };

        mark_frame_dirty(&frame, dirty_lsn);
        if root_append {
            self.stats.inc(CowBeTreeEvent::RootBufferAppends);
        }
        self.stats.inc(CowBeTreeEvent::RawBufferAppends);
        Ok(true)
    }

    fn try_append_buffer_kv(
        &self,
        page_id: u64,
        key: &[u8],
        value: &[u8],
        commit_ts: Timestamp,
    ) -> Result<bool, CowBeTreeError> {
        let mut frame = unsafe { self.pool().fix_orphan_frame(page_id) }.exclusive();
        let appended = append_internal_buffer_kv(
            frame.page_bytes_mut(),
            key,
            value,
            commit_ts,
            self.config.flush_threshold_messages,
            self.config.flush_threshold_bytes,
            self.config.max_internal_children,
        )?;
        let Some(_appended) = appended else {
            return Ok(false);
        };

        mark_frame_dirty(&frame, None);
        self.stats.inc(CowBeTreeEvent::RawBufferAppends);
        Ok(true)
    }

    fn try_append_leaf_message(
        &self,
        page_id: u64,
        message: &BufferedMessage,
        dirty_lsn: Option<Lsn>,
    ) -> Result<bool, CowBeTreeError> {
        let mut frame = unsafe { self.pool().fix_orphan_frame(page_id) }.exclusive();
        let appended = append_leaf_entry_message(
            frame.page_bytes_mut(),
            message,
            self.config.max_leaf_entries,
        )?;
        let Some(appended) = appended else {
            return Ok(false);
        };

        mark_frame_dirty(&frame, dirty_lsn);
        self.append_hint.store(page_id, Ordering::Release);
        self.stats
            .add(CowBeTreeEvent::RawLeafAppends, appended.message_count);
        self.stats.inc(CowBeTreeEvent::RawLeafAppendBatches);
        Ok(true)
    }

    fn try_append_leaf_prefix(
        &self,
        page_id: u64,
        batch: &[BufferedMessage],
        dirty_lsn: Option<Lsn>,
    ) -> Result<usize, CowBeTreeError> {
        if batch.is_empty() {
            return Ok(0);
        }

        let mut frame = unsafe { self.pool().fix_orphan_frame(page_id) }.exclusive();
        let appended =
            append_leaf_entry_prefix(frame.page_bytes_mut(), batch, self.config.max_leaf_entries)?;
        let Some(appended) = appended else {
            return Ok(0);
        };

        mark_frame_dirty(&frame, dirty_lsn);
        self.append_hint.store(page_id, Ordering::Release);
        self.stats
            .add(CowBeTreeEvent::RawLeafAppends, appended.message_count);
        self.stats.inc(CowBeTreeEvent::RawLeafAppendBatches);
        Ok(appended.message_count)
    }

    fn ensure_root_mutable(&self, dirty_lsn: Option<Lsn>) -> Result<u64, CowBeTreeError> {
        let old_root_pid = self.root_page_id();
        let new_root_pid = self.ensure_mutable(old_root_pid, dirty_lsn)?;
        if old_root_pid == new_root_pid {
            return Ok(old_root_pid);
        }
        self.install_root_page(new_root_pid)?;
        Ok(new_root_pid)
    }

    fn ensure_mutable(&self, page_id: u64, dirty_lsn: Option<Lsn>) -> Result<u64, CowBeTreeError> {
        if self.forks.active_roots.load(Ordering::Acquire) <= 1 {
            return Ok(page_id);
        }

        let mut shared_pages = self
            .forks
            .shared_pages
            .write()
            .expect("CoW B-epsilon fork registry lock poisoned");
        let Some(ref_count) = shared_pages.get_mut(&page_id) else {
            return Ok(page_id);
        };
        if *ref_count <= 1 {
            shared_pages.remove(&page_id);
            return Ok(page_id);
        }

        let node = self.load_orphan(page_id)?;
        let new_page_id = self.allocate_node(&node, dirty_lsn)?;
        *ref_count -= 1;
        if *ref_count == 1 {
            shared_pages.remove(&page_id);
        }
        self.stats.inc(CowBeTreeEvent::ForkPageCopies);
        Ok(new_page_id)
    }

    fn install_root_page(&self, new_root_pid: u64) -> Result<(), CowBeTreeError> {
        let old_root_pid = self.root_page_id();
        if old_root_pid == new_root_pid {
            return Ok(());
        }

        let mut old_root = unsafe { self.pool().fix_frame(&self.root) }.exclusive();
        let mut new_root = unsafe { self.pool().fix_orphan_frame(new_root_pid) }.exclusive();
        old_root.set_parent_link_none();
        new_root.set_parent_link_stable(unsafe { StableSwipRef::from_ref(self.root.as_ref()) });
        self.root.store(new_root.hot_swip(), Ordering::Release);
        self.append_hint.store(new_root_pid, Ordering::Release);
        self.stats.inc(CowBeTreeEvent::RootReplacements);
        Ok(())
    }

    fn mark_reachable_pages_shared(&self, root_page_id: u64) -> Result<(), CowBeTreeError> {
        let mut pages = Vec::new();
        let mut visited = HashSet::new();
        self.collect_reachable_pages(root_page_id, &mut visited, &mut pages)?;

        let mut shared_pages = self
            .forks
            .shared_pages
            .write()
            .expect("CoW B-epsilon fork registry lock poisoned");
        for page_id in pages {
            *shared_pages.entry(page_id).or_insert(1) += 1;
        }
        Ok(())
    }

    fn collect_reachable_pages(
        &self,
        page_id: u64,
        visited: &mut HashSet<u64>,
        pages: &mut Vec<u64>,
    ) -> Result<(), CowBeTreeError> {
        if !visited.insert(page_id) {
            return Ok(());
        }
        pages.push(page_id);
        let node = self.load_orphan(page_id)?;
        if let NodePage::Internal { children, .. } = node {
            for child in children {
                self.collect_reachable_pages(child, visited, pages)?;
            }
        }
        Ok(())
    }

    fn load_root(&self) -> Result<NodePage, CowBeTreeError> {
        let swip = self.root.load(Ordering::Acquire);
        if swip.is_evicted() {
            self.stats.inc(CowBeTreeEvent::ColdResolves);
            self.stats.inc(CowBeTreeEvent::PageFaults);
        }
        let frame = unsafe { self.pool().fix_frame(&self.root) }.shared();
        decode_page(frame.page_bytes())
    }

    fn lookup_root_step(
        &self,
        key: &[u8],
        read_ts: Timestamp,
    ) -> Result<VisibleLookupStep, CowBeTreeError> {
        let swip = self.root.load(Ordering::Acquire);
        if swip.is_evicted() {
            self.stats.inc(CowBeTreeEvent::ColdResolves);
            self.stats.inc(CowBeTreeEvent::PageFaults);
        }
        let frame = unsafe { self.pool().fix_frame(&self.root) }.shared();
        let root_pid = frame.pid();
        let step = lookup_step(frame.page_bytes(), key, read_ts)?;
        let mut step = own_lookup_step(step);
        if let VisibleLookupStep::Internal { parent_pid, .. } = &mut step {
            *parent_pid = root_pid;
        }
        Ok(step)
    }

    fn lookup_orphan_step(
        &self,
        parent_pid: u64,
        child_swip: Swip,
        key: &[u8],
        read_ts: Timestamp,
    ) -> Result<VisibleLookupStep, CowBeTreeError> {
        let frame = self.fix_child(child_swip).shared();
        let step = lookup_step(frame.page_bytes(), key, read_ts)?;
        let mut step = own_lookup_step(step);
        if let VisibleLookupStep::Internal {
            parent_pid: step_parent_pid,
            ..
        } = &mut step
        {
            *step_parent_pid = frame.pid();
        }
        let _ = parent_pid;
        Ok(step)
    }

    fn load_orphan(&self, page_id: u64) -> Result<NodePage, CowBeTreeError> {
        self.stats.inc(CowBeTreeEvent::ColdResolves);
        self.stats.inc(CowBeTreeEvent::PageFaults);
        let frame = unsafe { self.pool().fix_orphan_frame(page_id) }.shared();
        decode_page(frame.page_bytes())
    }

    /// Resolve a child SWIP read from a parent internal page to a pinned frame.
    ///
    /// Hot/Cool swizzled pointers bypass the page table entirely (direct frame
    /// pin); Evicted pointers fall back to [`BufferPool::fix_orphan_frame`],
    /// which faults the page in via the page table.
    ///
    /// The betree currently stores children as Evicted page IDs, so this always
    /// takes the fault path. The Hot/Cool branch is the forward-compatible fast
    /// path for a future swizzle-in step: [`BufferPool::pin_child`] returns
    /// `None` for a stale pointer (frame reused), so a concurrent eviction that
    /// unswizzles between reading the SWIP and pinning degrades to a panic
    /// rather than undefined behaviour. A correct swizzle-in implementation
    /// would hold the parent's latch until the child is pinned (B-link style),
    /// which is follow-up work tracked alongside optimistic reads.
    fn fix_child(&self, swip: Swip) -> PinnedFrame<'_> {
        if swip.is_hot() || swip.is_cool() {
            unsafe { self.pool().pin_child(swip) }
                .expect("hot/cool child swip must pin while parent edge is stable")
        } else {
            unsafe { self.pool().fix_orphan_frame(swip.as_page_id()) }
        }
    }

    fn materialize_node(
        &self,
        node: NodePage,
        path_buffer: &[BufferedMessage],
        rows: &mut BTreeMap<Vec<u8>, Vec<VersionRecord>>,
    ) {
        match node {
            NodePage::Leaf { entries, .. } => {
                for entry in entries {
                    rows.entry(entry.key).or_default().extend(entry.versions);
                }
                for message in path_buffer {
                    insert_version(
                        rows.entry(message.key.clone()).or_default(),
                        message.version.clone(),
                    );
                }
                if !path_buffer.is_empty() {
                    self.stats.inc(CowBeTreeEvent::PathBufferMerges);
                }
            }
            NodePage::Internal {
                children, buffer, ..
            } => {
                let mut next_buffer = path_buffer.to_vec();
                next_buffer.extend(buffer);
                for child in children {
                    if let Ok(child_node) = self.load_orphan(child) {
                        self.materialize_node(child_node, &next_buffer, rows);
                    }
                }
            }
        }
    }

    fn debug_walk(&self, node: NodePage, depth: u32, walk: &mut DebugWalk) {
        walk.snapshot.height = walk.snapshot.height.max(depth);
        match node {
            NodePage::Leaf { entries, .. } => {
                walk.snapshot.leaf_pages += 1;
                walk.snapshot.leaf_entries += entries.len();
            }
            NodePage::Internal {
                children, buffer, ..
            } => {
                walk.snapshot.internal_pages += 1;
                walk.snapshot.buffered_messages += buffer.len();
                walk.snapshot.max_buffered_messages_on_page = walk
                    .snapshot
                    .max_buffered_messages_on_page
                    .max(buffer.len());
                for child in children {
                    if let Ok(child_node) = self.load_orphan(child) {
                        self.debug_walk(child_node, depth + 1, walk);
                    }
                }
            }
        }
    }

    fn pool(&self) -> &BufferPool {
        self.pool.as_pool()
    }
}

/// Writeback preparer for `BeTreeInternal` pages.
///
/// The betree internal-node body stores child references as 8-byte SWIP words.
/// Resident pages may carry swizzled (Hot/Cool) pointers once the traversal
/// path swizzles them in; the on-disk format must contain Evicted page IDs.
/// This preparer rewrites every Hot/Cool child slot to `Evicted(pid)` in the
/// writeback copy, leaving the live frame's swizzled pointers intact.
///
/// Leaf pages carry no child pointers and short-circuit. The preparer is
/// registered for [`PageType::BeTreeInternal`] in [`CowBeTree::new`],
/// [`CowBeTree::open_with_config`], and [`CowBeTree::fork`].
struct BeTreeInternalWritebackPreparer;

impl PageWritebackPreparer for BeTreeInternalWritebackPreparer {
    fn prepare_page_copy_for_writeback(&self, page: &mut [u8], pool: &BufferPool) {
        if page_header::read_page_type(page) != PageType::BeTreeInternal {
            return;
        }
        let Some((child_count, child_offset)) = internal_child_array_range(page) else {
            return;
        };
        for i in 0..child_count {
            let off = match child_offset.checked_add(i * 8) {
                Some(off) => off,
                None => return,
            };
            let Some(slot) = page.get_mut(off..off + 8) else {
                return;
            };
            let raw = u64::from_le_bytes(slot.try_into().unwrap());
            let swip = Swip::from_raw(raw);
            if let Some(frame) = BufferFrameRef::from_hot_swip(swip)
                && pool.contains_frame(frame)
            {
                let child_pid = frame.pid();
                slot.copy_from_slice(&Swip::evicted(child_pid).raw().to_le_bytes());
            }
        }
    }
}

fn register_internal_writeback_preparer(pool: &BufferPool) {
    pool.register_page_writeback_preparer(
        PageType::BeTreeInternal,
        Arc::new(BeTreeInternalWritebackPreparer),
    );
}

fn mark_frame_dirty(frame: &ExclusiveFrame<'_>, dirty_lsn: Option<Lsn>) {
    match dirty_lsn {
        Some(lsn) => frame.mark_dirty_with_lsn(lsn),
        None => frame.mark_dirty(),
    }
}

fn insert_version(versions: &mut Vec<VersionRecord>, version: VersionRecord) {
    versions.retain(|existing| existing.commit_ts != version.commit_ts);
    let pos = versions.partition_point(|existing| existing.commit_ts > version.commit_ts);
    versions.insert(pos, version);
}

fn leaf_may_contain_gc_cursor_or_later(
    fence: &Fence,
    entries: &[LeafEntry],
    lower: Option<&[u8]>,
) -> bool {
    let Some(lower) = lower else {
        return true;
    };
    if fence
        .upper
        .as_ref()
        .is_some_and(|upper| upper.as_slice() <= lower)
    {
        return false;
    }
    if fence.upper.is_none()
        && entries
            .last()
            .is_none_or(|entry| entry.key.as_slice() < lower)
    {
        return false;
    }
    true
}

fn prune_version_records(
    versions: &mut Vec<VersionRecord>,
    watermark: Timestamp,
) -> CowBeTreeGcResult {
    let mut kept_watermark_floor = false;
    let mut result = CowBeTreeGcResult::default();
    versions.retain(|version| {
        if version.commit_ts > watermark {
            return true;
        }
        if !kept_watermark_floor {
            kept_watermark_floor = true;
            return true;
        }

        result.versions_pruned += 1;
        result.version_bytes_pruned += version_record_encoded_len(version);
        false
    });
    result
}

fn version_record_encoded_len(version: &VersionRecord) -> usize {
    8 + 1 + 4 + version.value.len()
}

fn own_lookup_step(step: LookupStep<'_>) -> VisibleLookupStep {
    match step {
        LookupStep::Leaf { visible } => VisibleLookupStep::Leaf {
            visible: visible.map(own_visible_candidate),
        },
        LookupStep::Internal {
            child_swip,
            child_slot,
            visible_buffer,
            buffer_count,
        } => VisibleLookupStep::Internal {
            parent_pid: 0,
            child_swip,
            child_slot,
            visible_buffer: visible_buffer.map(own_visible_candidate),
            buffer_count,
        },
    }
}

fn own_visible_candidate(version: RawVisibleVersion<'_>) -> VisibleCandidate {
    VisibleCandidate {
        commit_ts: version.commit_ts,
        deleted: version.deleted,
        value: version.value.to_vec(),
    }
}

fn merge_owned_visible_candidate(
    visible: &mut Option<VisibleCandidate>,
    candidate: VisibleCandidate,
) {
    if visible
        .as_ref()
        .is_none_or(|existing| candidate.commit_ts > existing.commit_ts)
    {
        *visible = Some(candidate);
    }
}

fn visible_from_versions(versions: &[VersionRecord], read_ts: Timestamp) -> Option<&[u8]> {
    versions
        .iter()
        .find(|version| version.commit_ts <= read_ts)
        .and_then(|version| (!version.deleted).then_some(version.value.as_slice()))
}

fn visible_candidate_from_versions(
    versions: &[VersionRecord],
    read_ts: Timestamp,
) -> Option<VisibleCandidate> {
    versions
        .iter()
        .find(|version| version.commit_ts <= read_ts)
        .map(|version| VisibleCandidate {
            commit_ts: version.commit_ts,
            deleted: version.deleted,
            value: version.value.clone(),
        })
}

fn range_contains(key: &[u8], lower: Bound<&[u8]>, upper: Bound<&[u8]>) -> bool {
    let above_lower = match lower {
        Bound::Unbounded => true,
        Bound::Included(bound) => key >= bound,
        Bound::Excluded(bound) => key > bound,
    };
    if !above_lower {
        return false;
    }

    match upper {
        Bound::Unbounded => true,
        Bound::Included(bound) => key <= bound,
        Bound::Excluded(bound) => key < bound,
    }
}

fn swip_page_id(swip: Swip) -> u64 {
    if swip.is_hot() || swip.is_cool() {
        unsafe { (*swip.as_ptr::<BufferFrame>()).header.core.pid }
    } else {
        swip.as_page_id()
    }
}
fn largest_child_batch(
    children: &[u64],
    separators: &[Vec<u8>],
    buffer: &[BufferedMessage],
) -> Result<(usize, Vec<BufferedMessage>, Vec<BufferedMessage>), CowBeTreeError> {
    if buffer.is_empty() {
        return Err(CowBeTreeError::CorruptPage("empty child flush batch"));
    }

    let mut child_counts = vec![(0usize, 0usize); children.len()];
    for message in buffer {
        let child_idx = route_child(separators, &message.key);
        if child_idx >= children.len() {
            return Err(CowBeTreeError::CorruptPage(
                "buffer message routed outside child array",
            ));
        }
        child_counts[child_idx].0 += 1;
        child_counts[child_idx].1 += message.encoded_len();
    }

    let mut best_child = None;
    for (child_idx, &(count, bytes)) in child_counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let replace_best = best_child
            .as_ref()
            .is_none_or(|&(_, best_count, best_bytes)| {
                count > best_count || (count == best_count && bytes > best_bytes)
            });
        if replace_best {
            best_child = Some((child_idx, count, bytes));
        }
    }

    let (best_child, batch_len, _) =
        best_child.ok_or(CowBeTreeError::CorruptPage("empty child flush batch"))?;
    let mut batch = Vec::with_capacity(batch_len);
    let mut remaining = Vec::with_capacity(buffer.len() - batch_len);
    for message in buffer {
        let child_idx = route_child(separators, &message.key);
        if child_idx == best_child {
            batch.push(message.clone());
            continue;
        }
        remaining.push(message.clone());
    }
    sort_buffer_messages(&mut batch);

    Ok((best_child, batch, remaining))
}

fn counter_shards() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1)
        .max(1)
}

fn default_internal_children(page_size: usize) -> usize {
    let pivot_budget = integer_sqrt(page_size).max(128);
    (pivot_budget / 32).clamp(4, 64)
}

fn default_flush_threshold_bytes(page_size: usize) -> usize {
    (page_size / 4).clamp(1024, 256 * 1024)
}

fn default_flush_threshold_messages(flush_threshold_bytes: usize) -> usize {
    (flush_threshold_bytes / 64).clamp(8, 4096)
}

fn integer_sqrt(value: usize) -> usize {
    if value <= 1 {
        return value;
    }

    let mut low = 1usize;
    let mut high = value;
    while low + 1 < high {
        let mid = low + (high - low) / 2;
        if mid <= value / mid {
            low = mid;
        } else {
            high = mid;
        }
    }
    low
}

struct DtProxy {
    pool: BufferPoolHandle,
}

impl ParentFinder for DtProxy {
    fn find_and_unswizzle_with_hint(
        &self,
        child: BufferFrameRef,
        child_pid: u64,
        parent_pid: u64,
        slot_index: u16,
        _is_upper: bool,
    ) -> Option<bool> {
        let pool = self.pool.as_pool();
        let parent_pin = unsafe { pool.try_fix_resident_page_frame(parent_pid) }?;
        let mut parent = parent_pin.try_exclusive().ok()?;
        if parent.pid() != parent_pid {
            return None;
        }
        let page_bytes = parent.page_bytes();
        if page_header::read_page_type(page_bytes) != PageType::BeTreeInternal {
            return None;
        }
        let current_swip = read_child_swip_at(page_bytes, slot_index);
        let child_hot_swip = child.hot_swip();
        let expected_evicted = Swip::evicted(child_pid);
        match current_swip {
            Some(swip)
                if swip.raw() == child_hot_swip.raw() || swip.raw() == expected_evicted.raw() =>
            {
                let page_mut = parent.page_bytes_mut();
                write_child_swip_at(page_mut, slot_index, expected_evicted);
                parent.mark_dirty();
                Some(true)
            }
            _ => None,
        }
    }

    fn find_and_unswizzle(&self, _child: BufferFrameRef, _child_pid: u64) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pagebox_storage::buffer_frame::PAGE_SIZE;
    use pagebox_storage::page_header;

    /// Build a `BeTreeInternal` page image whose child slots carry real
    /// resident frame SWIPs, run the writeback preparer, and assert every
    /// slot is rewritten to `Evicted(pid)`.
    ///
    /// This is the regression guard for the swizzled-pointer writeback path:
    /// if the preparer stopped iterating slots, mis-parsed the child-array
    /// offset, or wrote the wrong endianness, the Hot child would persist into
    /// the on-disk image and the next reopen would read a garbage page id.
    #[test]
    fn writeback_preparer_converts_hot_child_swips_to_evicted_page_ids() {
        let pool = BufferPool::new(8);
        register_internal_writeback_preparer(&pool);

        // Allocate two resident child frames and keep them pinned so their
        // Hot SWIPs stay valid pointers into the pool's frame array.
        let (left_pid, left_pinned) = pool.allocate_and_fix();
        let (right_pid, right_pinned) = pool.allocate_and_fix();
        let left_swip = left_pinned.hot_swip();
        let _right_swip = right_pinned.hot_swip();
        assert!(left_swip.is_hot(), "freshly allocated frame must be Hot");

        // Encode an internal page with both children as Evicted page IDs.
        let children = vec![left_pid, right_pid];
        let separators = vec![b"mid".to_vec()];
        let mut page = vec![0u8; PAGE_SIZE];
        encode_internal_page(&mut page, &Fence::root(), &children, &separators, &[]).unwrap();
        assert_eq!(
            page_header::read_page_type(&page),
            PageType::BeTreeInternal,
            "encoded page must carry the BeTreeInternal marker",
        );

        let (child_count, child_offset) =
            internal_child_array_range(&page).expect("internal page has a child array");
        assert_eq!(child_count, 2, "child count mismatch");

        // Swizzle the first child slot to Hot; leave the second as Evicted.
        page[child_offset..child_offset + 8].copy_from_slice(&left_swip.raw().to_le_bytes());

        // Run the preparer on the writeback copy.
        BeTreeInternalWritebackPreparer.prepare_page_copy_for_writeback(&mut page, &pool);

        // The Hot slot must be rewritten to Evicted(left_pid).
        let got_left = u64::from_le_bytes(page[child_offset..child_offset + 8].try_into().unwrap());
        assert_eq!(
            Swip::from_raw(got_left),
            Swip::evicted(left_pid),
            "preparer must convert the swizzled child back to an evicted page id",
        );
        assert!(
            !Swip::from_raw(got_left).is_hot(),
            "converted slot must not remain a swizzled pointer",
        );

        // The Evicted slot must be untouched.
        let right_off = child_offset + 8;
        let got_right = u64::from_le_bytes(page[right_off..right_off + 8].try_into().unwrap());
        assert_eq!(
            Swip::from_raw(got_right),
            Swip::evicted(right_pid),
            "preparer must leave an already-evicted child slot unchanged",
        );

        drop(left_pinned);
        drop(right_pinned);
    }

    /// A stray 8-byte word that happens to look like an aligned pointer but
    /// does not resolve to a pool-owned frame must be left alone, not
    /// rewritten to a garbage page id. This mirrors the `contains_frame` guard
    /// in `BufferFrame::convert_swips_in_buf`.
    #[test]
    fn writeback_preparer_leaves_non_pool_swips_untouched() {
        let pool = BufferPool::new(4);
        register_internal_writeback_preparer(&pool);

        let (child_pid, _pinned) = pool.allocate_and_fix();
        let children = vec![child_pid];
        let mut page = vec![0u8; PAGE_SIZE];
        encode_internal_page(&mut page, &Fence::root(), &children, &[], &[]).unwrap();

        let (_, child_offset) = internal_child_array_range(&page).unwrap();
        // A Hot-tagged word pointing at a plausible-but-unowned address: a
        // small aligned value well outside the pool's arena.
        let bogus = Swip::hot_ptr(0x1000_0000);
        assert!(bogus.is_hot(), "fixture must be a Hot swip");
        page[child_offset..child_offset + 8].copy_from_slice(&bogus.raw().to_le_bytes());

        let before: [u8; 8] = page[child_offset..child_offset + 8].try_into().unwrap();
        BeTreeInternalWritebackPreparer.prepare_page_copy_for_writeback(&mut page, &pool);
        let after: [u8; 8] = page[child_offset..child_offset + 8].try_into().unwrap();
        assert_eq!(
            before, after,
            "non-pool Hot swip must be left untouched (no garbage page id rewrite)",
        );
    }
}
