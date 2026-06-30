use std::fmt;

use pagebox_storage::buffer_frame::{BufferFrame, BufferFrameRef};
use pagebox_storage::page_header::{self, PageType};
use pagebox_swip_kernel::SwipWord as Swip;

use crate::message::{BufferedMessage, Timestamp, VersionRecord};

const PAGE_MAGIC: u32 = 0x4254_4552;
const PAGE_VERSION: u8 = 4;
const HEADER_SIZE: usize = 32;
const LEAF_DIR_ENTRY_SIZE: usize = 4;
const COUNT_OFF: usize = 8;
const BODY_LEN_OFF: usize = 10;
const MAGIC_OFF: usize = 14;
const VERSION_OFF: usize = 20;
const KIND_OFF: usize = 21;
const NONE_FENCE_LEN: u16 = u16::MAX;

#[derive(Debug)]
pub enum CowBeTreeError {
    CorruptPage(&'static str),
    EmptyInternalPage,
    PageOverflow {
        kind: &'static str,
        needed: usize,
        capacity: usize,
    },
}

impl fmt::Display for CowBeTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CorruptPage(msg) => write!(f, "corrupt CoW B-epsilon page: {msg}"),
            Self::EmptyInternalPage => write!(f, "internal CoW B-epsilon page has no children"),
            Self::PageOverflow {
                kind,
                needed,
                capacity,
            } => {
                write!(
                    f,
                    "{kind} page needs {needed} bytes, capacity is {capacity}"
                )
            }
        }
    }
}

impl std::error::Error for CowBeTreeError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageKindDebug {
    Leaf,
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PageKind {
    Leaf = 1,
    Internal = 2,
}

impl PageKind {
    fn from_byte(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Leaf),
            2 => Some(Self::Internal),
            _ => None,
        }
    }

    fn page_type(self) -> PageType {
        match self {
            Self::Leaf => PageType::BeTreeLeaf,
            Self::Internal => PageType::BeTreeInternal,
        }
    }

    pub(crate) fn debug(self) -> PageKindDebug {
        match self {
            Self::Leaf => PageKindDebug::Leaf,
            Self::Internal => PageKindDebug::Internal,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Fence {
    pub(crate) lower: Vec<u8>,
    pub(crate) upper: Option<Vec<u8>>,
}

impl Fence {
    pub(crate) fn root() -> Self {
        Self {
            lower: Vec::new(),
            upper: None,
        }
    }

    pub(crate) fn left_of(&self, upper: Vec<u8>) -> Self {
        Self {
            lower: self.lower.clone(),
            upper: Some(upper),
        }
    }

    pub(crate) fn right_of(&self, lower: Vec<u8>) -> Self {
        Self {
            lower,
            upper: self.upper.clone(),
        }
    }

    pub(crate) fn span(left: &Fence, right: &Fence) -> Self {
        Self {
            lower: left.lower.clone(),
            upper: right.upper.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LeafEntry {
    pub(crate) key: Vec<u8>,
    pub(crate) versions: Vec<VersionRecord>,
}

impl LeafEntry {
    pub(crate) fn insert_version(&mut self, version: VersionRecord) {
        self.versions
            .retain(|existing| existing.commit_ts != version.commit_ts);
        let pos = self
            .versions
            .partition_point(|existing| existing.commit_ts > version.commit_ts);
        self.versions.insert(pos, version);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum NodePage {
    Leaf {
        fence: Fence,
        entries: Vec<LeafEntry>,
    },
    Internal {
        fence: Fence,
        children: Vec<u64>,
        separators: Vec<Vec<u8>>,
        buffer: Vec<BufferedMessage>,
    },
}

impl NodePage {
    pub(crate) fn kind(&self) -> PageKind {
        match self {
            Self::Leaf { .. } => PageKind::Leaf,
            Self::Internal { .. } => PageKind::Internal,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawVisibleVersion<'a> {
    pub(crate) commit_ts: Timestamp,
    pub(crate) deleted: bool,
    pub(crate) value: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LookupStep<'a> {
    Leaf {
        visible: Option<RawVisibleVersion<'a>>,
    },
    Internal {
        child_swip: Swip,
        child_slot: u16,
        visible_buffer: Option<RawVisibleVersion<'a>>,
        buffer_count: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InternalBufferAppend {
    pub(crate) buffer_count: usize,
    pub(crate) body_len: usize,
    pub(crate) message_len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LeafEntryAppend {
    pub(crate) entry_count: usize,
    pub(crate) body_len: usize,
    pub(crate) entry_bytes: usize,
    pub(crate) message_count: usize,
}

pub(crate) fn lower_bound_entries(entries: &[LeafEntry], key: &[u8]) -> usize {
    entries
        .binary_search_by(|entry| entry.key.as_slice().cmp(key))
        .unwrap_or_else(|pos| pos)
}

pub(crate) fn apply_message_to_entries(entries: &mut Vec<LeafEntry>, message: &BufferedMessage) {
    let pos = lower_bound_entries(entries, &message.key);
    if entries
        .get(pos)
        .is_some_and(|entry| entry.key == message.key)
    {
        entries[pos].insert_version(message.version.clone());
        return;
    }

    let mut entry = LeafEntry {
        key: message.key.clone(),
        versions: Vec::new(),
    };
    entry.insert_version(message.version.clone());
    entries.insert(pos, entry);
}

pub(crate) fn route_child(separators: &[Vec<u8>], key: &[u8]) -> usize {
    separators.partition_point(|separator| separator.as_slice() <= key)
}

pub(crate) fn buffer_encoded_len(buffer: &[BufferedMessage]) -> usize {
    buffer.iter().map(BufferedMessage::encoded_len).sum()
}

pub(crate) fn encoded_page_len(node: &NodePage, capacity: usize) -> Result<usize, CowBeTreeError> {
    match node {
        NodePage::Leaf { fence, entries } => {
            encoded_body_len(PageKind::Leaf, encode_leaf_body(fence, entries)?, capacity)
        }
        NodePage::Internal {
            fence,
            children,
            separators,
            buffer,
        } => encoded_body_len(
            PageKind::Internal,
            encode_internal_body(fence, children, separators, buffer)?,
            capacity,
        ),
    }
}

pub(crate) fn encode_leaf_page(
    page: &mut [u8],
    fence: &Fence,
    entries: &[LeafEntry],
) -> Result<usize, CowBeTreeError> {
    finish_page(
        page,
        PageKind::Leaf,
        entries.len(),
        encode_leaf_body(fence, entries)?,
    )
}

#[derive(Debug)]
pub(crate) struct SplitLeafResult {
    pub(crate) separator: Vec<u8>,
    pub(crate) left_count: usize,
    pub(crate) right_count: usize,
}

/// Split a leaf page in-place by copying entries directly from `src` into two
/// destination pages.  Entries `[0, mid)` go to `dst_left`, entries
/// `[mid, entry_count)` go to `dst_right`.  The separator is the key of the
/// entry at index `mid`.
///
/// This is the zero-allocation split primitive: no `NodePage`, no
/// `Vec<LeafEntry>`, and no `BodyWriter` allocations.  The only allocation is
/// the separator key `Vec<u8>` returned in `SplitLeafResult`.
pub(crate) fn split_leaf_into_pages(
    src: &[u8],
    dst_left: &mut [u8],
    dst_right: &mut [u8],
    fence_left: &Fence,
    fence_right: &Fence,
    mid: usize,
) -> Result<SplitLeafResult, CowBeTreeError> {
    let reader = LeafPageReader::new(src)?;
    let entry_count = reader.entry_count();
    if mid == 0 || mid >= entry_count {
        return Err(CowBeTreeError::CorruptPage("invalid leaf split point"));
    }

    let separator = reader.entry_key(mid)?.to_vec();
    build_leaf_page_from_src(dst_left, fence_left, &reader, 0, mid)?;
    build_leaf_page_from_src(dst_right, fence_right, &reader, mid, entry_count)?;

    Ok(SplitLeafResult {
        separator,
        left_count: mid,
        right_count: entry_count - mid,
    })
}

fn build_leaf_page_from_src(
    dst: &mut [u8],
    fence: &Fence,
    src: &LeafPageReader<'_>,
    start: usize,
    end: usize,
) -> Result<usize, CowBeTreeError> {
    let count = end
        .checked_sub(start)
        .ok_or(CowBeTreeError::CorruptPage("entry range underflow"))?;
    if count == 0 {
        return Err(CowBeTreeError::CorruptPage(
            "cannot build leaf page from empty entry range",
        ));
    }

    let fence_len = fence_encoded_len(fence);
    let count_len = 2;

    let mut entry_bytes_total = 0usize;
    for idx in start..end {
        let entry_bytes = src.entry_raw(idx)?;
        entry_bytes_total = entry_bytes_total
            .checked_add(entry_bytes.len())
            .ok_or(CowBeTreeError::CorruptPage("entry bytes overflow"))?;
    }

    let directory_len =
        count
            .checked_mul(LEAF_DIR_ENTRY_SIZE)
            .ok_or(CowBeTreeError::CorruptPage(
                "leaf directory length overflow",
            ))?;
    let body_len = fence_len
        .checked_add(count_len)
        .and_then(|l| l.checked_add(entry_bytes_total))
        .and_then(|l| l.checked_add(directory_len))
        .ok_or(CowBeTreeError::CorruptPage("leaf body length overflow"))?;
    let needed = HEADER_SIZE
        .checked_add(body_len)
        .ok_or(CowBeTreeError::CorruptPage("page length overflow"))?;
    if needed > dst.len() {
        return Err(page_overflow("leaf", needed, dst.len()));
    }

    dst.fill(0);

    write_u16(dst, COUNT_OFF, count, "page item count too large")?;
    write_u32(dst, BODY_LEN_OFF, body_len, "page body too large")?;
    dst[MAGIC_OFF..MAGIC_OFF + 4].copy_from_slice(&PAGE_MAGIC.to_le_bytes());
    dst[VERSION_OFF] = PAGE_VERSION;
    dst[KIND_OFF] = PageKind::Leaf as u8;
    page_header::write_page_type(dst, PageType::BeTreeLeaf);

    let body_start = HEADER_SIZE;
    let mut pos = body_start;

    pos = write_fence_at_page(dst, pos, fence)?;

    let count_u16 =
        u16::try_from(count).map_err(|_| CowBeTreeError::CorruptPage("too many leaf entries"))?;
    dst[pos..pos + 2].copy_from_slice(&count_u16.to_le_bytes());
    pos += 2;

    let dir_start_in_body = (pos - body_start) + entry_bytes_total;
    let dir_start_in_page = body_start + dir_start_in_body;

    for idx in start..end {
        let entry_bytes = src.entry_raw(idx)?;
        let entry_offset_in_body = pos - body_start;
        let dir_idx = idx - start;
        let dir_pos = dir_start_in_page + dir_idx * LEAF_DIR_ENTRY_SIZE;
        let offset_u32 = u32::try_from(entry_offset_in_body)
            .map_err(|_| CowBeTreeError::CorruptPage("leaf entry offset too large"))?;
        dst[dir_pos..dir_pos + LEAF_DIR_ENTRY_SIZE].copy_from_slice(&offset_u32.to_le_bytes());
        dst[pos..pos + entry_bytes.len()].copy_from_slice(entry_bytes);
        pos += entry_bytes.len();
    }

    Ok(needed)
}

fn fence_encoded_len(fence: &Fence) -> usize {
    let lower_len = 2 + fence.lower.len();
    let upper_len = match &fence.upper {
        Some(upper) => 2 + upper.len(),
        None => 2,
    };
    lower_len + upper_len
}

fn write_fence_at_page(
    page: &mut [u8],
    mut pos: usize,
    fence: &Fence,
) -> Result<usize, CowBeTreeError> {
    let lower_len = u16::try_from(fence.lower.len())
        .map_err(|_| CowBeTreeError::CorruptPage("lower fence too large"))?;
    page[pos..pos + 2].copy_from_slice(&lower_len.to_le_bytes());
    pos += 2;
    page[pos..pos + fence.lower.len()].copy_from_slice(&fence.lower);
    pos += fence.lower.len();

    match &fence.upper {
        Some(upper) => {
            let upper_len = u16::try_from(upper.len())
                .map_err(|_| CowBeTreeError::CorruptPage("upper fence too large"))?;
            page[pos..pos + 2].copy_from_slice(&upper_len.to_le_bytes());
            pos += 2;
            page[pos..pos + upper.len()].copy_from_slice(upper);
            pos += upper.len();
        }
        None => {
            page[pos..pos + 2].copy_from_slice(&NONE_FENCE_LEN.to_le_bytes());
            pos += 2;
        }
    }
    Ok(pos)
}

fn encode_leaf_body(fence: &Fence, entries: &[LeafEntry]) -> Result<Vec<u8>, CowBeTreeError> {
    let mut body = BodyWriter::new();
    encode_fence(&mut body, fence)?;
    body.push_u16(entries.len(), "too many leaf entries")?;
    let mut entry_offsets = Vec::with_capacity(entries.len());
    for entry in entries {
        entry_offsets.push(body.len_u32("leaf entry offset too large")?);
        body.push_bytes_with_u16_len(&entry.key, "leaf key too large")?;
        body.push_u16(entry.versions.len(), "too many retained versions")?;
        for version in &entry.versions {
            encode_version(&mut body, version)?;
        }
    }
    for offset in entry_offsets {
        body.push_u32_raw(offset);
    }
    Ok(body.into_inner())
}

pub(crate) fn encode_internal_page(
    page: &mut [u8],
    fence: &Fence,
    children: &[u64],
    separators: &[Vec<u8>],
    buffer: &[BufferedMessage],
) -> Result<usize, CowBeTreeError> {
    finish_page(
        page,
        PageKind::Internal,
        children.len(),
        encode_internal_body(fence, children, separators, buffer)?,
    )
}

fn encode_internal_body(
    fence: &Fence,
    children: &[u64],
    separators: &[Vec<u8>],
    buffer: &[BufferedMessage],
) -> Result<Vec<u8>, CowBeTreeError> {
    if children.is_empty() {
        return Err(CowBeTreeError::EmptyInternalPage);
    }
    if separators.len() + 1 != children.len() {
        return Err(CowBeTreeError::CorruptPage(
            "internal separator count does not match child count",
        ));
    }

    let mut body = BodyWriter::new();
    encode_fence(&mut body, fence)?;
    body.push_u16(children.len(), "too many internal children")?;
    body.push_u16(separators.len(), "too many internal separators")?;
    body.push_u16(buffer.len(), "too many buffered messages")?;
    for child in children {
        body.push_u64(Swip::evicted(*child).raw());
    }
    for separator in separators {
        body.push_bytes_with_u16_len(separator, "internal separator too large")?;
    }
    for message in buffer {
        encode_message(&mut body, message)?;
    }
    Ok(body.into_inner())
}

pub(crate) fn decode_page(page: &[u8]) -> Result<NodePage, CowBeTreeError> {
    let (kind, mut reader) = page_body_reader(page)?;
    let fence = decode_fence(&mut reader)?;
    match kind {
        PageKind::Leaf => decode_leaf(reader, fence),
        PageKind::Internal => decode_internal(reader, fence),
    }
}

pub(crate) fn lookup_step<'a>(
    page: &'a [u8],
    key: &[u8],
    read_ts: Timestamp,
) -> Result<LookupStep<'a>, CowBeTreeError> {
    let (kind, mut reader) = page_body_reader(page)?;
    skip_fence(&mut reader)?;
    match kind {
        PageKind::Leaf => lookup_leaf(reader, key, read_ts),
        PageKind::Internal => lookup_internal(reader, key, read_ts),
    }
}

/// Route `key` to its child SWIP without scanning the internal buffer.
///
/// Returns `Ok(None)` for leaf pages, `Ok(Some(swip))` for the routed child on
/// internal pages. This is the write-path routing primitive: it avoids the
/// visible-buffer scan that [`lookup_step`] performs, since the write path
/// only needs the child edge, not the buffered-message visibility.
#[allow(dead_code)]
pub(crate) fn lookup_child_swip(page: &[u8], key: &[u8]) -> Result<Option<Swip>, CowBeTreeError> {
    Ok(lookup_child_slot(page, key)?.map(|(swip, _)| swip))
}

pub(crate) fn lookup_child_slot(
    page: &[u8],
    key: &[u8],
) -> Result<Option<(Swip, u16)>, CowBeTreeError> {
    let (kind, mut reader) = page_body_reader(page)?;
    skip_fence(&mut reader)?;
    if kind != PageKind::Internal {
        return Ok(None);
    }

    let child_count = reader.read_u16()? as usize;
    let separator_count = reader.read_u16()? as usize;
    let _buffer_count = reader.read_u16()?;
    if child_count == 0 {
        return Err(CowBeTreeError::EmptyInternalPage);
    }
    if separator_count + 1 != child_count {
        return Err(CowBeTreeError::CorruptPage(
            "internal separator count does not match child count",
        ));
    }

    let child_start = reader.pos;
    let child_bytes = child_count
        .checked_mul(8)
        .ok_or(CowBeTreeError::CorruptPage("child array length overflow"))?;
    reader.read_exact(child_bytes)?;

    let mut child_idx = 0usize;
    for _ in 0..separator_count {
        let separator = reader.read_bytes_u16_len()?;
        if separator <= key {
            child_idx += 1;
        }
    }

    let child_off = child_start
        .checked_add(child_idx * 8)
        .ok_or(CowBeTreeError::CorruptPage("child offset overflow"))?;
    let raw_child = reader
        .data
        .get(child_off..child_off + 8)
        .ok_or(CowBeTreeError::CorruptPage("child read out of bounds"))?;
    let child_swip = Swip::from_raw(u64::from_le_bytes(raw_child.try_into().unwrap()));
    Ok(Some((child_swip, child_idx as u16)))
}

/// Locate the contiguous child SWIP array in an internal page.
///
/// Returns `(child_count, child_array_page_offset)` where the offset is
/// relative to the page start. Returns `None` for non-internal pages or
/// malformed headers. Used by the writeback preparer to iterate child slots.
pub(crate) fn internal_child_array_range(page: &[u8]) -> Option<(usize, usize)> {
    let (kind, mut reader) = page_body_reader(page).ok()?;
    if kind != PageKind::Internal {
        return None;
    }
    skip_fence(&mut reader).ok()?;
    let child_count = reader.read_u16().ok()? as usize;
    let _separator_count = reader.read_u16().ok()?;
    let _buffer_count = reader.read_u16().ok()?;
    let child_array_offset = HEADER_SIZE
        .checked_add(reader.pos)
        .filter(|_| child_count > 0)?;
    Some((child_count, child_array_offset))
}

pub(crate) fn read_child_swip_at(page: &[u8], slot: u16) -> Option<Swip> {
    let (count, offset) = internal_child_array_range(page)?;
    if slot as usize >= count {
        return None;
    }
    let pos = offset + (slot as usize) * 8;
    let raw = u64::from_le_bytes(page.get(pos..pos + 8)?.try_into().ok()?);
    Some(Swip::from_raw(raw))
}

pub(crate) fn write_child_swip_at(page: &mut [u8], slot: u16, swip: Swip) -> Option<()> {
    let (count, offset) = internal_child_array_range(page)?;
    if slot as usize >= count {
        return None;
    }
    let pos = offset + (slot as usize) * 8;
    page.get_mut(pos..pos + 8)?
        .copy_from_slice(&swip.raw().to_le_bytes());
    Some(())
}

#[allow(dead_code)]
pub(crate) fn find_child_slot_by_frame(
    page: &[u8],
    child_bf: *const u8,
    child_pid: u64,
) -> Option<u16> {
    let (count, offset) = internal_child_array_range(page)?;
    let expected_hot = Swip::hot(child_bf as *mut BufferFrame).raw();
    let expected_evicted = Swip::evicted(child_pid).raw();
    for slot in 0..count {
        let pos = offset + slot * 8;
        let raw = u64::from_le_bytes(page.get(pos..pos + 8)?.try_into().ok()?);
        if raw == expected_hot || raw == expected_evicted {
            return Some(slot as u16);
        }
    }
    None
}

pub(crate) fn append_internal_buffer_message(
    page: &mut [u8],
    message: &BufferedMessage,
    flush_threshold_messages: usize,
    flush_threshold_bytes: usize,
    max_internal_children: usize,
) -> Result<Option<InternalBufferAppend>, CowBeTreeError> {
    let Some(layout) = internal_buffer_layout(page)? else {
        return Ok(None);
    };
    if layout.child_count > max_internal_children {
        return Ok(None);
    }

    let message_len = message.encoded_len();
    let new_buffer_count =
        layout
            .buffer_count
            .checked_add(1)
            .ok_or(CowBeTreeError::CorruptPage(
                "internal buffer count overflow",
            ))?;
    let new_buffer_len =
        layout
            .buffer_len
            .checked_add(message_len)
            .ok_or(CowBeTreeError::CorruptPage(
                "internal buffer byte length overflow",
            ))?;
    if new_buffer_count >= flush_threshold_messages || new_buffer_len >= flush_threshold_bytes {
        return Ok(None);
    }

    let new_body_len = layout
        .body_len
        .checked_add(message_len)
        .ok_or(CowBeTreeError::CorruptPage("internal body length overflow"))?;
    if HEADER_SIZE + new_body_len > page.len() {
        return Ok(None);
    }

    let message_offset = HEADER_SIZE + layout.body_len;
    encode_message_at(
        page.get_mut(message_offset..message_offset + message_len)
            .ok_or(CowBeTreeError::CorruptPage(
                "internal append message offset out of bounds",
            ))?,
        message,
    )?;
    write_u32(
        page,
        BODY_LEN_OFF,
        new_body_len,
        "internal page body too large",
    )?;
    write_u16(
        page,
        HEADER_SIZE + layout.buffer_count_offset,
        new_buffer_count,
        "too many buffered messages",
    )?;

    Ok(Some(InternalBufferAppend {
        buffer_count: new_buffer_count,
        body_len: new_body_len,
        message_len,
    }))
}

pub(crate) fn append_internal_buffer_kv(
    page: &mut [u8],
    key: &[u8],
    value: &[u8],
    commit_ts: Timestamp,
    flush_threshold_messages: usize,
    flush_threshold_bytes: usize,
    max_internal_children: usize,
) -> Result<Option<InternalBufferAppend>, CowBeTreeError> {
    let Some(layout) = internal_buffer_layout(page)? else {
        return Ok(None);
    };
    if layout.child_count > max_internal_children {
        return Ok(None);
    }

    let key_len = u16::try_from(key.len())
        .map_err(|_| CowBeTreeError::CorruptPage("message key too large"))?;
    let value_len = u32::try_from(value.len())
        .map_err(|_| CowBeTreeError::CorruptPage("version value too large"))?;
    let message_len = usize::from(key_len) + value_len as usize + 15;

    let new_buffer_count =
        layout
            .buffer_count
            .checked_add(1)
            .ok_or(CowBeTreeError::CorruptPage(
                "internal buffer count overflow",
            ))?;
    let new_buffer_len =
        layout
            .buffer_len
            .checked_add(message_len)
            .ok_or(CowBeTreeError::CorruptPage(
                "internal buffer byte length overflow",
            ))?;
    if new_buffer_count >= flush_threshold_messages || new_buffer_len >= flush_threshold_bytes {
        return Ok(None);
    }

    let new_body_len = layout
        .body_len
        .checked_add(message_len)
        .ok_or(CowBeTreeError::CorruptPage("internal body length overflow"))?;
    if HEADER_SIZE + new_body_len > page.len() {
        return Ok(None);
    }

    let message_offset = HEADER_SIZE + layout.body_len;
    encode_message_kv_at(
        page.get_mut(message_offset..message_offset + message_len)
            .ok_or(CowBeTreeError::CorruptPage(
                "internal append message offset out of bounds",
            ))?,
        key,
        value,
        commit_ts,
    )?;
    write_u32(
        page,
        BODY_LEN_OFF,
        new_body_len,
        "internal page body too large",
    )?;
    write_u16(
        page,
        HEADER_SIZE + layout.buffer_count_offset,
        new_buffer_count,
        "too many buffered messages",
    )?;

    Ok(Some(InternalBufferAppend {
        buffer_count: new_buffer_count,
        body_len: new_body_len,
        message_len,
    }))
}

pub(crate) fn append_leaf_entry_message(
    page: &mut [u8],
    message: &BufferedMessage,
    max_leaf_entries: usize,
) -> Result<Option<LeafEntryAppend>, CowBeTreeError> {
    append_leaf_entry_batch(page, std::slice::from_ref(message), max_leaf_entries)
}

pub(crate) fn append_leaf_kv(
    page: &mut [u8],
    key: &[u8],
    value: &[u8],
    commit_ts: Timestamp,
    max_leaf_entries: usize,
) -> Result<Option<LeafEntryAppend>, CowBeTreeError> {
    let Some(layout) = leaf_append_layout(page, key)? else {
        return Ok(None);
    };

    let key_len =
        u16::try_from(key.len()).map_err(|_| CowBeTreeError::CorruptPage("leaf key too large"))?;
    let value_len = u32::try_from(value.len())
        .map_err(|_| CowBeTreeError::CorruptPage("version value too large"))?;
    let entry_bytes = usize::from(key_len) + value_len as usize + 17;

    let new_entry_count = layout
        .entry_count
        .checked_add(1)
        .ok_or(CowBeTreeError::CorruptPage("leaf entry count overflow"))?;
    if new_entry_count > max_leaf_entries {
        return Ok(None);
    }

    let directory_bytes =
        new_entry_count
            .checked_mul(LEAF_DIR_ENTRY_SIZE)
            .ok_or(CowBeTreeError::CorruptPage(
                "leaf directory length overflow",
            ))?;
    let new_body_len = layout
        .directory_start
        .checked_add(entry_bytes)
        .ok_or(CowBeTreeError::CorruptPage("leaf body length overflow"))?;
    let new_body_len = new_body_len
        .checked_add(directory_bytes)
        .ok_or(CowBeTreeError::CorruptPage("leaf body length overflow"))?;
    if HEADER_SIZE + new_body_len > page.len() {
        return Ok(None);
    }

    let mut entry_offset = HEADER_SIZE + layout.directory_start;
    let mut entry_offsets = layout.entry_offsets.clone();
    entry_offsets.push(
        u32::try_from(entry_offset - HEADER_SIZE)
            .map_err(|_| CowBeTreeError::CorruptPage("leaf entry offset too large"))?,
    );
    encode_leaf_kv_at(
        page.get_mut(entry_offset..entry_offset + entry_bytes)
            .ok_or(CowBeTreeError::CorruptPage(
                "leaf append entry offset out of bounds",
            ))?,
        key,
        value,
        commit_ts,
    )?;
    entry_offset += entry_bytes;
    write_leaf_directory(page, entry_offset, &entry_offsets)?;
    write_u32(page, BODY_LEN_OFF, new_body_len, "leaf page body too large")?;
    write_u16(page, COUNT_OFF, new_entry_count, "too many leaf entries")?;
    write_u16(
        page,
        HEADER_SIZE + layout.entry_count_offset,
        new_entry_count,
        "too many leaf entries",
    )?;

    Ok(Some(LeafEntryAppend {
        entry_count: new_entry_count,
        body_len: new_body_len,
        entry_bytes,
        message_count: 1,
    }))
}

pub(crate) fn append_leaf_entry_batch(
    page: &mut [u8],
    messages: &[BufferedMessage],
    max_leaf_entries: usize,
) -> Result<Option<LeafEntryAppend>, CowBeTreeError> {
    let Some(first) = messages.first() else {
        return Ok(None);
    };
    let Some(layout) = leaf_append_layout(page, &first.key)? else {
        return Ok(None);
    };

    if let Some(last) = messages.last()
        && !std::ptr::eq(first, last)
        && leaf_append_layout(page, &last.key)?.is_none()
    {
        return Ok(None);
    }

    let mut previous_key = None;
    let mut entry_bytes = 0usize;
    for message in messages {
        if let Some(previous_key) = previous_key
            && previous_key >= message.key.as_slice()
        {
            return Ok(None);
        }
        previous_key = Some(message.key.as_slice());
        entry_bytes = entry_bytes
            .checked_add(leaf_entry_message_encoded_len(message)?)
            .ok_or(CowBeTreeError::CorruptPage("leaf append length overflow"))?;
    }

    let new_entry_count = layout
        .entry_count
        .checked_add(messages.len())
        .ok_or(CowBeTreeError::CorruptPage("leaf entry count overflow"))?;
    if new_entry_count > max_leaf_entries {
        return Ok(None);
    }

    let directory_bytes =
        new_entry_count
            .checked_mul(LEAF_DIR_ENTRY_SIZE)
            .ok_or(CowBeTreeError::CorruptPage(
                "leaf directory length overflow",
            ))?;
    let new_body_len = layout
        .directory_start
        .checked_add(entry_bytes)
        .ok_or(CowBeTreeError::CorruptPage("leaf body length overflow"))?;
    let new_body_len = new_body_len
        .checked_add(directory_bytes)
        .ok_or(CowBeTreeError::CorruptPage("leaf body length overflow"))?;
    if HEADER_SIZE + new_body_len > page.len() {
        return Ok(None);
    }

    let mut entry_offset = HEADER_SIZE + layout.directory_start;
    let mut entry_offsets = layout.entry_offsets.clone();
    for message in messages {
        entry_offsets.push(
            u32::try_from(entry_offset - HEADER_SIZE)
                .map_err(|_| CowBeTreeError::CorruptPage("leaf entry offset too large"))?,
        );
        let entry_len = leaf_entry_message_encoded_len(message)?;
        encode_leaf_entry_at(
            page.get_mut(entry_offset..entry_offset + entry_len).ok_or(
                CowBeTreeError::CorruptPage("leaf append entry offset out of bounds"),
            )?,
            message,
        )?;
        entry_offset += entry_len;
    }
    write_leaf_directory(page, entry_offset, &entry_offsets)?;
    write_u32(page, BODY_LEN_OFF, new_body_len, "leaf page body too large")?;
    write_u16(page, COUNT_OFF, new_entry_count, "too many leaf entries")?;
    write_u16(
        page,
        HEADER_SIZE + layout.entry_count_offset,
        new_entry_count,
        "too many leaf entries",
    )?;

    Ok(Some(LeafEntryAppend {
        entry_count: new_entry_count,
        body_len: new_body_len,
        entry_bytes,
        message_count: messages.len(),
    }))
}

pub(crate) fn append_leaf_entry_prefix(
    page: &mut [u8],
    messages: &[BufferedMessage],
    max_leaf_entries: usize,
) -> Result<Option<LeafEntryAppend>, CowBeTreeError> {
    let Some(first) = messages.first() else {
        return Ok(None);
    };
    let Some(layout) = leaf_append_layout(page, &first.key)? else {
        return Ok(None);
    };
    let upper = leaf_append_upper(page, &layout)?;

    let mut previous_key = None;
    let mut entry_count = layout.entry_count;
    let mut entry_bytes = 0usize;
    let mut message_count = 0usize;
    for message in messages {
        if let Some(previous_key) = previous_key
            && previous_key >= message.key.as_slice()
        {
            break;
        }
        if upper.is_some_and(|upper| message.key.as_slice() >= upper) {
            break;
        }

        let new_entry_count = entry_count
            .checked_add(1)
            .ok_or(CowBeTreeError::CorruptPage("leaf entry count overflow"))?;
        if new_entry_count > max_leaf_entries {
            break;
        }
        let entry_len = leaf_entry_message_encoded_len(message)?;
        let directory_bytes =
            new_entry_count
                .checked_mul(LEAF_DIR_ENTRY_SIZE)
                .ok_or(CowBeTreeError::CorruptPage(
                    "leaf directory length overflow",
                ))?;
        let new_body_len = layout
            .directory_start
            .checked_add(entry_bytes)
            .and_then(|len| len.checked_add(entry_len))
            .and_then(|len| len.checked_add(directory_bytes))
            .ok_or(CowBeTreeError::CorruptPage("leaf body length overflow"))?;
        if HEADER_SIZE + new_body_len > page.len() {
            break;
        }

        previous_key = Some(message.key.as_slice());
        entry_count = new_entry_count;
        entry_bytes = entry_bytes
            .checked_add(entry_len)
            .ok_or(CowBeTreeError::CorruptPage("leaf append length overflow"))?;
        message_count += 1;
    }

    if message_count == 0 {
        return Ok(None);
    }

    let directory_bytes =
        entry_count
            .checked_mul(LEAF_DIR_ENTRY_SIZE)
            .ok_or(CowBeTreeError::CorruptPage(
                "leaf directory length overflow",
            ))?;
    let body_len = layout
        .directory_start
        .checked_add(entry_bytes)
        .and_then(|len| len.checked_add(directory_bytes))
        .ok_or(CowBeTreeError::CorruptPage("leaf body length overflow"))?;
    let mut entry_offset = HEADER_SIZE + layout.directory_start;
    let mut entry_offsets = layout.entry_offsets.clone();
    for message in &messages[..message_count] {
        entry_offsets.push(
            u32::try_from(entry_offset - HEADER_SIZE)
                .map_err(|_| CowBeTreeError::CorruptPage("leaf entry offset too large"))?,
        );
        let entry_len = leaf_entry_message_encoded_len(message)?;
        encode_leaf_entry_at(
            page.get_mut(entry_offset..entry_offset + entry_len).ok_or(
                CowBeTreeError::CorruptPage("leaf append entry offset out of bounds"),
            )?,
            message,
        )?;
        entry_offset += entry_len;
    }
    write_leaf_directory(page, entry_offset, &entry_offsets)?;
    write_u32(page, BODY_LEN_OFF, body_len, "leaf page body too large")?;
    write_u16(page, COUNT_OFF, entry_count, "too many leaf entries")?;
    write_u16(
        page,
        HEADER_SIZE + layout.entry_count_offset,
        entry_count,
        "too many leaf entries",
    )?;

    Ok(Some(LeafEntryAppend {
        entry_count,
        body_len,
        entry_bytes,
        message_count,
    }))
}

fn page_body_reader(page: &[u8]) -> Result<(PageKind, BodyReader<'_>), CowBeTreeError> {
    if page.len() < HEADER_SIZE {
        return Err(CowBeTreeError::CorruptPage("page shorter than header"));
    }
    let magic = read_u32(page, MAGIC_OFF)?;
    if magic != PAGE_MAGIC {
        return Err(CowBeTreeError::CorruptPage("bad page magic"));
    }
    if page[VERSION_OFF] != PAGE_VERSION {
        return Err(CowBeTreeError::CorruptPage("unsupported page version"));
    }

    let kind =
        PageKind::from_byte(page[KIND_OFF]).ok_or(CowBeTreeError::CorruptPage("bad page kind"))?;
    let body_len = read_u32(page, BODY_LEN_OFF)? as usize;
    if HEADER_SIZE + body_len > page.len() {
        return Err(CowBeTreeError::CorruptPage(
            "page body extends past page end",
        ));
    }

    Ok((
        kind,
        BodyReader::new(&page[HEADER_SIZE..HEADER_SIZE + body_len]),
    ))
}

struct InternalBufferLayout {
    child_count: usize,
    buffer_count: usize,
    buffer_count_offset: usize,
    body_len: usize,
    buffer_len: usize,
}

struct LeafAppendLayout {
    entry_count: usize,
    entry_count_offset: usize,
    directory_start: usize,
    entry_offsets: Vec<u32>,
    upper_offset: Option<usize>,
    upper_len: usize,
}

fn internal_buffer_layout(page: &[u8]) -> Result<Option<InternalBufferLayout>, CowBeTreeError> {
    let (kind, mut reader) = page_body_reader(page)?;
    if kind != PageKind::Internal {
        return Ok(None);
    }

    skip_fence(&mut reader)?;
    let child_count = reader.read_u16()? as usize;
    let separator_count = reader.read_u16()? as usize;
    let buffer_count_offset = reader.pos;
    let buffer_count = reader.read_u16()? as usize;
    if child_count == 0 {
        return Err(CowBeTreeError::EmptyInternalPage);
    }
    if separator_count + 1 != child_count {
        return Err(CowBeTreeError::CorruptPage(
            "internal separator count does not match child count",
        ));
    }

    let child_bytes = child_count
        .checked_mul(8)
        .ok_or(CowBeTreeError::CorruptPage("child array length overflow"))?;
    reader.read_exact(child_bytes)?;
    for _ in 0..separator_count {
        reader.read_bytes_u16_len()?;
    }

    let body_len = reader.data.len();
    let buffer_start = reader.pos;
    let buffer_len = body_len
        .checked_sub(buffer_start)
        .ok_or(CowBeTreeError::CorruptPage(
            "internal buffer offset overflow",
        ))?;
    for _ in 0..buffer_count {
        decode_message(&mut reader)?;
    }
    if reader.pos != body_len {
        return Err(CowBeTreeError::CorruptPage(
            "internal buffer length mismatch",
        ));
    }
    Ok(Some(InternalBufferLayout {
        child_count,
        buffer_count,
        buffer_count_offset,
        body_len,
        buffer_len,
    }))
}

fn leaf_append_layout(page: &[u8], key: &[u8]) -> Result<Option<LeafAppendLayout>, CowBeTreeError> {
    let (kind, mut reader) = page_body_reader(page)?;
    if kind != PageKind::Leaf {
        return Ok(None);
    }

    let lower = reader.read_bytes_u16_len()?;
    if key < lower {
        return Ok(None);
    }
    let upper_len = reader.read_u16()?;
    let upper_offset = (upper_len != NONE_FENCE_LEN).then_some(reader.pos);
    if upper_len != NONE_FENCE_LEN {
        let upper = reader.read_exact(upper_len as usize)?;
        if key >= upper {
            return Ok(None);
        }
    }

    let entry_count_offset = reader.pos;
    let entry_count = reader.read_u16()? as usize;
    let entries_start = reader.pos;
    let directory_start = leaf_directory_start(reader.data.len(), entry_count)?;
    if entries_start > directory_start {
        return Err(CowBeTreeError::CorruptPage(
            "leaf directory overlaps entry header",
        ));
    }

    let mut entry_offsets = Vec::with_capacity(entry_count);
    for idx in 0..entry_count {
        entry_offsets.push(
            u32::try_from(read_leaf_entry_offset(reader.data, directory_start, idx)?)
                .map_err(|_| CowBeTreeError::CorruptPage("leaf entry offset too large"))?,
        );
    }
    let last_key = if entry_count == 0 {
        None
    } else {
        let offset = read_leaf_entry_offset(reader.data, directory_start, entry_count - 1)?;
        Some(read_leaf_entry_key_at(reader.data, offset)?)
    };
    if let Some(last_key) = last_key
        && last_key >= key
    {
        return Ok(None);
    }

    Ok(Some(LeafAppendLayout {
        entry_count,
        entry_count_offset,
        directory_start,
        entry_offsets,
        upper_offset,
        upper_len: if upper_len == NONE_FENCE_LEN {
            0
        } else {
            upper_len as usize
        },
    }))
}

fn leaf_append_upper<'a>(
    page: &'a [u8],
    layout: &LeafAppendLayout,
) -> Result<Option<&'a [u8]>, CowBeTreeError> {
    let Some(offset) = layout.upper_offset else {
        return Ok(None);
    };
    let start = HEADER_SIZE
        .checked_add(offset)
        .ok_or(CowBeTreeError::CorruptPage("upper fence offset overflow"))?;
    let end = start
        .checked_add(layout.upper_len)
        .ok_or(CowBeTreeError::CorruptPage("upper fence offset overflow"))?;
    page.get(start..end)
        .ok_or(CowBeTreeError::CorruptPage(
            "upper fence read out of bounds",
        ))
        .map(Some)
}

fn encode_fence(writer: &mut BodyWriter, fence: &Fence) -> Result<(), CowBeTreeError> {
    writer.push_bytes_with_u16_len(&fence.lower, "lower fence too large")?;
    match &fence.upper {
        Some(upper) => writer.push_bytes_with_u16_len(upper, "upper fence too large")?,
        None => writer.push_u16_raw(NONE_FENCE_LEN),
    }
    Ok(())
}

fn decode_fence(reader: &mut BodyReader<'_>) -> Result<Fence, CowBeTreeError> {
    let lower = reader.read_bytes_u16_len()?.to_vec();
    let upper_len = reader.read_u16()?;
    let upper = if upper_len == NONE_FENCE_LEN {
        None
    } else {
        Some(reader.read_exact(upper_len as usize)?.to_vec())
    };
    Ok(Fence { lower, upper })
}

fn skip_fence(reader: &mut BodyReader<'_>) -> Result<(), CowBeTreeError> {
    reader.read_bytes_u16_len()?;
    let upper_len = reader.read_u16()?;
    if upper_len != NONE_FENCE_LEN {
        reader.read_exact(upper_len as usize)?;
    }
    Ok(())
}

fn encode_version(writer: &mut BodyWriter, version: &VersionRecord) -> Result<(), CowBeTreeError> {
    writer.push_u64(version.commit_ts);
    writer.push_u8(u8::from(version.deleted));
    writer.push_bytes_with_u32_len(&version.value, "version value too large")
}

fn decode_version(reader: &mut BodyReader<'_>) -> Result<VersionRecord, CowBeTreeError> {
    let commit_ts = reader.read_u64()?;
    let deleted = match reader.read_u8()? {
        0 => false,
        1 => true,
        _ => return Err(CowBeTreeError::CorruptPage("bad version deletion flag")),
    };
    let value = reader.read_bytes_u32_len()?.to_vec();
    Ok(VersionRecord {
        commit_ts,
        value,
        deleted,
    })
}

fn encode_message(
    writer: &mut BodyWriter,
    message: &BufferedMessage,
) -> Result<(), CowBeTreeError> {
    writer.push_bytes_with_u16_len(&message.key, "message key too large")?;
    encode_version(writer, &message.version)
}

fn encode_message_at(dst: &mut [u8], message: &BufferedMessage) -> Result<(), CowBeTreeError> {
    if dst.len() != message.encoded_len() {
        return Err(CowBeTreeError::CorruptPage(
            "message append length mismatch",
        ));
    }
    let key_len = u16::try_from(message.key.len())
        .map_err(|_| CowBeTreeError::CorruptPage("message key too large"))?;
    let value_len = u32::try_from(message.version.value.len())
        .map_err(|_| CowBeTreeError::CorruptPage("version value too large"))?;

    let mut offset = 0usize;
    dst[offset..offset + 2].copy_from_slice(&key_len.to_le_bytes());
    offset += 2;
    dst[offset..offset + message.key.len()].copy_from_slice(&message.key);
    offset += message.key.len();
    dst[offset..offset + 8].copy_from_slice(&message.version.commit_ts.to_le_bytes());
    offset += 8;
    dst[offset] = u8::from(message.version.deleted);
    offset += 1;
    dst[offset..offset + 4].copy_from_slice(&value_len.to_le_bytes());
    offset += 4;
    dst[offset..offset + message.version.value.len()].copy_from_slice(&message.version.value);
    Ok(())
}

fn encode_message_kv_at(
    dst: &mut [u8],
    key: &[u8],
    value: &[u8],
    commit_ts: Timestamp,
) -> Result<(), CowBeTreeError> {
    let key_len = u16::try_from(key.len())
        .map_err(|_| CowBeTreeError::CorruptPage("message key too large"))?;
    let value_len = u32::try_from(value.len())
        .map_err(|_| CowBeTreeError::CorruptPage("version value too large"))?;
    let expected = usize::from(key_len) + value_len as usize + 15;
    if dst.len() != expected {
        return Err(CowBeTreeError::CorruptPage(
            "message kv append length mismatch",
        ));
    }

    let mut offset = 0usize;
    dst[offset..offset + 2].copy_from_slice(&key_len.to_le_bytes());
    offset += 2;
    dst[offset..offset + key.len()].copy_from_slice(key);
    offset += key.len();
    dst[offset..offset + 8].copy_from_slice(&commit_ts.to_le_bytes());
    offset += 8;
    dst[offset] = 0;
    offset += 1;
    dst[offset..offset + 4].copy_from_slice(&value_len.to_le_bytes());
    offset += 4;
    dst[offset..offset + value.len()].copy_from_slice(value);
    Ok(())
}

fn leaf_entry_message_encoded_len(message: &BufferedMessage) -> Result<usize, CowBeTreeError> {
    let key_len = u16::try_from(message.key.len())
        .map_err(|_| CowBeTreeError::CorruptPage("leaf key too large"))?;
    let value_len = u32::try_from(message.version.value.len())
        .map_err(|_| CowBeTreeError::CorruptPage("version value too large"))?;
    Ok(usize::from(key_len) + value_len as usize + 17)
}

fn encode_leaf_entry_at(dst: &mut [u8], message: &BufferedMessage) -> Result<(), CowBeTreeError> {
    if dst.len() != leaf_entry_message_encoded_len(message)? {
        return Err(CowBeTreeError::CorruptPage("leaf append length mismatch"));
    }
    let key_len = u16::try_from(message.key.len())
        .map_err(|_| CowBeTreeError::CorruptPage("leaf key too large"))?;
    let value_len = u32::try_from(message.version.value.len())
        .map_err(|_| CowBeTreeError::CorruptPage("version value too large"))?;

    let mut offset = 0usize;
    dst[offset..offset + 2].copy_from_slice(&key_len.to_le_bytes());
    offset += 2;
    dst[offset..offset + message.key.len()].copy_from_slice(&message.key);
    offset += message.key.len();
    dst[offset..offset + 2].copy_from_slice(&1u16.to_le_bytes());
    offset += 2;
    dst[offset..offset + 8].copy_from_slice(&message.version.commit_ts.to_le_bytes());
    offset += 8;
    dst[offset] = u8::from(message.version.deleted);
    offset += 1;
    dst[offset..offset + 4].copy_from_slice(&value_len.to_le_bytes());
    offset += 4;
    dst[offset..offset + message.version.value.len()].copy_from_slice(&message.version.value);
    Ok(())
}

fn encode_leaf_kv_at(
    dst: &mut [u8],
    key: &[u8],
    value: &[u8],
    commit_ts: Timestamp,
) -> Result<(), CowBeTreeError> {
    let key_len =
        u16::try_from(key.len()).map_err(|_| CowBeTreeError::CorruptPage("leaf key too large"))?;
    let value_len = u32::try_from(value.len())
        .map_err(|_| CowBeTreeError::CorruptPage("version value too large"))?;
    let expected = usize::from(key_len) + value_len as usize + 17;
    if dst.len() != expected {
        return Err(CowBeTreeError::CorruptPage(
            "leaf kv append length mismatch",
        ));
    }

    let mut offset = 0usize;
    dst[offset..offset + 2].copy_from_slice(&key_len.to_le_bytes());
    offset += 2;
    dst[offset..offset + key.len()].copy_from_slice(key);
    offset += key.len();
    dst[offset..offset + 2].copy_from_slice(&1u16.to_le_bytes());
    offset += 2;
    dst[offset..offset + 8].copy_from_slice(&commit_ts.to_le_bytes());
    offset += 8;
    dst[offset] = 0;
    offset += 1;
    dst[offset..offset + 4].copy_from_slice(&value_len.to_le_bytes());
    offset += 4;
    dst[offset..offset + value.len()].copy_from_slice(value);
    Ok(())
}

fn decode_message(reader: &mut BodyReader<'_>) -> Result<BufferedMessage, CowBeTreeError> {
    let key = reader.read_bytes_u16_len()?.to_vec();
    let version = decode_version(reader)?;
    Ok(BufferedMessage { key, version })
}

fn lookup_leaf<'a>(
    mut reader: BodyReader<'a>,
    key: &[u8],
    read_ts: Timestamp,
) -> Result<LookupStep<'a>, CowBeTreeError> {
    let entry_count = reader.read_u16()? as usize;
    let entries_start = reader.pos;
    let directory_start = leaf_directory_start(reader.data.len(), entry_count)?;
    if entries_start > directory_start {
        return Err(CowBeTreeError::CorruptPage(
            "leaf directory overlaps entry header",
        ));
    }

    let Some(entry_idx) = find_leaf_entry(reader.data, directory_start, entry_count, key)? else {
        return Ok(LookupStep::Leaf { visible: None });
    };
    let entry_offset = read_leaf_entry_offset(reader.data, directory_start, entry_idx)?;
    let mut entry_reader = BodyReader::at(reader.data, entry_offset)?;
    let entry_key = entry_reader.read_bytes_u16_len()?;
    if entry_key != key {
        return Err(CowBeTreeError::CorruptPage(
            "leaf directory selected the wrong key",
        ));
    }
    let version_count = entry_reader.read_u16()? as usize;
    Ok(LookupStep::Leaf {
        visible: first_visible_version(&mut entry_reader, version_count, read_ts)?,
    })
}

fn lookup_internal<'a>(
    mut reader: BodyReader<'a>,
    key: &[u8],
    read_ts: Timestamp,
) -> Result<LookupStep<'a>, CowBeTreeError> {
    let child_count = reader.read_u16()? as usize;
    let separator_count = reader.read_u16()? as usize;
    let buffer_count = reader.read_u16()? as usize;
    if child_count == 0 {
        return Err(CowBeTreeError::EmptyInternalPage);
    }
    if separator_count + 1 != child_count {
        return Err(CowBeTreeError::CorruptPage(
            "internal separator count does not match child count",
        ));
    }

    let child_start = reader.pos;
    let child_bytes = child_count
        .checked_mul(8)
        .ok_or(CowBeTreeError::CorruptPage("child array length overflow"))?;
    reader.read_exact(child_bytes)?;

    let mut child_idx = 0usize;
    for _ in 0..separator_count {
        let separator = reader.read_bytes_u16_len()?;
        if separator <= key {
            child_idx += 1;
        }
    }

    let child_off = child_start
        .checked_add(child_idx * 8)
        .ok_or(CowBeTreeError::CorruptPage("child offset overflow"))?;
    let raw_child = reader
        .data
        .get(child_off..child_off + 8)
        .ok_or(CowBeTreeError::CorruptPage("child read out of bounds"))?;
    let child_swip = Swip::from_raw(u64::from_le_bytes(raw_child.try_into().unwrap()));

    let mut visible_buffer = None;
    for _ in 0..buffer_count {
        let message_key = reader.read_bytes_u16_len()?;
        let version = read_raw_version(&mut reader)?;
        if message_key == key && version.commit_ts <= read_ts {
            merge_raw_visible(&mut visible_buffer, version);
        }
    }

    Ok(LookupStep::Internal {
        child_swip,
        child_slot: child_idx as u16,
        visible_buffer,
        buffer_count,
    })
}

fn first_visible_version<'a>(
    reader: &mut BodyReader<'a>,
    version_count: usize,
    read_ts: Timestamp,
) -> Result<Option<RawVisibleVersion<'a>>, CowBeTreeError> {
    let mut visible = None;
    for _ in 0..version_count {
        let version = read_raw_version(reader)?;
        if version.commit_ts <= read_ts {
            merge_raw_visible(&mut visible, version);
        }
    }
    Ok(visible)
}

fn leaf_directory_start(body_len: usize, entry_count: usize) -> Result<usize, CowBeTreeError> {
    let directory_len =
        entry_count
            .checked_mul(LEAF_DIR_ENTRY_SIZE)
            .ok_or(CowBeTreeError::CorruptPage(
                "leaf directory length overflow",
            ))?;
    body_len
        .checked_sub(directory_len)
        .ok_or(CowBeTreeError::CorruptPage(
            "leaf directory extends before body start",
        ))
}

fn read_leaf_entry_offset(
    data: &[u8],
    directory_start: usize,
    idx: usize,
) -> Result<usize, CowBeTreeError> {
    let offset_start =
        directory_start
            .checked_add(idx.checked_mul(LEAF_DIR_ENTRY_SIZE).ok_or(
                CowBeTreeError::CorruptPage("leaf directory offset overflow"),
            )?)
            .ok_or(CowBeTreeError::CorruptPage(
                "leaf directory offset overflow",
            ))?;
    let offset_end =
        offset_start
            .checked_add(LEAF_DIR_ENTRY_SIZE)
            .ok_or(CowBeTreeError::CorruptPage(
                "leaf directory offset overflow",
            ))?;
    let bytes = data
        .get(offset_start..offset_end)
        .ok_or(CowBeTreeError::CorruptPage(
            "leaf directory read out of bounds",
        ))?;
    let offset = u32::from_le_bytes(bytes.try_into().unwrap()) as usize;
    if offset >= directory_start {
        return Err(CowBeTreeError::CorruptPage(
            "leaf entry offset points into directory",
        ));
    }
    Ok(offset)
}

fn read_leaf_entry_key_at(data: &[u8], offset: usize) -> Result<&[u8], CowBeTreeError> {
    let mut reader = BodyReader::at(data, offset)?;
    reader.read_bytes_u16_len()
}

fn find_leaf_entry(
    data: &[u8],
    directory_start: usize,
    entry_count: usize,
    key: &[u8],
) -> Result<Option<usize>, CowBeTreeError> {
    let mut low = 0usize;
    let mut high = entry_count;
    while low < high {
        let mid = low + (high - low) / 2;
        let offset = read_leaf_entry_offset(data, directory_start, mid)?;
        let entry_key = read_leaf_entry_key_at(data, offset)?;
        match entry_key.cmp(key) {
            std::cmp::Ordering::Less => low = mid + 1,
            std::cmp::Ordering::Equal => return Ok(Some(mid)),
            std::cmp::Ordering::Greater => high = mid,
        }
    }
    Ok(None)
}

fn write_leaf_directory(
    page: &mut [u8],
    directory_offset: usize,
    entry_offsets: &[u32],
) -> Result<(), CowBeTreeError> {
    let directory_len =
        entry_offsets
            .len()
            .checked_mul(LEAF_DIR_ENTRY_SIZE)
            .ok_or(CowBeTreeError::CorruptPage(
                "leaf directory length overflow",
            ))?;
    let directory_end =
        directory_offset
            .checked_add(directory_len)
            .ok_or(CowBeTreeError::CorruptPage(
                "leaf directory offset overflow",
            ))?;
    let directory =
        page.get_mut(directory_offset..directory_end)
            .ok_or(CowBeTreeError::CorruptPage(
                "leaf directory write out of bounds",
            ))?;
    for (idx, offset) in entry_offsets.iter().copied().enumerate() {
        let start = idx * LEAF_DIR_ENTRY_SIZE;
        directory[start..start + LEAF_DIR_ENTRY_SIZE].copy_from_slice(&offset.to_le_bytes());
    }
    Ok(())
}

fn read_raw_version<'a>(
    reader: &mut BodyReader<'a>,
) -> Result<RawVisibleVersion<'a>, CowBeTreeError> {
    let commit_ts = reader.read_u64()?;
    let deleted = match reader.read_u8()? {
        0 => false,
        1 => true,
        _ => return Err(CowBeTreeError::CorruptPage("bad version deletion flag")),
    };
    let value = reader.read_bytes_u32_len()?;
    Ok(RawVisibleVersion {
        commit_ts,
        deleted,
        value,
    })
}

fn merge_raw_visible<'a>(
    visible: &mut Option<RawVisibleVersion<'a>>,
    candidate: RawVisibleVersion<'a>,
) {
    if visible
        .as_ref()
        .is_none_or(|existing| candidate.commit_ts > existing.commit_ts)
    {
        *visible = Some(candidate);
    }
}

fn finish_page(
    page: &mut [u8],
    kind: PageKind,
    count: usize,
    body: Vec<u8>,
) -> Result<usize, CowBeTreeError> {
    let needed = HEADER_SIZE + body.len();
    if needed > page.len() {
        return Err(page_overflow(
            match kind {
                PageKind::Leaf => "leaf",
                PageKind::Internal => "internal",
            },
            needed,
            page.len(),
        ));
    }

    page.fill(0);
    write_u16(page, COUNT_OFF, count, "page item count too large")?;
    write_u32(page, BODY_LEN_OFF, body.len(), "page body too large")?;
    page[MAGIC_OFF..MAGIC_OFF + 4].copy_from_slice(&PAGE_MAGIC.to_le_bytes());
    page[VERSION_OFF] = PAGE_VERSION;
    page[KIND_OFF] = kind as u8;
    page_header::write_page_type(page, kind.page_type());
    page[HEADER_SIZE..HEADER_SIZE + body.len()].copy_from_slice(&body);
    Ok(needed)
}

fn encoded_body_len(
    kind: PageKind,
    body: Vec<u8>,
    capacity: usize,
) -> Result<usize, CowBeTreeError> {
    let needed = HEADER_SIZE
        .checked_add(body.len())
        .ok_or(CowBeTreeError::CorruptPage("page length overflow"))?;
    if needed > capacity {
        return Err(page_overflow(
            match kind {
                PageKind::Leaf => "leaf",
                PageKind::Internal => "internal",
            },
            needed,
            capacity,
        ));
    }
    Ok(needed)
}

fn decode_leaf(mut reader: BodyReader<'_>, fence: Fence) -> Result<NodePage, CowBeTreeError> {
    let entry_count = reader.read_u16()? as usize;
    let directory_start = leaf_directory_start(reader.data.len(), entry_count)?;
    if reader.pos > directory_start {
        return Err(CowBeTreeError::CorruptPage(
            "leaf directory overlaps entry header",
        ));
    }

    let mut entries = Vec::with_capacity(entry_count);
    for idx in 0..entry_count {
        let expected_offset = read_leaf_entry_offset(reader.data, directory_start, idx)?;
        if expected_offset != reader.pos {
            return Err(CowBeTreeError::CorruptPage(
                "leaf entry offset directory mismatch",
            ));
        }
        let key = reader.read_bytes_u16_len()?.to_vec();
        let version_count = reader.read_u16()? as usize;
        let mut versions = Vec::with_capacity(version_count);
        for _ in 0..version_count {
            versions.push(decode_version(&mut reader)?);
        }
        entries.push(LeafEntry { key, versions });
    }
    if reader.pos != directory_start {
        return Err(CowBeTreeError::CorruptPage(
            "leaf entries do not end at offset directory",
        ));
    }
    for idx in 0..entry_count {
        let offset = reader.read_u32()? as usize;
        let expected_offset = read_leaf_entry_offset(reader.data, directory_start, idx)?;
        if offset != expected_offset {
            return Err(CowBeTreeError::CorruptPage(
                "leaf entry offset directory changed while decoding",
            ));
        }
    }
    validate_leaf_order(&entries)?;
    Ok(NodePage::Leaf { fence, entries })
}

fn decode_internal(mut reader: BodyReader<'_>, fence: Fence) -> Result<NodePage, CowBeTreeError> {
    let child_count = reader.read_u16()? as usize;
    let separator_count = reader.read_u16()? as usize;
    let buffer_count = reader.read_u16()? as usize;
    if child_count == 0 {
        return Err(CowBeTreeError::EmptyInternalPage);
    }
    if separator_count + 1 != child_count {
        return Err(CowBeTreeError::CorruptPage(
            "internal separator count does not match child count",
        ));
    }

    let mut children = Vec::with_capacity(child_count);
    for _ in 0..child_count {
        let raw = reader.read_u64()?;
        let swip = Swip::from_raw(raw);
        // Child SWIPs may be swizzled (Hot/Cool) in resident pages; resolve
        // to the page ID via the frame header. Evicted SWIPs decode directly.
        let child_pid = match BufferFrameRef::from_hot_swip(swip) {
            Some(frame) => frame.pid(),
            None => swip.as_page_id(),
        };
        children.push(child_pid);
    }

    let mut separators = Vec::with_capacity(separator_count);
    for _ in 0..separator_count {
        separators.push(reader.read_bytes_u16_len()?.to_vec());
    }
    validate_separator_order(&separators)?;

    let mut buffer = Vec::with_capacity(buffer_count);
    for _ in 0..buffer_count {
        buffer.push(decode_message(&mut reader)?);
    }
    if reader.pos != reader.data.len() {
        return Err(CowBeTreeError::CorruptPage(
            "internal buffer length mismatch",
        ));
    }

    Ok(NodePage::Internal {
        fence,
        children,
        separators,
        buffer,
    })
}

fn validate_leaf_order(entries: &[LeafEntry]) -> Result<(), CowBeTreeError> {
    if entries
        .windows(2)
        .any(|pair| pair[0].key.as_slice() >= pair[1].key.as_slice())
    {
        return Err(CowBeTreeError::CorruptPage(
            "leaf keys are not strictly sorted",
        ));
    }
    Ok(())
}

fn validate_separator_order(separators: &[Vec<u8>]) -> Result<(), CowBeTreeError> {
    if separators
        .windows(2)
        .any(|pair| pair[0].as_slice() >= pair[1].as_slice())
    {
        return Err(CowBeTreeError::CorruptPage(
            "internal separators are not strictly sorted",
        ));
    }
    Ok(())
}

fn write_u16(
    page: &mut [u8],
    offset: usize,
    value: usize,
    msg: &'static str,
) -> Result<(), CowBeTreeError> {
    let value = u16::try_from(value).map_err(|_| CowBeTreeError::CorruptPage(msg))?;
    let end = offset
        .checked_add(2)
        .ok_or(CowBeTreeError::CorruptPage("u16 offset overflow"))?;
    page.get_mut(offset..end)
        .ok_or(CowBeTreeError::CorruptPage("u16 write out of bounds"))?
        .copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_u32(
    page: &mut [u8],
    offset: usize,
    value: usize,
    msg: &'static str,
) -> Result<(), CowBeTreeError> {
    let value = u32::try_from(value).map_err(|_| CowBeTreeError::CorruptPage(msg))?;
    let end = offset
        .checked_add(4)
        .ok_or(CowBeTreeError::CorruptPage("u32 offset overflow"))?;
    page.get_mut(offset..end)
        .ok_or(CowBeTreeError::CorruptPage("u32 write out of bounds"))?
        .copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn read_u32(page: &[u8], offset: usize) -> Result<u32, CowBeTreeError> {
    let end = offset
        .checked_add(4)
        .ok_or(CowBeTreeError::CorruptPage("u32 offset overflow"))?;
    let bytes = page
        .get(offset..end)
        .ok_or(CowBeTreeError::CorruptPage("u32 read out of bounds"))?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn page_overflow(kind: &'static str, needed: usize, capacity: usize) -> CowBeTreeError {
    CowBeTreeError::PageOverflow {
        kind,
        needed,
        capacity,
    }
}

struct BodyWriter {
    data: Vec<u8>,
}

impl BodyWriter {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn len_u32(&self, msg: &'static str) -> Result<u32, CowBeTreeError> {
        u32::try_from(self.data.len()).map_err(|_| CowBeTreeError::CorruptPage(msg))
    }

    fn push_u8(&mut self, value: u8) {
        self.data.push(value);
    }

    fn push_u16(&mut self, value: usize, msg: &'static str) -> Result<(), CowBeTreeError> {
        let value = u16::try_from(value).map_err(|_| CowBeTreeError::CorruptPage(msg))?;
        self.push_u16_raw(value);
        Ok(())
    }

    fn push_u16_raw(&mut self, value: u16) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32(&mut self, value: usize, msg: &'static str) -> Result<(), CowBeTreeError> {
        let value = u32::try_from(value).map_err(|_| CowBeTreeError::CorruptPage(msg))?;
        self.push_u32_raw(value);
        Ok(())
    }

    fn push_u32_raw(&mut self, value: u32) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(&mut self, value: u64) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    fn push_bytes_with_u16_len(
        &mut self,
        bytes: &[u8],
        msg: &'static str,
    ) -> Result<(), CowBeTreeError> {
        self.push_u16(bytes.len(), msg)?;
        self.data.extend_from_slice(bytes);
        Ok(())
    }

    fn push_bytes_with_u32_len(
        &mut self,
        bytes: &[u8],
        msg: &'static str,
    ) -> Result<(), CowBeTreeError> {
        self.push_u32(bytes.len(), msg)?;
        self.data.extend_from_slice(bytes);
        Ok(())
    }

    fn into_inner(self) -> Vec<u8> {
        self.data
    }
}

#[derive(Clone, Copy)]
struct BodyReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BodyReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn at(data: &'a [u8], pos: usize) -> Result<Self, CowBeTreeError> {
        if pos > data.len() {
            return Err(CowBeTreeError::CorruptPage(
                "body reader offset out of bounds",
            ));
        }
        Ok(Self { data, pos })
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], CowBeTreeError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or(CowBeTreeError::CorruptPage("body offset overflow"))?;
        let bytes = self
            .data
            .get(self.pos..end)
            .ok_or(CowBeTreeError::CorruptPage("body read out of bounds"))?;
        self.pos = end;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8, CowBeTreeError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, CowBeTreeError> {
        Ok(u16::from_le_bytes(self.read_exact(2)?.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> Result<u32, CowBeTreeError> {
        Ok(u32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, CowBeTreeError> {
        Ok(u64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    fn read_bytes_u16_len(&mut self) -> Result<&'a [u8], CowBeTreeError> {
        let len = self.read_u16()? as usize;
        self.read_exact(len)
    }

    fn read_bytes_u32_len(&mut self) -> Result<&'a [u8], CowBeTreeError> {
        let len = u32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()) as usize;
        self.read_exact(len)
    }
}

/// Zero-copy reader for leaf pages.  Pre-parses the page header, fence, and
/// directory to provide O(1) entry access by index without any heap
/// allocation.
pub(crate) struct LeafPageReader<'a> {
    body: &'a [u8],
    directory_start: usize,
    entry_count: usize,
}

impl<'a> LeafPageReader<'a> {
    pub(crate) fn new(page: &'a [u8]) -> Result<Self, CowBeTreeError> {
        let (kind, mut reader) = page_body_reader(page)?;
        if kind != PageKind::Leaf {
            return Err(CowBeTreeError::CorruptPage("expected leaf page"));
        }
        let body = reader.data;
        skip_fence(&mut reader)?;
        let entry_count = reader.read_u16()? as usize;
        let directory_start = leaf_directory_start(body.len(), entry_count)?;
        if reader.pos > directory_start {
            return Err(CowBeTreeError::CorruptPage(
                "leaf directory overlaps entry header",
            ));
        }
        Ok(Self {
            body,
            directory_start,
            entry_count,
        })
    }

    pub(crate) fn entry_count(&self) -> usize {
        self.entry_count
    }

    pub(crate) fn entry_key(&self, idx: usize) -> Result<&'a [u8], CowBeTreeError> {
        let offset = read_leaf_entry_offset(self.body, self.directory_start, idx)?;
        read_leaf_entry_key_at(self.body, offset)
    }

    /// Returns the raw entry bytes (key_len + key + version_count + versions)
    /// for entry `idx`.  The bytes are a zero-copy slice into the page body.
    pub(crate) fn entry_raw(&self, idx: usize) -> Result<&'a [u8], CowBeTreeError> {
        let start = read_leaf_entry_offset(self.body, self.directory_start, idx)?;
        let end = if idx + 1 < self.entry_count {
            read_leaf_entry_offset(self.body, self.directory_start, idx + 1)?
        } else {
            self.directory_start
        };
        if start > end {
            return Err(CowBeTreeError::CorruptPage(
                "leaf entry has negative length",
            ));
        }
        self.body
            .get(start..end)
            .ok_or(CowBeTreeError::CorruptPage("leaf entry raw out of bounds"))
    }

    pub(crate) fn fence(&self) -> Result<Fence, CowBeTreeError> {
        let mut reader = BodyReader::new(self.body);
        decode_fence(&mut reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use pagebox_storage::buffer_frame::PAGE_SIZE;

    use crate::message::Timestamp;

    fn version(commit_ts: Timestamp, value: &[u8]) -> VersionRecord {
        VersionRecord {
            commit_ts,
            value: value.to_vec(),
            deleted: false,
        }
    }

    #[test]
    fn leaf_page_roundtrips_retained_versions_and_fences() {
        let fence = Fence {
            lower: b"a".to_vec(),
            upper: Some(b"z".to_vec()),
        };
        let entries = vec![
            LeafEntry {
                key: b"a".to_vec(),
                versions: vec![version(10, b"new"), version(1, b"old")],
            },
            LeafEntry {
                key: b"m".to_vec(),
                versions: vec![VersionRecord {
                    commit_ts: 8,
                    value: Vec::new(),
                    deleted: true,
                }],
            },
        ];

        let mut page = [0u8; PAGE_SIZE];
        encode_leaf_page(&mut page, &fence, &entries).unwrap();
        assert_eq!(page_header::read_page_type(&page), PageType::BeTreeLeaf);
        assert_eq!(
            decode_page(&page).unwrap(),
            NodePage::Leaf { fence, entries },
            "leaf page decode should preserve fence and retained MVCC versions"
        );
    }

    #[test]
    fn leaf_page_writes_offset_directory_for_binary_lookup() {
        let entries = vec![
            LeafEntry {
                key: b"k01".to_vec(),
                versions: vec![version(1, b"v01")],
            },
            LeafEntry {
                key: b"k02".to_vec(),
                versions: vec![version(2, b"v02")],
            },
            LeafEntry {
                key: b"k03".to_vec(),
                versions: vec![version(3, b"v03")],
            },
        ];

        let mut page = [0u8; PAGE_SIZE];
        encode_leaf_page(&mut page, &Fence::root(), &entries).unwrap();
        let body_len = read_u32(&page, BODY_LEN_OFF).unwrap() as usize;
        let directory_start = HEADER_SIZE + body_len - entries.len() * LEAF_DIR_ENTRY_SIZE;
        let offsets = (0..entries.len())
            .map(|idx| {
                let start = directory_start + idx * LEAF_DIR_ENTRY_SIZE;
                u32::from_le_bytes(page[start..start + LEAF_DIR_ENTRY_SIZE].try_into().unwrap())
                    as usize
            })
            .collect::<Vec<_>>();

        assert_eq!(
            offsets[0], 6,
            "first root-fence entry should start immediately after leaf entry count"
        );
        assert!(
            offsets.windows(2).all(|pair| pair[0] < pair[1]),
            "leaf directory offsets should increase with sorted entries"
        );

        let LookupStep::Leaf {
            visible: Some(visible),
        } = lookup_step(&page, b"k03", 3).unwrap()
        else {
            panic!("directory-backed lookup should find the final key");
        };
        assert_eq!(visible.value, b"v03", "lookup value mismatch");
    }

    #[test]
    fn internal_page_roundtrips_evicted_child_swips_and_buffered_messages() {
        let fence = Fence::root();
        let children = vec![11, 22, 33];
        let separators = vec![b"k10".to_vec(), b"k20".to_vec()];
        let buffer = vec![
            BufferedMessage::put(b"k05", 5, b"v05"),
            BufferedMessage::delete(b"k25", 7),
        ];

        let mut page = [0u8; PAGE_SIZE];
        encode_internal_page(&mut page, &fence, &children, &separators, &buffer).unwrap();
        assert_eq!(page_header::read_page_type(&page), PageType::BeTreeInternal);
        assert_eq!(
            decode_page(&page).unwrap(),
            NodePage::Internal {
                fence,
                children,
                separators,
                buffer,
            },
            "internal page decode should preserve children as page-id swips and buffered messages"
        );
    }

    #[test]
    fn raw_lookup_reads_leaf_visible_version_without_decoding_entries() {
        let entries = vec![
            LeafEntry {
                key: b"k10".to_vec(),
                versions: vec![version(20, b"new"), version(5, b"old")],
            },
            LeafEntry {
                key: b"k20".to_vec(),
                versions: vec![VersionRecord {
                    commit_ts: 7,
                    value: Vec::new(),
                    deleted: true,
                }],
            },
        ];

        let mut page = [0u8; PAGE_SIZE];
        encode_leaf_page(&mut page, &Fence::root(), &entries).unwrap();

        let LookupStep::Leaf {
            visible: Some(visible),
        } = lookup_step(&page, b"k10", 10).unwrap()
        else {
            panic!("raw lookup should find the older visible leaf version");
        };
        assert_eq!(visible.commit_ts, 5, "visible commit timestamp mismatch");
        assert_eq!(visible.value, b"old", "visible value mismatch");

        let LookupStep::Leaf { visible } = lookup_step(&page, b"k15", 10).unwrap() else {
            panic!("raw lookup should stay on the leaf page");
        };
        assert!(
            visible.is_none(),
            "missing key should not materialize a version"
        );
    }

    #[test]
    fn raw_lookup_routes_internal_child_and_reads_matching_buffer_message() {
        let children = vec![11, 22, 33];
        let separators = vec![b"k10".to_vec(), b"k20".to_vec()];
        let buffer = vec![
            BufferedMessage::put(b"k30", 30, b"other"),
            BufferedMessage::delete(b"k25", 20),
            BufferedMessage::put(b"k25", 12, b"new"),
            BufferedMessage::put(b"k25", 5, b"old"),
        ];

        let mut page = [0u8; PAGE_SIZE];
        encode_internal_page(&mut page, &Fence::root(), &children, &separators, &buffer).unwrap();

        let LookupStep::Internal {
            child_swip,
            visible_buffer: Some(visible),
            buffer_count,
            ..
        } = lookup_step(&page, b"k25", 15).unwrap()
        else {
            panic!("raw lookup should route through the internal page");
        };
        assert_eq!(
            child_swip.as_page_id(),
            33,
            "child routing should follow separators"
        );
        assert_eq!(buffer_count, 4, "buffer count should be reported");
        assert_eq!(
            visible.value, b"new",
            "newest visible buffer message should win"
        );
        assert_eq!(visible.commit_ts, 12, "buffer timestamp mismatch");
    }

    #[test]
    fn lookup_child_swip_routes_without_scanning_buffer() {
        let children = vec![11, 22, 33];
        let separators = vec![b"k10".to_vec(), b"k20".to_vec()];
        let buffer = vec![
            BufferedMessage::put(b"k25", 12, b"buffered"),
            BufferedMessage::put(b"k05", 5, b"early"),
        ];

        let mut page = [0u8; PAGE_SIZE];
        encode_internal_page(&mut page, &Fence::root(), &children, &separators, &buffer).unwrap();

        // Keys below the first separator route to child 0.
        assert_eq!(
            lookup_child_swip(&page, b"k00")
                .unwrap()
                .unwrap()
                .as_page_id(),
            11,
            "key below first separator routes to first child",
        );
        // Key equal to a separator routes right (separator <= key).
        assert_eq!(
            lookup_child_swip(&page, b"k10")
                .unwrap()
                .unwrap()
                .as_page_id(),
            22,
            "key equal to a separator routes past it",
        );
        // Key above the last separator routes to the final child.
        assert_eq!(
            lookup_child_swip(&page, b"zzz")
                .unwrap()
                .unwrap()
                .as_page_id(),
            33,
            "key above last separator routes to last child",
        );
    }

    #[test]
    fn lookup_child_swip_returns_none_for_leaf_pages() {
        let entries = vec![LeafEntry {
            key: b"k10".to_vec(),
            versions: vec![version(1, b"v01")],
        }];
        let mut page = [0u8; PAGE_SIZE];
        encode_leaf_page(&mut page, &Fence::root(), &entries).unwrap();

        assert!(
            lookup_child_swip(&page, b"k10").unwrap().is_none(),
            "leaf pages have no child to route to",
        );
    }

    #[test]
    fn internal_child_array_range_locates_child_slots_for_internal_pages() {
        let children = vec![11, 22, 33];
        let separators = vec![b"k10".to_vec(), b"k20".to_vec()];
        let mut page = [0u8; PAGE_SIZE];
        encode_internal_page(&mut page, &Fence::root(), &children, &separators, &[]).unwrap();

        let (child_count, child_offset) = internal_child_array_range(&page)
            .expect("internal page must expose its child array range");
        assert_eq!(child_count, 3, "child count mismatch");
        // Each child slot is an 8-byte little-endian SWIP word; the first one
        // at child_offset must encode the Evicted page ID 11.
        let raw = u64::from_le_bytes(page[child_offset..child_offset + 8].try_into().unwrap());
        assert_eq!(
            Swip::from_raw(raw).as_page_id(),
            11,
            "first child slot should encode page id 11",
        );
    }

    #[test]
    fn internal_child_array_range_rejects_leaf_pages() {
        let entries = vec![LeafEntry {
            key: b"k10".to_vec(),
            versions: vec![version(1, b"v01")],
        }];
        let mut page = [0u8; PAGE_SIZE];
        encode_leaf_page(&mut page, &Fence::root(), &entries).unwrap();

        assert!(
            internal_child_array_range(&page).is_none(),
            "leaf pages must not expose a child array range",
        );
    }

    #[test]
    fn raw_append_adds_internal_buffer_message_without_reencoding_page() {
        let children = vec![11, 22];
        let separators = vec![b"k10".to_vec()];
        let buffer = vec![BufferedMessage::put(b"k05", 5, b"old")];
        let appended = BufferedMessage::put(b"k15", 9, b"new");

        let mut page = [0u8; PAGE_SIZE];
        let before_body_len =
            encode_internal_page(&mut page, &Fence::root(), &children, &separators, &buffer)
                .unwrap()
                - HEADER_SIZE;
        let append = append_internal_buffer_message(&mut page, &appended, 8, 512, 8)
            .unwrap()
            .expect("below-threshold internal message should append in place");

        assert_eq!(append.buffer_count, 2, "buffer count after append mismatch");
        assert_eq!(
            append.body_len,
            before_body_len + appended.encoded_len(),
            "body length should grow by the encoded message only"
        );
        let NodePage::Internal { buffer, .. } = decode_page(&page).unwrap() else {
            panic!("appended page should remain internal");
        };
        assert_eq!(buffer.len(), 2, "decoded buffer should include append");
        assert_eq!(buffer[1], appended, "appended message mismatch");

        let LookupStep::Internal {
            visible_buffer: Some(visible),
            ..
        } = lookup_step(&page, b"k15", 10).unwrap()
        else {
            panic!("raw lookup should see appended buffer message");
        };
        assert_eq!(visible.value, b"new", "appended buffer value mismatch");
    }

    #[test]
    fn raw_append_declines_when_threshold_would_flush() {
        let children = vec![11, 22];
        let separators = vec![b"k10".to_vec()];
        let buffer = vec![BufferedMessage::put(b"k05", 5, b"old")];

        let mut page = [0u8; PAGE_SIZE];
        encode_internal_page(&mut page, &Fence::root(), &children, &separators, &buffer).unwrap();
        let body_len = read_u32(&page, BODY_LEN_OFF).unwrap();

        let result = append_internal_buffer_message(
            &mut page,
            &BufferedMessage::put(b"k15", 9, b"new"),
            2,
            512,
            8,
        )
        .unwrap();
        assert!(
            result.is_none(),
            "append should fall back when the new message reaches the flush threshold"
        );
        assert_eq!(
            read_u32(&page, BODY_LEN_OFF).unwrap(),
            body_len,
            "declined append must not mutate page body length"
        );
        let NodePage::Internal { buffer, .. } = decode_page(&page).unwrap() else {
            panic!("page should remain internal");
        };
        assert_eq!(buffer.len(), 1, "declined append should not alter buffer");
    }

    #[test]
    fn raw_internal_buffer_append_accepts_append_ordered_messages() {
        let children = vec![11, 22];
        let separators = vec![b"k10".to_vec()];
        let buffer = vec![
            BufferedMessage::put(b"k15", 7, b"other"),
            BufferedMessage::put(b"k05", 5, b"old"),
        ];
        let appended = BufferedMessage::put(b"k05", 8, b"new");

        let mut page = [0u8; PAGE_SIZE];
        encode_internal_page(&mut page, &Fence::root(), &children, &separators, &buffer).unwrap();
        let body_len = read_u32(&page, BODY_LEN_OFF).unwrap();

        let append = append_internal_buffer_message(&mut page, &appended, 8, 512, 8)
            .unwrap()
            .expect("append-order internal buffers should accept unsorted keys");

        assert_eq!(append.buffer_count, 3, "buffer count after append mismatch");
        assert!(
            read_u32(&page, BODY_LEN_OFF).unwrap() > body_len,
            "accepted append should grow the page body"
        );
        let NodePage::Internal { buffer, .. } = decode_page(&page).unwrap() else {
            panic!("appended page should remain internal");
        };
        assert_eq!(
            buffer[2], appended,
            "decoded buffer should preserve append order"
        );

        let LookupStep::Internal {
            visible_buffer: Some(visible),
            ..
        } = lookup_step(&page, b"k05", 10).unwrap()
        else {
            panic!("raw lookup should scan the whole append-ordered buffer");
        };
        assert_eq!(visible.commit_ts, 8, "newest appended version should win");
        assert_eq!(visible.value, b"new", "appended buffer value mismatch");
    }

    #[test]
    fn raw_leaf_append_adds_ordered_entry_without_reencoding_page() {
        let entries = vec![
            LeafEntry {
                key: b"k05".to_vec(),
                versions: vec![version(5, b"old")],
            },
            LeafEntry {
                key: b"k10".to_vec(),
                versions: vec![version(8, b"mid")],
            },
        ];
        let appended = BufferedMessage::put(b"k15", 9, b"new");

        let mut page = [0u8; PAGE_SIZE];
        let before_body_len =
            encode_leaf_page(&mut page, &Fence::root(), &entries).unwrap() - HEADER_SIZE;
        let append = append_leaf_entry_message(&mut page, &appended, 8)
            .unwrap()
            .expect("ordered new leaf entry should append in place");

        assert_eq!(append.entry_count, 3, "entry count after append mismatch");
        assert!(
            append.body_len > before_body_len,
            "body length should grow after leaf append"
        );
        let NodePage::Leaf { entries, .. } = decode_page(&page).unwrap() else {
            panic!("appended page should remain a leaf");
        };
        assert_eq!(entries.len(), 3, "decoded leaf should include append");
        assert_eq!(entries[2].key, b"k15", "appended key mismatch");
        assert_eq!(
            entries[2].versions[0].value, b"new",
            "appended value mismatch"
        );

        let LookupStep::Leaf {
            visible: Some(visible),
        } = lookup_step(&page, b"k15", 10).unwrap()
        else {
            panic!("raw lookup should see appended leaf entry");
        };
        assert_eq!(visible.value, b"new", "appended leaf lookup mismatch");
    }

    #[test]
    fn raw_leaf_batch_append_adds_ordered_entries_without_reencoding_page() {
        let entries = vec![LeafEntry {
            key: b"k10".to_vec(),
            versions: vec![version(8, b"mid")],
        }];
        let appended = vec![
            BufferedMessage::put(b"k15", 9, b"new-15"),
            BufferedMessage::put(b"k20", 10, b"new-20"),
        ];

        let mut page = [0u8; PAGE_SIZE];
        let before_body_len =
            encode_leaf_page(&mut page, &Fence::root(), &entries).unwrap() - HEADER_SIZE;
        let append = append_leaf_entry_batch(&mut page, &appended, 8)
            .unwrap()
            .expect("ordered new leaf entries should append in one batch");

        assert_eq!(append.entry_count, 3, "entry count after batch mismatch");
        assert_eq!(
            append.message_count, 2,
            "batch append should report appended message count"
        );
        assert!(
            append.body_len > before_body_len,
            "body length should grow after batch append"
        );
        let NodePage::Leaf { entries, .. } = decode_page(&page).unwrap() else {
            panic!("batch-appended page should remain a leaf");
        };
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.key.as_slice())
                .collect::<Vec<_>>(),
            vec![&b"k10"[..], &b"k15"[..], &b"k20"[..]],
            "decoded leaf should preserve appended key order"
        );

        let LookupStep::Leaf {
            visible: Some(visible),
        } = lookup_step(&page, b"k20", 10).unwrap()
        else {
            panic!("raw lookup should see batch-appended leaf entry");
        };
        assert_eq!(visible.value, b"new-20", "batch leaf lookup mismatch");
    }

    #[test]
    fn raw_leaf_append_declines_out_of_order_entry() {
        let entries = vec![LeafEntry {
            key: b"k10".to_vec(),
            versions: vec![version(8, b"mid")],
        }];

        let mut page = [0u8; PAGE_SIZE];
        encode_leaf_page(&mut page, &Fence::root(), &entries).unwrap();
        let body_len = read_u32(&page, BODY_LEN_OFF).unwrap();
        let result =
            append_leaf_entry_message(&mut page, &BufferedMessage::put(b"k05", 9, b"old"), 8)
                .unwrap();

        assert!(
            result.is_none(),
            "out-of-order key should fall back to the full leaf path"
        );
        assert_eq!(
            read_u32(&page, BODY_LEN_OFF).unwrap(),
            body_len,
            "declined leaf append must not mutate body length"
        );
        let NodePage::Leaf { entries, .. } = decode_page(&page).unwrap() else {
            panic!("page should remain a leaf");
        };
        assert_eq!(entries.len(), 1, "declined append should not add entry");
    }

    #[test]
    fn raw_leaf_batch_append_declines_unsorted_batch_without_partial_mutation() {
        let entries = vec![LeafEntry {
            key: b"k10".to_vec(),
            versions: vec![version(8, b"mid")],
        }];
        let appended = vec![
            BufferedMessage::put(b"k20", 9, b"new-20"),
            BufferedMessage::put(b"k15", 10, b"new-15"),
        ];

        let mut page = [0u8; PAGE_SIZE];
        encode_leaf_page(&mut page, &Fence::root(), &entries).unwrap();
        let body_len = read_u32(&page, BODY_LEN_OFF).unwrap();
        let result = append_leaf_entry_batch(&mut page, &appended, 8).unwrap();

        assert!(
            result.is_none(),
            "unsorted batch should fall back before mutating the leaf"
        );
        assert_eq!(
            read_u32(&page, BODY_LEN_OFF).unwrap(),
            body_len,
            "declined batch append must not mutate body length"
        );
        let NodePage::Leaf { entries, .. } = decode_page(&page).unwrap() else {
            panic!("page should remain a leaf");
        };
        assert_eq!(
            entries.len(),
            1,
            "declined batch append should not add any entries"
        );
    }

    #[test]
    fn raw_leaf_prefix_append_stops_before_capacity_without_reencoding_page() {
        let entries = vec![LeafEntry {
            key: b"k10".to_vec(),
            versions: vec![version(8, b"mid")],
        }];
        let appended = vec![
            BufferedMessage::put(b"k15", 9, b"new-15"),
            BufferedMessage::put(b"k20", 10, b"new-20"),
        ];

        let mut page = [0u8; PAGE_SIZE];
        encode_leaf_page(&mut page, &Fence::root(), &entries).unwrap();
        let append = append_leaf_entry_prefix(&mut page, &appended, 2)
            .unwrap()
            .expect("first ordered entry should fit before capacity");

        assert_eq!(append.message_count, 1, "prefix length mismatch");
        assert_eq!(append.entry_count, 2, "entry count after prefix mismatch");
        let NodePage::Leaf { entries, .. } = decode_page(&page).unwrap() else {
            panic!("prefix-appended page should remain a leaf");
        };
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.key.as_slice())
                .collect::<Vec<_>>(),
            vec![&b"k10"[..], &b"k15"[..]],
            "prefix append should stop before the full batch"
        );
    }

    #[test]
    fn sorted_validation_rejects_bad_leaf_order() {
        let mut page = [0u8; PAGE_SIZE];
        let entries = vec![
            LeafEntry {
                key: b"b".to_vec(),
                versions: vec![version(1, b"b")],
            },
            LeafEntry {
                key: b"a".to_vec(),
                versions: vec![version(1, b"a")],
            },
        ];
        encode_leaf_page(&mut page, &Fence::root(), &entries).unwrap();
        assert!(
            decode_page(&page).is_err(),
            "decode should reject unsorted leaf keys"
        );
    }

    #[test]
    fn leaf_page_reader_reads_entries_and_keys_zero_copy() {
        let entries: Vec<LeafEntry> = (0..5)
            .map(|i| LeafEntry {
                key: format!("key{i:02}").into_bytes(),
                versions: vec![version(i as Timestamp, format!("val{i:02}").as_bytes())],
            })
            .collect();
        let fence = Fence {
            lower: b"a".to_vec(),
            upper: Some(b"z".to_vec()),
        };

        let mut page = [0u8; PAGE_SIZE];
        encode_leaf_page(&mut page, &fence, &entries).unwrap();

        let reader = LeafPageReader::new(&page).unwrap();
        assert_eq!(reader.entry_count(), 5, "entry count mismatch");
        for (idx, entry) in entries.iter().enumerate() {
            assert_eq!(
                reader.entry_key(idx).unwrap(),
                entry.key.as_slice(),
                "entry key mismatch at index {idx}"
            );
        }
        let read_fence = reader.fence().unwrap();
        assert_eq!(read_fence, fence, "fence mismatch");
    }

    #[test]
    fn leaf_page_reader_entry_raw_preserves_multi_version_entries() {
        let entries = vec![
            LeafEntry {
                key: b"k01".to_vec(),
                versions: vec![version(10, b"new"), version(1, b"old")],
            },
            LeafEntry {
                key: b"k02".to_vec(),
                versions: vec![version(2, b"v02")],
            },
            LeafEntry {
                key: b"k03".to_vec(),
                versions: vec![
                    VersionRecord {
                        commit_ts: 7,
                        value: Vec::new(),
                        deleted: true,
                    },
                    version(3, b"v03"),
                ],
            },
        ];

        let mut page = [0u8; PAGE_SIZE];
        encode_leaf_page(&mut page, &Fence::root(), &entries).unwrap();

        let reader = LeafPageReader::new(&page).unwrap();
        for (idx, entry) in entries.iter().enumerate() {
            let raw = reader.entry_raw(idx).unwrap();
            let mut entry_reader = BodyReader::at(raw, 0).unwrap();
            let key = entry_reader.read_bytes_u16_len().unwrap();
            let version_count = entry_reader.read_u16().unwrap() as usize;
            assert_eq!(key, entry.key.as_slice(), "raw entry key mismatch");
            assert_eq!(
                version_count,
                entry.versions.len(),
                "raw entry version count mismatch"
            );
            for version in &entry.versions {
                let decoded = decode_version(&mut entry_reader).unwrap();
                assert_eq!(decoded, *version, "version mismatch");
            }
        }
    }

    #[test]
    fn split_leaf_into_pages_distributes_entries_and_preserves_versions() {
        let entries: Vec<LeafEntry> = (0..10)
            .map(|i| LeafEntry {
                key: format!("key{i:02}").into_bytes(),
                versions: vec![
                    version((i + 1) as Timestamp, format!("new{i:02}").as_bytes()),
                    version(1, format!("old{i:02}").as_bytes()),
                ],
            })
            .collect();
        let fence = Fence {
            lower: b"a".to_vec(),
            upper: Some(b"zzz".to_vec()),
        };

        let mut src = [0u8; PAGE_SIZE];
        encode_leaf_page(&mut src, &fence, &entries).unwrap();

        let mid = 4;
        let separator = entries[mid].key.clone();
        let left_fence = fence.left_of(separator.clone());
        let right_fence = fence.right_of(separator.clone());

        let mut dst_left = [0u8; PAGE_SIZE];
        let mut dst_right = [0u8; PAGE_SIZE];
        let result = split_leaf_into_pages(
            &src,
            &mut dst_left,
            &mut dst_right,
            &left_fence,
            &right_fence,
            mid,
        )
        .unwrap();

        assert_eq!(result.separator, separator, "separator mismatch");
        assert_eq!(result.left_count, mid, "left count mismatch");
        assert_eq!(
            result.right_count,
            entries.len() - mid,
            "right count mismatch"
        );

        let NodePage::Leaf {
            fence: lf,
            entries: le,
        } = decode_page(&dst_left).unwrap()
        else {
            panic!("left page should be a leaf");
        };
        assert_eq!(lf, left_fence, "left fence mismatch");
        assert_eq!(
            le,
            entries[..mid],
            "left entries should match the first half"
        );

        let NodePage::Leaf {
            fence: rf,
            entries: re,
        } = decode_page(&dst_right).unwrap()
        else {
            panic!("right page should be a leaf");
        };
        assert_eq!(rf, right_fence, "right fence mismatch");
        assert_eq!(
            re,
            entries[mid..],
            "right entries should match the second half"
        );
    }

    #[test]
    fn split_leaf_into_pages_rejects_invalid_split_point() {
        let entries = vec![
            LeafEntry {
                key: b"k01".to_vec(),
                versions: vec![version(1, b"v01")],
            },
            LeafEntry {
                key: b"k02".to_vec(),
                versions: vec![version(2, b"v02")],
            },
        ];
        let mut src = [0u8; PAGE_SIZE];
        encode_leaf_page(&mut src, &Fence::root(), &entries).unwrap();

        let mut dst_left = [0u8; PAGE_SIZE];
        let mut dst_right = [0u8; PAGE_SIZE];
        let fence = Fence::root();

        assert!(
            split_leaf_into_pages(&src, &mut dst_left, &mut dst_right, &fence, &fence, 0).is_err(),
            "mid=0 should be rejected"
        );
        assert!(
            split_leaf_into_pages(&src, &mut dst_left, &mut dst_right, &fence, &fence, 2).is_err(),
            "mid=entry_count should be rejected"
        );
    }
}
