/// Configurable page size. Change this to adjust the B+tree data page size.
/// Must be a power of two and at least 4096 (the hardware page size).
pub const PAGE_SIZE: usize = 64 * 1024;

pub type PageId = u64;
pub type Lsn = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InnerParentLink {
    pub parent_pid: PageId,
    pub slot_index: u16,
    pub is_upper: bool,
    pub dt_id: u16,
}

/// A page ID is just a positive u64 — no class-tag bits.
/// Page 0 is reserved for the page-store header.
///
/// Validate that `pid` refers to a real page (nonzero).
#[allow(dead_code)]
pub fn decode_page_id(pid: PageId) -> Option<((), u64)> {
    if pid == 0 {
        return None;
    }
    Some(((), pid))
}

/// The physical page number is the PID itself (no class remapping).
pub fn physical_page_number(pid: PageId) -> u64 {
    pid
}

/// The decoded page number is the PID itself.
pub fn decoded_page_number(pid: PageId) -> u64 {
    assert!(pid > 0, "invalid page id {pid}");
    pid
}

/// One page = one base page (no multi-base-page classes).
pub fn page_base_span(_pid: PageId) -> usize {
    1
}

/// All pages are PAGE_SIZE bytes.
pub fn page_size(_pid: PageId) -> usize {
    PAGE_SIZE
}

/// The last base page covered by this PID.
pub fn page_end_base_page(pid: PageId) -> u64 {
    pid
}
