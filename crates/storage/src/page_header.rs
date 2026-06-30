//! Common bytes shared by all buffer-managed pages.
//!
//! The buffer manager treats page bodies as opaque, but it does need a
//! small common prefix for recovery LSNs, coarse page classification, and
//! residency policy. Higher-level page formats own everything past this
//! prefix.

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageType {
    Unknown = 0,
    Index = 1,
    Tuple = 2,
    Delta = 3,
    Meta = 4,
    RootMeta = 5,
    BeTreeInternal = 6,
    BeTreeLeaf = 7,
}

impl PageType {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => PageType::Index,
            2 => PageType::Tuple,
            3 => PageType::Delta,
            4 => PageType::Meta,
            5 => PageType::RootMeta,
            6 => PageType::BeTreeInternal,
            7 => PageType::BeTreeLeaf,
            _ => PageType::Unknown,
        }
    }
}

const FLAGS_OFF: usize = 18;
const PAGE_LSN_OFF: usize = 0;
const PAGE_LSN_LEN: usize = 8;
const PAGE_TYPE_SHIFT: u32 = 8;
const PAGE_TYPE_MASK: u16 = 0x0F00;
const INDEX_LEAF_FLAG: u16 = 0x0002;

pub fn read_page_lsn(page: &[u8]) -> u64 {
    let Some(bytes) = page.get(PAGE_LSN_OFF..PAGE_LSN_OFF + PAGE_LSN_LEN) else {
        return 0;
    };
    u64::from_ne_bytes(bytes.try_into().unwrap())
}

pub fn write_page_lsn(page: &mut [u8], lsn: u64) {
    let Some(bytes) = page.get_mut(PAGE_LSN_OFF..PAGE_LSN_OFF + PAGE_LSN_LEN) else {
        return;
    };
    bytes.copy_from_slice(&lsn.to_ne_bytes());
}

pub fn read_page_type(page: &[u8]) -> PageType {
    let Some(bytes) = page.get(FLAGS_OFF..FLAGS_OFF + 2) else {
        return PageType::Unknown;
    };
    let flags = u16::from_ne_bytes(bytes.try_into().unwrap());
    PageType::from_u8(((flags & PAGE_TYPE_MASK) >> PAGE_TYPE_SHIFT) as u8)
}

pub fn write_page_type(page: &mut [u8], pt: PageType) {
    let Some(bytes) = page.get_mut(FLAGS_OFF..FLAGS_OFF + 2) else {
        return;
    };
    let mut flags = u16::from_ne_bytes(bytes.try_into().unwrap());
    flags = (flags & !PAGE_TYPE_MASK) | ((pt as u16) << PAGE_TYPE_SHIFT);
    bytes.copy_from_slice(&flags.to_ne_bytes());
}

pub fn is_inner_index_page(page: &[u8]) -> bool {
    let Some(bytes) = page.get(FLAGS_OFF..FLAGS_OFF + 2) else {
        return false;
    };
    let flags = u16::from_ne_bytes(bytes.try_into().unwrap());
    let pt = ((flags & PAGE_TYPE_MASK) >> PAGE_TYPE_SHIFT) as u8;
    pt == PageType::Index as u8 && (flags & INDEX_LEAF_FLAG) == 0
}

pub fn should_remain_resident(page: &[u8]) -> bool {
    matches!(read_page_type(page), PageType::Meta | PageType::RootMeta)
}

pub fn classify_loaded_page(page: &[u8]) -> PageType {
    read_page_type(page)
}
