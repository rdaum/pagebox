pub const PAGE_SIZE: usize = 4096;
pub const PAGE_CLASS_BITS: u64 = 6;
const PAGE_CLASS_SHIFT: u64 = 56;
const PAGE_NUMBER_MASK: u64 = (1u64 << PAGE_CLASS_SHIFT) - 1;
pub const MAX_PAGE_CLASS_TAG: u8 = (1u8 << PAGE_CLASS_BITS) - 1;

pub type PageId = u64;
pub type Lsn = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InnerParentLink {
    pub parent_pid: PageId,
    pub slot_index: u16,
    pub is_upper: bool,
    pub dt_id: u16,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageClass {
    Size4K = 0,
    Size8K = 1,
    Size16K = 2,
    Size32K = 3,
    Size64K = 4,
    Size128K = 5,
    Size256K = 6,
    Size512K = 7,
    Size1M = 8,
}

impl PageClass {
    pub const ALL: [PageClass; 9] = [
        PageClass::Size4K,
        PageClass::Size8K,
        PageClass::Size16K,
        PageClass::Size32K,
        PageClass::Size64K,
        PageClass::Size128K,
        PageClass::Size256K,
        PageClass::Size512K,
        PageClass::Size1M,
    ];

    pub const fn page_size(self) -> usize {
        PAGE_SIZE << self.tag()
    }

    pub const fn base_page_count(self) -> usize {
        1usize << self.tag()
    }

    pub const fn tag(self) -> u8 {
        self as u8
    }

    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Size4K),
            1 => Some(Self::Size8K),
            2 => Some(Self::Size16K),
            3 => Some(Self::Size32K),
            4 => Some(Self::Size64K),
            5 => Some(Self::Size128K),
            6 => Some(Self::Size256K),
            7 => Some(Self::Size512K),
            8 => Some(Self::Size1M),
            _ => None,
        }
    }

    pub fn for_size(size: usize) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|class| class.page_size() == size)
    }

    pub fn smallest_for_size(size: usize) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|class| class.page_size() >= size)
    }

    pub fn encode_page_id(self, page_number: u64) -> PageId {
        assert!(page_number > 0, "page 0 is reserved");
        assert!(
            page_number <= PAGE_NUMBER_MASK,
            "page number {page_number} exceeds encodable range"
        );
        ((self.tag() as u64) << PAGE_CLASS_SHIFT) | page_number
    }
}

pub fn decode_page_id(pid: PageId) -> Option<(PageClass, u64)> {
    if pid == 0 {
        return None;
    }
    let tag = (pid >> PAGE_CLASS_SHIFT) as u8;
    let page_number = pid & PAGE_NUMBER_MASK;
    if page_number == 0 {
        return None;
    }
    let class = PageClass::from_tag(tag)?;
    Some((class, page_number))
}

pub fn page_slot_index(pid: PageId) -> Option<(PageClass, usize)> {
    let (class, page_number) = decode_page_id(pid)?;
    let slot_idx = usize::try_from(page_number - 1).ok()?;
    Some((class, slot_idx))
}

pub fn physical_page_number(pid: PageId) -> u64 {
    decoded_page_number(pid)
}

pub fn decoded_page_number(pid: PageId) -> u64 {
    decode_page_id(pid)
        .map(|(_, page_number)| page_number)
        .expect("invalid encoded page id")
}

pub fn page_base_span(pid: PageId) -> usize {
    decode_page_id(pid)
        .map(|(class, _)| class.base_page_count())
        .expect("invalid encoded page id")
}

pub fn page_size(pid: PageId) -> usize {
    decode_page_id(pid)
        .map(|(class, _)| class.page_size())
        .expect("invalid encoded page id")
}

pub fn page_end_base_page(pid: PageId) -> u64 {
    let (class, page_number) = decode_page_id(pid).expect("invalid encoded page id");
    page_number + class.base_page_count() as u64 - 1
}
