mod frame;
mod page_id;
#[cfg(test)]
mod tests;

pub use frame::{AtomicFrameState, FrameCoreHeader, FrameState, PaddedAtomicU32};
pub use page_id::{
    InnerParentLink, Lsn, MAX_PAGE_CLASS_TAG, PAGE_CLASS_BITS, PAGE_SIZE, PageClass, PageId,
    decode_page_id, decoded_page_number, page_base_span, page_end_base_page, page_size,
    page_slot_index, physical_page_number,
};
