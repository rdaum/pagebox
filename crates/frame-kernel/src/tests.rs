use crate::{
    MAX_PAGE_CLASS_TAG, PAGE_SIZE, PageClass, decode_page_id, page_base_span, page_end_base_page,
    page_size, page_slot_index, physical_page_number,
};

/// Shift used for the page-number portion of a PageId (top 8 bits hold the class tag).
const PAGE_CLASS_SHIFT: u64 = 56;
const PAGE_NUMBER_MASK: u64 = (1u64 << PAGE_CLASS_SHIFT) - 1;

#[test]
fn page_id_roundtrip_preserves_class_and_slot() {
    let pid = PageClass::Size4K.encode_page_id(17);
    assert_eq!(decode_page_id(pid), Some((PageClass::Size4K, 17)));
    assert_eq!(page_slot_index(pid), Some((PageClass::Size4K, 16)));
    assert_eq!(physical_page_number(pid), 17);
}

#[test]
fn page_class_from_tag_accepts_power_of_two_classes() {
    for (tag, class) in PageClass::ALL.iter().copied().enumerate() {
        assert_eq!(PageClass::from_tag(tag as u8), Some(class));
        assert_eq!(class.page_size(), PAGE_SIZE << tag);
        assert_eq!(class.base_page_count(), 1usize << tag);
    }
    assert_eq!(PageClass::from_tag(MAX_PAGE_CLASS_TAG + 1), None);
    assert_eq!(PageClass::from_tag(u8::MAX), None);
}

#[test]
fn page_id_tracks_class_size_and_base_extent() {
    let pid = PageClass::Size1M.encode_page_id(257);
    assert_eq!(decode_page_id(pid), Some((PageClass::Size1M, 257)));
    assert_eq!(page_size(pid), 1024 * 1024);
    assert_eq!(page_base_span(pid), 256);
    assert_eq!(page_end_base_page(pid), 512);
}

#[test]
fn all_page_classes_roundtrip_through_encode_decode() {
    for class in PageClass::ALL {
        let page_number = 42u64;
        let pid = class.encode_page_id(page_number);
        assert_eq!(
            decode_page_id(pid),
            Some((class, page_number)),
            "{class:?}: encode/decode roundtrip failed"
        );
        assert_eq!(
            page_slot_index(pid),
            Some((class, page_number as usize - 1)),
            "{class:?}: slot index should be page_number - 1"
        );
        assert_eq!(
            page_size(pid),
            class.page_size(),
            "{class:?}: page_size mismatch"
        );
        assert_eq!(
            page_base_span(pid),
            class.base_page_count(),
            "{class:?}: base_page_count mismatch"
        );
    }
}

#[test]
fn page_slot_index_at_slot_zero_boundary() {
    let pid = PageClass::Size4K.encode_page_id(1);
    assert_eq!(page_slot_index(pid), Some((PageClass::Size4K, 0)));
}

#[test]
fn encode_page_id_at_max_page_number() {
    let pid = PageClass::Size4K.encode_page_id(PAGE_NUMBER_MASK);
    assert_eq!(
        decode_page_id(pid),
        Some((PageClass::Size4K, PAGE_NUMBER_MASK))
    );
    assert_eq!(physical_page_number(pid), PAGE_NUMBER_MASK);
}

#[test]
fn decode_page_id_rejects_zero() {
    assert_eq!(decode_page_id(0), None);
}

#[test]
fn decode_page_id_rejects_bogus_tag() {
    // A tag beyond MAX_PAGE_CLASS_TAG should return None.
    let bogus = ((MAX_PAGE_CLASS_TAG as u64 + 1) << PAGE_CLASS_SHIFT) | 1;
    assert_eq!(decode_page_id(bogus), None);
}

#[test]
#[should_panic(expected = "page 0 is reserved")]
fn encode_page_id_zero_panics() {
    let _ = PageClass::Size4K.encode_page_id(0);
}

#[test]
#[should_panic(expected = "exceeds encodable range")]
fn encode_page_id_overflow_panics() {
    let _ = PageClass::Size4K.encode_page_id(PAGE_NUMBER_MASK + 1);
}
