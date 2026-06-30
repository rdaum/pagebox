use crate::page_id::{
    PAGE_SIZE, PageId, decode_page_id, page_base_span, page_end_base_page, page_size,
    physical_page_number,
};

#[test]
fn page_id_basic_properties() {
    let pid: PageId = 42;
    assert_eq!(decode_page_id(pid), Some(((), 42)));
    assert_eq!(physical_page_number(pid), 42);
    assert_eq!(page_size(pid), PAGE_SIZE);
    assert_eq!(page_base_span(pid), 1);
    assert_eq!(page_end_base_page(pid), 42);
}

#[test]
fn decode_page_id_rejects_zero() {
    assert_eq!(decode_page_id(0), None);
}

#[test]
fn all_pages_are_same_size() {
    for pid in 1..1000 {
        assert_eq!(page_size(pid), PAGE_SIZE);
        assert_eq!(page_base_span(pid), 1);
        assert_eq!(page_end_base_page(pid), pid);
    }
}
