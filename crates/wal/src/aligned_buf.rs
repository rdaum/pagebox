use crate::format::DIRECT_IO_ALIGN;

pub(crate) struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    pub(crate) fn new(len: usize) -> Self {
        let layout =
            std::alloc::Layout::from_size_align(len, DIRECT_IO_ALIGN).expect("invalid layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "aligned allocation failed");
        Self { ptr, len }
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout =
            std::alloc::Layout::from_size_align(self.len, DIRECT_IO_ALIGN).expect("invalid layout");
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}
