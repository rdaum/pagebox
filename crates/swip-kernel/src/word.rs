use crate::state::{
    COOL_BIT, EVICTED_BIT, PTR_MASK, STATE_MASK, SwipState, classify_raw, raw_is_cool,
    raw_is_evicted, raw_is_hot,
};

/// ```anneal
/// ensures: ret.val = ptr.val
/// proof (h_anon):
///   unfold make_hot_ptr_word at h_returns
///   cases h_returns
///   rfl
/// ```
pub(crate) fn make_hot_ptr_word(ptr: u64) -> u64 {
    ptr
}

pub(crate) fn make_evicted_page_word(page_id: u64) -> u64 {
    page_id | EVICTED_BIT
}

pub(crate) fn pointer_bits_of(raw: u64) -> u64 {
    raw & PTR_MASK
}

pub(crate) fn page_id_of(raw: u64) -> u64 {
    raw & !EVICTED_BIT
}

pub(crate) fn cool_word(raw: u64) -> u64 {
    raw | COOL_BIT
}

pub(crate) fn warm_word(raw: u64) -> u64 {
    raw & !COOL_BIT
}

pub(crate) fn evict_word(raw: u64, page_id: u64) -> u64 {
    let _ = raw;
    page_id | EVICTED_BIT
}

/// ```anneal
/// ensures: ret.val = ptr.val
/// proof (h_anon):
///   unfold resolve_ptr_word at h_returns
///   cases h_returns
///   rfl
/// ```
pub(crate) fn resolve_ptr_word(raw: u64, ptr: u64) -> u64 {
    let _ = raw;
    ptr
}

/// Raw tagged swip word.
///
/// The top two bits encode state:
/// - `00`: hot pointer
/// - `01`: cool pointer
/// - `1x`: evicted page id
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct SwipWord(pub(crate) u64);

impl SwipWord {
    pub fn hot<T>(ptr: *mut T) -> Self {
        Self::hot_ptr(ptr as u64)
    }

    pub fn hot_ptr(ptr: u64) -> Self {
        debug_assert!(ptr & STATE_MASK == 0, "pointer uses tag bits");
        Self(make_hot_ptr_word(ptr))
    }

    pub fn evicted(page_id: u64) -> Self {
        Self::evicted_page(page_id)
    }

    pub fn evicted_page(page_id: u64) -> Self {
        debug_assert!(page_id & STATE_MASK == 0, "page ID uses tag bits");
        Self(make_evicted_page_word(page_id))
    }

    pub fn state(self) -> SwipState {
        classify_raw(self.0)
    }

    pub fn is_hot(self) -> bool {
        raw_is_hot(self.0)
    }

    pub fn is_cool(self) -> bool {
        raw_is_cool(self.0)
    }

    pub fn is_evicted(self) -> bool {
        raw_is_evicted(self.0)
    }

    pub fn pointer_bits(self) -> u64 {
        debug_assert!(
            !self.is_evicted(),
            "cannot read pointer bits from evicted swip"
        );
        pointer_bits_of(self.0)
    }

    /// # Safety
    ///
    /// Caller must ensure the swip is hot or cool, and that the pointed-to
    /// value is still live.
    pub unsafe fn as_ptr<T>(self) -> *mut T {
        self.pointer_bits() as *mut T
    }

    pub fn as_page_id(self) -> u64 {
        self.page_id()
    }

    pub fn page_id(self) -> u64 {
        debug_assert!(self.is_evicted(), "not evicted");
        page_id_of(self.0)
    }

    pub fn cool(&mut self) {
        debug_assert!(self.is_hot(), "can only cool a hot swip");
        self.0 = cool_word(self.0);
    }

    pub fn warm(&mut self) {
        debug_assert!(self.is_cool(), "can only warm a cool swip");
        self.0 = warm_word(self.0);
    }

    pub fn evict(&mut self, page_id: u64) {
        debug_assert!(self.is_cool(), "can only evict a cool swip");
        debug_assert!(page_id & STATE_MASK == 0, "page ID uses tag bits");
        self.0 = evict_word(self.0, page_id);
    }

    pub fn resolve_ptr(&mut self, ptr: u64) {
        debug_assert!(self.is_evicted(), "can only resolve an evicted swip");
        debug_assert!(ptr & STATE_MASK == 0, "pointer uses tag bits");
        self.0 = resolve_ptr_word(self.0, ptr);
    }

    pub fn resolve<T>(&mut self, ptr: *mut T) {
        self.resolve_ptr(ptr as u64);
    }

    pub fn raw(self) -> u64 {
        self.0
    }

    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
}

impl std::fmt::Debug for SwipWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.state() {
            SwipState::Hot => write!(f, "SwipWord::Hot({:#x})", self.raw()),
            SwipState::Cool => write!(f, "SwipWord::Cool({:#x})", self.pointer_bits()),
            SwipState::Evicted => write!(f, "SwipWord::Evicted({})", self.page_id()),
        }
    }
}
