pub(crate) const EVICTED_BIT: u64 = 1 << 63;
pub(crate) const COOL_BIT: u64 = 1 << 62;
pub(crate) const STATE_MASK: u64 = 0x3 << 62;
pub(crate) const PTR_MASK: u64 = !STATE_MASK;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwipState {
    Hot,
    Cool,
    Evicted,
}

/// ```anneal
/// ensures: ret = (mask != 0#u64)
/// proof (h_anon):
///   unfold evicted_mask_is_set at h_returns
///   simp_all
/// ```
pub(crate) fn evicted_mask_is_set(mask: u64) -> bool {
    mask != 0
}

pub(crate) fn evicted_mask(raw: u64) -> u64 {
    raw & EVICTED_BIT
}

pub(crate) fn raw_is_evicted(raw: u64) -> bool {
    evicted_mask_is_set(evicted_mask(raw))
}

pub(crate) fn state_mask(raw: u64) -> u64 {
    raw & STATE_MASK
}

pub(crate) fn is_cool_mask(mask: u64) -> bool {
    mask == COOL_BIT
}

pub(crate) fn raw_is_cool(raw: u64) -> bool {
    !raw_is_evicted(raw) && is_cool_mask(state_mask(raw))
}

pub(crate) fn raw_is_hot(raw: u64) -> bool {
    !raw_is_evicted(raw) && !raw_is_cool(raw)
}

pub(crate) fn classify_raw(raw: u64) -> SwipState {
    if raw_is_evicted(raw) {
        SwipState::Evicted
    } else if raw_is_cool(raw) {
        SwipState::Cool
    } else {
        SwipState::Hot
    }
}
