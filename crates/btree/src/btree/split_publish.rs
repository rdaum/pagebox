/// ```anneal
/// ensures: ret = (pos >= count)
/// proof (h_anon):
///   unfold split_right_parent_uses_upper at h_returns
///   cases ret <;> simp_all
/// proof (h_progress):
///   unfold split_right_parent_uses_upper
///   refine ⟨(pos >= count), ?_⟩
///   simp
/// ```
pub(crate) fn split_right_parent_uses_upper(pos: u16, count: u16) -> bool {
    pos >= count
}

/// ```anneal
/// ensures: ret = (if pos < count then pos else count)
/// proof (h_anon):
///   unfold split_separator_insert_pos at h_returns
///   by_cases h_lt : pos < count
///   · simp [h_lt] at h_returns
///     cases h_returns
///     simp [h_lt]
///   · simp [h_lt] at h_returns
///     cases h_returns
///     simp [h_lt]
/// proof (h_progress):
///   unfold split_separator_insert_pos
///   by_cases h_lt : pos < count <;> simp [h_lt]
/// ```
pub(crate) fn split_separator_insert_pos(pos: u16, count: u16) -> u16 {
    if pos < count { pos } else { count }
}
