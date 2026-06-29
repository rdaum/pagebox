/// ```anneal
/// ensures:
///   match edge with
///   | .Upper => ret = true
///   | .Slot _ => ret = false
/// proof (h_anon):
///   unfold parent_edge_is_upper at h_returns
///   cases edge <;> cases h_returns <;> simp_all
/// proof (h_progress):
///   unfold parent_edge_is_upper
///   cases edge <;> simp_all
/// ```
pub(crate) fn parent_edge_is_upper(edge: ParentEdge) -> bool {
    matches!(edge, ParentEdge::Upper)
}

/// ```anneal
/// ensures:
///   match edge with
///   | .Upper => ret = count
///   | .Slot pos => ret = pos
/// proof (h_anon):
///   unfold parent_edge_pos at h_returns
///   cases edge <;> cases h_returns <;> simp_all
/// proof (h_progress):
///   unfold parent_edge_pos
///   cases edge <;> simp_all
/// ```
pub(crate) fn parent_edge_pos(edge: ParentEdge, count: u16) -> u16 {
    match edge {
        ParentEdge::Slot(pos) => pos,
        ParentEdge::Upper => count,
    }
}

/// ```anneal
/// ensures:
///   match ret with
///   | .Upper => pos = count
///   | .Slot slot => slot = pos ∧ pos ≠ count
/// proof (h_anon):
///   unfold parent_edge_from_pos at h_returns
///   by_cases h_eq : pos = count
///   · simp [h_eq] at h_returns
///     cases h_returns
///     simp [h_eq]
///   · simp [h_eq] at h_returns
///     cases h_returns
///     simp [h_eq]
/// proof (h_progress):
///   unfold parent_edge_from_pos
///   by_cases h_eq : pos = count <;> simp [h_eq]
/// ```
pub(crate) fn parent_edge_from_pos(pos: u16, count: u16) -> ParentEdge {
    if pos == count {
        ParentEdge::Upper
    } else {
        ParentEdge::Slot(pos)
    }
}

/// ```anneal
/// ensures: ret = pos
/// proof (h_anon):
///   unfold parent_edge_round_trip_pos at h_returns
///   unfold parent_edge_pos parent_edge_from_pos at h_returns
///   by_cases h_eq : pos = count
///   · simp [h_eq] at h_returns
///     cases h_returns
///     exact h_eq.symm
///   · simp [h_eq] at h_returns
///     cases h_returns
///     rfl
/// proof (h_progress):
///   unfold parent_edge_round_trip_pos
///   unfold parent_edge_pos parent_edge_from_pos
///   by_cases h_eq : pos = count
///   · simp [h_eq]
///   · simp [h_eq]
/// ```
pub(crate) fn parent_edge_round_trip_pos(pos: u16, count: u16) -> u16 {
    parent_edge_pos(parent_edge_from_pos(pos, count), count)
}

#[derive(Clone, Copy)]
pub(crate) enum ParentEdge {
    Slot(u16),
    Upper,
}

impl ParentEdge {
    pub(crate) fn from_pos(pos: u16, count: u16) -> Self {
        parent_edge_from_pos(pos, count)
    }

    pub(crate) fn pos(self, count: u16) -> u16 {
        parent_edge_pos(self, count)
    }

    pub(crate) fn is_upper(self) -> bool {
        parent_edge_is_upper(self)
    }
}
