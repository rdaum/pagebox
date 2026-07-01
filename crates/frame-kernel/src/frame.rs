use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};

use crate::page_id::PageId;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameState {
    Free = 0,
    Loading = 1,
    Resident = 2,
    Evicting = 3,
}

/// ```anneal
/// ensures: ret = true ↔ (v = 0#u8 ∨ v = 1#u8 ∨ v = 2#u8 ∨ v = 3#u8)
/// proof (h_anon):
///   unfold frame.is_valid_frame_state_tag at h_returns
///   split at h_returns <;> simp_all
///   split at h_returns <;> simp_all
///   split at h_returns <;> simp_all
///   all_goals (constructor <;> intro h <;> simp_all)
/// proof (h_progress):
///   unfold frame.is_valid_frame_state_tag
///   by_cases h0 : v = 0#u8
///   · refine ⟨true, ?_⟩; simp [h0]
///   · by_cases h1 : v = 1#u8
///     · refine ⟨true, ?_⟩; simp [h1]
///     · by_cases h2 : v = 2#u8
///       · refine ⟨true, ?_⟩; simp [h2]
///       · exact ⟨(v = 3#u8), by simp [h0, h1, h2]⟩
/// ```
pub fn is_valid_frame_state_tag(v: u8) -> bool {
    v == 0 || v == 1 || v == 2 || v == 3
}

impl FrameState {
    pub fn from_u8(v: u8) -> Self {
        assert!(is_valid_frame_state_tag(v), "invalid FrameState: {v}");
        match v {
            0 => FrameState::Free,
            1 => FrameState::Loading,
            2 => FrameState::Resident,
            3 => FrameState::Evicting,
            _ => panic!("invalid FrameState: {v}"),
        }
    }

    /// Returns `true` if `from → to` is a valid state-machine transition.
    ///
    /// Valid transitions:
    /// ```text
    /// Free ──→ Loading ──→ Resident ──→ Evicting ──→ Free
    ///                                   │
    ///                                   └──→ Resident (revert)
    /// ```
    ///
    /// Any transition not listed here is illegal. This is used by
    /// [`AtomicFrameState::transition`] and
    /// [`AtomicFrameState::compare_exchange_transition`] to catch
    /// programmer errors.
    pub fn is_valid_transition(from: FrameState, to: FrameState) -> bool {
        use FrameState::*;
        matches!(
            (from, to),
            (Free, Loading)
                | (Free, Resident) // fresh allocation (allocate_and_fix zeroes page in place)
                | (Loading, Resident)
                | (Loading, Free) // loading aborted (page was already loaded by another thread)
                | (Resident, Evicting)
                | (Evicting, Free)
                | (Evicting, Resident) // revert: pin sneaked in or unswizzle failed
        )
    }
}

pub struct AtomicFrameState(AtomicU8);

impl AtomicFrameState {
    pub fn new(state: FrameState) -> Self {
        Self(AtomicU8::new(state as u8))
    }

    pub fn load(&self, order: Ordering) -> FrameState {
        FrameState::from_u8(self.0.load(order))
    }

    /// Store a new state without transition validation.
    ///
    /// Prefer [`AtomicFrameState::transition`] or
    /// [`AtomicFrameState::compare_exchange_transition`] for state changes
    /// — those methods validate the transition is legal in debug builds.
    /// This method exists for initialization (constructing a fresh frame)
    /// and for the few cases where the caller has already validated the
    /// transition externally.
    #[track_caller]
    pub fn store(&self, state: FrameState, order: Ordering) {
        self.0.store(state as u8, order);
    }

    /// Transition the frame state from `from` to `to`.
    ///
    /// In debug builds, panics if the transition is not in the valid
    /// state-machine graph (see [`FrameState::is_valid_transition`]).
    /// In release builds, the check is compiled out.
    ///
    /// Unlike [`compare_exchange_transition`](Self::compare_exchange_transition),
    /// this does NOT atomically verify the current state matches `from` —
    /// the caller must have already verified the state (e.g., under an
    /// exclusive latch, or after a successful CAS on a related field).
    #[track_caller]
    pub fn transition(&self, from: FrameState, to: FrameState, order: Ordering) {
        debug_assert!(
            FrameState::is_valid_transition(from, to),
            "invalid frame state transition: {from:?} → {to:?}"
        );
        self.0.store(to as u8, order);
    }

    /// Compare-and-swap a state transition.
    ///
    /// Like [`AtomicFrameState::compare_exchange`] but validates the
    /// transition is legal in debug builds. Returns the previous state
    /// on success (the `from` value) or the actual current state on failure.
    #[track_caller]
    pub fn compare_exchange_transition(
        &self,
        from: FrameState,
        to: FrameState,
        success: Ordering,
        failure: Ordering,
    ) -> Result<FrameState, FrameState> {
        debug_assert!(
            FrameState::is_valid_transition(from, to),
            "invalid frame state transition: {from:?} → {to:?}"
        );
        self.0
            .compare_exchange(from as u8, to as u8, success, failure)
            .map(FrameState::from_u8)
            .map_err(FrameState::from_u8)
    }

    pub fn compare_exchange(
        &self,
        expected: FrameState,
        new: FrameState,
        success: Ordering,
        failure: Ordering,
    ) -> Result<FrameState, FrameState> {
        self.0
            .compare_exchange(expected as u8, new as u8, success, failure)
            .map(FrameState::from_u8)
            .map_err(FrameState::from_u8)
    }
}

/// Cache-line padded atomic used for per-frame pin tracking.
///
/// The read hot path touches `pin_count` on every `fix`/`unfix`, while the
/// surrounding header fields are much colder. Padding it out keeps frequent
/// pin traffic from sharing a cache line with eviction/WAL metadata.
#[repr(align(128))]
pub struct PaddedAtomicU32(AtomicU32);

impl PaddedAtomicU32 {
    pub fn new(value: u32) -> Self {
        PaddedAtomicU32(AtomicU32::new(value))
    }

    pub fn load(&self, order: Ordering) -> u32 {
        self.0.load(order)
    }

    pub fn store(&self, value: u32, order: Ordering) {
        self.0.store(value, order);
    }

    pub fn fetch_add(&self, value: u32, order: Ordering) -> u32 {
        self.0.fetch_add(value, order)
    }

    pub fn fetch_sub(&self, value: u32, order: Ordering) -> u32 {
        self.0.fetch_sub(value, order)
    }
}

pub struct FrameCoreHeader {
    /// Number of active fixers. Atomic — modified concurrently on every
    /// read-side access, so keep it isolated from colder metadata.
    pub pin_count: PaddedAtomicU32,
    /// Which page is loaded in this frame (if any).
    pub pid: PageId,
    /// Set when the page has been modified since last write-back.
    pub dirty: AtomicBool,
    /// Clock-style second-chance bit. Set on page touches and cleared by
    /// the evictor before a page becomes a real victim candidate.
    pub referenced: AtomicBool,
    /// Explicit frame lifecycle state.
    pub state: AtomicFrameState,
    /// LSN of the most recent WAL record for this page.
    pub page_lsn: AtomicU64,
    /// WAL buffer epoch containing the most recent buffered page image.
    pub wal_buffer_epoch: AtomicU64,
    /// Byte offset within the WAL buffer for the buffered page image.
    pub wal_buffer_offset: AtomicU32,
}

impl FrameCoreHeader {
    pub fn new_free() -> Self {
        Self {
            pin_count: PaddedAtomicU32::new(0),
            pid: 0,
            dirty: AtomicBool::new(false),
            referenced: AtomicBool::new(false),
            state: AtomicFrameState::new(FrameState::Free),
            page_lsn: AtomicU64::new(0),
            wal_buffer_epoch: AtomicU64::new(0),
            wal_buffer_offset: AtomicU32::new(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_transitions_match_state_machine() {
        use FrameState::*;
        assert!(FrameState::is_valid_transition(Free, Loading));
        assert!(FrameState::is_valid_transition(Free, Resident));
        assert!(FrameState::is_valid_transition(Loading, Resident));
        assert!(FrameState::is_valid_transition(Loading, Free));
        assert!(FrameState::is_valid_transition(Resident, Evicting));
        assert!(FrameState::is_valid_transition(Evicting, Free));
        assert!(FrameState::is_valid_transition(Evicting, Resident));
    }

    #[test]
    fn invalid_transitions_are_rejected() {
        use FrameState::*;
        assert!(!FrameState::is_valid_transition(Free, Evicting));
        assert!(!FrameState::is_valid_transition(Loading, Evicting));
        assert!(!FrameState::is_valid_transition(Resident, Free));
        assert!(!FrameState::is_valid_transition(Resident, Loading));
        assert!(!FrameState::is_valid_transition(Evicting, Loading));
        assert!(!FrameState::is_valid_transition(Free, Free));
        assert!(!FrameState::is_valid_transition(Resident, Resident));
    }

    #[test]
    fn full_lifecycle_transitions_succeed() {
        let s = AtomicFrameState::new(FrameState::Free);
        s.transition(FrameState::Free, FrameState::Loading, Ordering::Relaxed);
        assert_eq!(s.load(Ordering::Relaxed), FrameState::Loading);
        s.transition(FrameState::Loading, FrameState::Resident, Ordering::Relaxed);
        assert_eq!(s.load(Ordering::Relaxed), FrameState::Resident);
        s.transition(
            FrameState::Resident,
            FrameState::Evicting,
            Ordering::Relaxed,
        );
        assert_eq!(s.load(Ordering::Relaxed), FrameState::Evicting);
        s.transition(FrameState::Evicting, FrameState::Free, Ordering::Relaxed);
        assert_eq!(s.load(Ordering::Relaxed), FrameState::Free);
    }

    #[test]
    fn revert_transition_succeeds() {
        let s = AtomicFrameState::new(FrameState::Evicting);
        s.transition(
            FrameState::Evicting,
            FrameState::Resident,
            Ordering::Relaxed,
        );
        assert_eq!(s.load(Ordering::Relaxed), FrameState::Resident);
    }

    #[test]
    #[should_panic(expected = "invalid frame state transition")]
    fn invalid_transition_panics_in_debug() {
        let s = AtomicFrameState::new(FrameState::Resident);
        s.transition(FrameState::Resident, FrameState::Free, Ordering::Relaxed);
    }

    #[test]
    fn compare_exchange_transition_succeeds() {
        let s = AtomicFrameState::new(FrameState::Resident);
        assert!(
            s.compare_exchange_transition(
                FrameState::Resident,
                FrameState::Evicting,
                Ordering::Relaxed,
                Ordering::Relaxed
            )
            .is_ok()
        );
        assert_eq!(s.load(Ordering::Relaxed), FrameState::Evicting);
    }
}
