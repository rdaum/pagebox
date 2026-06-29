//! Pure bit-layout helpers for the hybrid-latch version word.
//!
//! These functions are the single source of truth for how the 64-bit version
//! word is interpreted; `latch.rs` calls into them for every transition and
//! `anneal` blocks on each item state the post-condition the caller relies on.
//!
//! Word layout:
//!
//! ```text
//!   bit 0            : EXCLUSIVE_BIT — set iff a writer is in its critical section
//!   bits 1..=63      : base version, always even in the readable state
//! ```
//!
//! `VERSION_STEP` is `2`: each completed exclusive section advances the base
//! version by two so that any readable version observed before the write is
//! strictly unequal to any readable version observed after it. That inequality
//! is what lets [`OptimisticGuard::validate`](crate::OptimisticGuard::validate)
//! decide restart-vs-commit with a single equality test.
//!
//! `can_advance_readable_version` caps the budget at `u64::MAX - 1`; the
//! terminal exclusive state `u64::MAX` (bit 0 set, base `u64::MAX - 1`) cannot
//! be exited without overflowing and is rejected at exclusive-entry time by
//! the latch.

const EXCLUSIVE_BIT: u64 = 1;
const CLEAR_EXCLUSIVE_BIT: u64 = !EXCLUSIVE_BIT;
const VERSION_STEP: u64 = 2;

/// ```anneal
/// ensures:
///   ret = word &&& EXCLUSIVE_BIT
/// proof (h_anon):
///   unfold exclusive_mask at h_returns
///   simp_all [EXCLUSIVE_BIT]
/// ```
pub(crate) fn exclusive_mask(word: u64) -> u64 {
    word & EXCLUSIVE_BIT
}

/// ```anneal
/// ensures:
///   ret = (mask != 0#u64)
/// proof (h_anon):
///   unfold exclusive_mask_is_set at h_returns
///   cases ret <;> simp_all
/// ```
pub(crate) fn exclusive_mask_is_set(mask: u64) -> bool {
    mask != 0
}

/// ```anneal
/// ensures:
///   ret = ((word &&& EXCLUSIVE_BIT) != 0#u64)
/// proof (h_anon):
///   unfold version_is_exclusive at h_returns
///   unfold exclusive_mask exclusive_mask_is_set at h_returns
///   cases ret <;> simp_all [EXCLUSIVE_BIT]
/// ```
pub(crate) fn version_is_exclusive(word: u64) -> bool {
    exclusive_mask_is_set(exclusive_mask(word))
}

/// ```anneal
/// ensures:
///   match ret with
///   | true => current.val = snapshot.val
///   | false => current.val ≠ snapshot.val
/// proof (h_anon):
///   unfold optimistic_snapshot_still_valid at h_returns
///   cases ret <;> simp_all
/// ```
pub(crate) fn optimistic_snapshot_still_valid(current: u64, snapshot: u64) -> bool {
    current == snapshot
}

/// ```anneal
/// ensures:
///   ret = !(current = snapshot)
/// proof (h_anon):
///   unfold optimistic_restart_required at h_returns
///   unfold optimistic_snapshot_still_valid at h_returns
///   cases ret <;> simp_all
/// ```
pub(crate) fn optimistic_restart_required(current: u64, snapshot: u64) -> bool {
    !optimistic_snapshot_still_valid(current, snapshot)
}

/// ```anneal
/// ensures:
///   ret = current ||| EXCLUSIVE_BIT
/// proof (h_anon):
///   unfold enter_exclusive_version at h_returns
///   cases h_returns
///   simp_all [EXCLUSIVE_BIT]
/// ```
pub(crate) fn enter_exclusive_version(current: u64) -> u64 {
    current | EXCLUSIVE_BIT
}

/// ```anneal
/// ensures:
///   ret = exclusive_version &&& CLEAR_EXCLUSIVE_BIT
/// proof (h_anon):
///   unfold exclusive_base_version at h_returns
///   simp_all [CLEAR_EXCLUSIVE_BIT, EXCLUSIVE_BIT]
/// ```
pub(crate) fn exclusive_base_version(exclusive_version: u64) -> u64 {
    exclusive_version & CLEAR_EXCLUSIVE_BIT
}

/// ```anneal
/// ensures:
///   ret = (version <= 18446744073709551613#u64)
/// proof (h_anon):
///   unfold can_advance_readable_version at h_returns
///   cases ret <;> simp_all
/// ```
pub(crate) fn can_advance_readable_version(version: u64) -> bool {
    version <= 18_446_744_073_709_551_613
}

pub(crate) fn advance_readable_version_checked(version: u64) -> Option<u64> {
    version.checked_add(VERSION_STEP)
}

pub(crate) fn advance_readable_version(version: u64) -> u64 {
    assert!(
        can_advance_readable_version(version),
        "hybrid latch version overflow on exclusive release"
    );
    advance_readable_version_checked(version)
        .expect("hybrid latch version overflow on exclusive release")
}

pub(crate) fn exit_exclusive_version(exclusive_version: u64) -> u64 {
    debug_assert!(version_is_exclusive(exclusive_version));
    let base_version = exclusive_base_version(exclusive_version);
    advance_readable_version(base_version)
}

/// ```anneal
/// ensures:
///   ret = false
/// proof (h_anon):
///   unfold entered_version_disallows_optimistic_read at h_returns
///   unfold optimistic_read_allowed enter_exclusive_version version_is_exclusive exclusive_mask exclusive_mask_is_set at h_returns
///   cases ret <;> simp_all [EXCLUSIVE_BIT]
/// ```
#[allow(dead_code)]
pub(crate) fn entered_version_disallows_optimistic_read(current: u64) -> bool {
    optimistic_read_allowed(enter_exclusive_version(current))
}

/// ```anneal
/// ensures:
///   ret = !((version &&& EXCLUSIVE_BIT) != 0#u64)
/// proof (h_anon):
///   unfold optimistic_read_allowed at h_returns
///   unfold version_is_exclusive exclusive_mask exclusive_mask_is_set at h_returns
///   simp_all [EXCLUSIVE_BIT]
/// ```
pub(crate) fn optimistic_read_allowed(version: u64) -> bool {
    !version_is_exclusive(version)
}

/// ```anneal
/// ensures:
///   match ret with
///   | .none => ((version &&& EXCLUSIVE_BIT) != 0#u64)
///   | .some snapshot => !((version &&& EXCLUSIVE_BIT) != 0#u64) ∧ snapshot = version
/// proof (h_anon):
///   unfold optimistic_snapshot at h_returns
///   unfold optimistic_read_allowed version_is_exclusive exclusive_mask exclusive_mask_is_set at h_returns
///   by_cases h_exclusive : ((version &&& EXCLUSIVE_BIT) != 0#u64)
///   · simp [h_exclusive] at h_returns
///     cases h_returns
///     simp [h_exclusive]
///   · simp [h_exclusive] at h_returns
///     cases h_returns
///     simp [h_exclusive]
/// proof (h_progress):
///   unfold optimistic_snapshot
///   unfold optimistic_read_allowed version_is_exclusive exclusive_mask exclusive_mask_is_set
///   by_cases h_exclusive : ((version &&& EXCLUSIVE_BIT) != 0#u64)
///   · refine ⟨(.none), ?_⟩
///     simp [h_exclusive]
///   · refine ⟨(.some version), ?_⟩
///     simp [h_exclusive]
/// ```
pub(crate) fn optimistic_snapshot(version: u64) -> Option<u64> {
    if !optimistic_read_allowed(version) {
        None
    } else {
        Some(version)
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) const TEST_EXCLUSIVE_BIT: u64 = EXCLUSIVE_BIT;
