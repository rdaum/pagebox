#[cfg(loom)]
use loom::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicU64, Ordering};

use crate::SwipWord;

#[repr(transparent)]
pub struct AtomicSwipWord(AtomicU64);

impl AtomicSwipWord {
    pub fn new(swip: SwipWord) -> Self {
        Self(AtomicU64::new(swip.raw()))
    }

    pub fn load(&self, order: Ordering) -> SwipWord {
        SwipWord::from_raw(self.0.load(order))
    }

    pub fn store(&self, swip: SwipWord, order: Ordering) {
        self.0.store(swip.raw(), order);
    }

    pub fn compare_exchange(
        &self,
        current: SwipWord,
        new: SwipWord,
        success: Ordering,
        failure: Ordering,
    ) -> Result<SwipWord, SwipWord> {
        self.0
            .compare_exchange(current.raw(), new.raw(), success, failure)
            .map(SwipWord::from_raw)
            .map_err(SwipWord::from_raw)
    }
}

impl std::fmt::Debug for AtomicSwipWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AtomicSwipWord({:?})", self.load(Ordering::Relaxed))
    }
}
