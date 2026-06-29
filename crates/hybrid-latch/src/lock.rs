//! Blocking backend for the hybrid latch.
//!
//! The optimistic fast path never touches this lock; it is only acquired when a
//! caller needs a true shared or exclusive section (via `lock_shared` /
//! `lock_exclusive`) or when an optimistic guard is promoted (via the
//! `upgrade_to_*` family).
//!
//! Two backends are compiled:
//!
//! - **Normal builds** wrap `parking_lot::RawRwLock`, so shared holders coexist
//!   and an exclusive holder is the sole accessor — matching the semantics the
//!   optimistic-version machinery above assumes.
//! - **`cfg(loom)` builds** wrap `loom::sync::Mutex`, because loom has no
//!   `RwLock`. Under loom shared and exclusive acquires are therefore
//!   serialised through the same mutex; this is sound for model checking the
//!   version-word transitions (the property under test is whether optimistic
//!   readers observe a consistent version, not whether shared readers run
//!   concurrently) but does distort real shared/exclusive concurrency and
//!   should not be read as a throughput model.

/// Under loom: use loom::sync::Mutex (loom has no RwLock, so shared=exclusive).
/// Under normal builds: use parking_lot::RawRwLock.
#[cfg(loom)]
pub(crate) mod imp {
    pub(crate) struct InnerLock {
        mu: loom::sync::Mutex<()>,
    }

    impl Default for InnerLock {
        fn default() -> Self {
            Self::new()
        }
    }

    impl InnerLock {
        pub(crate) fn new() -> Self {
            InnerLock {
                mu: loom::sync::Mutex::new(()),
            }
        }

        pub(crate) fn lock_exclusive(&self) -> InnerGuard<'_> {
            InnerGuard(self.mu.lock().unwrap())
        }

        pub(crate) fn try_lock_exclusive(&self) -> Option<InnerGuard<'_>> {
            self.mu.try_lock().ok().map(InnerGuard)
        }

        pub(crate) fn try_lock_shared(&self) -> Option<InnerGuard<'_>> {
            self.mu.try_lock().ok().map(InnerGuard)
        }

        pub(crate) fn lock_shared(&self) -> InnerGuard<'_> {
            self.lock_exclusive()
        }
    }

    pub(crate) struct InnerGuard<'a>(#[allow(dead_code)] pub(crate) loom::sync::MutexGuard<'a, ()>);
}

#[cfg(not(loom))]
pub(crate) mod imp {
    use parking_lot::RawRwLock;
    use parking_lot::lock_api::RawRwLock as RawRwLockTrait;

    pub(crate) struct InnerLock {
        rw: RawRwLock,
    }

    unsafe impl Send for InnerLock {}
    unsafe impl Sync for InnerLock {}

    impl Default for InnerLock {
        fn default() -> Self {
            Self::new()
        }
    }

    impl InnerLock {
        pub(crate) fn new() -> Self {
            InnerLock {
                rw: <RawRwLock as RawRwLockTrait>::INIT,
            }
        }

        pub(crate) fn lock_exclusive(&self) -> ExclusiveInnerGuard<'_> {
            self.rw.lock_exclusive();
            ExclusiveInnerGuard { lock: &self.rw }
        }

        pub(crate) fn try_lock_exclusive(&self) -> Option<ExclusiveInnerGuard<'_>> {
            if self.rw.try_lock_exclusive() {
                Some(ExclusiveInnerGuard { lock: &self.rw })
            } else {
                None
            }
        }

        pub(crate) fn try_lock_shared(&self) -> Option<SharedInnerGuard<'_>> {
            if self.rw.try_lock_shared() {
                Some(SharedInnerGuard { lock: &self.rw })
            } else {
                None
            }
        }

        pub(crate) fn lock_shared(&self) -> SharedInnerGuard<'_> {
            self.rw.lock_shared();
            SharedInnerGuard { lock: &self.rw }
        }
    }

    pub(crate) struct ExclusiveInnerGuard<'a> {
        lock: &'a RawRwLock,
    }

    impl Drop for ExclusiveInnerGuard<'_> {
        fn drop(&mut self) {
            unsafe { self.lock.unlock_exclusive() };
        }
    }

    pub(crate) struct SharedInnerGuard<'a> {
        lock: &'a RawRwLock,
    }

    impl Drop for SharedInnerGuard<'_> {
        fn drop(&mut self) {
            unsafe { self.lock.unlock_shared() };
        }
    }
}
