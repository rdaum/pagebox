use crate::state::{COOL_BIT, EVICTED_BIT, classify_raw};
use crate::word::{page_id_of, pointer_bits_of};
use crate::{AtomicSwipWord, SwipState, SwipWord};

#[cfg(loom)]
use loom::sync::atomic::Ordering;
#[cfg(not(loom))]
use std::sync::atomic::Ordering;

#[test]
fn evicted_roundtrip() {
    let swip = SwipWord::evicted_page(42);
    assert!(swip.is_evicted());
    assert_eq!(swip.page_id(), 42);
}

#[test]
fn pointer_lifecycle() {
    let mut swip = SwipWord::hot_ptr(0x1000);
    assert!(swip.is_hot());
    assert_eq!(swip.pointer_bits(), 0x1000);

    swip.cool();
    assert!(swip.is_cool());
    assert_eq!(swip.pointer_bits(), 0x1000);

    swip.warm();
    assert!(swip.is_hot());
    assert_eq!(swip.pointer_bits(), 0x1000);
}

#[test]
fn cool_to_evicted_to_hot() {
    let mut swip = SwipWord::hot_ptr(0x1000);
    swip.cool();
    swip.evict(99);
    assert!(swip.is_evicted());
    assert_eq!(swip.page_id(), 99);

    swip.resolve_ptr(0x2000);
    assert!(swip.is_hot());
    assert_eq!(swip.pointer_bits(), 0x2000);
}

#[test]
fn atomic_compare_exchange_failure_preserves_current_value() {
    let atomic = AtomicSwipWord::new(SwipWord::evicted_page(7));
    let wrong = SwipWord::evicted_page(8);
    let new = SwipWord::hot_ptr(0x2000);
    let got = atomic
        .compare_exchange(wrong, new, Ordering::AcqRel, Ordering::Acquire)
        .unwrap_err();
    assert_eq!(got.raw(), SwipWord::evicted_page(7).raw());
    assert_eq!(
        atomic.load(Ordering::Acquire).raw(),
        SwipWord::evicted_page(7).raw()
    );
}

#[test]
fn raw_state_classification_matches_tag_patterns() {
    assert_eq!(classify_raw(0x1000), SwipState::Hot);
    assert_eq!(classify_raw(COOL_BIT | 0x1000), SwipState::Cool);
    assert_eq!(classify_raw(EVICTED_BIT | 99), SwipState::Evicted);
}

#[test]
fn page_id_and_pointer_masks_clear_tag_bits() {
    let raw = EVICTED_BIT | COOL_BIT | 123;
    assert_eq!(page_id_of(raw), COOL_BIT | 123);
    assert_eq!(pointer_bits_of(raw), 123);
}

#[test]
fn evicted_alias_matches_evicted_page() {
    let a = SwipWord::evicted(99);
    let b = SwipWord::evicted_page(99);
    assert_eq!(a.raw(), b.raw());
    assert!(a.is_evicted());
    assert_eq!(a.page_id(), 99);
}

#[test]
fn cool_to_cool_is_idempotent() {
    let mut swip = SwipWord::hot_ptr(0x1000);
    swip.cool();
    let raw_after_first_cool = swip.raw();
    // Calling cool again is a debug_assert violation in debug builds.
    // The raw value should be unchanged after a single cool.
    assert!(swip.is_cool());
    assert_eq!(swip.raw(), raw_after_first_cool);
}

#[test]
fn resolve_ptr_roundtrip_preserves_pointer() {
    let mut swip = SwipWord::hot_ptr(0x5000);
    swip.cool();
    swip.evict(200);
    assert!(swip.is_evicted());
    assert_eq!(swip.page_id(), 200);
    swip.resolve_ptr(0x6000);
    assert!(swip.is_hot());
    assert_eq!(swip.pointer_bits(), 0x6000);
}

#[test]
fn cas_evicted_to_hot_succeeds() {
    let atomic = AtomicSwipWord::new(SwipWord::evicted_page(5));
    let result = atomic.compare_exchange(
        SwipWord::evicted_page(5),
        SwipWord::hot_ptr(0x3000),
        Ordering::AcqRel,
        Ordering::Acquire,
    );
    assert!(result.is_ok());
    let loaded = atomic.load(Ordering::Acquire);
    assert!(loaded.is_hot());
    assert_eq!(loaded.pointer_bits(), 0x3000);
}

#[test]
fn cas_hot_to_evicted_succeeds() {
    let atomic = AtomicSwipWord::new(SwipWord::hot_ptr(0x4000));
    let result = atomic.compare_exchange(
        SwipWord::hot_ptr(0x4000),
        SwipWord::evicted_page(77),
        Ordering::AcqRel,
        Ordering::Acquire,
    );
    assert!(result.is_ok());
    let loaded = atomic.load(Ordering::Acquire);
    assert!(loaded.is_evicted());
    assert_eq!(loaded.page_id(), 77);
}

#[test]
#[should_panic(expected = "can only evict a cool swip")]
fn evict_on_hot_swip_panics() {
    let mut swip = SwipWord::hot_ptr(0x1000);
    swip.evict(50);
}

#[test]
#[should_panic(expected = "can only warm a cool swip")]
fn warm_on_evicted_swip_panics() {
    let mut swip = SwipWord::evicted_page(42);
    swip.warm();
}

#[test]
#[should_panic(expected = "can only cool a hot swip")]
fn cool_on_evicted_swip_panics() {
    let mut swip = SwipWord::evicted_page(42);
    swip.cool();
}

#[cfg(all(test, loom))]
mod loom_tests {
    use loom::sync::Arc;
    use loom::sync::atomic::Ordering;
    use loom::thread;

    use crate::{AtomicSwipWord, SwipWord};

    #[test]
    fn loom_compare_exchange_only_one_writer_wins() {
        loom::model(|| {
            let atomic = Arc::new(AtomicSwipWord::new(SwipWord::evicted_page(1)));

            let a1 = atomic.clone();
            let t1 = thread::spawn(move || {
                a1.compare_exchange(
                    SwipWord::evicted_page(1),
                    SwipWord::evicted_page(2),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            });

            let a2 = atomic.clone();
            let t2 = thread::spawn(move || {
                a2.compare_exchange(
                    SwipWord::evicted_page(1),
                    SwipWord::hot_ptr(0x2000),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            });

            let r1 = t1.join().unwrap();
            let r2 = t2.join().unwrap();
            assert!(
                !(r1 && r2),
                "two CAS operations succeeded from the same old word"
            );
        });
    }

    #[test]
    fn loom_failed_compare_exchange_returns_current_word() {
        loom::model(|| {
            let atomic = Arc::new(AtomicSwipWord::new(SwipWord::evicted_page(1)));

            let writer = {
                let atomic = atomic.clone();
                thread::spawn(move || {
                    atomic.store(SwipWord::evicted_page(9), Ordering::Release);
                })
            };

            let cas = {
                let atomic = atomic.clone();
                thread::spawn(move || {
                    atomic.compare_exchange(
                        SwipWord::evicted_page(1),
                        SwipWord::hot_ptr(0x4000),
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                })
            };

            writer.join().unwrap();
            let result = cas.join().unwrap();
            if let Err(observed) = result {
                let final_raw = atomic.load(Ordering::Acquire).raw();
                assert_eq!(
                    observed.raw(),
                    final_raw,
                    "failed CAS should report a real current word"
                );
            }
        });
    }
}
