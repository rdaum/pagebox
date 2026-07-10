# `pagebox-swip-kernel`

Swizzled pointer word representation and atomic operations.

## Role

`pagebox-swip-kernel` owns the raw tagged word used to represent either a
resident frame pointer or an evicted page identifier. Buffer-pool and tree code
build typed page-residency behaviour on top of this 64-bit state machine.

## Major Pieces

- `src/state.rs` defines state bits, masks, and `SwipState` classification.
- `src/word.rs` defines `SwipWord` constructors and accessors for hot, cool,
  and evicted references.
- `src/atomic.rs` defines `AtomicSwipWord` and atomic load/store/swap/CAS
  operations.
- `src/tests.rs` covers state-machine and round-trip behaviour.

## Key Concepts

- Bits 63-62 encode the state.
- Hot and cool states carry a direct pointer value.
- Evicted state carries a page ID.
- Higher layers decide when to cool, fix, evict, or unswizzle pages.

## Used By

- `pagebox-storage` for buffer-pool page references.
- `pagebox-btree` for tree child/page references.

## Uses

- No dependencies.
