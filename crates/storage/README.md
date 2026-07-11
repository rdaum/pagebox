# `pagebox-storage`

Low-level page, frame, buffer-pool, and page-store substrate.

## Role

`pagebox-storage` manages page residency, buffer frames, page stores, the
unified compile-time page format, free-page allocation, and the page-level operations
that higher layers compose into indexes or other data structures.

## Major Pieces

- `src/buffer_pool.rs` owns page fixing, residency, swizzling integration,
  eviction interaction, and buffer-pool telemetry.
- `src/buffer_frame.rs` defines typed frame state, latching, LSNs, parent links,
  dirty state, and the 4096-aligned two-half frame layout.
- `src/page_store.rs` and `src/page_provider.rs` provide file-backed and
  in-memory page-store access.
- `src/page_header.rs` defines common page headers and page type tags.
- `src/slotted_page.rs` implements the generic sorted key/value page format.
- `src/free_page_allocator.rs` provides sharded page reuse and monotonic
  allocation.
- `benches/` contains page-store, buffer-pool, allocator, and slotted-page
  microbenchmarks.

## Used By

- `pagebox-btree` for persistent tree pages.
- `kvstore` for file-backed composition and checkpointing.

## Uses

- `pagebox-frame-kernel` for page IDs, LSNs, and narrow frame-state types.
- `pagebox-swip-kernel` for raw swizzled pointer words and atomic operations.
- `pagebox-hybrid-latch` for optimistic/shared/exclusive frame latching.
- `pagebox-threading` for background worker support.
- `pagebox-wal` for page durability integration where needed.
