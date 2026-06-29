# `pagebox-storage`

Low-level page, frame, buffer-pool, and page-store substrate.

## Role

`pagebox-storage` is the storage layer beneath trees and tables. It manages page
residency, buffer frames, page stores, page formats, free-page allocation, and
the page-level operations that higher layers compose into indexes and tables.

## Major Pieces

- `src/buffer_pool.rs` owns page fixing, residency, swizzling integration,
  eviction interaction, and buffer-pool telemetry.
- `src/buffer_frame.rs` defines typed frame state, latching, LSNs, parent links,
  dirty state, and page-class handling.
- `src/page_store.rs` and `src/page_provider.rs` provide file-backed and
  in-memory page-store access.
- `src/page_header.rs` defines common page headers and page type tags.
- `src/slotted_page.rs`, `src/tuple_page.rs`, and `src/delta_page.rs` implement
  page-local row and delta formats used by table storage.
- `src/free_page_allocator.rs` and `src/row_id.rs` provide page reuse and rowid
  helpers.
- `src/bin/profile_slotted_page.rs` and `benches/` cover storage microprofile
  and benchmark work.

## Used By

- `pagebox-btree` and `pagebox-betree` for persistent tree pages.
- `pagebox-table` for tuple/delta page storage and rowid operations.
- `pagebox-runtime` for database-level storage ownership and recovery.
- `pagebox-wal`, `pagebox-query`, `pagebox-sexpr`, and `pagebox-tpcc-esque` for
  lower-level integration, tests, and workload telemetry.

## Uses

- `pagebox-frame-kernel` for encoded page IDs and narrow frame-state types.
- `pagebox-swip-kernel` for raw swizzled pointer words and atomic operations.
- `pagebox-hybrid-latch` for optimistic/shared/exclusive frame latching.
- `pagebox-threading` for background worker support.
- `pagebox-wal` for page durability integration where needed.
