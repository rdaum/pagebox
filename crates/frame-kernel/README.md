# `pagebox-frame-kernel`

Small page-id and frame-state kernel types.

## Role

`pagebox-frame-kernel` isolates representation-sensitive page, frame-state, and
LSN types that storage and WAL need to agree on. Keeping this crate narrow makes
the state machine and page identity easier to test independently from the full
buffer pool.

## Major Pieces

- `src/page_id.rs` defines the unified 64 KiB page size, plain nonzero `u64`
  page IDs, LSNs, and page-id helpers.
- `src/frame.rs` defines core frame lifecycle and metadata state shared by lower
  storage code.
- `src/tests.rs` covers state and encoding round trips.

## Used By

- `pagebox-storage` for buffer-frame and buffer-pool metadata.
- `pagebox-wal` for page-oriented log records.
- `kvstore` for shared page and LSN types at the composition boundary.

## Uses

- No dependencies.
