# `pagebox-frame-kernel`

Small page-id and frame-state kernel types.

## Role

`pagebox-frame-kernel` isolates representation-sensitive page and frame metadata
that storage, WAL, and row code need to agree on. Keeping this crate narrow
makes the bit-level state easier to test and reason about independently from the
full buffer pool.

## Major Pieces

- `src/page_id.rs` defines page identity, page class tagging, encoded page IDs,
  and page-id helpers.
- `src/frame.rs` defines core frame lifecycle and metadata state shared by lower
  storage code.
- `src/tests.rs` covers state and encoding round trips.

## Used By

- `pagebox-storage` for buffer-frame and buffer-pool metadata.
- `pagebox-wal` for page-oriented log records.
- `pagebox-runtime` and `pagebox-row` for lower-level integration points.

## Uses

- No other Boxter crates.
