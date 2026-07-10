# `pagebox-btree`

Concurrent B+tree index implementation.

## Role

`pagebox-btree` provides a persistent ordered byte-key/byte-value index above
the page and buffer-pool substrate. It intentionally contains no row, MVCC, or
database-level types.

## Major Pieces

- `src/btree.rs` contains the public tree implementation and tests.
- `src/btree/` contains node representation, parent-edge handling, split
  publication, split-child identities, and statistics.
- `src/lib.rs` exports `BTree`.
- `benches/` contains tree microbenchmarks and workload-shaped probes.

## Key Concepts

- Keys and payloads are byte strings; higher layers decide value encodings.
- SWIPs represent hot, cool, and evicted child/page references.
- Hybrid latches protect concurrent traversal, updates, and publication.
- Ordered, prefix, ascending-range, and descending-range scans are part of the
  public byte-oriented API.

## Used By

- `kvstore` as its durable ordered key/value index.

## Uses

- `pagebox-storage` for pages, frames, and buffer-pool residency.
- `pagebox-swip-kernel` for raw swizzled pointer representation.
- `pagebox-hybrid-latch` for concurrent tree access.
