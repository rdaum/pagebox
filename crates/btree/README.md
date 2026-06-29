# `pagebox-btree`

Concurrent B+tree index implementation.

## Role

`pagebox-btree` provides the ordered index used by the table layer for primary
and secondary access paths. It sits above the page/buffer substrate and below
MVCC/table semantics.

## Major Pieces

- `src/btree.rs` contains the tree implementation, page traversal, mutation,
  splitting, range scans, iterators, and stats.
- `src/lib.rs` exposes the public tree surface used by table and runtime code.
- `benches/` contains tree microbenchmarks and workload-shaped probes.

## Key Concepts

- Keys and payloads are byte strings; higher layers decide value encodings.
- SWIPs represent hot, cool, and evicted child/page references.
- Hybrid latches protect concurrent traversal, updates, and publication.
- Range and prefix scans support SQL index lookup, index scan, and range scan
  plans through the table/query layers.

## Used By

- `pagebox-table` for table primary and secondary indexes.
- `pagebox-runtime` for database-level index ownership and metadata operations.

## Uses

- `pagebox-storage` for pages, frames, and buffer-pool residency.
- `pagebox-swip-kernel` for raw swizzled pointer representation.
- `pagebox-hybrid-latch` for concurrent tree access.
