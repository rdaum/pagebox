# `boxter-betree`

Experimental copy-on-write B-epsilon tree storage.

## Role

`boxter-betree` is an experimental storage engine for copy-on-write B-epsilon-tree-style version
storage. It explores buffered messages, large page classes, versioned values, copy-on-write
rewrites, and incremental garbage collection as an alternative to the current B+tree/table version
storage paths.

## Major Pieces

- `src/tree.rs` defines `CowBeTree`, configuration, write routing, lookup, message flushing, page
  rewrites, fork tracking, debug snapshots, and incremental GC cursors.
- `src/message.rs` defines public `CowBeTreeMessage` operations and internal buffered version
  records.
- `src/page.rs` defines CoW B-epsilon leaf/internal page encoding, fences, buffered-message layout,
  lookup routing, and page-format errors.
- `src/stats.rs` defines tree event counters and their metric export surface.
- `benches/microbetreebench.rs` contains focused benchmark coverage for the experimental path.

## Key Concepts

- Writes enter the tree as messages and are flushed down internal buffers toward leaves instead of
  immediately mutating one leaf per write.
- Pages are rewritten copy-on-write, allowing forks and shared-page accounting while experiments
  run.
- Visible versions are timestamped and can be pruned by incremental GC once a watermark makes older
  versions unreachable.
- The crate currently works with encoded byte keys and byte payloads; table and row semantics are
  supplied by callers.

## Used By

- `boxter-table` for experimental versioned storage integration.

## Uses

- `boxter-storage` for page classes, buffer frames, pinned/exclusive frames, and buffer-pool access.
- `boxter-swip-kernel` for page references.
- `boxter-hybrid-latch` for page/tree concurrency primitives.
