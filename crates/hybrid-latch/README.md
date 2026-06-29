# `pagebox-hybrid-latch`

Optimistic/shared/exclusive latch primitive.

## Role

`pagebox-hybrid-latch` provides the concurrency primitive used in hot storage and
tree paths where optimistic reads should be cheap and exclusive updates still
need precise coordination.

## Major Pieces

- `src/latch.rs` implements `HybridLatch`, optimistic guards, shared guards,
  exclusive guards, upgrade/restart behaviour, and wait telemetry.
- `src/helpers.rs` defines version-word helpers and local verification
  annotations.
- `src/lock.rs` adapts the blocking lock backend for normal and loom builds.
- `src/lib.rs` exports the public surface.

## Public Types

- `HybridLatch`
- `OptimisticGuard`
- `SharedGuard`
- `ExclusiveGuard`
- `LatchGuard`
- `Restart`

## Used By

- `pagebox-storage` for frame latching.
- `pagebox-btree` and `pagebox-betree` for tree page traversal and mutation.
- `pagebox-runtime` and `pagebox-tpcc-esque` for telemetry and integration paths.

## Uses

- No other Boxter crates.
