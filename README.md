# pagebox

Pagebox is a high-performance storage substrate for data systems, written in
Rust.

It's meant for any data storage application that needs memory-fast access to a
hot working set but with disk-backed scale.

It is a LeanStore/Umbra-influenced buffer pool, page store, write-ahead log, and
concurrent B+tree.

It was extracted from my own unpublished "Boxter" project (which is a
high-performance hybrid in-memory/disk-focused OLTP relational database /
database engine that I've been yak-shaving for a long time...).

## Status

Pagebox is **pre-1.0 and not production-ready.** Do not trust it with data you
cannot afford to lose.

On-disk formats (page layouts, WAL records, manifest entries, user-meta slot
semantics) are unreleased and will change without migration scaffolding — a
format change means reinitializing local data, not upgrading it.

There is a substantial test suite: unit and integration tests across every
substrate crate, `#[should_panic]` contract tests for invariant violations,
differential / property tests against `BTreeMap` oracles, loom models for the
concurrency primitives that admit enumeration, and microbenchmarks for the hot
paths.

That said, this is a work in progress. Subtle correctness and concurrency bugs
are likely still hiding like broken glass in the grass. And I reserve the right to
change the API and storage formats.

The goal is a robust, measured storage engine; I'm not there yet. 

Use it, study it, break it, file issues.

## Why

The days of [constantly falling memory prices](https://dam.stanford.edu/memory-prices.html)
are over. Once we thought we'd be
holding ever larger datasets in ever cheaper DRAM. Now it's more important than
ever to make the best use of the RAM that we have.

Memory has gotten cheaper over decades, but not cheap enough to hold a working
set that outgrows RAM — and the gap between DRAM and persistent storage latency
remains large. Modern storage engines need to feel in-memory on the active
working set, scale beyond RAM without falling apart, and keep page-level access
cheap on multicore hardware. Pagebox provides the low-level primitives that make
that possible:

- A **swizzled-pointer buffer pool** with anonymous-`mmap` virtual-memory
  reservation (`MAP_NORESERVE`), resident-budget eviction, and latch-efficient
  page access.
- A **hybrid optimistic/exclusive latch** that keeps the common read path
  latch-free and falls back to exclusive locks only under contention.
- A **concurrent B+tree** with swizzled child/page references, hybrid-latched
  access, and ordered range and prefix scans.
- A **write-ahead log** with group commit, configurable sync backends, streaming
  replay, and crash recovery.
- A **file-backed page store** with a free-page allocator, header-resident user
  meta slots, and sync/fsync control.

## Crates

| Crate                  | Role                                                                                                          |
|------------------------|---------------------------------------------------------------------------------------------------------------|
| `pagebox-frame-kernel` | Page-id, frame-state, and LSN types shared by storage, WAL, and tree code.                                    |
| `pagebox-swip-kernel`  | Swizzled-pointer word representation and atomic state transitions for hot/cool/evicted pages.                 |
| `pagebox-hybrid-latch` | Optimistic/shared/exclusive latch primitive used by storage and tree hot paths.                               |
| `pagebox-threading`    | Linux-aware thread spawning, CPU topology detection, and optional CPU pinning helpers.                        |
| `pagebox-wal`          | Write-ahead log format, append path, group commit, sync, replay scanning, and WAL telemetry.                  |
| `pagebox-storage`      | Buffer pool, page store, page formats, buffer frames, slotted pages, free-page allocation, and page provider. |
| `pagebox-btree`        | Production concurrent B+tree with swizzled pointers, hybrid latching, and ordered scans.                      |
| `kvstore`              | Example durable KV store binary built on the substrate (see below).                                           |

Internal dependency DAG:

```
frame-kernel   (0 deps)
swip-kernel    (0 deps)
threading      (libc)
hybrid-latch   (parking_lot; + optional fast-telemetry)
wal            -> frame-kernel, threading
storage        -> frame-kernel, hybrid-latch, swip-kernel, threading, wal
btree          -> hybrid-latch, storage, swip-kernel
```

I've tried to keep Pagebox with very little outward coupling. It depends only on
`parking_lot`, `libc`, `crc-fast`, `crossbeam-queue`, and (optionally)
[`fast-telemetry`](https://crates.io/crates/fast-telemetry) for metrics.

## Quick Start

Build the substrate:

```bash
cargo build --workspace
```

Run the example KV store:

```bash
cargo run -p kvstore -- put hello world
cargo run -p kvstore -- put foo bar
cargo run -p kvstore -- get hello          # -> world
cargo run -p kvstore -- scan
cargo run -p kvstore -- del hello
cargo run -p kvstore -- checkpoint
```

Data persists across process restarts via WAL recovery. Killing the process
mid-write and reopening will recover committed page images from the WAL.

## Telemetry

Each crate that instruments hot paths (`pagebox-wal`, `pagebox-storage`,
`pagebox-btree`, `pagebox-hybrid-latch`) exposes a `metrics` feature, on by default. With `metrics` enabled, the crate
uses [`fast-telemetry`](https://crates.io/crates/fast-telemetry) counters,
histograms, and gauges. With it disabled, no-op shims take their place and the
crate pulls zero telemetry dependencies:

```bash
cargo build -p pagebox-storage --no-default-features
```

A downstream application can propagate the feature:

```toml
[features]
default = ["metrics"]
metrics = ["pagebox-storage/metrics", "pagebox-wal/metrics", "pagebox-btree/metrics"]
```

## Example: `kvstore`

`kvstore` is a standalone durable key-value store built on exactly four substrate
crates (`pagebox-btree`, `pagebox-storage`, `pagebox-wal`, `pagebox-frame-kernel`)
plus `clap`. It demonstrates that the substrate composes into a real crash-safe
engine with no database, SQL, or row-layer types in scope.

The open path mirrors a full database recovery sequence:

1. `FilePageStore::open` — open or create the data file.
2. `Wal::open_opts` — open or create the WAL.
3. `wal.recover(&store, checkpoint_lsn, read_page_lsn)` — replay page images.
4. `store.sync()` — flush recovered pages.
5. `BufferPool::with_store` + `pool.set_wal` — wire the pool to the store and WAL.
6. `BTree::new` or `BTree::open` — create or reopen the tree from `user_meta_0`
   (root page ID) and `user_meta_1` (height).

Checkpoint flushes dirty pages, persists tree metadata into the store's user
meta slots, advances the checkpoint LSN, and resets the WAL.

## Research Background

Pagebox is not a reimplementation of any one system, but its design is shaped by
research on memory-optimised disk-based engines:

- [LeanStore: In-Memory Data Management beyond Main Memory](https://db.in.tum.de/~leis/papers/leanstore.pdf)
  for swizzled pointers, hot/cool page management, and low-overhead buffer-managed
  storage.
- [Umbra: A Disk-Based System with In-Memory Performance](https://www.cidrdb.org/cidr2020/papers/p29-neumann-cidr20.pdf)
  for memory-optimised disk-based database architecture and adaptive execution.

## Testing and Benchmarks

```bash
# Substrate unit tests
cargo test -p pagebox-storage
cargo test -p pagebox-wal
cargo test -p pagebox-btree
cargo test -p pagebox-hybrid-latch

# Lint
cargo clippy --workspace --all-targets
cargo fmt --all

# Microbenchmarks
cargo bench -p pagebox-btree --bench btree
cargo bench -p pagebox-storage --bench microbufferpoolfaultbench
cargo bench -p pagebox-wal --bench wal
```

## License

Pagebox is free software, licensed under the **GNU Lesser General Public
License** version 3 or later. See [`LICENSE.md`](./LICENSE.md) for the full
text.
