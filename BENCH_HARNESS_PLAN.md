# KV Benchmark Harness — Implementation Plan

Status: **proposed, not started.** This document is the design for a new
top-level tool that drives synthetic workloads against `kvstore` and competitor
embedded KV engines through a single adapter, with a TUI for selecting and
comparing runs. It is a planning artifact, not a commitment; phases will be
landed incrementally and revised as evidence comes in.

## Goal & scope

A standalone user-run benchmark harness that:

- Drives the **same** synthetic workloads against `kvstore` and external
  embedded KV engines (`fjall`, `redb`, `sled`, optionally `rocksdb`/`leveldb`).
- Exposes the workload matrix as an interactive TUI: engine × workload × key
  distribution × record count (relative to buffer-pool budget) × value size ×
  thread count × deletion/TTL ratio.
- Produces machine-comparable results (throughput, latency histogram,
  engine-reported counters) persisted to JSON for run-vs-run and
  version-vs-version comparison.
- Exercises the substrate-specific failure modes the microbenches do not:
  buffer-pool eviction under uniform-scan, hot-spot zipfian residency, WAL
  group-commit contention under mixed writes, slotted-page suffix pressure
  under large values, delete churn.

The substrate-level microbenchmarks under `crates/{btree,storage,wal}/benches/`
(criterion / micromeasure, in-process against `BTree`+`BufferPool`) remain the
authoritative source for *substrate* performance regression. This harness is
one layer up: it measures `kvstore` as a composed binary against peer engines
that solve the same problem with different architectures (LSM-tree, B+tree,
copy-on-write B+tree). The two are complementary, not substitutes.

## Background & primary sources

Two canonical specs ground the workload set:

**YCSB** (Cooper, Silberstein, Tam, Ramakrishnan, Sears. *Benchmarking Cloud
Serving Systems with YCSB.* SoCC 2010; repo `brianfrankcooper/YCSB`). Six core
workloads over a shared synthetic dataset keyed by insert order. The three key
distributions are the heart of it: `uniform`, `zipfian` (theta≈0.99, hot-spot),
`latest` (recency-biased exponential). These directly exercise the buffer-pool
residency story — zipfian keeps the hot set in-frame, `latest` forces eviction
churn, `uniform` scans the whole keyspace.

**db_bench** (LevelDB / RocksDB; repo `facebook/rocksdb`). Named scenarios that
YCSB does not cover, specifically:

| db_bench scenario    | Models                                                        |
|----------------------|---------------------------------------------------------------|
| `fillseq`            | Sequential bulk load (write-optimised path).                 |
| `fillrandom`         | Random bulk load (split / free-page allocator pressure).     |
| `overwrite`          | In-place update churn (no growth, dirty-page no-steal path). |
| `readseq`            | Full ordered scan (slotted-page sequential read).           |
| `readrandom`         | Random point lookups (B+tree `find_leaf` traversal cost).    |
| `readwhilewriting`   | Mixed read/write contention (hybrid latch optimistic vs exclusive). |
| `deleterandom`       | Random deletion churn (free-page retirement, tombstones on LSM side). |
| `deleteseq`          | Sequential deletion (compaction-style mass retire).         |
| `seekrandom`         | Range-seek cost (B-link right-sibling chase).               |

YCSB does not define a deletion workload; `deleterandom`/`deleteseq` from
db_bench is the canonical source for that axis.

## Non-goals

- Not a network/server benchmark. All engines are embedded, in-process, called
  through their Rust API. No client/server transport is in scope.
- Not a row/MVCC/SQL layer benchmark. Workloads operate on `&[u8]` keys and
  `&[u8]` values only, matching the `kvstore::KvStore` surface
  (`crates/kvstore/src/main.rs:68`).
- Not a correctness oracle for the substrate. The harness validates *its own*
  workload generator against a `BTreeMap` model (see Test discipline); substrate
  correctness remains the responsibility of the substrate test suite audited in
  `TEST_AUDIT.md`.
- Not a replacement for the substrate microbenches. Use `crates/btree/benches/ycsb.rs`
  for in-tree B+tree regression; use this harness for engine-to-engine
  comparison.

## Crate placement & dependency boundary

**New workspace member:** `crates/kvbench` (binary crate, no `pagebox-` prefix,
matching the `kvstore` precedent at `crates/kvstore/Cargo.toml:2`).

**Workspace changes** (`Cargo.toml`):

- Add `crates/kvbench` to `members`.
- Do **not** add to `default-members`. The harness pulls heavy external deps
  (`fjall`, `redb`, `sled`, `librocksdb-sys`, `ratatui`); keeping it out of the
  default set preserves the clean substrate-only build that `cargo build` and
  `cargo test` currently give.

**Dependency justification.** `AGENTS.md` restricts substrate-crate
dependencies to `parking_lot`, `libc`, `crc-fast`, `crossbeam-queue`,
`fast-telemetry`. That policy applies to the substrate DAG; the harness is
explicitly *outside* that DAG (it depends on `kvstore` but nothing depends on
it) and is therefore entitled to pull external comparison + UI libraries. The
precedent is the existing dev-dep set (`criterion`, `micromeasure`, `proptest`,
`loom`, `clap`, `tempfile`), which is already beyond the substrate set.

Harness-only dependency list, with role:

| Dependency                | Role                                | Justification                          |
|---------------------------|-------------------------------------|----------------------------------------|
| `kvstore` (path)          | Adapter target under test           | In-workspace.                          |
| `fjall`                   | LSM-tree comparison engine          | Pure-Rust, simplest first external peer. |
| `redb`                    | COW B+tree comparison engine         | Pure-Rust, contrasting architecture.   |
| `sled`                    | LSM-tree comparison engine          | Pure-Rust; historical comparison baseline. |
| `librocksdb-sys` + `rocksdb` | C++ LSM-tree reference           | Heaviest build; gated behind a feature flag, not default. |
| `ratatui` + `crossterm`   | TUI                                 | Standard Rust TUI stack.               |
| `clap`                   | Non-interactive CLI mode            | Already in workspace deps.            |
| `serde` + `serde_json`    | Results persistence                 | Machine-comparable JSON output.        |
| `hdrhistogram`            | Latency histograms                  | Tail-latency reporting.                |

External engines are gated behind individual feature flags (`--features fjall`,
`--features redb`, etc.) so a user can build the harness with only the engines
they care about. `rocksdb` is off by default due to its C++ toolchain
requirement.

## Prerequisite: `kvstore` library surface

`kvstore` is currently a binary-only crate. `crates/kvstore/src/main.rs` holds
both the CLI parsing and the `KvStore` struct with its `pub fn
open/put/get/del/scan/checkpoint` methods — but because they live in `main.rs`
(the binary target), no other crate can name them. The harness must call them,
so the first piece of work is to split the crate into a real library.

Split shape:

- `crates/kvstore/src/lib.rs` — `pub struct KvStore`, options, error type.
- `crates/kvstore/src/main.rs` — thin CLI: `Cli` parse → `KvStore::open` /
  `put` / `get` / etc. No business logic. User-visible behaviour unchanged.

API design (this is the surface the harness `KvEngine` adapter will wrap):

```rust
pub struct KvStore { /* pool, tree, wal, store — private */ }

pub struct KvStoreOptions {
    pub pool_frames: usize,    // was POOL_FRAMES = 1024 (crates/kvstore/src/main.rs:11)
    pub domain_id: u16,        // was DT_ID = 1 (crates/kvstore/src/main.rs:12)
    pub sync_mode: SyncMode,   // Relaxed | Strict — see below
}

pub enum SyncMode {
    /// Writes return after the page is modified; WAL flush is asynchronous.
    /// Mirrors the WAL's relaxed commit mode.
    Relaxed,
    /// Every write blocks until the WAL has flushed it. Mirrors the WAL strict
    /// commit mode and the existing `--sync` Put flag (currently implemented
    /// ad-hoc in main.rs by calling `kv.wal.flush()` directly).
    Strict,
}

impl KvStore {
    /// Defaults: pool_frames = 1024, domain_id = 1, sync_mode = Relaxed.
    pub fn open<P: AsRef<Path>>(dir: P) -> std::io::Result<Self>;
    pub fn open_with<P: AsRef<Path>>(dir: P, opts: &KvStoreOptions) -> std::io::Result<Self>;

    pub fn put(&self, key: &[u8], value: &[u8]) -> bool;
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>>;
    pub fn del(&self, key: &[u8]) -> bool;

    /// Ordered range scan over [start, end). Calls `f` per (key, value).
    /// Replaces the full-scan-only `scan<F>` in the binary; backed by the
    /// underlying `BTree::scan_range`.
    pub fn scan_range(&self, start: &[u8], end: &[u8], f: &mut dyn FnMut(&[u8], &[u8]));

    /// Explicit durable flush — WAL fsync + dirty-page flush. Equivalent to
    /// the current `KvStore::checkpoint` (crates/kvstore/src/main.rs:84) less
    /// the user-meta slot writes (which remain internal to `open`/drop).
    pub fn sync(&self) -> std::io::Result<()>;

    /// Checkpoint: sync + advance user-meta slots + reset WAL. Existing
    /// behaviour preserved for the `checkpoint` CLI subcommand.
    pub fn checkpoint(&self) -> std::io::Result<()>;
}
```

Decisions:

- **`std::io::Result`, not a custom error type.** `AGENTS.md` permits custom
  errors "where appropriate"; at the kvstore lib layer (above the substrate
  hot paths), `io::Result` is sufficient and keeps the harness's `KvEngine:
  open/put/get/...` return-type uniform across adapters. WAL recovery failures
  map to `io::Error`; the substrate panic-on-invariant-violation convention is
  preserved underneath.
- **`SyncMode` instead of per-call `sync: bool`.** The existing `Put --sync`
  is ad-hoc (it's in `main.rs`, not on `KvStore`); the lib surfaces it as a
  knob on the options struct plus the explicit `sync()` method. Strict mode
  corresponds to calling `sync()` after every put; relaxed mode (the default)
  mirrors RocksDB's default buffering.
- **`scan_range` replaces `scan<F>`.** The current full-scan signature
  dictated the harness `KvEngine::scan_range` signature; this aligns the two.
  The CLI's `Scan` subcommand calls `scan_range` over the full keyspace (or a
  separate `scan_all` convenience if the special case turns out to be common).
- **WAL tuning stays env-var-driven.** The WAL crate exposes
  `PAGEBOX_WAL_*` env vars, not constructor knobs
  (`crates/wal/src/wal_impl.rs:944`). The lib preserves this —
  `KvStoreOptions` does not re-expose them. Users who need to tune WAL for
  the harness set the env vars. Reconsider if the harness needs to vary WAL
  config per-run (it would then surface them through `EngineOpts::engine_specific`).
- **No breaking-change concerns.** `AGENTS.md` is explicit: pre-1.0, prefer
  one current format over migration scaffolding. The binary's user-visible
  behaviour is preserved exactly; only the source layout changes.

## Architecture

```text
crates/kvbench/
├── Cargo.toml
└── src/
    ├── main.rs               # Binary entry: TUI by default, --no-tui for batch
    ├── engine.rs             # KvEngine trait + LoadState
    ├── engines/
    │   ├── kvstore_adapter.rs # Wraps kvstore::KvStore
    │   ├── fjall.rs
    │   ├── redb.rs
    │   ├── sled.rs
    │   └── rocksdb.rs        # Behind --features rocksdb
    ├── workload.rs           # WorkloadSpec -> WorkloadOp stream
    ├── distribution.rs      # uniform / zipfian / latest key generators
    ├── driver.rs            # Thread-pool workload executor, latency capture
    ├── stats.rs             # HdrHistogram + throughput aggregation
    ├── tui.rs               # ratatui app: axes, live view, results table
    ├── report.rs           # JSON serialisation + run comparison
    └── tests.rs             # BTreeMap oracle differential for workload.rs
```

### `KvEngine` trait

The adapter trait mirrors the `kvstore::KvStore` surface exposed by the
prerequisite lib refactor above and is the only contract competitor
adapters implement:

```rust
pub trait KvEngine: Send + Sync {
    /// Open a fresh instance rooted at `dir`. Each run uses a fresh dir.
    fn open(dir: &Path, opts: &EngineOpts) -> std::io::Result<Self> where Self: Sized;

    /// Insert or update. Returns true if the key was absent before.
    fn put(&self, key: &[u8], value: &[u8]) -> bool;
    /// Point lookup.
    fn get(&self, key: &[u8]) -> Option<Vec<u8>>;
    /// Remove. Returns true if the key was present.
    fn del(&self, key: &[u8]) -> bool;
    /// Ordered scan over [start, end). Calls `f` per (key, value).
    fn scan_range(&self, start: &[u8], end: &[u8], f: &mut dyn FnMut(&[u8], &[u8]));

    /// Strict-durability flush (engine-defined). Mirrors kvstore::KvStore::checkpoint
    /// (crates/kvstore/src/main.rs:84). Called at end of load phase and per
    /// `--sync-interval` if configured.
    fn sync(&self) -> std::io::Result<()>;

    /// Engine-reported stats for side-channel output (memtable size, cache hit
    /// rate, etc.). Adapter-defined; opaque to the driver.
    fn stats(&self) -> EngineStats { EngineStats::default() }
}

pub struct EngineOpts {
    pub value_size: usize,
    pub sync_mode: SyncMode,        // Relaxed | Strict (mirrors WAL commit modes)
    pub buffer_budget_frames: usize, // kvstore-only: maps to BufferPool::with_store frame count
    pub engine_specific: HashMap<String, String>,
}
```

Sync only at the trait boundary: all engines in scope expose synchronous APIs
(`fjall`, `redb`, `sled`, `rocksdb` are sync). An async driver adds complexity
without touching any adapter, so it is deferred.

### Workload generator

`workload.rs` is engine-agnostic. It consumes a `WorkloadSpec` and produces a
deterministic, seed-reproducible `WorkloadOp` stream:

```rust
pub struct WorkloadSpec {
    pub workload: Workload,   // YcsbA..=YcsbF, FillSeq, FillRandom, ReadRandom, ReadWhileWriting, DeleteRandom, DeleteSeq, SeekRandom, Overwrite
    pub distribution: Distribution, // Uniform | Zipfian { theta } | Latest
    pub record_count: u64,
    pub value_size: usize,
    pub operation_count: u64,
    pub seed: u64,
    pub threads: usize,
    pub deletion_ratio: f64,  // 0.0 = none, 1.0 = all
}

pub enum WorkloadOp { Put { key: Vec<u8>, value: Vec<u8> }, Get { key: Vec<u8> }, Del { key: Vec<u8> }, Scan { start: Vec<u8>, end: Vec<u8> } }
```

Each run is split into **load** (bulk `Put` of `record_count` keys, timed but
reported separately) and **run** (the actual mixed-operation phase, the
headline measurement). This matches the YCSB load/run convention and isolates
bulk-load cost from steady-state throughput.

### Driver, stats, results

- `driver.rs`: N worker threads pull from a shared `WorkloadOp` iterator
  (atomically-indexed for thread partitioning), call the engine, record
  per-operation latency via `std::time::Instant`. No `tokio`; plain
  `std::thread`.
- `stats.rs`: `HdrHistogram` for latency percentiles (p50/p95/p99/p99.9),
  throughput as ops/sec aggregated per workload × engine × thread count.
- `report.rs`: JSON per run; schema versioned (`{ schema: 1, engine, spec,
  load_phase: {...}, run_phase: {...} }`). A `compare` subcommand diffs two
  report files.

### TUI

`ratatui`-based. Layout:

- **Left pane**: matrix of axes (engine × workload), each cell checkable.
  Multi-select to enqueue a run batch.
- **Right pane, top**: live view of the currently running cell — throughput
  curve, latency histogram, ops/sec counter.
- **Right pane, bottom**: completed-run results table (engine, workload,
  record_count, threads, ops/sec, p50/p95/p99).
- **Bottom bar**: control — `r` run selected, `s` stop, `c` clear, `j`/`k`
  navigate, `e` export marked runs to JSON, `q` quit.

Non-interactive mode (`--no-tui`) takes a spec file (TOML) and runs the listed
workloads sequentially, writing JSON. Needed for CI and headless batch runs.

## Workload catalogue

YCSB core (implemented against the spec):

| Workload | Mix                                    | Distribution | Load phase              |
|----------|----------------------------------------|--------------|-------------------------|
| YcsbA    | 50% read / 50% update                  | uniform      | `record_count` puts     |
| YcsbB    | 95% read / 5% update                   | uniform      | same                    |
| YcsbC    | 100% read                              | uniform      | same                    |
| YcsbD    | 95% read / 5% insert (grows dataset)   | latest       | half of `record_count`  |
| YcsbE    | 95% short-range scan / 5% insert       | zipfian      | clustered by thread id  |
| YcsbF    | 100% read-modify-write                | zipfian      | `record_count` puts     |

db_bench scenarios (single-distribution, simpler): `FillSeq`, `FillRandom`,
`Overwrite`, `ReadSeq`, `ReadRandom`, `ReadWhileWriting` (reader threads +
1 writer), `DeleteRandom`, `DeleteSeq`, `SeekRandom`.

TTL / key-expiry workload (extension, phase 4): insert with TTL, let expire,
measure re-arrival cost. Outside both YCSB and db_bench; closer to RocksDB's
TTL bloom path. Flagged as a research extension, not phase 1.

## Key distributions

Implementation references:

- **Uniform**: `seed.wrapping_mul(0x517cc1b727220a95) % record_count` — same
  hashing pattern used in the existing `crates/btree/benches/ycsb.rs:13`.
- **Zipfian**: Hartmann-Leighton-McJones rejection sampler (the YCSB
  implementation, theta=0.99 default). Precompute CDF for `record_count`;
  sample by inverse-CDF. Theta exposed in TUI; default matches the YCSB paper.
- **Latest**: YCSB's `latest` distribution — exponential bias toward recently
  inserted keys. Maintains a moving window of recent insertions; samples
  `uniform(0..window_size)` into the window.

All three are deterministic given a seed. Per-run reproducibility is a hard
requirement: a failing bench must be exactly reproducible.

## Test discipline

Per `AGENTS.md` ("a benchmark with no correctness assertion is a measurement,
not a test"): the harness's own logic is unit-tested, separately from the
benchmark runs.

- **Workload generator oracle**: `crates/kvbench/src/tests.rs` drives a
  `WorkloadSpec` through a `BTreeMap`-backed `KvEngine` mock and asserts
  post-run invariants — final key count matches expected alive-set given the
  insert/delete ratio, no orphan keys, scan_range returns the contiguous
  expected slice. This catches generator bugs (wrong distribution bounds,
  double-counted ops, off-by-one on `record_count`) before they pollute
  measurements.
- **Distribution shape tests**: statistical assertions on the generators —
  uniform within chi-square tolerance, zipfian top-1% keys account for ~>30%
  of probes (theta=0.99), `latest` skews to the last-inserted window.
- **Adapter contract tests**: each `KvEngine` adapter (including the
  `kvstore` one) runs a fixed op sequence (`put A, put B, get A, del A, get A
  == None, scan_range A..C == [B]`) so a broken adapter surfaces as a failed
  test, not a misleading number.

No `assert!(true)` scaffolding, no `dbg!` prints. Genuinely-failing-if-wrong
tests only.

## Phasing

Each phase is independently mergeable and verifiable.

**Phase 0 — harness core, single engine, JSON only.**
- *Prerequisite* — `kvstore` lib refactor per the section above: split
  `main.rs` into `lib.rs` + thin CLI, expose `KvStore`/`KvStoreOptions`/`SyncMode`/`scan_range`/`sync`.
- `crates/kvbench` crate scaffold, added to `members` (not `default-members`).
- `KvEngine` trait + `kvstore_adapter` (wraps the new `kvstore::KvStore` lib).
- `WorkloadSpec` + uniform distribution + `FillRandom` + `ReadRandom` + `Overwrite`.
- `driver.rs` thread-pool executor with `Instant` latency capture.
- `report.rs` JSON output.
- `BTreeMap`-backed `MockEngine` + workload generator oracle tests.
- Verification: `cargo run -p kvbench -- --no-tui --spec specs/p0.toml` produces
  a JSON report; oracle tests pass; numbers are plausibly within an order of
  magnitude of the substrate microbench for the same op count.

**Phase 1 — full workload + distribution set, second engine.**
- Zipfian + `latest` generators + shape tests.
- Full YCSB A–F + remaining db_bench scenarios (`ReadSeq`, `ReadWhileWriting`,
  `DeleteRandom`/`DeleteSeq`, `SeekRandom`).
- `fjall` adapter behind `--features fjall`.
- HdrHistogram percentiles in the JSON report.
- Verification: side-by-side kvstore vs fjall JSON diff on the same spec
  shows sensible relative numbers (fjall should win on write-heavy LSM-shaped
  loads; kvstore should win or hold on point-read-heavy B+tree-shaped loads —
  if not, that itself is a substrate signal worth investigating).

**Phase 2 — TUI.**
- `ratatui` app: workload matrix, live run view, results table.
- `--no-tui` retained for batch / CI.
- Verification: TUI launches, selects a (kvstore, YcsbA, uniform, 10k, 1
  thread) run, displays live throughput + histogram, persists JSON on
  completion. Keyboard control tested manually (TUIs are not unit-testable in
  a meaningful way; document the manual test in the PR).

**Phase 3 — more engines + deletion workloads.**
- `redb` adapter (COW B+tree contrast).
- `sled` adapter.
- `rocksdb` adapter behind `--features rocksdb` (off by default).
- `DeleteRandom`/`DeleteSeq` used in a churn workload that explicitly measures
  deletion cost and post-delete reclamation.
- Verification: 5-engine comparison matrix on YcsbC read-only; adapter
  contract tests for each engine.

**Phase 4 — TTL expiry + comparison tooling.**
- TTL workload (insert with TTL, idle, measure re-arrival / reclamation).
- `compare` subcommand for two-report diff with pretty TUI table.
- Optional: substrate-metrics-side-channel — when `kvstore` is built with
  `metrics` on, surface its `fast-telemetry` counters alongside the engine
  stats (eviction rate, group-commit batch size, latch restart count).

## Open questions

1. **Workspace membership vs separate repo.** Recommendation: same workspace,
   not in `default-members`. Keeps the harness discoverable and the substrate
   build clean. Revisit if the external-engine dependency set causes CI
   pain.
2. **RocksDB default inclusion.** Recommendation: off by default (C++ build
   requirement), opt-in via `--features rocksdb`. The three pure-Rust engines
   (`fjall`, `redb`, `sled`) are sufficient for the first full comparison
   matrix.
3. **Engine option equity.** LSM-tree engines accept many tuning knobs
   (memtable size, compaction style, bloom bits); kvstore has fewer. The
   harness needs an "equitable default" policy: either run each engine at its
   documented defaults (simplest, defensible) or expose a small curated knob
   set per engine (richer, more work). Recommendation: defaults for phases
   0–3, curated knobs as an extension.
4. **Persistence of running config for reproducibility.** Each JSON report
   must contain the full `WorkloadSpec` + `EngineOpts` + git commit hash, so a
   reported number is reproducible from the report alone. Confirm this is
   sufficient vs requiring a make-like `specs/*.toml` file checked into the
   repo.

## Risks

- **External engines skewing the comparison through adapter overhead, not
  engine cost.** Mitigated by: trait is `&[u8]` in / `Option<Vec<u8>>` out
  only (no serialisation), `Instant` measured around the engine call not the
  op generation, load phase measured separately from run phase.
- **TUI latency under live updates contaminating measurements.** Mitigated
  by: live view renders from a stats snapshot read off the driver thread on an
  interval, never blocks the driver.
- **`kvstore` adapter not exercising the substrate's interesting paths.**
  Mitigated by: workload catalogue explicitly maps each YCSB/db_bench
  scenario to the substrate path it stresses (see db_bench table above); if a
  workload doesn't exercise a substrate path that matters, it's the wrong
  workload.
