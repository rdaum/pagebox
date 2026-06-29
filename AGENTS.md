# AGENTS.md

Quick-start context for AI coding agents. Detailed contribution rules and crate docs live in
[README.md](./README.md).

## Project Overview

Pagebox is a LeanStore/Umbra-influenced storage substrate for data systems, written in Rust: buffer pool, page store,
WAL, and concurrent B+tree. Not a database, SQL engine, or row/MVCC layer — the substrate those are built on.

Extracted from an unpublished OLTP database ("Boxter"). `kvstore` is the reference composition binary (four substrate
crates + `clap`), proving the substrate composes with no database-level types in scope.

Substrate components: swizzled-pointer buffer pool (anonymous-`mmap` `MAP_NORESERVE` reservation, resident-budget
eviction); file-backed page store (sharded free-page allocator, 4K/64K classes, header-resident user-meta slots);
WAL (group commit, configurable sync backends, relaxed/strict commit modes, streaming replay, crash recovery);
concurrent B+tree (swizzled references, hybrid latching, B-link right-sibling chase, ordered/prefix scans, reopen
recovery via user-meta slots); hybrid optimistic/shared/exclusive latch; SWIP hot/cool/evicted state machine on a
64-bit word; slotted pages (front-grown slots, back-grown heap, implicit compaction, reserved suffixes); Linux-aware
threading with CPU topology detection; optional `fast-telemetry` metrics (`--no-default-features` compiles them out).

## Repository Policy

- Follow [README.md](./README.md) for the substrate contract and crate boundaries.

For **now** because we are pre-1.0:
- Do not add backward-compatibility code for on-disk formats, page layouts, WAL records, manifest entries, or
  user-meta slot semantics unless the user explicitly asks for it.
- Treat persistent format work as unreleased: prefer one current format over migration scaffolding. If a format
  change is needed, update the current code path directly; older local data is unsupported. The `kvstore` example
  simply reinitializes.
- Same goes for APIs and refactorings. Do not build boilerplate "bridge" or "legacy" adapters unless prompted.

## Empirical Process And Test Discipline

Every change is an experiment. State the hypothesis, run it, report the result honestly — including the null result.

- **Start from evidence, not intuition.** Read the implementation and existing tests around a hot path before editing.
  Run the unmodified tests and benches so you have a real baseline; if you cannot get one, say so. Performance and
  correctness claims without a measured before/after are guesses.
- **Reproduce failures before fixing them.** A fix not anchored to a reproducing observation will be wrong in a
  different way than the bug. A flaky failure is a concurrency-bug signal — prefer a loom model that enumerates the
  race over a stress run that happens to trip it.
- **Distinguish tests from benchmarks.** A test asserts an invariant; a benchmark measures a quantity. A benchmark
  with no correctness assertion is a measurement, not a test — do not retcon it with `assert!(true)` and `#[test]`.
  When a benchmark exercises a hard concurrent path that should be asserted (concurrent splits, eviction-during-
  traversal, group-commit batching, direct-IO replay), write a separate `#[test]` driving the same scenario with
  explicit invariants rather than bloating the harness. A stress run asserting only "did not panic" is a smoke test,
  not a correctness test — do not propagate the pattern.
- **Write tests that would fail if the code were wrong.** Ask which bug a new test catches that no existing test
  catches; if the answer is "none", do not add it. Trivial construction smoke, tautological bit-pattern roundtrips,
  harness wrappers, and happy-path single-insert-then-lookup are not contributions.
- **Prefer behavioural oracles over happy paths.** Reach for `BTreeMap`/model differential tests, `#[should_panic]`
  contract tests, and loom enumerations first. For larger concurrency state spaces, write a stress test with a real
  invariant assertion (post-hoc scan, uniqueness check, count) — never "did not panic".
- **Keep the suite honest.** No temporary debugging tests, `dbg!` prints, or `assert!(true)` scaffolding in
  committed code. Delete tests added to investigate a specific failure if they are not permanent regression guards.
  Delete weaker tests subsumed by a stronger property test.

## Agentic Code of Conduct

AI coding agents are acceptable tools here, but generated output must meet the same standard as any human-written
contribution.

- Do not submit or propose code you cannot explain from repository evidence.
- Verify technical claims against the codebase, tests, docs, or primary sources (the LeanStore and Umbra papers are
  the primary sources for Pagebox's design).
- Avoid vague marketing language, filler, and generic AI prose in documentation and comments.
- Write factual, specific commit messages, PR descriptions, issue comments, and code review comments. Follow
  conventional commit format.
- Be explicit about uncertainty, test coverage, and remaining risks.
- Keep changes clean, organized, and narrowly scoped to the task. State exactly what was verified and what was not.

The standard is directed engineering work with human accountability, not blind acceptance of generated output.

Agents produce evidence-based, empirical results when making changes:

- Run relevant unit, integration, and `#[should_panic]` contract tests.
- Run `cargo clippy --workspace --all-targets` for code changes unless there is a clear reason not to, and report
  that reason.
- Run `cargo fmt --all` and ensure the result is a no-op before declaring a change ready.
- Use the [micro]benchmarks when changing performance-sensitive paths in the B+tree, buffer pool, slotted
  page, free-page allocator, or WAL. Measure before and after; do not rely on intuition for performance claims.
- For concurrency changes, prefer loom tests where they exist; add a new loom model rather than a flaky stress test
  when the invariant is small enough to enumerate.

Architectural changes should follow Pagebox's LeanStore-influenced hybrid in-memory design: keep the hot working set
memory-fast, preserve disk-backed scale, respect swizzled-pointer buffer-pool and page-residency invariants
(SWIP hot/cool/evicted transitions, hybrid latch optimistic-restart, referenced-bit second-chance, no-steal for
dirty B-e pages, stable-parent-link invalidation), and do not replace those assumptions with generic cache/database
patterns without evidence and explicit maintainer direction. Do not introduce dependencies beyond
`parking_lot`, `libc`, `crc-fast`, `crossbeam-queue`, and (optionally) `fast-telemetry` without explicit justification
— minimal outward coupling is a deliberate property of the substrate.

## Repository Structure

Cargo workspace (resolver v3, Rust edition 2024, MSRV 1.95). Each hot-path crate exposes a `metrics` feature (on by
default); disabled, the crate pulls zero telemetry dependencies and no-op shims take their place.

```text
crates/
├── kvstore/           # Example durable KV store binary (proves composition)
├── btree/             # Concurrent B+tree: swizzled pointers, hybrid latching, ordered scans
├── storage/           # Buffer pool, page store, slotted pages, free-page allocator, buffer frames
├── wal/               # WAL: format, append path, group commit, sync, replay, telemetry
├── hybrid-latch/      # Optimistic/shared/exclusive latch primitive (loom-tested)
├── threading/         # Linux-aware thread spawning, CPU topology detection, optional pinning
├── swip-kernel/       # Swizzled-pointer word representation and atomic state transitions (loom-tested)
└── frame-kernel/      # Page ID, frame state, and LSN types shared by storage, WAL, and tree code
```

Internal dependency DAG (kept deliberately narrow):

```text
frame-kernel   (0 deps)
swip-kernel    (0 deps)
threading      (libc)
hybrid-latch   (parking_lot; + optional fast-telemetry)
wal            -> frame-kernel, threading
storage        -> frame-kernel, hybrid-latch, swip-kernel, threading, wal
btree          -> hybrid-latch, storage, swip-kernel
kvstore        -> btree, storage, wal, frame-kernel  (+ clap)
```

## Pagebox-Specific Commands

Feature flags and metrics:

- `--features metrics` (default): enables `fast-telemetry` in `pagebox-wal`, `pagebox-storage`, `pagebox-btree`,
  and `pagebox-hybrid-latch`.
- `--no-default-features`: no-op telemetry shims, zero telemetry dependencies.
- Downstream embedders propagate the feature:
  ```toml
  [features]
  default = ["metrics"]
  metrics = ["pagebox-storage/metrics", "pagebox-wal/metrics", "pagebox-btree/metrics"]
  ```

WAL runtime configuration (environment variables, read in `pagebox-wal`):

- `PAGEBOX_WAL_DIRECT_IO=1`: open WAL segments with `O_DIRECT` and aligned writes (Linux-gated).
- `PAGEBOX_WAL_SYNC_BACKEND=fdatasync|pwritev2_dsync`: select the durable-sync backend.
- `PAGEBOX_WAL_GROUP_COMMIT_DELAY_MAX_US`, `PAGEBOX_WAL_GROUP_COMMIT_TARGET_RECORDS`: group-commit batching knobs.

Tests, runs, and benchmarks:

```bash
cargo test -p <crate>                     # any pagebox-* crate
cargo run -p kvstore -- put hello world   # full recovery sequence on open
cargo run -p kvstore -- get hello

# Loom-gated modules (pagebox-storage, pagebox-hybrid-latch); Miri likewise where applicable
RUSTFLAGS="--cfg loom" cargo test -p pagebox-hybrid-latch

# Workspace-wide lint and format (must be a no-op before declaring a change ready)
cargo clippy --workspace --all-targets
cargo fmt --all
```

Benches live per crate under `benches/`. Two harnesses are in use: `criterion` (statistical reporting, regression
detection, HTML reports; used when fancy charts and out-of-the-box statistical analysis are wanted) and `micromeasure`
(Linux-first; reports timing/throughput plus PMU-derived metrics — instructions retired, branch misses, cache misses
— together in one run, with persisted sample data for immediate last-run-vs-this-run comparison). `micromeasure` is
used on paths where co-observing latency and PMU metrics is the point — tiny operations where a change in instruction
count, branch behaviour, or cache-miss pattern is the signal. Run with `cargo bench -p <crate> --bench <name>`; pick
the harness the existing bench uses.

## Style Reminders

- Rust 2024 edition. Minimum Rust version is 1.95. Workspace resolver is v3. Lints are declared per-crate under
  `[lints.rust]` (e.g. `unexpected_cfgs` for `cfg(loom)`, `cfg(miri)`, `cfg(metrics)`); keep these in sync when
  introducing new cfgs.
- Use default `rustfmt` settings and `cargo fmt --all`.
- Organize imports in three groups: standard library, external crates, then internal `pagebox-*` crates. Keep all
  `use` statements at the top of the file/module; avoid per-function imports.
- Names describe what code does, not implementation details or history.
- Avoid deep nesting. Prefer early returns, `let else`, match guards, and match let-chains so the success path stays
  visible and failure cases are handled up front.
- Define custom errors with `Display` and `std::error::Error` where appropriate. On the storage and WAL hot paths,
  prefer infallible fast paths with explicit panic-on-invariant-violation guards (see `#[should_panic]` tests in the
  audit) over viral `Result` propagation when the failure is a genuine internal-corruption case.
- Add tests for new functionality and regressions; prefer real logic over mocked behaviour, and property/differential
  tests against a `BTreeMap` / model oracle over another happy-path smoke test. Use descriptive assertion messages.
- Use Canadian English spellings in docs and comments.
- Do not add dependencies beyond `parking_lot`, `libc`, `crc-fast`, `crossbeam-queue`, and (optionally)
  `fast-telemetry` without explicit justification; minimal outward coupling is a deliberate property of the substrate.

## Performance And Storage Invariants

The B+tree traversal, buffer pool fix/evict, slotted-page insert/scan, free-page allocator, and WAL append/sync
paths are all hot. Prefer low or zero-copy solutions (page bytes are accessed in place through buffer-frame slices;
do not copy them out unless an API boundary requires ownership — the B+tree `lookup_with` callback variant exists
specifically to avoid allocating on the read path). Avoid unnecessary allocations. Follow cache-friendly patterns
(slotted pages keep the slot array and key/value heap on the same 4 KiB page; do not introduce indirection that
defeats this). Use benchmark evidence for optimization claims; measure before and after.

Key patterns and invariants (preserved across changes):

- **SWIP state machine**: page references live in a 64-bit word encoding hot/cool/evicted states with a page ID.
  Atomic CAS drives transitions; failed CAS returns the live word (the contract optimistic-restart relies on).
  Mixed-state CAS transitions and hot→evict direct paths are subtle.
- **Hybrid latching**: optimistic readers take a snapshot, do their work, then `validate`. A version step on
  exclusive acquire invalidates in-flight optimistic guards. Upgrades from optimistic to shared/exclusive must
  restart if the version moved. The shared/exclusive compatibility matrix is non-trivial — see
  `pagebox-hybrid-latch/src/latch.rs::mod loom_tests`.
- **Buffer pool residency**: a resident budget limits how many frames are hot; eviction reclaims via
  referenced-bit second-chance (RandomSecondChance) or batch clock (BatchClock). Dirty B-e pages are no-steal:
  `try_evict_*` must refuse them until `flush_dirty_batch` cleans them. Stable parent-link pages (e.g. B-tree
  root) must not be evicted. Pin count is incremented on `fix` and decremented on `unfix`/drop; the pool panics
  (`buffer pool exhausted`) when every frame is pinned and a new fix is requested.
- **Free-page allocator**: sharded, with central and per-shard caches. Reusable (promoted) extents are consumed
  before monotonic growth. Adjacent same-class extents coalesce for larger-class allocations. Retired large pages
  split into smaller pages. Page IDs are never reused across reopen until the WAL checkpoint advances past them.
- **WAL durability**: `flush_at_least(lsn)` blocks under strict commit mode and returns the target LSN under
  relaxed mode without waiting. `recover(&store, checkpoint_lsn, read_page_lsn)` replays page images and patch
  records in LSN order, skipping records at or below the page's embedded LSN (idempotent recovery). Checkpoint
  advances the checkpoint LSN and resets the WAL.
- **Slotted pages**: slot array grows from the front, key/value heap grows from the back. `compactify` reclaims
  garbage between them. `copy_key_value_range` is the split primitive used by B-tree node splits. Reserved suffixes
  survive compaction. The page panics on overflow (`slotted page full`) — document overflow contracts with
  `#[should_panic]` rather than silently growing.
- **B+tree traversal**: `find_leaf_*` uses pin → validate → (possibly) re-validate because a child can be evicted
  between snapshot and pin. The `LOOKUP_OPTIMISTIC_RESTART_LIMIT` loop bounds restarts. `should_chase_right` follows
  B-link siblings when a split is in progress. Splits publish to parent via `publish_leaf_split_to_parent` with a
  retry budget. `owned_page_ids` and `find_parent` are the recovery and reachability primitives; reopen reads root
  page ID and height from `user_meta_0` / `user_meta_1`.

## Code Review Checklist

1. Does the change follow the import ordering and naming style?
2. Are errors surfaced through appropriate error types, or — on genuinely invariant-violation hot paths — documented
   with `#[should_panic]` tests?
3. Are new behaviours covered by tests or a clear explanation of why not?
4. Are tests and assertions descriptive? For stateful invariants, is there a `BTreeMap` / model oracle differential
   rather than a single-key happy path?
5. Is the code formatted with `cargo fmt --all` (no-op)?
6. Does `cargo clippy --workspace --all-targets` pass, or is there a stated reason it was not run?
7. Does the change avoid deep nesting and unnecessary abstraction?
8. Does any performance claim have a criterion benchmark behind it (before and after)?
9. Does the change preserve Pagebox's LeanStore-influenced storage assumptions: SWIP state transitions, hybrid-latch
   optimistic restart, buffer-pool residency and no-steal dirty B-e pages, stable parent-link, idempotent WAL
   recovery, slotted-page layout invariants?
10. Does the change introduce a new dependency? If so, is the justification explicit? The closed dependency set is
    `parking_lot`, `libc`, `crc-fast`, `crossbeam-queue`, `fast-telemetry` (optional), plus dev-deps
    `clap`, `criterion`, `loom`, `micromeasure`, `paste`, `proptest`, `tempfile`.
