# `pagebox-threading`

Thread spawning, CPU topology detection, and optional pinning helpers.

## Role

`pagebox-threading` centralizes Linux-aware thread creation and CPU pinning. It
detects heterogeneous CPU topology, classifies performance and efficient cores,
and provides helpers used by storage, WAL, table, runtime, and workloads.

## Major Pieces

- `src/lib.rs` contains topology detection, core classification, per-class
  round-robin selection, thread spawning, and direct pinning helpers.
- `ThreadClass` names the supported placement classes: performance, worker,
  efficient, and unpinned.
- `spawn_perf`, `spawn_worker_perf`, `spawn_efficient`, and `spawn_with_class`
  create named threads with optional pinning.
- `detect_performance_cores`, `pin_current_thread_to_core`, and
  `pin_current_thread_to_class` expose lower-level controls.

## Runtime Controls

- `PAGEBOX_ENABLE_THREAD_PINNING=1` enables pinning.
- Without the environment variable, or on unsupported systems, helpers fall back
  to normal unpinned threads.

## Used By

- `pagebox-storage`, `pagebox-wal`, `pagebox-table`, and `pagebox-runtime` for
  background/service work.
- `pagebox-tpcc-esque` for workload worker placement.

## Uses

- No other Boxter crates.
