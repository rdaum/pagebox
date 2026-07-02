# `pagebox-wal`

Write-ahead log format, append path, and durability coordination.

## Role

`pagebox-wal` owns the log records and IO path used to make table/storage changes
durable. It provides append, group commit, sync, recovery scanning, and WAL
benchmarking support for the runtime and lower storage layers.

## Major Pieces

- `src/format.rs` defines WAL record layout, record kinds, headers, checksums,
  and encode/decode helpers.
- `src/wal_impl.rs` implements append, buffering, flush, sync, group commit,
  replay iteration, and telemetry.
- `src/io.rs` and `src/aligned_buf.rs` contain file IO and aligned-buffer
  support.
- `src/backend.rs` defines the `WalIoBackend` trait (the completion-based seam
  between the driver loop and the I/O path) and the synchronous backends
  (`fdatasync`, `pwritev2_dsync`).
- `src/io_uring.rs` (Linux) implements the io_uring backend: hand-defined
  kernel uapi structs, `IORING_SETUP_NO_MMAP` ring creation, linked
  WRITE→FSYNC SQE chains, and CQE-driven completion reaping.
- `src/bin/profile_wal.rs` profiles WAL throughput and latency.
- `benches/` contains WAL benchmarks.

## Used By

- `pagebox-runtime` for database durability, checkpoint/replay, and recovery.
- `pagebox-table` for table-level mutation logging.
- `pagebox-storage` for storage-level durability integration.
- `pagebox-server` and `pagebox-tpcc-esque` for hosted and workload telemetry.

## Uses

- `pagebox-frame-kernel` for page/frame identifiers included in storage records.
- `pagebox-threading` for background flush and worker support.
- `pagebox-storage` for page-oriented integration points.
