# `pagebox-wal`

Write-ahead log format, append path, and durability coordination.

## Role

`pagebox-wal` owns the log records and I/O path used to make page and logical
changes durable. It provides append, group commit, sync, recovery scanning, and
WAL benchmarking support without depending on the storage crate.

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
  WRITE→FSYNC SQE chains, a shared in-flight slab, fd-based completion
  dispatch, and a dedicated CQE reaper thread.
- `src/bin/profile_wal.rs` profiles WAL throughput and latency.
- `benches/` contains WAL benchmarks.

## Used By

- `pagebox-storage` for storage-level durability integration.
- `kvstore` for recovery, strict/relaxed commits, and checkpoint/reset.

## Uses

- `pagebox-frame-kernel` for page/frame identifiers included in storage records.
- `pagebox-threading` for background flush and worker support.
