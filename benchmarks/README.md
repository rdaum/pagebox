# Microbenchmark workflow

Pagebox has 11 Micromeasure suites across the B+tree, storage, WAL, and hybrid-latch crates. Run a suite from the
workspace root. Add a substring after `--` to select a group or case.

- B+tree: `btree`, `microbtreebench`, `neworder_churn`, and `ycsb`.
- Storage: `microbufferpoolevictbench`, `microbufferpoolfaultbench`, `microfreepageallocatorbench`,
  `microslottedpagebench`, and `microstableswipbench`.
- WAL: `microwalbench`.
- Hybrid latch: `latchbench`.

```sh
cargo bench -p pagebox-storage --bench microslottedpagebench -- append_dense_reset
```

## Measurement configuration

All Micromeasure suites use the same measurement configuration. The defaults collect the full CPU PMU profile, do
not request RAPL energy, and let Micromeasure automatically probe for memory-controller bandwidth counters.

| Variable | Values | Default | Effect |
|---|---|---|---|
| `PAGEBOX_BENCH_PMU` | `full`, `compact`, `none` | `full` | Select the full CPU counter set, the four-counter compact profile, or timing only. |
| `PAGEBOX_BENCH_RAPL` | `off`, `package`, `package-core` | `off` | Add system-wide package energy, optionally with core-domain energy. |
| `PAGEBOX_BENCH_MEMORY_BANDWIDTH` | `auto`, `requested`, `off` | `auto` | Probe quietly, request IMC bandwidth explicitly, or skip the probe. |

Use the compact profile when the full event group must multiplex. For example:

```sh
PAGEBOX_BENCH_PMU=compact \
PAGEBOX_BENCH_MEMORY_BANDWIDTH=off \
cargo bench -p pagebox-btree --bench microbtreebench -- lookup_hot
```

RAPL and IMC measurements are system-wide. They include unrelated work on the same package or memory controller.
Use a quiet machine and samples long enough to dominate measurement noise.

Concurrent suites retain Micromeasure's per-worker PMU collection. Their reports identify the PMU scope as
`managed_workers`. The selected RAPL and IMC settings apply to the whole sample.

## Reports and comparisons

The common Micromeasure launcher writes versioned reports and supports explicit context, output, and baseline paths:

```sh
MICROMEASURE_CONTEXT_FILE=benchmark-context.json \
MICROMEASURE_BASELINE=artifacts/baseline/slotted-page.json \
MICROMEASURE_OUTPUT=artifacts/current/slotted-page.json \
cargo bench -p pagebox-storage --bench microslottedpagebench
```

The context file records comparison dimensions such as the commit, compiler, and host configuration. An explicit
invalid or incompatible baseline fails instead of silently selecting another local result.
