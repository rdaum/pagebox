#!/usr/bin/env python3
"""Generate benchmark charts from bench-results JSON files."""

import json
import os
import glob
import sys
import statistics

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "bench-results")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

ENGINES = ["kvstore", "lmdb", "redb", "fjall", "rocksdb"]
ENGINE_COLORS = {
    "kvstore": "#e41a1c",
    "lmdb": "#377eb8",
    "redb": "#4daf4a",
    "fjall": "#984ea3",
    "rocksdb": "#ff7f00",
}
ENGINE_MARKERS = {
    "kvstore": "o",
    "lmdb": "s",
    "redb": "^",
    "fjall": "D",
    "rocksdb": "v",
}
THREAD_COUNTS = [2, 4, 8, 16]
SCENARIOS = [
    "cache_pressure_ycsb_c",
    "resident_ycsb_c_small",
    "resident_ycsb_a",
    "cache_pressure_ycsb_a",
    "overwrite",
    "fillrandom",
]


def load_data():
    data = {}
    cohorts = {}
    sources = set()
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        basename = os.path.basename(f).removesuffix(".json")
        parts = basename.split("_")
        engine = parts[0]
        has_iteration = parts[-1].startswith("r")
        thread_part = parts[-2] if has_iteration else parts[-1]
        threads = int(thread_part.removesuffix("t"))
        scenario_end = -2 if has_iteration else -1
        scenario = "_".join(parts[1:scenario_end])
        with open(f) as fh:
            d = json.load(fh)
        if d.get("schema") != 3:
            raise RuntimeError(f"unsupported report schema in {f}")
        run = d.get("run_phase", {})
        ops_per_sec = run.get("ops_per_sec", 0)
        source = (d.get("git_commit", "unknown"), d.get("binary_hash", "unknown"))
        sources.add(source)
        key = (scenario, threads)
        expected = tuple(d.get("comparison", {}).get("engines", []))
        if key in cohorts and cohorts[key] != expected:
            raise RuntimeError(f"conflicting cohorts for {scenario} at {threads} threads")
        cohorts[key] = expected
        if key not in data:
            data[key] = {}
        data[key].setdefault(engine, []).append(ops_per_sec)
    if len(sources) > 1:
        raise RuntimeError("bench-results mixes executable builds; start with a clean result set")

    for key in list(data):
        expected = set(cohorts[key])
        counts = [len(data[key].get(engine, [])) for engine in expected]
        if set(data[key]) != expected or len(set(counts)) != 1:
            del data[key]

    source_label = "no reports"
    if sources:
        commit, binary_hash = next(iter(sources))
        source_label = f"{commit}/{binary_hash[:8]}"
    if not data:
        raise RuntimeError("no complete, balanced comparison")
    return data, source_label


def plot_scenario(ax, data, scenario, title):
    for engine in ENGINES:
        x = []
        y = []
        for t in THREAD_COUNTS:
            samples = data.get((scenario, t), {}).get(engine)
            if samples:
                x.append(t)
                y.append(statistics.median(samples) / 1e6)
        if x:
            line, = ax.plot(
                x, y,
                marker=ENGINE_MARKERS[engine],
                color=ENGINE_COLORS[engine],
                linewidth=2,
                markersize=7,
            )
            line.set_label(engine)
    ax.set_xlabel("Threads")
    ax.set_ylabel("ops/sec (millions)")
    ax.set_title(title)
    ax.set_xticks(THREAD_COUNTS)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)


def main():
    data, source_label = load_data()

    titles = {
        "cache_pressure_ycsb_c": "YCSB-C (65K x 2 KiB, 64 MiB app cache)",
        "direct_io_cache_pressure_ycsb_c": "YCSB-C direct I/O (65K x 2 KiB, 64 MiB app cache)",
        "resident_ycsb_c_small": "YCSB-C (10K records, resident)",
        "resident_ycsb_a": "YCSB-A (50K records, resident)",
        "cache_pressure_ycsb_a": "YCSB-A (65K x 2 KiB, 64 MiB app cache)",
        "direct_io_cache_pressure_ycsb_a": "YCSB-A direct I/O (65K x 2 KiB, 64 MiB app cache)",
        "overwrite": "Overwrite (50K records, in-place update)",
        "fillrandom": "FillRandom (50K records, bulk insert)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, scenario in enumerate(SCENARIOS):
        plot_scenario(axes[i], data, scenario, titles.get(scenario, scenario))

    fig.suptitle(
        "Pagebox kvstore comparison — median point-operation throughput\n"
        f"(relaxed sync; application-cache pressure does not control OS cache; {source_label})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "benchmark_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
