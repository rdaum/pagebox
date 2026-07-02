#!/usr/bin/env python3
"""Generate benchmark charts from bench-results JSON files."""

import json
import os
import glob
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "bench-results")
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
    "ycsb_c_evicting",
    "ycsb_c_small",
    "ycsb_a_oversize",
    "ycsb_a_evicting",
    "overwrite",
    "fillrandom",
]


def load_data():
    data = {}
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        basename = os.path.basename(f).removesuffix(".json")
        parts = basename.split("_")
        engine = parts[0]
        threads = int(parts[-1].removesuffix("t"))
        scenario = "_".join(parts[1:-1])
        with open(f) as fh:
            d = json.load(fh)
        run = d.get("run_phase", {})
        ops_per_sec = run.get("ops_per_sec", 0)
        key = (scenario, threads)
        if key not in data:
            data[key] = {}
        data[key][engine] = ops_per_sec
    return data


def plot_scenario(ax, data, scenario, title):
    for engine in ENGINES:
        x = []
        y = []
        for t in THREAD_COUNTS:
            v = data.get((scenario, t), {}).get(engine)
            if v is not None:
                x.append(t)
                y.append(v / 1e6)
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
    data = load_data()

    titles = {
        "ycsb_c_evicting": "YCSB-C Read (200K records, 32MB pool)",
        "ycsb_c_small": "YCSB-C Read (10K records, in-memory)",
        "ycsb_a_oversize": "YCSB-A Mixed R/W (50K records, 16MB pool)",
        "ycsb_a_evicting": "YCSB-A Mixed R/W (200K records, 32MB pool)",
        "overwrite": "Overwrite (50K records, in-place update)",
        "fillrandom": "FillRandom (50K records, bulk insert)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, scenario in enumerate(SCENARIOS):
        plot_scenario(axes[i], data, scenario, titles.get(scenario, scenario))

    fig.suptitle(
        "Pagebox kvstore vs competitors — ops/sec by thread count\n"
        "(uniform distribution, relaxed sync, commit aed7077)",
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
