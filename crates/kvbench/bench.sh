#!/usr/bin/env bash
# Run benchmark scenarios in release build for all engines, across multiple
# thread counts, then print a consolidated comparison table.
#
# Usage:
#   ./bench.sh                        # all scenarios, all thread counts
#   ./bench.sh ycsb_c_small           # one scenario, all thread counts
#   ./bench.sh ycsb_c_small 8         # one scenario, one thread count
#   ./bench.sh --keep                 # don't clear old results first
#   ./bench.sh --wal-backend io_uring  # override WAL backend (kvstore only)
#   ./bench.sh --wal-backend io_uring fillrandom 8
#
# Reports are written to ./bench-results/ as JSON.
# Output files are named: ${engine}_${spec}_${threads}t.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECS_DIR="$SCRIPT_DIR/specs"
RESULTS_DIR="$SCRIPT_DIR/bench-results"
ENGINES=(kvstore fjall redb rocksdb lmdb)
THREAD_COUNTS=(2 4 8 16)

# Parse args: separate --keep, --wal-backend, scenario names, and thread counts.
KEEP_RESULTS=false
WAL_BACKEND=""
SCENARIO_FILTER=()
THREAD_FILTER=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep) KEEP_RESULTS=true; shift ;;
        --wal-backend) WAL_BACKEND="$2"; shift 2 ;;
        --wal-backend=*) WAL_BACKEND="${1#--wal-backend=}"; shift ;;
        *[0-9]*) THREAD_FILTER+=("$1"); shift ;;
        *) SCENARIO_FILTER+=("$1"); shift ;;
    esac
done

# Scenarios to run (spec file basenames without .toml).
SCENARIOS=(
    ycsb_c_small
    ycsb_a_oversize
    ycsb_c_evicting
    ycsb_a_evicting
    fillrandom
    overwrite
)

if [[ ${#SCENARIO_FILTER[@]} -gt 0 ]]; then
    SCENARIOS=("${SCENARIO_FILTER[@]}")
fi
if [[ ${#THREAD_FILTER[@]} -gt 0 ]]; then
    THREAD_COUNTS=("${THREAD_FILTER[@]}")
fi

mkdir -p "$RESULTS_DIR"

if [[ "$KEEP_RESULTS" == false ]]; then
    rm -f "$RESULTS_DIR"/*.json
fi

echo "=== Building kvbench (release, --all-features) ==="
cargo build -p kvbench --release --all-features
echo
if [[ -n "$WAL_BACKEND" ]]; then
    echo "=== WAL backend: $WAL_BACKEND ==="
    echo
fi

BIN="target/release/kvbench"

run_one() {
    local spec="$1"
    local engine="$2"
    local threads="$3"
    local spec_path="$SPECS_DIR/${spec}.toml"
    local report="$RESULTS_DIR/${engine}_${spec}_${threads}t.json"

    if [[ ! -f "$spec_path" ]]; then
        echo "  ERROR: spec file not found: $spec_path"
        return 1
    fi

    local wal_arg=()
    if [[ -n "$WAL_BACKEND" ]]; then
        wal_arg=(--wal-backend "$WAL_BACKEND")
    fi

    echo "  [$engine] $spec @ ${threads}t ..."
    timeout 300 "$BIN" run \
        --spec "$spec_path" \
        --output "$report" \
        --engine "$engine" \
        --tmpdir "$RESULTS_DIR/tmp" \
        --threads "$threads" \
        "${wal_arg[@]}" 2>&1 | sed 's/^/    /'
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "    (run failed or timed out, rc=$rc)"
    fi
    echo
}

for spec in "${SCENARIOS[@]}"; do
    echo "=== Scenario: $spec ==="
    for threads in "${THREAD_COUNTS[@]}"; do
        echo "--- ${threads} threads ---"
        for engine in "${ENGINES[@]}"; do
            run_one "$spec" "$engine" "$threads"
        done
    done
done

echo "=== Consolidated Report ==="
"$BIN" summarize "$RESULTS_DIR"
echo

echo "=== Done. Reports in $RESULTS_DIR/ ==="
