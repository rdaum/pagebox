#!/usr/bin/env bash
# Run benchmark scenarios in release build for all engines, across multiple
# thread counts, then print a consolidated comparison table.
#
# Usage:
#   ./bench.sh                        # all scenarios, all thread counts
#   ./bench.sh resident_ycsb_c_small    # one scenario, all thread counts
#   ./bench.sh resident_ycsb_c_small 8  # one scenario, one thread count
#   ./bench.sh --keep                 # don't clear old results first
#   ./bench.sh --cooldown-secs 2      # pause between engine processes
#   ./bench.sh --wal-backend io_uring  # override WAL backend (kvstore only)
#   ./bench.sh --wal-backend io_uring fillrandom 8
#   ./bench.sh direct_io_cache_pressure_ycsb_c 2  # opt-in diagnostic
#
# Reports are written to ./bench-results/ as JSON.
# Output files are named: ${engine}_${spec}_${threads}t.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECS_DIR="$SCRIPT_DIR/specs"
RESULTS_DIR="$SCRIPT_DIR/bench-results"
LOGS_DIR="$RESULTS_DIR/logs"
THREAD_COUNTS=(2 4 8 16)

# Parse args: separate --keep, --wal-backend, scenario names, and thread counts.
KEEP_RESULTS=false
WAL_BACKEND=""
COOLDOWN_SECS=1
SCENARIO_FILTER=()
THREAD_FILTER=()
FAILURES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep) KEEP_RESULTS=true; shift ;;
        --cooldown-secs) COOLDOWN_SECS="$2"; shift 2 ;;
        --cooldown-secs=*) COOLDOWN_SECS="${1#--cooldown-secs=}"; shift ;;
        --wal-backend) WAL_BACKEND="$2"; shift 2 ;;
        --wal-backend=*) WAL_BACKEND="${1#--wal-backend=}"; shift ;;
        *[0-9]*) THREAD_FILTER+=("$1"); shift ;;
        *) SCENARIO_FILTER+=("$1"); shift ;;
    esac
done

# Scenarios to run (spec file basenames without .toml).
SCENARIOS=(
    resident_ycsb_c_small
    resident_ycsb_a
    cache_pressure_ycsb_c
    cache_pressure_ycsb_a
    fillrandom
    overwrite
)

if [[ ${#SCENARIO_FILTER[@]} -gt 0 ]]; then
    SCENARIOS=("${SCENARIO_FILTER[@]}")
fi
if [[ ${#THREAD_FILTER[@]} -gt 0 ]]; then
    THREAD_COUNTS=("${THREAD_FILTER[@]}")
fi
if [[ ! "$COOLDOWN_SECS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: --cooldown-secs must be a non-negative number" >&2
    exit 2
fi

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

if [[ "$KEEP_RESULTS" == false ]]; then
    rm -f "$RESULTS_DIR"/*.json
    rm -f "$LOGS_DIR"/*.log
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
    local log="$LOGS_DIR/${engine}_${spec}_${threads}t.log"

    if [[ ! -f "$spec_path" ]]; then
        echo "  ERROR: spec file not found: $spec_path"
        return 1
    fi

    local wal_arg=()
    if [[ -n "$WAL_BACKEND" ]]; then
        wal_arg=(--wal-backend "$WAL_BACKEND")
    fi

    echo "  [$engine] $spec @ ${threads}t ..."
    local rc=0
    local statuses=()
    if ! timeout 300 "$BIN" run \
        --spec "$spec_path" \
        --output "$report" \
        --engine "$engine" \
        --tmpdir "$RESULTS_DIR/tmp" \
        --threads "$threads" \
        "${wal_arg[@]}" 2>&1 | tee "$log" | sed -u 's/^/    /'; then
        statuses=("${PIPESTATUS[@]}")
        for status in "${statuses[@]}"; do
            if [[ $status -ne 0 ]]; then
                rc=$status
                break
            fi
        done
    fi
    if [[ $rc -ne 0 ]]; then
        echo "    (run failed or timed out, rc=$rc)"
        echo "    log: $log"
        FAILURES+=("${engine}/${spec}/${threads}t: rc=$rc")
    fi
    echo
    if [[ "$COOLDOWN_SECS" != "0" ]]; then
        sleep "$COOLDOWN_SECS"
    fi
}

scenario_index=0
for spec in "${SCENARIOS[@]}"; do
    echo "=== Scenario: $spec ==="
    spec_path="$SPECS_DIR/${spec}.toml"
    if [[ ! -f "$spec_path" ]]; then
        echo "  ERROR: spec file not found: $spec_path"
        exit 1
    fi
    mapfile -t cohort < <("$BIN" cohort --spec "$spec_path")
    if [[ ${#cohort[@]} -eq 0 ]]; then
        echo "  ERROR: comparison cohort is empty: $spec_path"
        exit 1
    fi
    thread_index=0
    for threads in "${THREAD_COUNTS[@]}"; do
        echo "--- ${threads} threads ---"
        # Rotate the declared cohort deterministically across scenario/thread
        # configurations. Runs remain wholly sequential.
        shift_by=$(((scenario_index + thread_index) % ${#cohort[@]}))
        for ((i = 0; i < ${#cohort[@]}; i++)); do
            engine="${cohort[$(((i + shift_by) % ${#cohort[@]}))]}"
            run_one "$spec" "$engine" "$threads"
        done
        thread_index=$((thread_index + 1))
    done
    scenario_index=$((scenario_index + 1))
done

echo "=== Consolidated Report ==="
"$BIN" summarize "$RESULTS_DIR"
echo

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "=== Failed Runs (${#FAILURES[@]}) ===" >&2
    printf '  %s\n' "${FAILURES[@]}" >&2
    echo "Logs are in $LOGS_DIR/" >&2
    exit 1
fi

echo "=== Done. Reports in $RESULTS_DIR/ ==="
