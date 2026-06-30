#!/usr/bin/env bash
# Run benchmark scenarios in release build for all engines, then print a
# consolidated comparison table.
#
# Usage:
#   ./bench.sh              # run all scenarios, all engines
#   ./bench.sh ycsb_c_small # run one scenario, all engines
#   ./bench.sh --keep       # don't clear old results first
#
# Reports are written to ./bench-results/ as JSON.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECS_DIR="$SCRIPT_DIR/specs"
RESULTS_DIR="$SCRIPT_DIR/bench-results"
ENGINES=(kvstore betree fjall redb sled)

# Check for --keep flag.
KEEP_RESULTS=false
for arg in "$@"; do
    if [[ "$arg" == "--keep" ]]; then
        KEEP_RESULTS=true
    fi
done

# Scenarios to run (spec file basenames without .toml).
SCENARIOS=(
    ycsb_c_small
    ycsb_a_oversize
    fillrandom
    overwrite
)

# Filter out --keep from scenario args.
FILTERED_ARGS=()
for arg in "$@"; do
    if [[ "$arg" != "--keep" ]]; then
        FILTERED_ARGS+=("$arg")
    fi
done
if [[ ${#FILTERED_ARGS[@]} -gt 0 ]]; then
    SCENARIOS=("${FILTERED_ARGS[@]}")
fi

mkdir -p "$RESULTS_DIR"

# Clear stale results unless --keep was passed.
if [[ "$KEEP_RESULTS" == false ]]; then
    rm -f "$RESULTS_DIR"/*.json
fi

echo "=== Building kvbench (release, --all-features) ==="
cargo build -p kvbench --release --all-features
echo

BIN="target/release/kvbench"

run_one() {
    local spec="$1"
    local engine="$2"
    local spec_path="$SPECS_DIR/${spec}.toml"
    local report="$RESULTS_DIR/${engine}_${spec}.json"

    if [[ ! -f "$spec_path" ]]; then
        echo "  ERROR: spec file not found: $spec_path"
        return 1
    fi

    echo "  [$engine] $spec ..."
    "$BIN" run \
        --spec "$spec_path" \
        --output "$report" \
        --engine "$engine" \
        --tmpdir "$RESULTS_DIR/tmp" 2>&1 | sed 's/^/    /'
    echo
}

for spec in "${SCENARIOS[@]}"; do
    echo "=== Scenario: $spec ==="
    for engine in "${ENGINES[@]}"; do
        run_one "$spec" "$engine"
    done
done

echo "=== Consolidated Report ==="
"$BIN" summarize "$RESULTS_DIR"
echo

echo "=== Done. Reports in $RESULTS_DIR/ ==="
