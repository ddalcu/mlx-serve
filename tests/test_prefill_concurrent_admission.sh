#!/bin/bash
# HTTP lifecycle sanity; deterministic memory-hold regression lives in scheduler.zig.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO/tests/_lib_models.sh"
MODEL=${PREFILL_TEST_MODEL:-$(find_model mlx-community/Qwen3.5-0.8B-MLX-4bit || true)}
if [[ ! -f "$MODEL/config.json" ]]; then
    echo 'SKIP prefill HTTP sanity: set PREFILL_TEST_MODEL to a small MLX model.'
    exit 0
fi
MODEL=$(cd "$MODEL" && pwd)
BINARY=${MLX_SERVE_BINARY:-$REPO/zig-out/bin/mlx-serve}
[[ -x "$BINARY" ]] || { echo "FAIL missing binary: $BINARY"; exit 1; }
WORK_ROOT=${PREFILL_TEST_WORK:-$REPO/tmp}
mkdir -p "$WORK_ROOT"
SCRATCH=$(mktemp -d "$WORK_ROOT/prefill-http.XXXXXX")
echo "HTTP artifacts: $SCRATCH"
python3 "$REPO/tests/prefill_concurrent_admission.py" \
    --binary "$BINARY" --model "$MODEL" --port "${1:-11385}" --scratch "$SCRATCH"
