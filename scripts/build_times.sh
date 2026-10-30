#!/usr/bin/env bash
# Clean ReleaseFast build time and peak compiler memory, then the rebuild time after a one-line edit in each named
# file (the edit is reverted). Run on a quiet machine: nothing else compiling or on the GPU.
#   scripts/build_times.sh [file ...]          e.g. scripts/build_times.sh src/engines/mlx/transformer.zig src/sdk/util/log.zig
# Env: ZIG (default .zig-toolchain/zig), ZIG_ARGS (extra `zig build` args, e.g. -Dstrip=true),
#      OUT (default ~/claude-tmp/build-times/<timestamp>). One line per build goes to stdout and $OUT/results.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
ZIG="${ZIG:-$PWD/.zig-toolchain/zig}"
OUT="${OUT:-$HOME/claude-tmp/build-times/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"

# build <label>: one timed ReleaseFast build; wall seconds and the peak RSS over the build's processes.
build() {
    local log="$OUT/$1.log" s e
    s=$(python3 -c 'import time; print(time.time())')
    # shellcheck disable=SC2086
    /usr/bin/time -l "$ZIG" build -Doptimize=ReleaseFast --summary all ${ZIG_ARGS:-} >"$log" 2>&1 || {
        echo "$1: build FAILED, see $log" | tee -a "$OUT/results.txt"; return 1; }
    e=$(python3 -c 'import time; print(time.time())')
    awk -v l="$1" -v w="$(python3 -c "print(round($e - $s, 1))")" -v sz="$(stat -f %z zig-out/bin/mlx-serve)" \
        '/maximum resident set size/ { printf "%-40s wall=%ss peak_rss=%.1fGB bin=%.1fMB\n", l, w, $1 / 2^30, sz / 1e6 }' \
        "$log" | tee -a "$OUT/results.txt"
}

echo "$(git rev-parse --short HEAD) $(uname -m) $(sysctl -n machdep.cpu.brand_string) zig=$("$ZIG" version) ${ZIG_ARGS:-}" \
    | tee "$OUT/results.txt"
rm -rf .zig-cache zig-out
build clean
for f in "$@"; do
    cp "$f" "$OUT/orig"
    trap 'cp "$OUT/orig" "$f"' EXIT
    echo "// build_times.sh one-line edit $(date +%s)" >>"$f"
    build "edit-$(basename "$f")" || true
    cp "$OUT/orig" "$f"
    trap - EXIT
done
echo "logs: $OUT"
