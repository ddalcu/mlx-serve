#!/bin/bash
# Hermetic clean/dirty provenance regression: no model, GPU, or network.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP_REPO="$(mktemp -d)"
trap 'rm -rf "$TMP_REPO"' EXIT

git -C "$TMP_REPO" init -q
git -C "$TMP_REPO" config user.name "DS4 provenance test"
git -C "$TMP_REPO" config user.email "ds4-provenance@example.invalid"
printf 'baseline\n' > "$TMP_REPO/tracked.txt"
git -C "$TMP_REPO" add tracked.txt
git -C "$TMP_REPO" commit -qm baseline

EXPECTED="$(git -C "$TMP_REPO" rev-parse --short=12 HEAD)"
CLEAN="$(bash "$ROOT/scripts/ds4-git-identity.sh" "$TMP_REPO")"
[ "$CLEAN" = "$EXPECTED" ]

printf 'dirty\n' >> "$TMP_REPO/tracked.txt"
TRACKED_DIRTY="$(bash "$ROOT/scripts/ds4-git-identity.sh" "$TMP_REPO")"
[ "$TRACKED_DIRTY" = "$EXPECTED-dirty" ]

git -C "$TMP_REPO" restore tracked.txt
printf 'untracked\n' > "$TMP_REPO/untracked.txt"
UNTRACKED_DIRTY="$(bash "$ROOT/scripts/ds4-git-identity.sh" "$TMP_REPO")"
[ "$UNTRACKED_DIRTY" = "$EXPECTED-dirty" ]

# rev-parse does not read the index, but status does. A broken index therefore
# pins the partial-failure case that must never be misreported as clean.
[ "$(GIT_INDEX_FILE=/dev/null git -C "$TMP_REPO" rev-parse --short=12 HEAD)" = "$EXPECTED" ]
if GIT_INDEX_FILE=/dev/null git -C "$TMP_REPO" status --porcelain --untracked-files=normal >/dev/null 2>&1; then
    printf 'expected git status to fail with GIT_INDEX_FILE=/dev/null\n' >&2
    exit 1
fi
PARTIAL_STDOUT="$TMP_REPO/helper.stdout"
if GIT_INDEX_FILE=/dev/null bash "$ROOT/scripts/ds4-git-identity.sh" "$TMP_REPO" >"$PARTIAL_STDOUT" 2>/dev/null; then
    printf 'identity helper accepted a failed git status\n' >&2
    exit 1
fi
[ ! -s "$PARTIAL_STDOUT" ]

# Git state is not a declared Zig build input, so plain builds deliberately
# identify DS4 as unknown. Exercise three consecutive ReleaseFast builds
# against ONE isolated local/global cache: a healthy checkout, a status-only
# failure, then a healthy checkout again. The failure must never inherit a
# stale clean revision; recovery must remain conservative. A fourth build
# proves an explicit helper-derived pin is still carried into the binary. This
# is compile/link-only: no model, GPU inference, or network access.
ZIG_BIN="${ZIG:-$ROOT/.zig-toolchain/zig}"
if [ ! -x "$ZIG_BIN" ]; then
    ZIG_BIN="$(command -v zig || true)"
fi
if [ -z "$ZIG_BIN" ] || [ ! -x "$ZIG_BIN" ]; then
    printf 'test_ds4_git_identity: SKIP (Zig unavailable for cache regression)\n'
    exit 0
fi

CACHE_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_REPO" "$CACHE_ROOT"' EXIT
OUT="$CACHE_ROOT/out"
LOCAL_CACHE="$CACHE_ROOT/local-cache"
GLOBAL_CACHE="$CACHE_ROOT/global-cache"

build_and_identity() {
    local expected="$1"
    local mode="$2"
    shift 2
    case "$mode" in
        clean)
            ZIG_GLOBAL_CACHE_DIR="$GLOBAL_CACHE" \
                "$ZIG_BIN" build -Doptimize=ReleaseFast -p "$OUT" --cache-dir "$LOCAL_CACHE" "$@" >/dev/null
            ;;
        broken-index)
            GIT_INDEX_FILE=/dev/null ZIG_GLOBAL_CACHE_DIR="$GLOBAL_CACHE" \
                "$ZIG_BIN" build -Doptimize=ReleaseFast -p "$OUT" --cache-dir "$LOCAL_CACHE" "$@" >/dev/null
            ;;
        *)
            printf 'unknown build mode: %s\n' "$mode" >&2
            exit 1
            ;;
    esac
    ACTUAL="$(DYLD_LIBRARY_PATH="$ROOT/lib/mlx/lib:$ROOT/lib/llama/lib" "$OUT/bin/mlx-serve" --version | awk '$1 == "ds4" { print $2; exit }')"
    [ "$ACTUAL" = "$expected" ]
}

ROOT_IDENTITY="$(bash "$ROOT/scripts/ds4-git-identity.sh" "$ROOT/lib/ds4")"
build_and_identity "unknown" clean
build_and_identity "unknown" broken-index
build_and_identity "unknown" clean
build_and_identity "$ROOT_IDENTITY" clean "-Dds4-commit=$ROOT_IDENTITY"

printf 'test_ds4_git_identity: PASS\n'
