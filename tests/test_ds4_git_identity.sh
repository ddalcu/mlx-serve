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

printf 'test_ds4_git_identity: PASS\n'
