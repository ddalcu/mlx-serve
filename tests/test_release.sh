#!/usr/bin/env bash
# Hermetic tests for release.sh — the CHANGELOG-version parse and the
# "only dispatch when versions match" gate. No network and no real release:
# release.sh is SOURCED (its main() doesn't auto-run), and the version
# computation + `gh` dispatch are overridden with stubs.
#
# Run: bash tests/test_release.sh
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

PASS=0; FAIL=0
ok() { # name  actual  expected
  if [ "$2" = "$3" ]; then PASS=$((PASS + 1)); echo "  PASS $1"
  else FAIL=$((FAIL + 1)); echo "  FAIL $1 — expected [$3], got [$2]"; fi
}

# Source the functions. main() is guarded by a BASH_SOURCE check, so this only
# defines functions; it does not trigger a release.
# shellcheck source=/dev/null
source "$ROOT/release.sh"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
DISPATCH_LOG="$TMP/dispatched.log"

# Stub the two externals so the gate can be exercised offline:
#  - computed_release_version → whatever the test pins via FAKE_COMPUTED
#  - gh → record the dispatch instead of calling GitHub
FAKE_COMPUTED="26.6.11"
computed_release_version() { echo "$FAKE_COMPUTED"; }
gh() { echo "DISPATCHED $*" >> "$DISPATCH_LOG"; }

dispatched() { [ -f "$DISPATCH_LOG" ] && echo yes || echo no; }
reset_dispatch() { rm -f "$DISPATCH_LOG"; }

echo "── changelog_top_version ──"
printf '# Changelog\n\n## v26.6.11 — Headline\n\n- a bullet\n\n## v26.6.10 — Older\n' > "$TMP/normal.md"
ok "parses the top entry's version"        "$(changelog_top_version "$TMP/normal.md")" "26.6.11"
printf '# Changelog\n\nnothing released yet\n'                                          > "$TMP/none.md"
ok "empty when no version heading present" "$(changelog_top_version "$TMP/none.md")"   ""

echo "── dispatch gate ──"
printf '## v26.6.11 — Match\n'    > "$TMP/match.md"
printf '## v26.6.10 — Stale\n'    > "$TMP/stale.md"

# Match → dispatches, exit 0.
reset_dispatch
( CHANGELOG="$TMP/match.md"; main -y ) >/dev/null 2>&1; rc=$?
ok "matching versions dispatch"     "$(dispatched)" "yes"
ok "matching versions exit 0"       "$rc"           "0"

# Mismatch → refuse, no dispatch, exit 1.
reset_dispatch
( CHANGELOG="$TMP/stale.md"; main -y ) >/dev/null 2>&1; rc=$?
ok "mismatched versions don't dispatch" "$(dispatched)" "no"
ok "mismatched versions exit 1"         "$rc"           "1"

# --dry-run on a match → no dispatch, exit 0.
reset_dispatch
( CHANGELOG="$TMP/match.md"; main --dry-run ) >/dev/null 2>&1; rc=$?
ok "dry-run never dispatches"       "$(dispatched)" "no"
ok "dry-run exit 0"                 "$rc"           "0"

# No CHANGELOG entry → refuse, exit 1.
reset_dispatch
( CHANGELOG="$TMP/none.md"; main -y ) >/dev/null 2>&1; rc=$?
ok "missing changelog entry refuses" "$(dispatched)" "no"
ok "missing changelog entry exit 1"  "$rc"           "1"

echo ""
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then echo "PASS $TOTAL/$TOTAL"; exit 0
else echo "FAIL $FAIL/$TOTAL"; exit 1; fi
