#!/bin/bash
# Which CI jobs a change set needs. Reads changed paths on stdin, one per line,
# and prints GITHUB_OUTPUT lines: zig, swift, slow.
#
#   zig    the macOS Zig build and `zig build test` jobs
#   swift  the Swift app job
#   slow   the tool-traffic replay inside `zig build test`
#
# An empty list, or any path not named below, runs the job: the sets only name
# what is known not to reach it. push to main and workflow_dispatch never call
# this, they run everything.
set -euo pipefail

# Files no job reads.
IGNORED_RE='^(docs|website|demo|Formula|Casks)/|^[^/]+\.md$|^firebase\.json$|(^|/)(AGENTS|CLAUDE)\.md$'
# What the Swift tests read outside app/ (guard: tests/test_ci_changes.sh).
SWIFT_RE='^(app|skills|containers|scripts|\.github)/|^build\.zig(\.zon)?$|^src/plugins/[^/]+/(config|plugin)\.zig$|^src/plugins/(ltx|acestep)/|^src/host/media_jobs/|^src/assets/|^(LICENSE|LICENSE-APACHE-2\.0|NOTICE|database\.rules\.json|\.gitattributes)$'
SLOW_RE='^src/host/chat/|^src/fixtures/tool_traffic\.jsonl$'

paths=$(cat)
if [ -z "$(printf '%s' "$paths" | tr -d '[:space:]')" ]; then
  printf 'zig=true\nswift=true\nslow=true\n'
  exit 0
fi

live=$(printf '%s\n' "$paths" | grep -Ev "$IGNORED_RE" || true)

# true when any live path matches (or, with -v, fails to match) the pattern.
any() {
  if printf '%s\n' "$live" | grep "$@" | grep -q .; then echo true; else echo false; fi
}

printf 'zig=%s\n' "$(any -Ev '^app/')"
printf 'swift=%s\n' "$(any -E "$SWIFT_RE")"
printf 'slow=%s\n' "$(any -E "$SLOW_RE")"
