#!/bin/bash
# scripts/ci-changes.sh decides which CI jobs a PR runs: the classification, and
# that the Swift job still runs for every repo file the Swift tests read.
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0
check() { # name, expected "zig swift slow", paths...
  local name=$1 want=$2; shift 2
  local got
  got=$(printf '%s\n' "$@" | bash scripts/ci-changes.sh | sed 's/^[a-z]*=//' | tr '\n' ' ' | sed 's/ $//')
  if [ "$got" != "$want" ]; then echo "FAIL $name: want [$want] got [$got]"; fail=1; fi
}

check "app only"            "false true false"  app/Sources/MLXServe/Views/ChatView.swift
check "zig only"            "true false false"  src/engines/mlx/scheduler.zig
check "chat touches replay" "true false true"   src/host/chat/tools/qwen.zig
check "docs only"           "false false false" docs/reference.md README.md src/plugins/inkling2/AGENTS.md
check "app docs"            "false false false" app/CLAUDE.md docs/gotchas/app.md
check "plugin config"       "true true false"   src/plugins/llama/config.zig
check "ci workflow"         "true true false"   .github/workflows/ci.yml
check "script"              "true true false"   scripts/build-mlx.sh
check "build graph"         "true false false"   build/macos.zig
check "lib"                 "true false false"  lib/jinja_cpp/jinja.cpp
check "app + zig"           "true true false"   app/Package.swift src/main.zig
check "empty list runs all" "true true true"    ""

# Every repo file a Swift test names outside app/ must keep the Swift job on.
tracked=$(mktemp); named=$(mktemp)
trap 'rm -f "$tracked" "$named"' EXIT
git ls-files | grep -v '^app/' | sort > "$tracked"
grep -rohE '"[A-Za-z0-9_./-]+"' app/Tests | tr -d '"' | sort -u | comm -12 - "$tracked" > "$named"
while IFS= read -r path; do
  got=$(printf '%s\n' "$path" | bash scripts/ci-changes.sh | grep '^swift=')
  case "$path" in
    docs/*|website/*|*.md) continue ;;
  esac
  if [ "$got" != "swift=true" ]; then echo "FAIL a Swift test names $path but a change to it skips the Swift job"; fail=1; fi
done < "$named"

[ "$fail" = 0 ] && echo "ci-changes: ok ($(wc -l < "$named" | tr -d ' ') Swift-test inputs checked)"
exit "$fail"
