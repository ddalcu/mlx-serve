#!/usr/bin/env bash
# Zig tests per file, source-scan tests (a test that @embedFile's Zig source), env-gated model tests and their names.
#   scripts/test_census.sh [repo-root]
set -euo pipefail
cd "${1:-$(dirname "$0")/..}"
exec python3 scripts/census.py tests
