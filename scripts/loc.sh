#!/usr/bin/env bash
# Zig lines per file and folder (production vs test) for src/ and the absorbed libs, files over the 1,500-line cap,
# ModelConfig fields and predicates, model-identity checks per file, kernel text inside Zig strings.
#   scripts/loc.sh [repo-root]
set -euo pipefail
cd "${1:-$(dirname "$0")/..}"
exec python3 scripts/census.py loc
