#!/usr/bin/env bash
# Every MLX_SERVE_* name in the Zig we compile: read sites (the quoted name, `"MLX_SERVE_X"`), every mention, and
# the files reading it; then the other environment names the code reads with getenv.
#   scripts/levers.sh [repo-root]
set -euo pipefail
cd "${1:-$(dirname "$0")/..}"
DIRS=(src)
echo "== MLX_SERVE_* names: read sites, mentions, files reading it"
rows=$({
    grep -rnoE --include='*.zig' '"MLX_SERVE_[A-Z0-9_]+"' "${DIRS[@]}" | sed 's/^/R:/'
    grep -rnoE --include='*.zig' 'MLX_SERVE_[A-Z0-9_]+' "${DIRS[@]}" | sed 's/^/M:/'
} | awk -F: '
    { n = $4; gsub(/"/, "", n) }
    $1 == "M" { m[n]++ }
    $1 == "R" { r[n]++; if (index(f[n], $2) == 0) f[n] = f[n] (f[n] ? "," : "") $2 }
    END {
        for (n in m) printf "%4d %4d  %s  %s\n", r[n], m[n], n, (f[n] ? f[n] : "-")
    }' | sort -k3)
echo "$rows"
echo "$rows" | awk '{ t++; if ($1 > 0) k++ } END { printf "%d names, %d with a read site, %d mentioned only (comments, logs, built names)\n", t, k, t - k }'
echo
echo "== other environment names read with getenv(\"...\") (tests included): reads, name"
grep -rhoE --include='*.zig' 'getenv\("[A-Za-z0-9_]+"\)' "${DIRS[@]}" | sed -E 's/getenv\("(.*)"\)/\1/' \
    | grep -v '^MLX_SERVE_' | sort | uniq -c | sort -nr
