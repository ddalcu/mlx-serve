#!/usr/bin/env bash
# Exercise the native ds4 store with graph payload I/O replaced by deterministic fixtures.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

"${CC:-clang}" -std=gnu99 -O1 -g -fsanitize=address,undefined \
  -DMLXSERVE_DS4_KV_CACHE_TEST -I"$root/lib/ds4" -I"$root/src" \
  "$root/src/ds4_kv_cache.c" "$root/lib/ds4/ds4_kvstore.c" \
  -o "$scratch/kv-store-test"
mkdir "$scratch/cache"
"$scratch/kv-store-test" "$scratch/cache"
