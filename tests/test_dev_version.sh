#!/bin/bash
# test_dev_version.sh — a build that cuts no release (FAST_DEV, ad-hoc) reports the newest release tag behind HEAD,
# so `mlx-serve --version`, and the plugin kit the app fetches for it, match the source; the plist only without a tag.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
plist="$tmp/Info.plist"
/usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string 26.10.3" "$plist" >/dev/null
fail=0
check() { if [ "$2" = "$3" ]; then :; else echo "FAIL $1: got '$2', want '$3'"; fail=1; fi; }
repo="$tmp/repo"; git init -q "$repo"
c() { git -C "$repo" -c user.name=t -c user.email=t@t commit -q --allow-empty -m "$1"; }
c one
check "no tag" "$(bash "$ROOT/scripts/dev-version.sh" "$repo" "$plist")" "26.10.3"
git -C "$repo" tag v26.11.1; c two; git -C "$repo" tag v26.11.2; c three
check "newest release tag behind HEAD" "$(bash "$ROOT/scripts/dev-version.sh" "$repo" "$plist")" "26.11.2"
git -C "$repo" tag v26.11.3-pre-release.1
check "a pre-release tag is not a release" "$(bash "$ROOT/scripts/dev-version.sh" "$repo" "$plist")" "26.11.2"
git -C "$repo" tag not-a-version
check "other tags are ignored" "$(bash "$ROOT/scripts/dev-version.sh" "$repo" "$plist")" "26.11.2"
exit $fail
