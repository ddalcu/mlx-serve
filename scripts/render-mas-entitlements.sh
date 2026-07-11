#!/bin/bash
# Usage: render-mas-entitlements.sh <src.entitlements> <team-id> <bundle-id> <out>
set -euo pipefail

SRC="$1"
TEAM_ID="$2"
BUNDLE_ID="$3"
OUT="$4"

cp "$SRC" "$OUT"
/usr/libexec/PlistBuddy -c "Add :com.apple.application-identifier string ${TEAM_ID}.${BUNDLE_ID}" "$OUT"
/usr/libexec/PlistBuddy -c "Add :com.apple.developer.team-identifier string ${TEAM_ID}" "$OUT"
