#!/bin/bash
# Creates a DMG installer with drag-to-Applications for MLX Core.
# Usage: scripts/create-dmg.sh <path-to-app> [output.dmg]
set -euo pipefail

APP_PATH="${1:?Usage: $0 <path/to/MLX Core.app> [output.dmg]}"
DMG_PATH="${2:-MLXCore.dmg}"

if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: $APP_PATH not found"
    exit 1
fi

rm -f "$DMG_PATH"

# create-dmg exits 2 when it can't set a custom background (we don't use one).
# The DMG is still created correctly, so treat exit 2 as success.
create-dmg \
    --volname "MLX Core" \
    --window-pos 200 120 \
    --window-size 660 400 \
    --icon-size 128 \
    --icon "MLX Core.app" 160 190 \
    --app-drop-link 500 190 \
    --no-internet-enable \
    --hdiutil-quiet \
    "$DMG_PATH" \
    "$APP_PATH" \
    || test $? -eq 2

echo "DMG created: $DMG_PATH ($(du -h "$DMG_PATH" | cut -f1))"
