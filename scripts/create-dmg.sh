#!/bin/bash
# Creates a DMG installer with drag-to-Applications for MLX Core.
# Usage: scripts/create-dmg.sh <path-to-app> [output.dmg]
#
# Self-contained (hdiutil + a tolerant Finder AppleScript) on purpose: the
# Homebrew `create-dmg` tool's generated AppleScript sets `statusbar visible ...
# to false`, which Finder REJECTS on macOS 26 (-10006) and aborts BEFORE
# finalizing the DMG, failing the build. We drop that one cosmetic line, keep the
# icon layout + drag-to-Applications, and make ALL Finder styling non-fatal — the
# .dmg is produced regardless of whether the (purely cosmetic) styling succeeds.
set -euo pipefail

APP_PATH="${1:?Usage: $0 <path/to/MLX Core.app> [output.dmg]}"
DMG_PATH="${2:-MLXCore.dmg}"
VOLNAME="MLX Core"

if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: $APP_PATH not found"
    exit 1
fi

APP_NAME="$(basename "$APP_PATH")"
rm -f "$DMG_PATH"

# Stage the contents: the app + a symlink to /Applications (drag-to-install).
STAGE="$(mktemp -d)"
MOUNT_DIR=""
cleanup() {
    [ -n "$MOUNT_DIR" ] && [ -d "$MOUNT_DIR" ] && hdiutil detach "$MOUNT_DIR" -force >/dev/null 2>&1 || true
    rm -rf "$STAGE"
    [ -n "${TMP_DMG:-}" ] && rm -f "$TMP_DMG" || true
}
trap cleanup EXIT

cp -R "$APP_PATH" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

# A stale volume of the same name blocks the mount — detach it first.
[ -d "/Volumes/$VOLNAME" ] && hdiutil detach "/Volumes/$VOLNAME" -force >/dev/null 2>&1 || true

# Create a writable image sized to the contents (+200 MB slack).
SIZE_MB=$(( $(du -sm "$STAGE" | cut -f1) + 200 ))
TMP_DMG="$(mktemp -u).dmg"
hdiutil create -srcfolder "$STAGE" -volname "$VOLNAME" -fs HFS+ \
    -format UDRW -size "${SIZE_MB}m" "$TMP_DMG" >/dev/null

# Mount and apply cosmetic styling (icon view, window size, icon positions).
# NOTE: no `set statusbar visible` — that is the line Finder rejects on macOS 26.
# The whole block is non-fatal: a Finder hiccup must not fail the build.
MOUNT_DIR="/Volumes/$VOLNAME"
hdiutil attach "$TMP_DMG" -mountpoint "$MOUNT_DIR" -nobrowse -noautoopen >/dev/null
if ! osascript <<EOF 2>/dev/null
tell application "Finder"
    tell disk "$VOLNAME"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set the bounds of container window to {200, 120, 860, 520}
        set theViewOptions to the icon view options of container window
        set arrangement of theViewOptions to not arranged
        set icon size of theViewOptions to 128
        set position of item "$APP_NAME" of container window to {160, 190}
        set position of item "Applications" of container window to {500, 190}
        update without registering applications
        delay 1
        close
    end tell
end tell
EOF
then
    echo "  (DMG window styling skipped — cosmetic only; image is still valid)"
fi
sync
hdiutil detach "$MOUNT_DIR" -force >/dev/null
MOUNT_DIR=""

# Compress to the final read-only DMG.
hdiutil convert "$TMP_DMG" -format UDZO -imagekey zlib-level=9 -o "$DMG_PATH" >/dev/null

[ -f "$DMG_PATH" ] || { echo "ERROR: DMG was not created"; exit 1; }
echo "DMG created: $DMG_PATH ($(du -h "$DMG_PATH" | cut -f1))"
