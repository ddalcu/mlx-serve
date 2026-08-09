#!/usr/bin/env bash
# Checked-in app resources must not carry the extended attributes that break
# code signing.
#
# An asset dragged out of Finder (a screen recording off the Desktop, an
# exported image) arrives with com.apple.FinderInfo attached. codesign then
# refuses the ENTIRE bundle:
#
#   MLX Core.app: resource fork, Finder information, or similar detritus not allowed
#
# — three minutes into a build, and naming the BINARY rather than the file that
# actually brought the metadata in, which sends you looking in the wrong place.
# build.sh strips them from the staged bundle so a build can't fail this way;
# this keeps the repo copies clean so the problem is caught in a second, with
# the offending file named and the fix printed.
#
# ONLY the attributes that actually break signing are checked. com.apple.macl
# and com.apple.provenance are added by macOS itself whenever a sandboxed app
# touches a file — every existing resource carries them and always has, and
# they sign fine. Failing on those would make this script fail forever through
# no fault of anyone's, and a guard that cries wolf gets deleted.
set -uo pipefail

cd "$(dirname "$0")/.."
RESOURCES="app/Sources/MLXServe/Resources"
BLOCKERS="com.apple.FinderInfo com.apple.ResourceFork com.apple.quarantine"

if [ ! -d "$RESOURCES" ]; then
    echo "FAIL: $RESOURCES is missing"
    exit 1
fi

fail=0
while IFS= read -r file; do
    attrs=$(xattr "$file" 2>/dev/null)
    for blocker in $BLOCKERS; do
        if echo "$attrs" | grep -qx "$blocker"; then
            echo "FAIL: $file carries $blocker"
            fail=1
        fi
    done
done < <(find "$RESOURCES" -type f ! -name '.gitkeep')

if [ "$fail" -ne 0 ]; then
    echo
    echo "Strip them with:  xattr -c <file>"
    echo "codesign rejects the whole bundle otherwise."
    exit 1
fi

echo "PASS: no app resource carries a signing-blocking extended attribute"
