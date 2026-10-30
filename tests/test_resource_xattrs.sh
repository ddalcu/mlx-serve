#!/usr/bin/env bash
# test_resource_xattrs.sh — checked-in app/Sources/MLXServe/Resources/* carry none of the extended attributes that break
# code signing (com.apple.FinderInfo, resource forks, quarantine): an asset dragged out of Finder brings them, and
# codesign then refuses the WHOLE bundle three minutes into a build, naming the binary instead of the file.
# com.apple.macl / com.apple.provenance are ignored on purpose: macOS adds them to any file a sandboxed app touches,
# every resource has them, and they sign fine. Hermetic (one `xattr` per file); a failure names the file and the fix.
#
#   ./tests/test_resource_xattrs.sh

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
