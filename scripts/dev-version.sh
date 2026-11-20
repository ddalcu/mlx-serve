#!/bin/bash
# The version a build that cuts no release stamps: the newest release tag (vYY.M.N) behind HEAD, so the engine and the
# plugin kit the app fetches for it match the source; the plist's version when the checkout has no such tag.
# Usage: dev-version.sh <repo> <Info.plist>
set -euo pipefail
tag="$(git -C "$1" describe --tags --abbrev=0 --match 'v[0-9]*.[0-9]*.[0-9]*' --exclude '*-*' 2>/dev/null || true)"
if [[ "$tag" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
else
    /usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$2"
fi
