#!/bin/bash
# Print the DwarfStar source identity embedded in mlx-serve build metadata.
# The explicit build option remains a release override; callers use this only
# to derive the truthful local checkout identity.
set -euo pipefail

DS4_REPO="${1:?usage: ds4-git-identity.sh <ds4-repo>}"
if ! DS4_REVISION="$(git -C "$DS4_REPO" rev-parse --short=12 HEAD 2>/dev/null)"; then
    exit 1
fi
[ -n "$DS4_REVISION" ]

if ! DS4_STATUS="$(git -C "$DS4_REPO" status --porcelain --untracked-files=normal 2>/dev/null)"; then
    exit 1
fi

if [ -n "$DS4_STATUS" ]; then
    printf '%s-dirty\n' "$DS4_REVISION"
else
    printf '%s\n' "$DS4_REVISION"
fi
