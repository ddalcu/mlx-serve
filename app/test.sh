#!/bin/bash
# The Swift suite in ~24s instead of ~55s.
#
# `swift test --parallel` runs each test CLASS in its own PROCESS, and a
# handful of classes read and write `UserDefaults.standard` — the real app
# domain, shared by every worker — so in parallel they read each other's
# writes and go red (measured 2026-09-10: ModelRootsTests reading another
# worker's model root, TranscriptTypographyTests reading its text size).
# Everything else parallelises cleanly, so those classes run serially after.
set -uo pipefail
cd "$(dirname "$0")"

SHARED_STATE='ModelRootsTests|ToolModelRootsTests|TranscriptTypographyTests|VoiceModeControllerTests|AgentWorkspaceDefaultTests'
FAIL=0

echo "=== parallel ==="
swift test --parallel --skip "$SHARED_STATE" || FAIL=1
echo "=== serial (shared UserDefaults) ==="
swift test --filter "$SHARED_STATE" || FAIL=1

[ "$FAIL" = 0 ] && echo "=== suite green ===" || echo "=== suite FAILED ==="
exit "$FAIL"
