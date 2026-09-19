#!/bin/bash
# The Swift suite, ~30s on an M4 Max once built.
#
# One process, deliberately not `swift test --parallel`: SwiftPM's parallel
# mode launches a fresh xctest process per TEST, ~3500 of them, and that
# launch cost now outweighs the parallelism (measured 2026-09-18: 30s serial
# vs 49s parallel). It also kept tripping over the classes that share
# `UserDefaults.standard`, which a single process never does.
set -euo pipefail
cd "$(dirname "$0")"
swift test "$@"
