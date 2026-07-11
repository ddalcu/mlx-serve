#!/bin/bash
# Xcode PRE-BUILD phase for project (app/project.yml):
# build the Zig server + guest agent and stage the engine/guest artifacts.
# Mirrors app/build.sh phases 1–2 (MAS branch) — keep the two in sync.
#
# Runs inside Xcode's script environment (ENABLE_USER_SCRIPT_SANDBOXING=NO);
# Xcode's PATH has no Homebrew, hence the explicit prefix.
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

command -v zig >/dev/null 2>&1 || { echo "error: zig not found — brew install zig" >&2; exit 1; }

# libllama (llama.cpp GGUF engine) must be staged before the Zig build links it.
bash scripts/fetch-llama.sh

# Guest kernel + prebaked rootfs — the MAS bundle ships them in Resources/guest.
if [ ! -f lib/guest/kernel ] || [ ! -f lib/guest/rootfs.tar.gz ]; then
    bash scripts/fetch-guest-rootfs.sh
fi

VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' app/Info-MAS.plist)"

# DEVELOPER_DIR override: zig must see the CommandLineTools SDK, not Xcode's
# (same clash app/build.sh works around). ReleaseFast always — a Debug
# mlx-serve is 2-4x slower and must never ship (see CLAUDE.md).
DEVELOPER_DIR=/Library/Developer/CommandLineTools zig build -Doptimize=ReleaseFast -Dmas=true -Dversion="$VERSION"
DEVELOPER_DIR=/Library/Developer/CommandLineTools zig build vz-agent

echo "mlx-serve: $(du -h zig-out/bin/mlx-serve | cut -f1), vz-agent: $(du -h zig-out/guest/vz-agent | cut -f1)"
