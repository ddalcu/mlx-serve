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

# Pinned Zig release (see build.zig's version-gate comptime block).
bash scripts/fetch-zig.sh
ZIG="$ROOT/.zig-toolchain/zig"

# libllama (llama.cpp GGUF engine) must be staged before the Zig build links it.
bash scripts/fetch-llama.sh

# Guest kernel + prebaked rootfs — the MAS bundle ships them in Resources/guest.
if [ ! -f lib/guest/kernel ] || [ ! -f lib/guest/rootfs.tar.gz ]; then
    bash scripts/fetch-guest-rootfs.sh
fi

VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' app/Info-MAS.plist)"

# DEVELOPER_DIR override: zig must see the CommandLineTools SDK, not Xcode's
# (same clash app/build.sh works around); a macOS upgrade can remove the CLT —
# fall back to the selected Xcode then (same as app/build.sh). ReleaseFast
# always — a Debug mlx-serve is 2-4x slower and must never ship (see CLAUDE.md).
ZIG_DEVELOPER_DIR=/Library/Developer/CommandLineTools
[ -d "$ZIG_DEVELOPER_DIR" ] || ZIG_DEVELOPER_DIR="$(xcode-select -p)"
DEVELOPER_DIR="$ZIG_DEVELOPER_DIR" "$ZIG" build -Doptimize=ReleaseFast -Dmas=true -Dversion="$VERSION"
DEVELOPER_DIR="$ZIG_DEVELOPER_DIR" "$ZIG" build vz-agent

echo "mlx-serve: $(du -h zig-out/bin/mlx-serve | cut -f1), vz-agent: $(du -h zig-out/guest/vz-agent | cut -f1)"
