#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_NAME="MLX Core"
BUNDLE_ID="com.dalcu.mlx-core"

# FAST_DEV=1 trades everything a RELEASE needs for iteration speed: an
# incremental (non-WMO) Swift build, an in-place bundle update, no icon
# regeneration, no framework restage/re-sign, no notarization, no DMG. Every
# deviation below is gated on this one variable and the unset path is the
# release path byte for byte — a fast build is for running the app on this Mac,
# never for shipping one. Guarded by tests/test_release_workflow_gates.sh.
FAST_DEV="${FAST_DEV:-0}"

# ZIG_DEBUG=1 builds mlx-serve -Doptimize=Debug. It is a SECOND lever rather
# than something FAST_DEV implies, because it costs more than it looks like:
# a Debug engine decodes 2–4× slower (root CLAUDE.md "Building"), so any
# latency read off that app is fiction, and Zig keys its cache on the optimize
# mode — alternating with a release build is a full rebuild each way, which is
# the opposite of fast. Worth it when the thing you are debugging is Zig-side
# (safety checks on, real panic traces, `-freference-trace` output), not when
# you are iterating on Swift. FAST_DEV-only: a Debug engine must never reach a
# bundle anyone else runs, and FAST_DEV is what stops before the shipping
# artifacts — so asking for one without it is refused BY NAME rather than
# quietly ignored.
ZIG_DEBUG="${ZIG_DEBUG:-0}"
if [ "$ZIG_DEBUG" = "1" ] && [ "$FAST_DEV" != "1" ]; then
    echo "ERROR: ZIG_DEBUG=1 is a FAST_DEV-only lever — a shipped mlx-serve is always ReleaseFast."
    echo "       Re-run as: FAST_DEV=1 ZIG_DEBUG=1 bash app/build.sh"
    exit 1
fi

# Signing identity from env (set in ~/.zshrc or CI). Unset = ad-hoc ("-"), so
# anyone can build without an Apple Developer account; a release sets the real
# identity (release.yml does).
IDENTITY="${APPLE_DEVELOPER_ID:--}"
TEAM_ID="${APPLE_TEAM_ID:--}"
if [ "$IDENTITY" = "-" ]; then
    echo "→ APPLE_DEVELOPER_ID not set — building with ad-hoc signing (dev build, not distributable)"
fi

cd "$SCRIPT_DIR"

# Preflight: make sure the pinned submodules (lib/ds4, lib/mlx-src, lib/mlxc-src)
# are actually checked out at their pins before anything links against them. A
# drifted engine submodule otherwise fails the Zig build with a cryptic
# missing-file error (or compiles against a mismatched struct ABI). --fix snaps
# a clean drift back to the pin; local edits inside a submodule are left alone.
# Runs in fast dev too: it is a no-op on a clean tree, and the failure it
# catches is exactly the one that wastes an iteration.
bash "$PROJECT_ROOT/scripts/check-submodules.sh" --fix

MAS="${MAS:-0}"
if [ "$MAS" = "1" ]; then
    echo "=== $MAS ==="
    INFO_PLIST_SRC="$SCRIPT_DIR/Info-MAS.plist"
    APP_ENTITLEMENTS="$SCRIPT_DIR/MLXCore-MAS.entitlements"
    HELPER_ENTITLEMENTS="$SCRIPT_DIR/mlx-serve-MAS.entitlements"
    SWIFT_MODE_FLAGS=(-Xswiftc -DMAS_BUILD)
    ZIG_MODE_FLAGS=(-Dmas=true)
else
    INFO_PLIST_SRC="$SCRIPT_DIR/Info.plist"
    APP_ENTITLEMENTS="$SCRIPT_DIR/MLXCore.entitlements"
    HELPER_ENTITLEMENTS=""            # DMG helper carries no entitlements
    SWIFT_MODE_FLAGS=()
    ZIG_MODE_FLAGS=()
fi

# Set calver version (YY.M.N) — auto-increment N from last GitHub release.
# The month comes from an explicitly pinned zone, matching the one in
# .github/workflows/release.yml: CI runners are UTC, so an evening release
# here lands in next month there (v26.8.1 was cut at 21:26 local on Jul 31
# against a CHANGELOG that said 26.7.12). Guarded by
# tests/test_release_workflow_gates.sh.
YM="$(TZ=America/New_York date +%y.%-m)"
# Only a build that could BECOME a release asks GitHub what has been published.
# A fast build cuts nothing, and neither does an ad-hoc one: `gh` is release
# tooling, not a build dependency, and someone who clones the repo to compile
# the app must not need a GitHub login to do it. The signing identity is the
# discriminator because it is already the flag deciding hardened runtime,
# notarization and productbuild below — no identity, no release, no `gh`.
# Nothing in CI reaches this: release.yml mirrors this script step by step and
# runs its own `gh release list`.
if [ "$FAST_DEV" = "1" ] || [ "$IDENTITY" = "-" ]; then
    # Reuse whatever version the plist already carries. It is only read back by
    # UpdateChecker, which has nothing to compare against on a dev build.
    CALVER="$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$INFO_PLIST_SRC" 2>/dev/null || echo "${YM}.0")"
else
    # `gh` is the only thing that knows what has already been published, so its
    # failure is not defaultable. It used to end in `|| echo "0"`, which turns a
    # missing login or an offline laptop into v<YY.M>.1 — BEHIND every release
    # already cut that month, which is the same UpdateChecker dead end as an
    # unstamped bundle from the other direction. An empty month still yields 0
    # through the jq's own `// 0`; only the COMMAND failing stops the build.
    if ! LAST_N=$(gh release list --limit 50 --json tagName --jq "[.[] | .tagName | select(startswith(\"v${YM}.\"))] | map(split(\".\")[2] | tonumber) | max // 0" 2>/dev/null); then
        echo "ERROR: \`gh release list\` failed, so the next CalVer number is unknown."
        echo "       Defaulting it would stamp v${YM}.1, behind what is already published."
        echo "       Fix it (\`gh auth login\`), or build for yourself: FAST_DEV=1 bash app/build.sh"
        exit 1
    fi
    NEXT_N=$((LAST_N + 1))
    CALVER="${YM}.${NEXT_N}"
fi
# The version is stamped into the BUNDLE below, never here. This block used to
# PlistBuddy the TRACKED app/Info.plist, which dirties the working tree on every
# build: release.yml's Homebrew step rebases and cannot take a dirty tree, and a
# contributor who merely builds the app ends up with a modified file they never
# touched. Not stamping at all is the other half of the same line and has
# already shipped — v26.8.1's DMG contained an app reporting 26.7.12, so
# UpdateChecker offered an update that installing could never satisfy, forever.
# Stamping the bundle copy is what release.yml does; both sides are pinned by
# tests/test_release_workflow_gates.sh.
export MLX_SERVE_VERSION="$CALVER"

if [ "$FAST_DEV" = "1" ]; then
    echo "=== Building MLX Core v$CALVER (FAST_DEV — not shippable) ==="
else
    echo "=== Building MLX Core v$CALVER ==="
fi

# ── Phase 1: Build Swift app ──
echo "→ Compiling Swift..."
# SwiftPM owns each target's language mode. MLXCore's 5.9 manifest keeps the app
# in Swift 5 mode, while modern dependencies (swift-sdk 0.12.x, SwaTex 0.5.x)
# compile in the mode their own manifests declare. A global `-swift-version 5`
# override reaches the whole graph and makes SwaTex's Swift 6.1 source invalid.
#
# The configuration is the single biggest lever in this script: `-c release`
# turns on Whole Module Optimization, so touching one file recompiles the whole
# module (and SwiftUI's type-checker is what makes that minutes, not seconds —
# see the "a view expression can stop the compiler finishing" rule). `-c debug`
# compiles incrementally. Debug and release keep SEPARATE build dirs, so
# alternating between them costs a full rebuild of whichever you left.
if [ "$FAST_DEV" = "1" ]; then
    SWIFT_CONFIG=debug
else
    SWIFT_CONFIG=release
fi
SWIFT_BUILD_FLAGS=(-c "$SWIFT_CONFIG" ${SWIFT_MODE_FLAGS[@]+"${SWIFT_MODE_FLAGS[@]}"})
# SwaTexRender's generated `Bundle.module` looks for its KaTeX fonts beside the
# .app, which codesign refuses to seal — so it can never find them in a bundle
# we assemble by hand and traps on the first equation (issue #233). Patch the
# lookup before compiling; the script is idempotent and fails loudly.
swift package resolve
bash "$PROJECT_ROOT/scripts/patch-swatex-font-lookup.sh" "$SCRIPT_DIR/.build/checkouts/SwaTex"
swift build "${SWIFT_BUILD_FLAGS[@]}" 2>&1 | tail -5
SWIFT_BIN_DIR="$(swift build "${SWIFT_BUILD_FLAGS[@]}" --show-bin-path)"
SWIFT_BIN="$SWIFT_BIN_DIR/MLXCore"
SWATEX_RESOURCE_BUNDLE="$SWIFT_BIN_DIR/SwaTex_SwaTexRender.bundle"
if [ ! -f "$SWIFT_BIN" ]; then
    echo "ERROR: Swift build failed"
    exit 1
fi
if [ ! -d "$SWATEX_RESOURCE_BUNDLE" ]; then
    echo "ERROR: SwaTex font resource bundle missing"
    exit 1
fi
echo "  Swift binary: $(du -h "$SWIFT_BIN" | cut -f1)"

# ── Phase 2: Build mlx-serve (Zig) ──
echo "→ Building mlx-serve (Zig)..."
cd "$PROJECT_ROOT"
# Stage libllama (llama.cpp GGUF engine) before the Zig build links against it.
echo "→ Fetching libllama..."
bash "$PROJECT_ROOT/scripts/fetch-llama.sh"
# Build the pinned mlx + mlx-c submodules into lib/mlx (NAX kernels enabled —
# the brew bottle ships without them). Idempotent: no-op when the stage
# matches the pinned SHAs. Needs full Xcode (Metal Toolchain), so it runs
# BEFORE the CLT-forced zig build below.
echo "→ Building mlx + mlx-c (pinned submodules)..."
bash "$PROJECT_ROOT/scripts/build-mlx.sh"
# Pinned Zig nightly (homebrew's `zig` formula still ships 0.16.0, which no
# longer builds — see build.zig's version-gate comptime block).
bash "$PROJECT_ROOT/scripts/fetch-zig.sh"
ZIG="$PROJECT_ROOT/.zig-toolchain/zig"
# The zig link resolves the SDK via xcrun. Prefer the CommandLineTools SDK
# (historical default), but a macOS upgrade can remove the CLT entirely —
# fall back to the selected Xcode then (same SDK CI links against).
ZIG_DEVELOPER_DIR=/Library/Developer/CommandLineTools
[ -d "$ZIG_DEVELOPER_DIR" ] || ZIG_DEVELOPER_DIR="$(xcode-select -p)"
# Engine-version pins surfaced by `mlx-serve --version` (parsed by the app so
# Settings can show engine versions without booting the server — src/version.zig).
# MLX + ggml self-report at runtime; these three have no runtime API:
MLXC_VERSION="$(git -C "$PROJECT_ROOT/lib/mlxc-src" describe --tags --always 2>/dev/null)"
DS4_COMMIT="$(git -C "$PROJECT_ROOT/lib/ds4" rev-parse --short HEAD 2>/dev/null)"
LLAMA_TAG="$(cat "$PROJECT_ROOT/lib/llama/.version" 2>/dev/null)"
# ReleaseFast unless ZIG_DEBUG asked otherwise (see the lever's comment at the
# top). Even under FAST_DEV the default stays ReleaseFast: Zig's cache already
# makes an unchanged rebuild a few seconds, so the Swift phase above is the
# real cost of an iteration, and the Debug arm buys nothing for Swift-side work.
ZIG_OPT=(-Doptimize=ReleaseFast)
if [ "$ZIG_DEBUG" = "1" ]; then
    ZIG_OPT=(-Doptimize=Debug)
    echo "  (ZIG_DEBUG=1 — mlx-serve built Debug: 2-4x slower decode, never read a latency off it)"
fi
DEVELOPER_DIR="$ZIG_DEVELOPER_DIR" "$ZIG" build "${ZIG_OPT[@]}" -Dversion="$MLX_SERVE_VERSION" \
  -Dmlx-c-version="${MLXC_VERSION:-unknown}" -Dds4-commit="${DS4_COMMIT:-unknown}" -Dllama-tag="${LLAMA_TAG:-unknown}" \
  ${ZIG_MODE_FLAGS[@]+"${ZIG_MODE_FLAGS[@]}"} 2>&1 | tail -3
# The bundled guest agent (static aarch64-linux ELF) rides inside the app.
DEVELOPER_DIR="$ZIG_DEVELOPER_DIR" "$ZIG" build vz-agent 2>&1 | tail -1
MLX_BIN="zig-out/bin/mlx-serve"
if [ ! -f "$MLX_BIN" ]; then
    echo "ERROR: Zig build failed"
    exit 1
fi
echo "  mlx-serve binary: $(du -h "$MLX_BIN" | cut -f1)"
cd "$SCRIPT_DIR"

# ── Phase 3: Generate app icon ──
APP="$SCRIPT_DIR/$APP_NAME.app"
CONTENTS="$APP/Contents"
ICON_DIR=$(mktemp -d)
# 12 sips invocations + iconutil for an image that changes about once a year.
# Fast dev reuses the icns already in the bundle — but only if there IS one, so
# the first fast build after a clean still produces a properly badged app
# instead of a generic-icon window in the Dock.
if [ "$FAST_DEV" = "1" ] && [ -f "$CONTENTS/Resources/AppIcon.icns" ]; then
    echo "→ Reusing bundled app icon (FAST_DEV)"
else
    echo "→ Generating app icon..."
    ICONSET="$ICON_DIR/AppIcon.iconset"
    mkdir -p "$ICONSET"
    for size in 16 32 64 128 256 512; do
        sips -z $size $size "$SCRIPT_DIR/appiconb.png" --out "$ICONSET/icon_${size}x${size}.png" > /dev/null 2>&1
        double=$((size * 2))
        sips -z $double $double "$SCRIPT_DIR/appiconb.png" --out "$ICONSET/icon_${size}x${size}@2x.png" > /dev/null 2>&1
    done
    iconutil -c icns "$ICONSET" -o "$ICON_DIR/AppIcon.icns" 2>/dev/null || echo "  (iconutil skipped, will use default icon)"
fi

# ── Phase 4: Create .app bundle ──
# A release build starts from an EMPTY bundle: anything that stops being
# produced (a retired resource, a renamed framework) has to stop shipping, and
# only `rm -rf` guarantees that. Fast dev updates in place instead — the
# frameworks are the bulk of the bundle and re-copying them per iteration is
# pure I/O. The trade is that a deleted resource lingers, so when a fast build
# behaves strangely, delete "$APP" (or run once without FAST_DEV) before
# believing it.
if [ "$FAST_DEV" = "1" ] && [ -d "$APP" ]; then
    echo "→ Updating app bundle in place (FAST_DEV)..."
else
    echo "→ Creating app bundle..."
    rm -rf "$APP"
fi
mkdir -p "$CONTENTS/MacOS"
mkdir -p "$CONTENTS/Frameworks"
mkdir -p "$CONTENTS/Resources"

# Main Swift executable
cp "$SWIFT_BIN" "$CONTENTS/MacOS/MLXCore"

# App resources (tray icon etc.)
cp -R "$SCRIPT_DIR/Sources/MLXServe/Resources/"* "$CONTENTS/Resources/" 2>/dev/null || true

# SwiftPM does not embed resource bundles when we assemble the .app by hand.
# SwaTex loads its KaTeX fonts from this bundle at runtime.
cp -R "$SWATEX_RESOURCE_BUNDLE" "$CONTENTS/Resources/"

# License + third-party attributions. The bundled mlx-serve binary links
# Apache-2.0 code (MTPLX/dflash/oMLX Metal kernels, jinja.cpp) whose section 4
# wants the license text and NOTICE attributions to travel with the binary, not
# just live in the git repo. Mirrors release.yml.
cp "$PROJECT_ROOT/LICENSE" "$PROJECT_ROOT/LICENSE-APACHE-2.0" "$PROJECT_ROOT/NOTICE" "$CONTENTS/Resources/"

# mlx-serve Zig binary
cp "$PROJECT_ROOT/$MLX_BIN" "$CONTENTS/MacOS/mlx-serve"

# Info.plist (the build-mode variant), stamped HERE rather than at the source:
# the tracked file stays clean, and the bundle can never ship the last release's
# version. Must land after this cp (which would overwrite it) and before the
# codesign below (a write after the seal invalidates it).
cp "$INFO_PLIST_SRC" "$CONTENTS/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $CALVER" "$CONTENTS/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $CALVER" "$CONTENTS/Info.plist"

# The vz-agent guest binary ships in EVERY build variant (issue #89: it was
# MAS-only, so Developer ID / ad-hoc builds always booted the legacy console
# shell and sandboxed MCP could never connect). Only the kernel + rootfs stay
# download-provisioned outside MAS.
echo "→ Staging vz-agent guest binary..."
mkdir -p "$CONTENTS/Resources/guest"
cp "$PROJECT_ROOT/zig-out/guest/vz-agent" "$CONTENTS/Resources/guest/vz-agent"
if [ ! -f "$CONTENTS/Resources/guest/vz-agent" ]; then
    echo "ERROR: vz-agent missing from the bundle — sandboxed MCP would be broken (issue #89)"
    exit 1
fi

if [ "$MAS" = "1" ]; then
    echo "→ Staging bundled guest (kernel + rootfs)..."
    bash "$PROJECT_ROOT/scripts/fetch-guest-rootfs.sh"
    cp "$PROJECT_ROOT/lib/guest/kernel"        "$CONTENTS/Resources/guest/kernel"
    cp "$PROJECT_ROOT/lib/guest/rootfs.tar.gz" "$CONTENTS/Resources/guest/rootfs.tar.gz"
    cp "$SCRIPT_DIR/PrivacyInfo.xcprivacy"     "$CONTENTS/Resources/PrivacyInfo.xcprivacy"
    # The embedded provisioning profile is required for a Mac App Store binary.
    if [ -n "${MAS_PROVISION_PROFILE:-}" ] && [ -f "$MAS_PROVISION_PROFILE" ]; then
        cp "$MAS_PROVISION_PROFILE" "$CONTENTS/embedded.provisionprofile"
    else
        echo "  WARNING: set MAS_PROVISION_PROFILE"
    fi
fi

# App icon
if [ -f "$ICON_DIR/AppIcon.icns" ]; then
    cp "$ICON_DIR/AppIcon.icns" "$CONTENTS/Resources/"
fi

# Bundle dylibs: the self-built mlx + mlx-c staged by scripts/build-mlx.sh —
# libmlxc + libmlx + @rpath sibling deps like libjaccl (missing siblings cause
# a "Library not loaded" dyld error at startup) + the NAX-enabled metallib.
MLX_STAGE_LIB="$PROJECT_ROOT/lib/mlx/lib"

# Staging the frameworks, rewriting their internal load commands and signing
# them are ONE decision, not three: install_name_tool rewrites the file and
# invalidates the signature, so skipping the re-sign while still running the
# fixups would leave broken signatures in the bundle. Fast dev reuses the copy
# already staged — these come out of lib/mlx and move only when the submodule
# pins do, which an mtime compare against the staged source detects. `cp` does
# not preserve mtimes, so the bundled copy is always newer until build-mlx.sh
# rewrites the source. webp (brew) and libllama (fetch-llama.sh) ride the same
# gate; both change rarely, and a build without FAST_DEV restages everything.
STAGE_FRAMEWORKS=1
if [ "$FAST_DEV" = "1" ] \
   && [ -f "$CONTENTS/Frameworks/libmlxc.dylib" ] \
   && [ -f "$CONTENTS/Frameworks/mlx.metallib" ] \
   && [ ! "$MLX_STAGE_LIB/libmlxc.dylib" -nt "$CONTENTS/Frameworks/libmlxc.dylib" ]; then
    STAGE_FRAMEWORKS=0
    echo "→ Reusing bundled frameworks (FAST_DEV)"
fi

if [ "$STAGE_FRAMEWORKS" = "1" ]; then
    if [ ! -f "$MLX_STAGE_LIB/libmlxc.dylib" ]; then
        echo "ERROR: lib/mlx not staged — run scripts/build-mlx.sh"
        exit 1
    fi
    for f in "$MLX_STAGE_LIB"/*.dylib; do
        [ -f "$f" ] && cp "$f" "$CONTENTS/Frameworks/"
    done

    # Metal shader library (NAX kernels included — guarded by tests/test_mlx_staged_nax.sh)
    cp "$MLX_STAGE_LIB/mlx.metallib" "$CONTENTS/Frameworks/"

    # libwebp + libsharpyuv for WebP image decoding in vision pipeline
    WEBP_LIB="$(brew --prefix webp 2>/dev/null || echo "/opt/homebrew/opt/webp")/lib"
    for wlib in libwebp.dylib libsharpyuv.dylib; do
        [ -f "$WEBP_LIB/$wlib" ] && cp "$WEBP_LIB/$wlib" "$CONTENTS/Frameworks/"
    done

    # libllama (llama.cpp GGUF engine) — single self-contained dylib staged by
    # scripts/fetch-llama.sh. Bundled + signed exactly like the others.
    [ -f "$PROJECT_ROOT/lib/llama/lib/libllama.dylib" ] && cp "$PROJECT_ROOT/lib/llama/lib/libllama.dylib" "$CONTENTS/Frameworks/"

    # SwiftOGG's binary deps: YbridOpus + YbridOgg dynamic frameworks (libopus +
    # libogg). VoicePreprocessor uses them to decode Telegram Ogg/Opus voice notes,
    # which AVFoundation can't read. SwiftPM fetches them as xcframeworks under
    # .build/artifacts; embed the macOS slice so the shipped app resolves
    # @rpath/Ybrid*.framework at runtime (without this the app won't launch).
    for fwname in YbridOpus YbridOgg; do
        fwsrc=$(find "$SCRIPT_DIR/.build/artifacts" -path "*macos-arm64*/$fwname.framework" -type d 2>/dev/null | head -1)
        if [ -n "$fwsrc" ]; then
            cp -R "$fwsrc" "$CONTENTS/Frameworks/"
        else
            echo "  WARNING: $fwname.framework not found under .build/artifacts (Telegram voice decode will fail)"
        fi
    done
fi

# Fix rpaths on mlx-serve binary. This one always runs: mlx-serve is re-copied
# (and re-signed) on every build, fast or not, so its load commands are back to
# the ones the linker wrote.
echo "→ Fixing rpaths..."
install_name_tool -change \
    "$(otool -L "$CONTENTS/MacOS/mlx-serve" | grep libmlxc | awk '{print $1}')" \
    "@executable_path/../Frameworks/libmlxc.dylib" \
    "$CONTENTS/MacOS/mlx-serve" 2>/dev/null || true

if [ "$STAGE_FRAMEWORKS" = "1" ]; then
    # Fix libmlxc -> libmlx dependency
    install_name_tool -change \
        "$(otool -L "$CONTENTS/Frameworks/libmlxc.dylib" | grep libmlx.dylib | head -1 | awk '{print $1}')" \
        "@loader_path/libmlx.dylib" \
        "$CONTENTS/Frameworks/libmlxc.dylib" 2>/dev/null || true

    # Add @loader_path to libmlx.dylib's rpath so its @rpath/libjaccl.dylib (and any future @rpath
    # sibling deps from mlx) resolves to the bundled Frameworks dir.
    install_name_tool -add_rpath @loader_path "$CONTENTS/Frameworks/libmlx.dylib" 2>/dev/null || true
fi

# Fix mlx-serve -> libwebp dependency
if [ -f "$CONTENTS/Frameworks/libwebp.dylib" ]; then
    install_name_tool -change \
        "$(otool -L "$CONTENTS/MacOS/mlx-serve" | grep libwebp | awk '{print $1}')" \
        "@executable_path/../Frameworks/libwebp.dylib" \
        "$CONTENTS/MacOS/mlx-serve" 2>/dev/null || true
    # Fix libwebp -> libsharpyuv dependency
    if [ "$STAGE_FRAMEWORKS" = "1" ]; then
        install_name_tool -change \
            "$(otool -L "$CONTENTS/Frameworks/libwebp.dylib" | grep libsharpyuv | awk '{print $1}')" \
            "@loader_path/libsharpyuv.dylib" \
            "$CONTENTS/Frameworks/libwebp.dylib" 2>/dev/null || true
    fi
fi

# Fix mlx-serve -> libllama dependency (@rpath/libllama.dylib -> bundled Frameworks)
if [ -f "$CONTENTS/Frameworks/libllama.dylib" ]; then
    install_name_tool -change \
        "@rpath/libllama.dylib" \
        "@executable_path/../Frameworks/libllama.dylib" \
        "$CONTENTS/MacOS/mlx-serve" 2>/dev/null || true
fi

# MLXCore (the Swift app) loads YbridOpus/YbridOgg via @rpath/Ybrid*.framework.
# Point @rpath at the bundled Frameworks dir — SwiftPM's dev rpath into
# .build/artifacts doesn't survive into the shipped binary, so without this
# added rpath dyld can't find the frameworks and the app fails to launch.
if [ -d "$CONTENTS/Frameworks/YbridOpus.framework" ] || [ -d "$CONTENTS/Frameworks/YbridOgg.framework" ]; then
    install_name_tool -add_rpath "@executable_path/../Frameworks" "$CONTENTS/MacOS/MLXCore" 2>/dev/null || true
fi

rm -rf "$ICON_DIR"

echo "  Bundle created: $APP"
echo "  Size: $(du -sh "$APP" | cut -f1)"

# ── Phase 5: Code sign ──
echo "→ Code signing..."
# Fix permissions for signing
chmod -R u+w "$APP"

# Strip extended attributes before signing. An asset added to
# Sources/MLXServe/Resources by dragging it out of Finder carries
# com.apple.FinderInfo and com.apple.quarantine, and codesign refuses the whole
# bundle with "resource fork, Finder information, or similar detritus not
# allowed" — three minutes into a build, naming the BINARY rather than the file
# that actually brought the metadata in. Cleaning the staged bundle fixes the
# class rather than one asset; `tests/test_resource_xattrs.sh` keeps the
# checked-in copies clean too, so the failure is caught before a build starts.
xattr -cr "$APP" 2>/dev/null || true

# Hardened runtime requires a real Team ID — skip it for ad-hoc ("-") signing,
# otherwise dyld rejects framework loads with "different Team IDs".
ENTITLEMENTS="$APP_ENTITLEMENTS"
if [ "$IDENTITY" = "-" ]; then
    SIGN_OPTS=(--force --sign -)
    # Ad-hoc dev builds run without hardened runtime (mic prompt works without
    # entitlements), but the Agent Sandbox needs com.apple.security.virtualization
    # on the PROCESS or VZVirtualMachine refuses to start — and that entitlement
    # works fine with ad-hoc signing. Apply the same entitlements file.
    APP_SIGN_OPTS=("${SIGN_OPTS[@]}" --entitlements "$ENTITLEMENTS")
elif [ "$MAS" = "1" ]; then
    SIGN_OPTS=(--force --sign "$IDENTITY")
    MAS_BUNDLE_ID="$(/usr/libexec/PlistBuddy -c "Print :CFBundleIdentifier" "$INFO_PLIST_SRC")"
    RENDERED_ENTITLEMENTS="$(mktemp -t MLXCore-MAS-rendered).entitlements"
    bash "$PROJECT_ROOT/scripts/render-mas-entitlements.sh" \
        "$APP_ENTITLEMENTS" "$TEAM_ID" "$MAS_BUNDLE_ID" "$RENDERED_ENTITLEMENTS"
    APP_SIGN_OPTS=("${SIGN_OPTS[@]}" --entitlements "$RENDERED_ENTITLEMENTS")
else
    SIGN_OPTS=(--force --options runtime --sign "$IDENTITY")
    # Under hardened runtime the mic-using process (MLXCore) and the .app need
    # the com.apple.security.device.audio-input entitlement, or macOS denies
    # microphone access in Voice mode without ever prompting. Frameworks/dylibs
    # and the headless mlx-serve binary must NOT carry it (over-broad
    # entitlements can fail notarization).
    APP_SIGN_OPTS=("${SIGN_OPTS[@]}" --entitlements "$ENTITLEMENTS")
fi

# Sign frameworks first (inside-out) — no entitlements. Skipped only when the
# staging block above skipped them too: untouched files keep the signature the
# previous build gave them, and the whole-bundle seal below is re-taken either
# way.

# The metallib is signed on BOTH paths. It is not Mach-O, so codesign stores its
# signature in the com.apple.cs.CodeDirectory EXTENDED ATTRIBUTE — which the
# `xattr -cr` strip above wipes on every build. The dylibs (embedded
# LC_CODE_SIGNATURE) and the .framework bundles (_CodeSignature) survive it; the
# metallib does not, and an unsigned nested code object fails the whole-bundle
# seal ("code object is not signed at all"), leaving an .app that will not open.
# Costs ~0.1s, so the FAST_DEV skip below has nothing to gain by covering it.
for mlib in "$CONTENTS/Frameworks/"*.metallib; do
    [ -f "$mlib" ] && codesign "${SIGN_OPTS[@]}" "$mlib" && echo "  Signed: $(basename "$mlib")"
done

if [ "$STAGE_FRAMEWORKS" = "1" ]; then
    for fw in "$CONTENTS/Frameworks/"*.dylib; do
        [ -f "$fw" ] && codesign "${SIGN_OPTS[@]}" "$fw" && echo "  Signed: $(basename "$fw")"
    done

    # Sign embedded .framework bundles (YbridOpus/YbridOgg) — re-sign over the
    # vendor signature with our identity so dyld accepts them under hardened runtime.
    for fw in "$CONTENTS/Frameworks/"*.framework; do
        [ -d "$fw" ] && codesign "${SIGN_OPTS[@]}" "$fw" && echo "  Signed: $(basename "$fw")"
    done
else
    echo "  Frameworks unchanged — keeping their signatures (FAST_DEV)"
fi


if [ "$MAS" = "1" ] && [ -n "$HELPER_ENTITLEMENTS" ]; then
    codesign "${SIGN_OPTS[@]}" --entitlements "$HELPER_ENTITLEMENTS" "$CONTENTS/MacOS/mlx-serve" && echo "  Signed: mlx-serve (sandbox+inherit)"
else
    codesign "${SIGN_OPTS[@]}" "$CONTENTS/MacOS/mlx-serve" && echo "  Signed: mlx-serve"
fi
codesign "${APP_SIGN_OPTS[@]}" "$CONTENTS/MacOS/MLXCore" && echo "  Signed: MLXCore"
codesign "${APP_SIGN_OPTS[@]}" "$APP" && echo "  Signed: $APP_NAME.app"

# Verify
codesign -vv "$APP" 2>&1 | head -3

# A fast build stops at a signed, launchable .app. Everything past this point
# produces something to HAND SOMEONE — a .pkg, a notarized + stapled bundle, a
# DMG — and a debug-configuration binary must not be any of them.
if [ "$FAST_DEV" = "1" ]; then
    echo ""
    # Name both configurations: which arms were on is exactly what you forget
    # two hours later, and a Debug engine is invisible from the running app.
    echo "=== Fast build complete (v$MLX_SERVE_VERSION, swift $SWIFT_CONFIG, mlx-serve ${ZIG_OPT[0]#-Doptimize=}) ==="
    echo "App: $APP"
    echo "Not notarized, no DMG — run without FAST_DEV to ship."
    exit 0
fi

if [ "$MAS" = "1" ]; then
    echo "→ Building App Store .pkg..."
    PKG_PATH="$SCRIPT_DIR/MLXCore-MAS.pkg"
    INSTALLER_IDENTITY="${APPLE_INSTALLER_ID:-3rd Party Mac Developer Installer: David Dalcu ($TEAM_ID)}"
    if [ "$IDENTITY" = "-" ]; then
        echo "  (ad-hoc build — skipping productbuild; a real 'Apple Distribution' identity is required to submit)"
    else
        productbuild --component "$APP" /Applications \
            --sign "$INSTALLER_IDENTITY" \
            "$PKG_PATH"
        echo ""
        echo "=== Mac App Store build complete! ==="
        echo "App: $APP"
        echo "Pkg: $PKG_PATH"
        echo "Version: v$MLX_SERVE_VERSION"
        echo ""
        echo "To submit: open the .pkg in Transporter, or:"
        echo "  xcrun altool --upload-app -f \"$PKG_PATH\" -t macos --apple-id \"\$APPLE_ID\" --password \"\$APPLE_ID_PASSWORD\""
    fi
    exit 0
fi

# ── Phase 6: Notarize ──
if [ "${SKIP_NOTARIZE:-}" = "1" ]; then
    echo "→ Skipping notarization (SKIP_NOTARIZE=1)"
elif [ "$IDENTITY" = "-" ]; then
    # An ad-hoc signed app can't be notarized (notarization requires a real
    # Developer ID signature), so don't fail a no-account build at the finish line.
    echo "→ Skipping notarization (ad-hoc signing)"
else
    echo "→ Notarizing..."
    APPLE_ID="${APPLE_ID:?Set APPLE_ID in env for notarization}"
    APPLE_ID_PASSWORD="${APPLE_ID_PASSWORD:?Set APPLE_ID_PASSWORD in env (app-specific password)}"

    ZIP_PATH="$SCRIPT_DIR/MLXCore.zip"
    ditto -c -k --keepParent "$APP" "$ZIP_PATH"

    xcrun notarytool submit "$ZIP_PATH" \
        --apple-id "$APPLE_ID" \
        --password "$APPLE_ID_PASSWORD" \
        --team-id "$TEAM_ID" \
        --wait

    xcrun stapler staple "$APP"
    echo "  Notarization complete and stapled"
    rm -f "$ZIP_PATH"
fi

# ── Phase 7: Create DMG installer ──
echo "→ Creating DMG..."
DMG_PATH="$SCRIPT_DIR/MLXCore.dmg"
bash "$PROJECT_ROOT/scripts/create-dmg.sh" "$APP" "$DMG_PATH"

echo ""
echo "=== Build complete! ==="
echo "App: $APP"
echo "DMG: $DMG_PATH"
echo "Version: v$MLX_SERVE_VERSION"
echo ""
echo "To release:"
echo "  gh release create v$MLX_SERVE_VERSION $DMG_PATH --title \"mlx-serve v$MLX_SERVE_VERSION\" --notes-file CHANGELOG_LATEST.md"
