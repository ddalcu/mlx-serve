#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_NAME="MLX Core"
BUNDLE_ID="com.dalcu.mlx-core"

# Signing identity from env (set in ~/.zshrc or CI)
IDENTITY="${APPLE_DEVELOPER_ID:?Set APPLE_DEVELOPER_ID in env (e.g. 'Developer ID Application: Name (TEAMID)')}"
TEAM_ID="${APPLE_TEAM_ID:?Set APPLE_TEAM_ID in env}"

cd "$SCRIPT_DIR"

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

# Set calver version (YY.M.N) — auto-increment N from last GitHub release
YM="$(date +%y.%-m)"
LAST_N=$(gh release list --limit 50 --json tagName --jq "[.[] | .tagName | select(startswith(\"v${YM}.\"))] | map(split(\".\")[2] | tonumber) | max // 0" 2>/dev/null || echo "0")
NEXT_N=$((LAST_N + 1))
CALVER="${YM}.${NEXT_N}"
# Stamp the version into whichever Info.plist this build mode ships.
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $CALVER" "$INFO_PLIST_SRC"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $CALVER" "$INFO_PLIST_SRC"
export MLX_SERVE_VERSION="$CALVER"

echo "=== Building MLX Core v$CALVER ==="

# ── Phase 1: Build Swift app ──
echo "→ Compiling Swift..."
# `-Xswiftc -swift-version -Xswiftc 5` forces Swift 5 language mode globally
# across the build graph. The `swift-sdk` 0.10.x pin (kept for Swift 6.1 / macos-14
# CI compat) declares `swift-tools-version:6.1`; under Swift 6.3 (current Xcode 26)
# its NetworkTransport.swift hits `[#SendingRisksDataRace]` errors that didn't
# exist in 6.1. Swift 5 mode downgrades those to warnings. Until the swift-sdk
# pin can move past 0.11 (or CI moves to a Swift 6.3 runner), this flag keeps
# the build green on both old and new toolchains.
SWIFT_BUILD_FLAGS=(-c release -Xswiftc -swift-version -Xswiftc 5 ${SWIFT_MODE_FLAGS[@]+"${SWIFT_MODE_FLAGS[@]}"})
swift build "${SWIFT_BUILD_FLAGS[@]}" 2>&1 | tail -5
SWIFT_BIN="$(swift build "${SWIFT_BUILD_FLAGS[@]}" --show-bin-path)/MLXCore"
if [ ! -f "$SWIFT_BIN" ]; then
    echo "ERROR: Swift build failed"
    exit 1
fi
echo "  Swift binary: $(du -h "$SWIFT_BIN" | cut -f1)"

# ── Phase 2: Build mlx-serve (Zig) ──
echo "→ Building mlx-serve (Zig)..."
cd "$PROJECT_ROOT"
# Stage libllama (llama.cpp GGUF engine) before the Zig build links against it.
echo "→ Fetching libllama..."
bash "$PROJECT_ROOT/scripts/fetch-llama.sh"
# Engine-version pins surfaced by `mlx-serve --version` (parsed by the app so
# Settings can show engine versions without booting the server — src/version.zig).
# MLX + ggml self-report at runtime; these three have no runtime API:
MLXC_VERSION="$(brew list --versions mlx-c 2>/dev/null | awk '{print $2}')"
DS4_COMMIT="$(git -C "$PROJECT_ROOT/lib/ds4" rev-parse --short HEAD 2>/dev/null)"
LLAMA_TAG="$(cat "$PROJECT_ROOT/lib/llama/.version" 2>/dev/null)"
DEVELOPER_DIR=/Library/Developer/CommandLineTools zig build -Doptimize=ReleaseFast -Dversion="$MLX_SERVE_VERSION" \
  -Dmlx-c-version="${MLXC_VERSION:-unknown}" -Dds4-commit="${DS4_COMMIT:-unknown}" -Dllama-tag="${LLAMA_TAG:-unknown}" \
  ${ZIG_MODE_FLAGS[@]+"${ZIG_MODE_FLAGS[@]}"} 2>&1 | tail -3
# The bundled guest agent (static aarch64-linux ELF) rides inside the app.
DEVELOPER_DIR=/Library/Developer/CommandLineTools zig build vz-agent 2>&1 | tail -1
MLX_BIN="zig-out/bin/mlx-serve"
if [ ! -f "$MLX_BIN" ]; then
    echo "ERROR: Zig build failed"
    exit 1
fi
echo "  mlx-serve binary: $(du -h "$MLX_BIN" | cut -f1)"
cd "$SCRIPT_DIR"

# ── Phase 3: Generate app icon ──
echo "→ Generating app icon..."
ICON_DIR=$(mktemp -d)

ICONSET="$ICON_DIR/AppIcon.iconset"
mkdir -p "$ICONSET"
for size in 16 32 64 128 256 512; do
    sips -z $size $size "$SCRIPT_DIR/appiconb.png" --out "$ICONSET/icon_${size}x${size}.png" > /dev/null 2>&1
    double=$((size * 2))
    sips -z $double $double "$SCRIPT_DIR/appiconb.png" --out "$ICONSET/icon_${size}x${size}@2x.png" > /dev/null 2>&1
done
iconutil -c icns "$ICONSET" -o "$ICON_DIR/AppIcon.icns" 2>/dev/null || echo "  (iconutil skipped, will use default icon)"

# ── Phase 4: Create .app bundle ──
echo "→ Creating app bundle..."
APP="$SCRIPT_DIR/$APP_NAME.app"
CONTENTS="$APP/Contents"
rm -rf "$APP"
mkdir -p "$CONTENTS/MacOS"
mkdir -p "$CONTENTS/Frameworks"
mkdir -p "$CONTENTS/Resources"

# Main Swift executable
cp "$SWIFT_BIN" "$CONTENTS/MacOS/MLXCore"

# App resources (tray icon etc.)
cp -R "$SCRIPT_DIR/Sources/MLXServe/Resources/"* "$CONTENTS/Resources/" 2>/dev/null || true

# mlx-serve Zig binary
cp "$PROJECT_ROOT/$MLX_BIN" "$CONTENTS/MacOS/mlx-serve"

# Info.plist (the build-mode variant, already version-stamped)
cp "$INFO_PLIST_SRC" "$CONTENTS/Info.plist"

if [ "$MAS" = "1" ]; then
    echo "→ Staging bundled guest (kernel + rootfs + vz-agent)..."
    bash "$PROJECT_ROOT/scripts/fetch-guest-rootfs.sh"
    mkdir -p "$CONTENTS/Resources/guest"
    cp "$PROJECT_ROOT/lib/guest/kernel"        "$CONTENTS/Resources/guest/kernel"
    cp "$PROJECT_ROOT/lib/guest/rootfs.tar.gz" "$CONTENTS/Resources/guest/rootfs.tar.gz"
    cp "$PROJECT_ROOT/zig-out/guest/vz-agent"  "$CONTENTS/Resources/guest/vz-agent"
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

# Bundle dylibs (prefer /opt/homebrew/lib for source-built versions, then Homebrew Cellar)
if [ -f "/opt/homebrew/lib/libmlxc.dylib" ]; then
    MLXC_LIB="/opt/homebrew/lib"
else
    MLXC_LIB=$(brew --prefix mlx-c 2>/dev/null || echo "/opt/homebrew/opt/mlx-c")/lib
fi
if [ -f "/opt/homebrew/lib/libmlx.dylib" ]; then
    MLX_LIB="/opt/homebrew/lib"
else
    MLX_LIB=$(brew --prefix mlx 2>/dev/null || echo "/opt/homebrew/opt/mlx")/lib
fi

for lib in libmlxc.dylib; do
    if [ -f "$MLXC_LIB/$lib" ]; then
        cp "$MLXC_LIB/$lib" "$CONTENTS/Frameworks/"
    fi
done

# Copy ALL dylibs from mlx's lib dir, not just libmlx.dylib — newer mlx (0.31.2+) introduced
# sibling deps like libjaccl.dylib referenced via @rpath. Missing them causes a "Library not loaded"
# dyld error at startup. We deliberately use the keg-only path (`brew --prefix mlx`/lib) instead
# of `$MLX_LIB` because the latter can fall back to `/opt/homebrew/lib` (the symlink dir for ALL
# Homebrew libs), which would copy thousands of unrelated dylibs.
MLX_KEG_LIB="$(brew --prefix mlx 2>/dev/null || echo "/opt/homebrew/opt/mlx")/lib"
for f in "$MLX_KEG_LIB"/*.dylib; do
    [ -f "$f" ] && cp "$f" "$CONTENTS/Frameworks/"
done

# Metal shader library
if [ -f "$MLX_LIB/mlx.metallib" ]; then
    cp "$MLX_LIB/mlx.metallib" "$CONTENTS/Frameworks/"
fi

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

# Fix rpaths on mlx-serve binary
echo "→ Fixing rpaths..."
install_name_tool -change \
    "$(otool -L "$CONTENTS/MacOS/mlx-serve" | grep libmlxc | awk '{print $1}')" \
    "@executable_path/../Frameworks/libmlxc.dylib" \
    "$CONTENTS/MacOS/mlx-serve" 2>/dev/null || true

# Fix libmlxc -> libmlx dependency
install_name_tool -change \
    "$(otool -L "$CONTENTS/Frameworks/libmlxc.dylib" | grep libmlx.dylib | head -1 | awk '{print $1}')" \
    "@loader_path/libmlx.dylib" \
    "$CONTENTS/Frameworks/libmlxc.dylib" 2>/dev/null || true

# Add @loader_path to libmlx.dylib's rpath so its @rpath/libjaccl.dylib (and any future @rpath
# sibling deps from mlx) resolves to the bundled Frameworks dir.
install_name_tool -add_rpath @loader_path "$CONTENTS/Frameworks/libmlx.dylib" 2>/dev/null || true

# Fix mlx-serve -> libwebp dependency
if [ -f "$CONTENTS/Frameworks/libwebp.dylib" ]; then
    install_name_tool -change \
        "$(otool -L "$CONTENTS/MacOS/mlx-serve" | grep libwebp | awk '{print $1}')" \
        "@executable_path/../Frameworks/libwebp.dylib" \
        "$CONTENTS/MacOS/mlx-serve" 2>/dev/null || true
    # Fix libwebp -> libsharpyuv dependency
    install_name_tool -change \
        "$(otool -L "$CONTENTS/Frameworks/libwebp.dylib" | grep libsharpyuv | awk '{print $1}')" \
        "@loader_path/libsharpyuv.dylib" \
        "$CONTENTS/Frameworks/libwebp.dylib" 2>/dev/null || true
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

# Sign frameworks first (inside-out) — no entitlements.
for fw in "$CONTENTS/Frameworks/"*.metallib "$CONTENTS/Frameworks/"*.dylib; do
    [ -f "$fw" ] && codesign "${SIGN_OPTS[@]}" "$fw" && echo "  Signed: $(basename "$fw")"
done

# Sign embedded .framework bundles (YbridOpus/YbridOgg) — re-sign over the
# vendor signature with our identity so dyld accepts them under hardened runtime.
for fw in "$CONTENTS/Frameworks/"*.framework; do
    [ -d "$fw" ] && codesign "${SIGN_OPTS[@]}" "$fw" && echo "  Signed: $(basename "$fw")"
done


if [ "$MAS" = "1" ] && [ -n "$HELPER_ENTITLEMENTS" ]; then
    codesign "${SIGN_OPTS[@]}" --entitlements "$HELPER_ENTITLEMENTS" "$CONTENTS/MacOS/mlx-serve" && echo "  Signed: mlx-serve (sandbox+inherit)"
else
    codesign "${SIGN_OPTS[@]}" "$CONTENTS/MacOS/mlx-serve" && echo "  Signed: mlx-serve"
fi
codesign "${APP_SIGN_OPTS[@]}" "$CONTENTS/MacOS/MLXCore" && echo "  Signed: MLXCore"
codesign "${APP_SIGN_OPTS[@]}" "$APP" && echo "  Signed: $APP_NAME.app"

# Verify
codesign -vv "$APP" 2>&1 | head -3

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
