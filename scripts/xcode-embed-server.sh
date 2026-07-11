#!/bin/bash
# Xcode POST-BUILD phase for project (app/project.yml):
# populate the built .app with everything Xcode doesn't know about — the Zig
# mlx-serve helper, the MLX/webp/llama engine dylibs, the Metal shader
# library, the guest kernel/rootfs/agent, the icon — then fix install names
# and sign the nested pieces. Mirrors app/build.sh phases 3–4 + the MAS
# signing section — keep the two in sync.
#
# What this deliberately does NOT handle: the YbridOpus/YbridOgg frameworks
# (SwiftOGG's binary deps) — Xcode embeds + signs SPM binary frameworks into
# Contents/Frameworks itself, unlike the SwiftPM CLI path build.sh patches up.
#
# Signing: nested code is signed here (inside-out); Xcode signs the outer
# .app after all phases. mlx-serve gets the HELPER entitlements — exactly
# {app-sandbox, inherit}, never more (see PackagingTests). On export/upload,
# the Organizer re-signs nested code preserving these entitlements.
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APPDIR="$ROOT/app"
APP="${CODESIGNING_FOLDER_PATH:?run from an Xcode build phase}"
CONTENTS="$APP/Contents"

mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Frameworks" "$CONTENTS/Resources"

# ── App icon (build.sh phase 3) ──
ICON_DIR=$(mktemp -d)
ICONSET="$ICON_DIR/AppIcon.iconset"
mkdir -p "$ICONSET"
for size in 16 32 64 128 256 512; do
    sips -z $size $size "$APPDIR/appiconb.png" --out "$ICONSET/icon_${size}x${size}.png" > /dev/null 2>&1
    double=$((size * 2))
    sips -z $double $double "$APPDIR/appiconb.png" --out "$ICONSET/icon_${size}x${size}@2x.png" > /dev/null 2>&1
done
iconutil -c icns "$ICONSET" -o "$CONTENTS/Resources/AppIcon.icns" 2>/dev/null \
    || echo "warning: iconutil failed — app ships without AppIcon.icns" >&2
rm -rf "$ICON_DIR"

# ── mlx-serve helper ──
cp "$ROOT/zig-out/bin/mlx-serve" "$CONTENTS/MacOS/mlx-serve"

# ── Guest assets (MAS ships the prebaked guest in Resources/guest) ──
mkdir -p "$CONTENTS/Resources/guest"
cp "$ROOT/lib/guest/kernel"        "$CONTENTS/Resources/guest/kernel"
cp "$ROOT/lib/guest/rootfs.tar.gz" "$CONTENTS/Resources/guest/rootfs.tar.gz"
cp "$ROOT/zig-out/guest/vz-agent"  "$CONTENTS/Resources/guest/vz-agent"

# ── Engine dylibs + Metal shaders (same source-resolution as build.sh) ──
if [ -f "/opt/homebrew/lib/libmlxc.dylib" ]; then
    MLXC_LIB="/opt/homebrew/lib"
else
    MLXC_LIB=$(brew --prefix mlx-c 2>/dev/null || echo "/opt/homebrew/opt/mlx-c")/lib
fi
[ -f "$MLXC_LIB/libmlxc.dylib" ] && cp "$MLXC_LIB/libmlxc.dylib" "$CONTENTS/Frameworks/"

# ALL dylibs from mlx's keg (libmlx + @rpath siblings like libjaccl) — the
# symlink dir /opt/homebrew/lib would drag in thousands of unrelated libs.
MLX_KEG_LIB="$(brew --prefix mlx 2>/dev/null || echo "/opt/homebrew/opt/mlx")/lib"
for f in "$MLX_KEG_LIB"/*.dylib; do
    [ -f "$f" ] && cp "$f" "$CONTENTS/Frameworks/"
done
[ -f "$MLX_KEG_LIB/mlx.metallib" ] && cp "$MLX_KEG_LIB/mlx.metallib" "$CONTENTS/Frameworks/"
[ ! -f "$CONTENTS/Frameworks/mlx.metallib" ] && [ -f "/opt/homebrew/lib/mlx.metallib" ] \
    && cp "/opt/homebrew/lib/mlx.metallib" "$CONTENTS/Frameworks/"

WEBP_LIB="$(brew --prefix webp 2>/dev/null || echo "/opt/homebrew/opt/webp")/lib"
for wlib in libwebp.dylib libsharpyuv.dylib; do
    [ -f "$WEBP_LIB/$wlib" ] && cp "$WEBP_LIB/$wlib" "$CONTENTS/Frameworks/"
done

[ -f "$ROOT/lib/llama/lib/libllama.dylib" ] && cp "$ROOT/lib/llama/lib/libllama.dylib" "$CONTENTS/Frameworks/"

# ── Install-name surgery (byte-for-byte the build.sh rules) ──
chmod -R u+w "$CONTENTS/MacOS/mlx-serve" "$CONTENTS/Frameworks"

install_name_tool -change \
    "$(otool -L "$CONTENTS/MacOS/mlx-serve" | grep libmlxc | awk '{print $1}')" \
    "@executable_path/../Frameworks/libmlxc.dylib" \
    "$CONTENTS/MacOS/mlx-serve" 2>/dev/null || true

install_name_tool -change \
    "$(otool -L "$CONTENTS/Frameworks/libmlxc.dylib" | grep libmlx.dylib | head -1 | awk '{print $1}')" \
    "@loader_path/libmlx.dylib" \
    "$CONTENTS/Frameworks/libmlxc.dylib" 2>/dev/null || true

# @loader_path rpath so libmlx's @rpath siblings (libjaccl…) resolve in-bundle.
install_name_tool -add_rpath @loader_path "$CONTENTS/Frameworks/libmlx.dylib" 2>/dev/null || true

if [ -f "$CONTENTS/Frameworks/libwebp.dylib" ]; then
    install_name_tool -change \
        "$(otool -L "$CONTENTS/MacOS/mlx-serve" | grep libwebp | awk '{print $1}')" \
        "@executable_path/../Frameworks/libwebp.dylib" \
        "$CONTENTS/MacOS/mlx-serve" 2>/dev/null || true
    install_name_tool -change \
        "$(otool -L "$CONTENTS/Frameworks/libwebp.dylib" | grep libsharpyuv | awk '{print $1}')" \
        "@loader_path/libsharpyuv.dylib" \
        "$CONTENTS/Frameworks/libwebp.dylib" 2>/dev/null || true
fi

if [ -f "$CONTENTS/Frameworks/libllama.dylib" ]; then
    install_name_tool -change \
        "@rpath/libllama.dylib" \
        "@executable_path/../Frameworks/libllama.dylib" \
        "$CONTENTS/MacOS/mlx-serve" 2>/dev/null || true
fi

# ── Sign nested code (skipped when Xcode builds unsigned) ──
if [ "${CODE_SIGNING_ALLOWED:-YES}" != "NO" ] && [ -n "${EXPANDED_CODE_SIGN_IDENTITY:-}" ]; then
    for fw in "$CONTENTS/Frameworks/"*.metallib "$CONTENTS/Frameworks/"*.dylib; do
        [ -f "$fw" ] && codesign --force --sign "$EXPANDED_CODE_SIGN_IDENTITY" "$fw"
    done
    codesign --force --sign "$EXPANDED_CODE_SIGN_IDENTITY" \
        --entitlements "$APPDIR/mlx-serve-MAS.entitlements" \
        "$CONTENTS/MacOS/mlx-serve"
    echo "signed nested: $(ls "$CONTENTS/Frameworks" | wc -l | tr -d ' ') items + mlx-serve (sandbox+inherit)"
else
    echo "code signing disabled — nested items left unsigned"
fi
