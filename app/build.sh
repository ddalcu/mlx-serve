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

# Set calver version (YYYY.M.D)
CALVER="$(date +%Y.%-m.%-d)"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $CALVER" Info.plist
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $CALVER" Info.plist

echo "=== Building MLX Core $CALVER ==="

# ── Phase 1: Build Swift app ──
echo "→ Compiling Swift..."
swift build -c release 2>&1 | tail -5
SWIFT_BIN="$(swift build -c release --show-bin-path)/MLXCore"
if [ ! -f "$SWIFT_BIN" ]; then
    echo "ERROR: Swift build failed"
    exit 1
fi
echo "  Swift binary: $(du -h "$SWIFT_BIN" | cut -f1)"

# ── Phase 2: Build mlx-serve (Zig) ──
echo "→ Building mlx-serve (Zig)..."
cd "$PROJECT_ROOT"
DEVELOPER_DIR=/Library/Developer/CommandLineTools zig build -Doptimize=ReleaseFast 2>&1 | tail -3
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

# Info.plist
cp "$SCRIPT_DIR/Info.plist" "$CONTENTS/"

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

for lib in libmlx.dylib; do
    if [ -f "$MLX_LIB/$lib" ]; then
        cp "$MLX_LIB/$lib" "$CONTENTS/Frameworks/"
    fi
done

# Metal shader library
if [ -f "$MLX_LIB/mlx.metallib" ]; then
    cp "$MLX_LIB/mlx.metallib" "$CONTENTS/Frameworks/"
fi

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

rm -rf "$ICON_DIR"

echo "  Bundle created: $APP"
echo "  Size: $(du -sh "$APP" | cut -f1)"

# ── Phase 5: Code sign ──
echo "→ Code signing..."
# Fix permissions for signing
chmod -R u+w "$APP"

# Sign frameworks first (inside-out)
for fw in "$CONTENTS/Frameworks/"*.metallib "$CONTENTS/Frameworks/"*.dylib; do
    [ -f "$fw" ] && codesign --force --options runtime --sign "$IDENTITY" "$fw" && echo "  Signed: $(basename "$fw")"
done

# Sign executables
codesign --force --options runtime --sign "$IDENTITY" "$CONTENTS/MacOS/mlx-serve" && echo "  Signed: mlx-serve"
codesign --force --options runtime --sign "$IDENTITY" "$CONTENTS/MacOS/MLXCore" && echo "  Signed: MLXCore"
codesign --force --options runtime --sign "$IDENTITY" "$APP" && echo "  Signed: $APP_NAME.app"

# Verify
codesign -vv "$APP" 2>&1 | head -3

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
