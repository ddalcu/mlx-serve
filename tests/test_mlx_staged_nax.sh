#!/bin/bash
# test_mlx_staged_nax.sh — static guard for the self-built, NAX-enabled mlx + mlx-c (scripts/build-mlx.sh from the
# pinned submodules at deployment target 26.2; the Homebrew bottle ships NAX off): the staged dylibs + metallib +
# .version stamp exist, the metallib contains `*_nax` kernels (MLX's CMake gate fails SILENTLY below 26.2),
# libmlx.dylib minos >= 26.2, libmlxc.dylib links the staged mlx, a built zig-out/bin/mlx-serve links no Homebrew
# mlx/mlx-c, and when it loads @rpath/libmlx.dylib itself (the mlx-stream shims) its rpaths reach libmlx in the
# app's Frameworks/ and the CLI zip's lib/.
#
#   ./tests/test_mlx_staged_nax.sh   # hermetic: no GPU, no server

set -u
cd "$(dirname "$0")/.." || exit 1

STAGE="lib/mlx"
PASS=0
FAIL=0

ok()   { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

echo "== staged files =="
for f in "$STAGE/lib/libmlx.dylib" "$STAGE/lib/libmlxc.dylib" "$STAGE/lib/mlx.metallib"; do
  if [ -f "$f" ]; then ok "$f exists"; else fail "$f missing (run scripts/build-mlx.sh)"; fi
done
if [ -f "$STAGE/.version" ]; then
  ok ".version stamp present ($(tr '\n' ' ' < "$STAGE/.version"))"
else
  fail "$STAGE/.version missing (run scripts/build-mlx.sh)"
fi

echo "== NAX kernels present in metallib =="
if [ -f "$STAGE/lib/mlx.metallib" ]; then
  NAX_COUNT=$(strings "$STAGE/lib/mlx.metallib" | grep -c "_nax" || true)
  if [ "$NAX_COUNT" -gt 0 ]; then
    ok "metallib contains NAX kernels ($NAX_COUNT symbol hits)"
  else
    fail "metallib has ZERO *_nax kernels — the 26.2 CMake gate failed silently (wrong SDK/deployment target or missing Metal Toolchain)"
  fi
else
  fail "cannot check NAX kernels: metallib missing"
fi

echo "== deployment target >= 26.2 =="
if [ -f "$STAGE/lib/libmlx.dylib" ]; then
  MINOS=$(otool -l "$STAGE/lib/libmlx.dylib" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')
  MAJOR=${MINOS%%.*}
  MINOR=$(echo "$MINOS" | cut -d. -f2)
  if [ -n "$MINOS" ] && { [ "$MAJOR" -gt 26 ] || { [ "$MAJOR" -eq 26 ] && [ "${MINOR:-0}" -ge 2 ]; }; }; then
    ok "libmlx.dylib minos is $MINOS"
  else
    fail "libmlx.dylib minos is '$MINOS' — need >= 26.2 for the NAX kernel gate"
  fi
else
  fail "cannot check minos: libmlx.dylib missing"
fi

echo "== mlx-c links the staged mlx, not Homebrew =="
if [ -f "$STAGE/lib/libmlxc.dylib" ]; then
  if otool -L "$STAGE/lib/libmlxc.dylib" | grep -q "/opt/homebrew"; then
    fail "libmlxc.dylib still references /opt/homebrew:"$'\n'"$(otool -L "$STAGE/lib/libmlxc.dylib" | grep /opt/homebrew)"
  else
    ok "libmlxc.dylib has no /opt/homebrew references"
  fi
else
  fail "cannot check linkage: libmlxc.dylib missing"
fi

echo "== mlx-serve binary linkage + min-OS (if built) =="
BIN="zig-out/bin/mlx-serve"
if [ -f "$BIN" ]; then
  if otool -L "$BIN" | grep -Eq "/opt/homebrew/(opt|Cellar)/(mlx|mlx-c)/"; then
    fail "mlx-serve still links Homebrew mlx/mlx-c:"$'\n'"$(otool -L "$BIN" | grep -E '/opt/homebrew/(opt|Cellar)/(mlx|mlx-c)/')"
  else
    ok "mlx-serve links no Homebrew mlx/mlx-c"
  fi
  # The binary's own minos must state the honest floor (26.2, matching the
  # libmlx it links) — a 14.0-minos binary "loads" on old macOS only to die
  # on the dylib with a worse error.
  BIN_MINOS=$(otool -l "$BIN" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')
  BMAJOR=${BIN_MINOS%%.*}
  BMINOR=$(echo "$BIN_MINOS" | cut -d. -f2)
  if [ -n "$BIN_MINOS" ] && { [ "$BMAJOR" -gt 26 ] || { [ "$BMAJOR" -eq 26 ] && [ "${BMINOR:-0}" -ge 2 ]; }; }; then
    ok "mlx-serve minos is $BIN_MINOS"
  else
    fail "mlx-serve minos is '$BIN_MINOS' — must be >= 26.2 (build.zig os_version_min must match the libmlx floor)"
  fi
  if otool -L "$BIN" | grep -q "@rpath/libmlx.dylib"; then
    RPATHS=$(otool -l "$BIN" | awk '/LC_RPATH/{f=1} f && / path /{print $2; f=0}')
    for want in "@executable_path/../Frameworks" "@executable_path/lib"; do
      if echo "$RPATHS" | grep -qx "$want"; then
        ok "mlx-serve reaches libmlx through rpath $want"
      else
        fail "mlx-serve loads @rpath/libmlx.dylib but has no rpath $want (the app keeps it in Frameworks/, the CLI zip in lib/)"
      fi
    done
  fi
else
  echo "  NOTE: $BIN not built — linkage + minos checks skipped (build with: zig build -Doptimize=ReleaseFast)"
fi

echo
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
