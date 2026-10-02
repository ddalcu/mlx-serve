#!/usr/bin/env bash
# Build llama.cpp's libllama (+ split ggml libs) with the Vulkan backend and stage
# them for linking into the Linux mlx-serve binary (gguf_only build).
#
# Unlike scripts/fetch-llama.sh (which downloads the macOS Metal XCFramework), Linux
# has no prebuilt Vulkan llama.cpp, so we build from source at the pinned tag.
# Produces lib/llama/lib/{libllama.so,libggml*.so} + lib/llama/include + .version.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_TAG="${LLAMA_TAG:-b10809}"          # keep in lockstep with scripts/fetch-llama.sh
SRC="${LLAMA_SRC:-$REPO_ROOT/.llama-src}"
DEST="$REPO_ROOT/lib/llama"
STAMP="$DEST/.version"
if [ -f "$STAMP" ] && [ -f "$DEST/lib/libllama.so" ] && [ "$(cat "$STAMP")" = "$LLAMA_TAG" ]; then
  echo "[build-llama-linux] lib/llama already at $LLAMA_TAG — nothing to do"; exit 0
fi
[ -d "$SRC/.git" ] || git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp.git "$SRC"
cmake -S "$SRC" -B "$SRC/build-vulkan" -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
      -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$SRC/build-vulkan" --target llama -j "$(nproc)"
mkdir -p "$DEST/lib" "$DEST/include"
cp -a "$SRC"/build-vulkan/bin/libllama.so* "$SRC"/build-vulkan/bin/libggml*.so* "$DEST/lib/"
cp "$SRC/include/llama.h" "$DEST/include/"; cp "$SRC"/ggml/include/*.h "$DEST/include/"
echo "$LLAMA_TAG" > "$STAMP"
echo "[build-llama-linux] staged libllama ($LLAMA_TAG, Vulkan) into $DEST"
