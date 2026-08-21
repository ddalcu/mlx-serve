# Zig is NOT a brew dep: homebrew's `zig` formula still ships 0.16.0, but this
# project requires a 0.17 nightly (comptime check at the top of build.zig —
# 0.16.0's bundled libc++ fails against the macOS 27 SDK) until 0.17.0 stable
# ships. `scripts/fetch-zig.sh` fetches the pinned nightly into
# .zig-toolchain/ instead (CI: same script in release.yml).
#
# Brewfile can't pin versions, so this floor is enforced at build time:
#   webp >= 1.6.0   build.zig verifyBrewDeps
# mlx + mlx-c are NOT brew deps: pinned submodules (lib/mlx-src, lib/mlxc-src)
# built by scripts/build-mlx.sh so the NAX (M5) kernels ship enabled — the
# brew bottle is compiled at deployment target 26.0 and silently disables them.
#
# cmake is what BUILDS those submodules (four calls in scripts/build-mlx.sh),
# so it is a build-time tool rather than something the binary links against.
# macOS ships no cmake and GitHub's runner image does, which is why its absence
# here went unnoticed: CI was green while a clean Mac died at the first
# `cmake -S`. Guard: tests/test_brewfile_covers_build_tools.sh.
brew "cmake"
brew "webp"
