# Contributing to mlx-serve

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

**Prerequisites:**
- macOS with Apple Silicon (M1+)
- [Zig 0.15+](https://ziglang.org/download/)
- mlx-c: `brew install mlx-c`

**Build:**
```bash
zig build -Doptimize=ReleaseFast        # Zig server
cd app && swift build -c release        # Swift app
cd app && SKIP_NOTARIZE=1 bash build.sh # Full .app bundle
```

**Test:**
```bash
zig build test                          # Zig unit tests
cd app && swift test                    # Swift unit tests
./tests/integration_test.sh [model_dir] # API integration tests (needs a model)
```

## How to Contribute

### Bug Reports
Open an issue with:
- What you expected vs what happened
- Model name and quantization (e.g., `gemma-4-e4b-it-4bit`)
- macOS version and chip (e.g., macOS 15.4, M4)
- Server log output (`--log-level debug`)

### Pull Requests
1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run `zig build test` and `cd app && swift test` — all tests must pass
4. Keep PRs focused — one feature or fix per PR
5. Follow existing code style (no auto-formatters, match surrounding code)

### Code Style
- **Zig:** Minimal, DRY. Tests at the bottom of each source file. No unnecessary abstraction.
- **Swift:** Match existing patterns. No inline CSS. Use SwiftUI idioms.
- Avoid adding dependencies unless absolutely necessary.

### Areas Where Help is Appreciated
- Model architecture support (new transformer variants)
- Performance optimization (GPU utilization, memory efficiency)
- Test coverage for edge cases
- Documentation improvements

## Project Structure

| Path | What it does |
|------|-------------|
| `src/server.zig` | HTTP server, OpenAI + Anthropic API |
| `src/chat.zig` | Chat templates, tool call parsing |
| `src/generate.zig` | Autoregressive generation, sampling |
| `src/model.zig` | Model loading, architecture dispatch |
| `src/arch/` | Per-architecture forward passes for the MLX path |
| `src/ds4_ffi.zig` + `src/arch/ds4.zig` | FFI + bridge for the embedded ds4 engine |
| `lib/<engine>/` | Embedded model-specific engines (currently `ds4`) |
| `app/Sources/MLXServe/` | Swift macOS app (MLX Core) |
| `tests/` | Integration test scripts |

## Embedding a model-specific engine (`lib/<engine>/`)

Some architectures (DSV4-Flash is the first) ship with a purpose-built inference engine that's faster and more correct than re-implementing the forward pass against mlx-c. `lib/<engine>/` is the convention for keeping those engines vendored under our submodule tree without forking them.

A new embedded engine should:

1. **Live as a git submodule** under `lib/<engine>/` and be pinned to an explicit SHA. Add via `git submodule add <url> lib/<engine> && git -C lib/<engine> checkout <sha>`.
2. **Expose a C library API** in a single `<engine>.h`. The header should be narrow enough that mlx-serve's FFI bindings can mirror it 1:1 without semantic drift.
3. **Build cleanly via `zig cc`** — no Makefile-only flags. Add the C/Objective-C/whatever sources to `build.zig` next to the existing `addCSourceFile` calls for `stb_image` / `jinja_cpp` / `ds4`.
4. **Ship an MIT/Apache/BSD-compatible license.** Add a note in `CHANGELOG.md` on first inclusion.
5. **Stay patch-free.** If you need build-time configuration (e.g. kernel source paths), use the upstream engine's existing env-var hooks rather than patching files. We rely on submodule integrity to make `git clone --recursive` reproducible.

Add the FFI bindings at `src/<engine>_ffi.zig` and a Zig-friendly wrapper at `src/arch/<engine>.zig` that owns lifetimes and converts errors. The bridge should expose the same shape the rest of mlx-serve already uses (`open` / `tokenize` / `createSession` / `eval` / `sample`) so generation dispatch is a tag check, not a per-engine switch.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
