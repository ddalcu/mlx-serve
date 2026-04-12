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
| `app/Sources/MLXServe/` | Swift macOS app (MLX Core) |
| `tests/` | Integration test scripts |

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
