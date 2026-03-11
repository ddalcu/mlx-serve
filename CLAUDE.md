# mlx-serve – project context for AI

Native Zig server that runs MLX-format LMs on Apple Silicon and exposes an OpenAI-compatible HTTP API. No Python.

## Stack

- **Zig** 0.15+
- **mlx-c** (Apple) via Homebrew; FFI in `src/mlx.zig`
- **Jinja_cpp** (lib/jinja_cpp): chat templates; replaces previous vibe-based Jinja (macros caused infinite loop)
- **safetensors** for weights; BPE tokenizers (SentencePiece / byte-level)

## Layout

| Path | Role |
|------|------|
| `src/main.zig` | Entry, CLI (`--model`, `--serve`, `--host`, `--port`, `--prompt`, `--max-tokens`, `--temp`, `--ctx-size`, `--timeout`, `--log-level`, `--version`, `--help`) |
| `src/mlx.zig` | mlx-c FFI |
| `src/model.zig` | Config + safetensors loading; supports Gemma-3, Qwen3, Qwen3.5 MoE, Qwen3-next, Llama, Mistral |
| `src/tokenizer.zig` | BPE tokenizer |
| `src/transformer.zig` | Forward pass (embedding, attention, MLP, MoE, GatedDeltaNet); architecture dispatch |
| `src/generate.zig` | Autoregressive generation, sampling (temperature, top-k, top-p, repeat penalty, presence penalty, logprobs) |
| `src/chat.zig` | Chat template formatting (ChatML, Gemma turns, Llama-3, Jinja2 via Jinja_cpp); thinking/reasoning tags |
| `src/server.zig` | HTTP server: `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions` (stream + non-stream, tool calling) |
| `src/status.zig` | TUI status bar (CPU, memory, GPU metrics) |
| `src/log.zig` | Leveled logging (error, warn, info, debug) |
| `build.zig` | Zig build; links mlx-c and Jinja_cpp |

## Testing

- `zig build test` — 69 unit tests (log, chat, model, generate, server)
- `./tests/integration_test.sh [model_dir] [port]` — 36 end-to-end API tests (needs a model)
- Always run unit tests before submitting changes
- Add tests for new pure logic functions in the same source file

## Conventions

- Prefer minimal, DRY Zig; avoid unnecessary abstraction.
- Chat templates live in model dirs; Jinja_cpp renders them (with fallback formatting).
- Server supports concurrent health checks via threaded connections, single-slot generation.
- KV cache reuse across requests via prompt prefix matching.
- Tests go at the bottom of each source file (Zig convention).
