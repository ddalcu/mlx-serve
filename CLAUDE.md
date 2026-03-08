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
| `src/main.zig` | Entry, CLI (`--model`, `--serve`, `--port`, `--prompt`, `--max-tokens`, `--temp`) |
| `src/mlx.zig` | mlx-c FFI |
| `src/model.zig` | Config + safetensors loading |
| `src/tokenizer.zig` | BPE tokenizer |
| `src/transformer.zig` | Forward pass (embedding, attention, MLP, norms); architecture dispatch (Gemma-3, Qwen3/Qwen3.5, Llama) |
| `src/generate.zig` | Autoregressive generation, temperature sampling |
| `src/chat.zig` | Chat template formatting (Gemma turns, ChatML, Jinja2 via Jinja_cpp); thinking/reasoning tags |
| `src/server.zig` | HTTP server: `/v1/models`, `/v1/chat/completions` (stream + non-stream) |
| `src/status.zig` | TUI status bar (CPU, memory, prefill/decode state) |
| `build.zig` | Zig build; links mlx-c and Jinja_cpp |

## Conventions

- Prefer minimal, DRY Zig; avoid unnecessary abstraction.
- Chat templates live in model dirs; Jinja_cpp renders them (including `enable_thinking` for reasoning/thinking tags).
- Server is single-threaded; one request at a time.
