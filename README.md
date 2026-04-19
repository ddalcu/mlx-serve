# mlx-serve

[![Release](https://img.shields.io/github/v/release/ddalcu/mlx-serve?style=flat-square)](https://github.com/ddalcu/mlx-serve/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-black?style=flat-square&logo=apple)](https://github.com/ddalcu/mlx-serve/releases/latest)
[![Zig](https://img.shields.io/badge/built%20with-Zig-f7a41d?style=flat-square&logo=zig)](https://ziglang.org)

**[ddalcu.github.io/mlx-serve](https://ddalcu.github.io/mlx-serve/)**

Native Zig server that runs MLX-format language models on Apple Silicon and exposes OpenAI-compatible and Anthropic-compatible HTTP APIs. No Python. Comes with **MLX Core**, a macOS menu bar app with built-in chat, agent mode, and model management.

![MLX Core](docs/mlxcore-screenshot-1.png)

[<img src="docs/appiconb.png" width="48" align="center">](https://github.com/ddalcu/mlx-serve/releases/latest) **[Download MLX Core.app](https://github.com/ddalcu/mlx-serve/releases/latest)** — latest release for macOS (Apple Silicon)

### Install via Homebrew

```bash
brew tap ddalcu/mlx-serve https://github.com/ddalcu/mlx-serve
brew install --cask mlx-core   # GUI menu bar app
brew install mlx-serve          # CLI server only
```

## Features

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`)
- Anthropic-compatible API (`/v1/messages`) — works with Claude Code
- Streaming and non-streaming responses
- Tool calling (function calling) with automatic detection
- KV cache reuse across requests for fast multi-turn conversations
- Sampling: temperature, top-k, top-p, repeat penalty, presence penalty
- Vision/image support (Gemma 4 SigLIP encoder) — send images via `image_url` content blocks
- Reasoning/thinking mode support
- Chat templates via Jinja2 (Jinja_cpp) with fallback formatting
- TUI status bar with CPU, memory, and GPU metrics

## MLX Core (macOS App)

Menu bar app that wraps the server with a full UI:

- **Model browser** -- download models from HuggingFace with resumable downloads
- **Chat interface** -- multi-session chat with markdown rendering
- **Agent mode** -- 10 built-in tools (shell, cwd, readFile, writeFile, editFile, searchFiles, listFiles, browse, webSearch, saveMemory) with automatic tool calling loop
- **Editable system prompt** -- customize agent behavior via `~/.mlx-serve/system-prompt.md` (Agent menu → Edit System Prompt)
- **Persistent memory** -- agent can save memories across sessions to `~/.mlx-serve/memory.md`
- **Prompt-based skills** -- drop `.md` files in `~/.mlx-serve/skills/` to teach the agent custom capabilities
- **Server management** -- start/stop server, view logs, configure max tokens
- **Image Generation (FLUX.2)** -- optional, tray button; requires Python (see below)
- **Video Generation (LTX-Video 2.3, MLX-native, with audio)** -- optional, tray button; requires Python + ffmpeg (see below)

### Image / Video Generation (optional)

The tray has **ImageGen** and **VideoGen** buttons that run [FLUX.2](https://huggingface.co/black-forest-labs) and [LTX-Video 2.3](https://github.com/dgrauet/ltx-2-mlx) through a Python subprocess. Both run natively on MLX — no MPS/diffusers path. This is completely optional — the Zig server itself remains Python-free.

**Prerequisite:** Python 3 and ffmpeg must be installed on your Mac.

```bash
brew install python ffmpeg
```

Then launch MLX Core, click the ImageGen (or VideoGen) tray icon, and hit **Install** in the window. The app will:

1. Create a dedicated venv at `~/.mlx-serve/venv` (does not touch your system Python)
2. Install mflux (FLUX), ltx-pipelines-mlx (LTX-2.3), and shared utilities. ~3 GB pip install.
3. Download the model weights on first generation (HuggingFace cache, resumable)

**Models:**

| Feature | Default | Other options | Approx. RAM |
|---|---|---|---|
| Image | FLUX.2-klein 4B 4-bit (mflux, ~5 GB pre-quantized) | FLUX.1-schnell / dev 4-bit and 8-bit | 8 / 12 / 16 GB |
| Video | LTX-Video 2.3 Q4 | — | 24 GB RAM, ~50 GB first-run download (LTX 41 GB + Gemma 8 GB) |

> The 41 GB LTX snapshot ships **both** transformer variants (1-stage distilled + 2-stage dev, ~11 GB each) plus a 7.6 GB distillation LoRA, so you can switch between Fast/Good/Quality/Super offline without re-downloading.

The image path uses [`mflux`](https://github.com/filipstrand/mflux) for native MLX inference with built-in 4/8-bit quantization — the only way FLUX fits on Apple Silicon under 32 GB. The video path uses [`ltx-2-mlx`](https://github.com/dgrauet/ltx-2-mlx), a native MLX pipeline for LTX-Video 2.3 with audio generation (muxed via system `ffmpeg`).

Outputs go to `~/.mlx-serve/generations/images/YYYY-MM-DD/` and `.../videos/YYYY-MM-DD/`.

> The app won't let you start a generation if there isn't enough free RAM. If the mlx-serve server is running and competing for memory, you'll be prompted to stop it first.

## Supported Models

| Architecture | `model_type` | Examples | Chat Format | Vision |
|---|---|---|---|---|
| **Gemma 4** | `gemma4` | `gemma-4-e2b-it-4bit`, `gemma-4-e4b-it-8bit`, `gemma-4-26b-a4b-it-4bit` | Gemma turns | SigLIP |
| **Gemma 3** | `gemma3` | `gemma-3-12b-it-qat-4bit` | Gemma turns | -- |
| **Qwen 3 / 3.5 / 3.6** | `qwen3`, `qwen3_5`, `qwen3_5_moe`, `qwen3_next` | `Qwen3-4B`, `Qwen3.5-4B`, `Qwen3.6-35B-A3B` | ChatML | -- |
| **Nemotron-H** | `nemotron_h` | Nemotron-3-Nano-4B | ChatML | -- |
| **LFM2** | `lfm2` | LFM2.5-350M | ChatML | -- |
| **Llama** | `llama` | Llama 3, Llama 3.1, Llama 3.2 | Llama-3 | -- |
| **Mistral** | `mistral` | Mistral 7B | ChatML | -- |

Any quantized MLX model using one of the above architectures should work. Models with unsupported architectures are flagged in the Model Browser but can still be downloaded.

### Not Yet Supported

| Architecture | `model_type` | Examples | Reason |
|---|---|---|---|
| **LFM2-VL** | `lfm2-vl` | LFM2.5-VL-450M | Needs vision encoder integration |
| **Phi** | `phi`, `phi3` | Phi-3, Phi-4 | Different attention/MLP layout, untested |
| **Cohere** | `command-r` | Command R+ | Different architecture, untested |
| **BERT** (partial) | `bert` | -- | Encoder-only, config parsing exists but no serving endpoint |

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- [Zig 0.15+](https://ziglang.org/download/)
- mlx-c and libwebp:

```bash
brew install mlx-c webp
```

## Quick Start

### Download a model

The MLX Core app can download models directly, or use the CLI:

```bash
pip install huggingface-hub
huggingface-cli download mlx-community/gemma-4-e4b-it-4bit --local-dir ~/.mlx-serve/models/gemma-4-e4b-it-4bit
```

### Build and run

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/mlx-serve --model ~/.mlx-serve/models/gemma-4-e4b-it-4bit --serve --port 8080
```

### Build the app

```bash
cd app && SKIP_NOTARIZE=1 bash build.sh
open "MLX Core.app"
```

Requires `APPLE_DEVELOPER_ID` and `APPLE_TEAM_ID` environment variables for code signing.

## Usage

### Interactive mode

```bash
./zig-out/bin/mlx-serve --model /path/to/model --prompt "What is 2+2?"
```

### HTTP server

```bash
./zig-out/bin/mlx-serve --model /path/to/model --serve --port 8080
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | required | Path to the MLX model directory |
| `--serve` | off | Start the HTTP server |
| `--host ADDR` | `127.0.0.1` | Host address to bind |
| `--port N` | `8080` | Port for the HTTP server |
| `--prompt TEXT` | `"Hello"` | Prompt for interactive mode |
| `--max-tokens N` | `100` | Maximum tokens to generate |
| `--temp F` | `0.0` | Sampling temperature (0 = greedy) |
| `--ctx-size N` | auto | Context window size (auto = computed from GPU memory) |
| `--timeout N` | `300` | Request timeout in seconds |
| `--reasoning-budget N` | `0` | Thinking token budget (0 = disabled) |
| `--no-vision` | off | Disable vision encoder even if model supports it |
| `--log-level` | `info` | Log level (error, warn, info, debug) |

## API

### POST /v1/chat/completions

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a haiku about programming."}],
    "max_tokens": 256,
    "stream": true
  }'
```

Supports `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, `tools`, `repetition_penalty`, `presence_penalty`, and `logprobs`. Messages can include `image_url` content blocks (base64 or URL) for vision-capable models.

### POST /v1/messages (Anthropic)

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "mlx-serve",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Write a haiku about programming."}]
  }'
```

Compatible with Claude Code (`ANTHROPIC_BASE_URL=http://localhost:8080 claude`) and Anthropic SDKs. Supports streaming, tool calling, and extended thinking.

### Other endpoints

- `GET /health` -- health check
- `GET /v1/models` -- list loaded models
- `POST /v1/completions` -- text completions
- `POST /v1/messages` -- Anthropic Messages API

## Performance

Benchmarked on Apple M4 (16 GB unified memory):

| Model | Prefill | Decode | Memory |
|---|---|---|---|
| Gemma-4 E4B (4-bit) | ~390 tok/s | ~32 tok/s | 4.3 GB |
| Qwen3.5-4B (4-bit) | ~155 tok/s | ~33 tok/s | 2.4 GB |
| LFM2.5-350M (8-bit) | ~3800 tok/s | ~210 tok/s | 0.4 GB |
| Nemotron-3-Nano-4B (8-bit) | -- | ~22 tok/s | 4.3 GB |

Matches mlx-lm (Python) generation speed while using less memory and starting 3x faster. Key optimizations: fully-lazy async pipeline with reordered eval (submit-first pattern), JIT-compiled activations (GELU, GeGLU, softcap via `mlx_compile`), and GPU memory wiring.

<details>
<summary>Benchmark reproduction</summary>

```bash
# Prefill (~840 token prompt):
./zig-out/bin/mlx-serve --model ~/.mlx-serve/models/gemma-4-e4b-it-4bit \
  --prompt "$(python3 -c "print('Explain the following topics in extreme detail: ' + ', '.join([f'topic {i} about science and technology and its impact on human civilization throughout history' for i in range(1,50)]))")" \
  --max-tokens 1

# Decode (256 tokens, temp=0):
./zig-out/bin/mlx-serve --model ~/.mlx-serve/models/gemma-4-e4b-it-4bit \
  --prompt "Write a detailed essay about quantum computing" \
  --max-tokens 256
```

Run 3 times and take the average of runs 2-3 (run 1 includes model loading from disk).
</details>

## License

MIT
