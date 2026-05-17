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
- OpenAI Responses API (`/v1/responses`) — stateful chains via `previous_response_id`, streaming SSE with `sequence_number`, plus `/v1/responses/compact` for opaque round-trippable history blobs
- WebSocket transport on `/v1/responses` — same JSON-per-frame contract over `Upgrade: websocket`
- Anthropic-compatible API (`/v1/messages`) — works with Claude Code
- Streaming and non-streaming responses
- Tool calling (function calling) with automatic detection
- KV cache reuse across requests for fast multi-turn conversations
- Sampling: temperature, top-k, top-p, repeat penalty, presence penalty
- **Speculative decoding** — PLD (model-agnostic n-gram lookup, on by default) and the Gemma 4 assistant drafter. Adaptive prompt-time gate keeps novel content at parity; agentic code-edit loops see up to 1.6×.
- Vision/image support (Gemma 4 SigLIP encoder) — send images via `image_url` content blocks
- Reasoning/thinking mode support
- Chat templates via Jinja2 (Jinja_cpp) with fallback formatting
- TUI status bar with CPU, memory, and GPU metrics

## MLX Core (macOS App)

Menu bar app that wraps the server with a full UI:

- **Model browser** -- download models from HuggingFace with resumable downloads. Auto-discovers models from LM Studio (reads `~/.lmstudio/settings.json` `downloadsFolder`); they appear in a separate "Other Discovered Models" section in the picker so you don't re-download what's already on disk.
- **Chat interface** -- multi-session chat with markdown rendering. Drop in PDFs or images alongside text; PDFs are extracted via PDFKit and inlined into the prompt.
- **Agent mode** -- 10 built-in tools (shell, cwd, readFile, writeFile, editFile, searchFiles, listFiles, browse, webSearch, saveMemory) with automatic tool calling loop
- **Editable system prompt** -- customize agent behavior via `~/.mlx-serve/system-prompt.md` (Agent menu → Edit System Prompt)
- **Persistent memory** -- agent can save memories across sessions to `~/.mlx-serve/memory.md`
- **Prompt-based skills** -- drop `.md` files in `~/.mlx-serve/skills/` to teach the agent custom capabilities
- **Server management** -- start/stop server, view logs, full **Settings window** (Cmd+,) for every server-launch flag and per-request default
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

Any quantized MLX model using one of the above architectures should work. Models with unsupported architectures are flagged in the Model Browser but can still be downloaded. Architectures we haven't implemented yet are tracked in [TODO.md](TODO.md).

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
| `--port N` | `11234` | Port for the HTTP server |
| `--prompt TEXT` | `"Hello"` | Prompt for interactive mode |
| `--max-tokens N` | `100` | Maximum tokens to generate |
| `--temp F` | `0.0` | Sampling temperature (0 = greedy) |
| `--ctx-size N` | auto | Context window size (auto = computed from GPU memory) |
| `--timeout N` | `300` | Request timeout in seconds |
| `--reasoning-budget N` | `-1` | Thinking token budget (`-1` = unlimited, `0` = no thinking) |
| `--no-vision` | off | Disable vision encoder even if model supports it |
| `--pld` / `--no-pld` | on | Prompt Lookup Decoding (model-agnostic spec-decode) |
| `--pld-draft-len N` | `5` | Max draft tokens per PLD step |
| `--pld-key-len N` | `3` | N-gram match key length for PLD |
| `--drafter DIR` | none | Gemma 4 assistant drafter checkpoint (e.g. `gemma-4-E4B-it-assistant-bf16`) |
| `--draft-block-size N` | `4` | Drafts per round for the Gemma 4 drafter |
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

### POST /v1/responses (OpenAI Responses API)

```bash
curl http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "input": "Write a haiku about programming.",
    "stream": true
  }'
```

Stateful chains via `previous_response_id`, full streaming SSE with per-event `sequence_number`, schema-conformant envelope with `tools` / `tool_choice` / `text` / `reasoning` / `usage` echo. `POST /v1/responses/compact` returns an opaque base64 history blob that round-trips back as a `compaction` input item without any LLM call. Same endpoint also accepts an `Upgrade: websocket` handshake — each text frame is a `response.create` JSON message, and each SSE event becomes one outbound text frame.

### Other endpoints

- `GET /health` -- health check
- `GET /v1/models` -- list loaded models
- `POST /v1/completions` -- text completions
- `POST /v1/messages` -- Anthropic Messages API
- `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}` -- fetch / delete stored responses

## Performance

Benchmarked on Apple M4 (16 GB unified memory):

| Model | Prefill | Decode | Memory |
|---|---|---|---|
| Gemma-4 E4B (4-bit) | ~390 tok/s | ~33 tok/s | 4.3 GB |
| Qwen3.5-4B (4-bit) | ~380 tok/s | ~38 tok/s | 2.3 GB |
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

## Speculative Decoding

Two flavors, both greedy-equivalent (byte-identical at temp=0 within the first 30 tokens; mathematically exact at temp > 0 via the Leviathan probability-ratio sampler):

- **PLD** (Prompt Lookup Decoding) — model-agnostic n-gram match in `prompt + generated_tokens`. Default-on (`--pld`); zero per-model setup. Wins on agentic loops, RAG, code editing, anywhere the answer echoes prompt content.
- **Gemma 4 assistant drafter** — Google's small 4-layer cross-attention drafters (`gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant-bf16`). Opt-in via `--drafter <dir>`. The drafter cross-attends into the target's KV cache — no separate weights duplicated.

Both share an **adaptive prompt-time gate**: a 3-gram repetition score on the prompt (`spec_gate_threshold = 0.01`) auto-disables speculation on novel content, so creative writing and one-shot Q&A run at parity with `--no-pld` instead of paying per-step verify overhead. A **runtime acceptance gate** further disables speculation mid-decode if accept rate falls below break-even (0.30 after 5 attempts). Sticky for the rest of the request.

### Speedup on the realistic agentic code-edit workload

Apple M-series, MLX 4-bit weights, temp=0, function in prompt + small modification requested (the canonical mlx-serve workload). `nospec` = same binary with `--no-pld`:

| Model | nospec | PLD | Drafter |
|---|---:|---:|---:|
| Gemma 4 E4B (4-bit) | 28.0 tok/s | **45.0 tok/s · 1.61×** | **44.6 tok/s · 1.59×** |
| Qwen 3.5 4B (4-bit) | 28.1 tok/s | **40.5 tok/s · 1.44×** | — |
| LFM2.5 350M (8-bit) | 162 tok/s | 160 tok/s · 0.99× | — |

On creative / novel-content prompts both features stay at parity (≈1.0×) thanks to the gate — **no regression**. The 350M LFM2.5 is roughly neutral on spec-decode — its forward is small enough that the verify pass costs about the same as AR.

Reproduce with **`./tests/bench_spec_matrix.sh`** (release-comparison matrix vs. the shipped binary) or **`./tests/bench_spec.sh --corpus`** (9-prompt threshold-tuning corpus across echo, code-rename, JSON, RAG, agent, plain Q&A, code-translate, summarize, creative).

### vs. LM Studio (HTTP-vs-HTTP)

**+39% faster overall** (geomean across 18 cells, best mlx-serve vs best LMS, identical 4-bit weights, ctx=4096, temp=0).

| Model | Echo | Code | Free-form |
|---|---:|---:|---:|
| Gemma 4 E2B | **+122%** | **+47%** | +20% |
| Gemma 4 E4B | **+97%** | **+53%** | **+35%** |
| Gemma 4 31B | +20% | +4% | -1% |
| Gemma 4 26B-A4B-MoE | **+66%** | +23% | +31% |
| Qwen 3.6 27B | **+60%** | +24% | +32% |
| Qwen 3.6 35B-A3B-MoE | **+88%** | +20% | +25% |

![Gemma 4](docs/perf-vs-lmstudio-gemma-26.5.5.png)
![Qwen 3.6](docs/perf-vs-lmstudio-qwen36-26.5.5.png)

Reproduce: `./tests/bench_vs_lmstudio_omlx.sh --family gemma` (or `qwen36`). Requires `lms`, `jq`, `python3`, `matplotlib`; install `omlx` to include it in the comparison (it auto-skips otherwise).

## Acknowledgements

mlx-serve stands on a lot of open-source shoulders. Huge thanks to all of these projects.

### Inference + math

- [**MLX**](https://github.com/ml-explore/mlx) (Apple) — the C++/Metal tensor framework that does the actual GPU work. We link against it via [`mlx-c`](https://github.com/ml-explore/mlx-c), Apple's stable C API, so a Zig binary can drive it without a Python runtime.
- [**mlx-lm**](https://github.com/ml-explore/mlx-lm) (Apple) — the reference Python implementation we cross-check against on every release. Many architecture quirks were nailed down by reading mlx-lm side-by-side.
- [**antirez/ds4**](https://github.com/antirez/ds4) — the embedded engine that serves DeepSeek-V4-Flash via GGUF. Vendored under `lib/ds4/` pinned at commit `613e9b2`; native Metal kernels, official-logits-validated. Salvatore did the hard part.

### Model architectures + tokenizers

- [**llama.cpp**](https://github.com/ggerganov/llama.cpp) — vendored under `lib/jinja_cpp/` for the C++17 Jinja2 chat-template engine plus the bundled [**nlohmann/json**](https://github.com/nlohmann/json) header.
- [**Google Gemma**](https://ai.google.dev/gemma), [**Qwen team**](https://huggingface.co/Qwen), [**Meta Llama**](https://www.llama.com/), [**Mistral AI**](https://mistral.ai/), [**NVIDIA Nemotron-H**](https://huggingface.co/nvidia), [**Liquid LFM2.5**](https://www.liquid.ai/), [**DeepSeek**](https://www.deepseek.com/) — the model families this server runs. The Zig forward paths were written against each project's official reference implementations.
- The [**HuggingFace `tokenizers`**](https://github.com/huggingface/tokenizers) library — the byte-level BPE reference our Zig tokenizer matches against.

### Image + video

- [**stb_image**](https://github.com/nothings/stb) — single-header JPEG/PNG decode for vision input.
- [**libwebp**](https://chromium.googlesource.com/webm/libwebp) — WebP decode.
- [**Black Forest Labs FLUX.2**](https://huggingface.co/black-forest-labs) and [**LTX-Video 2.3 (dgrauet/ltx-2-mlx)**](https://github.com/dgrauet/ltx-2-mlx) — the optional MLX-native image / video generators MLX Core can drive.

### MLX Core (Swift app) integrations

- [**Anthropic swift-sdk**](https://github.com/anthropics/swift-sdk) — the Claude API client the agent loop uses.
- [**Model Context Protocol (Swift SDK)**](https://github.com/modelcontextprotocol/swift-sdk) — powers the MCP marketplace + tool routing.
- Apple frameworks (PDFKit, WKWebView, AVFoundation, AppKit, SwiftUI) — the menu-bar app, browser tool, video player, and PDF attachment pipeline all ride on these.

### Build + ship

- [**Zig**](https://ziglang.org) — the systems language the server is written in. The 0.16 migration was painless thanks to the team's documentation.
- [**Homebrew**](https://brew.sh/) — distribution channel for both the server (`brew install mlx-serve`) and the GUI (`brew install --cask mlx-core`).

If we missed you, please open a PR — happy to add anyone who landed code, fixtures, or a fix here.

## License

MIT
