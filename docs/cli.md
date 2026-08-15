# CLI & server flags

## Ollama-style commands

```bash
mlx-serve run gemma4        # downloads Gemma 4 E4B (4-bit), serves it, chats right in your terminal
mlx-serve pull qwen3.6:27b  # just download (resumable, straight from Hugging Face)
mlx-serve list              # what's on disk
mlx-serve serve             # serve everything you've pulled — models load on demand by name
```

Short names, `org/repo` HuggingFace ids, and `name:tag` all work. Models land in a shared `~/.mlx-serve/models` store the MLX Core app uses too.

## Driving the server directly

What you want for scripts, launchd, or a headless Mac:

```bash
# one model, pinned for the life of the process
mlx-serve --model ~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit --serve --port 11234

# a whole folder, every model loaded on demand by name
# (this is exactly what `mlx-serve serve` and the app do)
mlx-serve --serve --model-dir ~/.mlx-serve/models

# GGUF takes the same flags; the server detects it and routes to the embedded llama.cpp
mlx-serve --model ~/models/Qwen3.5-4B-Q4_K_M.gguf --serve
```

`mlx-serve --help` lists every flag. Defaults are `--host 0.0.0.0 --port 11234`.

## One-shot, without a server

```bash
mlx-serve --model /path/to/model --prompt "What is 2+2?"
```

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | required | Path to the model directory or a `.gguf` file |
| `--serve` | off | Start the HTTP server |
| `--host ADDR` | `0.0.0.0` | Bind address (all interfaces — set `127.0.0.1` for strictly local) |
| `--port N` | `11234` | Port for the HTTP server |
| `--prompt TEXT` | `"Hello"` | Prompt for interactive mode |
| `--max-tokens N` | `100` | Maximum tokens to generate |
| `--temp F` | `0.0` | Sampling temperature (0 = greedy) |
| `--ctx-size N` | auto | Context window size (auto = computed from GPU memory) |
| `--embedding-max-length N` | auto | Per-input token ceiling for `/v1/embeddings` (auto = the model's declared window; over-limit inputs get a 400, never silent truncation) |
| `--timeout N` | `300` | Stall timeout — seconds *without a new token* (a request that keeps producing never times out) |
| `--reasoning-budget N` | `-1` | Thinking token budget (`-1` = unlimited, `0` = no thinking) |
| `--no-vision` | off | Disable vision encoder even if model supports it |
| `--pld` / `--no-pld` | on | Prompt Lookup Decoding (model-agnostic spec-decode) |
| `--pld-draft-len N` | `5` | Max draft tokens per PLD step |
| `--pld-key-len N` | `3` | N-gram match key length for PLD |
| `--drafter DIR` | none | Speculative-decoding drafter checkpoint: a Gemma 4 assistant or a DFlash draft companion. Models that ship a `drafter/` subdir (Muse-Glimmer builds) load theirs automatically |
| `--no-drafter` | off | Never load a drafter, including one shipped inside the checkpoint |
| `--draft-block-size N` | auto | Drafts per round for the drafter (auto-sized to what this Mac's verify path can use) |
| `--no-mtp` / `--mtp` | on when sidecar present | Disable / force the native MTP head (MoE trunks default off) |
| `--mtp-depth N` | `3` | Max tokens drafted per MTP round (adaptive controller tunes within `[1, N]`) |
| `--mtp-history-window N` | `0` (full) | Prompts past 16K tokens only build MTP head history for the last N tokens (windowing costs acceptance on stock Qwen heads) |
| `--dspark` | off | DeepSeek V4's own block-parallel draft stages (~11 GB on top of the model) |
| `--ssd-streaming` | off | ds4 / DeepSeek-V4-Flash GGUF only: stream expert weights from SSD instead of holding the whole model in RAM |
| `--prefill-chunk N` | `8192` | Max tokens forwarded per prefill chunk (auto-capped further per model); lower it to cut prefill peak memory |
| `--no-decode-attn-quant` | on | Disable the decode-only requant of dense bf16 attention weights (the "Fast decode for bf16-attention models" toggle) |
| `--kv-quant {off,4,8,turbo2,turbo4}` | off | KV-cache quantization scheme (MLX path) |
| `--kv-attn-mode {auto,dense,fused}` | auto | Decode read path for quantized KV: `fused` reads the packed cache in place, `auto` engages it from 8K prompt tokens (only at `--kv-quant 4/8`; per-request `kv_attn_mode` overrides) |
| `--llama-kv-quant {off,q8,q4}` | off | KV-cache quantization for GGUF (llama.cpp path) |
| `--llama-cache-entries N` | `4` | Multi-session LRU for llama.cpp (warm multi-doc agents) |
| `--tokenize-cache-entries N` | `4` | Chat-template + tokenize cache size |
| `--max-concurrent N` | `1` | Continuous-batch decode parallelism |
| `--prefix-cache-entries N` | auto | Shared-prefix KV cache entry cap |
| `--prefix-cache-mem N{KB,MB,GB}` | `2 GB` | Shared-prefix KV cache memory cap |
| `--prefix-cache-disk N{MB,GB}` | off | SSD tier: prefixes survive restarts (11K-token restart TTFT 5.9 s → 0.7 s) |
| `--metrics` | off | Prometheus `/metrics` + live dashboard panel on `/` |
| `--api-key KEY` | none | Require a key for non-localhost requests (localhost stays open) |
| `--lan-share <all\|id,...>` | off | Share the listed models (or all) with your local network over Bonjour — only inference is exposed, model management stays host-local |
| `--lan-discover` | off | Discover models other Macs share: they appear in `/v1/models` as `model@peer` and requests proxy to that Mac |
| `--lan-name NAME` | hostname | The Bonjour name other Macs see |
| `--model-dir PATH` | none | Discover and serve every model in a folder (LRU resident set). Repeatable — folders merge first-wins |
| `--max-resident-mem N{MB,GB}` | auto | Summed memory cap across loaded models; decides whether a model may load at all (auto = 80% of the MLX wired limit, `0` disables) |
| `--max-resident-models N` | `3` | How many models stay loaded at once (LRU-evicted) |
| `--idle-evict-secs N` | off | Unload models nobody is using after this many idle seconds |
| `--no-warmup-eager` | off | Skip the eager warmup at boot (benchmarking / minimal-footprint deployments) |
| `--skip-mem-preflight` | off | Skip the free-RAM pre-flight on load (the cap above is checked first, and still applies) |
| `--no-tool-autocorrect` | off | Turn off schema-driven repair of model-emitted tool arguments |
| `--log-level` | `info` | Log level (error, warn, info, debug) |
| `--log-file PATH` | `~/.mlx-serve/logs/` | Where the server log goes |
