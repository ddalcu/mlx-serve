[English](cli.md) · [简体中文](zh-CN/cli.md)

# CLI & server flags

## Ollama-style commands

```bash
mlx-serve run gemma4        # downloads Gemma 4 E4B (4-bit), serves it, chats right in your terminal
mlx-serve pull qwen3.6:27b  # just download (resumable, straight from Hugging Face)
mlx-serve list              # what's on disk
mlx-serve serve             # serve everything you've pulled — models load on demand by name
mlx-serve launch claude     # configure + launch a coding agent CLI against the local server
mlx-serve plugin new toy    # start a plugin from the plugin kit (also: check, fixture, install)
```

Outside plugins (dylibs built with the plugin kit, [plugins.md](plugins.md)) load from `~/.mlx-serve/plugins/` when
`~/.mlx-serve/plugins.json` says `"enabled": true`; `--no-plugins` turns them off for one run.

`launch` supports claude, pi, omp, opencode, opencode2, codex, hermes, and aider. It reads the running server's model list and real context window, writes the agent's config into a dedicated `~/.mlx-serve/<agent>/` folder (never your real agent config), and starts the agent; codex keeps its settings as `-c` overrides over your own Codex home and gets one written file, `~/.mlx-serve/codex/model-instructions.md`, a current tool-contract prompt for local models (`MLX_SERVE_CODEX_MODEL_INSTRUCTIONS=off` drops the override). If the server is down it starts the MLX Core app first. `--model <id>` picks a model, `--print` shows the launch script instead of running, and anything after `--` goes to the agent (`mlx-serve launch codex -- resume`). `launch opencode` detects the installed OpenCode's version (even `--print` runs `opencode --version` to route) and picks its v1 or v2 integration automatically; `launch opencode2` stays as a compatibility alias that forces the v2 integration, which also installs the mlx-serve monitor plugin and needs the server started with `--metrics`. Full per-agent detail in [integrations.md](integrations.md).

Short names, `org/repo` HuggingFace ids, and `name:tag` all work. Models land in a shared `~/.mlx-serve/models` store the MLX-Serve app uses too.

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
| `--serve` | on unless `--prompt` | Start the HTTP server (implied; kept for older scripts) |
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
| `--no-mtp` / `--mtp` | on when a head is loaded (Metal, CUDA) | Disable the native MTP head; `--mtp` turns it on where it is off by default (Vulkan). Dense and MoE alike |
| `--mtp-min-depth N` | `1` | Minimum MTP draft depth (1..8); lifts the automatic cap if needed |
| `--mtp-max-depth N` | auto | Maximum MTP draft depth (1..8); every planner choice stays within the bounds. Equal min/max pins the depth. `--mtp-depth` is the old spelling of this flag. Native head count and remaining tokens can shorten a round. |
| `--mtp-greedy-tail` | off | Sampled requests draft only the first MTP token by sampling, later ones by argmax; pairs with `--mtp-typical`. A model's `mtp_greedy_tail` in `model-settings.json` outranks it |
| `--mtp-history-window N` | `0` (full) | Prompts past 16K tokens only build MTP head history for the last N tokens (windowing costs acceptance on stock Qwen heads) |
| `--dspark` | off | DeepSeek V4's own block-parallel draft stages (~11 GB on top of the model). DeepSeek-V4.1 runs the stages its pack ships unless `--no-mtp` |
| `--ssd-streaming` | off | ds4 / DeepSeek-V4-Flash GGUF only: stream expert weights from SSD instead of holding the whole model in RAM |
| `--ane` | off | Serve dense qwen3_5 and Qwen3 MLX packs on the Neural Engine instead of the GPU, contexts up to 64K (a model's `"ane"` in `model-settings.json` wins); the first load on a Mac compiles its programs (minutes, then cached) |
| `--ane-slots N` | `1` | Requests each Neural Engine model runs at once (1..4); slots share the weights, each adds its own KV and state |
| `--w8a8` / `--no-w8a8` | on with an M5-class GPU and 64 GiB RAM, else off | Image, video, music and 3D models (Krea, Qwen-Image, MageFlow, FLUX.2 klein, LTX, MiniMax-H3, ACE-Step, Music 3, Hunyuan3D) run their transformer linears and, at head_dim 128, attention as rotated int8 x int8 on the Neural Accelerators: faster, slightly lossy, and the int8 copy replaces the packed weights (a 4-bit pack holds more memory, a bf16 pack half). `--w8a8` opts in on a smaller Mac, `--no-w8a8` keeps the packed weights |
| `--prefill-chunk N` | `8192` | Max tokens forwarded per prefill chunk (auto-capped further per model); lower it to cut prefill peak memory |
| `--no-decode-attn-quant` | on | Disable the decode-only requant of dense bf16 attention weights (the "Fast decode for bf16-attention models" toggle) |
| `--kv-quant {off,4,8}` | off | KV-cache quantization scheme (MLX path) |
| `--kv-attn-mode {auto,dense,fused}` | auto | Decode read path for quantized KV: `fused` reads the packed cache in place, `auto` engages it from 8K prompt tokens (only at `--kv-quant 4/8`; per-request `kv_attn_mode` overrides) |
| `--int8-qk` | off | Lossy, NAX GPUs (M5-class and later) only: the causal head-dim-256 prefill attention of dense Qwen 3.5/3.6/3.8 models multiplies int8 Q and K (K with its per-head mean removed), P and V stay half. Flat at 8k tokens, +3% at 32k, +7% at 64k, +14% at 128k on an M5 Ultra (256 GB). Details in [performance.md](performance.md#int8-qk-prefill-attention-nax-gpus) |
| `--lm-head-mxfp8` / `--no-lm-head-mxfp8` | on for quantized Flash Next packs on integrated CUDA | Lossy: a dense bf16 `lm_head` is quantized to MXFP8 (group 32) at load, byte-identical to vLLM's `VLLM_MXFP8_LM_HEAD=1`, and the bf16 copy freed; the head reads half the bytes per token (Flash Next FP4 on a DGX Spark: ~8% faster decode, +0.06% perplexity). On by default only where it was measured: Flash Next (`qwen4_exp`) on an integrated NVIDIA GPU (DGX Spark), when the pack is quantized but its head is dense; other models, bf16 packs, discrete GPUs and Macs opt in. A model's `lm_head_mxfp8` in `model-settings.json` wins; a quantized or tied head is declined by name (`[lm-head]`) when asked for |
| `--llama-kv-quant {off,q8,q4}` | off | KV-cache quantization for GGUF (llama.cpp path) |
| `--llama-cache-entries N` | `4` | llama.cpp sequences per model: requests decoded together in one batch, each keeping its prompt KV warm; each holds a full `--ctx-size` of KV from load |
| `--llama-mtp-drafts N` | `2` | Draft tokens per MTP round for a GGUF with an MTP head (its own, or an `mtp-*.gguf` beside it); drafts only while one request decodes; `0` = off |
| `--llama-ubatch N` | libllama's 512 | llama.cpp prefill batch in tokens; 1024-2048 prefill faster, mostly on MoE, for more scratch memory |
| `--tokenize-cache-entries N` | `4` | Chat-template + tokenize cache size |
| `--max-concurrent N` | `1` | Continuous-batch decode parallelism |
| `--prefill-decode-share S` | `0.25` | Wall-time fraction (0..0.9) decoding streams keep during another request's prefill; the prefill runs 1024-token chunks meanwhile and admits small newcomers at its chunk boundaries. `0` = the old schedule. |
| `--prefix-cache-entries N` | auto | Shared-prefix KV cache entry cap |
| `--prefix-cache-mem N{KB,MB,GB}` | `2 GB`, or one session at the working context on hybrid (Qwen3.5+, GLM-5.3, Nemotron-H, LFM2, Ling), sliding-ring (MiMo) and qwen4_exp models when larger | Shared-prefix KV cache memory cap |
| `--prefix-cache-disk N{MB,GB}` | off | SSD tier: prefixes survive restarts (11K-token restart TTFT 5.9 s → 0.7 s) |
| `--shutdown-persist-secs N` | `45` | On SIGTERM, finish writing the SSD-tier slices not yet on disk for at most N seconds, then exit; 0 disables. Under launchd, set `ExitTimeOut` above it |
| `--metrics` | off | Prometheus `/metrics` + live dashboard panel on `/` |
| `--api-key KEY` | none | Require a key for non-localhost requests (localhost stays open) |
| `--lan-share <all\|id,...>` | off | Share the listed models (or all) with your local network over Bonjour — only inference is exposed, model management stays host-local |
| `--lan-discover` | off | Discover models other Macs share: they appear in `/v1/models` as `model@peer` and requests proxy to that Mac |
| `--lan-name NAME` | hostname | The Bonjour name other Macs see |
| `--dist pipeline\|tensor` | off | Serve `--dist-model` across this Mac and a second one over Thunderbolt 5 RDMA: `pipeline` gives each Mac a run of layers (Qwen3.8-Flash-Next, DeepSeek-V4.1), `tensor` gives each half of every MoE expert (Flash Next); with a MiniMax-H3 pack each video step's rows are split instead, when that is faster. The other Mac runs `mlx-serve worker --coordinator IP:PORT --device rdma_enX` and finds the same model in its own library |
| `--dist-model PATH` | `--model` | The model the two Macs serve together |
| `--dist-coordinator IP:PORT` | none | This Mac's address the worker dials |
| `--dist-device rdma_enX` | none | The RDMA port facing the worker (`ibv_devinfo`) |
| `--dist-split N` | by free memory | Pipeline: layers kept on this Mac, on a full-attention boundary |
| `--model-dir PATH` | none | Discover and serve every model in a folder (LRU resident set). Repeatable — folders merge first-wins |
| `--max-resident-mem N{MB,GB}` | auto | Summed memory cap across loaded models; decides whether a model may load at all (auto = the GPU working-set limit, `0` disables) |
| `--max-resident-models N` | `3` | How many models stay loaded at once (LRU-evicted) |
| `--idle-evict-secs N` | off | Unload models nobody is using after this many idle seconds |
| `--no-warmup-eager` | off | Skip the eager warmup at boot (benchmarking / minimal-footprint deployments) |
| `--skip-mem-preflight` | off | Skip the free-RAM pre-flight on load (the cap above is checked first, and still applies). Either way, a load that would not fit in free RAM first unloads idle models when that makes it fit |
| `--no-tool-autocorrect` | off | Turn off schema-driven repair of model-emitted tool arguments |
| `--log-level` | `info` | Log level (error, warn, info, debug) |
| `--log-file PATH` | `~/.mlx-serve/logs/` | Where the server log goes |
