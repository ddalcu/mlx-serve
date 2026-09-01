# FAQ

## Is mlx-serve faster than LM Studio?

Yes, though it depends what you run. On the v26.8.3 matrix (M4 Max, LM Studio 0.4.19+2, identical MLX weight files, **both engines on shipping defaults**), mlx-serve decodes **+26% geomean** and prefills **+36% geomean** across the four models LM Studio also has.

The shape matters more than the average. On dense Gemma, raw single-stream decode is now a wash (−0.5% on E4B, −0.8% on 31B) — LM Studio has caught up there. The separation is prefill, which is +117% on E4B and +35% on the 26B-A4B MoE, and speculative decoding: on Qwen 3.6 27B mlx-serve loads the checkpoint's MTP head and LM Studio does not, which is **+145%** decode on the same file. Earlier releases quoted a larger geomean by picking the best speculative configuration per model; this one is defaults against defaults, which is the number you actually get.

## Does mlx-serve replace LM Studio?

For most use cases, yes. mlx-serve runs the same MLX and GGUF models, exposes an OpenAI-compatible API on the same kind of port, and ships a native menu-bar app instead of an Electron one. It goes deeper on the API surface than LM Studio's newer compatibility endpoints — fuller Anthropic Messages and OpenAI Responses coverage, plus a WebSocket transport and response compaction — and adds things LM Studio doesn't have: MCP tool calling, agent mode with 10 built-in tools, KV-cache quantization, continuous batching, and the [antirez/ds4](https://github.com/antirez/ds4) engine for DeepSeek V4 Flash.

## Does mlx-serve replace Ollama?

On Apple Silicon, yes — mlx-serve **speaks the Ollama API natively** (`/api/chat`, `/api/generate`, `/api/tags`, `/api/embed`, `/api/pull`, …), so Raycast, Obsidian, Enchanted, Open WebUI, and `ollama-python`/`js` work unchanged: drop in `http://localhost:11234` wherever you had `http://localhost:11434`. The CLI workflow matches too (`mlx-serve run gemma4`, `pull`, `list`, `serve`). Underneath, you get llama.cpp **and** native MLX with the Mac-specific optimizations Ollama doesn't ship (Metal kernels through mlx-c, speculative decoding, shared-prefix KV cache, the Gemma 4 cross-attention drafter).

## Can I run GGUF models on my Mac without Python?

Yes. mlx-serve embeds llama.cpp's inference library (`libllama`) inside the same signed, notarized binary. Point `--model` at any `.gguf` and the server auto-detects the format and routes to the right engine — no `pip`, no venv, no `llama-server` to install separately. DeepSeek V4 Flash GGUFs go through the dedicated [antirez/ds4](https://github.com/antirez/ds4) engine instead, also embedded.

## Does mlx-serve work with Claude Code?

Yes — natively. mlx-serve implements Anthropic's `/v1/messages` endpoint including streaming, tool calling, and extended thinking. Point Claude Code at it with `ANTHROPIC_BASE_URL=http://localhost:11234`. The MLX Core app ships a one-click "Launch Claude Code" button that wires up the env vars for you, and `mlx-serve launch claude` does the same from the terminal. Other agents too: pi, oh-my-pi, OpenCode, Codex, hermes, aider, plus editors like Zed. Setup for each is in [integrations.md](integrations.md).

## Can my Macs share models over the network?

Yes — LAN Sharing, off by default. Turn on sharing where the models live (Settings ▸ LAN Sharing, or `mlx-serve --serve --lan-share all`) and discovery on the Mac that wants to use them (`--lan-discover`). They find each other over Bonjour — no IPs, no config — and shared models appear in every model picker as "model · peer" and in `/v1/models` as `model@peer`, so even Claude Code pointed at `localhost` can run on the other Mac's model. Works for chat and image/speech/music/video/3D generation; models cold-load on demand on the host; only inference is exposed (model management, metrics, and the status page stay private to each Mac).

## What about the OpenAI SDK, Continue, Cursor, Open WebUI?

All work — anything that talks the OpenAI chat-completions or Anthropic Messages wire protocol does. mlx-serve also implements the newer OpenAI Responses API (`/v1/responses`) for clients that want stateful chains via `previous_response_id`, plus a WebSocket transport on the same endpoint.

## Can mlx-serve run DeepSeek V4 Flash locally?

Yes, on 128 GB+ Apple Silicon Macs. Open the MLX Core Model Browser, pick DeepSeek-V4-Flash, hit Download. Since v26.7.12 the safetensors build runs on our own MLX engine rather than through GGUF: 284B with 13B active, 1M context, chat, thinking, tool calls and streaming, about 30 tok/s serial decode on an M4 Max and roughly twice that with DSpark (`--dspark`, the checkpoint's own draft stages). `.gguf` builds still route to the embedded [ds4](https://github.com/antirez/ds4) engine. Agent mode and MCP tools work on DSV4 too. It needs the 0731 release of the checkpoint; the earlier preview is turned away at load.

## What models are supported?

Native MLX dispatch for Gemma 3/4, DiffusionGemma, Qwen 3 / 3.5 / 3.6 / 3.8 / 3-Next, Meta's Muse-Glimmer-30B, inclusionAI Ling 3.0, Tencent Hunyuan 3 (295B), Thinking Machines Inkling Small (276B), poolside Laguna S 2.1, Llama 3.x, Mistral, Nemotron-H, LFM2.5 (including the VL vision builds), and DeepSeek V4 Flash. Anything else as GGUF via embedded llama.cpp — Qwen, Llama, Mistral, Gemma, DeepSeek, Phi, Yi, and thousands more available on HuggingFace. Full table: [models.md](models.md).

## Can mlx-serve run Tencent's Hunyuan 3 (295B) locally?

Yes — the largest open model mlx-serve runs. The 2-bit mixed-precision build (`mlx-serve run hy3`, ~105 GB on disk) decodes at ~26 tok/s with ~235 tok/s prefill on an M4 Max, with thinking, tool calling, and all four API surfaces working. It's recommended for Macs with **more than 128 GB** of unified memory; on a 128 GB Mac it loads and answers correctly, but only a minimal context window (~3K tokens) fits beside the weights — fine for short chats, tight for agent work. The checkpoint's native multi-token-prediction head is supported too (`enable_mtp: true` per request, best with `--mtp-depth 1`).

## How does it compare to MTPLX for Qwen MTP models?

[MTPLX](https://github.com/youssofal/MTPLX) is a focused Python runtime built around Qwen's native multi-token-prediction heads, and it set the bar here. mlx-serve loads the same MTP sidecar artifacts (including MTPLX-published ones) with zero setup and, in a same-machine head-to-head on the identical checkpoint, prompts, and sampling (v26.8.3 vs MTPLX 2.5.3, both on shipping defaults), decodes **+10%** faster with **+17%** prefill and a third of the time to first token (494 ms vs 1528 ms). You also get the rest of the stack — OpenAI/Anthropic/Ollama APIs, GGUF, the agent app — in one binary with no Python.

## Does it support tools / function calling?

Yes, on both API surfaces. The server detects tool-call patterns across architectures (Hermes XML, Gemma 4 `<|tool_call>`, MiniCPM5 V3's attribute-quoted `<function name="…">` XML, raw JSON, ChatML), repairs common Qwen 3.5/3.6 escape quirks, and emits OpenAI-style `tool_calls` deltas in the SSE stream. The MLX Core app ships 10 built-in tools (shell, file I/O, search, browse, web search, memory) and connects to MCP servers from a curated marketplace.

## How does it stay this small / fast?

Zig with direct `mlx-c` FFI — no Python runtime, no Electron, no IPC bridge. The release binary is ~7 MB. Eager warmup at boot page-faults weights and pre-compiles decode kernels (first request 3.5× faster). Multi-turn agent loops reuse KV across turns and skip re-prefilling system prompts via a shared-prefix cache. Tokenize caching turns the second hit on a long conversation into a memcpy.

## Is the inference exact, or quantized output drift?

For greedy decoding (temp=0), mlx-serve is byte-identical to the reference for the first ~30-80 generated tokens, with the long-tail divergence inherent to INT4 float-reduction order (documented in `CLAUDE.md`). For temp > 0, the Leviathan probability-ratio sampler keeps speculative decoding mathematically exact in distribution. Equivalence is pinned by `tests/test_pld_equivalence.sh`, `test_drafter_equivalence.sh`, and `test_kv_quant_equivalence.sh`.

## Can I get byte-identical greedy output across runs and settings?

Yes, if you turn off the things that legitimately reorder float math. Speculative decoding verifies several tokens in one forward, and a wider forward picks different Metal kernels than a one-token forward, so near-tie argmaxes can flip. That's not corruption and not a seed issue (`seed` does nothing at temp 0); it's the same batch-width property every serving engine has. The byte-stable recipe:

- disable speculation: `enable_mtp: false` in the request (or launch with `--no-mtp`), plus `--no-drafter` and no `--pld` if you enabled those
- `--kv-quant off` or `8`
- on hybrid architectures (Qwen 3.5/3.6/3.8, LFM2, Nemotron-H): `--prefix-cache-entries 0`, since a cache hit re-runs the recurrence in a different block size

If your test gate needs exact-string matches, run it with this recipe. Making the speculative path itself byte-identical would mean forcing every kernel's reduction order to be independent of batch width, which is exactly what the fast verify kernels trade away.

## Where does my data go?

Nowhere off your machines. Everything runs locally — no analytics, no telemetry, no cloud calls. The HTTP server listens on your local network interface by default (`--host 0.0.0.0`) so your own devices can reach it; set `--host 127.0.0.1` to make it strictly local, or `--api-key` to gate every non-localhost request. With LAN Sharing on, prompts sent to a shared model travel only across your local network to the Mac hosting that model. Open source under MIT.

## How do I update?

The MLX Core app self-updates by checking the GitHub releases feed. CLI: `brew upgrade --cask mlx-core` or `brew upgrade mlx-serve`.
