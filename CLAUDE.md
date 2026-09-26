# mlx-serve — project context for AI

Native Zig server running MLX-format LMs on Apple Silicon; OpenAI/Anthropic/Ollama-compatible HTTP APIs + native media generation. No Python. This file is the COMPRESSED layer; detail lives one hop away.

## Detail map (read on demand)

- `docs/reference.md` — deep detail: per-file contracts, media-gen schemas, API surfaces, LAN design, observability, embedded engines, arch numerics (dsv4/inkling/H3, Kokoro, runtime LoRA), website/tier list, licensing. Read its section BEFORE deep work on a subsystem.
- `docs/gotchas/{tool-calling,server-http,engine-mlx,models-media,app}.md` — full war stories behind every rule in `## Rules`.
- `tests/CLAUDE.md` — integration-test matrix. `app/CLAUDE.md` — Swift app layout + rules (auto-load in their dirs).
- Skills: `/release` (pre-release checklist, CalVer, CHANGELOG), `/bench` (llmprobe methodology + comparison traps).
- `containers/{agent-shell-mlxserve,guest-kernel}/` — Agent Sandbox guest image + kernel, two pinned artifacts that BUMP TOGETHER. Detail: `docs/reference.md`.
- `website/` — GitHub Pages site + `llm-tier-list/`. Design: `docs/reference.md`. Guards: `tests/test_website_pages.sh`, `tests/website_tier_list_logic.mjs`.
- **Growth policy (ENFORCED)**: this file stays under 100k bytes, short-form and minimal; EVERY rule bullet is ≤ 3 lines. Be PICKY: a rule earns a line only for a real bug or a strong gotcha we actually hit and that someone would repeat. Not a diary: no log of what happened, no measurements, no round-by-round notes, no restating what the code or git history says. Same bar for `docs/gotchas/*.md` (one short story per gotcha) and `docs/reference.md` (one section per subsystem). When in doubt, leave it out; App content → `app/CLAUDE.md`.

## Stack

Zig 0.17 (pinned nightly via `scripts/fetch-zig.sh`; brew 0.16 no longer builds); mlx + mlx-c PINNED SUBMODULES (`lib/mlx-src` v0.32.2, `lib/mlxc-src` 56b2d39 = PR #127) self-built NAX-enabled by `scripts/build-mlx.sh` into `lib/mlx/` (FFI `src/mlx.zig`); jinja.cpp (wangzhaode, Apache-2.0, NOT llama.cpp's) as `lib/jinja_cpp/libjinja.a`; stb_image + libwebp; safetensors; BPE. Embedded engines: ds4 (`lib/ds4`, DSV4-Flash GGUF) + libllama (`lib/llama`, generic GGUF).

## Layout (`src/`)

| File | Role |
|---|---|
| `main.zig` | Entry, CLI flags + subcommands (`run/pull/list/serve/launch`) |
| `cli.zig` | Ollama-grade CLI: alias → HF repo, resumable pull into `~/.mlx-serve/models/<org>/<repo>`, `list`, `run` REPL |
| `launch.zig` | `mlx-serve launch <agent>` (claude/pi/omp/opencode/opencode2/codex/hermes/aider): reads `/v1/models` (models + ADVERTISED context), writes configs into `~/.mlx-serve/<agent>/`, starts the app if the server is down. Swift `CLILauncher`+`AgentConfigs` is the twin (omp `PI_CODING_AGENT_DIR`, codex Responses-only `CODEX_HOME`, hermes `HERMES_HOME`; opencode2 `XDG_CONFIG_HOME` + monitor plugin) |
| `mlx.zig` | mlx-c FFI |
| `model.zig` | Config parse + safetensors loading |
| `tokenizer.zig` | BPE; single special-token splitter (first-byte-bucketed); per-model `digit_group` |
| `transformer.zig` | Forward pass, arch dispatch (attention/MLP/MoE/GatedDeltaNet), quant resolution, custom kernels (`msv_attn_p256`, `verifyQmm` lanes incl. NAX) |
| `generate.zig` | Generation, sampling, PLD/drafter/MTP orchestration, `StallClock`, prefill chunking, loop-stop tiers |
| `reasoning_protocol.zig` | Bounded reasoning/header masks, recovery, and authoritative JSON response routing (see `docs/reasoning-protocols.md`) |
| `chat.zig` | Chat templates (ChatML/Gemma/Llama-3/Jinja2), thinking tags, tool-call parsing/repair/coercion |
| `vision.zig` / `qwen_vision.zig` + `mrope.zig` | Gemma SigLIP / Qwen3-VL ViT + M-RoPE |
| `muse_vision.zig` / `lfm2_vision.zig` | Muse-Glimmer ViT / LFM2-VL SigLIP2-NaFlex tower + projector + tiling |
| `server.zig` | All HTTP: `/v1/*` (chat/completions/messages/responses/embeddings/load/unload/models), media endpoints, `/metrics(.json)`, WS, Ollama glue, `--api-key`, console at `GET /` (`src/html/` as `{s}` args, renders with NO model). Embeddings: BERT + EmbeddingGemma, per-checkpoint pooling, `--embedding-max-length` |
| `lan.zig` | LAN sharing: Bonjour, `SharedSet` + `routeClass` allowlist, `<id>@<peer>` mirroring, streaming proxy. Pure transport |
| `model_settings.zig` | Per-model settings (`~/.mlx-serve/model-settings.json`, keyed by model path): `ctx_size`, `kv_quant`, `mtp`, `mtp_acceptance` (`exact|typical|tokenv3` at the default thresholds), `chat_template_kwargs` (JSON object of template variables, vLLM/llama.cpp vocabulary); read at every load construction site, stamped on `ModelConfig.{ctx,kv_quant,mtp,mtp_acceptance}_override` + `ChatConfig.chat_template_kwargs` |
| `providers.zig` | Upstream OpenAI-compatible chat providers (`~/.mlx-serve/providers.json`): background `/v1/models` probe, `<id>@<name>` rows (`models` = filter, or the list for a listless provider), curl-backed `/v1/chat/completions` proxy. `GET /v1/providers`, `POST /v1/providers/reload` |
| `metrics.zig` | Lock-free zero-when-off observability (`--metrics`): `vllm:`+`mlx_serve:` Prometheus + JSON |
| `ollama.zig` | `/api/*` translation, SSE→NDJSON `Sink`, tags/show/ps, `resolveName` |
| `gen.zig` | Unified media gen: modality-named engine slots, `detectModality`/`peekModelType`, per-request handlers, img2img/edit/LoRA, residency estimators |
| `krea.zig` / `flux.zig` | Image backends (Krea-2-Turbo / FLUX.2 klein 4B+9B); `MixedLinear` infers quant geometry |
| `qwen_image.zig` | Qwen-Image-2.1 (`qwen_image21`; converter emits quantized packs only, bf16 preset owed): block-causal single-stream DiT (two sdpa calls, shared t=0/t modulation), 64-ch /16 VAE, `mage_flow.TextEncoder` at 8B width; 40 steps, real CFG, img2img; text encoder STAGED per request where the pack crowds the GPU (`gen.qwenImageStagesTextEncoder`) |
| `multipart.zig` | RFC 7578 form parsing, zero-copy `Part` (only non-JSON shape: `POST /v1/images/edits`) |
| `mage_flow.zig` | MageFlow Turbo/Edit: flow DiT + DiCo VAE + Qwen3-VL TE; `MfLinear` shared with H3; DiT/TE bf16, VAE f32 (load-bearing) |
| `hunyuan3d.zig` / `hunyuan3d_paint*.zig` | 3D shape + texture paint; converted layouts BAKE OUT per-head QKV interleaves — never "fix" it |
| `acestep.zig` | ACE-Step music (Qwen3 encoder, AdaLN DiT, Euler flow-match, Oobleck VAE 48 kHz; Snake/encode f32) |
| `music3.zig` | MiniMax Music 3: Qwen3-8B global LLM (batch-2 CFG) + depth decoder → hidden states condition a flow DiT (temb as TOKEN) + Snake/DAC vocoder 44.1 kHz |
| `ltx_video.zig` / `ltx_audio.zig` | LTX video (one/two-stage/HQ, i2v, a2vid) + audio VAE/BigVGAN. `LtxVersion` (from `model_version`) keys 2.3-vs-2.5: text encoder, `ff_bias`, `keyframes_abs_pos_embedding` |
| `ltx_diffvae*.zig` | LTX-2.5 DiffVAE decoder (`"decoder":"diffusion"`): geometry/tiling, fused 3D NA Metal kernel, MLX pass. Sampler contract is MEASURED (x0, 1 step, timesteps x1000) |
| `minimax_h3*.zig` | MiniMax-H3 text-to-audio-video: joint video+audio DiT, staged residency, fast recipe, Turbo LoRA, chained windows — detail in `docs/reference.md` |
| `tts.zig` | Qwen3-TTS incl. ECAPA-TDNN voice clone |
| `kokoro.zig` / `kokoro_g2p.zig` | Kokoro-82M TTS + text→IPA G2P (no espeak — GPLv3) |
| `laya.zig` | Laya typed decisions (`POST /v1/decisions`): mmBERT/ModernBERT encoder + decision head, prompt layout and output JSON mirror `laya_mlx`; `.decision` modality slot |
| `marching_cubes.zig` / `glb.zig` / `uvwrap.zig` / `rasterize.zig` / `texinpaint.zig` | Pure-Zig mesh/GLB/xatlas/rasterizer/inpaint (zero MLX, hermetic tests) |
| `preview.zig` / `latent_rgb.zig` / `jpeg.zig` | Opt-in per-step video previews (#208): published latent→RGB map per backend (GENERATED — `tests/dump_latent_rgb_factors.py`), temporal pick + filmstrip, bilinear resize, baseline JPEG. Zero MLX; `zig build preview-test` is the Linux-runnable graph |
| `responses.zig` | Responses API pure data: parser, envelope, `ResponseStore`, compaction |
| `ws.zig` | RFC 6455 framing (server-side) |
| `pld_index.zig` | PLD n-gram index (`findMatch`, `ngramRepeatScore`) |
| `prefix_cache.zig` / `kv_disk_cache.zig` / `kv_disk_writer.zig` | Hot prefix cache + SSD tier (`--prefix-cache-disk`, default OFF); SSD-first mode (qwen4_exp) + its background writer thread |
| `drafter.zig` | Gemma 4 assistant drafter (cross-attention spec-decode) |
| `dflash.zig` | DFlash block-drafter: config-contract detection (root OR nested `dflash_config`), per-request context cache, block forward; DFlash2 adds `selectPath` + 2-tap grouped convs; trunk seam = `ForwardCtx.capture_layers` + `rawEmbedding` |
| `mtp.zig` | Qwen 3.5/3.6/3.8 native MTP head (sidecar OR in-checkpoint `mtp.*` via `resolveMtpSource`; per-weight quant re-solve; committed-history cache) |
| `diffusion.zig` | DiffusionGemma block-diffusion canvas loop |
| `deepseek_v4.zig` | DeepSeek-V4-Flash NATIVE arch (module-owned decode state on `Dsv4Model.dec_state`, NOT the KVCache) |
| `qwen4_exp.zig` | Qwen3.8-Flash-Next (`qwen4_exp`) host side: n-gram hash (splitmix multipliers, per-head primes, eos-segment shifts) + the mmapped 4-bit `ngram_table.bin` row gather. Trunk forward = `transformer.forwardQwen4With` (hyper-connections, PLE, QSA mask over `gatedFullAttnWith`) |
| `scheduler.zig` | Slots, inference thread (sole MLX caller), queues, batching, loop-stop guard, spec wiring, single-flight admission |
| `round_cost.zig` | Measured per-model/width/KV-bucket spec round-cost table (`Transformer.round_cost`), persisted |
| `model_discovery.zig` / `model_registry.zig` | Discovery (two-level org/name, multi-root, GGUF classification, stub meta), multi-model registry |
| `arch/ds4.zig` / `arch/llama.zig` (+ `*_ffi.zig`, `lib/llama_shim`) | Embedded-engine bridges |
| `ane.zig` + `lib/ane/` | ANE prefill offload (`--ane-prefill`, opt-in, LOSSY int8/fp16, M4-and-below): SwiGLU-MLP + fused GDN in-proj MIL programs on the private AppleNeuralEngine framework (`msv_ane_*`, attribution in NOTICE), `/props` `"ane"` + `mlx_serve:ane_*`. Rules: `docs/reference.md` "ANE prefill rules" |
| `rht.zig` / `qmv2.zig` / `gdn_decode.zig` / `mtp_graft.zig` | Prism Hadamard packs (`prism_hadamard_qwen35`): `<linear>.signs` bound to weight handles, `qmatmul` reads `H_block(signs*x)`, the embedding gather gets the inverse; exact 2-bit GEMVs (`qmv2`: M 1..8 over ternary weights, geometry per GPU generation via `planFor`, also non-Hadamard ternary packs via `ternary_2bit`); 2-dispatch GDN decode step; MTP head grafted from the Qwen3.8-27B pack |
| `lora.zig` | Runtime unfused STACKED LoRA (8 max, summed never merged) across QLinear/MixedLinear/MfLinear |
| `status.zig` / `log.zig` | TUI status bar; leveled logging + file sink (`~/.mlx-serve/logs/mlx-serve-<port>.log`, 32 MB rotation) |
| `format_corpus_test.zig` / `tool_traffic_replay_test.zig` / `mtp_replay_test.zig` | Hermetic format corpus + real-traffic replay (`src/fixtures/tool_traffic.jsonl`) + MTP depth-policy replay over recorded acceptance traces (`src/fixtures/mtp_accept_traces.txt`) |

CLI flags: `--model --serve --host --port --prompt --max-tokens --temp --top-p --top-k --ctx-size --config-overrides --embedding-max-length --timeout --reasoning-budget --no-vision --pld --pld-draft-len --pld-key-len --drafter --draft-block-size --no-mtp --mtp --mtp-depth --mtp-history-window --max-mtp-ctx --ane-prefill --ane-image --ane-video --ane-audio --ane-split --dspark --decode-attn-quant --no-decode-attn-quant --kv-quant --kv-attn-mode --prefix-cache-entries --prefix-cache-mem --prefix-cache-disk --max-concurrent --skip-mem-preflight --os-reserve-gib --wired-margin-gib --mtp-head-kv-quant --metrics --api-key --lan-share --lan-discover --lan-name --no-drafter --no-tool-autocorrect --no-prevent-sleep --ssd-streaming --no-ds4-mtp --model-dir --log-level --log-file --version --help`

Sampling defaults for omitted fields: body > launch flags > model `generation_config.json` > hardcoded (1.0/1.0/off). Missing generation_config = wild-sampling signature.

## Building

- First-time: `./scripts/fetch-zig.sh` stages the pinned Zig at `.zig-toolchain/`.
- Zig caches configure-time subprocess output: after a toolchain/SDK change `rm -rf .zig-cache` or the link references a ghost path.
- **ALWAYS `zig build -Doptimize=ReleaseFast`, never bare `zig build`** (Debug 2–4× slower ⇒ fake regressions). `zig build test` does NOT refresh `zig-out/bin/mlx-serve` — rebuild before any live A/B.
- Swift app: `bash app/build.sh`. The two bundle binaries move together.
- mlx + mlx-c: submodules built by `scripts/build-mlx.sh` (deployment target 26.2 → NAX kernels; script + `tests/test_mlx_staged_nax.sh` ASSERT `*_nax` in the metallib). Min macOS 26.2. Bump = checkout tag → rerun → re-diff `src/mlx.zig` externs. Brew: webp ≥ 1.6.0.
- Rebuild Jinja after `lib/jinja_cpp/*.cpp` changes: compile the 7 `.cpp` (`clang++ -std=c++17 -O2 -DNDEBUG -I .`) into `obj/` and `ar rcs libjinja.a obj/*.o`.

## Testing — TDD is mandatory

Order: (1) failing test FIRST, for the right reason; (2) minimum code to green; (3) full suite (`zig build test` 6/6 0 fail + `bash app/test.sh`/`swift build` + relevant `tests/*.sh`); (4) refactor. A live curl is a sanity check, NOT a test.

Feature = unit test that fails without it (+ integration script if HTTP-observable). Bug fix = regression test red→fix→green, red-on-revert. Cross-arch = cover every touched arch. Refactor = characterization test first. UI/build scripts = factor a pure helper and test that.

**Class bugs get class guards.** A live failure revealing a CLASS ships: the instance regression test; a corpus entry or universal invariant in `src/format_corpus_test.zig`; a rule here + story in `docs/gotchas/`.

Hermetic suites: `zig build test -Dtest-filter="format corpus"`, `-Dtest-filter="tool traffic"`. Full matrix: `tests/CLAUDE.md`.

## Releases & benchmarking

- Process/CalVer/CHANGELOG: `/release`. Per-release perf gate = `./tests/bench.sh` on the FINAL tree vs the previous column in `benchmarks.md` (ONE new column per release).
- Methodology: `/bench` (llmprobe measures; bench.sh drives mlx-serve, `--url` probes another engine). Only diff same-methodology cells (llmprobe from 26.8); spec cells are variance — sample across runs/boot-orders, same-session ratios only, name the engine in every win, and an A/B arm is proven by ENGAGEMENT lines in its log. Record = `benchmarks.md` + `~/claude-tmp/bench-<tag>/`.

## Conventions

- Minimal DRY Zig; tests at the bottom of each source file; shell integration tests in `tests/`. Env levers only for paths with two arms worth comparing (lossy/tradeoff), never for an obvious win/fix.
- Inference thread is the SOLE mlx caller (even frees) — media gen posts to `gen_queue`, never a gpu mutex. A long gen blocks chat decode (accepted).
- Concurrent requests batch-decode on pure-attention archs + the qwen3_5 family incl. `qwen3_5_moe` and `qwen4_exp` (`configBatchesDecode`); `--max-concurrent` sizes the submit queue, not a decode gate. The per-slot verdict is a `BatchVerdict` reason: `[batched] slot serial: <reason>` once per slot, `/props` `batching`, `/v1/models` `batched_decode`, `mlx_serve:decode_serial_total{reason}`. Slots entering a batch mid-generation drain lazy pipeline state first.
- A batched group past 1024 KV tokens attends PER SLOT (`perSlotBatchedAttn`: own view, no pad/stack/array mask; causal + quantized-KV kernels eligible). Below that, and on qwen4's QSA reads, the STACKED arm is capped by PADDING WASTE (`groupKeepCount`, `MAX_PAD_WASTE` 1.5, must stay < 2.0); longest slots fall to serial.
- A batched-decode guard that only runs at N=1 pins a shape that never ships: `MLX_SERVE_FORCE_BATCHED=1` at one slot still has `batch == 1` inside the forward (decode-attn-quant + fused QK-norm gates key on it). `tests/test_batched_equivalence.sh` runs a real two-stream arm; both kernels log `[batched] ... engaged (slots=N)`.
- A cold prefill YIELDS to decode ticks at chunk boundaries (`scheduler.interleaveDecodeTick`, up to 8 ticks for a quarter of wall time, `interleaveTicksFor`) and narrows to 2048 while anyone decodes (`companyPrefillChunk`: a chunk boundary is the only yield point, an 8192 chunk was a 33 s stall on the 27B); bounds the GAP, not 4-way aggregate. Idle cost zero, `MLX_SERVE_PREFILL_INTERLEAVE=0` restores. Guard: `tests/test_prefill_interleave.sh`.
- KV reuse via prompt-prefix matching; invalidated after tool calls + pad-only gens; hot cache spills to SSD; RAM invalidation propagates to disk.
- Chat templates live in model dirs; Jinja renders with fallback formatting.
- **Diffs are read by a human. Keep them small.** A comment says what the code cannot (a non-obvious WHY, a contract, a unit), in one to three lines. Never: bug history, measurements, audit trails, review item numbers, dates, "PR #NNN", what an older commit did, or a restatement of the code. That belongs in the commit message and, if it is a rule, in `docs/gotchas/*.md`.
- **No source-scan tests** (`@embedFile` + "this string appears in that function"): they pin text, not behaviour, and pass against their own needles. Test the behaviour or state the rule in a comment. A test comment is one line saying what the bar is.
- **One story per gotcha, one line per rule.** A gotcha entry is the defect, the cause, the fix and the guard in under 20 lines, written once after the work lands. No round-by-round logs, no ledgers, no "what the reviewer said". CHANGELOG: one user-facing sentence per change, no provisional numbers.

## Supported architectures

Dispatch on `config.json` `model_type`. GGUF bypasses MLX → embedded engine by header (`gguf_meta.preferredEngine`: antirez DSV4-Flash + the ds4-only archs `deepseek41`/`qwen4exp`/`glm-dsa`/`glm5-next` → ds4, else llama.cpp).

| model_type | Notes |
|---|---|
| `gemma4`, `gemma4_text` | `language_model.model` prefix; SigLIP vision; clipped linears, PLE |
| `diffusion_gemma` | Gemma 4 26B-A4B trunk, BLOCK-DIFFUSION (diffusion.zig): ≤48-step canvas denoise; PLD/drafter/MTP/batching/prefix-cache never apply; instruct-only |
| `gemma3`, `gemma3_text` | + flat text-only sibling; EmbeddingGemma encoder when `use_bidirectional_attention` |
| `qwen3` | QK norm |
| `qwen3_5`, `qwen3_5_moe(_text)` | GatedDeltaNet + optional MoE, shared expert; Qwen3-VL vision. Qwen3.8 packs serve on this arch |
| `prism_hadamard_qwen35` | prism-ml Bonsai 2 = qwen3_5 behind block-1024 Hadamard rotations (`rht.zig`, `hadamard_block` from `modules[].block`); served in the pack's own numerics: f16 activations over its f16 scales, f32 GDN state (`ModelConfig.actDtype`/`ssmStateDtype`); fused QKV declines; MTP depth 2 |
| `qwen3_next` | DeltaNet |
| `nemotron_h` | Hybrid transformer + Mamba2 (`backbone` prefix) |
| `lfm2`, `lfm2_moe`, `lfm2_vl` | Hybrid gated conv + attention; `lfm2_moe` = sparse MoE past `num_dense_layers` (sigmoid routing, selection-only `expert_bias`, no shared expert); dense MLP `w1/w3/w2` OR `gate/up/down_proj` (probed); `lfm2_vl` = siglip2 tower + projector, NaFlex 64-256 merged tokens, tiling past `max_image_tokens x 2.0` into 512px tiles + thumbnail (`<|img_row_R_col_C|>`/`<|img_thumbnail|>`) |
| `hy_v3` | Hunyuan 3 MoE (expert container probed) |
| `laguna` | poolside Laguna S 2.1 (117.6B-A8.5B): nvfp4 experts, softplus attn out gate, YaRN + sliding, sigmoid routing, UNGATED shared expert. Serial |
| `inkling_mm_model` | Thinking Machines Inkling Small (276B-A12B, REAP-prunable). NO RoPE (RelativeLogits bias + causal short-convs); role-less markers; serial, spec off |
| `muse_glimmer` | Muse-Glimmer-30B (dense VL, GQA 32/2 hd128): sandwich norms (post-norm eps 1e-8), weight-less qk-norm folded into `attnScale()`, sigmoid attn out-gate, NoPE every 4th layer, RMS-normed embeddings, logits softcap 20. Harmony channels + ATEM tools; thinking silently ON with tools / OFF without (`defaultEnableThinking`), thinking-off commits ` to=user<|message|>`. Llama-3 pretokenizer. Vision via `muse_vision.zig` (window attn 3/4 layers, 2D RoPE, CHANNEL-major pixel-shuffle). DFlash via `-assistant` sidecar. Mirror `ddalcu/Muse-Glimmer-30B-MLX-Serve-8bit` |
| `llama`, `mistral` | Standard |
| `spark2_5` | XHToken Spark-X2.5 (1.7B/4B dense, hd 256, 16/4 heads): 3 sliding(512)+1 full layers with PER-TYPE RoPE (sliding full-rotary 1e4, full 25%-rotary 5e6), exact-erf GELU, plain norms, PER-HEAD sigmoid attn out-gate (`g_proj`, `attn_gate_headwise`), fused `q_k_v_proj` split at load (`attn_fused_qkv`), `model.embedding` tied head. DeepSeek-style template (`<think>` open/close by `enable_thinking`, GLM `<arg_key>` tools); tokenizer = DSV4 regex + `Digits` rule ⇒ per-digit. Packs `abenzerps/Spark-X2.5-4B-MLX-{4,8}bit` |
| `k2_horizon` | IFM K2-Horizon dense (0.9B/3.7B/7B/32B; 7B = 36 layers, 32/8 heads hd 128, theta 1e7, 512k ctx): Llama trunk + GROUPED RMS norms (`layernorm_num_groups` → `norm_groups`, `groupedRmsNorm`). Markers `<ifm|think>`/`<ifm|think_fast>`/`<ifm|think_faster>` (one per `reasoning_effort` high/medium/low) + `<ifm|tool_calls>`/`<ifm|arg_key>` decode to `<think>`/GLM tags (`Tokenizer.installMarkerAliases`); the template RAISES on an assistant turn with no thinking field (`serializeMessagesJsonFor`) and on any other effort word (`k2EffortFor`). Thinking default ON (the template opens a marker unconditionally); JSON-schema recovery closes through the pack's own atomic closer (`Tokenizer.markerCloserFor`, resolved from the prompt's opener token in `resolveReasoningProtocol`). A GLM call with a bare NAME and no `<arg_key>` parses (`chat.isBareToolName`). Stop set = generation_config `eos_token_id` list (`<|ifm|im_end|>`). MoVA (`mova_num_experts` > 0) not served. Pack `mlx-community/K2-Horizon-7B-oQ6e` (`pull k2`) |
| `deepseek_v4` | DeepSeek-V4-Flash NATIVE (284B-A13B, 1M ctx; safetensors only). Serial, spec hard-off, prefix cache OFF, single-flight; DSML tools on OUR byte-pinned template; checkpoint 0731+ only; only our converter's layout; mirror `ddalcu/…mixed-2-3-8bit`. DSpark spec |
| `qwen4_exp` | Qwen3.8-Flash-Next (125B-A6B + 51B n-gram + 4B MTP): qwen3_5 GDN+MoE trunk inside 4 hyper-connection streams (`hcRead`/`hcWrite`, norms folded by the converter), n-gram PLE at layer 1 (host gather from `ngram_table.bin`, never resident), QSA sparse attention past 2048 tokens (`qsaMask`), `hyper_connection_mixer` replaces `model.norm`. Module state is READ-ONLY: text slots BATCH-decode (`forwardMoeBatchedDecode`), only the MTP slot is exclusive (`scheduler.slotExclusiveDecode`), prefix cache ON. MTP = the checkpoint's own QSA+MoE layer over the PRE-mixer stream (`Qwen4Mtp`, `MtpHeadRef.qwen4`, opt-in `--mtp`). Vision = the Qwen3-VL tower (`qwen_vision.zig`, `model.visual.` prefix) + M-RoPE through attention AND the indexer, decoding serially. Oracle `tests/dump_qwen4_exp_fixtures.py`, converter `tests/convert_qwen38_flash_next.py`; the 64 GB pack `ddalcu/…-iQ-MLX-3.3bpw` (52 GB resident, exact-bf16 imatrix via `tests/qwen38_flash_next_imatrix_collect.py`, per-layer widths via `qwen38_flash_next_iq_allocate.py`, bar = `qwen38_flash_next_score.py`) |
| `bailing_hybrid` | Ling 3.0 (BailingMoeV3): KDA + MLA hybrid MoE, `layer_group_size` → `full_attention_interval`; KDA = GDN with PER-CHANNEL gate (`_vec` kernel), BOUNDED-SIGMOID gate (`kda_lower_bound` REPLACES softplus), sigmoid out-gate; MLA = naive DeepSeek-V3 (ASYMMETRIC K192/V128 cache; `--kv-quant 4|8` ok); `noaux_tc` routing. Thinking ON; GLM tool tags. Mirror `rapid-mlx/Ling-3.0-tiny-MLX-4bit` |
| `*.gguf` | ds4/llama.cpp; GGUF presence WINS over stray config.json. ds4 DSpark: `--dspark` arms when a `-DSpark-` GGUF sits beside the model (gate keys on `mtpDraftTokens()>1` NOT `hasMtp()`); ~0 net on 0731 |
| `minimax_h3` | MiniMax-H3 text-to-audio-video: joint denoise, 17k+5 frame ladder, 24 fps, two partitions (fl2va/ref2va — `tasks` is the ONLY discriminator), Turbo LoRA, chained windows, fast recipe default-on |
| `laya` | Laya typed-decision checkpoints (no root config.json — classified from `encoder/config.json` + `rl_agent_config.json` by `model_discovery.peekLayaCheckpoint`, `gen.peekModelType` delegates; app twin `DownloadManager.configlessModelType` + `MediaBundle.laya`): ModernBERT encoder (RoPE, GeGLU, global/sliding bool masks) + 2 head layers + marker scorer + act head, fp16; tokenizer.json `Metaspace` pre-tokenizer implemented in `tokenizer.zig` |
| media types | `flux2*`/`krea*`/`mage_flow*`/`qwen_image*`/`qwen3_tts`/`acestep`/`minimax_music3`/`AudioVideo` (LTX 2.3 + 2.5 by `model_version`)/`hunyuan3d*` → gen.zig slots (`mage_flow` has NO root config.json — classified from `model_index.json` by `gen.peekModelType` + `model_discovery.peekMageFlowIndex`, kept in sync) |

Models with `vision_config` but no vision weights disable vision. Embedded-engine detail: `docs/reference.md`.

## Unified media generation

One server, one registry — image/audio/video/3D coexist with chat. Engine slots are MODALITY-named unions on `LoadedModel`; new backend = one union arm + impl file. Gen runs on the inference thread via `gen_queue`; app flow load→generate→unload; headless starts idle. `model_discovery.isMediaModelType` and `gen.modalityFromType` are documented duplication — keep in sync. Downloads → `~/.mlx-serve/models` (Swift + Zig).

- `POST /v1/images/edits` = OpenAI multipart translated by `gen.openaiEditFormToJson` into the `mode:"edit"` JSON body; unhonored fields = NAMED 400.
- LoRAs are STACKED, ONE grammar across image/LTX/H3: `lora_paths`+`lora_scales` (cap 8, `gen.parseLoraFields`), summed at forward — never merged. Resident backends reconcile via `setLoras`; H3 pre-validates (`lora.validatePath`), Turbo = file 0.
- **FLUX real CFG** (`guidance_scale`+`negative_prompt`, `gen.ImageEngine.supportsGuidance`): distilled klein bakes guidance into the weights and takes NEITHER field (1.0 default = the unconditional forward never runs, `flux.ditVelocity` called once); the undistilled "base" 9B checkpoint (`flux2-klein-9b-base`, app preset `flux2Klein9BBase_Q4`) needs it — TWO forwards per denoise step, blended `uncond + scale·(cond−uncond)`. Same DiT geometry as the distilled 9B (checkpoint-derived), so nothing else about loading changes.
- Endpoint/field/backend detail: `docs/reference.md`. Guards: `tests/test_unified_gen.sh` + per-modality scripts; parity via env-gated cos oracles (`tests/dump_*_fixtures.py`).
- **A denoise step is the ANE case the LM prefill seam never was** (opt-in, LOSSY: `--ane-image` Krea, `--ane-video` H3, `--ane-audio` ACE-Step; ONE value `ane.media_offload` set in `main()`): batch job at the compute roofline; ONE compiled 256-row tile, looped (`ane.mediaMlp`), serves every step AND every request size, so the ANE cache never grows per size. M4 Max: Krea 1.30x, H3 1.22x/step, ACE-Step 1.33x; M4 base ACE-Step 1.58x. Tables: `docs/reference.md`.
- **The media share is CALIBRATED once per (chip, model), then STICKS** (`ane.planMediaOffload` → `calibrate`): block 0 alone, compiled at the probe's seed, timed on the ANE and GPU over the same 4096 rows, the GPU fed the dtype the MLP really sees (bf16 ones read ACE's f32 GPU 27% fast); later builds reuse the set's `calibrated share=` tag (`ane.cachedShare`; an explicit `--ane-split` never is), since re-solving recompiled + pruned on size and probe jitter.
- **"Free disk" is what the OS will GRANT, not statfs** (`msv_volume_free_for_use` = `volumeAvailableCapacityForImportantUsage`; `kv_disk_cache.volumeSpace` + the ANE cap read it, statfs is the fallback): purgeable space is released on demand, and a volume df called 36 GB free had 117 GB — the SSD tier refused every persist on it.
- **The ANE compile cache is capped by FREE DISK and pruned per LINEAGE** (`msv_ane_cache_lineage`: same seam+shape, other share → gone on the next cold compile; byte cap = min(40 GB, volume room − 8 GB reserve)): a fixed cap above free space is not a cap, and two small-disk boxes shipped `ready: N/M`. The gate bills the build's f32 transient against a 4 GB swap floor (`mediaGateRefusal`), refusing by NAME.
- The seam is the channel-mode one (`ane.packUnitPlanes`/`readPlane`/`dequantToHostF32` shared with `transformer.zig`): ANE holds gate/up channels [0..k) + the matching down K-slabs, GPU the complement, partials ADD. H3's fc1 is FUSED, so its complement is TWO row views (`aneBuildRest`).
- **The ANE graph is fp16 END TO END, so a partial-sum seam must SCALE** (`ane.OUT_PLANE_SCALE` 256, folded into the `up` copy at build time, multiplied back on read — exact: per-row int8 puts it in the row scale, and `up` is linear into `silu(gate)*up`). Unscaled, H3 saturated to INF from block 36 and rendered BLACK; Krea only lost precision (cos vs GPU 0.996 -> 0.9993).
- **A LoRA-attached block DECLINES** (`aneBlockEligible` in both): the adapter is summed at forward from the FULL activation, half of which never leaves the ANE program. Turbo binds `blocks.N.mlp.fc1/fc2`, so H3's fast path is GPU-only until the LoRA is folded into the int8 snapshot.
- **GPU work built AFTER a blocking ANE wait is serial, and a small piece of it is LAUNCH LATENCY, not rows** (`ane.mediaMlp`): a 23-row tail after the tile loop cost ACE 30 s 7% of diffusion. Every GPU piece goes out with the complement BEFORE the loop; a partial tile pads onto the ANE only past `tail > T x (1 - share)` (`ane.mediaTilePlan`).
- **H3 stages the DiT per REQUEST, so the ANE build is paid per request** (cold 32 s, warm 8 s + a 4.9 GB int8 copy): noise against a 275 s/step dense render, dominant on a short one. Caching the engine across requests is owed.

## HTTP APIs

- **OpenAI chat/completions + Responses**: usage ALWAYS carries `prompt_tokens_details.cached_tokens` (ONE `formatChatUsage`); thinking opt-ins = `reasoning_effort` OR `enable_thinking`, top level or in `chat_template_kwargs` (`reasoning_budget_tokens` outranks); `n>1` 400s. `/v1/responses`: every SSE event has `sequence_number`, `background:true` → 400, stateful via `ResponseStore`, WS via Upgrade, NO `[DONE]` on WS.
- **Continuing a partial reply** (`chat.continuationRequested`/`continuationPrefill`): trailing assistant message becomes a PREFILL (whitespace trimmed). `/v1/chat/completions` takes `continue_final_message` EXPLICITLY; `/v1/messages` INFERS; `/v1/responses` neither. Tool-call replies never continue; in the tokenize-cache key; ds4 can't (`continuationRejectReason`).
- Context-overflow 400s name BOTH counts (`contextOverflowMessage`); the app renders it as a card.
- **Anthropic `/v1/messages`** (Claude Code): typed blocks, `input_schema`→`parameters`, stop-reason map incl. `stop_sequence` echo, full SSE block lifecycle. Launcher env: `ANTHROPIC_BASE_URL` + dummy keys + `ANTHROPIC_DEFAULT_*_MODEL=mlx-serve`.
- **Ollama `/api/*`**: pure translation, no-model endpoints answered pre-scheduler; `resolveName` handles `name:tag`. One path must never register under TWO ids (`registry.peekByPath` — double-residency OOM).
- `/v1/models` rows carry `context_length`+`max_model_len` at TOP level twinning `meta.context_length` (#188): discovery clients never read `meta.*`. Both emitters. Guard: `tests/test_models_capabilities.sh` [4b].
- **Providers**: `<model>@<name>` from `~/.mlx-serve/providers.json` proxies `/v1/chat/completions` ONLY (other surfaces = named 400) through curl with the provider's key; rows ride `/v1/models` with a `provider` badge while the probe answers, a non-empty `models` list FILTERS the provider's list (and IS the list when it has none), none when unreachable. App: "Pick…" fetches the list into a checklist. Never shared to the LAN. Design: `docs/reference.md`.
- **LAN sharing**: proxy is a TRANSPORT; keyless gate = `routeClass` × `SharedSet`; `<id>@<peer>` mirroring; loops impossible by construction (self-token + tunnel marker, one hop). Design: `docs/reference.md`.
- **Observability** (`--metrics`): zero cost off; TTFT at prefill completion; live tok/s via ONE atomic per tick. `--api-key`: loopback exempt; `/health`+OPTIONS open; `constTimeEql`. No admin surface.

## Tool calling (server pipeline)

With `tools`, tokens buffer for detection (all tag families + raw JSON); thinking buffers separately. Parse chain: strict → tolerant repairs → truncation salvage, then the ONE chokepoint `server.parseToolCallsForRequest` = parse → inferred-name filter → parallel clamp → buried-param hoist → schema coercion (the last two gated by `--no-tool-autocorrect`; emitted `arguments` ALWAYS valid JSON). Serialization `chat.serializeMessagesJson`: role "tool" native, args as JSON STRINGS, every string via `appendJsonString`. Streaming: full args in ONE SSE delta, thinking → `reasoning_content`. Fallback `fallbackFormatChat`. Client: `app/CLAUDE.md`.

## Prompt-based skills / downloads / debugging

- Product skills: `~/.mlx-serve/skills/*.md` (frontmatter trigger substring → body into system prompt; `SkillManager` rescans on mtime).
- `DownloadManager`: streams to `.partial`, Range resume, 3 retries, cancel preserves partial, size-matching files skipped.
- Server log `~/.mlx-serve/logs/mlx-serve-<port>.log` is THE post-mortem file (`--log-level debug`). Grep: `jinja error:`, `[cache]`, `<- N+M tokens`, `tool_msgs=`, `[spec-stats]`, `spec-gate:`, `[loop-stop]`, `[lan] proxy`, `[disk-cache]`, `[hot-cache]`, `[admission]`, `[providers]`. Capture: `MLX_SERVE_RAW_DUMP_FILE=<abs>` → `tests/harvest_tool_traffic.py`. Reproduce tool bugs `stream:false` first; `pkill -f mlx-serve` between KV-poison tests.

## Rules (distilled gotchas — stories in docs/gotchas/; every bullet ≤ 3 lines)

### Tool calling & formats (→ docs/gotchas/tool-calling.md)

- **Control bytes**: ONE raw byte <0x20 in history kills the strict render → SILENT `fallbackFormatChat` (model loses its stop token). Everything through `appendJsonString`; wrong-family tags out ⇒ suspect silent fallback first.
- **A NUL byte in any message TRUNCATED the rendered prompt** (`grep -a` output pasted into an agent turn): the `\u0000` escape decodes back to a real NUL inside jinja and the shim's bare `char*` was read with `std.mem.span`, so the prompt ended mid-user-turn with no assistant header and the model answered with an immediate EOS (`P+0 tokens … [stop]`, empty agent turn). `jinja_render_chat` returns its LENGTH; byte-level BPE round-trips 0x00. Tell: the same `prompt=` count on consecutive turns.
- **A `system`-role message past index 0 FOLDS into the leading system message on `/v1/messages` and `/v1/responses`** (`chat.foldSystemMessages`): Claude Code / Codex carry a non-leading system that way, every served template raises on a system turn that is not first, and the raise is the SILENT generic fallback.
- **`developer` is OpenAI's spelling of the system turn and every template we serve reads it as `system` or raises** (`chat.canonicalRole` at both parse sites — chat + Responses; issue #348: pi sends it for reasoning models unless `supportsDeveloperRole:false`, Qwen's template raised, the SILENT generic fallback lost the stop token and pi "stalled").
- **A `chat_template` value can be a POINTER** (`{% include 'chat_template.jinja' %}`, transformers ≥5, issue #169): `chat.isIncludeStub` reads an include-only value as "no inline template" so the sidecar loads. Grep the log for `jinja` first.
- **A JSON-dialect call cut INSIDE the object still names its tool** (`chat.truncatedJsonCallName`, depth-1 `"name"` only): the Hermes JSON arm had no NAME + `{}` salvage, so a 4 KB `edit` that hit EOS mid-string vanished as an empty `stop`. Shipping the raw markup as content is NOT the fix.
- **A template that iterates any non-string `content` raises on `null`** (LFM2-VL 1.6B): every tool-call turn fell to the generic fallback, and with an image it became a 400. `EmptyContent.empty_string` is sniffed for it like Harmony (`if content is not string`).
- **Model-mangled arg JSON**: strict parse fails → `looseRepairToolCallJson`; never drop the whole call. **Truncated opener**: recover NAME + `{}` — NEVER ship partial values.
- **Delimiter-drop tolerance** (hy3): a tag parser never bails on ONE missing delimiter; plural-wrapper recovery keys strictly on the SUFFIXED `<tool_calls:` form.
- **A `<tool_call>` body carrying `<function=` is the XML dialect and is read FIRST** (qwen 3.5+ template mandates it): the JSON branch once snapped a package.json out of a `content` parameter and shipped its `"name"` as the tool. A parameter VALUE never decides the call.
- **A `<parameter>` VALUE may spell the dialect's own close tags** (`hermesValueEnd` = LAST `</parameter>` before the next opener; `hermesParamSpanEnclosing` skips values when scanning for `</function>`/`</tool_call>`): first-occurrence scans emptied or dropped a `content` documenting the format. The TRUNCATED-call branch reads `<function=` first too; `isJsonNumber` (JSON grammar, not `parseFloat`) decides literal vs string.
- **A hand-written partial-prefix ladder DERIVES its test from the marker** (MiniCPM5 `<function`): an enumerating test copies the array's gaps — `<funct` was missing from both. A gate check is a strict SUPERSET of its parser, else a real opener flushes AND duplicates.
- **A `</think>` inside a tool ARGUMENT is payload** (`chat.thinkCloseIsToolCallPayload`): decline a close whose nearest preceding tool opener is still OPEN AND whose block closes afterwards — both halves load-bearing.
- **A short exact cycle convicts on SPAN, not reps** (`degenerate_loop_min_span` 128, `exactCyclePeriod`): 16 reps at period 1 is a 24-wide map wall row `"1111…"` on a per-digit tokenizer, or a zeroed array at period 2. Legit code repeats short cycles; a loop runs on.
- **Near-repeat loop tier** (`generate.isNearRepeatTailLoop`): 1024-token window, THREE ratios must ALL be low (distinct ≤0.12, 4-grams ≤0.35, novelty ≤0.10); the third (PROGRESS) tells a loop from procedural code. Bar errs toward acquittal.
- **A loop tier convicts on SPAN, because a low-entropy FILE ends and a loop does not** (`near_repeat_min_span` 4096, `degenerate_loop_long_min_span` 1024): a 27B's tile maps of identical `100000000000000000000001` rows are a period-28 cycle AND a zero-novelty window by content, and no content measure can tell them apart from a loop. A real loop still dies in 13-50 s instead of max_tokens.
- **Loop-stop cuts are intentional stops**: `finish_reason "stop"` (`scheduler.loopStopReason`), `[loop-stop]` logged, tool parsing suppressed; `finish_details:{"type":"repetition_loop"}` on chat+completions; non-streaming trimmed to the span start (`loopTrimmedIds`, `MLX_SERVE_LOOP_TRIM=0`). Guard: `tests/test_loop_stop_signal.sh`.
- **A container param string with a key repeated at the SAME value still coerces** (#402, `parseContainerAllowingRepeats`: first-wins and last-wins parses must agree, a CONFLICTING repeat stays a string): Qwen Code's `questions` array spelled `header` twice per item and both strict parses rejected it.
- **Types come from the SCHEMA, never the value's spelling** (`coerceToolArgsToSchema`; undecidable → untouched). ONE chokepoint `server.parseToolCallsForRequest` — never call `chat.parseToolCalls` from a handler.
- **Buried required params**: `hoistMisplacedRequiredParams` lifts only on all-schema-read unanimity. Pristine args = the parse layer is innocent.
- **Heuristic raw-JSON inference must name a DECLARED tool** (`filterInferredBySchema`); explicit tag calls never filtered; new heuristics set `.inferred`; not autocorrect-gated.
- **Hard invariants (replay-pinned)**: emitted args ALWAYS valid JSON; every converter escapes + dedups; coercion never worsens conformance; broken output stays honest. Harness: `src/tool_traffic_replay_test.zig`.
- **A Hermes param value keeps its own whitespace** (`chat.stripHermesValueFraming`): the template frames it with EXACTLY one newline per side and that is all the parser strips — trimming ate an `old_string`'s indentation. Padded SCALARS still type from their spelling (`isJsonLiteral` probe).
- **Gemma dropped `<|"|>`**: rich bare values run to the CONFIRMED closing delimiter or top-level final `}`, never the first `,`/`}` inside markup.
- **Think-tag leaks**: strip pos-0 unclosed openers; `trimTrailingThinkClosers`; universal no-tag-leak corpus invariant covers new entries.
- **Unparsed tool markup never rides out on a TOOLS request** (`chat.trimLeakedToolMarkup`): one cut at the first wrapper opener, applied ONCE; `/v1/messages` cuts at emission; streaming end-flush concatenates BEFORE cutting. No `tools` = raw text on both paths, like vLLM/llama.cpp. Guard: `tests/test_no_tools_markup_passthrough.sh`.
- **Streaming + tools + thinking**: buffer until pattern resolution; reasoning streams INCREMENTALLY on the tools path (`.hold_thinking` + `chat.unstreamedReasoning`, never a resend; a BUDGET keeps buffering). Pinned by corpus streaming replay + `tests/test_messages_stream_thinking_tools.sh`.
- **The streaming think gate scans with a CURSOR** (`chat.ThinkScan`, O(n)); close tag cannot latch once a `tool_call` substring is seen — falls back to the full scan.
- **Pythonic tool calls (lfm2/LFM2.5)**: call-expression grammar; TYPES live in the SPELLING; marker-gated ⇒ additive; truncation ships NAME + `{}`.
- **A template can open `<think>` unconditionally** (LFM2.5): whether a prompt ends inside a think block is a property of the RENDERED BYTES (`server.promptOpensThink`), never ANDed with `enable_thinking`; thinking-off + prompt-opened ⇒ a stream arm DROPS the block.
- **A model can open its OWN think block when the template did not** (Gemma 4 `<|channel>`+`thought`, a bare `<think>` token on LFM2.5-8B-A1B, muse's ` to=self<|message|>` header at position 0): the plain arm latches via `chat.modelThinkOpener` (checked BEFORE the marker skip, which swallows `<think>`) and `promptOpensMuseHeader` arms the header skip at stream start. Bar = the stream/non-stream split invariant on a template that opens nothing; `tests/test_smoke_matrix.sh` runs it per arch.
- **`in_think_block` seeds from `prompt_opened_think` ALONE at every stream site, never the request flag** (LFM2-VL shipped its whole answer as reasoning with empty content). Class guard = source scan; bar = stream-vs-non-stream byte invariant (`tests/test_lfm2_vision.sh` [6/7]).
- **An integration assertion that a MODEL must think/answer/call is a checkpoint expectation**: assert the INVARIANT, branch on the model's choice.
- **A contract COMMENT is read as a spec** — pin it with a test or it gets filed as a bug (#94).
- **Assistant-history reasoning round-trips** (`Message.reasoning_content`, OMITTED when absent): reasoning-persisting templates (laguna, inkling) otherwise render nothink signatures from turn 2.
- **A template can raise_exception on OUR extra-context values** (Inkling, Qwen3.8): `serializeExtraContext` sniffs the family; tool-call `arguments` stay OBJECTS; history tool_calls carry `"id"`. Qwen3.8 = THIRD effort vocabulary (`xhigh|medium|low`); the thinking-off refusal is sniffed on the refusal string and only a refusing template gets `noThinkTailSuffix`.
- **A transcribed chat template is worth what it is PINNED against; whitespace is a token-level contract** (`tests/dsv4_template_ab.py`, byte equality over the shapes the server emits).
- **Inkling channel markers are single special tokens, but tool turns need more**: `streamShouldBufferForTools` buffers on the invoke marker and HOLDS bare-identifier segments after boundary markers.
- **A muse channel HEADER is ordinary text between single-token markers**: every stream surface HOLDS an unresolved header (incl. sub-`to=` prefixes) and drops header tokens (`museHeaderSkipNext`); `to=self`→reasoning, `to=user`/bare→content, else→ATEM tool (`splitMuseChannels`/`museStreamVerdict`). ATEM strings RAW, JSON keeps type.
- **An error echo teaches the model the error** (Inkling name salvage): NAME = trailing identifier run; body = BALANCED JSON; a parsed NAME never contains `<|` (corpus invariant); bare-JSON inference never runs on invoke-marker text; `applyFamilySamplingDefaults` fills top_p 0.95 when no generation_config.
- **Thinking-off is enforced in the PROMPT; generated reasoning is ALWAYS delivered** (`chat.noThinkTailSuffix`: muse commits ` to=user<|message|>` without tools, LFM2.5 gets `</think>` appended). Every delivery site splits via `splitThinkBlock(text, true, …)` — no strip-and-discard path.
- **Muse renders round-tripped reasoning as `to=self` HISTORY, so PRIOR-turn reasoning is dropped from the prompt** (`chat.dropPriorTurnReasoning`): only assistant messages AFTER the last user message keep `reasoning_content`. laguna/inkling untouched — per-family.
- **A template that reads `preserve_thinking` gets it FALSE unless `chat_template_kwargs` say otherwise** (`serializeExtraContext`): Qwen3.8 keeps EVERY turn's `<think>` when undefined, and round-tripped agent reasoning re-seeded loops. Kwargs layer request > model settings > generation_config > arch (`resolveChatThinking`, `mergeTemplateKwargs`); `messages`/`tools` are never kwargs.

### Server, HTTP, lifecycle (→ docs/gotchas/server-http.md)

Reasoning, budgets, agents:
- **A reasoning budget is enforced at DECODE** (`armThinkBound` → `think_bound`, `thinkBoundTick`): early-stop line + atomic closer committed as ONE forward (`commitForcedTokens`); bare-`<think>` family only; all three surfaces (`reasoning_budget_tokens`). Guard: `tests/test_reasoning_budget_stream.sh`.
- **Effort budgets are pi's ladder** (`responses.effortBudget`): minimal 1024, low 2048, medium 8192, high/xhigh uncapped; the word maps to a budget only where the bound can arm (`effortWordOnly`).
- **Agent output share is ctx/2; compaction reserve ctx/4 capped 20000** (`launch.budgetForContext` + `compactionReserve`, Swift `AgentBudget`, the pi extension JS — three copies); pi `settings.json` is MERGED, opencode2 carries `compaction` + `limit.output`.
- **A launch below the agent's context floor WARNS, never blocks** (`launch.contextFloor` / `AgentBudget.contextFloor`: claude 64k, opencode 32k, others 16k; app NSAlert BEFORE launch).
- **A constrained JSON payload offset is AUTHORITATIVE** (`reasoning_protocol.Delivery`): nothing after it is re-parsed or cleaned. Guard: `tests/test_json_schema_protocol_routing.py`.
- **Schema-mask surfaces share one policy** (`schemaMasksThinking`, #331): defer only across bare `<think>` + atomic closer with no finite budget; tools present = no mask. Guard: `tests/test_json_schema_thinking.sh`.

Request parsing + media:
- **`messages.deinit` frees the Message array and NOTHING it points at**: media is owned by ONE `server.RequestMedia`; `Message` BORROWS; slots are INDICES.
- **Undecodable media anywhere in the conversation is a NAMED 400** (`IMAGE_DECODE_REJECT`; remote URLs never fetched); `stop: ""` skipped; schema-less `json_schema` 400; empty embedding input 400; Ollama promptless generate = load handshake. Guard: `tests/test_api_edges.sh`.
- **Latest-turn user media on a tower-less model is refused by NAME** (`mediaRejectReason`), older/tool media replaced by a note the agent can read (`dropMedia`); non-text model on a text surface 400s BEFORE prefill (`textGenRejectReason`; new surface → `isTextGenRoute`, new modality → `modalityFromType`).
- **Every media item renders as the model's OWN template placeholder** (`chat.appendMediaContentParts`), all items encoded in prompt order, the k-th placeholder expanded to item k's run (`expandMediaPlaceholders`); a count/kind mismatch is a NAMED 400, never a shift. Encoder outputs are cached by pixel hash (`VisionEncoder.emb_cache`), or history re-encodes every turn. Text parts JOIN in order (#195).
- **A dispatch field must be readable from every body SHAPE** (`parseModelFromRequest(body, content_type)`); header parameter lookups key at a boundary (`name=` vs `filename=`); binary bodies are not logged (`bodyIsText`); request ints clamp (`parseRequestSeed`, `clampJsonI32`).
- **A client-supplied PATH is proven on OUR side of the mlx boundary** (`lora.loadFile` stat → 400). `/v1/images/edits` forwards the LoRA fields (#268); unhonored = NAMED 400.
- **Hand-written error text is not JSON**: escape at the SINK (`jsonEscapeMessage`), truncate on a UTF-8 boundary. NO model-byte string is guaranteed UTF-8 — sanitizing lives INSIDE every escaper (`chat.utf8Next`); logprobs `bytes` keeps exact bytes.

Sampling + logprobs + streams:
- **A `seed` binds EVERY sampler with a fresh key PER DRAW** (`generate.seedKey` + `SamplingParams.draw`).
- **Logprobs are the MODEL's distribution**: pre-temperature, `logits - logsumexp` in f32 (`computeLogprobs`), ids travel with values, entry belongs to the RETURNED token (`pending_logprob`), pre-GRAMMAR-mask under `response_format` (`nextConstrained`). Bar: temp-0 rank 1 == chosen.
- **Streaming logprobs**: a SIBLING of `delta`/`text` (`ChunkExtras`), ONE collector `StreamLogprobs` with a high-water mark, shipped once; `logprobs.content` describes `message.content` (`contentTokenRange`, `skipToContent`, `dropPending`). `/v1/completions` logprobs is an INTEGER + four arrays. Guard: `tests/test_logprobs.sh`.
- **A client stop string ends the ANSWER, never the reasoning** (#549, `chat.answerStopIndex`): a match counts only where the split delivers it as content, judged on the text UP TO the match, so stream and non-stream cut the same byte. Guard: corpus `a stop string ends the answer`.
- **Stream and non-stream are the SAME BYTES**; only an all-whitespace lead chunk may be withheld (`streamContentLead`). Also agree: spent budget WITHHOLDS the rest; tool replies carry `visibleToolPreamble`; disconnect = `client_disconnect`; stop sequences cut at INDEX (`stopSequenceCut`).
- **A client cannot time our stream**: final-chunk server `timings`; the `include_usage` chunk ships `"choices": []`. Guard: `tests/test_loop_stop_signal.sh`.
- **Liveness is a property of the SOCKET** (`beatStreamKeepalive`, 5 s byte-silence; `StreamHeartbeat` mirrors `StallClock`). GAP: Ollama sink drops SSE comments. `--timeout` is a STALL timeout; `toolCallFinishReason` preserves "length".
- **A long job's abort must not depend on the RESPONSE SHAPE** (`gen_sse.StreamCtx.stream`, `Conn.peerClosed`).
- **Grammar**: every state has a legal byte; the mask never walks the vocabulary (#380, `token_mask.buildMask`, `nextConstrained` lazy); NO whitespace OUTSIDE the root, the model's own layout inside (`MAX_FREE_WS` 16); token→bytes tables are per-MODEL (`grammarTokenBytes`). Guard: `tests/test_json_mode_multi_model.sh`.

Loading + residency:
- **A missing tensor is a load ERROR, never `unreachable`** (#217, `error.MissingWeight` → named 503); load failures cross the inference thread by NAME (`req.error_name` → `loadErrorFromName`, #144).
- **The load preflight's "available" is capped by Metal's working-set limit** (`effectiveAvailableBytes(…, mlx.maxRecommendedWorkingSet())`): a lowered `iogpu.wired_limit_mb` binds below free RAM and weights past it OOM in warmup, leaving a server that refuses every request.
- **`modelDiskBytes` bills the shards the INDEX names** (#274); an index naming NO shard on disk is STALE, not a filter (`indexShardSet` → null).
- **Tokenizer-derived config is set in ONE place for both load paths** (`ModelConfig.applyTokenizer`: EOS merges, LFM2 image tokens): the registry path once lacked them and app-loaded models misplaced every image.
- **A launch flag that shapes a LOAD is retained on the Scheduler** (`ensureLoaded`'s cold-load `LoadRequest` is a second site; `--drafter` via `coldLoadDrafterDir`). Guard: `tests/test_cold_load_launch_flags.sh`. An arg loop needs an else branch (`classifyUnparsedArg`); guard `tests/test_headless_spec_flags.sh`.
- **Per-model settings** (`model-settings.json`, #269) are stamped at BOTH load sites (`applyModelSettings`); read `manualContext(config)` / `configuredKvQuantFor(config)`, never the raw flags. Guard: `tests/test_model_settings.sh`.
- **Endpoint EXISTENCE never depends on model state**: 404 BEFORE resolution (`ROUTE_PATHS`); a status route never reaches `ensureLoaded` (`handlePropsNoModel`); `GET /` renders with no model. Guards: `tests/test_route_404_no_load.sh`, `tests/test_index_page.sh`.
- **`--model-dir` is REPEATABLE**, merged FIRST-WINS; app reads via `readRoots`, writes via `ownedRoots`. Guard: `tests/test_multi_model_dir.sh`.
- **A reload FREES retained CPU state off-mutex while `.loading`** (`releaseRetainedCpuState`): a refcount-less reader takes the mutex AND skips `.loading` entries.
- **Embedded engines**: ONE persistent session per model (`ds4_session` + `session_busy`); ds4 in-checkpoint MTP only when the GGUF declares `nextn_predict_layers` (`embedded_mtp`); the `Tokenizer` is a STUB — count via `server.encodeText`; embeddings refuse by NAME; a llama session trim is FALLIBLE (#286/#287, cold-prefill on refusal). Guard: `tests/test_ds4_serve.sh`.
- **An embedding SUB-BATCH is its own forward** (cache reset per sub-batch). Guard: `tests/test_embeddings.sh` [4c].
- **A READY model never advertises LESS than its stub** (`readyHasChat`). Default bind 0.0.0.0 WARNS (`shouldWarnOpenBind`).

Memory bills + admission:
- **KV is billed per CACHING LAYER at the arch's OWN K/V widths** (`kvBytesPerToken` ← `attnCacheLayerCount`; only `.attention` blocks of a `layer_block_types` hybrid); STORED and SCORED widths are two parameters (`prefillScoreHeadDim`).
- **The prefill working set includes the arch's OWN streams** (`prefillStreamBytesPerToken`: GDN q/k/v across cadence layers, MoE `top_k` replication) plus a RUNTIME floor (`PREFILL_RUNTIME_FLOOR_BYTES`); a lazily copied side-channel state must be NAMED in the cadence eval (`evalCadencePoint`, #366).
- **The guard bills what the ARCH reads** (`prefillAttnKeys`, `prefillFfnWidth`, dsv4 own term); a bill follows its buffer (QSA ring once per slot, `qsaRingBytes`); qwen4 MTP head KV billed at its effective width (`--mtp-head-kv-quant` opt-in).
- **The prefill CHUNK is a machine decision** (`resolvePrefillChunk`: widest rung whose reserve ≤ a quarter of the budget; `--prefill-chunk` wins); on qwen4 per-REQUEST and re-chosen per chunk (`chooseRequestPrefillChunk`, `adaptivePrefillWidth`). Vision chunks like text (#197).
- **Auto-context**: KV at the CONFIGURED width, activations ONCE (`prefillTransientReserve`), PINNED at load (`pinAutoContext`); ask `getEffectiveContextLength`. Load-time bills run inside `Scheduler.init` — use `configuredKvQuant()` + the constant `CTX_SIZING_CACHE_RESERVE`.
- **The GPU ceiling sees EXTERNAL pressure** (`currentGpuMemoryCeiling`); Metal OOM is UNCATCHABLE; embedded-engine requests are exempt (`mlxMemoryGuardApplies`).
- **Every request is re-billed against LIVE memory before prefill** (`slotHoldsForMemory`): no fit + company = back to `pending`; alone it proceeds. Guard: `tests/test_memory_pressure_4way.sh`.
- **Siblings are billed against un-allocated promises and the plan leaves the OS an eighth of RAM** (`PromiseLedger`, 2 GB `SIBLING_MARGIN_BYTES`; `osReserveBytes` 2..8 GB, `--os-reserve-gib 0` / app toggle).
- **A long prefill EVICTS the hot cache to be admitted, on the inference thread** (#353, `evictLruToAdmit`); refusals by NAME (`PrefillDoesNotFit` → 400); warm prompts bill via `WarmPrefix`; ONE `[admission]` line.
- **Past 32k a request RESERVES its KV up front** (`KVCache.reservedTokens`, `MLX_SERVE_KV_RESERVE=0`); `max_tokens` clamped FIRST; serve-mode `--max-tokens` is the omitted-field default.
- **Long-context mechanisms of #363 gate on ONE predicate** (`ModelConfig.longCtxGated()`, qwen4_exp).
- **Gates, preflight and COMMIT read ONE estimator** (#126, `estimatePeakResidentBytes` → `gateEstimateBytes`); a refusal quotes the number it COMPARED (`loadRequirementBytes`); staged bills are per-STAGE and a residency bill is a PLAN (`stagedPeakBytes`, `ltxPeakBytes`).
- **A video response is ONE JSON body**: the transport cap bills DELIVERED frames incl. chain windows (#283, `videoRgbTransportReason`, 768 MB).
- **Serial ≠ exclusive**: module-owned decode state needs admission single-flight (`admitPendingTick` on `modelExclusiveDecode`); held slots stay in `pending`.
- **"Free disk" is what the OS will GRANT** (`msv_volume_free_for_use`), statfs is the fallback.

Prefix cache (RAM + SSD):
- **The hot-cache budget is CLAMPED at load, a HARD cap, and FOLLOWS residency** (#364, `clampedPrefixCacheMem`, `reviseHotCacheBudgets`). Guard: `tests/test_prefix_cache_budget_revisit.sh`.
- **An oversized candidate is TRIMMED to the longest restorable prefix** (#330, `trimLenForBudget`, `trimmedCopy` a real copy; QSA bank priced via `trimmedCheckpointBytes`); the replace path sheds inherited checkpoints first; commit owns `ssm_cps` on EVERY outcome.
- **A restored cache shares its donor's buffer; a SHORT restore regrows from the prefix** (`KVCache.restoredOversized`, #492): copying the donor's capacity let a "hi" chat hold 1.9 GB and evict the long session it matched.
- **A share that does not fit is taken over, not refused** (PR #518, `checkoutRestored`): the admission pass checks out a full-entry hit on demand and credits its rows on every arch. A restore always leaves a token to forward (a 1-token full hit segfaulted).
- **Checkpoint retention thins the INTERIOR, dense newest quarter** (`spanPreservingDropIndex`, `ThinPolicy`); a decline is observable (`CommitStatus`, `TrimDecline`).
- **Eviction is WORKLOAD-fair** (#378, `cache_key` via `requestCacheKey`, `lruIndexExcluding`). Guard: `tests/test_prefix_cache_workloads.sh`.
- **State AT/AFTER media is keyed per ITEM on its PIXELS** (`Entry.media: []MediaSpan`; a match stops at the first item differing in position or pixels, `mediaSharedBound`; an entry keeps only the items it covers); inheritance + thinning obey the first item (`bestCheckpointDonor`, `boundaryCheckpointIndex`). Guard: `tests/test_vision_prefix_cache.sh`.
- **SSD tier**: serves the text before the first media item, entries with media never spill; checkpoints come off the TOP of the flush budget; hybrid arm ranks by restorable checkpoint (`bestHybridMatch`); a RAM decline spills (`spillDeclinedToDisk`, 4 GB floor).
- **A disk restore evals each chunk before loading the next** (a lazy `mlx_load_safetensors` holds its fd until eval: 256 files = ~250k tokens); restore entry points drop their own latch, or the cold fallback fails.
- **SSD-first** (qwen4 + disk tier, `ssdFirstActive`): RAM floors at one session; spill and EVICT are two decisions (`PersistOutcome`); writes ride `kv_disk_writer.zig`; a checkout is a PROMISE until the append DONATES (`donateCheckout`/`releaseCheckout`).
- **The batched pad-waste cap reads `KVCache.kvLenForBatching`**, never `cache.step`.

MLX errors + threads:
- **An MLX failure is CATCHABLE** (#353, `installErrorHandler`): `checkError` per chunk, `checkErrorDecode` per tick, a latched error never 200s; a swallowed failure DROPS its latch (`dropLatchedErrorUnless(had_error)`); never hand a null `mlx_array` to the tensor-map insert. Guard: `tests/test_mlx_error_recovery.sh`.
- **Threads**: detach every conn thread; drain (`active_conn_threads` + `cancelAllInFlight`) BEFORE `scheduler.deinit`; handler sampling state outlives every pass (`Slot.in_pass`); an adopted spec cache has ONE owner (#266); sleep inhibition follows the inference-thread wait (#251, `tests/test_sleep_inhibit.sh`).
- **A weight outside every warmup forward is still LAZY at serve time**: force-eval at init; DSpark is opt-in `--dspark`.
- **Ownership by PROVENANCE, never content** (`{slice, owned}`).

LAN + console:
- **LAN**: per-INTERFACE dns_sd callbacks, loopback-first fetches, eviction via `attemptKnown`; proxying bounded by the TUNNEL MARKER (`isTunneledRequest` at gate AND dispatch, ONE hop, `error.SelfFetch`).
- **Console**: `index.html` is a std.fmt FORMAT string; mic only in `listening`; markdown from ESCAPED input; ONE gen per turn, tool `model` enum == `editableIds`. Guard: `tests/html_console_test.mjs`.

### Engine: KV, spec-decode, kernels, MLX (→ docs/gotchas/engine-mlx.md)

Attention + KV:
- **Prefill-chunk cap reads the SCORE width** (`ModelConfig.prefillScoreHeadDim`); hd-256 policy branches key on 256 EXACTLY. Every qwen3.5/3.6/3.8 checkpoint is hd 256 — read the CHECKPOINT before porting an hd-128 kernel.
- **On NAX the stock sdpa is the hd-256 kernel** (`naxSdpaPreferred`, `MLX_SERVE_NAX_SDPA=0|1`); `force_fused` only where mlx has one (`sdpaForceFusedFor`). Band arm stays ours; hd 512 declined.
- **MLX sdpa has a WIDTH WALL** (`q_len*gqa <= 32`): at hd 256 only, causal qL 2..15 ride `splitCausalSdpa` in groups of min(8, 32/gqa) rows, yielding to NAX force-fuse past 8 rows; array masks ride `splitMaskedSdpa256` (`MLX_SERVE_SDPA_SPLIT=0` turns both off). `--mtp-depth` is a CAP — force verify width via PLD draft-len in A/Bs.
- **hd-256 prefill kernel (`msv_attn_p256`)**: band always fused; causal via kv-chunk budget; q_len < 16 ALWAYS declined; its mask arm (`fusedSdpa256Masked`) is the QSA prefill FALLBACK. Guards bill through `prefillHeadDimFused`.
- **Fused decode QK-norm+RoPE** (`fusedQkNormRope`, laguna, `MLX_SERVE_QK_NORM_ROPE_FUSED=0`): bit-identical; live paired A/B is the bar.
- **Decode-only dense-attention requant** (`--decode-attn-quant`, default ON, LOSSY): side copies at decode AND verify; prefill dense; tail layers nvfp4-g16 (`attnDqFor`). A/B per newly-adopted dense arch. dsv4 comp_in requant is EXPLICIT opt-in (`decodeAttnQuantExplicit`).
- **KV invalidation** after tool calls + pad-only gens. Sliding layers keep the full BUFFER; the VIEW is sized by the whole BLOCK (`slidingViewFor`, `window + q_len - 1`, `MLX_SERVE_SLIDING_BLOCK_TRIM=0`). Guard: `tests/test_sliding_window_trim.sh`.
- **Hot-cache restore ALWAYS clamps** (`truncate(final_len)`); a failed restore hands back an EMPTY cache (`errdefer`); every eviction loop needs a no-progress exit (`evictOneLruProgress`). The hybrid DISK arm adopts the spec sidecar of the entry it RESTORED, honours `poisoned`.
- **kv-quant contract**: attention reads `KVCache.denseView` on EVERY path; schemes extend via enum + two switch arms; `HotEntry` records its scheme; quant snapshots carry 6 handles.
- **kv8/kv4 attention at hd 256 rides `matmul2d` over a threadgroup tile** (`qkvAttnMppKernel`, window `qkvMppWins`: t_q>=4 from 2K KV, 2-3 from 8K, 1 from 16K). `MLX_SERVE_KV_ATTN_VERIFY=1` keeps the old kernels.
- **Quantized-KV packed reads are kernel-or-DENSE, per WIDTH** (`kvAttnFusedEligible` t_q==1; `kvAttnVerifyEligible` t_q 2..8 only while gqa x t_q <= 12 rows, `qkvVerifyRowsPay`); verify kernel OFF on G17; floor 2048. Guard: `tests/test_kv_quant_fused_equivalence.sh`.
- **A parity loop asserts FINITENESS before it diffs**; additive masks via `mlx_where`. Threadgroup memory is an OCCUPANCY decision (≤ ~10 KiB, `qkvDecBlockFor`).
- **Every KV buffer is sized from the OPERAND it stores** (K/V widths differ on MLA; an MLA cache is billed per ATTENTION head); a scheme with a shape constraint refuses at LOAD.
- **A fallible re-init BEHIND a `deinit` leaves a freed object** — build first, then swap (`KVCache.reinit`). A handle freed before a fallible op is reset AT the free (`updateDense`). Guard: `KVCache dense update` fault sweep.
- **KV growth is PROPORTIONAL** (`nextCapacity` +25%, floor one chunk, cap 8192). `active` flat while phys climbs = the POOL (`/props` `memory.cache_bytes`).
- **Allocator cache**: `mlx_clear_cache()` per CHUNK and per emitted block, un-skippable (#110): `applyMlxCacheLimit` once in `main()`; cadence interval-based; `Generator.advanceStep` is the one step mover; `finishSlot` clears the tail.
- **Hybrid-SSM prefix cache retains ~3.4× what it reports** (lever = ENTRY COUNT, `ramCappedPrefixCacheEntries`). SSM-checkpoint stride never sub-divides the chunk; dense hd-256 chunk cap 8192, MoE 4096.
- **Hybrid prefix candidates rank by RESTORABLE SSM position** (`findBestRestorableMatch`). The always-on SSM snapshot sits `SSM_SNAPSHOT_BACKOFF` (30) tokens BEFORE prompt end.
- **A restored tail inside the backoff window forwards as ONE span** (`ssmSnapshotBackoff(…, restored)`): same prompt warm == cold BYTES. Guard: `tests/test_hybrid_reuse_equivalence.sh`.
- **INT4 long-greedy divergence is legit**; byte-stable greedy ⇒ no spec + `--kv-quant off/8`. A prefix-cache HIT is not bit-identical on a HYBRID (≤0.047 nats) ⇒ `--prefix-cache-entries 0`.
- **SSM/hybrid**: init checks `ssm_state.ctx == null`; param-free RMS norm passes `ones()`; Nemotron dt clip = only `time_step_limit`; PLD snapshots need per-FIELD null guards.
- **A GDN trunk's `KVCache.step` is 0 forever**: batched rope offsets read the slot's `moe_seq_offset`, which the batched tick ADVANCES. Bar: `test_batched_equivalence.sh`.
- **A model whose per-request state lives OUTSIDE the KVCache is excluded from prefix-cache reuse AND owed a `Transformer.deinit` arm.** `openDirAbsolute` on an empty/relative path is ReleaseFast UB — guard every site.

Spec decode:
- **Verify invariant** (all drafters): `cache.step = prompt_len + emitted`, t1 NOT in cache on entry, verify input `[t1, draft…]`, partial-accept correction from ORIGINAL `verify_logits[accepted]`.
- **A block decoder checks its ENTRY token first** (`generate.tokenStops`, all five); only an ALL-pad generation declines commit (`commitDeclinesPadOnly`); a cancel mid batched tick still RECORDS the row (`batchedTickAction`).
- **The token budget is a PRE-COMMIT invariant in every block decoder**; blocks publish through ONE `+= 1` loop.
- **A committed argmax is a `CommittedArgmax`** from `verifyArgmax` only (masks reserved ids); acceptance-test argmaxes are exempt.
- **No KV snapshots across verify**: rollback = scalar anchors + per-position SSM capture + offset-only `truncate`. A multi-token forward is not a prefill (`prefillEvalCadenceApplies`, seq ≥ 32).
- **Dispatch discipline**: all four surfaces × stream/non-stream wire `use_*` via ONE `server.requestSpecModes`; priority DFlash > MTP > gemma drafter > PLD; MTP default = `defaultEnableMtp` (MoE OFF); logprobs>0 + grammar disable spec; tools disable NOTHING; engagement COUNTS in tests.
- **Gates read DISPATCH, not ARMED flags** (`slotTicksRegular` → `specTickMode` + `spec_disabled_runtime`); a guard shaping INIT options does not bind dispatch; `nextPld` self-declines on `!pld_enabled`.
- **Batched MTP verify is ONE `[N, S]` trunk forward sized by the verify lane** (`mtpSubGroupPlan`; 7 rows split-K, 16-row M4 shader tile for 4..8 lanes, `mtp_group_fill`); past the crowd threshold slots decode PLAIN (`mtp_plain_tick`). `MLX_SERVE_MTP_BATCHED=0`.
- **Group lanes draft as rows of ONE head forward** (`mtp.forwardLanes`), greedy off the exact trunk head, sampled off its top-32; next chain pre-dispatched (`mtpGroupPreDraft`). M-RoPE/confidence-gated lanes keep `mtpChainBuild`.
- **The group planner prices each width per row** (`mtp_group_planner.zig`, `MLX_SERVE_MTP_GROUP_PLANNER=0`, per-request `enable_batch_mtp:false`) and falls to plain batched decode at width 0; `[mtp-planner]` names the choice.
- **A group's sampled accept is ONE filtered block on the group's own eval** (`mtpGroupSampledAccept`; one shared sampler, no seeded row). A padded verify row reads `1+m` rows (`verifyRows2d`, #446).
- **qwen4 MTP head state is a `Qwen4MtpState` swapped onto the module** (`qwen4MtpActivate` before EVERY head touch); batched verify there is opt-in (`MLX_SERVE_MTP_BATCHED_QWEN4`, `mtpRoundsStaySolo`).
- **A DFlash sidecar yields to the MTP head when the request has company** (`requestSpecModes(..., has_company)`); a burst's first request goes plain after 2 ticks with company (`dflashYieldTick`, one-way).
- **DFlash is a METHOD keyed on the CONFIG CONTRACT** (`block_size` + `mask_token_id` + `target_layer_ids`; `--drafter` the one flag): drafts from ONE assistant forward, anchor row DROPPED, context via `capture_layers`, block K/V never cached.
- **DFlash2** = v1 + selector + convs on NESTED `dflash_config`; selector codebooks bf16 gather tables NEVER quantized; discovery classifies by contract. `MLX_SERVE_DFLASH_SELECTOR=0`.
- **DSpark** = DFlash + Markov head (`dflash.Contract`, `MarkovHead`): `block_size` at ROOT; `rope_is_neox_style:false` = MLX `traditional=true`; `w1` DENSE; `MLX_SERVE_DFLASH_MARKOV=0`. Serves SAMPLED requests via one-hot acceptance (`MLX_SERVE_DSV4_DSPARK_STOCH=0`).
- **Hybrid forward is capture-capable; partial accept rolls the CONV state back** (`ssmRollbackFromCapture`); `hybrid_path` separate from `moe_path`. The hybrid veto belongs to the GEMMA drafter only (`archBlocksAssistantSidecar`).
- **DFlash yield gate** = round cost / serial step, calibrated on DENSE trunks; MoE takes an absolute floor (1.8 accepted/round). The ngram spec-gate scores the PROMPT; DFlash is EXEMPT. Acceptance is a THINKING-MODE property.
- **A shipped spec sidecar is a load-time dependency** (`resolveInDirDrafter`, `<model_dir>/drafter`); `--drafter` wins, `--no-drafter` opts out; discovery refuses `*_assistant` standalone.
- **The spec BLOCK is a MACHINE property** (`resolveBlockSize`, `blockCapForMachine`: M3 Ultra 8, else 5; wide lane NAX-only); a wider block loses off-NAX at N=1. `WidthChooser` opt-in (`MLX_SERVE_DFLASH_CHOOSER=1`).
- **Trunk-derived draft state rides the prefix cache** (`DflashSnap`, `Entry.mtp`, `restoreSpecSnap`): adopt only on `base + step == matched`; survives the SSD tier (`spec.safetensors`).
- **A sidecar's weights are a per-round READ**: dense bf16 assistants quantized at load (`MLX_SERVE_DFLASH_QUANT_BITS` 8); MTP head trunk requantized (`MLX_SERVE_MTP_HEAD_QUANT_BITS` 4/g64, fc bf16 for m5Nax); head `fc` may ship quantized (`QLinear`).
- **MTP head load**: delta norms AUTO-FOLD (`mtpNormNeedsRepair` also reads the norm's own negative fraction); quant mode solved PER WEIGHT from GEOMETRY (`quantParamsFromGeometry`, `.biases` optional); verification is ALWAYS the trunk head; every lever kill-switched.
- **`--no-mtp` gates an IN-CHECKPOINT head too** (`entry.mtp` reads `params.mtp_enabled`). `MTP_FORCE_ENABLE=1` on MoE.
- **Per-arch exclusions read ONE predicate** (`ownsModuleDecodeState()` + `specInitWiring`); "module-owned" is a property of the STATE (`module_owned_state_fields` vs `module_shared_readonly_fields`); only the slot driving a module-owned head is exclusive (`slotExclusiveDecode`).
- **A module-owned forward must honour `ctx.capture_hidden`** (publish last-row + all-rows via `skip_lm_head`).
- **Round cost is MEASURED** (`round_cost.zig`, `MLX_SERVE_MTP_COST_TABLE=0`): per model/width/KV bucket from solo SINGLE-CHUNK rounds; `tok` is a workload MIXTURE (plan on `planTok`); per-silicon rows are COLD-START caps (`mtp_depth_free`), never deleted.
- **Table hygiene**: interleaved prefill drops the round clock; refuse past `IMPLAUSIBLE_STEP` 1.5x / `SELF_SPIKE` 3x at fold AND load; load sweep also `IMPLAUSIBLE_WIDER` 1.25x (#382). Persistence OPT-IN (`MLX_SERVE_ROUND_COST_PERSIST=1`), keyed on chip/model/quant/OS/engine build.
- **Width trials** (`mtpWidthTrialTarget`): m_lo then m_lo+1, NEVER m_lo−1; `[spec-stats]` carries `width_trials= table=`.
- **Auto MTP depth**: below 8192 KV plan tokens from acceptance EMAs + table round TIME in ONE chunk (`MtpDepthPolicy.accept`, `MLX_SERVE_MTP_DEPTH_POLICY=legacy`); past 8k legacy. Rank policies in `src/mtp_replay_test.zig`, never live tok/s.
- **The adaptive serial switch serves sidecar heads too** (`mtpSerialProbeEarned`: 16 rounds, EMA < 2.0, floor 32k KV).
- **Auto-mode MTP output is NOT byte-reproducible**; byte bar = `MLX_SERVE_MTP_FORCE_DEPTH`. Acceptance is a PROMPT-TYPE property — measure per index (`acc_idx=`) before touching round cost.
- **EV cost tables are refit whenever the verify forward changes** (`MTP_EV_DEFAULT_COSTS`); `MtpCostProfile` comes from the runtime fingerprint, never the sidecar; qwen4 has its own G17 surface (`MLX_SERVE_MTP_QWEN4_PROFILE=0`). A/B profiles with persistence OFF on both arms.
- **Drafts shortlist on a coarse lm_head and re-score exactly** (`buildRerankCoarse`/`rerankShortlist`, from the MIXER output; `MLX_SERVE_MTP_DRAFT_RERANK=0`). Proposal is per REQUEST (`mtpDraftStepPath`): greedy = argmax, sampled = q over the exact top-32; sampled group rows stay batched (`shortlistProposalRows`).
- **qwen4 MTP specifics**: head projects only the consumed row (`Qwen4MtpProject`); EV seed lives on `Qwen4Mtp`, declines under FORCE_DEPTH; head rides the slot's M-RoPE table on image turns; a verify row is BYTES (MTP stays opt-in); grouped-expert NAX tile is a measured LOSS. Solo greedy rounds pad the head history and build the next chain lazily (`MLX_SERVE_MTP_PADDED_HEAD=0` / `MLX_SERVE_MTP_LAZY_PREDRAFT=0`); the lazy plan lags one round at auto depth BY DESIGN (see engine-mlx gotchas).

Sampling:
- **`top_p` 0 is GREEDY** (`applyTopP` floors at `floatMin`). Filters cut by RANK with lowest-id tie break (`ranksDescending`, `topRanksDescending`); cumsum in f32; top-k + top-p are ONE pass (`filterTopKTopP`). Block helpers use `_axis` ops.

Kernels + numerics:
- **Slice-born weights into gather_qmm/quantized_matmul are `mlx_contiguous`-materialized at load**. mlx `Copy`/`contiguous` are VIEW ops: a slice outliving its parent goes through `materializedOwnedCopy` + eval; a raw data-pointer read PROVES row-major contiguity. A view-materializing helper owns the view (`sliceContig`).
- **`mx.quantize` packs DENSELY** (element i at bit `i*bits`, straddling words at 3/5/6 bits, #305): every hand-rolled unpack is tested at EVERY shipped width. Fused MoE kernels take 3-bit as a BYTE TRIPLE (`mlxserve_qpack`).
- **An f32 SCALAR promotes every bf16 operand** — scalars via `scalarOf(v, dtype)`; a load-time const table in the wrong dtype widens every read (`constTableAs`); a chain that returns f32 by design makes the CALLER own the dtype; a quantized KV cache returns the dtype it was FED (bf16 scales widened f16 Bonsai under `--kv-quant`). Tell: `[dtype-trace] residual widened`.
- **Every forward path carries a `[dtype-trace]`**; 1-D f16 tables narrowed at load (`narrowsLoadedF16`); a kernel's dtype and threadgroup BLOCK SIZE are ONE decision (`gdnBlockTFor`); signatures come from each input's ACTUAL dtype, <8-element arrays land in `constant`.
- **A decode kernel can be LATENCY-bound** (qwen4 `hcReadFused`, `MLX_SERVE_HC_FUSED=0`): split-K + unrolled loads beat the op chain; redistributing a reduction into every threadgroup LOSES. Meter: `MLX_SERVE_DECODE_FWD_UBENCH`.
- **A decode kernel keyed on `batch*seq == 1` declines verify rows AND batched slots** — the grid carries the rows (`HC_FUSED_MAX_ROWS`/`GDN_FUSED_MAX_ROWS` 16).
- **Prefill fusions take the chunk WIDTH as a scalar INPUT, never a template** (`hc_prefill.zig`, `MLX_SERVE_HC_PREFILL=0` / `MLX_SERVE_GDN_PREFILL_FUSED=0`); a per-token-varying template value is a fresh JIT per value. Every `metal_kernel` helper owes a `streamIsGpu` guard.
- **GDN decode is three fused dispatches per layer** (`gdnPreworkFused`, `gdnNormGateFused`, `MLX_SERVE_GDN_DECODE_FUSED=0`); a greedy byte flip there is legit.
- **A host read inside a graph build or layer loop is a GPU BARRIER**: qwen4 PLE defers (`ple_defer` + `flushDeferredPle`, claims its spec slot at BUILD time via `pleClaimSpecCapture`); batch non-consumed reads into ONE eval. A rollback that re-forwards is a SECOND forward (`DsparkAnchors`).
- **A cold 32 GB mmap read is a serial SSD fault** (`NgramTable.gather` → `PrefetchPool`, `QWEN4_PLE_PREFETCH=0`; `startWarm`, `MLX_SERVE_NGRAM_WARM=0`).
- **A chain the GPU already OVERLAPS is not a dispatch to fuse**; a fusion pays only if it shortens the DEPENDENCY CHAIN; a weight-layout fusion is not output-preserving. Measure the marginal in-situ; "duplicate and read the marginal" is unsound for anything that MUTATES state.
- **Reproducing an MLX op means its REDUCTION TREE and ACCUMULATOR**; `mlx_compile` is not output-preserving; JIT vs metallib transcendentals disagree — sweep the 16-bit domain (`swigluSigTable`).
- **MoE decode**: gate+up fused `gatherQmv` default by the kernel's OWN conditions (`useGatherQmvDecode`); down+reduce ONE dispatch over 8 lanes (`gatherQmvDownReduce`, `DOWNRED_LPR`, `MLX_SERVE_MOE_DOWN_REDUCE_FUSED=0`; bar = fp32-truth RMS). lm_head prune OPT-IN. MoE PREFILL uses `_gather_sort`.
- **verifyQmm lanes** (`vqmmLaneForTile`): split-K M 2–7 / wide tile / NAX m16 M 8–16 / shader matmul2d on G16 (4-bit, M 8–24) / crossrow opt-in; plain-SIMD tiles BITS-templated and SHAPE-gated (`mixedPlainShapeEnabled`). M=8 is the plain-SIMD cliff.
- **A verify lane is never byte-identical to stock**; parity = fp32-dequant truth per width, RMS ratio vs stock ≤ 3.0x, never cosine (`VerifyQmmParity`). 2-bit GEMV accumulates in f32 from exact products (`qmv2.zig`, M 1..8; unmeasured GPU generations keep the old M 1..3 dispatch).
- **A `metal_kernel` config cache is keyed by FULL SHAPE** (`ShapeKey`); a borrowed-handle cache evicts LRU (`vqmmScalarEvictIndex`). A bandwidth bench smaller than a real step measures CACHE.
- **A diagnostic env goes through `diagEnvOn`** (absent or `0` = off). A default belongs to the engine that MEASURED it.
- **Kernel testing**: parity = no-worse-than fp32 truth, never kernel-vs-kernel; same-boot A/Bs at per-cell MEDIANS, interleaved in one process; every adopted shape gets its own A/B; content-forking arms need 3+ reps with a NONCE per rep.
- **QSA** (qwen4): GATHERS selected blocks, never a dense `[S, kv]` mask (`gatherQsa256`, `qsaSparseAttn`, `qsaDecodeGatherAttn`); exact select `msv_qsa_select` (split 16-way at decode, `MLX_SERVE_QSA_SELECT_SPLIT=0`); score sheet one NAX kernel (`msv_qsa_score`, `qsaScoreFusedActiveFor`); verify gather floor per KV scheme (`qsaVerifyGatherMinKvFor`); NAX gather via `qsaNaxEligible` (`MLX_SERVE_QSA_NAX=0`, bar = error vs float64).
- **QSA indexer**: ONE YaRN mscale on every arm (`yarnScaleRotated`, never double-scale M-RoPE); ropes with attention's M-RoPE table (`beginMropeChunk`). A pointer-keyed cache is invalidated by an ATOMIC MARK (`markQsaPooledRopeStale`), freed on the inference thread.
- **qwen4 residual stream is bf16; the f32 fixture is the MATH oracle** (`QWEN4_STREAM_F32=1`; `qwen4 fixture bf16` pins streams at 0.999).
- **Gates**: a lower bound may REPLACE the formula (KDA `kdaUsesBoundedGate`); per-head vs per-channel gating is an INDEXING contract (`gdnKernelSource(vectorized, …)`). A struct in a generic fn must capture its comptime params.
- **Prefill perf kernels A/B PER ARCH before default-on** (`gdnBlockedEligible`, `prefillDqGemm`). dsv4 decode glue is FUSED (`dsv4_emit_win`, `dsv4_dec_chain`). A sparse-attention PATTERN cannot be tuned into correctness (H3, DEAD).
- **Memory**: RSS is blind to Metal (`/props` `active_bytes`); a weights MAP outliving the model pins every buffer; `iterator_next` hands a +1; `mlx_array_new_data` COPIES shape-worth of bytes; Metal at the working-set edge returns ZEROS before it aborts.
- **Tokenizer**: special-token splitting only in `Tokenizer.encode` (first-byte buckets); any per-position loop over a vocab-derived collection needs an index.
- **Upgrades**: mlx-c → diff `lib/mlxc-src/mlx/c/*.h` vs `src/mlx.zig`; ds4 → re-sync `ds4_ffi.zig` EngineOptions + CORE_OBJS; NAX presence asserted, never assumed.
- **Kokoro**: safetensors READ on CPU stream; depthwise `ConvTranspose1d` transposes `{0,2,1}`; `AdaIN1d` is pure instance norm; `Vocab.encode` DROPS unknowns; reproducing an upstream NO-OP means NOT doing it (SineGen).
- **ANE prefill (opt-in, M4-and-below, LOSSY)**: rules in `docs/reference.md` "ANE prefill rules"; parity = per-program cos/rms + perceived-content equivalence; `ready: N/M` under-count = silent partial coverage. Guard: `tests/test_ane_prefill.sh`.

### Model loading, configs, converters, media parity (→ docs/gotchas/models-media.md)

Configs, templates, tokenizers:
- **A marker family that is GLM/`<think>` under another spelling is ALIASED at decode, never re-parsed** (K2 `<ifm|…>` → `Tokenizer.marker_aliases`); only the rendered prompt keeps the pack's spelling (`k2ThinkOpenerAt`). JSON-schema + thinking on K2 keeps thinking off (known).
- **`generation_config.json` `eos_token_id` joins the stop set** (`mergeEosTokens`, additive); Gemma terminators merge additively (`ensureGemmaTerminators`). jinja.cpp: `is sameas true` / `is divisibleby 3` parse a BARE test argument.
- **Config reads**: when the reference IGNORES a field, the field is not the truth (laguna YaRN mscale); a field HF allows in two SHAPES is read as both (`chat_template`); `text_config` FIRST, then root, PER FIELD; a default only ONE family wants is pinned PER LAYER TYPE (muse `rope_local_base_freq`, `tests/test_muse_repetition.sh`).
- **`*_text` siblings**: accept the tag, collapse to base type, prefix by `text_config` presence, force `tie_word_embeddings` for Gemma, add to BOTH visibility allowlists.
- **A sampler never draws a RESERVED special or a PADDING row** (`reservedOutputIds` + `definedVocabSize` → `installSuppressMask`, `MLX_SERVE_SUPPRESS_RESERVED=0`; `unpadded_vocab_size` = ONE trim); logprobs stay RAW.
- **Metaspace `prepend_scheme` is THREE-valued** (`MetaspacePrepend`): `first` prepends ▁ only at offset 0, never after a special token (Mistral `[INST]Use`); `always` prepends per segment (laya). Diff `/tokenize` vs HF on a prompt WITH specials.
- **Digit GROUPING is per-model** (`Tokenizer.digit_group`; a COMBINED Split regex hides it, `pretok_style = .llama3`); cross-check `/tokenize` vs HF at bring-up. The degenerate-tail guard has a LONG-period tier (`isDegenerateTailLoopRange`).

Weights, quant, loading:
- **Tree, prefix and axis order are CONVERTER choices — probe** (`resolveWeightPrefix`, `lagunaRouterBase`, `hy3ExpertContainer`, `resolveVisionPrefix` + `patchProjLayout`). A family's geometry comes from the CHECKPOINT once a second size exists (FLUX klein 4B/9B).
- **Quant resolves PER WEIGHT** (`computeQuantParams`; scales dtype decides fp8 vs affine; overrides can hide inside the fp family, `fpParamsFromGeometry`); affine bits outside {2,3,4,5,6,8} rejected at PARSE; no engine hardcodes a width (`affineParamsFromGeometry`, scan-pinned).
- **Dense checkpoints**: scales absence PER-TENSOR (`getLayerScaleOpt`); dense contracted weights owe `maybeTransposeForBf16` — never depthwise conv or SSM state.
- **A pack that declares its activation dtype is served in it** (`actDtype`, `LoadOpts.keep_f16`; Bonsai 2 f16 + f32 GDN state); every constant takes the activation dtype. Guard: `tests/test_hadamard_fidelity.sh`.
- **Every new LM or media port supports a bf16 pack AND the quantized ones**: loader, converter presets and app catalog carry both; quantized is the default download, bf16 the quality reference.
- **A gather-read table is quantized only where the READER has a quantized-gather path** (media `NEVER_QUANTIZE`; LM `embed_tokens` via `gatherQuantizedRows`).
- **Calibrated quant**: weights per-input-channel and per-expert; bit width beats group granularity ≤3 bits; round (s,b) to the STORED dtype first; an imatrix is valid only for the WEIGHTS it was collected on; uniform ≤2-bit experts to the LAST layer cause agent loops (4-bit tail fixes it).
- **Publish MTP head norms FOLDED** (`--fold-mtp-norms`).
- **Discovery**: a configless repo SHAPE is taught to ONE predicate (`peekMageFlowIndex`; `gen.peekModelType` delegates); size sums stat THROUGH symlinks; an incomplete media pack is invisible until its marker lands (`requiredMediaMarker`, `error.IncompleteMediaPack`); a GGUF folder is a SHELF (`id#<file>`). Guard: `tests/test_model_rescan.sh`.
- **The `--no-vision` drop list is a GLOBAL prefix filter**: a tower loads through `loadWeightsWithVision`, never `loadWeights` (MageFlow Edit lost `model.visual.*`).

qwen4_exp:
- **NOT a qwen3_5 pack**: hyper-connections, n-gram PLE, QSA around the trunk; HF `hidden_states[i]` is the INPUT of layer i. Vision rows splice BEFORE the hc tile (`forwardQwen4With`).
- **Tiny MoE oracles tie everywhere**: fixtures dump the reference's OWN margins, `Qwen4Ties` acquits by those; k < E selection coverage = the MTP head's one MoE layer (`--topk 2`, `route_gap`).
- **QSA's visible tail is PER QUERY**; scores in f32; `torch.topk` keeps the LOWER index on ties; n-gram hash eos is the TEXT config's (`ngram_eos`).
- **The spec "hidden" IS the pre-mixer stream** (`[B,L,hc*hidden]`); head row r = (stream r, token r+1) at position r+1 (`pos_base`).
- **Per-request state outside conv/ssm rides `SSMCacheEntry.aux_state` + `ple_prev`**; every reset/free via `ssmFreeQsaState`; restore-parity bar = the CHUNKING class (~0.3 nats top-5).

Vision:
- **`x-mlx-pixels` is a GEMMA format** (`wantsServerPreprocess`; server refuses when `vp.mode != .gemma`); `VisionEncoder.forward` returns named errors.
- **A placeholder id the splice mask does not name never reaches the prompt** (`spliceVisionRows` takes `video_token_id`); a chunked splice resumes its ROW INDEX (#197, `vision_splice_offset`, `tests/test_vision_chunked_prefill.sh`); every generative forward arm splices vision (scan-pinned).
- **Per-arch preprocessing**: patch feature order (LFM2-VL channel INNERMOST); two GELUs in one checkpoint (LFM2-VL); NaFlex resample is `bilinear` WITH antialias and TILING is the resolution; muse = aspect-closest grid, Lanczos, CHANNEL-major shuffle (`VisionPreproc.mode`). Guards: `tests/test_lfm2_vision.sh`, `tests/test_muse_vision.sh`.
- **A processor pixel bound is not the ENGINE's** (`effectivePixelBounds`, `ENGINE_MAX_PIXELS` 1536²).

Media backends:
- **ACE-Step**: timbre slot = silence latent OR the reference clip's VAE mean (`ref_audio`, #259); a TASK is the context stream + instruction line (`complete`, `cover` with soft clamp THEN hard grid; `fsq.safetensors` separate dense bf16). Guards assert log lines, never output.
- **`instrumental` reaches the checkpoint only as TEXT** (`[Instrumental]`); flag + lyrics = NAMED 400 (`instrumentalConflicts`).
- **Music3 is NOT an ACE-Step variant** (timestep TOKEN, reversed SwiGLU, alpha-only Snake, hardcoded DiT RoPE). Probe laps (`MUSIC3_COST_PROBE`) before bandwidth arithmetic.
- **LTX**: a standalone-frame latent belongs in slot 0 ONLY (#260; `keyframeMask` + `keyframePositions`); the RELEASE is a config field deciding the text encoder (`LtxVersion`; 2.5 runs the real gemma4 via `gemmaCapture4`, `prefill_mask_add`); `keyframes_abs_pos_embedding` parsed, never added.
- **LTX DiffVAE**: constructor args are not its config (`Sampler`: 1-step x0 at t×1000); tile budget is a per-REQUEST memory decision (`tileTokensForMemory`). 4-bit affine on a video DiT is a QUALITY setting (8-bit mirror).
- **A full-resolution f32 VAE stage is BANDED, exact because everything but its 3x3 convs is per-pixel** (`qwen_image.Stage.banded` + `Conv.forwardStrips`): whole, a 1024² decode peaked 18 GB on a 5 GB engine while MLX's peak counter read 4.9. Bar = the fixture oracle with bands forced.
- **A pinned library's op TRANSIENT is invisible to residency bills** (#321, #424): `conv3dDepthChunked` windows decoder convs; H3 `encodeMoments` EVALS per tile pass. Diff conv DISPATCH per mlx bump.
- **Canvas**: `recommendedResolution(totalGB:)` per Mac, capped by `autoCanvasCapPixels`; two-stage denoises at HALF; `maxFramePayloadBytes` is a TRANSPORT limit. A tiled decoder whose positions normalize over the EXTENT refuses (`error.TilingUnsupported`).
- **H3**: condition stream assembled in SEGMENT order by ONE resolver (`resolveRefs`); identical-file partitions need a DECLARED discriminator (`tasks`).
- **LoRA alpha lives wherever the exporter put it** (`lora.fileAlphaScale`; per-module tensor WINS); test with `tests/lora_noise.py` + `tests/test_real_loras.sh`.
- **A reference-image editor keeps its input's geometry** (`resolveEditTargetSize`); a few-step distilled model can REQUIRE bf16 (MageFlow Turbo: `roundBf16`, `scalarLike`).
- **Latent→RGB previews are GENERATED fits per latent SPACE** (#208, `tests/dump_latent_rgb_factors.py`; `LTXAV` ≠ `LTXV`).
- **TTS**: `TimeDelayNetBlock` = conv + IMPLICIT ReLU; embedding lookups reshape to the TABLE's width; voice clone = ECAPA embedding as one codec-prefix position.

Parity method:
- **Cosine cannot see SCALE** — concatenated tensors assert `rms_ratio`; a permutation-invariant checksum cannot see a permutation (`pos_weighted`); an oracle that cannot execute the reference must say so; a faithful port can make the checkpoint WORSE (H3 splice default-off).
- **Deep ViT/DiT fixtures dump fp32 on CPU**; transformers ≥5 zeroes custom rotary buffers (assert non-zero); an oracle that IMPROVES when a transform is removed had it disabled. DiffusionGemma parity = converged-canvas self-consistency (`MLX_SERVE_DIFFUSION_TRACE=1`).

### Licensing & third-party code

Ported kernels + vendored code are enumerated in `NOTICE` (the ONE place); `LICENSE`/`LICENSE-APACHE-2.0`/`NOTICE` ride every packaging path, pinned by `tests/test_release_workflow_gates.sh`. To enumerate ports, grep comments for `mlxfast|oMLX|MTPLX|mlx-lm|port`. Full story: `docs/reference.md` "Licensing".

### App (Swift)

All app-side rules: `app/CLAUDE.md`; stories: `docs/gotchas/app.md`. Crossovers: agent budgets derive from advertised `context_length`; the app ALWAYS emits memory-critical launch flags (`ServerOptions` defaults mirror Zig defaults — change both together); the two bundle binaries move together.
