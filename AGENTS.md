# mlx-serve — project context for AI

Native Zig server running MLX-format LMs on Apple Silicon; OpenAI/Anthropic/Ollama-compatible HTTP APIs + native media generation. No Python. This file is the COMPRESSED layer; detail lives one hop away.

## Detail map (read on demand)

- `docs/reference.md` — pre-rewrite deep detail (a folder's `AGENTS.md` wins where they disagree): per-file contracts, media-gen schemas, API surfaces, LAN design, observability, embedded engines, arch numerics (dsv4/inkling/H3, Kokoro, runtime LoRA), website/tier list, licensing. Read its section BEFORE deep work on a subsystem.
- `docs/gotchas/{tool-calling,server-http,engine-mlx,models-media,app}.md` — full war stories behind every rule in the folders' `AGENTS.md`.
- `src/plugins/<name>/AGENTS.md` — each model's rules, quirks, packs and oracles. `docs/known-issues.md` — what is broken now.
- `src/{host,host/chat,engines/mlx,plugins}/AGENTS.md` — each area's rules, one line each. `docs/recipes/` — add a model, extend one, a media backend, a kernel, an engine, an API field. `docs/plugins.md` — the plugin guide.
- `docs/CUDA.md` — NVIDIA (Linux, Windows): builds, our own CUDA kernels, per-device defaults, memory, measuring, MLX patches.
- `tests/CLAUDE.md` — integration-test matrix. `app/CLAUDE.md` — Swift app layout + rules; `app-web/CLAUDE.md` — console source (auto-load in their dirs).
- Skills: `/release` (pre-release checklist, CalVer, CHANGELOG), `/bench` (llmprobe methodology + comparison traps).
- `containers/{agent-shell-mlxserve,guest-kernel}/` — Agent Sandbox guest image + kernel, two pinned artifacts that BUMP TOGETHER. Detail: `docs/reference.md`.
- `website/` — GitHub Pages site + `llm-tier-list/`. Design: `docs/reference.md`. Guards: `tests/test_website_pages.sh`, `tests/website_tier_list_logic.mjs`.
- **Growth policy (ENFORCED)**: this file stays under 30k bytes (the map, the global rules, the inner loop), a folder's rules live in its own `AGENTS.md`, short-form and minimal; EVERY rule bullet is ≤ 3 lines. Be PICKY: a rule earns a line only for a real bug or a strong gotcha we actually hit and that someone would repeat. Not a diary: no log of what happened, no measurements, no round-by-round notes, no restating what the code or git history says. Same bar for `docs/gotchas/*.md` (one short story per gotcha) and `docs/reference.md` (one section per subsystem). When in doubt, leave it out; App content → `app/CLAUDE.md`.

## Stack

Zig 0.17.0 (pinned via `scripts/fetch-zig.sh`); mlx + mlx-c PINNED SUBMODULES (`lib/mlx-src` v0.32.3, `lib/mlxc-src` 56b2d39 = PR #127 + `patches/mlxc-gather-qmm-global-scale.patch`, applied by the build scripts) self-built NAX-enabled by `scripts/build-mlx.sh` into `lib/mlx/` (FFI `src/mlx/ffi.zig`); jinja.cpp (wangzhaode, Apache-2.0, NOT llama.cpp's) as `lib/jinja_cpp/libjinja.a`; stb_image + libwebp; safetensors; BPE. Embedded engines: ds4 (`lib/ds4`, DSV4-Flash GGUF), libllama (`lib/llama`, generic GGUF), the Neural Engine runtime (`lib/ane`). Absorbed in-tree: sushi's EXL3 + STQ (`src/sdk/quant/exl3/sushi/`, module `sushi_exl3`, `zig build exl3-test`) and davidtai's mlx-stream (`src/mlx_stream.zig`).

## Layout (`src/`)

Two seams: the ENGINE interface (`abi/engine.zig`, host ↔ an engine, per request) and the PLUGIN tables
(`abi/plugin.zig` model, `abi/media.zig` media, `abi/quant.zig` quant; per load and per forward, never per layer).

| Area | Role |
|---|---|
| `main.zig`, `env.zig` | Entry, CLI flags + subcommands; every environment variable registered in `env.zig` (kind, owner, since) |
| `exe.zig`, `host_unit.zig`, `engine_unit.zig`, `modules_unit.zig` | `-Dunits` roots: the host, the MLX engine, the module archs it calls by symbol (`engines/mlx/modules/call.zig`: DSpark, the cluster worker, EXL3 experts, qwen4's MTP head; the stream glue) and each plugin group (`unit_<g>.zig`, generated) as their own objects, each under 3 GB of compiler memory (`scripts/unit_memory.py`); `shared_all.zig` lists every `Shared` the host owns |
| `host/` | Everything model-agnostic: `http/` (router, conn, SSE, WS, multipart, auth), `api/` (OpenAI, Anthropic, Responses, Ollama, embeddings, `/v1/models`, providers, transcription), `chat/` (templates, `tools/` one file per dialect, `reasoning/`, stops, format corpus + traffic replay), `registry/` (discovery, modality map, model settings, load progress), `gpu_thread/` (what connection threads hand the GPU thread: the `Scheduler`'s queues, load/submit/media/vision/embed requests), `engines/` (`mlx.zig` = the ONE way into the MLX engine, a table of calls; `stream.zig` = a request's output every engine fills through `Sink.event`; `served.zig` = the other engines), `media/` (request media), `media_jobs/` (media-gen jobs, previews), `lan/`, `launch/`, `cli/`, `metrics/`, `plugins/` (outside dylibs, `mlx-serve plugin new\|check\|install`), `sandbox/` |
| `abi/` | Extern tables only, no globals: engine, model plugin, media, quant, `Host`, dylib manifest; `contract_test.zig` + `exactness_test.zig` run every engine |
| `engines/mlx/` | Our engine, reached by the host only through `api.zig` (the table `host/engines/mlx.zig` declares): `runtime/` (the inference thread: sole MLX caller, loads, media jobs, offloaded engines), `scheduler.zig` + `step.zig` (slots, admission in `admission/`, batching in `batching/`/`batched/`), `loop_stop.zig`, `generate/`, `sampling/`, `grammar/`, `spec/` (PLD, drafter, DFlash/DSpark, MTP rounds, round cost), `kv/`, `cache/` (prefix cache, SSD tier `cache/ssd/`), `ssm/`, `cluster/` (Thunderbolt RDMA), `plugin/` (the DRIVER that serves a plugin table, its proofs and tapes), `load/` + `forward/` + `transformer.zig` (the legacy `Transformer` the unflipped archs still run) |
| `engines/{ds4,llama,ane}/` | One adapter each over its runtime (`lib/ds4`, `lib/llama` + `lib/llama_shim`, `lib/ane`); `engines/own_thread/` is what the three share |
| `sdk/` | Everything a plugin imports: `bases/` (decoder, hybrid, encoder, DiT skeletons + `bases/blocks/` the blocks they compose), `blocks/` (attention, mlp, moe, gdn, kda, ssm, conv, norms, rope, hc, linear, qmm, dit, schedulers), `quant/` (affine/fp8/mxfp8/nvfp4, `hadamard/`, `exl3/`, `gguf/`, `fp8_block128`), `weights/` (resident, streamed SSD banks, LoRA), `spec/` (MTP head), `vision/` (SigLIP, Qwen3-VL, DINOv3, M-RoPE), `media/` (job progress, previews, JPEG) + `mesh/` (pure-Zig mesh/GLB/xatlas), `tokenizer/`, `testing/` (fixtures, conformance, fake host, `kernel_contract/`), `util/` (`shared.zig`, `log.zig`, `rtfmt.zig`) |
| `plugins/<name>/` | One model family or media backend each: `plugin.zig` (the table), `config.zig` (MLX-free claim + parse + bill), its own blocks, `kernels/`, `fixtures/`, proofs, and its `AGENTS.md` (its rules and quirks: read it before touching the model). `_template*` = the barebones plugins, `_example*` = the kit's |
| `mlx/` | `ffi.zig` (mlx-c), `ops.zig` (the op layer: a `Graph` owns what it makes), `scope.zig`, `kernels.zig` (the kernel loader), `state.zig` |
| `kernels/<area>/` | Metal/CUDA kernels two or more plugins use (one model's own: `plugins/<name>/kernels/`); every file listed with its sha256 in `kernels/digests.txt` |
| `mlx_stream.zig` | The absorbed mlx-stream module: GLM-5.3 (`plugins/glm_moe_dsa/`, its only implementation, still on the engine's `modules/stream_glue.zig`), MiMo's expert streamer (`sdk/weights/streamed/mimo/`), GLM-5.3-Flash's EXL3 lane (`sdk/weights/streamed/glm5n/`), EXL3 kernels in `sdk/quant/exl3/stream/`; `zig build mlx-stream-test` |
| `kit.zig`, `kit_header.zig`, `plugin_host.zig` | The plugin kit's module, its generated C header, and the host surface a plugin's own test build imports |
| `testing/fake_engine.zig`, `tests.zig`, `preview_tests.zig`, `ios_lib.zig` | Test roots, the no-model engine, the iOS entry |
| `html/index.html` | The console, BUILT from `app-web/` (never hand-edited) |

Build glue: `build.zig` lists the steps; `build/` holds each graph (`macos`, `linux`, `ios`, `standalone`, `lint`), what
they link (`libs`), the configure-time reads and checks (`config`), `plugins.zig` (scans `src/plugins/*/plugin.zig` into
the generated registry, refuses a `config.zig` that reaches MLX), `test_runner.zig` (shards the tests) and `kit/`.

CLI flags: `mlx-serve --help` (`printUsage` in `src/main.zig` is the source of truth; `docs/cli.md` for users).

Generation defaults for omitted fields: body > model `generation_defaults` > global `~/.mlx-serve/generation-settings.json` > launch flags > model `generation_config.json` > hardcoded (1.0/1.0/off); a rule with `ignore_client` outranks the body. Read per request. Missing generation_config = wild-sampling signature.

## Building

- First-time: `./scripts/fetch-zig.sh` stages the pinned Zig at `.zig-toolchain/`.
- Zig caches configure-time subprocess output: after a toolchain/SDK change `rm -rf .zig-cache` or the link references a ghost path.
- **ALWAYS `zig build -Doptimize=ReleaseFast`, never bare `zig build`** (Debug 2–4× slower ⇒ fake regressions). `zig build test` does NOT refresh `zig-out/bin/mlx-serve` — rebuild before any live A/B.
- **Edit loop: `zig build check`** (35 s, Sema only) → `zig build test` (8 parallel shards, `-Dtest-shards=N`; a `sharded:` test splits its own work) → ReleaseFast only to RUN or MEASURE. ReleaseFast is single-threaded LLVM `-O3` per object (no flag tunes it): a plugin edit rebuilds in 0:30 with `-Dunits`, 3:30 without.
- **`-Dunits=true`** (off by default; `test`/`check` run without it) builds the host and 8 plugin groups as parallel objects; a plugin edit rebuilds its group only while `zig build unit-closure` holds (the plugin's imports stay in `sdk/`, `mlx/`, `abi/`, `kernels/`).
  A unit copies every global it reaches: host state is `Shared` / `SharedThreadLocal` (`sdk/util/shared.zig`), checked by `zig build units-check -Dunits=true` (data AND thread-local sections).
- **Every kernel registers with the kernel contract** (`sdk/testing/kernel_contract/REGISTERING.md`: one `contract.zig`
  per area or plugin, five gates against a float64 reference, planted controls must fail); it runs as ONE `sharded:`
  test split across the shards. Lint: `python3 scripts/lint.py --strict` (`zig build lint`) stays 0.
- Swift app: `bash app/build.sh`. The two bundle binaries move together.
- mlx + mlx-c: submodules built by `scripts/build-mlx.sh` (deployment target 26.2 → NAX kernels; script + `tests/test_mlx_staged_nax.sh` ASSERT `*_nax` in the metallib). Min macOS 26.2. Bump = checkout tag → rerun → re-diff `src/mlx/ffi.zig` externs. Brew: webp ≥ 1.6.0.
- **Linux/Windows CUDA**: build with `./scripts/build-linux.sh` / `scripts\build-windows.ps1`; MLX's JIT needs `CUDA_HOME`; custom MSL runs on Metal only. Kernels, per-device defaults, memory, measuring, and its rules (GPU-pool loads, graphs, kernel cache, matching MLX's bits): `docs/CUDA.md`.
- Rebuild Jinja after `lib/jinja_cpp/*.cpp` changes: compile the 7 `.cpp` (`clang++ -std=c++17 -O2 -DNDEBUG -I .`) into `obj/` and `ar rcs libjinja.a obj/*.o`.

## Testing — TDD is mandatory

Order: (1) failing test FIRST, for the right reason; (2) minimum code to green; (3) full suite (`zig build test` all steps, 0 fail + `bash app/test.sh`/`swift build` + relevant `tests/*.sh`); (4) refactor. A live curl is a sanity check, NOT a test.

Feature = unit test that fails without it (+ integration script if HTTP-observable). Bug fix = regression test red→fix→green, red-on-revert. Cross-arch = cover every touched arch. Refactor = characterization test first. UI/build scripts = factor a pure helper and test that.

**Class bugs get class guards.** A live failure revealing a CLASS ships: the instance regression test; a corpus entry or universal invariant in `src/host/chat/format_corpus_test.zig`; a rule here + story in `docs/gotchas/`.

Hermetic suites: `zig build test -Dtest-filter="format corpus"`, `-Dslow-tests -Dtest-filter="tool traffic"` (the replay is gated behind `-Dslow-tests`; the bare filter compiles and reports success without running it). Full suite: `zig build test -Dslow-tests`. Full matrix: `tests/CLAUDE.md`.

## Releases & benchmarking

- Process/CalVer/CHANGELOG: `/release`. Per-release perf gate = `./tests/bench.sh` on the FINAL tree vs the previous column in `benchmarks.md` (ONE new column per release).
- Methodology: `/bench` (llmprobe measures; bench.sh drives mlx-serve, `--url` probes another engine). Only diff same-methodology cells; spec cells are variance: same-session ratios only, name the engine in every win, an A/B arm is proven by ENGAGEMENT lines. Record = `benchmarks.md` + `~/claude-tmp/bench-<tag>/`.

## Conventions

- KISS, DRY, YAGNI; simple is not easy: the elegant solution, minimal and fast code, no speculative abstractions or knobs.
- Tests at the bottom of each source file; shell integration tests in `tests/`. Env levers only for paths with two arms worth comparing (lossy/tradeoff), never for an obvious win/fix; every one is registered in `src/env.zig` (kind, owner, since) and read through it.
- Commit messages and PR bodies carry no `Co-Authored-By` or generated-by line.
- Inference thread is the SOLE mlx caller (even frees) — media gen posts to `gen_queue`, never a gpu mutex. Chat runs between gen steps (`Progress.boundary`), CPU-only stages on a worker (`gen_sse.offload`), the job's peak billed to admission (`gen_reserve_bytes`; per request via `GenRequest.peak_bytes` where the request moves it: a 1536 3D cascade vs 1024).
- Concurrent requests batch-decode on pure-attention archs, the qwen3_5 family incl. `qwen3_5_moe`/`qwen4_exp` (`configBatchesDecode`) and llama.cpp GGUFs; `--max-concurrent` sizes the submit queue, not a decode gate. A slot's verdict is a `BatchVerdict` reason (`[batched] slot serial: <reason>`, `/props` `batching`, `/v1/models` `batched_decode`, `decode_serial_total{reason}`). Slots joining mid-generation drain lazy pipeline state first.
- A batched group past 1024 KV tokens attends PER SLOT (`perSlotBatchedAttn`: own view, no pad/stack/array mask; causal + quantized-KV kernels eligible). Below that, and on qwen4's QSA reads, the STACKED arm is capped by PADDING WASTE (`groupKeepCount`, `MAX_PAD_WASTE` 1.5, must stay < 2.0); longest slots fall to serial.
- A batched-decode guard that only runs at N=1 pins a shape that never ships: `MLX_SERVE_FORCE_BATCHED=1` at one slot still has `batch == 1` inside the forward (decode-attn-quant + fused QK-norm gates key on it). `tests/test_batched_equivalence.sh` runs a real two-stream arm; both kernels log `[batched] ... engaged (slots=N)`.
- A cold prefill YIELDS to decode ticks at chunk boundaries (`scheduler.interleaveDecodeTick`, up to 8 ticks for a quarter of wall time, `interleaveTicksFor`) and narrows to 2048 while anyone decodes (`companyPrefillChunk`: a chunk boundary is the only yield point); bounds the GAP, not 4-way aggregate. Idle cost zero. Guard: `tests/test_prefill_interleave.sh`. The decode share DEFAULTS to 0.25 (`DEFAULT_DECODE_SHARE`; `--prefill-decode-share 0` = the old schedule): without one a quick question waited 90-120 s behind a 90k prefill. With a decode share, the same boundary also admits small pending requests (`admitSmallAtBoundary`: uncached tail at most one chunk, up to 4 per boundary within one decode share of wall time, never from a nested prefill) so a newcomer waits one chunk, not the whole prefill. Guard: `tests/test_boundary_admission.sh`.
- KV reuse via prompt-prefix matching; invalidated after tool calls + pad-only gens; hot cache spills to SSD; RAM invalidation propagates to disk.
- Chat templates live in model dirs; Jinja renders with fallback formatting.
- **Diffs are read by a human. Keep them small.** A comment says what the code cannot (a non-obvious WHY, a contract, a unit), in one to three lines. Never: bug history, measurements, audit trails, review item numbers, dates, "PR #NNN", what an older commit did, or a restatement of the code. That belongs in the commit message and, if it is a rule, in `docs/gotchas/*.md`.
- **A dependency bump moves the pin and nothing else**: docs never name a submodule commit (git records it), a doc gets a line only for a user-facing change, and an app catalog entry lands only where a surface shows it (the tray lists `gemmaModelOptions` ids with `4bit`).
- **No source-scan tests** (`@embedFile` + "this string appears in that function"): they pin text, not behaviour, and pass against their own needles (lint enforces it). Test the behaviour; a class guard that can only be text is one line in `scripts/lint_pins.txt` (`!needle` forbids). A test comment is one line saying what the bar is.
- **One story per gotcha, one line per rule.** A gotcha entry is the defect, the cause, the fix and the guard in under 20 lines, written once after the work lands. No round-by-round logs, no ledgers, no "what the reviewer said". CHANGELOG: one user-facing sentence per change, no provisional numbers.
- **New model support starts from mlx-lm's layout**: before writing a converter or loader, read mlx-lm's model file (merged or open PR) and the MLX packs already on Hugging Face, and match their weight names, config keys and quant layout so packs load across engines. Invent only what mlx-lm has no answer for.
- **Code two plugins need belongs in the SDK** (recommendation): when a second plugin wants a block, kernel or helper, promote it to `sdk/blocks/` or `sdk/bases/blocks/` with both plugins' tests rather than copying it, and land a speed-up in `sdk/` first. A deliberate one-model change copies the block into the plugin, marked `//! copied_from: <sdk path>`: a debt to fold back.
- **One feature, one PR**: UI + Zig in the same PR; split only genuinely unrelated features.

## Supported architectures

Dispatch on `config.json` `model_type`: the first plugin whose `config.zig` claims it serves it (`build/plugins.zig`
registry). Each plugin's `AGENTS.md` holds its rules, quirks, packs and oracles (`src/plugins/<name>/AGENTS.md`; the
name column below is the folder). With `--mlx-gguf` (opt-in, experimental) a GGUF that `sdk/quant/gguf/claim.zig`
`servablePath` accepts (arch `qwen35`/`qwen35moe`/`gemma4`, every tensor type known) runs on the MLX path as the
matching arch; `--engine ds4|llama` still wins. Every other GGUF bypasses MLX → embedded engine by header
(`engine_pick.preferredEngine`: antirez DSV4-Flash + the ds4-only archs `deepseek41`/`qwen4exp`/`glm-dsa`/`glm5-next`
→ ds4, else llama.cpp); a SPLIT GGUF (`split.count` > 1) always → llama.cpp (ds4 `exit()`s on a missing shard). ds4
DSpark arms with `--dspark` when a `-DSpark-` GGUF sits beside the model (`mtpDraftTokens() > 1`, not `hasMtp()`).

| model_type | Plugin | Notes |
|---|---|---|
| `gemma4`, `gemma4_text`, `gemma4_unified(_text)` | `gemma4` | SigLIP vision, clipped linears, PLE; Gemma 4 assistant drafter |
| `gemma3`, `gemma3_text` | `gemma3` | + EmbeddingGemma encoder when `use_bidirectional_attention` (`embedding_gemma`) |
| `embedding_gemma2` | `embedding_gemma2` | Gemma 4 trunk trained bidirectional; images + video frames on `/v1/embeddings` `messages` |
| `bert` | `bert` | Embedding encoder, per-checkpoint pooling |
| `llama`, `mistral`, `qwen2`, `qwen3`, `qwen3_moe(_text)`, `qwen3_next`, `hy_v3` | `llama`, `qwen3`, `qwen3_next`, `hy_v3` | Standard dense / DeltaNet / Hunyuan 3 MoE |
| `qwen3_5(_text)`, `qwen3_5_moe(_text)`, `minicpmv4_7` | `qwen3_5` | GDN + optional MoE; Qwen3-VL vision; Qwen3.8 packs; MiniCPM-V 4.7 tower + canvas M-RoPE; native MTP |
| `prism_hadamard_qwen35` | `prism_hadamard_qwen35` | Bonsai 2: qwen3_5 behind block Hadamard rotations, f16 activations, exact 2-bit GEMVs |
| `qwen4_exp(_text)` | `qwen4_exp` | Qwen3.8-Flash-Next: hyper-connections, n-gram PLE, QSA; batched text, exclusive MTP slot |
| `nemotron_h`, `nemotron_h_puzzle`, `nemotron_h_omni` | `nemotron_h` | Transformer + Mamba2 hybrid, NoPE attention, RADIO vision, LiLiCorr DFlash, MTP sidecar |
| `lfm2`, `lfm2_moe`, `lfm2_vl` (`lfm2-vl`) | `lfm2` | Gated conv + attention hybrid; SigLIP2 NaFlex tower with tiling |
| `lfm3`, `lfm3_vl`, `lfm3_audio`, `lfm3_omni` | `lfm3` | Gated conv + full/sliding gated attention, sandwich norms, zero-tail partial RoPE; VL + speech towers via `MediaHooks`; speech OUT via `SpeechHooks` (OpenAI `modalities`, `/v1/audio/speech`) |
| `laguna`, `kolibri1`, `flint` | same | Sliding + NoPE attention arms with sigmoid MoE and an UNGATED shared expert |
| `gpt_oss`, `spark2_5`, `k2_horizon`, `muse_glimmer(_text)` | same | Sink attention / sliding dense / grouped-RMS Llama / Muse VL (Harmony channels, ATEM tools) |
| `mimo_v2`, `mimo_v2_flash` | `mimo_v2` | MiMo-V2.6-Flash, mlx-lm's layout (release refused by name); legacy forward deleted; MTP = 3 heads (`mtp_head.zig`); stream packs (`expert_streaming`) stream their `experts.bin` through the same table |
| `inkling_mm_model` | `inkling` | NO RoPE (relative-logit bias + short convs); serial, spec off |
| `deepseek_v4`, `deepseek_v41` | `deepseek_v4`, `deepseek_v41` | 1M ctx, module-owned state (V4 single-flight; V4.1 one state per slot, plain decode batched), DSML tools on our byte-pinned templates, DSpark; V4.1 images + EXL3 stream packs |
| `bailing_hybrid` | `bailing_hybrid` | Ling 3.0: KDA + MLA hybrid MoE |
| `glm5_next(_text)`, `glm_moe_dsa` | `glm5_next` | GLM-5.3-Flash: KDA + DSA inside mHC, 288-expert MoE, batched decode, `mtp.0` head, vision |
| `kev`, `laya`, `d1`, `clef` | same | Typed decisions (`POST /v1/decisions`) |
| `qwen3_asr` | `qwen3_asr` | `/v1/audio/transcriptions` only, + the forced aligner (`qwen3_forced_aligner`) |
| `nemotron_diarization` | `nemotron_diarization` | `diarized_json` speakers, run once over the whole clip |
| `*.gguf` | engines | ds4 / llama.cpp (above); GGUF presence WINS over a stray config.json |
| media | `flux2`, `krea`, `mage_flow`, `qwen_image`, `moss_image`, `ltx`, `minimax_h3`, `hunyuan3d`, `trellis2`, `pixal3d`, `acestep`, `music3`, `stable_audio3`, `yue2`, `qwen3_tts`, `kokoro` | Media plugins (`abi/media.zig`); configless packs are claimed from `model_index.json` / `pipeline.json` / `model_config.json` (`sdk/media/pack_claims.zig`) |

Models with `vision_config` but no vision weights disable vision. Embedded-engine detail: `docs/reference.md`.

## Unified media generation

One server, one registry — image/audio/video/3D coexist with chat. Engine slots are MODALITY-named unions on `LoadedModel`; new backend = one union arm + impl file. App flow load→generate→unload; headless starts idle. Which pack or `model_type` is which modality is its media plugin's answer (`MediaTable.claims` + `modality`, read through `host/media_jobs/plugin/served.zig`): discovery, `list`, the load arms and each `/v1/models` row's `modality` read it. Downloads → `~/.mlx-serve/models` (Swift + Zig).

- `POST /v1/images/edits` = OpenAI multipart translated by `gen.openaiEditFormToJson` into the `mode:"edit"` JSON body; unhonored fields = NAMED 400.
- LoRAs are STACKED, ONE grammar across image/LTX/H3: `lora_paths`+`lora_scales` (cap 8, `gen.parseLoraFields`), summed at forward — never merged. Resident backends reconcile via `setLoras`; H3 pre-validates (`lora.validatePath`), Turbo = file 0.
- **FLUX real CFG** (`guidance_scale`+`negative_prompt`, `gen.ImageEngine.supportsGuidance`): distilled klein takes NEITHER field (one forward per step); the undistilled base 9B (`flux2-klein-9b-base`) needs it — TWO forwards per step, `uncond + scale·(cond−uncond)`.
- Endpoint/field/backend detail: `docs/reference.md`. Guards: `tests/test_unified_gen.sh` + per-modality scripts; parity via env-gated cos oracles (`tests/dump_*_fixtures.py`).
- **ANE media offload** (opt-in, LOSSY: `--ane-image` Krea, `--ane-video` H3, `--ane-audio` ACE-Step; ONE value `ane.media_offload`): calibrated share that sticks, free-disk-capped lineage cache, fp16 seam SCALE, LoRA blocks decline. Rules: `docs/reference.md` "ANE prefill rules" → media offload.
- **Media DiT linears run int8 on NAX (LOSSY; default ON only with NAX + 64 GiB RAM, `--w8a8` / `--no-w8a8`, app toggle) and the int8 copy REPLACES the packed weights**: each loader frees a block's packed arrays as it converts it (`Weights.dropBlock`), so the load peak is the converted size; the bill is `w8a8.residentBytes` (a 4-bit pack GROWS ~1.8x). Kernels need NAX (`w8a8.nax()`, also gates int8 attention); name-keyed maps hold the copy in a `w8a8.Overlay`. Bar per engine: `w8a8.agreement` cos AND rms vs the packed forward, resident bytes vs the bill (`expectBilled`); `zig build test` runs with it OFF.
- **"Free disk" is what the OS will GRANT, not statfs** (`msv_volume_free_for_use` = `volumeAvailableCapacityForImportantUsage`; `kv_disk_cache.volumeSpace` + the ANE cap read it, statfs is the fallback): purgeable space is released on demand.

## Prompt-based skills / downloads / debugging

- Product skills: `~/.mlx-serve/skills/*.md` (frontmatter trigger substring → body into system prompt; `SkillManager` rescans on mtime).
- `DownloadManager`: streams to `.partial`, Range resume, 3 retries, cancel preserves partial, size-matching files skipped.
- Server log `~/.mlx-serve/logs/mlx-serve-<port>.log` is THE post-mortem file (`--log-level debug`). Grep: `jinja error:`, `[cache]`, `<- N+M tokens`, `tool_msgs=`, `[spec-stats]`, `spec-gate:`, `[loop-stop]`, `[lan] proxy`, `[disk-cache]`, `[hot-cache]`, `[admission]`, `[providers]`. Capture: `MLX_SERVE_RAW_DUMP_FILE=<abs>` → `tests/harvest_tool_traffic.py`. Reproduce tool bugs `stream:false` first; `pkill -f mlx-serve` between KV-poison tests.

## Rules

Area rules live with the code, one line each (stories in `docs/gotchas/`):
- `src/host/AGENTS.md`: API surfaces, request parsing, streams, LAN.
- `src/host/chat/AGENTS.md`: templates, tool calls, reasoning.
- `src/engines/mlx/AGENTS.md`: loading, memory bills, prefix cache, KV, spec decode, kernels.
- `src/plugins/AGENTS.md`: configs, tokenizers, weights, vision, media, parity; each plugin's own in its folder.

### Licensing & third-party code

Ported kernels + vendored code are enumerated in `NOTICE` (the ONE place); `LICENSE`/`LICENSE-APACHE-2.0`/`NOTICE` ride every packaging path, pinned by `tests/test_release_workflow_gates.sh`. To enumerate ports, grep comments for `mlxfast|oMLX|MTPLX|mlx-lm|port`. Full story: `docs/reference.md` "Licensing".

### App (Swift)

All app-side rules: `app/CLAUDE.md`; stories: `docs/gotchas/app.md`. Crossovers: agent budgets derive from advertised `context_length`; the app ALWAYS emits memory-critical launch flags (`ServerOptions` defaults mirror Zig defaults — change both together); the two bundle binaries move together.
