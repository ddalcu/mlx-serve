# mlx-serve – project context for AI

Native Zig server that runs MLX-format LMs on Apple Silicon and exposes OpenAI-compatible and Anthropic-compatible HTTP APIs. No Python.

## Stack

- **Zig** 0.16+; **mlx-c** (Apple) via Homebrew, FFI in `src/mlx.zig`
- **Jinja engine** (`lib/jinja_cpp`): llama.cpp's C++17 Jinja2 + nlohmann/json, pre-compiled as `libjinja.a`
- **stb_image** (JPEG/PNG) + **libwebp** (WebP) for vision; **safetensors** weights; BPE tokenizers

## Layout

### Zig server (`src/`)

| Path | Role |
|------|------|
| `main.zig` | Entry, CLI flags |
| `mlx.zig` | mlx-c FFI |
| `model.zig` | Config + safetensors loading |
| `tokenizer.zig` | BPE tokenizer |
| `transformer.zig` | Forward pass; arch dispatch (attention, MLP, MoE, GatedDeltaNet) |
| `generate.zig` | Autoregressive generation, sampling, PLD/drafter/MTP orchestration |
| `chat.zig` | Chat templates (ChatML, Gemma, Llama-3, Jinja2); thinking tags; tool call parsing |
| `vision.zig` | Gemma 4 SigLIP vision encoder |
| `server.zig` | HTTP: `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/responses`, `/v1/embeddings` (encoder-only BERT; whole input array batched into padded masked GPU forwards), WebSocket on `/v1/responses` |
| `responses.zig` | OpenAI Responses API: parser, envelope, in-memory `ResponseStore`, compaction blob |
| `ws.zig` | RFC 6455 WebSocket framing (server-side, generic over `Conn`) |
| `pld_index.zig` | PLD n-gram index (`PldLookup.findMatch`, `ngramRepeatScore`) |
| `drafter.zig` | Gemma 4 assistant drafter (cross-attention spec-decode) |
| `mtp.zig` | Qwen 3.5/3.6 native MTP head (sidecar spec-decode; self-contained — delete this + `Generator.nextMtp` to remove the feature) |
| `diffusion.zig` | DiffusionGemma block-diffusion generation: canvas denoising loop, entropy-bound sampler, self-conditioning, per-slot `Runner` |
| `status.zig` | TUI status bar |
| `log.zig` | Leveled logging |
| `build.zig` | Links mlx-c, libjinja, libwebp, stb_image |

CLI flags: `--model --serve --host --port --prompt --max-tokens --temp --top-p --top-k --ctx-size --timeout --reasoning-budget --no-vision --pld --pld-draft-len --pld-key-len --drafter --draft-block-size --no-mtp --mtp-depth --kv-quant --prefix-cache-entries --prefix-cache-mem --max-concurrent --model-dir --log-level --version --help`

Sampling defaults for request fields the client OMITS resolve as: request body > `--temp`/`--top-p`/`--top-k` launch flags (the app passes its Settings values) > the model's `generation_config.json` (Qwen 3.6: top_k 20 / top_p 0.95; Gemma 4: top_k 64 / top_p 0.95) > hardcoded (1.0/1.0/off). Claude Code omits all sampling params, so pre-2026-06 it sampled the full untruncated distribution at temp 1.0.

### Swift macOS app (`app/Sources/MLXServe/`)

| Path | Role |
|------|------|
| `MLXServeApp.swift` | App entry, menu bar + Chat/Browser windows |
| `AppState.swift` | Global state, chat session persistence |
| `Models/{ChatModels,AgentModels}.swift` | `ChatMessage`, `ChatImage`, `SerializedToolCall`, `AgentPlan` |
| `Services/APIClient.swift` | HTTP + SSE streaming client |
| `Services/AgentPrompt.swift` | System prompt, 10 tools, `SkillManager` |
| `Services/AgentEngine.swift` | Shared agent logic: history, tool exec, repetition tracking, overflow |
| `Services/ToolExecutor.swift` | Tool handlers (shell, file, search, browse, webSearch, saveMemory) |
| `Services/DocumentIndex.swift` | Mini in-memory RAG for "attach a folder": chunker, hybrid retrieval (cosine + IDF lexical), `searchDocuments` tool. Embeds via `ServerEmbedding` only (server `/v1/embeddings`, GPU): `autoProvider` probes `/v1/models`, auto-downloads `bge-small-en-v1.5-8bit` (35 MB) when no encoder is known, and registers it via `/v1/load-model` by absolute path. Server down / provisioning failure → lexical-only retrieval (no NLEmbedding fallback) |
| `Services/{ImagePreprocessor,BrowserManager,ServerManager,TestServer,AgentMemory}.swift` | Image prep, WKWebView, server lifecycle, embedded test server (port 8090), agent context |
| `Views/{ChatView,StatusMenuView,BrowserView}.swift` | Chat UI + `runAgentLoop()`, menu bar, browser |

## Building

- **ALWAYS build Zig with `-Doptimize=ReleaseFast`. Never use bare `zig build`.** The default optimize mode is Debug, which is 2-4× slower than ReleaseFast for the same workload (e.g. Gemma 4 E4B 4-bit prefill measured at 230 vs 558 tok/s on the same machine — pure build-flag difference, no code change). A Debug binary at `zig-out/bin/mlx-serve` will silently make every benchmark look like a regression. Zig caches incremental builds aggressively, so `-Doptimize=ReleaseFast` is essentially free on rebuild — there is no reason to ever skip it.
- **Always use `./app/build.sh` for the Swift app, not direct `swift build`** — the script knows the right Swift-version flags, links Zig artifacts, signs the bundle, and keeps `MLXCore` + `mlx-serve` in lockstep. Skip notarization in dev with `SKIP_NOTARIZE=1 bash app/build.sh`.
- Zig only: `zig build -Doptimize=ReleaseFast` (needs `brew install webp`). **No exceptions** — never bare `zig build` for performance work or for any artifact that ends up in `zig-out/bin/`. A quick way to spot a Debug binary that slipped through: it's noticeably larger than the ~4.5 MB ReleaseFast binary (Debug runs ~2× that), and `du -h zig-out/bin/mlx-serve` should match `/Applications/MLX Core.app/Contents/MacOS/mlx-serve` in size.
- Direct `swift build` (escape hatch only — fast iteration on a Swift-only change): `cd app && swift build -c release -Xswiftc -swift-version -Xswiftc 5`. Don't ship a build that didn't go through `build.sh`.
- **Rebuild Jinja** (after `lib/jinja_cpp/*.cpp` changes): `cd lib/jinja_cpp && for f in jinja_wrapper caps lexer parser runtime jinja_string value; do clang++ -std=c++17 -O2 -DNDEBUG -I . -c $f.cpp -o obj/$f.o; done && ar rcs libjinja.a obj/*.o`

The `-Xswiftc -swift-version -Xswiftc 5` flag forces Swift 5 mode under Swift 6.3 (Xcode 26+) — required because the pinned `swift-sdk` 0.10.x emits `[#SendingRisksDataRace]` errors otherwise. Pin held at 0.10.x for macos-14 / Swift 6.1 CI compat. `app/build.sh` already passes the flag; only direct `swift build`/`swift test` need it.

## Testing

**TDD is mandatory for every code change — features, bug fixes, refactors, build-script tweaks, all of it.** No exceptions, no "I'll add the test after the smoke test passes." A live curl against a running server is a useful sanity check but it is NOT a test — it doesn't survive in CI, it doesn't pin the behavior against future regressions, and it doesn't prove the fix is on the right code path.

The order is fixed:

1. **Write the failing test first.** It must fail for the right reason — i.e. the symptom you're fixing or the capability you're adding. If you can't articulate a failing test, you don't yet understand the change you're making.
2. **Make it pass.** Minimum code to flip the test green; no opportunistic refactoring in the same step.
3. **Run the full suite** (`zig build test` + `cd app && swift build` + the relevant integration script) — a green new test doesn't excuse a red regression elsewhere.
4. **Then iterate** — refactor / clean up / extend, with the test as a tripwire.

Concrete rules for each change class:

- **Feature**: at least one unit test that fails without the feature and passes with it. Add an integration test (`tests/test_*.sh`) when the feature is observable over HTTP.
- **Bug fix**: a regression test that reproduces the bug from the user's perspective. The diff order must be `test (red) → fix → test (green)` — verify locally that reverting the fix turns the test red again.
- **Cross-arch work**: cover every architecture the change touches, not just the one you're holding (`gemma4`, `qwen3_5_moe`, `lfm2`, `nemotron_h`, `deepseek_v4` via ds4, etc.). One arch's pass is not the suite's pass.
- **Refactor / no-behavior-change**: at least one existing test must exercise the refactored path before the change. If coverage isn't there, add a characterization test first, then refactor under it.
- **Untestable surface (UI, build scripts, etc.)**: factor a pure helper out and test that. "I can't test SwiftUI" usually means "I didn't extract the testable piece yet."
- **Class bugs get class guards — no whack-a-mole.** When a live failure reveals a CLASS of bug (an escaping gap, a format-family leak, a dispatch hole — anything where "fix one model" could break or miss another), the fix ships with THREE things: (1) the instance regression test; (2) a corpus entry or universal invariant in `src/format_corpus_test.zig` that exercises the whole class across every family/entry, so future additions are covered automatically; (3) a CLAUDE.md gotcha naming the class, its symptom signature, and the rule that prevents reintroduction. Precedents: the no-tag-leak + valid-JSON-args invariants, the history round-trip serialization invariant, the spec-decode dispatch engagement counts.

Existing tools:

- Unit tests OK; integration tests with real models are the real tests.
- After big features: build `mlx-serve` + the app bundle, run `.app` with `TestServer.swift` enabled, run the agentic harness.
- Always run `zig build test` and `swift test` (or `swift build`) before submitting.

| Command | Purpose |
|---|---|
| `zig build test` | Zig unit tests |
| `zig build test -Dtest-filter="format corpus"` | Hermetic format-correctness corpus (`src/format_corpus_test.zig`): real captured model outputs across families through splitThinkBlock/stripThinkBlock/parseToolCalls, plus universal no-tag-leak + valid-JSON-args invariants. No weights; harvest workflow in the file header. |
| `FORMAT_MODELS=<csv> ./tests/test_format_matrix.sh` | Live format matrix: 7 format-family representatives (Qwen think-tags, Gemma 4 channel-tags + custom tool args, Qwen MoE raw-JSON, Gemma 3 fallback/fenced-JSON, GGUF/llama.cpp, DSV4-Flash/ds4) × 8 checks: thinking split, stream+tools content routing, tool-arg byte-fidelity (chat + /v1/messages tool_use), omitted max_tokens. One model at a time on port 11297 with `--log-level debug`; FAILs echo raw-output dumps for corpus harvesting. `FORMAT_MODELS=gemma4-e4b` ≈ 3 min smoke; missing paths skip cleanly. |
| `VALIDATOR_MODELS=<csv> ./tests/test_validator_matrix.sh` | API-compliance + agentic matrix: per architecture (gemma4, gemma3, qwen3_5 dense, qwen3_5_moe, qwen3_moe, llama-engine GGUF, ds4 GGUF) boots a server, runs llmprobe (`~/projects/agents/responses-chat-messages-validator`, 112 checks across /v1/responses + /v1/chat/completions + /v1/messages), then the pi 2-turn html agentic case (multi-turn tool calls) via pi_integration_run.sh. `SKIP_PI=1`/`SKIP_PROBE=1` run one layer; pkills ALL serving mlx-serve instances (81GB ds4 coexistence). Logs in tests/validator-results/. |
| `cd app && swift test` | Swift unit tests |
| `./tests/integration_test.sh [model_dir] [port]` | 36 end-to-end API tests |
| `./tests/test_tool_response.sh [port]` | Tool calling round-trip |
| `./tests/test_kv_cache_poison.sh [port]` | KV cache poisoning regression |
| `./tests/test_anthropic_api.sh [port]` | Anthropic Messages API |
| `LLAMA_GGUF_MODEL=<file.gguf> ./tests/test_llama_gguf.sh [port]` | Embedded llama.cpp GGUF engine end-to-end (auto-routing, chat, streaming, Anthropic) |
| `PLD_TEST_MODEL=<dir> ./tests/test_pld_equivalence.sh` | PLD byte-equivalence (default gemma-4-e4b-it-8bit) |
| `./tests/test_streaming_pld.sh [port]` | Streaming PLD byte-identical to non-streaming |
| `./tests/test_pld_tools.sh [port]` | PLD with tools present: engages (`[spec-stats] mode=pld`), tool calls + echo text byte-identical to no-PLD baseline (starts its own server) |
| `./tests/test_drafter_tools.sh [port]` | Drafter with tools present: same contract as test_pld_tools.sh for the Gemma 4 assistant drafter, across chat non-stream, /v1/messages stream + non-stream, and /v1/responses non-stream, with a per-request `[spec-stats]` engagement count — equality checks alone can't catch a dispatch hole because the regular-decode fallback is output-identical (starts its own server) |
| `MTP_TEST_MODEL=<dir> ./tests/test_mtp_equivalence.sh [port]` | Qwen native-MTP head: auto-load, per-request opt-out, first-100-char temp-0 equivalence vs `--no-mtp`, and a per-request `[spec-stats] mode=mtp` ENGAGEMENT count on chat (stream + non-stream) + /v1/messages — output equality alone can't see a silent fallback to regular decode. Includes an ACCEPTANCE-FLOOR check (best per_draft_pct >= 50%): a structurally broken head — e.g. a sidecar built without folding +1 into Qwen's delta-encoded norms — engages, accepts ~0%, gate-falls-back to regular decode, and passes both equivalence and engagement. Sidecars rebuild from the original Qwen repo via tests/build_mtp_sidecar.py. |
| `./tests/test_completions_spec.sh [port]` | /v1/completions spec-decode: PLD + drafter engage on the FIM endpoint, first-N-token equivalence, stream/non-stream leading-space parity (starts its own server) |
| `./tests/test_batched_transition.sh [port]` | Concurrent-stream consistency: legacy→batched decode transition must not duplicate/drop tokens (starts its own server) |
| `./tests/test_drafter_equivalence.sh [port]` | Gemma 4 drafter byte-equivalence |
| `UD_MOE_MODEL=<dir> ./tests/test_ud_moe.sh` | Unsloth UD MoE load + generate (default Qwen3.6-27B-UD-MLX-4bit) |
| `NVFP4_TEST_MODEL=<dir> ./tests/test_nvfp4.sh` | NVFP4 checkpoint (issue #24): load, banner reports nvfp4 mode, temp-0 coherence canary. Default Qwen3-0.6B-nvfp4; pass a gemma-4 QAT nvfp4 dir to exercise mixed affine-override resolution. Hermetic counterparts: `computeQuantParams` + nvfp4 qmatmul/gather tests in transformer.zig. |
| `./tests/test_diffusion_gemma.sh [port]` | DiffusionGemma block diffusion end-to-end: load, greedy chat coherence + EOS semantics, diffusion-loop engagement + early-stop convergence (steps < max), block-wise SSE streaming, /v1/messages, tool calls, thinking, multi-canvas long generation. Hermetic counterparts: sampler math tests in diffusion.zig. Live parity: `DIFFUSION_TEST_MODEL=<dir> zig build test` runs the converged-canvas self-consistency test (reference 63/64 no-sc, 64/64 with sc) — this is the test that catches the lazy-slice expert-zeros class (red on revert) where toy-scale synthetic gathers stay green. |
| `./tests/test_long_agent_memory.sh [port]` | 10-turn Claude-Code-style agent: plants 3 facts in turn 1, recalls them across mode transitions (tools on/off, thinking on/off). Catches "model acts like first-time-seen" regressions. |
| `EMBED_TEST_MODEL=<dir> ./tests/test_embeddings.sh [port]` | /v1/embeddings (encoder-only BERT): OpenAI shape, batched-vs-single cosine parity across mixed lengths (pins the padding mask + masked mean-pool), >EMBED_MAX_BATCH round-trip, 400 on generation, hot-load + stub `embeddings` capability beside a chat default, and `/v1/load-model` register-by-absolute-path (incl. the JSON `\/`-escaped form Swift clients send) for encoders outside --model-dir. Default mlx-community/bge-small-en-v1.5-8bit. Hermetic counterparts: `buildPaddedBatch`/`buildKeyPadMask`/`maskedMeanPoolNormalize` tests in generate.zig. Live app-side flow: `DOCS_RAG_AUTODL_PORT=<port> swift test --filter testAutoProviderProvisionsEncoderEndToEnd` (downloads from HF). |
| `./tests/test_disconnect_cancel.sh [model_dir] [port]` | Client-disconnect handling during long prefills: SSE keepalives flow every 5s while a request waits on first tokens (Anthropic `ping` events / OpenAI `: keepalive` comments), and a vanished client cancels its slot within one probe interval — the chunk loop aborts the ghost prefill (`error.Cancelled`) instead of grinding minutes of abandoned work that piles up behind Claude Code retries. Starts its own server. |
| `./tests/test_messages_stream_thinking_tools.sh [model_dir] [port]` | /v1/messages STREAMING with thinking + tools together (Claude Code's exact shape) on a template-opened-think model: no think-tag leak in text deltas, SSE content-block lifecycle validity (every delta/stop references an open block), thinking_delta + tool_use emission. Starts its own server. Hermetic counterpart: `chat.streamThinkGate` unit tests + the corpus streaming-gate replay. |
| `./tests/test_kv_quant_equivalence.sh [model_dir] [port]` | Off-vs-4-bit-vs-8-bit KV cache equivalence (per-arch first-N-token thresholds, env-overridable). |
| `./tests/test_kv_quant_per_request.sh [model_dir] [port]` | Per-request `kv_quant` body field plumbing — confirms parse + scheduler routing for each scheme. |
| `./tests/test_prefix_cache_mem.sh [model_dir] [port]` | `--prefix-cache-mem` budget — long-prompt traffic must stay within the byte cap and produce eviction log lines. |
| `./tests/test_turboquant_equivalence.sh [model_dir] [port]` | TurboQuant smoke (non-NaN, ≥5 tokens, first-N diff vs dense). |
| `SOAK_DURATION_HOURS=N ./tests/test_soak_24h.sh [model_dir]` | Plan 01 A8 — 4-way concurrent soak (chat/agent/Anthropic/tool). RSS drift bounded; default 24h, set `SOAK_DURATION_HOURS=1` for CI smoke. |
| `./tests/bench.sh --family <gemma\|qwen36> [--lmstudio --omlx --concurrent N]` | Unified bench: prefill/decode/echo/code prompts × {none,pld,drafter} cells. Default mlx-serve only; opt in to comparison engines and concurrent throughput. |

## Versioning & Releases

CalVer `YY.M.N` (e.g., `v26.4.25` = 2026, April, 25th release). `N` auto-increments from the last GitHub release for that `YY.M` prefix; `build.sh` computes via `gh release list`.

**Version sources**: `app/Info.plist` (`CFBundleVersion`/`CFBundleShortVersionString`), Zig `-Dversion` build option (`build_options.version`), git tag (`gh release create v{version}`).

**Release**:
1. Update `CHANGELOG.md` with NEXT version (check `gh release list --limit 1` first — never reuse an existing tag)
2. Commit + push
3. `cd app && SKIP_NOTARIZE=1 bash build.sh` — prints the `gh release create` command
4. Run that command

### CHANGELOG style

**One entry per shipped release. No new entries for unshipped work — fold it into the next pending entry.** Always run `gh release list --limit 1` first; if the topmost CHANGELOG entry is newer than the latest GitHub release, that entry is unshipped and any new bullets get merged into it. Never bump version numbers ahead of an actual release.

Tone: high-level executive bullets, marketing-style. The audience is users/integrators, not contributors reading the diff.

- Lead each bullet with **what changed for the user** (capability, speed, model support), not the implementation.
- Quantify where impressive — concrete tok/s percentages, model names, the workload it applies to.
- Avoid: file paths, function names, internal symbol renames, line-count diffs, "we discovered that…", PR/issue numbers.
- 4–7 bullets per release. If you need more, the release is too big and should ship sooner.

Template:

```markdown
## vYY.M.N — Two-to-five-word headline

- **<User-visible thing>**: one or two sentences on the impact. Numbers if you have them.
- **<New model / API / behavior>**: what unlocks, when it kicks in, what stays the same.
- **<Speed or reliability win>**: workload + measured gain.
- **<Removed / deprecated thing, if any>**: why, and what users should do instead.

---
```

When in doubt, look at the existing entries (v26.5.4 and earlier) — keep the same density and tone.

## Benchmarking

Run `./tests/bench.sh --family <gemma|qwen36>` after major features/optimizations; CSVs go in `docs/perf-csvs/`, narrative summary in `BenchmarkLog.md`. Pass `--lmstudio` and/or `--omlx` to add comparison engines (chart renders into `docs/`); `--concurrent N` adds a batched-throughput row per cell. Default run is mlx-serve only — fast enough for tight dev loops.

## Conventions

- Minimal, DRY Zig; avoid unnecessary abstraction
- Tests at the bottom of each source file (Zig convention)
- Shell integration tests in `tests/`, need a running server
- Chat templates live in model dirs; Jinja renders them with fallback formatting
- Concurrent requests batch-decode together on batchable archs (pure attention); `--max-concurrent` sizes the submit queue, not a decode gate. Slots entering a batch mid-generation drain their lazy pipeline state first (`Generator.drainPipelineForBatch`) — see `tests/test_batched_transition.sh`
- KV cache reuse via prompt-prefix matching; invalidated after tool calls and pad-only generations

## Supported Architectures

Dispatched on `model_type` in `config.json` via `model.zig` (config/weights) and `transformer.zig` (forward). **GGUF files bypass this MLX dispatch entirely** and route to an embedded engine by file format — see "GGUF auto-routing" below.

| `model_type` | Family | Weight prefix | Vision | MoE | Notes |
|---|---|---|---|---|---|
| `gemma4`, `gemma4_text` | Gemma 4 | `language_model.model` | SigLIP | -- | Full vision, clipped linears, PLE |
| `diffusion_gemma` | DiffusionGemma | `model.decoder` | -- (tower not wired) | 128/top-8 | Gemma 4 26B-A4B trunk; BLOCK-DIFFUSION generation (src/diffusion.zig), not AR decode. See "DiffusionGemma" below. |
| `gemma3` | Gemma 3 | `language_model.model` | -- | -- | |
| `qwen3` | Qwen 3 | `model` | -- | -- | QK norm |
| `qwen3_5`, `qwen3_5_moe(_text)` | Qwen 3.5/3.6 | `language_model.model` | -- | Optional | GatedDeltaNet + MoE/dense, shared expert routing |
| `qwen3_next` | Qwen 3-next | `model` | -- | Optional | DeltaNet |
| `nemotron_h` | Nemotron-H | `backbone` | -- | -- | Hybrid transformer + Mamba2 SSM |
| `lfm2` | Liquid LFM2.5 | `model` | -- | -- | Hybrid gated conv + full attention |
| `llama`, `mistral` | Llama/Mistral | `model` | -- | -- | |
| `*.gguf` (any) | via llama.cpp | -- | -- | -- | Embedded libllama engine; reported as `model_type=gguf`. See Embedded engines. |

**TODO**: `lfm2-vl` (vision encoder), `phi`/`phi3` (different layout), `command-r` (different arch).

### GGUF auto-routing

A `.gguf` model path (or a directory containing one) bypasses the MLX safetensors path. `main.zig` picks the embedded engine by family: **DeepSeek-V4-Flash → ds4**, **every other GGUF → llama.cpp** (`model_discovery.isDs4GgufBasename` — case-insensitive `deepseek-v4-flash` prefix; mirrored client-side by `DownloadManager.ggufModelType`). The Swift app surfaces any `.gguf` as a selectable base model and the server auto-routes — no engine selector. Both engines share the same HTTP/chat-template/tool/SSE machinery via per-request dispatch on `LoadedModel.{ds4,llama}_engine`.

Models with `vision_config` but no vision weights (e.g., text-only quantized Qwen 3.5) gracefully disable vision at init. Swift app flags unsupported archs via `supportedModelTypes` in `HFModels.swift`.

### DiffusionGemma (block diffusion)

`model_type: diffusion_gemma` (google/diffusiongemma-26B-A4B-it, June 2026) reuses the Gemma 4 26B-A4B MoE trunk verbatim — same `forwardMoeWith` gemma4 branch, sigma-MoE router, v_norm, K=V alias on full layers, proportional RoPE — but GENERATION is a canvas-denoising loop (`src/diffusion.zig`), not autoregressive decode. `config.isDiffusion()` (canvas_length > 0) gates everything; `config.isGemma4Layers()` collapses gemma4 + diffusion onto the shared binding/forward arms.

- **Encoder/decoder split over one trunk.** The *encoder* runs the trunk causally (normal prefill, fills the slot's KV cache) using its own per-layer scalars (`model.encoder.language_model.layers.N.layer_scalar` — the only untied encoder weights; `ForwardCtx.use_encoder_scalars`). The *decoder* (`forwardDiffusionDecoder`) re-forwards a 64–256-token canvas of noise tokens BIDIRECTIONALLY each denoising step: q/k/v computed for the canvas only, cache K/V concat-READ (never written), sliding layers sliced to the last `sliding_window − 1` cached positions, no mask at all. `ctx.skip_lm_head` skips the 262K-vocab projection on encoder passes.
- **The loop** (`diffusion.Runner.nextCanvas`, one canvas per scheduler tick): uniform-RANDOM canvas init (no mask token), ≤48 steps, linear temp schedule (t_max 0.8 → t_min 0.4 on countdown), entropy-bound acceptance (sort entropies asc, accept while `cumsum − cummax ≤ 0.1`, renoise the rest), self-conditioning via soft embeddings (`softmax(processed) @ dequantized_embed_table × √hidden` — table dequantized once per request, ~1.5 GB transient), early stop when argmax is stable across `stability_threshold` steps AND mean entropy < 0.005. COMMIT = the argmax canvas; committed tokens are causally re-encoded at the START of the next tick (so an EOS-terminated request never pays for it). User temp < 0.01 → greedy denoiser; ≥ → categorical (temp 1.0 ≡ HF reference). Measured: 256-token canvas in ~10 steps on prose (25.6 tokens/forward), ~30% faster than the mlx-vlm reference.
- **Dispatch**: third engine-style branch (`runPrefillDiffusion` / `runDiffusionDecodeTick`, `Slot.diffusion`) — tokens flow through normal slot machinery, so all four HTTP surfaces + block-wise streaming work unchanged. PLD/drafter/MTP/batching never apply (MoE already non-batchable; `modelBatchable` also gates `isDiffusion`). Hot prefix cache is intentionally NOT consulted (restored snapshots leave cache VIEWS stale; the decoder reads views via denseView before any update would rebuild them) — wire it later by refreshing views post-restore.
- **Weights**: prefix `model.decoder`; experts packed as `experts.gate_up_proj` [E, 2M, X] (gate rows first) → `splitPackedGateUp` slices + MATERIALIZES halves at load; self-conditioning module under `model.decoder.self_conditioning.*`; `model.encoder.vision_tower.*` dropped at load (tower not wired — `has_vision` forced false).
- **Template/EOS**: standard Gemma 4 `<|turn>`/`<|channel>thought` template; EOS set {1, 106, 50}. Raw `/v1/completions` (no chat template) degenerates — reference behaves identically; instruct-only checkpoint trait, not a bug.

## OpenAI Responses API (`/v1/responses`)

Pure data in `responses.zig`; HTTP/orchestration in `server.zig`. Supports `POST`/`GET`/`DELETE /{id}`.

- **Envelope** (`buildResponsesEnvelope` + `ResponseEcho`): echoes `tools`, `tool_choice`, `text`, `reasoning`, `usage` (with `cached_tokens` + `reasoning_tokens`), `truncation`, `parallel_tool_calls`, sampling params, `metadata`, etc. Renderers reshape into the strict ResponseResource schema (flat tool form, not nested chat-completions).
- **Streaming SSE**: `response.created`, `response.in_progress`, `response.output_item.added`, per-type deltas/`.done`, `response.completed`. **Every event needs `sequence_number`** — `sendResponsesEvent` injects it; the POST handler threads a per-request `seq_num`.
- **Stateful chains**: `ResponseStore` keyed by id. `previous_response_id` replays history; missing → 404. `inputContainsFunctionCallOutput` triggers final-answer mode (tools disabled).
- **Compliance**: `experiments/openresponses` validates strict schema; currently 17/17. `top_level response_format` accepted as alias for `text.format`.
- **Compaction (`POST /v1/responses/compact`)**: pure data, no LLM call. Synthesizes opaque base64 `encrypted_content` over `{"v":1,"msgs":[...]}`. `appendCompactionInputItem` reconstitutes on round-trip. `model` required (422 on missing). Drops tool calls + images.
- **WebSocket transport (`ws[s]://host/v1/responses`)**: same endpoint, opt-in via `Upgrade: websocket`. Each text frame is a `response.create`-shaped JSON; SSE events become single WS text frames via `WsBridge` on `Conn.ws_mode`. **No `[DONE]` on success** (`response.completed`/`.failed`/`.incomplete` is the terminator). Sequence numbers reset per response. `WsLocalCache` holds `store: false` responses for the connection lifetime; failed continuations evict the chain root.

## Anthropic Messages API (`/v1/messages`)

For Claude Code and Anthropic SDK clients with local models.

- `system` (top-level) → internal system message
- Typed content blocks (`text`, `tool_use`, `tool_result`, `thinking`) → internal `Message` structs
- `input_schema` → OpenAI `parameters` for chat templates
- `tool_result` in user messages → internal `role: "tool"`
- `thinking` → `enable_thinking` + `reasoning_budget`; thinking blocks emit fake `signature`
- Stop reasons: `stop`→`end_turn`, `length`→`max_tokens`, `tool_calls`→`tool_use`
- SSE events: `message_start`, `content_block_{start,delta,stop}` (with `text_delta`/`thinking_delta`/`signature_delta`/`input_json_delta`), `message_{delta,stop}` — explicit start/stop lifecycle per indexed block

**Claude Code launcher**: app sets `ANTHROPIC_BASE_URL`, dummy `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_DEFAULT_*_MODEL=mlx-serve`, `CLAUDE_CODE_SUBAGENT_MODEL=mlx-serve`.

## Tool Calling

### Server (Zig)
- **Detection**: when `tools` present, server buffers tokens, checks `<tool_call>`, Hermes XML, Gemma 4 `<|tool_call>`, raw JSON. Gemma 4 double-brace args (`{{...}}`) unwrapped before parse. Thinking tokens buffered separately if enabled.
- **Serialization** (`chat.serializeMessagesJson`): `role: "tool"` passed natively (Gemma 4 templates render `<|turn>tool` directly). Tool call `arguments` serialized as JSON strings, not objects.
- **Streaming**: full args sent in one SSE delta (avoids client accumulation bugs). Thinking buffered until close tag, emitted as `reasoning_content`.
- **Fallback** (`chat.fallbackFormatChat`): ChatML, Llama (`ipython` role), Gemma (`Tool result:` in user turn).
- **KV cache**: `reuseKVCache()` token-by-token prefix compare; auto-invalidated after tool calls and pad-only gens. Sliding-window layers keep full buffer; views slice to last `sw` during decode.

### Client (Swift)
- **Agent loop** (`ChatView.runAgentLoop`): up to 150 iters; tools → parse → exec → feed back → repeat. Synthetic user nudge after tool results for some models.
- **History builder**: filters errors/pad-only/summaries, truncates assistant at 500 chars, budget-aware (walks backward, fits within `context_length - max_tokens - system_prompt`), pins first user + first assistant, auto-compacts tool results when tight.
- **SSE parsing**: accumulates tool-call deltas; emits `.toolCalls` on `finish_reason: "tool_calls"`; fallback if stream drops without `finish_reason`.
- **Storage**: `SerializedToolCall` (id, name, args as JSON string) on `ChatMessage.toolCalls`; Codable-persisted; backwards-compat (optional field).
- **Error recovery**: tool errors include sent args + retry hint → enables self-correction.

## Prompt-based Skills

`.md` files in `~/.mlx-serve/skills/` with YAML frontmatter:
```markdown
---
name: deploy
description: Deploy to production
trigger: deploy, release, ship it
---
Steps...
```
- `trigger`: comma-separated, case-insensitive substring match in user message → body injected into system prompt
- Skill index (name + description) always included
- `SkillManager` re-scans on dir mtime change
- UI: folder icon opens `~/.mlx-serve/` in Finder

## Resumable Downloads

`DownloadManager` streams to `<file>.partial` via `URLSessionDataTask`. Resume sends `Range: bytes=<existing>-`; 206 → continue, 200 → truncate+restart. 3 retries with 2s/4s backoff. Cancel preserves `.partial`. UI shows "Resume" when `hasPartialDownload()`. Already-complete files (size matches HF metadata) are skipped.

## Debugging

### Server logs
- `--log-level debug` for verbose output
- App captures stderr in `ServerManager.serverLog` (64KB rolling buffer); view via log button in menu bar
- Manual: `./zig-out/bin/mlx-serve --model <path> --serve --port 8080 --log-level debug 2>&1`
- Patterns: `jinja error:`, `[cache] reusing N/M tokens`, `[cache] invalidated`, `<- N+M tokens (Xms) [reason]`, `tool_msgs=N`, `[spec-stats] mode=...`, `spec-gate: ngram-score=...`

### Swift logs
- `print()` invisible when launched via `open` — run binary directly or write to file
- Every agent request dumped to `~/.mlx-serve/last-agent-request.json`; replay via curl
- Chat history at `~/.mlx-serve/chat-history.json`

### Reproducing
- Tool calling: curl with `stream: false` first, then `stream: true`
- Jinja offline: `pip3 install jinja2`, render `chat_template.jinja` with dumped request
- KV cache: `pkill -f mlx-serve` between tests — one bad request can poison the cache

## Gotchas

### Hand-rolled JSON and control bytes (silent prompt-format downgrade)
Everything fed to the C++ Jinja engine is hand-serialized JSON (`chat.serializeMessagesJson` → `appendJsonString`), and nlohmann is strict: ONE raw control byte (< 0x20) anywhere in the history makes the whole render fail. `renderChatTemplate` then silently downgrades to `fallbackFormatChat`, whose generic tags are NOT the family's trained format — for Gemma 4 the fallback's `<start_of_turn>`/`<end_of_turn>` aren't even special tokens (its template uses `<|turn>`/`<turn|>`), so the model loses its stop token and degenerates into hallucinating both sides of the conversation. Live failure 2026-06-11: gemma-4-31b via pi went insane on turn 3 because turn 2's tool result captured an interactive npm CLI's ANSI codes (`\x1b[?25l`). Rules:
- Any string that can carry arbitrary bytes (tool results, model output echoed into history) must be escaped by a helper that `\u`-escapes ALL control chars. `appendJsonString` does this now; never add a new field with ad-hoc escaping (compare `server.zig jsonEscape`, `responses.zig jsonEscape`, `json_schema.zig writeJsonString` — all already correct).
- The corpus test "history round-trip serialization survives any byte content" (`src/format_corpus_test.zig`) pins the invariant for every corpus entry plus hostile ANSI/control-byte tool results — new corpus entries are covered automatically.
- A Jinja render failure logs at **warn** (`jinja render failed`), not debug. If a model suddenly emits wrong-family tags (`<start_of_turn>` from a Gemma 4, raw role names like `assistant`), suspect a silent fallback render before suspecting the model or spec-decode.

### KV cache after tool calls
Generated tool-call tokens are in the cache but not in `cached_prompt_ids` → reusing for the next request (with tool results) corrupts attention. Auto-invalidated. Pad-only generations also trigger invalidation.

### Sliding window KV cache
Gemma 4 E4B (512-token window) keeps full buffer — no trimming. Prefill returns all entries (Q/K dim match); decode views return last `sw`. Mask handles attention scope. Matches mlx-lm `RotatingKVCache`.

### Lazy mlx_slice views fed to gather_qmm (silent wrong/zero expert outputs)
A weight produced by `mlx_slice` is a lazy VIEW with parent strides. Feeding such a view (weight, scales, or biases) to `mlx_gather_qmm` SILENTLY computes wrong expert outputs at real-checkpoint scale — observed as exactly-zero MoE branch output on DiffusionGemma's split `experts.gate_up_proj` (the model still "worked", just incoherently: the dense shared-MLP branch carried it). Three traps make this class nasty:
- Reading the slice via data pointers materializes it correctly, so value-equality tests on the split pass while the gather is broken — the assertion must go THROUGH gather_qmm.
- Toy/synthetic geometries stay green: a full-geometry random-weight repro (E=128, M=704, IN=2816, sorted path) showed ZERO diff between view and contiguous. Only the real checkpoint reproduces it. The guard is therefore the LIVE converged-canvas self-consistency test (`DIFFUSION_TEST_MODEL`), verified red-on-revert (49/64 vs 63/64).
- Rule: any weight that goes through `mlx_gather_qmm`/`mlx_quantized_matmul` and was born from a slice must be materialized with `mlx_contiguous` at load (see `splitPackedGateUp`).

### DiffusionGemma numerics: garbage canvases amplify chaos; parity-test on converged canvases
Logit/argmax comparisons against the reference on RANDOM-token canvases are meaningless for correctness: garbage input puts the sigma-MoE router on knife-edge ties, and a single bf16 kernel-order difference flips expert sets, ballooning into multi-unit logit deltas while both outputs are "valid". The meaningful invariant is SELF-CONSISTENCY on a converged canvas (real prompt + the canvas the reference committed → one decoder forward must reproduce it as argmax; reference 63/64 no-sc / 64/64 with-sc). Related: sliding-layer prefill now takes the `"causal"` fast path whenever `total_kv <= sliding_window` (same kernel + reduction order the mlx-lm/mlx-vlm reference picks) — the `"array"`-mask kernel's different reduction order alone degraded diffusion convergence via router-tie flips. Debug aids: `MLX_SERVE_DIFFUSION_TRACE=1` prints per-step mean entropy + acceptance counts and decoder layer-0/self-conditioning tensor heads.

### Config fields omitted by nested `text_config` fall back to 12b-shaped struct defaults
`ModelConfig`'s field defaults (`src/model.zig`) are the Gemma-3/4 **12b/27b** shape — `num_attention_heads=16`, `num_key_value_heads=8`, `hidden_size=3840`, 48 layers. A multimodal checkpoint's nested `text_config` is read at `parseConfigFromJson` (lines 458-460), but **omitted** fields silently keep those struct defaults instead of the model family's HF defaults. `gemma-3-4b-it-4bit`'s `text_config` omits `num_attention_heads`/`num_key_value_heads`/`head_dim` and leans on HF Gemma3TextConfig defaults (8/4/256) — so it loaded with 16/8, and the Q projection (`8*256=2048`) crashed reshaping into `(1,1,16,256)` at warmup (issue #43). Symptom signature: **`MLX error: [reshape] Cannot reshape array of size N into shape (1,1,H,D)` during warmup, where N = (real heads)×head_dim and H = the wrong (default) head count.** Rule: when an arm relies on HF per-arch defaults for fields a minimal `text_config` may omit, fill them explicitly in that arm gated on `cfg_obj.get(field) == null` (see the gemma3 arm). Guarded by the two `ModelConfig … gemma3 … head counts` tests in `model.zig` (omitted → HF defaults; explicit → preserved).

### Gemma 4 tool calling
Templates render `role: "tool"` natively as `<|turn>tool` — no transformation. Don't add `tool_responses` field (causes duplicate content). Args serialized as JSON strings.

### Streaming with tools + thinking
Server buffers tokens to detect tool patterns. With thinking enabled, `<|channel>thought` is buffered (not flushed) until closing `<channel|>`. After generation, thinking is split into `reasoning_content`; channel tags stripped from visible content.

### Quantization modes (nvfp4 / mxfp4 / mxfp8)
`config.json`'s `quantization.mode` lands on `ModelConfig.quant_mode` (default `affine`; unknown → `error.UnsupportedQuantMode` at parse). Non-affine modes store NO `.biases` tensors and use uint8 fp8-encoded scales — but mixed QAT checkpoints (gemma-4 `*-qat-nvfp4`) override some layers to affine 8-bit/gs64 WITH bf16 scales + biases, so resolution is PER WEIGHT (`transformer.computeQuantParams`, cached by scales ctx): uint8 scales → config's fp8 mode; float scales → affine, with (bits, group_size) solved from the activation inner dim (`w_cols*32/s_cols` alone only pins bits×gs). Two loader rules: `.biases` fetches are mandatory under affine, OPTIONAL under non-affine modes (`getLayerBias`) — never skipped, or the affine-override matmuls break; and never gate fp8-vs-affine on `biases.ctx == null` alone (that's the legacy mxfp8-inside-affine heuristic in `qmatmulBits`, kept for NVIDIA Nemotron mxfp8 layers).

### SSM/GatedDeltaNet state init
`conv1dWithCache` sets `ssm.initialized = true` after conv update but BEFORE SSM recurrence state exists. Init code must check `ssm.ssm_state.ctx == null`, NOT `!ssm.initialized`. Used by both `mamba2Mixer` and `gatedDeltaNet`.

### Parameter-free RMS norm
mlx-c crashes on null/empty weight for `mlx_fast_rms_norm`. Pass `ones([dim], bfloat16)` for parameter-free norm. Affects GatedDeltaNet Q/K norm and Mamba2 group norm.

### Nemotron-H time_step_limit
Python defaults to `(0.0, inf)` (no dt clipping). `time_step_min`/`time_step_max` in config.json are NOT used by Python for SSM clipping. Only `time_step_limit` JSON array overrides.

### Speculative decoding (PLD + drafter) — overview

Two paths share a verify invariant: `cache.step = prompt_len + tokens_emitted`, t1 NOT in cache on entry, no pending state. Verify input is `[t1, draft[0..m-1]]` length `1+m`; full accept samples `new_t1` from `verify_logits[m]` (bonus prediction); partial accept rolls back via `KVCache.snapshot/restore` + `ssmSnapshot/Restore` and re-forwards `[t1, draft[0..accepted-1]]`. `accepted=0` still re-forwards `[t1]`. Pending correction sampled from *original* `verify_logits[accepted]` (NOT re-forward); index is `accepted` not `accepted-1` — off-by-one silently corrupts output, guarded by `tests/test_pld_equivalence.sh`.

**PLD** (`src/pld_index.zig`, `Generator.nextPld`): model-agnostic n-gram match in `prompt + generated`. CLI `--pld --pld-draft-len 5 --pld-key-len 3`; per-request `enable_pld`. `Generator.initWithOptions` clones prompt to `prompt_ids_owned` (caller-supplied freed before `nextPld`). Stochastic verify: draft as one-hot; `accept_prob = min(1, target_p[draft[i]])`, residual `max(target_p − one_hot, 0)` renormalized — preserves marginal per Leviathan. One-hot built via `pldOneHotRow` (no scatter).

**Drafter** (`src/drafter.zig`, `Generator.nextDrafter`): Gemma 4 only. 4-layer, hidden 256, no K/V projections — cross-attends into target's K/V via layer-type mapping (drafter sliding → target last sliding; drafter full → target last full). Loaded via `--drafter <dir>`. `block_size` auto-detected per target (E2B=2, E4B=4, 26B-A4B=4, 31B=8 — matches vLLM PR #41745) via `recommendedBlockSize`; override with `--draft-block-size`. Input: `concat([target.embed(prev) * sqrt(target.hidden), h_prev], -1)` → drafter hidden 256. Autoregressive within round (`block_size − 1` drafts), constant RoPE offset. Sparse `MaskedEmbedding` LM head (~2048 centroids, top-32 → ~4096 token logits of 262144). Linear weights pre-transposed at load.

**Validation**: `error.UnsupportedDrafterArch` (model_type mismatch), `error.DrafterTargetMismatch` (hidden_size or layer_types incompatible).

**`forwardCaptureHidden`**: `forwardStandard` and `forwardMoe` honor `capture_hidden`, slicing post-final-norm hidden at LAST position. Drafter seeds first `h_prev`; PLD uses during partial-accept rollback. Other forward paths (BERT, hybrid) leave it empty — drafter/PLD not wired there.

**Coverage**: PLD, drafter, and MTP dispatch on ALL FOUR HTTP surfaces — `/v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/v1/completions` — in both streaming (`pickStreamMode`) and non-streaming (`nonStreamingViaScheduler` `use_pld`/`use_drafter`) modes. When adding an endpoint or dispatch path, wire BOTH flags through both modes and extend the engagement-count check in `tests/test_drafter_tools.sh` — two non-streaming call sites shipped with a hardcoded `use_drafter=false` for a month because output-equality tests can't see a silent fallback to regular decode.

**MTP (native multi-token prediction)** (`src/mtp.zig`, `Generator.nextMtp`): Qwen 3.5/3.6 checkpoints with a trained one-layer MTP sidecar (`mtp/weights.safetensors`, ~15 tensors, e.g. ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve — buildable from any Qwen 3.6 checkpoint via tests/build_mtp_sidecar.py) auto-load it and speculate with the model's own head. Forward: `fc(concat([rmsnorm(embed(tok)), rmsnorm(trunk_hidden_post_final_norm)]))` → one qwen3_5 gated full-attention layer with its OWN single-layer dense KV cache (partial RoPE at cache-relative offset) → `mtp.norm` → trunk lm_head. The head keeps a COMMITTED-HISTORY cache: entry j pairs (trunk hidden at position p, token at p+1), built chunk-wise during prefill (`ForwardCtx.capture_hidden_all`) and rebuilt each round — drafts append temporary entries from MTP-predicted hiddens, then commit restores the round-boundary snapshot and re-appends from TRUE verify hiddens (one batched MTP forward), so history never drifts. Verify invariant identical to PLD/drafter. Quant: MTP linears ride their own (bits, group_size), INFERRED from tensor geometry at load (`inferBits` pins bits exactly via packed cols vs the fc-derived hidden; sidecars are often g32/8-bit over a g64/4-bit trunk); fc + norms bf16. GOTCHA: the original Qwen repo stores RMS-norm weights delta-encoded (layer computes 1+w) — a sidecar built without the +1 fold-in is structurally broken (0% acceptance). Priority MTP > drafter > PLD; defaults ON when loaded (`!isMoe()`), per-request `enable_mtp`; NOT subject to the prompt n-gram spec-gate (the trained head holds ~73% per-draft on fully novel content). Default depth 1 — measured M4 Max Qwen3.6-27B 4-bit: story 32.4 tok/s (1.11× AR, 73% accept), code 41.6 tok/s (1.43× AR, 93% accept), vs the reference MTP runtime's best 32.3/38.1 on the same hardware/prompts. Depth >1 (`--mtp-depth`) engages a windowed adaptive controller (`mtpDepthDecision`): demote fast on a 5-round sample, promote only on ≥8 hot rounds with a 32-round post-demotion cooldown, disable (sticky) only on a FULL 16-round window below 50% at depth 1 — a 5-round disable was observed killing requests that ran 1.11× overall. Depth 2 adds an extra full-vocab lm_head projection per draft and only ~2% on code; it regresses creative via demotion churn — hence 1.
The parity test feeds the reference implementation's trunk hiddens through our head and requires exact draft-token agreement, isolating head math from INT4 trunk-numerics differences (live drafts agree 25/26 with the reference; rejections are near-tie trunk argmax flips, not head bugs).

**Auto-disable**: `logprobs > 0` and grammar-constrained sampling disable both. Tools disable NEITHER — agent traffic (tool results echoed into edits) is spec-decode's best workload; equivalence with tools is pinned by `tests/test_pld_tools.sh` and `tests/test_drafter_tools.sh` (~2.1× decode on file-edit tool calls, Gemma 4 E4B). PLD works on hybrid SSM (LFM2.5, Nemotron-H) — see snapshot null-state guard below; the drafter does not (verify forward hits the SSM-state issue). Drafter streams since spec-decode v3 (`pickStreamMode` routes streaming requests to `.drafter`). Drafter > PLD > regular priority when both enabled.

**Adaptive prompt-time gate** (`spec_gate_threshold = 0.01` in `server.zig`): n-gram repetition score on tokenized prompt (`pld_index.ngramRepeatScore`, 3-grams). If `score < threshold` AND user didn't set `enable_pld:true`/`enable_drafter:true`, the flag is silently disabled. Runs in all three request paths; chat-completions logs `spec-gate: ngram-score=X.XXX` once per request. v4 corpus validation: 9/9 correct decisions; threshold 0.01 cleanly separates "any 3-gram repeats" from pure-novel prompts.

**Runtime acceptance gate** (`RUNTIME_GATE_MIN_PER_DRAFT_RATE = 0.50`, warmup 5): when per-draft acceptance falls below 50% mid-decode, `Generator.spec_disabled_runtime` flips on (sticky). Subsequent calls short-circuit to `Generator.next`, which has a transition shim: when no pending logits/token, sync `forward([next_token_id])` to seed pending_logits. Pre-v26.5.6 the gate compared per-round against 0.30 → almost never fired; 0.50 cleanly cuts creative-content tail (22-47%) while leaving heavy-echo (84-97%) untouched. Does NOT save MoE+drafter regressions where per-draft is high but verify cost dominates — handled by MoE default-off in `serve()`.

**Default-on policy**: PLD is ON by default at the CLI (`main.zig` `enable_pld = true`; `--no-pld` disables). The drafter is opt-in via `--drafter <dir>`. While a drafter is loaded:
- Dense Gemma 4 (E2B/E4B/31B) drafter: `enable_drafter` defaults TRUE per-request; gates handle creative content
- MoE Gemma 4 (26B-A4B) drafter: `enable_drafter` defaults FALSE — verify forward MoE expert-routing penalty makes drafter regress at batch=1 even at 97.8% per-draft (every block_size tested). PLD remains default-on (1.43× echo). Per-request override still works.

### PLD/drafter long-greedy byte-divergence at INT4
AR (`next`) forwards `[1,1,d]` qmv; verify forwards `[1,K+1,d]` qmm. INT4 float reductions in slightly different orders → near-tie argmax can flip → divergence cascades. First ~30–80 generated tokens at temp=0 are byte-identical (equivalence tests live here); beyond that, paths may diverge char-by-char while both being mathematically valid greedy outputs. At temp ≥ 0.01 the Leviathan sampler preserves the target distribution → exact past 30 tokens. **For byte-stable long-greedy at temp=0 on INT4: `--no-pld`, no `--drafter`.** For chat/agent (temp>0) spec-decode is exact and free.

The same float-reduction issue compounds when **KV is also INT4** — see "KV cache quantization" below.

### KV cache quantization (`--kv-quant {off, 4, 8, turbo2, turbo4}`)
Group-wise affine quantization of K/V via `mlx_quantize`/`mlx_dequantize` (no new kernels). Storage swaps dense `[B,H,T,D]` bf16 buffers for a triple `(q, scales, biases)` where `q` is packed uint32 and `scales`/`biases` are per-group bf16; SDPA always reads dense data via `KVCache.denseView`, which dequantizes on the fly in quant mode. `--kv-quant` sets the **process default**; individual requests can override via the `kv_quant` body field on `/v1/chat/completions`, `/v1/messages`, `/v1/responses` (`"off"`, `4`, `8`, `"turbo2"`, `"turbo4"`). Memory: ~4× smaller at 4-bit (4.5 bits/elem including scale+bias overhead at group=64), ~2× at 8-bit. TurboQuant adds a Hadamard rotation before affine quant; `turbo2` halves bits-per-element again at the cost of an extra `[head_dim,head_dim]` matmul per K/V per token. Implemented in `src/kv_quant.zig` + `src/transformer.zig` (KVCache).

- **Equivalence thresholds** (`tests/test_kv_quant_equivalence.sh`, default 30/30; raise via env vars for stricter testing):
  - Gemma 4 E4B 4-bit weights: 30/30 passes; 8-bit KV stays identical past 60 in practice.
  - Qwen 3.5/3.6 MoE 4-bit weights (GatedDeltaNet + MoE): 4-bit KV passes 30 tokens. 8-bit KV diverges around token 41 from MoE+GDN float-reduction noise — same class as the INT4-weight long-greedy tail.
  - LFM2.5 8-bit weights (hybrid SSM): 4-bit KV diverges around token 12 — recommend `--kv-quant 8` for byte-stable long-greedy on this family.
  - Override per-arch via `KV_QUANT_FIRST_N_4BIT` / `KV_QUANT_FIRST_N_8BIT` env vars.
- **Compounding with INT4 weights**: Both the existing weight-quant divergence (PLD/drafter note above) and KV-quant divergence stack. For byte-stable long-greedy at temp=0 on INT4-weight models: prefer `--kv-quant 8` if you need a quant; `--kv-quant off` if you don't.
- **Drafter**: target's KV may be quantized; drafter cross-attends through `cache.denseView` so it never sees the quantized representation directly. Drafter's own cache stays dense. No special handling needed.
- **Snapshot / prefix cache**: snapshot/restore copy 6 array handles per entry instead of 2 (4 extra for scale/bias); hot prefix cache works unchanged because it operates on `KVCacheSnapshot` opaquely. Each `HotEntry` records its scheme; `findBestMatch` filters by `(prompt_ids, has_tools, scheme)` so per-request overrides never produce a cross-scheme hit.
- **TurboQuant (`turbo2`, `turbo4`)**: same affine-write/read path with a per-layer Hadamard rotation applied before quantization and undone after dequantization. `TurboState` builds `2 × num_layers` symmetric `[head_dim, head_dim]` bf16 matrices via Sylvester construction with per-layer column-sign flips (deterministic, no RNG seed). `head_dim` MUST be a power of two — caller passes via `KVCache.initWithConfigAndHeadDim`. State lives on `KVCache.quant_state` and refcount-shares through `snapshot`/`restore`. The rotation matters when inputs have outliers that would inflate per-group ranges in straight affine; on smooth data it can be slightly *worse* than straight affine because the rotation spreads tight local ranges into a wider global range.
- **1-bit TurboQuant**: not yet shipped. `mlx_quantize`/`mlx_dequantize` only support bits ∈ {2,4,8} natively, so 1-bit requires a custom pack/unpack. Land alongside the future fused-kernel work.
- **Extending the scheme** (e.g. fused quant-SDPA Metal kernel): the contract between cache and attention is `KVCache.denseView`. To add a new scheme:
  1. Add an enum variant to `kv_quant.Scheme`.
  2. (Optional) Add per-cache state (e.g. `quant_state: ?TurboState` for rotation matrices).
  3. Add `quantizeX` / `dequantizeX` functions in `src/kv_quant.zig`.
  4. Extend the `switch (config.scheme)` arms in `KVCache.update` and `KVCache.denseView`.
  SDPA call sites don't change. See top-of-file comment in `src/kv_quant.zig` for the worked TurboQuant example (now shipped).

### Hot prefix cache memory budget (`--prefix-cache-mem`)
Wave 1.B — the hot prefix cache used to cap on entry count alone; with 4 KB-ctx entries on Gemma 4 E4B that's an 8 GB worst case. `--prefix-cache-mem N{KB,MB,GB}` (default 2 GB) caps resident KV bytes; `commit` evicts LRU entries until `current_kv_bytes + new_bytes <= budget`. `0`/`off` disables the byte cap (count cap still applies). Each `HotEntry` records its bytes at commit time (sum of `mlx_array_size × mlx_array_itemsize` across keys/values plus the scales/biases triples in quant mode). Log line: `[hot-cache] resident=X.XX / Y.YY MB (E entries)` on every commit / eviction.

### PLD on hybrid SSM (snapshot null-state guard)
`SSMCacheEntry` has two slots (`conv_state`, `ssm_state`) populated by different layer types: LFM2's `gated_conv` writes only `conv_state` (sets `initialized=true`) and never touches `ssm_state`. `mlx_array_set` with null source aborts via mlx-c's default handler (`exit(-1)`). `ssmSnapshot`/`ssmRestore` and `PrefillCache` save/restore must check each field's `.ctx != null` independently — `initialized` alone insufficient. This was the previous "off on hybrid SSM" auto-disable; lifted once per-field guard landed.

### mlx-c API changes
mlx-c 0.6.0 added a `global_scale` param (may be null) to `mlx_dequantize` between `mode` and `dtype`. FFI in `mlx.zig` must match installed header. When upgrading, diff `/opt/homebrew/include/mlx/c/ops.h` against `extern "c"` decls.

### Two binaries in the app bundle
`.app` contains `MLXCore` (Swift UI) AND `mlx-serve` (Zig server). Both must be updated together. Forgetting one after rebuild is a common "still doesn't work" cause.

### WebSearch + Browse
`webSearch` navigates DuckDuckGo HTML, extracts results via JS. `browse.readText` navigates first then extracts — ensures correct page (not previous).

### WKWebView main thread
`BrowserManager` is `@MainActor`. All WKWebView ops (navigate, readText, evaluateJS) on main thread. Created eagerly at app launch so tools work without Browser window open.

### Swift JSONSerialization quirks
- `[String: Any]` non-deterministic key order
- `""` stays `""` in JSON (not `null`); server treats both as empty
- `Double` like `0.7` → `0.69999999999999996` — fine
- `arguments` in tool_calls must be a JSON String (e.g., `"{\"command\":\"ls\"}"`), not nested dict; server checks `if (v == .string)`

### Embedded engines (`lib/<engine>/`)

DSV4-Flash is served by the embedded **ds4** engine (antirez/ds4, since renamed "DwarfStar" upstream), pinned at `lib/ds4/@477c0e8`. `build.zig` compiles `ds4.c` + `ds4_distributed.c` + `ds4_metal.m` (the library path; `ds4_kvstore/web/help/agent.c` are CLI/server-only and not embedded). The Zig MLX path no longer carries a DSV4 forward — `model_type=deepseek_v4` on a `.safetensors` checkpoint returns `error.UnsupportedDsv4MlxFormat`; users load the GGUF artifact through ds4 instead.

- C library API: `lib/ds4/ds4.h` (~200 lines, engine + session model).
- FFI: `src/ds4_ffi.zig` (mechanical mirror of the header).
- Bridge: `src/arch/ds4.zig` (`Ds4Engine` + `Ds4Session` — owns Metal-kernel extraction, tokenizer lookups, and the speculative-decode API).
- Build: `build.zig` compiles `lib/ds4/ds4.c` + `lib/ds4/ds4_metal.m` and links Foundation/Metal. Metal kernel sources are embedded via `@embedFile` (`lib/ds4_metal_sources.zig`); at first launch we stage them under `~/.mlx-serve/ds4-metal/<binary-hash>/` and point ds4 at them via its `DS4_METAL_<NAME>_SOURCE` env-var overrides — zero patches to the submodule.
- Convention: each new model-specific engine lives at `lib/<engine>/`, exposes a single-header C API, and gets a thin Zig wrapper at `src/arch/<engine>.zig`. See `CONTRIBUTING.md` for the embedding checklist.

Generic GGUF models (everything except DSV4-Flash) are served by the embedded **llama.cpp** engine via `libllama`.

- Staging: `scripts/fetch-llama.sh` downloads the pinned llama.cpp XCFramework (`LLAMA_TAG`, currently `b9496`), thins the macOS slice to a single self-contained arm64 `lib/llama/lib/libllama.dylib` (llama + ggml + ggml-metal merged, Metal shaders embedded), rewrites its install-name to `@rpath/libllama.dylib`, and copies headers to `lib/llama/include`. **Not vendored** — `lib/llama/` is git-ignored and re-fetched by CI / `app/build.sh`. Bump `LLAMA_TAG` to upgrade.
- C shim: `lib/llama_shim/llama_shim.{h,c}` — a small, stable C API over the ABI-fragile `llama.h` structs (compiled by clang against the real header, so the ABI is always correct). Mirrors the ds4.h discipline.
- FFI: `src/llama_ffi.zig` (mechanical mirror of the shim header).
- Bridge: `src/arch/llama.zig` (`LlamaEngine` + `LlamaSession` — model load, tokenize/detokenize, chat-template string, prefill `sync`, `eval`, `argmax`/`sample`). Tests gate on `LLAMA_TEST_MODEL`.
- Build: `build.zig` `addLlamaLib` adds `lib/llama/include`, links `libllama` (`use_pkg_config = .no` so a Homebrew `llama.cpp` install can't hijack the link), compiles the shim, and adds a dev rpath. Bundling: `release.yml` / `app/build.sh` copy `libllama.dylib` into the app `Frameworks/` (and the CLI tarball `lib/`), rewrite `@rpath/libllama.dylib` → `@executable_path/...`, and sign it through the existing dylib loop — no new executable, so notarization is unchanged.
- Chat templating reuses mlx-serve's Jinja engine: the GGUF's embedded template is adopted into the stub `ChatConfig.chat_template` at load (`Scheduler.doLoadLlamaOnInferenceThread`), then `chat.encodeChatViaLlama` renders via `renderChatTemplate` (which supplies the tool-synthesis fallback) and tokenizes through libllama (`add_special=false`, the template owns BOS).
- Scope (v1, like ds4): serial (`max_concurrent=1`), no PLD/drafter/kv-quant/hot-switch (those are MLX-specific). Tested by `tests/test_llama_gguf.sh` (gated on `LLAMA_GGUF_MODEL`).
