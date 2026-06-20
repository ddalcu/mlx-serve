# TODO

Ordered by **expected speed gain per unit of work**, highest first. Cross-arch parity audit (2026-05-16) put mlx-serve at or above mlx-lm on every remaining benchmarked architecture except Qwen 3.6 27B hybrid (-4.4% after `mlx_split`). DSV4 is now served by the embedded **ds4** engine (antirez/ds4 @ `477c0e8`) under `lib/ds4/` — the legacy Zig forward and its tuning items below are retired.

| Model                              | mlx-serve | mlx-lm | Δ      |
|---|---|---|---|
| Gemma 4 E4B 4-bit                  | 112.48    | 114.02 | -1.4% (noise) |
| Gemma 4 E2B 4-bit                  | 182.68    | 184.70 | -1.1% (noise) |
| Qwen 3.6 35B-A3B MoE 4-bit         | 122.95    | 120.39 | **+2.1%** |
| Qwen 3.6 27B hybrid 4-bit          | 27.74     | 29.01  | -4.4%    |

DSV4-Flash decode is whatever ds4 reports on the host; track upstream rather than re-benching here.

Multi-slot batching shipped: Gemma 4 E4B `--max-concurrent 2` → **1.50× throughput** at 1.33× per-request latency.

---

## High-leverage performance (next big wins)

1. ~~**`mlx_compile(forward)` wiring.**~~ **RETIRED 2026-06-10 — measured no-op.** Whole-forward compile (env-gated `MLX_SERVE_COMPILE_FORWARD=1`, prefill-chunk path) showed zero prefill gain on E4B; decode can't use it (KV growth → per-step retrace). mlx-lm doesn't compile its full forward either; our activation/MoE-routing/GDN-gate closures cover its actual `@mx.compile` usage.

2. **Qwen 3.6 27B hybrid gap, now −1.2% (was −4.4%, then −2.4%).** 2026-06-10: GDN decay gate fused into one compiled closure (`compiled_gdn_gate`, mirrors mlx-lm `compute_g`) + cached parameter-free rms_norm ones-weight and Q/K scale scalars (were rebuilt every layer/token) → 28.74→29.12 tok/s. Remaining −1.2% suspect: conv1d-step composition / per-layer dispatch overhead in the GDN block. Tap-diff to localize. **~half day.**

3. ~~**Eval-boundary audit.**~~ **CLOSED 2026-06-10 — no leak signal.** Decode loop already `mlx_async_eval`-pipelined; dense models at/above mlx-lm and at the bandwidth ceiling (31B = 79% of theoretical peak). The "0% if not" case.

4. **Fused quant-attention Metal kernel** (Plan 02 Path B). `--kv-quant 4/8/turbo*` dequantizes triples on every SDPA read; ~5% decode TPS on Gemma 4 E4B short context. Pattern lives in `src/kv_quant.zig` top-of-file comment. **Multi-day.**

4b. **Drafter tuning (2026-06-10 baseline findings).** (a) 12B drafter is below raw on EVERY cell (echo 35.9 vs 39.0 raw) — investigate block size (`recommendedBlockSize` has no 12B entry?) and verify cost; consider default-off like MoE. (b) 31B drafter loses the code cell (23.4 vs 24.3; block_size=8 suspect — try 4). (c) **PLD-first hybrid**: drafter echo is far below PLD echo everywhere (E2B: 224 vs 352); in `nextDrafter`, try the n-gram lookup first and use its draft when a match exists, drafter otherwise — best of both on agent traffic. (d) Drafter disabled mode is sticky (no mid-request re-enable) — needs an h_prev re-seed via `forwardCaptureHidden` on top of the pipeline drain that PLD re-enable uses. **~1 day total.**

4c. **PLD cold-path pipelining (remove the yield gate's reason to exist).** The cold no-match `nextPld` step does a synchronous forward+eval (loses the async pipeline, −14% measured). Restructure PLD around the pipelined state: keep `pending_logits`, lookup on the RESOLVED token, and on match verify `[draft]` against `pending_logits` + verify-forward (t1 already in cache — partial-accept rollback point shifts by one). Would make PLD-on-novel-content free from token 1 and the 32-step warmup tax disappear. Equivalence pins exist. **~1 day, invariant-sensitive.**

## Medium / niche performance

5. **1-bit TurboQuant.** Custom pack/unpack since mlx-c bits ∈ {2,4,8}. Niche — only matters for users hitting KV memory budgets. **Multi-day.**

## Infrastructure / packaging

6. **GGUF load path wiring.** The ds4 bridge + Metal-kernel extraction are in place at `src/arch/ds4.zig`; main.zig still needs the `.gguf` early-branch that constructs a `Ds4Engine` and routes generation through `Ds4Session.eval/sample` instead of the MLX `Generator`. Today, pointing `--model` at a GGUF file falls through to the safetensors path and fails. **~1 day.**

7. **Cold-tier (SSD) prefix cache** (Plan 03 Phase 2). Persists hot tier under `~/.mlx-serve/kv-cache/<model_id>/`. Affects cold-prompt latency, not steady-state. **Multi-day.**

8. **Mask-build cache for cold prefill** (Plan 04 Phase 2). 5–80ms per cold first request. Modest.

9. **Page-aligned slicing in live cache** (Plan 03 Phase 4). Only matters once cold tier lands.

10. **Vendored deps (mlx-c, libwebp)** as git submodules under `lib/` matching `libjinja.a` / `libds4`. Removes Homebrew dependency from CI / end-user setup.

11. **Multi-slot 24h soak test** with `--max-concurrent 4`. The unified bench's `--concurrent N` mode exercises the same path; long-duration soak validation still pending.

12. **Recurrent-state snapshots for hybrid-GGUF prefix reuse.** Today `LlamaSession.sync` trims attention KV via `seq_rm` but the recurrent state (GatedDeltaNet on Qwen3.5 / Qwen3-Next, Mamba on Nemotron-H GGUFs, etc.) can't be rolled back the same way — warm-decode after a trim is *not* byte-identical to cold-decode, and the `prefix reuse is byte-identical to cold decode` Zig test fails on hybrid `LLAMA_TEST_MODEL`s. Fix: thread `llama_state_seq_{get,set}_data` through the shim, hold one snapshot buffer per LRU entry on `LlamaSession`, save at the current resident-end after each successful sync, restore at the next sync's common-prefix point (fall through to `reset()` when `common < snapshot_pos`). Detect once via `llama_model_is_recurrent`. Preserves the multi-turn prefix-reuse speedup on hybrids that phase 5 #1 set up. Mid-term mitigation if not done: force cold-prefill on `llama_model_is_recurrent == true` so we don't ship wrong-output prefix reuse. **~1 day incl. a Mamba/GDN round-trip test.**

13. **Separate reasoning tokens from content — LANDED 2026-06-10 (except max_tokens semantics, kept as-is by design).** The real leak was the *template-opened* think block: Qwen 3.5/3.6 render `…assistant\n<think>\n` into the generation prompt, so a length-truncated thought contains no think tags and the non-streaming paths dumped it into `content`. Fixed via `chat.promptTailOpensThink` (decodes the prompt tail) + an `opened_by_template` arg on `splitThinkBlock`, wired through chat-completions, /v1/messages, and /v1/responses (stream + non-stream). `usage.completion_tokens_details.reasoning_tokens` now populated on chat-completions; /v1/responses `output_tokens_details.reasoning_tokens` now real (was hardcoded 0). Pinned by `tests/test_thinking_split.sh` (14 checks). **Deliberately NOT changed**: `max_tokens` still caps total output including reasoning — that matches OpenAI's own `max_output_tokens` semantics; LM Studio's content-only interpretation is the outlier. Honest cross-engine decode comparisons should use the stream-measured rate (`tests/_decode_stream.py` counts reasoning deltas) or subtract `reasoning_tokens`, both of which now work.

14. **Qwen3-VL vision — faithful M-RoPE on Anthropic + /v1/responses.** Only `/v1/chat/completions` (stream + non-stream) threads the real M-RoPE position table today; `/v1/messages` and `/v1/responses` pass `MropeData{}` → scalar-RoPE fallback (images still decode correctly; only spatial grounding on large/counting tasks is softer). Those handlers (`handleAnthropicMessages`, `handleResponses`) compute `prompt_ids` + have the image grids but submit through sub-scopes where `computeQwenMrope`'s result isn't in scope. Thread it the same way the chat path does (compute after `insertMultimodalTokens`, hand off as `sub_mrope`, pass to the submit). Matters for Claude Code (Anthropic API) image use. **~half day.** Also: 35B (`mlx-community/Qwen3.6-35B-A3B-4bit`) untested — same arch as the validated 27B (head_dim 72), low risk; run `tests/test_qwen_vision.sh <35B>` to confirm. Multi-image per request also uses one `vision_start` block (single-image is the validated path).

## Deviations kept (no work)

- **`Conn.writeAllNoFlush`** — kept (ws.zig uses the explicit-flush pattern). Removing requires a deeper `writeAll` semantics refactor.

---

## Gotchas worth remembering

- **NEVER bare `zig build`.** Always `-Doptimize=ReleaseFast`. Debug binary is 2–4× slower; silently makes every benchmark look like a regression. ReleaseFast ~3.6 MB; Debug ~7–8 MB.
- **GPU OOM risk.** Never run `mlx-serve` and `dump_python_taps.py` concurrently — both load the 90 GB checkpoint. `pkill -9 -f mlx-serve` and `pkill -9 -f dump_python` between them.
- **`pkill -9 -f mlx-serve`** before any new tap-diff run, otherwise stale cache poisons the new dump.
- **Two binaries in the `.app` bundle.** `MLXCore` (Swift) AND `mlx-serve` (Zig). Both must be updated together after a rebuild.

---

## Recent landings (compact)

- **Qwen3.5/3.6 (Qwen3-VL) image input.** 2026-06-20: `qwen3_5`/`qwen3_5_moe` checkpoints now read images. New `src/qwen_vision.zig` (Qwen3-VL ViT, mirrors mlx-vlm `qwen3_vl/vision.py`; dense bf16 even on 4-bit; encoder parity mean_abs 0.004 vs reference via `tests/test_qwen_vision_parity.sh` + `build_qwen_vision_fixture.py`) + `src/mrope.zig` (faithful interleaved M-RoPE, `get_rope_index`). Preprocessing = smart_resize + `(x/255−0.5)/0.5` + merge-order patchify in `server.zig decodeImageToPixels`; `<|vision_start|>…<|vision_end|>` wrap in `insertMultimodalTokens`. M-RoPE wired into `gatedFullAttnWith` (prefill manual cos/sin, decode scalar at `offset+rope_delta`), threaded server→slot→`ForwardCtx` as `MropeData`. MTP/drafter auto-disabled for image requests; hybrid arch already bypasses the hot prefix cache. App sends raw JPEG for Qwen (`MultimodalContent.serverPreprocess`; `x-mlx-pixels` is Gemma-square only). Validated live on 0.8B + 27B (`ddalcu`, head_dim 72, +MTP): accurate descriptions, OCR, top-to-bottom spatial ordering. Tests: `tests/test_qwen_vision.sh` (OpenAI + Anthropic), hermetic mrope/qwen/config, Swift `MultimodalContentTests`.

- **DSV4 moves to embedded ds4 engine.** Legacy Zig DSV4 forward (~7K LOC) retired; `lib/ds4/` submodule pinned at `477c0e8` carries antirez's hand-tuned Metal kernels. Bridge at `src/arch/ds4.zig` extracts the 19 Metal kernel sources via `@embedFile` at first launch (no patches to upstream). MLX-format DSV4 returns `error.UnsupportedDsv4MlxFormat`; the GGUF artifact is the supported path. `--mtp` / `--mtp-block-size` CLI flags removed.
