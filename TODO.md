# TODO

Ordered by **expected speed gain per unit of work**, highest first. Cross-arch parity audit (2026-05-16) put mlx-serve at or above mlx-lm on every remaining benchmarked architecture except Qwen 3.6 27B hybrid (-4.4% after `mlx_split`). DSV4 is now served by the embedded **ds4** engine (antirez/ds4 @ `613e9b2`) under `lib/ds4/` — the legacy Zig forward and its tuning items below are retired.

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

1. **`mlx_compile(forward)` wiring for dense archs.** `Transformer.compileForward` is wired but dead — `compiled_forward` never invoked at runtime. mlx-lm uses `@mx.compile` extensively. Expected 5–15% on Gemma 4, Llama, Mistral, and Qwen 3 dense — wide blast radius. **~1 day per arch.**

2. **Qwen 3.6 27B hybrid -4.4% gap.** `mlx_split` closed 1.4% (5.7 → 4.4%); remaining is in op-level dispatch overhead or a missing `@mx.compile` in the full-attention layers. Tap-diff vs mlx-lm to localize, then fix. **~1 day.**

3. **Eval-boundary audit / drop redundant materializations.** mlx-lm calls `mx.eval` only at chunk boundaries; we may eval more often. Profile under `MLX_TRACE`; look for `mlx_array_eval` calls outside the resolve-pending-token site. **5–15% if leaks exist, 0% if not. ~4 hours.**

4. **Fused quant-attention Metal kernel** (Plan 02 Path B). `--kv-quant 4/8/turbo*` dequantizes triples on every SDPA read; ~5% decode TPS on Gemma 4 E4B short context. Pattern lives in `src/kv_quant.zig` top-of-file comment. **Multi-day.**

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

13. **Separate reasoning tokens from content in mlx-serve OpenAI API.** With `enable_thinking:true`, Qwen 3.6 generates `<think>…</think>` blocks that mlx-serve currently emits inside `choices[].message.content` and counts toward `completion_tokens`. That makes `max_tokens` budget include thinking, while LM Studio (which reports reasoning separately in `completion_tokens_details.reasoning_tokens` and treats `max_tokens` as content-only) decodes much more in the same wall-clock window — turning honest decode-rate comparisons into apples-to-oranges. Fix: parse `<think>…</think>` server-side, emit reasoning text into `choices[].message.reasoning_content`, populate `usage.completion_tokens_details.reasoning_tokens`, and make `max_tokens` cap visible content only (with a separate `reasoning_budget` for the thinking budget, matching what the Anthropic Messages API already does in mlx-serve). Until this lands, `tests/bench.sh` keeps thinking suppressed on Qwen via the stacked workaround in `build_body_lms`.

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

- **DSV4 moves to embedded ds4 engine.** Legacy Zig DSV4 forward (~7K LOC) retired; `lib/ds4/` submodule pinned at `613e9b2` carries antirez's hand-tuned Metal kernels. Bridge at `src/arch/ds4.zig` extracts the 19 Metal kernel sources via `@embedFile` at first launch (no patches to upstream). MLX-format DSV4 returns `error.UnsupportedDsv4MlxFormat`; the GGUF artifact is the supported path. `--mtp` / `--mtp-block-size` CLI flags removed.
