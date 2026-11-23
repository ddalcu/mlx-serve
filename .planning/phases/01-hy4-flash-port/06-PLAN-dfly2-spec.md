# PLAN 06 — DFly2 drafter (speculative decode)

Phase: 01-hy4-flash-port | Wave: 3 | Depends on: 03, 04
Requirements: HY4-06
Files: `src/plugins/hy_v4/blocks/dfly.zig`, `src/engines/mlx/spec/` touch (engagement + verify path), `tests/test_hyv4_spec.sh` (add)

## Objective

The built-in DFly2 drafter runs: 5 sequential draft steps, each conditioning on fused (embed, corrected hidden) state via `mtp_fusion` slot0, reading the target's latent KV and sparse indices from layers 25/29/33/37/41, drafting ≤5 tokens verified by the target in one forward. Output distribution identical with spec ON vs OFF.

**Semantics now pinned** (SEMANTICS.md §8 + `dfly_reference_sglang.py`, vLLM PR #68296 / SGLang PR #51159 / AngelSpec — HF reference genuinely omits the drafter):
- fusion: `e₀ + gate·eh_proj(cat(enorm(e₀), hnorm(h_target_last)))` on **slot 0 (the bonus token) only** — that is what `dfly_mtp_fusion: "slot0"` means
- hidden_correction (AngelSpec eq.15): residual MLP `6144→2·3072 SiLU-mul→3072` on each draft hidden, conditioned on the previous step's SAMPLED token; draft logits = target `lm_head(z + δ)`
- draft attention per query row: **target indexer's top-2048 snapshot at the last-accepted row ∪ own causal block slots**; draft KV is a **storage alias of the target's sparse layers 25/29/33/37/41** (zero extra KV; draft-written rows feed later rounds)
- `confidence_head` = acceptance-prob logit (3072→1), shipped but **skipped by both engines** (feeds D-Cut adaptive verification) — load the weights, do not wire it; record as a gotcha line
- draft sampling: gumbel at request temp (probabilistic) or greedy; verification = standard rejection sampling
- remaining [AMBIGUOUS], resolve by reading `dfly_reference_sglang.py` before coding: pre- vs post-final-norm target hidden in fusion; gate parametrization; RNG keying

## Tasks

### Task 1: draft-step forward
- `dfly.zig`: per §8 confirmed section + `dfly_reference_sglang.py`: draft layer n = its own MLA (own `q_a/q_b/kv_a/kv_b/linear_gate/o`, sink [32], dense MLP 9216) attending over the **aliased** KV rows of target layer `25+4n`, restricted to the target indexer's top-2048 snapshot at the last-accepted row plus the draft's own causal block slots; `hidden_correction` residual MLP on each z_i conditioned on the previous step's sampled token; draft logits = target lm_head(z+δ) through the resident affine path; `final_norm` last; `dfly_mask_token_id` 120009 per the reference's insertion rule; fusion applied to slot 0 only (above).
- Alias KV: no new cache allocation beyond the draft block rows written into the target layers' storage; billing = weights only (0.85B bf16, tiny) + snapshot buffer (2048 × bytes) per slot.
- **Verify:** unit: one draft step vs a dump from the SGLang/vLLM reference path (512 GB host; HF dumps cover the target model, the drafter dump comes from Tencent's engines); if a dump is unavailable at execution time → `checkpoint:decision`: shape-only bring-up, parity deferred to the ON/OFF acceptance test in Task 2.
- **Done:** one draft step produces logits; alias + slicing math pinned with citations.

### Task 2: loop + verify + engagement
- Spec loop: `num_speculative_tokens=5`, `draft_sample_method: probabilistic` (draft sampling temp per our sampling path); verify in one target forward (5 extra batch rows); accept via standard speculative acceptance; `dfly_enable_hidden_correction` respected.
- Gate the engine honestly until engaged: `spec-gate` lines, `[spec-stats]`, `/props` reporting dfly support; drafter weights load only when spec enabled (bill at load, evict with model).
- **Verify:** `tests/test_hyv4_spec.sh`: greedy outputs identical spec ON/OFF over 20 prompts (incl. a 32k-context arm), acceptance-rate log ≥ baseline PLD, engagement lines show draft tokens proposed+verified; no distribution drift at temp 0.9/top-p 1.
- **Done:** spec decode engages with parity, and a win recorded per /bench etiquette (name the engine, same-methodology cells).

## Verification (plan-level)
ON/OFF parity script green; spec gates honest before engagement; lint 0.

## Execution notes (plan 06, WIP)

- `blocks/dfly.zig` loads the pack's 5 dense draft layers (own MLA, dense FFN 9216), `mtp_fusion`
  (enorm/hnorm/eh_proj/gate), `hidden_correction` (3072 inter) and `final_norm`; `trunk.zig` carries the round:
  fusion on slot 0, target KV history read through `CacheOps.view` at layers 25/29/33/37/41 plus the block's
  local rows, target verify in one forward, argmax accept, cache rollback through a new `CacheOps.truncate`
  hook, and a sampled `verify`/`finish` split.
- DISARMED: `p.spec.native_stages` stays 0 and the table exposes no native_drafter. The round loaded and ran
  to the draft attention, where `mlx_astype` on the cache view faults with `EXC_ARM_DA_ALIGN` (SIGBUS); the
  view handle needs a dense/quantized-aware arm (or `kv.dense`) before the round is safe. The host cache
  `truncate` hook landed but has no live test.
- No acceptance/pacing number is claimed; HF transformers omits DFly2, so parity is SGLang/vLLM-only and the
  full-pack greedy ON/OFF script is deferred until the view path works.
