# PLAN 03 — Core forward: iHC + MLA + SWA + MoE (DSA layers fall back to local window)

Phase: 01-hy4-flash-port | Wave: 2 | Depends on: 01, 02
Requirements: HY4-03 (HY4-04 arrives Plan 04)
Files: `src/plugins/hy_v4/plugin.zig` (forward), `src/plugins/hy_v4/blocks/` (ihc.zig, mla.zig, swa.zig, router.zig), `src/plugins/hy_v4/kernels/*` + contracts, parity fixtures

## Objective

Text-only forward runs end to end at reference parity for greedy tokens: embed (4-bit) → iHC 4-stream init → 42 layers [hc_pre mix → MLA attn (DSA layers: local causal window as indexer placeholder; SWA layers: window 512) → hc mix → dense L0 / MoE 41] → hc_head reduce → final_norm → lm_head (4-bit). Correctness first; DSA selection and Engram slot in later plans.

## Reuse bill (from the codebase inventory — copy `//! copied_from:` per repo policy)

- **iHC:** `sdk/blocks/hc/{hc_decode2,hc_prefill,contract}.zig` reusable as-is (generic over hc count); the `hc_base/hc_fn/hc_scale` mixing shape exists — copy the arm from `glm5_next/trunk.zig` `Mhc` (fn_rows/base/scale) + `glm5_next/kernels/{mixes,hc_pre_*,collapse,expand}.metal`, marked `//! copied_from: plugins/glm5_next`; the per-module (attn/mlp separate) mix and `hc_head` reduce are new code on that skeleton.
- **MLA:** copy `glm5_next/table_forward.zig` `dsaAttention()` skeleton (q_a→norm→kv_a→norm, latent cache read/write, embed/unembed folds) + `trunk.zig` `pub const Dsa` struct as the cleanest second-arch template; ADD the rope arm from `deepseek_v41` (`ropeQat`/`ropeTail`/`ropeTailOnly` + `rope_tail*.metal`) — GLM's MLA is NoPE, ours needs partial-rotary on the 64-dim tail + M-RoPE.
- **M-RoPE:** `sdk/vision/mrope.zig` `interleavedSelector(sel, [12,10,10])` + `sdk/blocks/rope/mrope.zig` reusable as-is (sections arbitrary, tested); wiring pattern from `qwen4_exp/trunk/attn.zig` (parse in config, build cos/sin per chunk).
- **MoE router:** `sdk/blocks/moe/router_kernel.zig` `RouterMode.sigmoid_bias` (selection-only bias) + `routing.zig` reusable as-is; wiring pattern = `glm5_next/trunk.zig` `routed()`; expert exec `sdk/blocks/moe/*` as-is. The LatentMoE up/down wrappers are new (thin linear wrappers around existing expert-bank execution).
- Per-slot `LState`/`State` pool state shape: copy-in from `deepseek_v41` when Plan 04 lands.


## Tasks

### Task 1: iHC streams + MLA/SWA attention
- `ihc.zig`: per-module mix from `hc_base[8]`/`hc_fn[8,12288]`/`hc_scale[2]` per SEMANTICS.md §1; identity residual + gated delta over 4 streams; `hc_head` [4,12288]/[4]/[1] final reduce; `hc_magnitude=2.0`, `hc_eps` exactly per reference; f32 math for the mixing (config `fast_op_fp32_linear_gemm`).
- `mla.zig`: q_a→RMSNorm→q_b (32 heads × (192+64)); kv_a (512 latent + 64 k_pe); interleaved M-RoPE [12,10,10] with t=h=w=pos for text; per-head q/k RMSNorm; learnable sinks; `linear_gate` sigmoid gate on attention output; `post_proj_norm` placement per reference; o_proj reads the 8192-wide (padded-nope) form raw.
- `swa.zig`: same machinery, window 512 (511+current) hard bound, rope theta 1e4; KV kept in latent form (576/row), expand via kv_b at read — [AMBIGUOUS: expand-at-read vs materialize]; decide with a measured A/B, log the decision.
- KV-cache layout: latent KV per layer, sliding window for SWA layers, full for DSA placeholder.
- **Verify:** unit tests per SEMANTICS equation (mixing, rope interleaving, sink, gate) against f64 references; kernel contract for the RoPE/norm/gate fusions; on the tiny fixture pack, layer-1 forward cos ≥ 0.999 vs `tests/dump_hyv4_fixtures.py` outputs.
- **Done:** attention+HC math proven against reference on fixtures; kernels pass 5-gate contract.

### Task 2: MoE + dense FFN + top-level wiring
- `router.zig`: sigmoid scores + `e_score_correction_bias` (selection only vs weighting per reference [check citation]), n_group=1/topk_group=1 so no group masking, top-16, `norm_topk_prob`, `routed_scaling_factor=3.998`; LatentMoE path: `routed_expert_up_proj`(3072→1536) → 512× experts(1536→1536, swiglu clamp 10) → `routed_expert_down_proj`(1536→3072); shared expert (13824) always added; `combine_output_norm` sandwich location per reference.
- Dense layer 0: gate/up swiglu-limit + down + `post_fc2_norm` (sandwich post).
- Wire 42-layer loop + final `model.norm` + lm_head; stream/sink events; sampling untouched.
- **Verify:** greedy parity vs HF reference on ≥3 prompts ≤2k tokens (fixture oracles on the tiny pack first, full pack on the 512 GB host); expert-routing unit test reproduces top-16 ids + weights from a fixture hidden state; `zig build test` green.
- **Done:** full text forward at parity; `MLX_SERVE_RAW_DUMP_FILE` replay of a chat request matches reference greedy tokens.
- **HY4-08 (serving, inherited host work — verify not rebuild):** after Task 2, `POST /v1/chat/completions` works stream + non-stream through the existing host for a claimed chat model; `/v1/models` lists the pack; discovery finds it; the pack's `chat_template.jinja` renders via jinja.cpp (fallback format documented if unsupported — check first); a format-corpus entry pins the template. These are acceptance checks here; the implementation is host machinery, so no new host code expected — if a check fails, fix in `host/` with a regression test.

## Verification (plan-level)
Tiny-pack parity test in CI; full-pack greedy parity script (`tests/test_hyv4_parity.sh`, 512 GB host, env-gated like other cos-oracles); lint 0.
Deferred-scope note: tiny dev pack `test-org-copilot/HYV4-0.7B` (`-mlx`, ~838 MB) covers the text path in CI; engram/DFly/vit fixtures are synthetic (Plan 02 fixtures).
