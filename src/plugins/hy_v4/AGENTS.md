# hy_v4 — Tencent Hy4-flash (220B-A12B multimodal MoE)

`model_type: hy_v4_flash`. The trunk is a DeepSeek-family MLA + compressed-DSA MoE with iHC streams, an Engram
n-gram PLE on 4 layers, and a DFly2 draft stack in `model.dfly_layers.*`. Plans done: 01 claim/parse/bill/cache
layout, 02 loader + Engram stream tables, 03 text forward (iHC, MLA with sinks, SWA, latent MoE, dense L0).
Pending: 04 indexer, 05 Engram, 06 DFly2, 07 vision, 08 proofs/docs.

## Rules

- **Layer kinds come from WEIGHT presence, never `layer_types`**: the config list is HF-offset by one. The pack's
  DSA layers are `[0,1,5,9,...,41]` (12, with `compressed_indexer.*` tensors); the rest are SWA (64 heads, window
  512). Engram tensors sit at pack layers 1/11/21/31 while `engram_layer_ids` says 2/12/22/32; config stores the
  pack index (`id-1`) and the head-vocab rows are cross-checked against the pack's table rows at load.
- **Two attention geometries, chosen per layer**: DSA layers `num_attention_heads` 32 (`q_b_proj` 8192 =
  32×(192+64)), SWA layers `swa_num_attention_heads` 64 (`q_b_proj` 16384). `head_dim` 64 is NOT the MLA width;
  nope+rope is 256. There is no padding — the earlier "2× padded" reading was wrong.
- **Every attention layer has learned per-head sinks** (`learnable_sink_param`, f32 `[heads]`): one extra logit
  with a zero value, read through MLX's SDPA `sinks` arm. The dense MLP/SWA layers carry them too.
- **MLA absorbs K into Q**: `q_nope @ kv_b`'s K half lands the query in the latent, keys are
  `[latent | k_rot]` (576), values are the latent (512), and V is lifted by `kv_b`'s V half AFTER the reduction.
  Both halves are dense bf16 copies made at load (billed).
- **Text M-RoPE reduces to plain rope**: all three axes share the position, so the interleave selection is a
  no-op; the rotation is adjacent pairs (`traditional=true`) over the trailing 64 rope channels, theta 1e7 on
  DSA layers and 1e4 on SWA. Vision (Plan 07) is where the [12,10,10] interleave actually matters.
- **iHC is the DeepSeek-V4 form WITHOUT Sinkhorn** (`comb`, GLM's arm, not needed): `hc_fn` `[2*lanes,
  lanes*hidden]` mixes the flattened lanes, `pre = sigmoid(logits*scale[0]+base[:N]) + hc_eps`,
  `post = hc_magnitude*sigmoid(logits*scale[1]+base[N:]) + hc_eps`, the sublayer's update is spread over lanes
  by `post` and added to each lane's residual; `hc_head` collapses the lanes with its own gates. All f32.
- **The latent MoE draws its experts from the LATENT width**: `routed_expert_down_proj` hidden→1536,
  `experts.gate_up_proj` `[512, 3072, 1536/8]`, `experts.down_proj` `[512, 1536, 1536/8]`,
  `routed_expert_up_proj` latent→hidden. The router and the shared expert see the full width; SwiGLU clamps at
  `swiglu_limit` 10 (gate ≤ 10, up in ±10), the shared expert has no clamp and no post norm,
  `combine_output_norm` closes the block.
- **Router**: f32 logits, sigmoid scores, `e_score_correction_bias` picks only (weights are the unbiased scores
  at the chosen ids), top-16 sorted ascending by expert id (the reference's accumulation order), normalized by
  `sum + 1e-20`, × `routed_scaling_factor` 3.998. `n_group`/`topk_group` are 1 here, so no group masking.
- **The Engram tables never go resident**: 4× `U32[165M,12]` 4-bit·g32 ≈ 39.6 GB packed (60 B/row: 12 U32 codes
  + 3 bf16 scales + 3 bf16 biases). They stream from an SSD bank (Plan 05 reads rows; Plan 02 opens the ranges);
  the bill excludes them.
- **Fine indexer stays off** (`compressed_indexer_skip_fine_select: true`): compressed IndexPool only — 4×
  pooling → top-512 groups → top-2048 tokens. A pack with it on is refused by name.
- The DFly2 draft reads the target's indexer top-2048 snapshot at the last-accepted row ∪ its own causal block;
  its KV aliases target layers 25/29/33/37/41; fusion happens on slot 0 only. HF transformers does NOT implement
  DFly2 — the semantics live in SGLang #51159 / vLLM #68296 (see `~/claude-tmp/hy4-port/SEMANTICS.md`).
- `confidence_head` ships in the pack; Tencent's engines skip it — load it, do not wire it.

## Packs & fixtures

- Full pack: `~/.mlx-serve/models/TensorFold/Hy4-flash-MLX-4bit-MTP` (148 shards, 129.9 GB; 90.3 GB minus Engram).
- CI dev pack: `test-org-copilot/HYV4-0.7B-mlx` (~838 MB, no engram/DFly/vit weights); engram/DFly fixtures are
  synthetic. Fixture config: `src/fixtures/model-configs/hy_v4-flash.json`. Conformance + loader fixtures:
  `src/plugins/hy_v4/fixture.zig` (a tiny 2-layer pack: DSA layer 0 dense, SWA layer 1 with Engram + latent MoE;
  the conformance suite loads it and forwards prefill/decode/verify).
- Quant census: trunk linears/embed/lm_head at g64 4-bit, MoE experts at g64 (24 groups over the 1536 latent) and
  Engram embeddings at g32 (per-tensor overrides in `quantization`), iHC f32, DFly bf16. `enable_lm_head_fp32`
  lies — the lm_head IS quantized.
- KV per token: 1088 halves/layer (576 K + 512 V; MLA reads its values from the latent). DSA layers are full
  length, SWA layers a `window + 2048` ring.

## Files

`config.zig` claim/parse, `weights.zig` load + validate + Engram tables, `trunk.zig` per-layer build + forward,
`blocks/{ihc,mla,proj,moe,router}.zig` the math, `fixture.zig` the tiny pack, `loader_test.zig` census + load.
