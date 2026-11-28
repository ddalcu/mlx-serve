# hy_v4 — Tencent Hy4-flash (220B-A12B multimodal MoE)

`model_type: hy_v4_flash`. The trunk is a DeepSeek-family MLA + compressed-DSA MoE with iHC streams, an Engram
n-gram PLE on 4 layers, and a DFly2 draft stack in `model.dfly_layers.*` (config parsed and validated; the
drafter is not served by this plugin). Done: claim/parse/bill/cache
layout, loader + Engram stream tables, text forward (iHC, MLA with sinks, SWA, latent MoE, dense L0), compressed
indexer (recall beyond the budget), Engram (hash + streamed gather + injection), vision (SigLIP tower + projector +
image/newline rows + Hy4 M-RoPE), and the numeric oracle for every block.

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
  DSA layers and 1e4 on SWA. Vision is where the [12,10,10] interleave actually matters.
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
  + 3 bf16 scales + 3 bf16 biases). They stream from an SSD bank (`weights.Engram` ranges); the forward preads
  16 rows per token per layer and dequantizes off-disk; the bill excludes them.
- **The Engram hash is numpy-exact**: the odd multipliers are NOT in the pack (transformers redraws them), so
  `blocks/numpy_rng.zig` reproduces `SeedSequence -> PCG64 -> Lemire` bit for bit; `seed + 10007*HF_layer_id`,
  bound `INT64_MAX // (max(lookup_map)+1) // 2` — the pack's map is TRAINED (max 92619), not its 120818-row
  length, so a bound off the table size draws different multipliers and reads the wrong rows. The frozen
  `lookup_table` maps tokens to compressed ids (past it, or before the sequence, reads the pad row), the ids
  multiply and XOR (order 2 then 3), and each of the 16 heads (`(max_ngram-1) * n_head_per_ngram`) reduces
  modulo its own vocab size. Row = head offset (cumsum of the 16 sizes) + id.
- **Every Engram norm is torch's DEFAULT eps, not the config's**: the per-head 96-wide norms, the signal's
  unweighted norm and the key/query norms are plain `nn.RMSNorm`, whose kernel normalizes in f32 and takes
  `finfo(float32).eps` (1.19e-7), NOT bf16's 2^-7. Only the layer norms outside Engram read `rms_norm_eps`.
  The numeric oracle pins it (`defaultEps`).
- **fp8 is a HOST round trip**: the pinned MLX FFI has no float8 dtype, so each gathered 96-vector is scaled by
  `max|x|/448`, clamped, snapped to the e4m3fn grid (ties to even) and rescaled in Zig before upload.
- **The n-gram is the CURRENT call's tokens**: HF's `build_ngram_tokens` pads positions before the call, so a
  decode step is `[t, pad, pad]`. A chunked prompt therefore pads at chunk boundaries where HF (no chunking)
  would see history — parity holds for prompts forwarded whole.
- **The compressed indexer is the ONLY selector** (`compressed_indexer_skip_fine_select: true`, no second
  per-token stage): the DSA layer's cached token keys pool by a gated fp32 convex combination
  (`key_value_projection` * softmax(`key_gate_projection` + `intra_group_ape`)), the group key takes `key_norm`
  and the RoPE phase of its group's FIRST token, scores are `relu(q @ group_keys^T)` per head weighted by
  `head_weight_projection(x) * (heads*dim)^-0.5`, and each query keeps the best `group_topk - 1` CLOSED groups
  plus its own — 512 groups × ratio 4 = the 2048-token budget. Below `index_topk` tokens (2048) the layer attends
  densely, exactly as the reference does. A pack with the fine stage on is refused by name.
- **Attention reads the recall set as an additive mask over the slot axis** (dense masked attention — the
  reference's own arm). A gather kernel that reads only the recalled rows is a perf step, not a correctness one.
- DFly2 (the pack's 5 draft layers, fusion and hidden correction) is NOT served by this PR: its semantics live
  in SGLang #51159 / vLLM #68296, not HF transformers, and the round + rollback land with the PR that proves
  them. The pack's tensors are still validated at load.
- `confidence_head` ships in the pack; Tencent's engines skip it — load it, do not wire it.
- **The tower has NO RoPE**: the learned 128x128 position grid (row 0 is an unused cls row) is bilinearly
  resampled to the request's patch grid with Torch's `(size+0.1)/edge` scale_factor; attention is full and
  bidirectional per image. `vit.perceive.norm` is the pre-merge RMSNorm, `proj.0` the 2x2 conv and `proj.2`
  the 1x1 conv (the reference names them `before_rms`/`proj_conv`/`proj_out`).
- **Preprocessing is `HYV4VLImageProcessor`, not Pillow**: smart resize on the patch*merge grid (32) within
  [4096, min(2048², the engine's 1536² budget)], Torch's antialiased bicubic (`resizeRgbBicubicFloatChw`),
  rescale 1/255 then `(x-0.5)/0.5`. Patchify is row-major patches with channel-major features (C, 16, 16).
- **The merger appends one `image_newline` row per merged row**, so an image run is `(gh/2)*(gw/2+1)` image
  tokens (the processor's `_num_frame_tokens`); the forward splices the tower's rows at those ids in order
  from `Forward.media_offset` (a chunked prefill resumes there).
- **Hy4 M-RoPE has a newline column**: each frame starts a block of `max(gh/2, gw/2+1)` positions and text
  resumes at `max+1`; every layer ropes with its own theta (DSA 1e7, SWA 1e4). The raw 3-D table rides
  `Forward.mrope_positions` — the engine's one-theta `Forward.mrope` angles are not used.
- Video ids exist in the pack but video is refused by name (`prepare` refuses a non-image input).
- **A block scope per ViT layer**: a large image's attention scores are hundreds of MB; 27 blocks held in one
  graph blew Metal's resource limit. Free each block's intermediates at its scope end.
- **Prefix cache is OFF**: a restored prefix may hold image placeholder rows, and the splice resumes only
  inside a chunked prefill (`vision_splice_offset`); a cache restore across an image run would splice tower rows
  at the wrong offset. Re-enable only once the cache carries that offset.

## Packs & fixtures

- Full pack: `~/.mlx-serve/models/TensorFold/Hy4-flash-MLX-4bit-MTP` (148 shards, 129.9 GB; 90.3 GB minus Engram).
- CI dev pack: `test-org-copilot/HYV4-0.7B-mlx` (~838 MB, no engram/DFly/vit weights); engram/DFly fixtures are
  synthetic. Fixture config: `src/fixtures/model-configs/hy_v4-flash.json`. Conformance + loader fixtures:
  `src/plugins/hy_v4/fixture.zig` (a tiny 2-layer pack: DSA layer 0 dense, SWA layer 1 with Engram + latent MoE;
  the conformance suite loads it and forwards prefill/decode/verify).
- Numeric oracle: `tests/dump_hyv4_fixtures.py` builds tiny random modules with the HF modeling code and writes
  `src/plugins/hy_v4/fixtures/num/` (weights, inputs, outputs); `blocks/oracle_test.zig` pins iHC, gated MLA with
  sinks, the compressed indexer's recall mask, the latent MoE and the Engram gate at cos >= 0.999, rms +-0.02.
- Quant census: trunk linears/embed/lm_head at g64 4-bit, MoE experts at g64 (24 groups over the 1536 latent) and
  Engram embeddings at g32 (per-tensor overrides in `quantization`), iHC f32, DFly bf16. `enable_lm_head_fp32`
  lies — the lm_head IS quantized.
- KV per token: 1088 halves/layer on SWA layers and 1216 on DSA layers (K = latent 512 + k_rot 64 +
  indexer token key 128, V = the latent 512; MLA lifts values from the latent). DSA layers are full length,
  SWA layers a `window + 2048` ring.

## Files

`config.zig` claim/parse, `weights.zig` load + validate + Engram tables, `trunk.zig` per-layer build + forward,
`vision.zig` tower + processor + media hooks, `mrope.zig` the newline-row position table,
`blocks/{ihc,mla,proj,moe,router,indexer,engram,numpy_rng}.zig` the math, `blocks/oracle_test.zig` the numeric
oracle, `fixture.zig` the tiny pack, `loader_test.zig` census + load, `vision_test.zig` tower parity dump.
