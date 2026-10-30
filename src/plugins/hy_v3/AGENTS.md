# hy_v3

Owns `hy_v3`: Tencent Hunyuan 3 (Hy3, 295B-A21B MoE), mlx-lm's layout (PR #1211). Packs: mlx-community `Hy3-oQ2e`,
pipenetwork `Hy3-REAP*`, ox-ox builds.

## Files
- `plugin.zig`: the table. `Decoder(TableEmbed, RmsNorm, GatedAttention{output_gate = false}, EitherMlp(GatedMlp,
  Moe), AddNorm, LmHead)`; decode reads each dense or 8-bit attention projection's requantized side copy, built at
  load as the engine's `attnDqFor` builds it (`sdk/bases/blocks/decode_copies.zig`); one compiled routing chain for every layer.
- `config.zig`: MLX-free claim + parse. `proof_test.zig`: bit-equal to the legacy `forwardMoeWith`.
- `mtp_head.zig`: the MTP head (`SpecHooks.head`): `mtp.*` from `model-mtp.safetensors` (or the trunk's shards),
  eh_proj + a full hy3 layer + final norm, on the trunk's routing chain; the step split shared with nemotron_h's head.
  `mtp_head_test.zig`: bit-equal to the legacy `MtpModel` (`.hy3`), then served against the legacy sidecar arm.
- `load.zig`: legacy (`hy3ExpertContainer`, read by the engine's layered loader).

## Facts the table declares
- `trunk_weight` hands drafters the table and head.
- Serial decode (the blocks batch, the table refuses `batched_rows`: the engine never batched hy_v3), prefix cache on,
  `dflash_capture`, `moe`. No exactness contract declared (drafted, resumed and concurrent are unverified).
- The stop set always holds `<｜hy_eos:opensource｜>` (120025). No family sampling fill: tencent's generation_config is
  untruncated (temp 0.9, top_k -1, top_p 1) by design.
- Parse declines (`unsupported`), each a shape the engine never served: a layer span (the cluster refuses hy_v3,
  `DistArchUnsupported`), `expert_quant` (the engine reads no EXL3 banks for hy_v3: `MissingWeight`), a
  `quantization_config` with a `quant_method` (no reader; the engine took the tensors as plain ones), `modules` (the
  engine read a Prism pack without its rotations), `pooling_mode` / `use_bidirectional_attention` (an embedding
  hy_v3: no pack, no plugin embedding path).
- Served shapes the parse once declined: mlx-lm's `quantization_config` copy (ignored, as the engine does); a 2-bit
  group-128 pack, ternary when every bias is exactly -scale (`armTernary`: the dense projections then read qmv2's
  kernels, `Linear.ternary`, as the engine's `qmatmul`); `qk_norm: false` (q and k still pass a WEIGHTLESS RMS norm,
  as the engine computes it, where mlx-lm skips the norm); a `vision_config` (`caps.images`: the engine's generic
  tower, rows spliced at `image_token_id`).

## Rules (moved from root AGENTS.md)
- Full-attention MoE: GQA with per-head q/k RMS norm, full rotary, scale `query_pre_attn_scalar^-0.5`, no output
  gate. Sigmoid router `mlp.router.gate` (8-bit or bf16) with the f32 selection bias `mlp.expert_bias` (zeros when the
  pack ships none) + top-k renorm (`route_norm`) + `router_scaling_factor`; an UNGATED shared expert
  `mlp.shared_mlp.*` (`num_shared_experts` x the expert width); layers below `first_k_dense_replace` are dense.
- The expert container is a CONVERTER choice: `mlp.experts.*` (ox-ox) or `mlp.switch_mlp.*` (mlx-lm), probed per
  layer, `experts` first. The expert width may ride as `expert_hidden_dim`.
- Tool calls (hy3): a tag parser never bails on ONE missing delimiter; plural-wrapper recovery keys strictly on the
  SUFFIXED `<tool_calls:` form.

## Known issues
- The hd-128 fused q/k norm+RoPE decode arm and the dense side copies are proven here; 8-bit side copies (requantized
  from values to nvfp4) are not exercised by a proof pack.
- A head with dense (bf16) experts is refused by name (`DenseExpertHead`): the legacy head binds the bank
  untransposed and `gather_mm` refuses it, so there is nothing to be equal to.

## Served

In `served.zig`; the shell runs on the legacy parse's `ModelConfig`. Proof: `proof_test.zig` servedCase (4-bit, and
bf16 with decode-attn-quant on: steps past a chunk, greedy decode) and `mtp_head_test.zig` servedCase (4-bit and
8-bit/3-bit heads: the engine's head calls bit-equal to the `.qwen` sidecar arm, a pinned-depth round loop with equal
tokens and accepted counts). The once-declined shapes (`proof_test.zig` shapeCase: 2-bit group 128 with the
`quantization_config` copy, ternary, `qk_norm: false`, a `vision_config` with spliced rows) bind to the plugin and are
bit-equal served. The legacy arms to delete are the hy_v3 arms of
`load/layers_layered.zig` and `load/weights.zig` with this folder's `load.zig`, `compiled_hy3_routing` and its
callback (`transformer.zig`, `sdk/blocks/moe/routing.zig`), the `expert_bias` routing arm of `moeMLP2WithRouter`
(`computeHy3Routing`), and the `.hy3` layout of the legacy `MtpModel` (`loadHy3Mtp`, the `eh_proj` arms of
`fcConcat`/`bind`). The config parse stays.

## Fixtures
- `conformanceFixture`: a dense and an MoE layer, width 64, f32. Proof packs: 4-bit and bf16, three layers, both
  expert containers, one layer without a selection bias.
