# laguna plugin

Owns `laguna` (poolside Laguna S 2.1, 117.6B-A8.5B coder). Served (see "Served"). `config.zig` and `family.zig` are
the attention-MoE family's shared parse and load: `flint` and `kolibri1` reuse them from their own folders.
`ubench_test.zig` is an env-gated GEMV bench.

- Forward: `Decoder(TableEmbed, RmsNorm, DecodeCopies(LagunaAttention(.softplus_gate)), family.Mlp, AddNorm, LmHead)`.
  `Mlp` is a dense `GatedMlp` on `mlp_only_layers` (found by `mlp.switch_mlp.gate_proj.weight` presence, as the
  legacy) or the routed experts (`Moe`: sigmoid + selection bias, renormed per `norm_topk_prob`,
  `moe_routed_scaling_factor`, an ungated shared expert). Serial: the table refuses batched rows, as the legacy never
  batches Laguna.
- Parse: per-layer heads, YaRN on full layers with the mscale COMPUTED (0.1·ln(factor)+1, never the config's
  `attention_factor`; XS ships 1.0), the context the YaRN window resolves. `gating` may be a bool. Read as the
  engine reads them: a `text_config` nesting, a flat `rope_parameters` of type default (flint-int4), an empty
  `vision_config` (Kolibri-1), `pooling_mode`, a missing `vocab_size`/`moe_intermediate_size` (widths off the
  tensors). Declined (`proof_test.zig` lists every shape, with what the legacy reading did): router logit
  softcapping, non-per-head gating, no `layer_types`/`sliding_window`/`head_dim` (Laguna: no `rope_parameters`),
  feature keys the legacy dropped (`rope_scaling`, `expert_quant`, KV sharing, `global_head_dim`, `swiglu_limit`,
  `attention_k_eq_v`), a full-layer rope type but default (YaRN on Laguna), YaRN without
  `original_max_position_embeddings`, a malformed value.
- Proof against a released config: `poolside/Laguna-XS-2.1-NVFP4-mlx`, `RecursiveSI/flint-int4` and
  `here-be-dragons-ai/Kolibri-1-MLX-3bit` (`engines/mlx/plugin/fixtures/mlx_lm/`) at the proof geometry
  (`both.atGeometry`) replay tapes recorded live against the legacy forward; the XS YaRN `beta_fast` 64 fails it.
- Load: router `mlp.gate` or HF-native `mlp.gate.proj` (probe); a missing selection bias is zeros (the reference
  zero-inits it). Decode-attn-quant side copies built at load and read at solo one-row steps through
  `DecodeCopies` (`sdk/bases/blocks/decode_copies.zig`). Refused by name: a dense expert bank in a quantized pack.
- Proof (`proof_test.zig`): 4-bit, and bf16 with decode-attn-quant off and on, bit-equal to the recorded
  `forwardMoeWith` over steps crossing the window (prefill, decode, verify, captures) and greedy decode. Mutations:
  route scale 1, no mscale.
- Drafters: `SpecHooks.trunk_weight` (the bare table and head) + `caps.dflash_capture`, so a pack with an in-dir
  DFlash sidecar binds the plugin (the legacy served it); the drafted stream replays one recorded from the legacy
  forward, whose embedding rows and draft head matched bit for bit.

## Served

In `served.zig`, on the legacy parse's `ModelConfig`, whose sliding and MoE fields size and bill the cache. The
legacy forward is deleted with flint's and kolibri1's (`served.legacy_removed`: a pack the plugin declines is refused
by name): `lagunaAttnWith`, the laguna arms of `forward/layered.zig` and `load/layers_layered.zig`, the laguna
batching line, the `proj_bench` rung that called it, and its bases proofs (these plugins cover them). Proofs replay
tapes recorded from it (`fixtures/*.tape`, `engines/mlx/plugin/tape.zig`). Not ported: the opt-in fused QKV, which
never won.

## Rules moved from the root AGENTS.md

- Supported-architectures row: nvfp4 experts, softplus attn out gate, YaRN + sliding, sigmoid routing, UNGATED shared
  expert. Serial.
- Config reads: when the reference IGNORES a field, the field is not the truth (laguna YaRN mscale).
- Assistant-history reasoning round-trips (`Message.reasoning_content`): laguna's template persists reasoning.
