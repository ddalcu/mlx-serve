# kolibri1 plugin

Owns `kolibri1` (Aleph Alpha Kolibri-1, 78B-A3.5B, mlx-lm layout). Served (see "Served"). Parse and load are the
attention-MoE family's (`plugins/laguna/config.zig`, `family.zig`).

- Forward: `Decoder(TableEmbed, RmsNorm, PostNormed(DecodeCopies(LagunaAttention)), PostNormed(Moe), AddNorm,
  LmHead)`: 40 sliding(513) + 10 NoPE full layers (every full layer is NoPE), QK norm, sandwich norms at
  `rms_norm_eps`, routed experts on EVERY layer: top-k on the f32 logits + `expert_bias`, weights sigmoid(logits), no
  renorm, an ungated shared expert (`moe_logit_select`, the fused `logit_bias` router kernel). Serial: the table
  refuses batched rows.
- Checkpoint names: `post_attn_norm` normalizes the attention OUTPUT, `post_attention_layernorm` is the pre-FFN norm,
  `post_ffn_norm` the MoE output's; experts under `mlp.experts.*` (or `mlp.switch_mlp.*`), shared under
  `mlp.shared_experts.*`, the bias `mlp.expert_bias`. The router ships bf16 and is widened to f32 at load; a quantized
  router is refused (`UnsupportedKolibriConfig`), as the legacy.
- Parse: `norm_topk_prob: true` declined, as the legacy; sliding layers rotate at `rope_theta`.
- Proof (`proof_test.zig`): 4-bit, and bf16 with decode-attn-quant off and on, bit-equal to the recorded
  `forwardMoeWith` (steps crossing the window, captures, greedy). Mutation: RoPE on the full layers fails both. The legacy's plain-op sandwich
  tail matches `AddNorm`'s fused seams bit for bit.
- Drafters: `SpecHooks.trunk_weight` (the bare table and head) + `caps.dflash_capture`, so a pack with an in-dir
  DFlash sidecar binds the plugin (the legacy served it); the drafted stream replays one recorded from the legacy
  forward, whose embedding rows and draft head matched bit for bit.

## Served

In `served.zig`, on the legacy parse's `ModelConfig`. The legacy forward and load arms are deleted
(`served.legacy_removed`): a pack the plugin declines is refused by name. The proofs replay tapes recorded from that
forward (`fixtures/*.tape`, `engines/mlx/plugin/tape.zig`); the legacy `moeMLP` logit-bias routing stays as the
bases MoE block's oracle. The host's Kolibri effort words key on the template (`isKolibriEffortTemplate`).

## Rules moved from the root AGENTS.md

- Supported-architectures row: 40 sliding(513) + 10 NoPE full layers on laguna's attention arm, sandwich norms,
  UNGATED shared expert; top-k on logits + `expert_bias`, weights sigmoid(logits), router in fp32
  (`moe_logit_select`, fused `logit_bias` kernel). Serial.
