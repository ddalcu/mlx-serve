# spark2_5 plugin

Owns `spark2_5` (XHToken Spark-X2.5, 1.7B/4B dense, hd 256; packs `abenzerps/Spark-X2.5-4B-MLX-{4,8}bit`, mlx-lm
layout). Served (see "Served").

- Forward: `Decoder(TableEmbed, RmsNorm, Attention, GatedGelu(.erf), ResidualTail, LmHead)`. `attention.zig` is a
  copy of `sdk/bases/blocks/decoder_attention.zig` (sliding + per-head gate) plus the hd-256 fused RoPE decode arm
  the shared block refuses (`FusedQkNormRope256NotPorted`); fold it back once the shared block carries that arm.
- Load: the checkpoint's one `q_k_v_proj` is row-sliced into q, k, v (dense: transposed per slice, as the legacy
  `splitFusedQkvRows`), then a quantized layer joins q|k|v|g for one-row steps; `model.embedding`, tied head.
- Parse declines what the legacy reading fills from Gemma defaults: no `head_dim`, no (or null) `sliding_window`, no
  `layer_types`; a non-default full-layer rope type, a flat rotary fraction; MoE/KV-sharing keys; a malformed value
  (`proof_test.zig` lists every shape). Read as the engine reads them: `text_config` nesting, a flat
  `rope_parameters` of type default, `vision_config`, `pooling_mode`, a missing `vocab_size`/`intermediate_size`.
  The released `abenzerps/Spark-X2.5-4B-MLX-4bit` config at the proof geometry replays a live-recorded tape.
- Proof (`proof_test.zig`): 4-bit and bf16 packs at hd 256 (3 sliding + 1 full, 25% rotary, window 4) bit-equal to
  the recorded `forwardStandardWith` / `forwardBatchedDecode` over steps crossing the window, greedy lazy decode,
  batched rows.
  Mutation: full rotary on the full layer fails both. Disabling the fused RoPE arm passes too (its bits equal the
  composed chain at these shapes).
- Drafters: `SpecHooks.trunk_weight` (the bare table and head) + `caps.dflash_capture`, so a pack with an in-dir
  DFlash sidecar binds the plugin (the legacy served it); the drafted stream replays one recorded from the legacy
  forward, whose embedding rows and draft head matched bit for bit.

## Served

In `served.zig`, on the legacy parse's `ModelConfig` (sliding layers and window from it). The legacy arms are
deleted (`served.legacy_removed`: a pack the plugin declines is refused by name): the `embedding` table name, the
`q_k_v_proj` split, `out_proj` and the per-head `g_proj` gate of the dense load and `attnOutGate`. Proofs replay
tapes recorded from that forward (`fixtures/*.tape`); `bases_test.zig` proves the bases
`DecoderAttention(.gate = .head)` block, built from this plugin's weights, against tapes of the same.

## Rules moved from the root AGENTS.md

- Supported-architectures row: 3 sliding(512)+1 full layers with PER-TYPE RoPE, PER-HEAD sigmoid out-gate, fused
  `q_k_v_proj` split at load, tied head. DeepSeek-style template, GLM `<arg_key>` tools; per-digit tokenizer.
- Story: `docs/gotchas/models-media.md` "Spark-X2.5 (`spark2_5`) port".
