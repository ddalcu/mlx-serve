# flint plugin

Owns `flint` (Recursive Flint, 39.8B-A2.5B, rec_mlx layout). Served (see "Served"). Parse and load are the
attention-MoE family's (`plugins/laguna/config.zig`, `family.zig`); `fixtures.zig` is the rec_mlx oracle test
(env-gated).

- Forward: `Decoder(TableEmbed, RmsNorm, DecodeCopies(LagunaAttention(.sinks)), family.Mlp, AddNorm, LmHead)`: QK
  norm, 35 sliding + 14 NoPE full layers (`no_rope_layer_indices`), learned sinks (`self_attn.softmax_offset`, rec_mlx
  exports f32; cast to bf16 at load since MLX's SDPA takes sinks only in a dtype that promotes to its output's), the
  band kernel on sliding prefill; a dense `GatedMlp` on `mlp_only_layers` (weight presence) or renormed sigmoid
  experts with an ungated shared expert. Batches decode (per-slot views), as the legacy.
- Parse: sliding layers rotate at `rope_theta` (no local base), scale `1/sqrt(head_dim)`, route scale 1.
- Load: the selection bias is `mlp.gate.e_score_correction_bias` or rec_mlx's `mlp.gate.expert_bias`, zeros when
  absent. Decode-attn-quant side copies as laguna's.
- Proof (`proof_test.zig`): 4-bit, and bf16 with decode-attn-quant off and on, bit-equal to the recorded
  `forwardMoeWith` (steps crossing the window, captures, greedy) and `forwardMoeBatchedDecode` (batched rows).
  Mutations: no sinks, the `expert_bias` name unread, a serial table: each fails. Sliding layers through a ring
  cache (`ring` = window + 2) read the same rows as the full buffer.
- Drafters: `SpecHooks.trunk_weight` (the bare table and head) + `caps.dflash_capture`, so a pack with an in-dir
  DFlash sidecar binds the plugin (the legacy served it); the drafted stream replays one recorded from the legacy
  forward, whose embedding rows and draft head matched bit for bit.

## Served

In `served.zig`, on the legacy parse's `ModelConfig`; batching reads `Caps.batches_decode`. The legacy load arms
(sinks, `expert_bias`) and the `model_type == "flint"` batching line are deleted (`served.legacy_removed`): a pack
the plugin declines is refused by name. The proofs replay tapes recorded from the legacy forward (`fixtures/*.tape`);
`fixtures.zig` runs the served path against rec_mlx (env-gated).

## Rules moved from the root AGENTS.md

- Supported-architectures row: laguna's attention arm (QK norm, 35 sliding(513) + 14 NoPE full layers, no gate)
  plus learned sinks, laguna's MoE binding (sigmoid top-k renormed, UNGATED shared expert, dense `mlp_only_layers`).
  The model opens its own `<think>`; GLM-4.7 `<arg_key>` calls with no newlines. Oracle: `tests/dump_flint_fixtures.py`.
- jinja.cpp lexes `\xHH` (Flint's `replace("\x00", "")`); a DOUBLED opener keeps what follows the last one (Flint
  `<arg_key><arg_key>K`): host chat rules that name Flint, kept in the root.
