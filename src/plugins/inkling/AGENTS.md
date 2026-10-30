# inkling

`inkling_mm_model` (Thinking Machines Inkling Small, 276B-A12B, REAP-prunable), text only:
`Decoder(TableEmbed, RmsNorm, Attention, Mlp, AddNorm, Head)` with `Hooks` (embedding RMS norm, the head's input
divided by the muP width multiplier). `config.zig` parses `text_config`; `attention.zig` holds `Attention`, `Mlp` and
`Head`, `moe.zig` the routed + shared `Moe` and the router chain.

- Attention has NO RoPE: a hidden-conditioned relative bias plus the causal (or band) constraint as one additive mask,
  causal short convs on k and v, per-head q/k RMSNorm at scale 1/head_dim, log-scaling of q and the mask on global
  layers past `log_scaling_n_floor`. Its output and the MLP's each pass a short conv.
- State: every layer's four conv windows are ONE f32 array `[k | v | attn | mlp]`, the slot's `.conv` state (the
  layout the legacy writes and the SSD tier stores). `Attention` writes the first three windows and carries the old
  MLP window; `Mlp` replaces it. Keep that order.
- Positions come from `moe_seq_offset` (`cache.ssm_state.entries` = layers, layer 0 not linear).
- MLP: dense layers below `dense_mlp_idx` times `mlp.global_scale` in its stored dtype; MoE: the f32 router over the
  routed and shared experts' rows, selection bias on sigmoid scores, ONE softmax over the logsigmoids of the
  selected and shared logits (`inklingRouterChain`), weights cast to the stream's dtype, the shared sum in f32.
- Head: the padded vocab's real rows (`unpadded_vocab_size`); the shell's `vocab_size` is that count.
- Serial (`batched_rows` refused), spec off, as the legacy. `trunk_weight` is on the table so a pack carrying a
  drafter sidecar binds the plugin; no `dflash_capture` (the legacy could not draft Inkling either: its verify has no
  conv-state rollback).
- Parse: nested or flat config, a missing vocab or expert width (off the tensors), as the legacy read them. Declined
  (`proof_test.zig` lists every shape, with what the legacy did): `expert_quant`, `use_sconv`/`shared_expert_sink`
  off and a non-sigmoid gate (the legacy ignored them), a sliding geometry other than the global one, a missing
  structural key, a malformed value.
- Proof (`proof_test.zig`): 4-bit and bf16 packs (4 layers: dense + 3 MoE, layer 2 global with log-scaling engaged,
  extent 16, window 4, padded vocab) and the released `Inkling-Small-mlx-2bit` config at the proof geometry, over
  `long_steps` and the greedy lazy decode, decode-attn-quant off and on, on the legacy parse's `ModelConfig`.
  Mutations: no gain, no MLP or k window carry, no log-scaling fail all four; no side copies fails the dq-on proof.

## Served

In `served.zig`, on the legacy parse's `ModelConfig`. The legacy forward is deleted (`served.legacy_removed`: a pack
the plugin declines is refused by name): `inklingAttnWith`, `inklingMoeMlp`, `inklingExpertsApply`, the inkling arms
of `forward/layered.zig` (embedding norm, layer tail, muP, vocab slice), `load/layers_layered.zig` and
`load/init.zig`. Proofs replay tapes recorded from it (`fixtures/*.tape`). Two places the plugin and the legacy
differ, both legacy defects: `mlp.gate.global_scale` is read at any float dtype (the legacy read f32 only), and the
expert reshape keeps the row axis at one row and one shared expert (the legacy squeeze dropped it).

## Rules moved from the root AGENTS.md

- Supported-architectures row: NO RoPE (RelativeLogits bias + causal short-convs); role-less markers; serial,
  spec off.
- Assistant-history reasoning round-trips (laguna, inkling templates persist reasoning); the extra-context
  `raise_exception` sniff, the invoke-marker buffering and the name-salvage rules are host chat rules and stay.
