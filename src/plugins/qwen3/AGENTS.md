# qwen3 plugin

Owns `qwen3` (dense, QK norm) and `qwen3_moe` / `qwen3_moe_text` (Qwen3-30B-A3B, Qwen3-Coder-30B-A3B). `load.zig` and
the JSON readers in `config.zig` are the Qwen family's shared load/parse (qwen3_5 reads through them);
`twin_test.zig` is the family's legacy-vs-table parity harness.

- Dense: `Decoder(TableEmbed, RmsNorm, Attention + q/k norms, SwiGlu, ResidualTail, LmHead)`, served on the plugin
  path (`served.zig`). Bit-equal to `forwardStandardWith` / `forwardBatchedDecode` (`proof_test.zig`: 4-bit hd 32,
  bf16 hd 256 with the fused decode norm+RoPE and a tied head; steps, captures, lazy greedy, batched rows).
  Drafters bind through `SpecHooks.trunk_weight` + `caps.dflash_capture` (proof: a DFlash round loop emits the
  legacy stream with the same accepted counts, dense and MoE).
- MoE: `Decoder(TableEmbed, RmsNorm, GatedAttention (no output gate), Moe (softmax top-k renormed, no shared expert),
  AddNorm, LmHead)`, bit-equal to the layered forward through the table (4-bit; bf16 hd 256 with decode side
  copies). Served (`served.zig`); serial decode, as the engine serves it (its blocks could batch: a measured win waits
  on a gate run). DFlash through `trunk_weight` + `dflash_capture`, proven like the dense trunk.
- Facts: `chat.sampling_defaults` top_k 20 / top_p 0.95 (the family's documented defaults when a pack ships no
  generation_config; the engine still fills them via `applyFamilySamplingDefaults` until P8).
- Qwen3-Embedding (dense, `pooling_mode` / a `1_Pooling` sidecar / a folder named `*embedding*`): served. The engine
  config is the legacy parse, so pooling stays the engine's; one sequence rides the slot path, a padded batch the
  dense attention's cache-less arm (`embedBatchForward`: causal over its own rows, the key mask unread since padding
  is on the right). Pooled rows bit-equal to the legacy dense forward (`proof_test.zig`, 4-bit and bf16 hd 256).
- Parse declines (`unsupported`, legacy keeps the pack): an embedding pack on the MoE layout, vision/audio/linear
  keys, `layer_types` other than all-full, any rope scaling but llama3 (llama3 on the MoE layout too), a partial
  rotary factor, non-silu, MoE without `head_dim` (legacy keeps a stale 256), `mlp_only_layers`, a shared expert,
  2-bit group-128 (possible ternary), `quantization_config.quant_method`.

## Rules
- Qwen3-Embedding packs with no pooling sidecar are named `*embedding*`: last-token pooling is inferred from the
  folder (`poolingFromDirName`), which `parse` cannot see; the driver keeps such a pack on the plugin only for the
  types in `served.decoder_embedders`.
- The MoE parse never derives `head_dim` from hidden/heads (the dense one does): refuse a config without it.
