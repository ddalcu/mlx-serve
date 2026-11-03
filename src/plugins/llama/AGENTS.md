# llama plugin

Owns `llama`, `mistral`, `qwen2` (mlx-lm's layouts), all three served by the MLX engine through this table
(`src/engines/mlx/plugin/served.zig`).

- Forward: `Decoder(TableEmbed, RmsNorm, Attention, SwiGlu, ResidualTail, LmHead)` from `sdk/bases`. Qwen2's q/k/v
  biases ride `Linear.bias` (a biased layer never joins q|k|v, as the legacy loader). Bit-equal to the legacy
  `forwardStandardWith` / `forwardBatchedDecode`: `src/engines/mlx/plugin/proof.zig` (llama 4-bit, bf16 + llama3
  rope + tied head) and `proof_test.zig` (Qwen2 4-bit and bf16 with biases and a tied head, Mistral 4-bit): steps,
  captures, greedy lazy decode, batched rows; the served `ModelConfig` equals the legacy parse.
- Load: the engine hands its loaded tensors (`LoadCtx.weights`); each projection goes through the pack's quant format
  (`sdk/quant/formats.zig`). Quantized layers join q|k|v and gate|up rows into one buffer (views, originals freed),
  exactly as the engine's loader did, so memory and the decode matmuls match. Embedding and head stay `[out, in]`.
- Parse declines (`unsupported` = the legacy path keeps the pack): hidden_act other than silu, head_dim 256 (the
  base lacks the fused hd-256 norm+RoPE decode kernel), any rope scaling but llama3, `layer_types`,
  `query_pre_attn_scalar`, vision/MoE/pooling packs, `quantization_config.quant_method` (the Mistral fixture's
  bitsandbytes block), 2-bit group-128 (possible ternary kernels), a `1_Pooling` sidecar, missing core dimensions.
- Mistral's `sliding_window` and Qwen2's `use_sliding_window` are ignored, as the legacy did (every layer attends the
  whole context, mlx-lm's reading).
- The engine refuses by name on this path: a drafter sidecar (DFlash/Gemma need embed+head hooks), int8 prefill,
  the MXFP8 head, media rows, draft trees.

Known gaps vs legacy (none changes output): no `[dtype-trace]`/ladder diagnostics, decode async ladder env ignored.

## Rules moved from the root AGENTS.md

- A struct default is ONE family's: Gemma's 1024 sliding default hit every Llama 3.x (no `sliding_window` key) past
  1k tokens; `rope_type: llama3` is read. Guard: smoke `long prompt`.
