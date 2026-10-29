# k2_horizon plugin

Owns `k2_horizon` (IFM K2-Horizon dense, 0.9B to 32B; 7B: 36 layers, hd 128, 512k ctx), served by the MLX engine
through this table.

- Forward: llama's decoder with `GroupedRmsNorm` (`norm.zig`, `K2HorizonRMSNorm`: rms per channel group, then the
  full weight; per-head widths stay plain). The norm is never fused into a residual seam, as the legacy declined
  every fused seam for grouped norms. Loaded by llama's `loadDense`; parsed by llama's `parseFamily` with
  `grouped_norms`.
- Proof (`proof_test.zig`): 4-bit and bf16 packs bit-equal to the legacy `forwardStandardWith` /
  `forwardBatchedDecode` (steps with every layer captured, greedy lazy decode, batched rows); served `ModelConfig`
  equals the legacy parse (`norm_groups` is plugin-owned).
- Parse declines MoVA / MoE sizes (`mova_num_experts`, `num_experts` > 0), `query_key_norm: true` and an
  `attention_gate_func`: those configs run another attention.
- Chat facts set here (the host still sniffs them until P8 reads Parsed): thinking default ON, effort vocabulary
  `k2`, `reasoning_round_trip = required_field`, the 14 `<ifm|…>` marker aliases.

## Rules moved from the root AGENTS.md

- Supported-architectures row: Llama trunk + GROUPED RMS norms. `<ifm|…>` markers decode to `<think>`/GLM tags
  (`installMarkerAliases`); the template RAISES on a thinking-less assistant turn and on any other effort word
  (`k2EffortFor`). Thinking default ON; MoVA not served.
- A marker family that is GLM/`<think>` under another spelling is ALIASED at decode, never re-parsed (K2 `<ifm|…>` →
  `Tokenizer.marker_aliases`); only the rendered prompt keeps the pack's spelling (`k2ThinkOpenerAt`). JSON-schema +
  thinking on K2 keeps thinking off (known).
- Story: `docs/gotchas/models-media.md` "K2-Horizon (`k2_horizon`) port".

## The row from the root architecture table

IFM K2-Horizon dense (7B: 36 layers, hd 128, 512k ctx): Llama trunk + GROUPED RMS norms (`groupedRmsNorm`). `<ifm|…>`
markers decode to `<think>`/GLM tags (`installMarkerAliases`); the template RAISES on a thinking-less assistant turn
and on any other effort word (`k2EffortFor`). Thinking default ON; MoVA not served.
