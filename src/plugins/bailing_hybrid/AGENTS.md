# bailing_hybrid plugin

Owns `bailing_hybrid` (Ling 3.0, BailingMoeV3: tiny and flash packs, mlx-lm's layout and oMLX's). Reads its config
with GLM-5.3-Flash's readers and loads through `glm5_next/trunk.zig`; its own parse arm, forward and MLA.

- Parse (`config.zig`, MLX-free): `layer_group_size` -> `full_attention_interval`, KDA heads = attention heads, the
  bounded-sigmoid gate (`kda_lower_bound` < 0 REPLACES softplus), `no_kda_lora`/`use_kda_lora`, MLA geometry
  (`qk_head_dim` must equal nope + rope), head-wise sigmoid out gate, `rope_interleave`, grouped sigmoid routing
  (`n_group`/`topk_group`), one ungated shared expert. Equal to the engine's parse on both Ling fixtures
  (`glm5_next/model_config.zig`). Refuses (as the engine does): KDA LoRA gates, a lower bound >= 0, `use_mla_nope`,
  value/up-proj norms and nGPT, a non-sigmoid score, a gate granularity other than head-wise, no `kv_lora_rank`.
- Forward (`table_forward.zig`, a copy of the deleted `forwardMoeWith` bailing arms): KDA through `slots.kda`
  (`glm5_next/kda.zig`), naive MLA (`mla.zig`, a copy of the deleted `mlaAttnWith`) appending `[B, H, S, nope+rope]` keys and `[B, H, S, v]`
  values per head through `CacheOps`, the residual + norm fusions in the legacy order, captures = the layer output.
  One slot at a time (the engine never batches it: `kda_vector_gate`).
- Proof: `glm5_next/proof_test.zig` (`expectSteps` on `fixture.lingFiles`): logits, captures, KDA states and verify
  captures bit-equal to the deleted legacy forward's tape (`fixtures/steps.tape`). Conformance passes on the same pack.
- Bill: `kv_per_token` per attention head (MLA expands the latent before the write), bf16 `ssm_checkpoint`,
  `prefill_score_width` = nope + rope.

Served, legacy forward deleted (`engines/mlx/plugin/served.zig`; `glm5_next/served_test.zig`: steps and greedy through the driver). DFlash via
`trunk_weight` + captures. No MTP head: the legacy path serves none (a NextN `eh_proj` head binds to hy_v3/nemotron_h
only, `MtpModel.bind`), so a Ling-flash head is a new feature with no oracle yet.

## Rules (from root AGENTS.md)
- KDA = GDN with a PER-CHANNEL gate (the `_vec` kernel); per-head vs per-channel gating is an INDEXING contract.
- A lower bound may REPLACE the formula (`kdaUsesBoundedGate`).
- MLA = naive DeepSeek-V3 with an ASYMMETRIC K192/V128 cache; `--kv-quant 4|8` serve it.
- The depthwise convs ship flat (`<x>_conv1d.weight`) in the public MLX packs, nested in the reference: both bind.
- Thinking ON; GLM tool tags.

## Known issues
- A capturing verify leaves the last row of `spec_state_seq` unwritten by design (the final state is its own output):
  the proof compares the rows rollback reads.
- No `[dtype-trace]` or ladder diagnostics on the table forward; the dense layer has no ANE arm (the legacy
  `denseMLPMaybeAne` never engaged here: ANE prefill serves `qwen3_5_moe` only).

## The row from the root architecture table

Ling 3.0 (BailingMoeV3): KDA + MLA hybrid MoE, `layer_group_size` → `full_attention_interval`; KDA = GDN with
PER-CHANNEL gate (`_vec` kernel), BOUNDED-SIGMOID gate (`kda_lower_bound` REPLACES softplus), sigmoid out-gate; MLA =
naive DeepSeek-V3 (ASYMMETRIC K192/V128 cache; `--kv-quant 4|8` ok); `noaux_tc` routing. Thinking ON; GLM tool tags.
Mirror `rapid-mlx/Ling-3.0-tiny-MLX-4bit`
