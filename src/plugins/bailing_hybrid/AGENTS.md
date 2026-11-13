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
- Ling-3.1 streaming pack (the EXL3 repack beside `experts.bin`): it keeps the release's config, so its EXL3
  `quantization_config` (not `expert_streaming`) sets `streamed_bank`; glm5_next's streaming rules then apply. Its prompt
  bill is `prefill_bytes`: the engine's unstreamed arithmetic over `sdk/bill/prefill.zig`, pinned by `bill_test.zig`
  (the closure stays in sdk/); `prompt_last_row` reads the head's last 32 rows only.
- Exact verify (streaming packs, `serialVerify`, ≤ 16 rows): every verify row carries serial decode's bytes
  (projections at `QuantTable.exact_rows`, KDA and MLA per row, `moe.mlpRows`), so greedy MTP text equals MTP off;
  not `exact_drafted` (sampled MTP is not position-keyed). Bar: `served_test.zig`.
- Ling-3.1's clamps AFTER the SiLU, per layer: `expert_swiglu_limit_list` (routed, the bank's or the composed chain's)
  and `share_expert_swiglu_limit_list` (shared, `moe.Moe.Weights.shared_limit_out`); GLM's clamp is BEFORE it.

Served, legacy forward deleted (`engines/mlx/plugin/served.zig`; `glm5_next/served_test.zig`: steps and greedy through the driver). DFlash via
`trunk_weight` + captures.
- MTP head (`head.zig`, streaming packs only): the NextN layer at `num_hidden_layers` (`trunk.lingHead`), one
  MLA + MoE block over `cat[enorm(embed), hnorm(h)]`, `final_layernorm` out; no legacy oracle (the served test runs it).
- A head the outside plugin runs (EXL3) is no `trunk_weight`: drafts re-score on a 4-bit g64 copy of its rows
  (`trunk.draftReadout`, the evaluated width; billed in `resident_bytes`) after shortlisting on the engine's 3-bit
  copy of it (billed by the engine). The trunk's logits keep the plugin's head.

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
