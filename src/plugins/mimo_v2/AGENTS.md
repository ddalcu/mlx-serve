# mimo_v2

Owns `mimo_v2` and `mimo_v2_flash` (MiMo-V2-Flash; TensorFold's V2.6 label): Xiaomi MiMo-V2.6-Flash (309B-A15B) in
mlx-lm's layout (`tests/convert_mimo_v2.py`, mlx-community's mxfp4-q8 packs).

## Files
- `plugin.zig`: the table. `Decoder(TableEmbed, RmsNorm, SinkAttention, EitherMlp(GatedMlp, Moe), AddNorm, LmHead)`;
  `trunk_weight` hands drafters the table and head.
- `sink_attention.zig`: MiMo's attention block (copied from the engine's sink arm).
- `config.zig`: MLX-free claim + parse (+ `layout`: global layers full KV at their own KV heads, sliding layers a
  ring of window + 2048 rows; K 192 / V 128).
- `proof_test.zig`: bit-equal to the deleted legacy `forwardMoeWith` and `forwardMoeBatchedRows`, replayed from
  `fixtures/*.tape` (`both.Both.replay`).
- `mtp_head.zig` (+ `mtp_head_test.zig`): the MTP heads behind `SpecHooks.head`, bit-equal to the recorded legacy heads.
- `fixtures.zig`: the served MiMo against `modeling_mimo_v2.py` (model-gated).
- `streamed_test.zig`: the experts read from an `experts.bin` (the stream packs' MXFP4 bank) through the MoE block's
  `Routed` hook (`moe.Moe.Weights.streamed`), bit-equal to the in-memory banks (served logits and the block alone,
  evictions included); at the release's 4096 / 2048 the hook's decode MoE against fp32 truth. A streaming pack (fp8 residents, `experts.bin`) loaded through the table serves the same pack's
  in-memory logits bit for bit.
- `dec_attn.zig`: the split-K kernel's names (`sdk/blocks/attention/dec_attn.zig`) for gpt_oss.

## Facts the table declares
- Batched decode and batched verify rows (the engine batches MiMo: no per-slot recurrent state), prefix cache on,
  `dflash_capture`, `moe`, thinking ON by default (the template thinks unless `enable_thinking: false`).
- MTP: `mtp_heads` = `num_nextn_predict_layers` (at most 3), `in_checkpoint`, reading the final-normed hidden; the
  pack must ship exactly that many `model.mtp.layers.{k}` (read from the shards the index names for `model.mtp.*`).
- Streaming packs (`expert_streaming`: mlx-stream's MXFP4 bank): the routed experts
  come from `experts.bin` through `sdk/weights/streamed.zig` (`Moe.Weights.streamed`, the trunk's own MoE math), the
  residents from the index (`fp8_block128` q/k/v and dense MLP: the shared Linear's e4m3 arm), the MTP heads from
  `mtp-heads.safetensors`. The engine serves them here (`ModelConfig.expert_bank`): slot rows per routed layer are the
  model's `stream_experts` setting, a quarter of the experts when it sets none; the load bills the residents and
  those rows (`preflight.shardResidentBytes`).
- Parse declines (`unsupported`): another `expert_streaming` bank or `residents_format`, `n_group` > 1, a layer span. The original release (a `quantization_config` with a `quant_method` and no `expert_quant`) FAILS by
  name: convert it with `tests/convert_mimo_v2.py`.
- Sushi packs (`expert_quant`): the engine's loader adapts the shards (`sushi_pack.adapt`: the fused e4m3 QKV split
  into row-scaled e4m3 q/k/v, the dense MLP decoded to bf16); the routed experts load and run through the pack's
  quant format (`QuantTable.load_bank` + `swiglu`, `moe.FormatBanks`); the MTP heads, read from the shards, get the
  same adaptation (`sushi_pack.adaptMimoHeads`).

## Rules (moved from root AGENTS.md)
- gpt_oss's sink arm with PER-LAYER geometry: global layers `num_key_value_heads` at `rope_theta`, sliding layers
  `swa_num_key_value_heads` at `swa_rope_theta` under a 128-row band with learned sinks (`self_attn.sinks` or
  `attention_sink_bias`); K 192 / V 128, RoPE on `int(192 * partial_rotary_factor)` = 64 channels, V scaled by
  `attention_value_scale` (in V's dtype) BEFORE caching; additive q/k/v/o biases, no q/k norm.
- A full-buffer sliding layer caches every token it never reads: the sliding layers keep a RING of window + slack
  rows (`CacheFacts.sliding_ring`), billed per slot, restored above the ring floor, never spilled.
- Decode-attn-quant side copies are read at decode width AND on a speculative verify window up to `dec_attn.MAX_ROWS`
  (a verify scoring drafts on other weights than the serial steps would diverge); a prompt chunk prefills dense.
- Global layers decode through the split-K kernel (`dec_attn`: a KV group's heads x positions are one MMA tile's
  rows); prefill past 16 rows takes NAX attention at q/k 192 / v 128 (`fusedSinkAttnPrefill`).
- Experts: laguna's MoE binding (sigmoid top-k renormed, a selection-only bias `mlp.gate.e_score_correction_bias`,
  zeros when dropped, NO shared expert), router `mlp.gate` or `mlp.gate.proj`, routed in f32 (a bf16 router widens
  once at load); the dense MLP layers lead the stack.
- A template that renders tool CALLS but never names a `'tool'` role is tool-aware (MiMo's generic `{{ role }}`
  header): the host probe-renders one tool turn and keeps it native.

## Known issues
- Head k's row p pairs the trunk hidden at p with token p+k+1, so a request's state (each head's rows, pending
  hiddens and tokens, the round's drafts) rides the head cache's recurrent entries; KV layer 0 only counts the
  committed rows. The head cache is never persisted (`MtpCacheRef.kv` null): a prefix-cache hit rebuilds the heads
  from the uncached tail, as the legacy heads did.

## Served

In `served.zig` (`mimo_v2`, `mimo_v2_flash`), the legacy forward deleted (`legacy_removed`); the shell runs on the
legacy parse's `ModelConfig`. Proofs replay tapes recorded from the legacy forward: steps, greedy and layered batched
decode; the engine's head calls (history, a `.none` opening, `.mixed`/`.logits` drafts, a partial-accept truncate and
a reopen); a pinned-depth MTP round loop with its tokens and accepted counts. Sushi packs too (`sushi_test.zig`: e4m3
QKV, e4m3 MLPs, EXL3 banks). The config parse
and `round.zig`/`round_finish.zig`'s `isMimo` rollback stay (the served shell's config keys them).

## Fixtures
- `conformanceFixture`: a dense global and an MoE sliding layer with sinks, width 64, head 32 / V 16, f32. Proof
  packs: 4-bit and bf16, four layers, both router and sink names, one MoE layer without a bias; a 2200-row ring case.
  Oracles (model-gated): `tests/dump_mimo_v2_fixtures.py`, `MIMO_V2_MODEL` / `MIMO_V2_FIXTURE`.
