# glm5_next plugin

Owns `glm5_next`, `glm5_next_text` (GLM-5.3-Flash: mlx-community, NVIDIA ModelOpt FP4, Sushi EXL3 and EXL3 streaming
packs) and claims `glm_moe_dsa` to refuse it (GLM-5.3, `plugins/glm_moe_dsa`). Ling 3.0 (`plugins/bailing_hybrid`) shares this plugin's
config readers, trunk (`trunk.zig`), KDA layer (`kda.zig`), slot access (`slots.zig`) and fixture writer.

- Parse: `config.zig` (MLX-free) reads every key the engine's GLM arm reads into `Config`; `model_config.zig` maps it
  to the engine's `ModelConfig` field for field (`expectEngineParse`: equal on the three GLM fixtures, a flat pack, the
  refusals, and the file parse with processor + generation config). Refuses: an `expert_streaming` bank other than the EXL3 K=3 mul1 manifest-v2 one, `qk_rope_head_dim`,
  any Gemma/BERT/sliding key (`foreign_keys`), `glm_moe_dsa`.
- Load (`trunk.zig`): reads the driver's resident source (`LoadCtx.weights`: the engine's map with the Sushi and
  ModelOpt adapters applied) in the layout the deleted legacy load built: q|k|v and the convs joined, GLM's decode row
  group `[qkv | z | a | b]`, f32 KDA tables, DSA and MLA projections, the MoE block's weights, the `mtp.0` head.
  Quant params resolve per weight at load (`resolve`, the legacy `computeQuantParams` over known input widths).
  `Model` carries the helpers the forwards took from `Transformer` (`mm`, `rmsNorm`, `narrowAddNorm`, `embedding`,
  `lmHead`, `denseMlp`, `headMm`), same op order.
- KDA (`kda.zig`): the per-channel-gate arms of the legacy `gatedDeltaNetProjected` on a slot's `Entry`: GLM's
  one-dispatch decode/verify (`kda_recurrence.decodeStep`), low-rank gates, the fused prework, the per-core prompt
  recurrence; Ling's softplus gate (compiled at load) and the stock vector-gate kernels.
- MoE (`moe.zig`, Ling's too, the MTP head's): the shared `sdk/bases/blocks/moe.zig` block; GLM's one-kernel decode
  (`glm5.moeDecode`, affine, up to 16 rows) is the block's `fused` arm, ModelOpt's per-expert up x down scale its
  `route_scale`, Sushi's EXL3 banks its `format_banks`.
- Forward (`table_forward.zig`): the [B,S,4,D] stream through KDA / DSA sublayers inside mHC; KV and recurrent state
  only through `CacheOps` (`slots.kda` writes conv, ssm and a verify's per-row capture back); positions from
  `Forward.positions`; captures = the stream MEAN; batched rows in groups of 8 (`MOE_DECODE_BATCH_MAX`), each slot's
  KDA and DSA by its own call.
- Spec: `trunk_weight` (embedding table and head, both KDA hybrids); `SpecHooks.head` = `head.zig`, the `mtp.0` layer
  read with the trunk and made resident by `load`, its block over the head's one-layer cache at the cache's OWN length (`CacheOps.len`),
  never the trunk position. DFlash: captures through `Forward.capture_layers`; the policy stays engine-side.
- Proof (`proof_test.zig`, tiny random pack at the release's DSA geometry, `fixture.zig`): every row's logits, both
  captured layers, every KDA state and verify capture bit-equal to the deleted legacy forward's tapes
  (`fixtures/*.tape`, same-device digests) over prefill, decodes, verify windows past the indexer budget, a later
  chunk; batched rows (two row groups); a device token; media rows over two chunks; the `mtp.0` head; a Sushi pack
  (EXL3 experts through `sdk/quant/exl3/format.zig`). Served (`served_test.zig`, the driver on the legacy config via
  `both.zig`): steps, pipelined greedy, media; the head through `MtpHeadRef.plugin` incl. a pinned-depth round loop.
- Bill: `kv_per_token` (latent + indexer key and gate per DSA layer), `ssm_checkpoint` (f32). The prefill bill stays
  `bill.zig`'s `glm5PrefillMemoryNeeded` (the served config is the legacy parse, so `isGlm5()` arms still read it).
- Discovery: caps advertise what legacy discovery did (no images for a `_text` pack's vestigial tower; video beside a
  `video_token_id`, which the engine never reads). One change: a ModelOpt pack's unloaded row names 4 bits, as its
  loaded row always did (legacy discovery read only `quantization.bits`).

Served, legacy forward deleted (`engines/mlx/plugin/served.zig` `legacy_removed`: a pack the plugin declines is
refused by name). Streamed packs (`expert_streaming`, `glm_moe_dsa`) stay mlx-stream's.
Batched verify rows (`batched_rows`, rows > 1) are refused by name: the engine keeps GLM's MTP rounds solo
(`mtpRoundsStaySolo`), so nothing calls it.

## Rules (from root AGENTS.md)
- A norm's epsilon sits where the REFERENCE puts it: fla's l2norm adds it to the SUM (`gdnQkNormEps`: 1e-6/dk).
- A capture gate is a promise the forward keeps: GLM taps the MEAN of its hyper-connection streams.
- A sparse target's DFlash is policy-driven (`dflash_policy.zig`).
- Generation growth is billed to the KV reserve's HEADROOM, never the auto budget (`glm5PrefillMemoryNeeded`).
- A cache row no op reads is never computed: the indexer's rows below its budget ride `mlx_depends`.
- Partials crossing threadgroups in ONE dispatch go by atomic store/load (`hcPre`).
- jinja.cpp reads `x.0` as an INTEGER index (GLM-5 tool history).
- A prefix-cache HIT is not bit-identical on a hybrid: low-margin tokens move >1 nat with chunk width (numerics).

## Known issues
- With a vocab under 16384 rows the engine's served-head readout builds a 3-bit
  draft head the legacy GLM head never built (drafts only, never output): needs a `SpecFacts` fact.
- Kernel contracts for the 20 `kernels/*.metal` rows are not registered.
- No `[dtype-trace]` or layer-cap diagnostics on the table forward.
- The dense layer's one-row decode takes the unclamped joined SwiGLU (`Model.denseMlp`), as the legacy forward did,
  though the config declares `swiglu_limit`; wider rows clamp. Kept for bit-equality; unverified against the reference.
- Not ported (no GLM or Ling pack uses them): the ternary 2-bit arm of `qmatmul`, the RHT arms, ANE prefill.
- `vision.zig` (the tower the engine calls) and `bill.zig` (the engine's prefill bill) still import the engine's
  `ModelConfig`/`Transformer`.

## The row from the root architecture table

GLM-5.3-Flash: KDA (f32 state) + DSA (absorbed NoPE MLA, indexer past position 2051) inside mHC, 288-expert MoE.
Thinking ON. Batches decode (`forwardGlmBatchedDecode`); MTP = its `mtp.0` layer. Vision = `vision.zig`, rows spliced
at `<|image|>` on 1-D positions.
- Streaming pack (`expert_streaming`, EXL3 K=3 mul1 bank `experts.bin`, `fp8_block128` residents): `config.streamed`;
  `plugin.zig` `openBank` binds `sdk/weights/streamed/glm5n_source.zig` (MiMo's streamer, the GLM-shape EXL3 texts)
  behind `Moe.Weights.streamed`; slot rows from free memory (`bill.streamRows`, all 288 on 256 GB); the `mtp.0` head's
  NVFP4 experts load resident (`sdk/quant/nvfp4/stream_bank.zig`). Serial (no batched decode).
