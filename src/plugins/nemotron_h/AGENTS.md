# nemotron_h

Owns `nemotron_h`, `nemotron_h_puzzle` (the heterogeneous latent-MoE trunk) and `nemotron_h_omni` (Bolt / Super:
the trunk under `llm_config`, the RADIO tower beside it). Packs: mlx-community's Nemotron 3 Nano / 3.5 Bolt / Super.

## Files
- `plugin.zig`: the table. `Hybrid(TableEmbed, RmsNorm, Either(Either(Mamba2, HybridAttention), Either(NemotronMoe,
  ReluSq)), -, AddNorm, LmHead)`; the MoE folds its combine + shared expert into the closing add_norm seam (the
  latent trunk declines the fold, as legacy). Prefix `backbone` or `language_model.backbone` (probed).
- `config.zig`: MLX-free claim + parse (+ `layout`: Mamba2 = conv window + scan state, attention = NoPE full KV).
- `proof_test.zig`: tables bit-equal to the deleted legacy forward, replayed from `fixtures/` (harness
  `../lfm2/proof.zig`), with a drafter bound too (`Forward.row_exact`), where a verify also equals serial decode row
  for row; the kernel arms against their chains (fused add+norm, fused Mamba2 step) and the NoPE order bar.
- `mtp_head.zig` (+ `mtp_head_test.zig`): the MTP head behind `SpecHooks.head` (sevren-ai `mtp_head.safetensors`,
  TensorFold `mtp-4bit.safetensors`), bit-equal to the deleted legacy `MtpModel` nemotron arm (tapes); greedy and
  seeded MTP rounds equal the serial stream on the row-exact kernels.
- `radio_vision.zig`, `media.zig`: the engine's RADIO tower and its rows (legacy, behind media hooks).

## Facts the table declares
- `trunk_weight` hands drafters the table and head; `SpecHooks.head` serves the MTP head (`mtp_head.zig`).
- `exact_drafted`: with `Forward.row_exact` set every projection, the MoE router and experts and a 2..16-row window's
  attention take the row-exact lanes (`sdk/bases/blocks/row_exact.zig`, lane `rows` = `rowqmv`).
- Serial decode, prefix cache on, `exact_resumed` (test_hybrid_reuse_equivalence.sh), `dflash_capture` (every
  layer's output), `moe` when experts exist, SSD namespace `nemotron-h-nope-v1`.
- Spec: MTP = a bare-key `mtp_head.safetensors` (`mtp_layout = nemotron`), reading the FINAL-NORMED hidden, depth 2
  when auto. LiLiCorr rides DFlash.
- A 2-bit group-128 pack whose every bias is exactly -scale is ternary (`reader.armTernary`): the mixers' and the
  dense MLPs' 2-bit projections read qmv2's kernels (`Linear.ternary`); the MoE keeps stock kernels, as the engine
  runs it outside `qmatmul`.
- A `--mlx-gguf` file (its rebuilt config's `quantization.mode: gguf`) reads through `sdk/quant/gguf`'s table: a
  block-quantized tensor keeps its bytes beside its sentinel (`QuantMode.gguf`; projections, banks and the table run
  the gguf kernels, the row-exact lanes stay off), a float one loads bf16 and dense (`reader.ggufProj`).
- Parse declines (`unsupported`): a `quantization_config` with a `quant_method` (compressed-tensors FP8), a
  `layers_block_type` name it does not know, a layer span (the cluster refuses nemotron_h).

## Rules (moved from root AGENTS.md)
- **The MTP head reads the FINAL-NORMED hidden and pays per verify ROW** (`ModelConfig.mtpReadsFinalNorm`; more drafts kept at depth 1 and 2 than from the residual): every extra row routes to more experts, so round cost climbs with depth; depth is 1 or 2 per round (`mtpExactDepth`).
- Pattern from `hybrid_override_pattern` (string wins) OR `layers_block_type`; Bolt's `llm_config` omits
  `num_hidden_layers`: the block list is the depth. `time_step_limit` bounds: numbers or positive infinity only.
- Attention is NoPE whatever `rope_theta` the config carries, in every trunk layer and the MTP head.
- The MTP head pays per verify ROW: every extra row routes to more experts; depth is 1 or 2 per round.
- Row-exact decode makes drafted output byte-identical to serial ONLY while a DFlash drafter is bound; the serial
  reference is the same server with `"enable_drafter": false`.
- Mamba2 decode is ONE dispatch per layer; MoE decode reads experts in place (`gatherQmv` fc1 + ReLU², fc2), never
  the sort machinery; a single routing group takes the ungrouped fused router.
- Omni's native MTP config requires the RADIO final LayerNorm BEFORE pixel shuffle; projector quant is per linear.

## Known issues
- The RADIO tower (image and video, its final LayerNorm, EVS) stays the engine's: a served omni pack's tower loads into
  the engine's map (`driver.loadWith`) and its rows splice at `<image>`, a tokenizer id the driver hands the table from
  the engine config. The head reads its sidecar through the engine's `loadWeightsSingleFile` (a plugin has no file
  source of its own yet).
- `exact_drafted` holds only where every projection fits the row kernels (`row_exact.fits`, read at load into
  `Model.row_exact`); a pack outside them serves plain kernels with the flag set and is not exact, as on the legacy
  path (`firstRowExactMiss`, which the driver runs over the pack for the engine's `row_exact_covered`).
- At the proof shapes the row-exact router, the fused shared up+ReLU² and the row-by-row attention give the same bits
  as the composed ops; the projection and expert lanes are what the bar discriminates.
- The engine reads an unknown `layers_block_type` name as attention: the Nano fixture's `linear_attention` (its
  Mamba layers) loads as attention there; this plugin declines it.

## Served

`nemotron_h`, `nemotron_h_puzzle` and `nemotron_h_omni` in `served.zig`; the shell runs on the legacy parse's
`ModelConfig`. Proof: `proof_test.zig` servedCase (4-bit, a Puzzle latent-MoE pack, a bf16 list-pattern pack with a
tied head, a row-kernel pack with a DFlash drafter bound, an omni pack with media rows spliced across a chunk: steps
past a chunk and greedy decode bit-equal, `row_exact_covered` equal; the omni pack's tower in the engine's map; two
`--mlx-gguf` packs, MoE and dense, `ggufCase`, and two real GGUF files through the engine's GGUF loader, `ggufFileCase`) and
`mtp_head_test.zig` servedCase (both head layouts: the engine's head calls through `MtpHeadRef.plugin` bit-equal to the
`.qwen` sidecar arm, a pinned-depth round loop with equal tokens and accepted counts). The legacy forward, load, MoE
and the legacy `MtpModel`'s nemotron layout are deleted (`served.legacy_removed`; the engine's init refuses a hybrid
config by name, `loadMtp` a Nemotron head); the config parse stays.

## Fixtures
- `conformanceFixture`: `M*E-`, width 64, f32. Proof packs in `proof_test.zig`. Oracles: Super RADIO reference
  fixtures, LiLiCorr tests, `tests/test_nemotron_bolt.py`.

## The row from the root architecture table

Hybrid transformer + Mamba2 (`backbone` prefix; nested trunks probed); pattern from `hybrid_override_pattern` OR
`layers_block_type`; sigmoid/ReLU² MoE + shared expert, including latent full-hidden routing; attention is NoPE
(`layer_no_rope`). Puzzle/omni aliases, RADIO image/video and LiLiCorr via DFlash. MTP = a bare-key
`mtp_head.safetensors`, depth 2.
