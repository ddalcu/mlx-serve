# lfm2

Owns every `lfm2*` model_type (the engine's prefix rule): `lfm2` (LFM2 / LFM2.5 dense), `lfm2_moe` (LFM2.5-8B-A1B),
`lfm2_vl` (LFM2-VL). Packs: mlx-community's `LFM2.5-*` and LiquidAI's own bf16 repos. `d1` imports this trunk.

## Files
- `plugin.zig`: the table. `Hybrid(TableEmbed, RmsNorm, Either(GatedConv, HybridAttention), EitherMlp(GatedMlp, Moe),
  ResidualTail, LmHead)`; `embedding_norm` is the final norm; head = `lm_head` (nested or root), else the embedding.
- `config.zig`: MLX-free claim + parse (+ `layout`: conv layers keep a window, attention layers full KV).
- `reader.zig`, `json.zig`: the load's tensor reader and the config reader (shared with the other misc plugins).
- `proof.zig` (+ `proof_test.zig`): the proof harness (`Twin`, `hostLogits`, the `Gguf` file writer); the lfm2
  proofs replay tapes recorded against the deleted legacy forward (`fixtures/`).
- `vision.zig`, `media.zig`: the engine's tower and its rows (legacy, behind media hooks).

## Facts the table declares
- `trunk_weight` hands drafters the table and head.
- Serial decode (no block batches), prefix cache on, `exact_resumed` (test_hybrid_reuse_equivalence.sh),
  `dflash_capture`, `moe` on lfm2_moe. One recurrent entry per layer (bf16). Thinking and tool dialect come from the
  template probe. The weight prefix is probed (`language_model.model`, `model.language_model`, `model`).
- LFM2-VL (`lfm2_vl` with a `vision_config` object): `caps.images`, `placeholder_ids.image` from `image_token_id` /
  `image_token_index`; the engine runs the tower and hands its rows in `Forward.media`, the trunk splices them after
  the embedding (`Run.splice`).
- `lfm2-vl` (mlx-community's LFM2-VL-1.6B spelling) is claimed as TEXT: the engine sets its LFM2 tower only for
  `lfm2_vl`, so that pack never served images (a post-refactor fix).
- An LFM2-VL config with no image id parses with `placeholder_ids.image` 0: the driver hands the table the engine
  config's id (the tokenizer's `<image>`).
- A 2-bit group-128 pack whose every bias is exactly -scale is ternary (`reader.armTernary`, the engine's verdict over
  the loaded tensors): its 2-bit projections read qmv2's kernels (`Linear.ternary`), as the engine's `qmatmul`.
- A `--mlx-gguf` file (its rebuilt config's `quantization.mode: gguf`) reads through `sdk/quant/gguf`'s table: a
  block-quantized tensor keeps its bytes beside its sentinel (`QuantMode.gguf`; projections, banks and the table run
  the gguf kernels, the row-exact lanes stay off), a float one loads bf16 and dense (`reader.ggufProj`).
- Parse declines (`unsupported`): embedding packs, `quantization_config.quant_method`, `expert_quant`, a dense lfm2_moe pack (the
  engine fails it too), a layer span (the cluster refuses lfm2), more than 128 layers.

## Rules (moved from root AGENTS.md)
- Hybrid gated conv + attention; `lfm2_moe` = sparse MoE past `num_dense_layers` (sigmoid routing, selection-only
  `expert_bias`, zero when the pack ships none, no shared expert); dense MLP `w1/w3/w2` OR `gate/up/down_proj` (probed).
- A depthwise conv weight is `[C, K, 1]` (mlx-lm) or `[C, 1, K]` (LiquidAI's bf16 repos): a converter choice, read
  from the shape (`reader.depthwiseConv`), never the arch.
- Pythonic tool calls (LFM2.5): call-expression grammar; TYPES live in the SPELLING; marker-gated; truncation ships
  NAME + `{}`.
- A template can open `<think>` unconditionally (LFM2.5): whether a prompt ends inside a think block is a property of
  the RENDERED BYTES, never ANDed with `enable_thinking`; thinking-off appends `</think>`; LFM2.5-8B-A1B may open a
  bare `<think>` itself.
- A template that iterates any non-string `content` raises on `null` (LFM2-VL 1.6B): `EmptyContent.empty_string`.
- LFM2-VL: SigLIP2 NaFlex tower, patch features channel INNERMOST, two GELUs in one checkpoint, bilinear resample WITH
  antialias, tiling past `max_image_tokens x 2.0` into 512px tiles + thumbnail (`<|img_row_R_col_C|>`).

## Known issues
- A dense (bf16) lfm2_moe pack fails on the legacy path: the hybrid loader binds its expert banks untransposed and
  `gather_mm` refuses them at the first MoE layer. The plugin declines it (nothing to be equal to).
- The tower and its preprocessing (`vision.zig`, `media.zig`) stay the engine's: a served `lfm2_vl` pack's tower loads
  into the engine's map (`driver.loadWith`, both nestings), its rows reach the table as media rows.

## Served

`lfm2`, `lfm2_moe` and `lfm2_vl` in `served.zig`; the shell runs on the legacy parse's `ModelConfig`. Proof
(`servedCase`, the VL test): 4-bit, bf16 with torch-layout conv taps and an untied head, lfm2_moe, and LFM2-VL, steps
past a chunk (logits, the published hidden, every layer's capture), media rows spliced across a chunk boundary, and
the pipelined greedy decode bit-equal; two `--mlx-gguf` files (`ggufCase`: k-quant, legacy and i-quant blocks, a float
projection, a tied q6_k table; lfm2_moe's gguf banks and float router), and two real GGUF files through the engine's
GGUF loader (`ggufFileCase`, `proof.Gguf`), one addressed by the file, one by its folder. The published hidden is
the residual before the final norm (`hooks.residual_hidden`), as the legacy capture. A D1 decision pack
(`modeling_d1`) is claimed by d1 and refused on the LM path. The legacy forward and load are deleted
(`served.legacy_removed`, the engine's init refuses a hybrid config by name); the config parse stays.

## Fixtures
- `conformanceFixture`: conv + attention layer, width 64, f32. Proof packs: `proof_test.zig` (4-bit; bf16 nested +
  torch conv + untied head; lfm2_moe; an LFM2-VL image prompt over two chunks). VL oracle: `tests/test_lfm2_vision.sh`.

## The row from the root architecture table

Hybrid gated conv + attention; `lfm2_moe` = sparse MoE past `num_dense_layers` (sigmoid routing, selection-only
`expert_bias`, no shared expert); dense MLP `w1/w3/w2` OR `gate/up/down_proj` (probed); `lfm2_vl` = siglip2 tower +
projector, NaFlex 64-256 merged tokens, tiling past `max_image_tokens x 2.0` into 512px tiles + thumbnail
(`<|img_row_R_col_C|>`/`<|img_thumbnail|>`)
