# gemma3 plugin

Owns `gemma3` and `gemma3_text` (mlx-lm's layouts): the decoders, with SigLIP image rows spliced at
`image_token_index` (the tower stays in `sdk/vision`). EmbeddingGemma (`use_bidirectional_attention`) is the
`embedding_gemma` plugin's (claimed by marker). Served (`served.zig`): `gemma3`, `gemma3_text`.

## Files
- `config.zig`: MLX-free claim + parse; also the Gemma decoder `Config` (ModelConfig's names and defaults) and
  `readCommon`/`readGemma3`/`finish`/`readMedia`/`readPooling`/`encoderLayout`, which the other Gemma plugins share; `layout` (sliding/full per layer, KV source, head widths).
- `compiled.zig`: the compiled GELU / GeGLU / softcap closures and the model's GELU tables (the legacy compiles
  them at load on a GPU, so served bits come from them).
- `load.zig`: `Attention` = `DecoderAttention(.{ .sliding, .kv_sharing, .value_norm })` (sdk, with the fused
  hd-256 decode step); the engine's tensors through the pack's quant format, q|k|v and gate|up joined, (1 + w) norms,
  value-norm ones, the proportional rope table, per-layer geometry. Gemma 4 loads through it too.
- `plugin.zig`: `Decoder(TableEmbed, RmsNorm, PostNormed(Attention), PostNormed(GeGlu), ResidualTail,
  SoftcapHead)`; `proof_test.zig`: the bit-equality proof (shared helpers Gemma 4's proof imports).

## Facts the table declares
- `batches_decode` (from the base); thinking off; `caps.images` + `placeholder_ids` with a tower; stop set `{config eos} + {1, 106}` (also `tokenizer.extra_eos`); sampling
  defaults top_k 64 / top_p 0.95; `weights.prefix` `model` (flat) or `language_model.model` (nested).
- Exactness: drafted no, resumed unverified, concurrent no (`docs/known-issues.md`): none set.

## Rules (moved from root AGENTS.md)
- `*_text` siblings: accept the tag, collapse to the base type (`gemma3`), prefix by `text_config` presence, force
  `tie_word_embeddings` (the abliterated text-only builds ship no head and omit the flag).
- A silent config takes Gemma3TextConfig's head defaults (8 q heads, 4 KV heads, head_dim 256, issue #43), and the
  linear rope scale of 8 on global layers unless `rope_scaling.factor` is named.
- Gemma terminators merge additively: `<eos>` (1) and `<end_of_turn>` (106) join whatever the config declares.
- hd-256 branches key on 256 EXACTLY (decode: fused q/k norm+RoPE; prefill: the band/causal p256 kernels).
- The tanh GELU, GeGLU and softcap run COMPILED where a GPU exists; the GELU tables are the model's own (the
  engine keeps one per process, built by whichever model asks first).

## Known issues
- The engine's config of a served pack is the legacy parse with the binding attached; `bill` returns the defaults
  until the engine prices from the layout (P8).
- mlx-lm writes `quantization` twice (also as `quantization_config`): only a `quantization_config` naming a
  `quant_method` is declined (`foreignQuant`); `src/fixtures/model-configs` pins the released configs (`plugin/audit.zig`).
- A 2-bit group-128 pack may be ternary: every Gemma load (`plugin.zig`, `gemma4/plugin.zig`, `bidirectional.zig`)
  calls `load.armTernary` over what it loaded, as the engine's `firstNonTernaryBias` + qmv2 arm.
- `encoder.zig` is the legacy EmbeddingGemma forward (moved in P1); it goes with the legacy path.

## Fixtures
See `fixtures/README.md`.
