# bert

Owns `bert`: BERT encoders as mlx-community converts them (MiniLM, bge-, mxbai-embed). Served on `/v1/embeddings`.

## Files
- `plugin.zig`: the table. `Encoder(BertEmbed, BidirAttention, GeluMlp, LayerNorm)`: word + position + token-type
  (always 0) rows under a LayerNorm, post-norm layers, every projection with its additive `.bias`. The tanh GELU runs
  through ONE closure compiled at load (`compiled.compile`, shapeless), as the engine's `compileGelu`.
- `config.zig`: MLX-free claim + parse (pooling contract: config `pooling_mode`, then `1_Pooling/config.json`).
- `proof_test.zig`: bit-equal to the deleted legacy `forwardBertWith`, replayed from `fixtures/*.tape` (4-bit and
  2-bit group 128).

## Facts the table declares
- `caps.embed`: prefill only (decode, verify, batched rows refused `unsupported`); the forward returns the hidden rows
  `[1, T, hidden]`, no logits; no KV (`layout` = no cache per layer), prefix cache off.
- Pooling: explicit `pooling_mode` > the sentence-transformers sidecar (booleans or ST 6's string, `lasttoken`); a
  sidecar declaring only modes we do not implement FAILS the parse, never a silent mean-pool. Unset = the host's
  name fallback (`bge-`/`mxbai-embed` → cls), else mean.
- Parse declines (`unsupported`): a dense pack (the engine read every BERT tensor quantized),
  `quantization_config.quant_method`, a layer span, `text_config`/`modules`/`expert_quant`. A 2-bit group-128 pack
  runs stock kernels (the engine's BERT load never armed the ternary ones). A declined pack is refused by name
  (`served.legacy_removed`).

## Rules
- None moved from root AGENTS.md: WordPiece's `max_input_chars_per_word` and embedding sub-batches are host rules.

## Known issues
- At the proof's sizes the uncompiled GELU chain is also bit-equal to the compiled closure, so the proof does not
  discriminate the closure; it is carried because compiling is not output-preserving in general.

## Served

In `served.zig`; the shell runs on the legacy parse's `ModelConfig` (pooling, encoder-only caps), so a pack whose
folder name picks the pooling (`bge-`, `mxbai-embed`) serves under that config's pooling too (`bindServed` keeps only
a decoder's name-picked embedding arm legacy). A padded batch rides `Forward.key_mask` (proof: a two-sequence padded
batch and one unmasked sequence through the driver bit-equal to `forwardBertWith`). The legacy forward and load are
deleted; the legacy config parse stays.

## Fixtures
- `conformanceFixture`: one 4-bit layer, width 64. Proof packs: two layers, width 128, 4-bit and 2-bit g128, a cls
  pooling contract; tapes in `fixtures/`.
  Integration: `tests/test_embeddings.sh`.
