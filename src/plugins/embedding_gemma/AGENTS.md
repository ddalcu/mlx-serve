# embedding_gemma plugin

Owns EmbeddingGemma: a `gemma3`/`gemma3_text` pack with `use_bidirectional_attention: true` (mlx-community's
layout), claimed by that marker (`by_marker` outranks the gemma3 plugin's claim by type). Served (`served.zig`) on `/v1/embeddings`; `EmbedHooks.project` is the served path's head.

## Files
- `config.zig`: MLX-free claim + parse (the Gemma 3 arm `gemma3/config.zig readGemma3`, the pooling contract
  `readPooling`: config `pooling_mode`, then `1_Pooling/config.json`; an unknown mode fails the parse).
- `plugin.zig`: the table over `gemma4/bidirectional.zig` (shared with EmbeddingGemma 2), `EmbedHooks.project` =
  the root `dense.0`/`dense.1` head; `conformanceFixture`.
- `proof_test.zig`: the hidden rows (table, padded batch under its key mask through the base) and the projection
  bit-equal to `forwardGemma3EncoderWith` / `embedProjection`. `Pair` is shared with EmbeddingGemma 2's proof.

## Facts the table declares
- `caps.embed` (prefill only), no prefix cache, no KV (`encoderLayout`), `embed.pooling`.
- `embed.has_projection` stays false at parse: the head rides inside `model.safetensors`; `project` returns the
  pooled rows unchanged when the pack has none.

## Known issues
- No floor under the band + key mask (the engine's): a padded query whose band holds only padding is NaN and, through
  V, so is its row in every later layer. A padded batch whose length gap exceeds `sliding_window` (512 on the
  release) returns NaN for the short row on BOTH paths; EmbeddingGemma 2 floors at -1e9. Fix both sides together.

## Fixtures
See `fixtures/README.md`.
