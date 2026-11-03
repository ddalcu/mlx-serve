# embedding_gemma2 plugin

Owns `embedding_gemma2` (the release's layout): the Gemma 4 text trunk trained as a bidirectional encoder, served on
`/v1/embeddings`; images and video frames ride `messages` as media rows spliced at `image_token_id`/`video_token_id`.

## Files
- `config.zig`: MLX-free claim + parse (Gemma 4 traits under `language_model`, `per_layer_config` geometry, the
  card's 8192 window, the text config's stop set without chat terminators, placeholder ids).
- `plugin.zig`: the table over `gemma4/bidirectional.zig` (projection-only per-layer inputs, layer scalars, the
  inclusive band floored under the key mask), `EmbedHooks.project` = `embedding_projection`; `conformanceFixture` =
  the release's tiny checkpoint (`fixtures/embedding_gemma2_tiny*`).
- `proof_test.zig`: 4-bit and bf16 packs through a served shell, one sequence / padded batch / image rows / head /
  vectors bit-equal to tapes recorded from the legacy path (`fixtures/`).
- `generate.zig`: the transformers-reference tests (tiny checkpoint, towers) on the served model. The legacy forward
  is deleted: a pack the plugin declines is refused by name.

## Facts the table declares
- `caps.embed` (prefill only), `images` + `video` with a `vision_config`, no prefix cache, no KV, `has_projection`.

## Rules (moved from root AGENTS.md)
- The window is the INCLUSIVE radius of a symmetric band (`band = sliding_window + 1` keys exclusive).
- Per-layer inputs read the MERGED embeddings: media rows splice before `ple.per_layer_model_projection`.

## Known issues
- Audio is not served (as the engine).

## The row from the root architecture table

EmbeddingGemma 2: Gemma 4 trunk trained bidirectional projection-only PLE, INCLUSIVE band, head in the dense0 slot;
images (Gemma 4 SigLIP tower, budgeted size) and video FRAMES ride `messages` on `/v1/embeddings`; no audio.
