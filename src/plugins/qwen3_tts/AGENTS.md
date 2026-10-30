# qwen3_tts

Qwen3-TTS (12 Hz) on `/v1/audio/speech`: a Qwen3 talker writes codebook 0 per frame, a 5-layer code predictor the
other 15, the speech tokenizer's decoder turns 16 codebooks into 24 kHz mono. `-Base` checkpoints ship an
ECAPA-TDNN speaker encoder for zero-shot cloning. Packs: `mlx-community/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base-{8bit,bf16}`.

## Files
- `plugin.zig`: the media table (speech and warm-up jobs) and the tiny conformance pack.
- `config.zig`: claims (`model_type: qwen3_tts`, complete once `speech_tokenizer/model.safetensors` lands), bills.
- `tts.zig`: talker, code predictor (compiled chain, KV-cached steps), speaker encoder + clone cache, codec decoder.

## Request contract
`input` (or `text`), `ref_audio` (base64 24 kHz mono PCM16 WAV: clones on a checkpoint with a speaker encoder; an
unusable one (bad base64, undecodable WAV, no speaker encoder) is a named 400, empty or absent = the plain voice), `warm_only` (embed + cache `ref_audio` without synthesizing;
answers `{"warmed":true,"cache":"hit"|"miss"}`), `language` (a `talker_config.codec_language_id` name, any case, dialects
excluded: the codec prefix becomes `[think, think_bos, <id>, think_eos]`; absent or `auto` = the auto prefix; anything
else a named 400). Generation stops at codec EOS or 2048 frames.

## Rules
- `TimeDelayNetBlock` = conv + IMPLICIT ReLU.
- Embedding lookups reshape to the TABLE's width (0.6B's text table is wider than its talker; the projection bridges).
- A clone voice is the ECAPA embedding as one codec-prefix position (the codec stream grows 5 → 6).
- The codec decoder's geometry is `speech_tokenizer/config.json`'s `decoder_config` (published values = defaults).

## Shared blocks
- `sdk/blocks/mlp/gelu.geluErf` (ConvNeXt GELU), `sdk/blocks/vae/snake.snakeBetaLog` (decoder SnakeBeta).

## Levers
Diag `tts-trace` only.

## Fixtures
- `conformanceFixture`: talker + predictor at hidden 32 (one layer each), byte-level tokenizer, decoder at latent
  16 / decoder 32; the codec head's EOS bias keeps a seeded job to a few frames.
- Live: `tests/test_tts.sh`, `test_tts_quant.sh`, `test_tts_clone_cache.sh`.
