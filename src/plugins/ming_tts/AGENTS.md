# ming_tts

Ming-3.0-tts-0.5b on `/v1/audio/speech`: a Qwen2.5 LM writes one hidden per step, a CFM flow head (10 Euler steps of an
8-layer DiT with CFG) turns it into a 64-d latent, an aggregator feeds it back, and a VAE decoder (Qwen2 + ISTFT) makes
44.1 kHz mono. CAM++ and the VAE encoder clone a reference. Packs: `mlx-community/Ming-3.0-tts-0.5B-8bit` (the default
download), `jorgepb96/Ming-3.0-tts-0.5B-4bit`, and bf16, the quality reference (`inclusionAI/Ming-3.0-tts-0.5b`,
`mlx-community/Ming-3.0-tts-0.5B-bf16`).

## Files
- `plugin.zig`: the media table, the request's fields, the tiny conformance pack.
- `config.zig`: claims (`model_type: dense` plus `ditar_config` and `aggregator_config`), bills, the geometry parse.
- `engine.zig`: tasks to samples: fragments, retries, joins, prompt audio. `text.zig`, `fr.zig`, `nfc_table.zig`: the
  frontend. `audio.zig`: resampler, fbank, trims. `campplus.zig`: CAM++ on the CPU stream.
- `model.zig`: the op chain, the pipelined decode loop. `fused.zig`, `kernels/`, `contract.zig`: the fused kernels.

## Request contract
`docs/api.md`, "Ming-3.0-tts speech". An unknown field, a field its task does not take, or a reference that cannot be
used is a named 400, never another voice.

## Rules
- CAM++ runs on the CPU stream: Metal fp32 convolutions drift enough to change a bf16 clone's trajectory.
- The decode loop stays pipelined one step ahead: it queues CFM for step k and the LM for k+1, then reads stop k. Noise
  is drawn only for steps that run, so the RNG stream matches a serial loop.
- A fragment is the job's step: `Progress.boundary` runs before each one. Chat waits at most one fragment.
- fr/de/es/ja/ko text without a `language` is detected and skips the zh/en
  frontend, which spells numbers in English and turns accented letters into punctuation.
- The `Target language` prompt line joins only TTS, BASIC and EMOTION: on captioned voices it moves the voice.
- Fused kernels run only at the release's geometry on a bf16 GPU stream; anything else takes the op chain.
- A quantized pack has the bf16 pack's tensor names plus `.scales` and `.biases`, and declares one affine scheme in
  `config.json`. `quantOf` checks each tensor against that scheme. `.bias` is additive and `.biases` holds the
  zero-points. `embed_tokens` and `word_embedder` are read through `gatherQuantizedRows`. Other modes are a named refusal.

## Known gaps
- A seed repeats a take only while no chat runs on the server. With chat alongside, the same seed gave a different length. The cause is not traced.
- No parity test against the reference in this tree; `tests/test_ming_tts.sh` proves the HTTP surface on a real pack.

## Levers
`MLX_SERVE_MING_FUSED=0`: the op chain, the control arm for the fused kernels.

## Fixtures
- `conformanceFixture`: every stack one layer at hidden 32, no CAM++ (a clone is refused by name), a stop head biased
  to end the job early. Config: `src/fixtures/model-configs/ming_tts-0.5b.json` is the release's.
- Live: `tests/test_ming_tts.sh`.
