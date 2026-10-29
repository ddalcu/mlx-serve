# nemotron_diarization

NVIDIA Nemotron 3 Diarization (mlx-audio layout; mlx-community bf16 + 8-bit packs): who speaks when, up to eight
speakers in order of first speech, one probability per speaker every 10 ms. NeMo log-mels, eight to a row, into a
31-layer RoPE encoder; a sub-pixel convolution returns to 10 ms.

## Files
- `plugin.zig`: the media table (`.diarization`: the clip in, the speaker activity out) and the tiny conformance pack.
- `config.zig`: claims (`model_type: nemotron_diarization`), no route, bills (weights resident).
- `diarization.zig`: config parse (refuses by name what the port lacks), log-mel, encoder, the chunked stream with
  the FIFO and the Arrival-Order Speaker Cache, `aoscKeep` (the cache compression, on the host).

## Job contract
No route of its own: `diarized_json` on `/v1/audio/transcriptions` loads it (capability `diarization`) and runs it ONCE
over the whole clip. `inputs[0]` = f32 mono PCM at 16 kHz (native byte order); the result = speaker count (u32
little-endian), then f32 probabilities `[frames][speakers]`. The host builds `diarized_json` from it.

## Rules
- One pass over the whole clip, never per 180 s ASR piece: a voice keeps its letter across pieces only that way.
- 27.2 s chunks attend to the FIFO and the speaker cache; a compressed cache is out of order, so only its stored
  probabilities describe it. Among equal scores the earlier frame wins (`aoscKeep`, pinned by the official cases).
- The final sigmoid runs f32: the cache scores tell close frames apart.

## Fixtures
- `conformanceFixture`: one layer at 16, 8 mels stacked two to a row, two speakers, a cache small enough to compress.
- Reference oracle: `tests/dump_nemotron_diarization_fixtures.py` (mlx-audio and transformers' official model) →
  `NEMOTRON_DIARIZATION_MODEL` / `_FIXTURE` env-gated test in `diarization.zig`.
