# stable_audio3

Stable Audio 3 small, text-to-audio on `/v1/audio/sound-generations` (the `sound` route): T5Gemma encoder, a 20-layer
DiT with memory tokens, the ping-pong rectified-flow sampler, the SAME-S decoder, 44.1 kHz stereo WAV. Loads the
official `stabilityai/stable-audio-3-small-*` repo as published (`model_config.json`, `t5gemma-b-b-ul2/`).

## Files
- `plugin.zig`: the media table over the engine, the tiny conformance pack.
- `config.zig`: claims (by type, or by `model_config.json` for the repo as published), caps (the sound route), bills.
- `stable_audio.zig`: encoder, DiT, sampler, decoder; GELU is sdk's `geluTanh` (`SQRT_2_PI_LOW`, the operand's dtype).

## Rules
- `seconds_total` conditions in WHOLE seconds >= 1 (`trainedSeconds`): a fraction is noise, in the reference too.
- Every activation is f32: the decoder's differential attention cancels catastrophically in f16.
- Parity: env-gated `SA3_*` oracles from `tests/dump_stable_audio_fixtures.py`; the served arm is pinned
  byte-for-byte against the legacy handler (`host/media_jobs/plugin/http.zig`).
