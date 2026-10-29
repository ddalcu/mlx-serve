# kokoro

Kokoro-82M text-to-speech (hexgrad's StyleTTS 2 + iSTFTNet) on `/v1/audio/speech`: one non-autoregressive forward
per utterance window, 24 kHz mono WAV, f32 end to end. Packs: `ddalcu/Kokoro-82M-MLX-Serve` (`model.safetensors` +
`voices.safetensors` + `g2p/`).

## Files
- `plugin.zig`: the media table (load, the speech job, WAV out) and the tiny conformance pack.
- `config.zig`: claims (`model_type: kokoro`), caps (speech route), bills (both safetensors resident).
- `kokoro.zig`: the model (ALBERT, prosody predictor, decoder, iSTFTNet generator) and the engine.
- `g2p.zig`: text to IPA from the pack's lexicon (no espeak: GPLv3).

## Request contract
`input` (or `text`, up to 4096 characters), `voice` (one of the pack's voices, or a comma-separated blend averaged
over whole packs; default `af_heart`), `speed` in (0, 5], `seed`. Refused by name: `ref_audio` and `warm_only`
(Qwen3-TTS's cloning controls), an unknown voice, input with nothing speakable.

## Rules
- safetensors READS run on the CPU stream: mlx's Load has no GPU kernel and dies uncatchably.
- A voice is a `[510, 1, 256]` table indexed by the utterance's phoneme count; a fixed row degrades prosody silently.
- Depthwise `ConvTranspose1d` weights transpose `{0,2,1}`, not the groups=1 `{1,2,0}`.
- `AdaIN1d` is a pure instance norm then the style affine; `AdaLayerNorm` has no affine of its own.
- `Vocab.encode` DROPS unknown symbols, as the reference does.
- Reproducing an upstream NO-OP means NOT doing it: SineGen's random initial phase is never read by the reference's
  downsample; adding it here dropped waveform cosine from 0.99 to 0.48.
- `torch.round` is half-to-even (`roundHalfToEven`); `@round` drifts a frame.
- Input past the BERT context is spoken window by window (`phonemeWindow`), cut at sentence, clause, then word marks.

## Shared blocks
None: Kokoro's Snake (no epsilon, `square`) and its norms are other op sequences than `sdk/blocks`.

## Fixtures
- `conformanceFixture` (plugin.zig): a seeded random pack at hidden 16, one ALBERT layer, a two-stage generator.
- Reference oracles: `tests/dump_kokoro_fixtures.py` (the published model), gated tests in `kokoro.zig`.
