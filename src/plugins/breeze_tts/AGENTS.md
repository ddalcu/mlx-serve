# breeze_tts

Breeze-TTS-2 (`model_type: breeze_tts`) as a media plugin: `/v1/audio/speech` with instruction voice
design, `S0`..`S9` speaker slots, and zero-shot voice cloning (base64 reference WAV + its transcript).
24 kHz mono WAV out.

## Files

- `config.zig` — claims, caps, bills. MLX-free by rule. `required_file` gates discovery on the Mimi
  codec landing (`audio_tokenizer/model.safetensors`).
- `plugin.zig` — `MediaTable` wiring. Job body parsing, refuse-by-name for every unhonored field,
  base64/ref validation, the f32 chunk sink, WAV encode.
- `engine.zig` — the model: Qwen3-1B backbone (codebook 0 + text), llama-100M depth decoder
  (codebooks 1..15), T5Gemma2 text encoder, instruction wrapping. Host-gathered audio/TE table
  boundaries; forwards run on-device; sampling keyed off a counter for seed determinism.
- `mimi.zig` — Kyutai Mimi codec, 16 codebooks @ 12.5 Hz: the RVQ decoder the frame loop streams
  through and the encoder that folds the reference clip into clone frames. Self-contained (plugins do
  not import plugins); the low-level mlx helpers ride along. If a second codec consumer appears this
  moves to `src/sdk/media/` and `qwen3_tts` re-imports it.

## Rules this plugin honors

- MLX runs on the stream the host handed `LoadCtx.stream`; the engine never creates a GPU stream
  (`Engine.load` takes `s: S` and pins it into the file-local `g_stream`). `g_stream`/`mlx_allocator`
  are plugin-file-private state, only ever set by this plugin's own `load`.
- Weights load through `sdk/weights/load.zig` (CPU-stream internals); tokenizer through
  `sdk/tokenizer/load.zig`.
- `step` calls `progress.boundary()` before every frame via `StreamSpec.cancel` — that is where a
  cancel lands (`error.Cancelled` → 499) and where chat requests get the GPU back.
- Unknown body fields are refused by name (`'language' is not supported...`), never ignored. Host-
  consumed keys (`model`, `stream`, `response_format`, `speed`) pass through; `language` is refused
  because Breeze detects it itself.
- `bill_request` returns 0 (all state is resident at load; the per-request working set is the frame
  history + codec decode, small against the load bill).

## Gaps

- `conformanceFixture` is a stub: the conformance suite needs a tiny converted pack (a 2-layer
  backbone via the convert script). Until then the suite reports `error.MissingFixture`.
- Chunked PCM out is not in the media ABI: `encode` hands the host one WAV. Real-time consumers
  (fleet's WS lane) currently re-request short turns; a `step`-side PCM chunk callback would make the
  media ABI cover streaming speech.
