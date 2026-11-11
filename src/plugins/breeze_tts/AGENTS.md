# breeze_tts

Breeze-TTS-2 (`model_type: breeze_tts`) as a media plugin: `/v1/audio/speech` with instruction voice
design, `S0`..`S9` speaker slots, and zero-shot voice cloning (base64 reference WAV + its transcript).
24 kHz mono WAV out.

## Files

- `config.zig` — claims, caps, bills and the pack's `Config`: every width, id, theta and eps from
  `config.json` (defaults are the published Breeze-TTS-2 values, `layer_types` gives the TE's
  full-attention ladder); `supportedLayout` refuses anything the engine doesn't implement BY NAME
  (`error.UnsupportedBreezeGeometry`). MLX-free by rule; `billRequest` reads the same config through
  the PackDir seam. `required_file` gates discovery on the Mimi codec landing
  (`audio_tokenizer/model.safetensors`).
- `plugin.zig` — `MediaTable` wiring. Job body parsing, refuse-by-name for every unhonored field,
  base64/ref validation, the f32 chunk sink, WAV encode.
- `engine.zig` — the model: backbone (codebook 0 + text), depth decoder (codebooks 1..15), text
  encoder, instruction wrapping — all built from `Config` (layer counts, hiddens, head geometry,
  rope thetas, rms eps come from the pack; only the 16-codebook frame loop is pinned).
  Host-gathered audio/TE table boundaries; forwards run on-device; sampling keyed off a counter
  for seed determinism.
- `mimi.zig` — Kyutai Mimi codec, 16 codebooks @ 12.5 Hz: the RVQ decoder the frame loop streams
  through and the encoder that folds the reference clip into clone frames. Its own geometry rides
  `audio_tokenizer/config.json` (`readCodecConfig` + `CodecConfig.supported`); the 1920-sample
  upsample ladder is the floor. Self-contained (plugins do not import plugins); the low-level mlx
  helpers ride along. If a second codec consumer appears this moves to `src/sdk/media/` and
  `qwen3_tts` re-imports it. The codec is a separate file on purpose: qwen3_tts's copy lives
  inside `tts.zig`, and merging means promoting it whole, which is a qwen3_tts change with its
  own parity bar.
- A `Bundle` (one safetensors file + handle map, `openBundle`) serves the decoder and the encoder
  from one weight load; every load path that opens one `defer b.close()` — the parse phase
  duplicates key strings and leaves nothing mapped after it.
- Frame rows the engine returns are views into one flat buffer (`Synthesis.backing`); the caller
  frees it through `freeSynthesisFrames` and never the rows.

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
- `billRequest` prices the frame loop's host rows (stream + span + depth span + codec chain) from
  the pack's config and the body's frame budget, with margin; the engine refuses a request past
  the window at step time, this keeps admission honest.
- Every debug lever that shipped with the port was deleted; the only env vars left are the
  fixture-gated ones (`BREEZE_MODEL`, `MIMI_*`, registered in `src/env.zig`). A smaller model is
  a smaller `config.json`, never an env override.

## Gaps

- `conformanceFixture` ships a toy pack (2/4/2 layers at 64/32 hiddens, a 256-row TE table, a real
  codebook ladder) whose config.json states its geometry — the same code path the published pack
  takes. The published pack itself has no oracle end-to-end yet: the Mimi encoder parity test is
  the only numerical bar. `fixtures/mimi_clip.wav` + `fixtures/mimi_codes.bin` are that oracle:
  u16-LE codes from the `qwen_tts` reference tokenizer pointed at a real pack's `audio_tokenizer/`;
  `MLX_SERVE_MIMI_ENCODE` must still name a real pack (the encoder is real-sized). Regenerate with
  `qwen_tts.Qwen3TTSTokenizer.encode` on an exact multiple of 1920 samples — partial trailing
  frames floor on this side and ceil in torch.
- `fixtures/mimi_clip.wav` + `fixtures/mimi_codes.bin` are the encoder-parity oracle: u16-LE codes
  from the `qwen_tts` reference tokenizer pointed at a real pack's `audio_tokenizer/`. The parity
  test defaults to these; `MLX_SERVE_MIMI_ENCODE` must still name a real pack (the encoder is
  real-sized). Regenerate with `qwen_tts.Qwen3TTSTokenizer.encode` on an exact multiple of 1920
  samples — partial trailing frames floor on this side and ceil in torch.
- Chunked PCM out is not in the media ABI: `encode` hands the host one WAV. Real-time consumers
  (fleet's WS lane) currently re-request short turns; a `step`-side PCM chunk callback would make the
  media ABI cover streaming speech.
