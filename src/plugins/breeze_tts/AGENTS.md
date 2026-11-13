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
- The codec is `sdk/media/mimi_codec.zig` (shared with qwen3_tts): Kyutai Mimi, 16 codebooks
  @ 12.5 Hz — the RVQ decoder the frame loop streams through and the encoder that folds the
  reference clip into clone frames. Its geometry rides `audio_tokenizer/config.json`
  (`readCodecConfig` + `CodecConfig.supported`); the 1920-sample upsample ladder is the floor.
- A `Bundle` (one safetensors file + handle map, `openBundle`/`openBundleAt`) serves the decoder
  and the encoder from one weight load; every load path that opens one `defer b.close()` — the
  parse phase duplicates key strings and leaves nothing mapped after it. `openBundle` is the
  live-pack seam (published widths); the toy fixture shares the published upsampling at
  scaled widths and loads through the same path.
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
- The turn's stop is `Config.cbEos()` — the lm_head's LAST row (`audio_vocab_size`, the
  reference's `eos_token = vocab_size`). `audio_vocab_size - 1` is the PAD row: a stop
  compared there can never fire and every request runs to the frame cap (field report: 30 s
  of babble per line). The toy fixture's frame-0 EOS guard must leave the row otherwise
  noise-filled — a zeroed row is an unreachable EOS, the same failure mode in fixture form.
- `billRequest` prices the frame loop's host rows (stream + span + depth span + codec chain) from
  the pack's config and the body's frame budget, with margin; the engine refuses a request past
  the window at step time, this keeps admission honest.
- Every debug lever that shipped with the port was deleted; the only env vars left are the
  fixture-gated ones (`BREEZE_MODEL`, `MIMI_*`, registered in `src/env.zig`). A smaller model is
  a smaller `config.json`, never an env override.
- Shared math rides the sdk blocks where they fit the shape, composed chains where they do not:
  the frame loop (decode step + prefill) runs `sdk/blocks/rope/qk_norm_rope.fusedQkNormRope`,
  `sdk/blocks/norms/residual_norm.fusedAddRmsNorm` and `sdk/blocks/mlp/fused_activations.fusedSwiGLU`
  (all bit-identical to the composed chains, optional-fallback like qwen3_tts's; a `QLayer.eps_arr`
  0-d f32 gates the norm kernels). The TE keeps its composed chain — its per-layer rope position
  scale has no fused arm. The decode step's `concat2` aliases kr/vt into the kv entry on the first
  frame, so those frees sit at the loop tail, never in a `defer`.

## Gaps

- `conformanceFixture` ships a toy pack (2/4/2 layers at 64/32 hiddens, a 256-row TE table, a real
  codebook ladder) whose config.json states its geometry — the same code path the published pack
  takes. The published pack itself has no oracle end-to-end yet: the Mimi encoder parity test is
  the only numerical bar. `fixtures/mimi_clip.wav` + `fixtures/mimi_codes.bin` are that oracle:
  u16-LE codes from the `qwen_tts` reference tokenizer pointed at a real pack's `audio_tokenizer/`;
  the parity test defaults to these, `MLX_SERVE_MIMI_ENCODE` must still name a real pack (the
  encoder is real-sized). Regenerate with `qwen_tts.Qwen3TTSTokenizer.encode` on an exact multiple
  of 1920 samples — partial trailing frames floor on this side and ceil in torch.
- Chunked PCM out is not in the media ABI: `encode` hands the host one WAV. Real-time consumers
  (fleet's WS lane) currently re-request short turns; a `step`-side PCM chunk callback would make the
  media ABI cover streaming speech.
