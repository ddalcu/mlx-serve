# lfm3

Owns `lfm3` (LiquidAI LFM3-1.7B / -Base), `lfm3_vl` (LFM3-VL-2B: images and clips), `lfm3_audio` (LFM3-Audio-2B:
speech in and out) and `lfm3_omni` (LFM3-Omni-2B: all of it). Packs: `LiquidAI/LFM3-{1.7B,VL-2B,Audio-2B,Omni-2B}-MLX-
{8bit,bf16}`. Oracle: mlx-vlm's `lfm3`, `lfm3_vl`, `lfm3_audio` (Blaizzy/mlx-vlm merge efd4b9f1),
`tests/dump_lfm3_fixtures.py` (`fixtures/tiny`, `tiny_vl`, `tiny_audio`; `--codec`, `--speech` on a real pack).
Live: `tests/test_lfm3.sh`.

## Files
- `plugin.zig`: `Decoder(TableEmbed, RmsNorm, PostNormed(Either(GatedConv, DecoderAttention(.sliding, .gate=head))),
  PostNormed(GatedMlp), ResidualTail, LmHead)`; the post norms are the sandwich (`post_operator_norm`,
  `post_ffn_norm`), `embedding_norm` the final norm, the head tied.
- `config.zig`: MLX-free claim + parse (+ `layout`: conv layers keep a window, full and sliding layers KV), the
  tower's `Vision`, the speech encoder's `Audio`, the speech output's `Speech` (depth decoder, ids, sampling,
  the codec's rate from `audio_tokenizer/config.json`).
- `vision.zig`: the tower and projector. `audio.zig`: the speech encoder and projector. `trunk.zig`: the cache-free
  LFM trunk the encoder, the depth decoder and the codec share. `depth.zig`: the depth decoder (MaskGIT frames,
  frame rows). `codec.zig`: the speech codec (FSQ, trunk, iSTFT, `Stream`). `speech.zig`: the `SpeechHooks`.
  `media.zig`: the `MediaHooks` (prepare, tower, the placeholder run).
- `proof_test.zig`: the table, towers, depth decoder and frame rows against mlx-vlm on the tiny fixtures (CPU);
  `served_test.zig`: the pack through `model.parseConfig` and the driver, and the codec (env-gated).

## Facts the table declares
- Serial decode (the gated conv serves no batched rows), prefix cache, `exact_resumed`, `dflash_capture`. One
  recurrent entry per layer (bf16). Tool dialect `qwen_xml`, `<think>` tags, thinking ON by default only without a
  speech encoder (the text and VL cards).
- Weight prefix `language_model.model` (mlx-vlm's layout); the load also probes `model.language_model`, `model`.
- The legacy parse has no lfm3 arm: the engine config is the plugin's facts (`adapted`).
- VL and Omni: `caps.images` + `caps.video`; vision start/end ids default to the tokenizer's (Omni names none).
- Audio and Omni: `caps.audio` (encoder from lfm3_audio's `encoder` + `lfm` or Omni's `audio_encoder_config` +
  `text_config`), and `Parsed.speech` + `placeholder_ids.speech` (`<|audio_out|>`) only beside a codec.

## Rules
- Full layers rotate only `int(head_dim * partial_rotary_factor) / 2` pairs at the FULL head's progression
  (`theta^(2i / head_dim)`); the tail never turns: MLX rope with wavelengths and an infinite tail
  (`zeroTailWavelengths`). Sliding layers turn every pair at their own theta (1e4).
- The fixture's thetas are small on purpose: at 1e6 the tail pairs barely turn within a prompt, and dropping the
  zero tail passed the proof.
- Proof bar: bf16 activations vs the f32 reference, cos > 0.999 and no decided miss (mlx-vlm's own bf16 run sits
  below it); dropping the zero tail falls to 0.75.
- An image is Qwen3-VL's processor (smart resize, Pillow bicubic, mean/std 0.5, duplicated across the temporal
  patch); a clip is LFM3's own rule (`clipSize`: one size from the WHOLE clip's bounds, round-half-even snaps, the
  last frame repeated to fill a group). Proof: pixel values within 1e-5 of the reference.
- A clip's run is `<t seconds><|vision_start|>` + rows + `<|vision_end|>` per frame group, inside the template's own
  start/end; `t` is the mean of the group's first and last real frame, formatted as Python's `.1f` (the exact
  double, ties to even: 0.25 is "0.2"). A clip with no `fps` was sampled at the processor's (2.0).
- A speech clip is 16 kHz mono, right-padded to `400 + k * 160` samples, its log-mel UNCENTERED, floored at -12 (not
  Whisper's max - 8), `(x + 4) / 4`; one row per 8 frames. The encoder's attention reads the last 32 rows (window
  mask); SiLU and GELU run in f32 over bf16 activations. Proof: log-mel within 1e-3 of the f64 reference.
- The codec's two input projections run in f32 and round ONCE into its trunk, which amplifies its input's rounding
  (rounded twice: cos 0.996 against the f32 reference; once: 0.9989, the reference's own bf16 floor). A stream
  decodes each chunk over 20 frames of left context (the trunk's reach) and equals the whole decode.
- Greedy speech codes sit on exact bf16 ties (mlx-vlm's own f32 depth decoder picks otherwise), so a real-pack test
  compares the TEXT slots; the depth decoder's exactness is the tiny proof's. One loaded model keeps ONE conv state:
  a test running two prompts on it loads it per prompt.
- Tools with a spoken reply are a CARD fact (`SpeechFacts.tools`): LFM3-Audio documents the tools recipe, Omni's card
  answers tools in text, and under the Audio recipe Omni called `get_weather` for a joke.
- The card's recipes are the API's: voice chat `Respond with interleaved text and audio.` + `<|mixed_start|>`, TTS
  `Perform TTS in <language> using a <gender> voice.` + `<|audio_start|>`, ASR `Perform ASR in <Language>.`.

## Parse declines
- `use_sandwich_norm: false`, `conv_bias: true`, a rope type other than `default`, an unknown layer type, a layer
  span, a `quantization_config` naming a `quant_method`, a depth decoder with conv layers or a schedule that does not
  cover its codebooks.
