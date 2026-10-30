# yue2

YuE2-3B songs on `/v1/audio/music-generations`: ONE Qwen3-shaped trunk with AR and NAR weight sets per layer. ABC
score (planned or given) → semantic ids (host sampler, CFG) → flow-matching latents → tiled Oobleck decode,
48 kHz stereo. Pack = ahmadw/YuE2-3B-MLX's layout (`model.safetensors`, `vae.safetensors` with the decoder folded
in, `vae_config.json`, `qwen.tiktoken` via `tokenizer.loadTiktoken`); ours: `ddalcu/YuE2-3B-MLX-Serve-8bit`.

## Files
- `plugin.zig`: the media table (the music job, WAV out) and the tiny conformance pack.
- `config.zig`: claims (`model_type: yue2`, complete once `vae.safetensors` lands), music route, bills.
- `yue2.zig`: engine, prompt protocol, sampler, AR/NAR forwards, decoder; verbatim Scope oracle at the bottom.

## Request contract
`prompt` (style tags), `lyrics` (structure tags on their own lines), `cot` (`full` | `melody` | `off`), `abc` (an
external score: needs cot melody or full), `duration_seconds` [5, 360], `steps` [1, 100], `cfg_scale` [0, 20], `seed`.
ACE-Step's controls (`bpm`, `keyscale`, `timesignature`, `vocal_language`, `ref_audio`, `src_audio`, `task`,
`instrumental`) are refused by name. A request whose prompt and score leave no room in the 24576-token context is
refused before any forward.

## Rules
- A stage-wide scope keeps every temporary alive through its eval: `snake` and `resUnit` own short-lived scopes
  (a 1056-frame tile's Snake/conv temporaries are half a gigabyte each).
- This NAX-enabled MLX rounds f32 convs to TF32: the decoder oracle is cosine + a 1% scale band, never exact.
- Logits are judged by argmax and KL (bf16 moves the background one ulp); the NAR by cosine and `rms_ratio`.
- The decode tile size never moves the audio (halo covers the receptive field).

## Shared blocks
- `sdk/blocks/vae/snake.snakeBetaLog` (the decoder's Snake). Before/after on the tiny pack: bit-equal WAV.

## Known issues
- The planned ABC score rides the streamed `complete` event in the legacy path; the media `encode` carries the WAV
  only (no field for it in the table yet).

## Fixtures
- `conformanceFixture`: hidden 64, two layers, 4-bit `lm_head` + AR `q_proj`, byte-level tiktoken, strides [8,8,6,5].
- `Fixture` in `yue2.zig` (in-memory engine) pins the Graph port against the Scope oracle.
- Reference oracles: `tests/dump_yue2_fixtures.py` → `YUE2_*` env-gated tests; live: `tests/test_yue2_gen.sh`.
