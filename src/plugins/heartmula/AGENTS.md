# heartmula

HeartMuLa-oss-3B on `/v1/audio/music-generations` (a heartlib port): style tags + lyrics → a Llama-3.2-3B backbone
(batch 2 for CFG 3, top-k 50) samples codebook 0 of each 80 ms frame, a 3-layer depth decoder (head_dim 384, KV reset per
frame) codebooks 1..7; HeartCodec turns the codes into 48 kHz stereo: RVQ dequant → a two-stack AdaLN flow DiT (10 Euler
steps, CFG 1.25 vs a ZERO condition) in 29.76 s windows continued in context → a PReLU conv decoder per stereo band.
Packs: `ddalcu/HeartMuLa-3B-MLX-Serve-{8bit,4bit}` (`scripts/convert_heartmula_weights.py`, also `--bits 16`).

## Files
- `plugin.zig`: the media table (the music job, WAV out), the request contract, the tiny conformance pack.
- `config.zig`: claims (`model_type: heartmula`, complete once `codec.safetensors` lands), bills.
- `heartmula.zig`: config parse, prompt byte contract, backbone + depth loop + sampler, codec (RVQ, DiT, Euler,
  scalar decoder), decode windows and crossfade, the engine; tests and the env-gated oracles at the bottom.

## Request contract
`prompt` (tags, required), `lyrics` (required), `duration_seconds` [5, 360] (default 360, an upper bound: the model
ends the song), `temperature` (0, 5], `top_k` [1, 1000], `cfg_scale` [1, 10] (default 3), `steps` [4, 50] (codec
Euler, default 10), `seed`. `bpm`, `keyscale`, `timesignature`, `vocal_language`, `ref_audio`, `src_audio`,
`instrumental` and a non-text2music `task` are named 400s, as is a prompt + duration past the 8192 positions.

## Rules
- Names follow mlx-audio's Sesame CSM port; q/k rows stay torchtune-INTERLEAVED, so every RoPE is `traditional`
  (backbone and decoder with llama3-scaled freqs, the DiT at base 10000). `audio_head` is split into seven Linears.
- The unconditional CFG row replaces EVERY text embedding AND the reference-audio slot; one draw feeds both rows.
- The codec's CFG negative is an all-zero `mu`; `zero_cond_embedding1` is never reached (a window is always full).
- Window arithmetic is heartlib's float truncation (`songSamples`): `int(n / 12.5 * 48000)` is not `n * 3840`.
- The 4-bit pack (trunk 4-bit, depth + heads 8-bit) agrees with the fp32 greedy codes far less often than 8-bit
  (replay oracle); 8-bit matches bf16.
- Tags are joined with no space around a comma (`joinTags`, heartlib's README format), then lowercased.
- Training kept at most ONE short tag per category (paper Table 6: genre .95, timbre .5, gender .375, mood .325,
  instrument .25, scene .2, region .125, topic .1): a long tag list or several genres is off-distribution and the
  style stops following it. The app's templates and Enhance write that shape.

## Levers
None.

## Fixtures
- `conformanceFixture`: one layer per stage at 32 over a byte tokenizer, a 4-bit text table and codebook-0 head, a
  one-block-per-stack DiT at 16/32 and the published 1920-sample decoder at 1-32 channels.
- Reference oracles: `tests/dump_heartmula_fixtures.py` (fp32 CPU; `--replay-bf16` measures PyTorch bf16's own
  replay rate, the bar) → `HEARTMULA_TEST_MODEL` (a pack) + `HEARTMULA_FIXTURES`; rerun the built test binary.
- Live: `tests/test_heartmula_gen.sh`.
