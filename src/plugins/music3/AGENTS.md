# music3

MiniMax Music 3 on `/v1/audio/music-generations`: a Qwen3-8B global LLM (AR at 25 Hz, batch 2 for CFG 1.5, top-k 50)
emits one semantic code per frame, a 0.6B depth decoder the 7 residual codes; the 8 per-frame hidden states feed a
condition encoder (softmax mix → Conv1d → nearest resample to 86.13 Hz) that conditions a 2.4B flow DiT, denoised in
200-frame windows (hop 100, overlap blended every Euler step, CFG 1.7 vs a ZERO condition); a Snake/DAC vocoder →
44.1 kHz stereo. Pack: `ddalcu/MiniMax-Music3-MLX-Serve-8bit` (`scripts/convert_music3_weights.py`).

## Files
- `plugin.zig`: the media table (the music job, WAV out) and the tiny conformance pack.
- `config.zig`: claims (`model_type: minimax_music3`, complete once `vocoder.safetensors` lands), bills.
- `music3.zig`: geometry parse (config.json sections), prompt byte contract, LLM + pruned head, depth decoder,
  condition encoder, DiT, vocoder, load-time fixups (weight-norm fusion, PT conv axes), the engine.

## Request contract
`prompt` (required), `lyrics` (required unless `instrumental`), `instrumental`, `bpm` [30, 300] and `keyscale` (both
become caption text, `captionWithFacts`), `duration_seconds` [1, 360] (an upper bound: the model may stop earlier),
`steps` [4, 100] (default 30), `seed`. ACE-Step fields (`timesignature`, `vocal_language`, `ref_audio`, `src_audio`,
a non-text2music `task`) are named refusals. The assembled prompt is capped at 5000 tokens.

## Rules
- NOT an ACE-Step variant: the timestep is a prepended TOKEN (plain LayerNorms, no AdaLN), SwiGLU is REVERSED
  (`value * silu(gate)`), Snake is alpha-only (no beta, no exp), the DiT RoPE is partial (rotary 32, θ 10000).
- A condition works only as training fed it: `instrumental` is the lyric TEXT `[Instrumental]` (the open weights have
  no flag) plus a caption clause; with real lyrics it is a named refusal (`plugin_job.instrumentalConflicts`).
- Probe laps (`MLX_SERVE_DIAG=music3-cost-probe`) before bandwidth arithmetic.
- The pruned semantic head (16385 rows) is the full head restricted to the legal outputs (row slices of a
  per-row-group affine weight are exact); a dense head falls back to the full vocabulary + mask.

## Shared blocks
- `sdk/blocks/vae/snake.snake` with the epsilon in α's dtype (vocoder).

## Levers
`MLX_SERVE_DIAG=music3-cost-probe` (a diagnostic; no serving lever).

## Fixtures
- `conformanceFixture`: one layer per stage at 32 over the real vocabulary layout, 4-bit head, vocoder 16 → 4.
- Reference oracles: `tests/dump_music3_fixtures.py` → `MUSIC3_*` env-gated tests in `music3.zig`.

## Rules (moved from the root AGENTS.md)

- **Music3 is NOT an ACE-Step variant** (timestep TOKEN, reversed SwiGLU, alpha-only Snake, hardcoded DiT RoPE). Probe laps (`MLX_SERVE_DIAG=music3-cost-probe`) before bandwidth arithmetic.
