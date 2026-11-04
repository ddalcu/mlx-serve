# acestep

ACE-Step v1.5 XL Turbo on `/v1/audio/music-generations`: Qwen3-Embedding prompt encoder (lyrics by table lookup),
condition encoder (lyric + timbre CLS + text projector), 32-layer AdaLN DiT with alternating sliding/full attention
and once-per-generation cross K/V, 8-step Euler flow match (shift 3) + DCW, Oobleck VAE → 48 kHz stereo, peak
normalized to −1 dBFS. Packs: `ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-{4bit,8bit}` (converted by
`tests/convert_acestep_weights.py`).

## Files
- `plugin.zig`: the media table (the music job, WAV out) and the tiny conformance pack.
- `config.zig`: claims (`model_type: acestep`, complete once `text_encoder/model.safetensors` lands), bills.
- `acestep.zig`: geometry parse (config.json + text_encoder/config.json), encoders, DiT, DCW, VAE, FSQ (cover),
  the ANE MLP seam (`--ane-audio`), the engine.

## Request contract
`prompt` (required), `lyrics`, `instrumental`, `vocal_language`, `bpm` [30, 300], `keyscale`, `timesignature`,
`duration_seconds` [10, 600], `seed`, `task` (`text2music` | `cover` | `complete`), `src_audio` (cover/complete source,
10-600 s, decides the length), `cover_strength` / `cover_noise_strength` [0, 1] (cover only), `track_classes`
(complete only, from `TRACK_NAMES`), `ref_audio` (style/timbre clip, ≥ 1 s). Every mismatch is a named refusal.

## Rules
- The timbre slot holds the silence latent OR the reference clip's VAE mean, never both.
- A TASK is the context stream + the instruction line (`complete`: the source latent; `cover`: the source through
  pooler → FSQ (soft clamp, THEN the hard grid) → detokenizer). `fsq.safetensors` is a separate dense bf16 file
  loaded on the first cover; a pack without it refuses cover by name.
- A condition works only as training fed it: `instrumental` is the TEXT `[Instrumental]`; with real lyrics it is a
  named refusal (`plugin_job.instrumentalConflicts`, shared with Music3).
- Snake and the VAE encode run f32 (exp headroom; a bf16 round trip of the source signal costs cos 0.9999 → 0.958).
- Low-mem (iOS, ≤ 17 GiB RAM, `MLXSERVE_LOWMEM`): the text encoder loads per request and the VAE decodes in 5 s
  windows; identical audio in the cores.
- Guards assert log lines, never output (the model is distilled and seed-stable only per machine).

## Shared blocks
`sdk/blocks/vae/snake.zig` (`snakeBetaLog`, on its f32 cast), `sdk/blocks/text_encoders/qwen3.zig` (`.fast`, the
encoder), `sdk/blocks/dit/timestep.zig` (`sinusoid`, f32, t x 1000), `sdk/blocks/schedulers/flow_match.zig`
(`timeShift`, f32).

## Fixtures
- `conformanceFixture`: DiT 64 x 2 layers, encoders at 32, one-layer Qwen3 text encoder, VAE 64 → 4 channels.
- Reference oracles: `tests/dump_acestep_fixtures.py` → `ACESTEP_*` env-gated tests in `acestep.zig`.
