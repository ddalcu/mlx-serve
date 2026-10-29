# minimax_h3

MiniMax-H3 text-to-audio-video on `/v1/video/generations`: Qwen3-VL text encoder (+ tower for references), a 50-layer
joint audio/video flow DiT, a transformer video VAE decoder (24 fps) and an audio VAE (32 kHz stereo). Partitions by
the pack's declared `tasks` (the ONLY discriminator; the files are identical): FL2VA (keyframes, chained windows),
REF2VA (references), FastH3 Omni (REF2VA's 8-step PDD student with trained VSA sparse attention). Packs:
`ddalcu/MiniMax-H3-MLX-Serve-{4bit,8bit}`, converter `tests/convert_minimax_h3_weights.py`.

## Files
- `plugin.zig`: the media table (request parse, keyframes, references, residency plan, the clip as JSON members), the
  tiny conformance pack.
- `config.zig`: claims, `geometry()` from config.json, the bills (staged peak, per-request activations, residency).
- `geometry.zig`: MLX-free geometry and arithmetic shared by the bills and the forward (frame ladder, configs, speed).
- `h3.zig`: encoders, DiT, sampler, PDD, residency, `generate`. `vae.zig` / `audio.zig` / `vision.zig`: the VAEs and
  the vision presentation. `vsa.zig`: VSA. `glue.zig`: fused glue kernels. `cluster.zig`: the Thunderbolt step split.
- `contract.zig`: kernel contract registrations (qkv_prep, swiglu, VSA pool/assemble/simdgroup/NAX).

## Request contract
`prompt`, `width`/`height` (multiples of 32), `num_frames` (snapped UP to 17k+5, said so in the log), `steps` ([1,100];
a student runs exactly its own), `seed`, `turbo` (needs `turbo_lora.safetensors`, >= 4 steps), `fast` (default on),
`chain_windows` 1-6 (FL2VA only; `num_frames` per window), `first_frame_image` / `last_frame_image`, `ref_images` /
`ref_videos` (`{frames, audio}`) / `ref_audios` + `ref_image_size` (REF2VA only), `lora_paths`/`lora_scales`.
`cfg_scale`/`stg_scale`/`pipeline` are ignored, not refused (the app sends them for every video backend).

## Rules
- Every unusable field is a NAMED refusal: a silently dropped reference generates while ignoring the user.
- The condition stream is assembled in SEGMENT order by ONE resolver (`resolveRefs`): keyframe conds, then each
  reference block, then the target; request order lands a cond row on the wrong reference.
- A request is billed on its OWN sequence (`requestRows` x `ROW_BYTES` + the fast recipe's attention cache): it prices
  the resident plan, and the staged plan's DiT stage is refused by name before any stage loads (Metal OOM is
  uncatchable). Live RAM comes from `Request.machine`.
- A resident DiT skips the AdaLN precompute and keeps its adapters in the cache (`h3.Resident`): precompute frees the
  modulation weights for ONE schedule. Residency is all-or-nothing; another load releases it before refusing.
- Two reuse levers never stack: a forward after a step-cache skip recomputes attention (`attnBroadcastRefresh`).
- The converted layouts BAKE OUT per-head interleaves; a sparse-attention pattern cannot be tuned into correctness.
- FastH3 VSA: a per-block temporary of GBs is a fresh allocation in the server; pooling and assembly are kernels.

## Fixtures
- `conformanceFixture`: DiT at 64 (two 32-wide heads, one block + one refiner), text encoder at 64, VAE decoder at 32,
  byte tokenizer, no audio VAE (the clip is silent); 64x64 x 5 frames, 2 steps on CPU.
- Oracles: `tests/dump_minimax_h3_fixtures.py` (toy DiT block, layout, vision) in `h3.zig` tests.

## The row from the root architecture table

MiniMax-H3 text-to-audio-video: joint denoise, 17k+5 frame ladder, 24 fps, two partitions (fl2va/ref2va — `tasks` is
the ONLY discriminator), Turbo LoRA, chained windows, fast recipe default-on. FastH3 Omni = REF2VA's 8-step PDD
student: 32 stacked output heads fused per step from the config's `pdd_*` contract, trained VSA sparse attention
(`vsa.zig`)
