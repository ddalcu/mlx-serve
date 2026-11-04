# krea — Krea-2-Turbo (krea, gated licence), ported from `avlp12/krea2_alis_mlx`

Text to image and variation. Qwen3-VL-4B text stack (12 tapped hidden states), single-stream MMDiT (28 adaLN
blocks behind a text-fusion stage), flow-match Euler, the Qwen-Image 3D causal VAE run per frame in f16. The pack
is assembled: `transformer_*.safetensors` + `text_encoder/` + `vae/` + `tokenizer/` + config.json `krea2_turbo`.

## Files

| File | Owns |
|---|---|
| `config.zig` | MLX-free: claims (`krea*`), caps (variation when `vae/` holds the encoder), bills |
| `plugin.zig` | the media table: the job (body via `sdk/media/image_request`), LoRA, PNG, the tiny pack |
| `krea.zig` | `MixedLinear` (bf16 or affine, bits/group from geometry), encoder on `sdk/blocks/text_encoders/qwen3` (`.table`, bf16 −1), DiT on `sdk/bases/dit` (`Hooks`, adaLN `.plus_one(f32)`), VAE on `sdk/blocks/vae` (RMS 1e-12), schedule, ANE MLP seam, the engine |

## Rules

- Widths come off the checkpoint (`KreaConfig.ditFrom` / `encoderFrom`: output rows, quantized or not); the
  sinusoid width, patch 2, 16 latent channels, RoPE θ and the 12 taps are fixed.
- Only the main blocks' attn + MLP linears are quantized in the release; every `MixedLinear` casts its input to
  bf16 first (the reference computes there). Encoder norm weights stay f32 (the late-layer outliers need it).
- The prompt template pads to 541 + a 5-token suffix; the first 34 rows (the system prefix) are dropped after the
  encoder. Pad text rows only ever meet a −1e9 mask, so they are dropped before the DiT (no mask); an all-pad
  prompt keeps the masked layout.
- No edit training and no negative path: `mode:"edit"` and `guidance_scale` are named 400s (`caps`).
- `--w8a8` fuses q/k/v/gate into one int8 weight (norm + RoPE in one kernel) and gate/up interleaved with SwiGLU in
  the epilogue; an adapter at a nonzero scale takes the projections apart. `--ane-image` keeps the DiT packed (the
  ANE seam reads it), and the bill follows (`converts = !ane_image`).
- The ANE seam calls the SwiGLU through `forwardRaw` (it owns no graph); a LoRA-attached block declines.

## Proofs

- Tiny pack (`conformanceFixture`): 36-layer encoder at hidden 64, two quantized blocks at features 128 with dense
  text fusion, the 32-channel VAE and its encoder; the conformance suite runs a 256x256 job.
- Moved onto sdk blocks bit-equal: `MLX_SERVE_GOLDEN_DIR=<dir> zig build test -Dtest-filter=krea` saved a
  rebalanced text-to-image run and a variation on the old code; the port compares bit for bit.
- Reference oracles (env-gated, `KREA_TEST_MODEL` + `tests/dump_krea_fixtures.py`): encoder, DiT velocity, VAE,
  encoder round trip, end to end, the w8a8 bar.

## Known issues

- The golden runs with `--w8a8` off on the CPU stream: the fused int8 heads and int8 attention are covered only by
  the NAX-gated SwiGLU test and the env-gated w8a8 bar, not by a bit-equal before/after run.
- `krea.zig` is past the 1500-line cap (lint `file-over-cap`, as before the port) and loads weights through the
  engine's `model.zig` (lint `media-plugin-imports-engine`).
