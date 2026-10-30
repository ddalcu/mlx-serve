# flux2 — FLUX.2 klein 4B / 9B and the undistilled base 9B (black-forest-labs, mflux's MLX layout)

Text to image, variation (SDEdit renoise at `strength`) and instruction edits with up to four in-context
references. Qwen3 encoder (36 layers, hidden states after layers 9/18/27 side by side), double-stream then
single-stream DiT, FlowMatchEuler, the 32-channel VAE with its encoder. Port of mflux.

## Files

| File | Owns |
|---|---|
| `config.zig` | MLX-free: claims (`flux2*`, or a configless pack whose `transformer/` names `double_stream_modulation_img`), caps, bills |
| `plugin.zig` | the media table: the job (body via `sdk/media/image_request`, upload decode + edit geometry), LoRA, PNG, the tiny pack |
| `engine.zig` | the loaded pipeline: low-memory encoder policy, the Qwen3 prompt template padded to 512, CFG, variation/edit wiring |
| `flux.zig` | weights (affine linears, geometry read off the checkpoint), encoder on `sdk/blocks/text_encoders/qwen3`, DiT on `sdk/bases/dit` (`Hooks`), VAE on `sdk/blocks/vae`, schedule, the pipeline |

## Rules

- The 4B and 9B differ in six numbers (blocks, heads, joint width, encoder hidden and MLP); they are read off the
  WEIGHT SHAPES (`ditConfigFrom`, `teConfigFrom`), never assumed: the 9B mirror ships no config.json, and a
  hardcoded 4B count against a 9B DiT loads without an error and draws a plausible wrong image. Encoder heads come
  from `q_proj` rows, not hidden/head_dim (20 on the 4B, wrong).
- Bits and group size come off each linear's geometry (`inferQuantGeometry`, group 64 preferred): one loader serves
  the 3/4/6/8-bit builds.
- Real CFG only for the undistilled base: `guidance_scale` 1.0 (the default) is ONE forward per step and
  byte-identical to the no-CFG path; anything else encodes the negative prompt (default "") and runs two.
- The DiT's joint sequence is [text, image]; double blocks split it, run both streams, and join it again.
  Reference tokens ride after the latents at t = 10·(i+1) on their own 0-based grid and are never noised.
- Edits keep each reference's own aspect (~1 MP, /32, never upscaled); no `size` = the primary's resolution
  (`resolveEditTargetSize`), an explicit size is honoured as asked within 1536. Variation covers the output grid.
- Low-memory mode (the host's `Machine.low_mem`: iOS, `MLXSERVE_LOWMEM=1`, ≤ 16 GB) loads the encoder per request, frees it
  after the MATERIALIZED prompt encode (`encodePrompt` evaluates: a lazy graph would pin the weights).
- `--w8a8` converts the block linears (not modulation/embedders) and drops each block's packed arrays as it goes;
  the bill reads the DiT at its int8 size (`pack_bytes.residentWeights(.., converts = true)`).
- LoRA arch `flux2`: diffusers, BFL and kohya spellings resolve onto `transformer_blocks.N.attn.to_q` etc.,
  stacked and summed at forward (`lora.deltaSum`), never merged.

## Proofs

- Tiny pack (`conformanceFixture`): 27-layer encoder at hidden 64, one double + one single block (inner 128), the
  64-channel VAE with its encoder; the conformance suite runs a 256x256 job.
- Blocks moved onto the sdk bit-equal: `MLX_SERVE_GOLDEN_DIR=<dir> zig build test -Dtest-filter=flux2` saved
  CFG + rebalance, an edit and a variation on the old code; the port compares bit for bit.
- Reference oracles (env-gated, `FLUX_TEST_MODEL` + mflux dumps): encoder `prompt_embeds`, one DiT forward, VAE
  decode and encoder round trips, the 4-step pipeline, the w8a8 route's bar.

## Known issues

- `flux.zig` loads weights through the engine's `model.zig` (lint `media-plugin-imports-engine`) and the plugin
  reads the host tokenizer, as every media plugin does today.
- `steps: 1` now runs the full-sigma step: the old schedule divided by `steps - 1` and drew NaN.
