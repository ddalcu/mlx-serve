# moss_image — MOSS-Image (OpenMOSS-Team/MOSS-Image-SFT / -Base, Apache-2.0)

Text to image: Qwen3.5-9B encoder (final hidden state, no head), 5B single-stream DiT (dim 3072, 24 heads, 42
layers), FLUX.2's small VAE decoder in f32. Port of the release's `pipeline_moss_image.py`; detail in
`docs/reference.md` "MOSS-Image".

## Files

| File | Owns |
|---|---|
| `config.zig` | MLX-free: claims (`moss_image`, or a configless pack whose `model_index.json` names `MOSSImagePipeline`), caps, load + per-request bills |
| `plugin.zig` | the media table: load on the host's stream, `set_loras`, the job (body via `sdk/media/image_request`), PNG encode, the tiny conformance pack |
| `moss_image.zig` | encoder sanitize + forward (our qwen3_5 Transformer), DiT on `sdk/bases/dit` (`Hooks`), VAE on `sdk/blocks/vae`, the snapped schedule, the pipeline |

## Rules

- No root config.json: discovery classifies on `model_index.json`; the pack is complete once
  `vae/diffusion_pytorch_model.safetensors` exists (`required_file`).
- The DiT is the sdk skeleton with `.folded` modulation (`1 + scale` folded once per step in `modulation`); the
  adapters ride the block linears through `linL` (LoRA arch `generic`, module names `layers.N.proj.q`).
- CFG 3 when the request names no `guidance_scale`; the negative branch skips block 22 (`uncond_skip`; a pack
  shallower than 23 blocks skips none). `guidance_scale: 1` is one forward per step.
- The schedule is the release scheduler's grid snapped to the 1000-step training grid with repeats dropped: a step
  count can come back shorter than asked (`stepsFor`).
- Text to image only: an `image` (variation) or `mode:"edit"` is a named 400 (`caps` declares neither).
- The VAE runs in f32; each up block's graph is released BEFORE its eval (holding it pinned every full-resolution
  intermediate: 25.5 GiB at 1024²).
- `--w8a8` leaves this DiT packed (the loaders do not convert it; `converts = false` in the bill).
- Bill: weights + `genTransientBytes(1024, 1024)` at load; the job refuses by name a canvas whose transient
  (`genTransientBytes(w, h)`: 1.5 MiB per latent token, at least a full prompt's 3 GiB) is past live headroom.

## Proofs

- Tiny pack (`conformanceFixture`): 2-layer Qwen3.5 encoder (full attention), 2-layer DiT, 4-level VAE; the
  conformance suite runs a 256x256 job on it.
- The DiT, VAE and Euler step moved onto sdk blocks bit-equal: `MLX_SERVE_GOLDEN_DIR=<dir> zig build test
  -Dtest-filter=moss_image` saves the tiny pack's CFG pipeline on the old code and compares bit for bit after.
- Reference oracle: `tests/dump_moss_image_fixtures.py` + `MOSS_IMAGE_TEST_MODEL`/`MOSS_IMAGE_FIXTURE`
  (cosine and rms ratio for the encoder, DiT with and without the skipped block, VAE and a 4-step pipeline).

## Known issues

- `moss_image.zig` still imports the engine (`sdk/config/model.zig`, `transformer.zig`) for weight loading and
  the Qwen3.5 encoder forward (lint `media-plugin-imports-engine`) until the qwen3_5 plugin exports its blocks.
