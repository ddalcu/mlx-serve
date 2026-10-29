# ltx

LTX video (2.3 and 2.5 by `model_version`; one-stage, two-stage, HQ, i2v, a2vid) + its audio VAE / BigVGAN, and the
LTX-2.5 DiffVAE decoder (`"decoder":"diffusion"`).

## Files
- `plugin.zig`: the media table (request parse, keyframes, a2vid, decoder choice, the clip as JSON members), the tiny
  conformance pack (a 2.5 layout: DiT, connector and decoder at 64/32 widths, a two-layer Gemma-3 in the encoder slot).
- `config.zig`: claims (`AudioVideo` + `connector.safetensors`), the resident bill (one variant, plus the 2.3 encoder).
- `engine.zig`: the loaded pack (variant swap, lazy upsampler/DiffVAE, LoRA stack, `resolveGemmaDir`), prompt padding,
  guider defaults, the a2vid gates; the legacy handler shares it until the flip.
- `video.zig` / `audio.zig`: the pipeline, the text encoders, the DiT, the VAEs. Geometry is `LtxConfig`: the released
  packs omit the `transformer`/`connector` sections and keep the defaults.
- `diffvae.zig` / `diffvae_forward.zig` / `diffvae_kernel.zig`: DiffVAE geometry and tiling, the MLX pass, the fused
  3D neighborhood-attention Metal kernel (`kernels/`).
- `contract.zig`: the kernel contract registrations (`kernel_contract/registry.zig`).

## Rules
- A standalone-frame latent belongs in slot 0 ONLY (`keyframeMask` + `keyframePositions`); `keyframes_abs_pos_embedding`
  is parsed, never added.
- The RELEASE is a config field (`LtxVersion`): it decides the text encoder (2.5 runs the real gemma4).
- DiffVAE: constructor args are not its config; the sampler contract is MEASURED (x0, 1 step, timesteps x1000). NATTEN
  shifts its window inward at an edge (`naWindowStart`), never clamp-and-mask. Tile budget is a per-request memory
  decision (`tileTokensForMemory`).
- 4-bit affine on a video DiT is a QUALITY setting (the 8-bit mirror is the reference).
- The 2.3 text encoder lives OUTSIDE the pack (`LTX_GEMMA_DIR`, else the shared `mlx-community/gemma-3-12b-it-4bit`
  download): the bill reads it from the filesystem and the engine resolves it at load.
- Every request field the pack cannot honor is a NAMED refusal (two-stage pieces, a2vid, last frame, DiffVAE); only a
  missing first-frame encoder degrades to text-to-video, as it always has.

## Known issues (kernel contract)
- `ltx_na3d_*` (gate 1, nonfinite): a +Inf score (an Inf in K) returns that key's value: the online softmax's
  new-maximum branch sets the running max to +Inf and never computes exp(Inf - Inf); float64 softmax and the
  reference's torch softmax give NaN. Only an activation overflow reaches it.
