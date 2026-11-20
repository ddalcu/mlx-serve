# minimax_h3_image

MiniMax-H3-Image text to image and multi-reference editing on `/v1/images/generations` (`mode:"edit"`, `image` +
`ref_images`, at most 16; the prompt names them `<Picture i>`). It is H3's DiT without the audio stream, run as a
six-step PDD student (one output head per step, its `pdd_sigmas` grid, velocity Euler, no CFG), H3's text encoder
unchanged (the tower presents each reference), and the release's 2D image VAE (H3's causal encoder; a 2D ViT decoder).
Pack: `ddalcu/MiniMax-H3-Image-MLX-Serve-4bit`, converter `tests/convert_minimax_h3_weights.py --partition image`.

## Files
- `plugin.zig`: the media table (request parse, reference decode, the canvas and its bill), the tiny conformance pack.
- `config.zig`: claims, `geometry()` (H3's plus `image_vae` and `pdd_sigmas`), canvas and reference sizes, the bill.
- `image.zig`: `generate` (references -> text encoder -> DiT -> VAE, each stage freed before the next).
  Everything else is `minimax_h3`'s: `h3.zig` (encoder, DiT, layout, PDD), `vae.zig` (`imageMoments`, `decodeImage`).

## Request contract
`prompt`, `size` (each side 16-6272), `seed`, `steps` (omit, or the student's own count), `min_generation_area`
(default 2048², [1024, 4096²]), `mode:"edit"` + `image` + `ref_images`. `guidance_scale`, LoRAs and variation refuse.

## Rules
- The release interleaves each attention's Q/K/V per head; the converter reorders them to H3's `[q | k | v]`. A
  pass-through pack runs and decodes noise on the 16-px latent grid (`test_convert_minimax_h3_image.py`, the live
  script's seam check).
- The layout is H3's REF2VA at one latent frame: `[text | each reference | target]`, references held at t = 0.999
  for every step; the reference label is `Picture i: ` (no brackets), unlike the video packs' `<Picture i>: `.
- A reference keeps its own size inside short-edge bounds [1024, 2048] (the tower sees it at 1024), cropped
  top-left to whole VAE cells, its latent to whole patches; its posterior is SAMPLED and rounded through f16.
- A request smaller than `min_generation_area` is generated on a larger canvas at its aspect and resized back as the
  release does: truncated to u8, then Pillow Lanczos (`qwen_vision.resizeInterleavedPil`).
- The DiT stage is priced on its own rows (`requestRows`) before any stage loads, as H3's staged plan is.

## Fixtures
- `conformanceFixture`: H3's tiny DiT without audio and two PDD heads, H3's tiny text encoder, a one-block image VAE
  decoder at 32; 64x64 at `min_generation_area` 1024, two steps on CPU. No encoder: edits run only live.
- Live: `tests/test_minimax_h3_image.sh`.
