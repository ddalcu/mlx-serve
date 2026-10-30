# qwen_image — Qwen-Image-2.1 text to image, variation and instruction edit (mflux PR #736's pure-MLX reference)

A Qwen3-VL-8B text encoder (MageFlow's, `mage_flow.TextEncoder`), a block-causal single-stream DiT (text causal over
text, the image block over everything: two SDPA calls, never a dense mask), a 64-channel /16 VAE, FlowMatchEuler
with a dynamic shift (or the checkpoint's own `sample_sigmas`), 40 steps, real CFG on request. The edit pack carries
the Qwen3-VL tower in `text_encoder/`; mlx-community's spelling ships no root config.json (`model_index.json` names
`QwenImage21Pipeline`; the 2.0 family's `QwenImagePipeline` is another architecture).

## Files

| File | Owns |
|---|---|
| `config.zig` | MLX-free: claims, caps (the tower decides `edit`), the load bill, the canvas bill, the edit plan and dims |
| `plugin.zig` | the media table: the job (`sdk/media/image_request`), the edit's memory check, PNG, the tiny pack |
| `qwen_image.zig` | DiT (blocks on `sdk/bases/dit.adaLnBlock`, `.scale_only`), prefix cache, VAE, engine |
| `edit.zig` | ti2i prompt, image-pad expansion, tower preprocessing, the joint VLM encode |

## Rules

- The text encoder is half the pack and idle for the whole denoise: past 3/4 of the working set it is STAGED (loaded
  per request, freed before the first DiT forward). The engine and the bill read ONE answer (`config.staged`).
- A full-resolution f32 VAE stage is BANDED, exact because everything but its 3x3 convs is per-pixel
  (`Stage.banded` + `Conv.forwardStrips`); MLX's peak counter does not see the whole-stage peak. Bar = the fixture
  oracle with bands forced. So the VAE resnets keep their own strip-aware convs (not `sdk/blocks/vae/resnet`).
- An edit's output follows the LAST condition image (its aspect at 1024², /32); a request size is a ceiling floored
  to /32, never an upscale (`config.editDims`). Up to 10 references; each ~1 MP one adds ~4096 joint tokens.
- An edit whose working set (references + target) runs past the working set hangs the first denoise step with no
  error: the job refuses it by name with the numbers (`config.editPlan`), and drops the prefix cache when only the
  uncached forward fits. A text-to-image canvas past live headroom is refused by name in the job.
- The edit prefix cache keeps the text + reference K/V of the first step; target-only steps reuse it. The block-causal
  segment walk (`segmentSdpaWalk`) equals the dense mask; two adjacent image blocks stay separate segments.
- Text tokens read the t=0 modulation row, image (target) tokens the sampled-t row; an edit's references read t=0.
- The fused RoPE equals the chain bit for bit (separate f32 roundings, `fp contract(off)`); the edit forward never
  takes it.

## Proofs

- Tiny pack (`conformanceFixture`): the 36-layer encoder at hidden 64, a 2-block DiT (hidden 64, two heads of 32,
  4-bit MLP), the VAE at 8/16 channels with its encoder. Goldens (`MLX_SERVE_GOLDEN_DIR=<dir> zig build test
  -Dtest-filter=qwen_image` / `QwenImage edit forward`): CFG text to image plus a variation on the pack, and the
  edit DiT forward on the in-memory tiny DiT, saved on the old code and bit-equal after the sdk port.
- Kernel contract: `zig build test -Dtest-filter="kernel contract"` (f32 to bf16 and f16, gates 1-5).
- Reference oracles (env-gated, `tests/dump_qwen_image21_fixtures.py`): DiT, VAE, edit rope/geometry/forward/encode.

## Known issues

- The edit golden covers the DiT forward, not the tower (mage_flow's code) or the VAE `encodeRgba` (the variation
  golden covers `encode`, which shares every stage).
