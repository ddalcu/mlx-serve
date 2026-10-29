# pixal3d — TencentARC/Pixal3D (MIT): one photo or 2-8 posed views to a textured GLB

TRELLIS.2's cascade and decoders (`../trellis2`), every flow also reading the photo pixel by pixel: each voxel token
is projected through a pinhole camera onto the DINOv3 patch map (past the sparse structure, onto its NAF upsampling),
and every DiT block adds a projection of what it lands on to its cross-attention output. Multi-view: posed photos
through the `_mv` checkpoints `pipeline_mv.json` names, each through its own camera, the features averaged.

## Files

| File | Owns |
|---|---|
| `config.zig` | MLX-free: claims (TRELLIS.2's `pipeline.json` name, told apart by `image_attn_mode: proj`), caps |
| `plugin.zig` | the media table: views, `fov`/`cameras` and TRELLIS.2's fields, the refusals, the tiny pack |
| `pixal3d.zig` | cameras and the default rig, view images, the conditioner, the engine over `trellis2/pipeline` |
| `naf.zig` | the NAF upsampler, evaluated only at the projected points |

## Rules

- The bills are TRELLIS.2's (`trellis2/config`): the same cascade and decoders; there is no 512 pipeline.
- `images` + `cameras` + `fov` exist only here; TRELLIS.2 refuses `images` by name. A pack the app downloads for one
  photo carries `pipeline_mv.json` without the `_mv` weights: multi-view is then absent and several views are
  refused by name (`viewsRefusal`), never a load failure.
- NAF is never run as a map: a 1024 x 1024 x 1024 upsampled map is gigabytes, and Pixal3D reads it only bilinearly
  at projected points (`naf.atPixels`). Its window is natten's: it shifts inward at the borders, never pads.
- A single view is cropped around its subject (`prepareImageMargin` at 1.1); multi-view images are never cropped (the
  framing is the camera's) and resize through Pillow's premultiplied round trip.
- The cascade cell rounds half to even as torch does (`roundCell`); the camera is a horizontal FOV from the request
  (the reference estimates it with MoGe-2, not ported).

## Proofs

- Tiny pack (`conformanceFixture`): TRELLIS.2's tiny pack with projecting flows plus a NAF of 8-wide encoders.
  Golden (`MLX_SERVE_GOLDEN_DIR=<dir> zig build test -Dtest-filter=pixal3d`): the decoded voxels at two steps,
  saved on the code before TRELLIS.2's sdk port and bit-equal after it.
- Reference oracles (env-gated `PIXAL3D_*`, `tests/dump_pixal3d_fixtures.py`) and `tests/test_pixal3d_3d.sh`.

## Known issues

- The tiny pack's job peaks near 12 GB RSS on the CPU stream: NAF's 256-wide guide at the fixed 1024 input size is
  the architecture's, whatever the pack's widths.
- The golden covers a single view; multi-view conditioning has only the env-gated oracle.

## The row from the root architecture table

TencentARC/Pixal3D (MIT): TRELLIS.2's `pipeline.json` name, told apart by its flows' `image_attn_mode: proj`
(`pack_claims`); `images` + `cameras` + `fov` only here. Pack adds `naf/` beside `dinov3/`
