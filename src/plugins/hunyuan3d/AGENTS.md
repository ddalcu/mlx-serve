# hunyuan3d — Tencent Hunyuan3D-2.1: one photo to a mesh GLB, textured on request

Shape: DINOv2-L conditioner, a 3.3B flow DiT (U-ViT skips, the timestep as token 0, six MoE blocks), the ShapeVAE
decoder whose geometry head answers SDF queries, a hierarchical volume decode, marching cubes. Paint (`texture: true`):
SD2's VAE, DINOv2-giant, the 2.5D dual UNet over six fixed views, bake to an atlas. Weights arrive pre-converted
(`tests/convert_hunyuan3d_weights.py`): `config.json` (`hunyuan3d_2_1`) beside `conditioner`, `dit`, `vae.safetensors`.

## Files

| File | Owns |
|---|---|
| `config.zig` | MLX-free: claims (any `hunyuan3d*` type), caps, the load bill |
| `plugin.zig` | the media table: fields, the paint folder (`paintDir`), the shape then paint job, the tiny pack |
| `hunyuan3d.zig` | the shape engine: conditioner, DiT (+ its int8 route), ShapeVAE, volume decode |
| `paint.zig` / `paint_unet.zig` | the paint engine: VAE (blocks on `sdk/blocks/vae`), DINO-giant, the 2.5D UNet |

## Rules

- Converted layouts BAKE OUT the reference's per-head QKV interleaves: never "fix" them back.
- The dense dtype is FLOAT16, not bf16 (the source checkpoints are fp16); timestep features, the scheduler step, the
  SDF grid and marching cubes stay f32. `MixedLinear` reads (bits, group) from each tensor's geometry.
- The paint stage is a folder of its own: `HY3D_PAINT_DIR` (the only candidate when set), `<pack>/paint`, or the
  converter's sibling `hunyuan3d-2-1-paint-8bit`; it loads per request and is freed after. `texture: true` without it
  is a named refusal, never a silent untextured mesh.
- The shape stage reads its grid on the host, so a Metal failure inside it hands back zeros with no error: the job
  checks the MLX latch before minutes of paint, and the host after `step`.
- The paint stage decimates to 40k faces before xatlas (`PAINT_MAX_FACES`; a raw marching-cubes mesh unwrapped for
  tens of minutes) and keeps the one-thread `parametrize`.
- The DiT's linears run int8 on NAX under `--w8a8` (LOSSY), replacing the packed weights block by block; the bill
  is `pack_bytes.residentWeights` (`dit.safetensors` at its int8 size).
- MLX's CPU `gather_mm` hangs on f16 (`gather_qmm` runs): the tiny pack's experts are 8-bit, as shipped.

## Proofs

- Tiny pack (`conformanceFixture`): the shape engine at 32 wide (DINOv2 one layer, DiT depth 3 with one 8-bit MoE
  block, ShapeVAE one block); the job meshes at octree 64. Goldens (`MLX_SERVE_GOLDEN_DIR=<dir> zig build test
  -Dtest-filter=hunyuan3d`), all saved on the code before the sdk port and bit-equal after it: the conditioning,
  latent and SDF grid; a 32-channel SD VAE's encode and decode (`hy3d-paint`); the UNet's norm, convs, GELU and
  timestep features.
- Reference oracles (env-gated `HY3D_*` / `HY3DP_*`, `tests/dump_hunyuan3d_fixtures.py`).

## Known issues

- No tiny paint pack: the 2.5D UNet hardcodes SD2's geometry (320..1280 channels, 77 x 1024 text), so `texture: true`
  is covered by its block goldens and the env-gated oracles, not by a job on a pack.
- A sibling paint folder is outside the pack, so its ~4.6 GB are not in the load bill (as before).
- `steps: 1` passes the field check and fails in `buildSigmas` (needs 2), as before.
