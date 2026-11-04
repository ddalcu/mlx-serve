# trellis2 — microsoft/TRELLIS.2-4B (MIT): one RGBA cutout to a textured PBR GLB

DINOv3 tokens (512, and 1024 for the cascade) condition three flow DiTs (one `Dit` for all: a dense 16^3 grid and a
sparse voxel list differ only in the RoPE coordinates): the sparse structure (decoded by a conv3d decoder to 64^3
occupancy pooled to 32^3), the shape latent (512, then a 1024/1536 cascade over the voxels its decoder upsamples to),
and the texture latent; two sparse SC-VAE decoders (the shape one predicts the subdivisions, the texture one follows
them) give the dual-grid head and the PBR attributes. The CPU half: dual grid, remesh, cleanup, xatlas, bake, GLB.
A pack has no root config.json: `pipeline.json` names it; Pixal3D keeps that name (`config.kind` reads its flows'
`image_attn_mode: proj`). No background remover: RMBG-2.0 is non-commercial, a cutout's alpha is the subject.

## Files

| File | Owns |
|---|---|
| `config.zig` | MLX-free: claims (`kind`, shared with Pixal3D), the load bill, the per-resolution request bill |
| `plugin.zig` | the media table: views, the body's fields (`parseOpts`, shared with Pixal3D), the refusals, the tiny pack |
| `pipeline.zig` | the stages (`generateVoxels` = GPU half, `meshToGlb` = CPU half), image prep, export axes |
| `trellis2.zig` | the flow DiT, sampler, sparse-structure decoder |
| `sparse.zig` | voxel sets, the submanifold conv, the SC-VAE decoder and its `Plan` |
| `mesh.zig` / `remesh.zig` | dual grid, cleanup, the volume sample and texels / the sign-free narrow-band remesh |

## Rules

- Every model loads for its own stage and is freed before the next: the bill is one DiT plus its activations,
  measured per resolution (`config.peakBytes`; 512 skips the cascade). The load bills the default request's peak
  and falls to 512 where the GPU cannot pass it and the pack ships the 512 texture flow; a request past the GPU's
  live headroom is refused by name (`plugin.resolutionRefusal`). The scheduler reserves the request's own peak
  (`billRequest`).
- The raw dual grid is NON-MANIFOLD by design (a fifth of the edges on 3+ faces): xatlas on it is thousands of charts
  and minutes, so it always goes through `remesh` first. The remesh skin sits 1-2 voxels off the decoded surface, so
  bake texels are `snapToSurface`d onto the RAW mesh before the volume read (else black stripes).
- xatlas segments one connected surface on ONE thread: `uvwrap.parametrizeParallel` splits it into median-cut
  regions, one chart group per core (1M faces: 424 s to 3.4 s).
- A flow-DiT forward over tens of thousands of tokens needs a scope PER BLOCK (`Dit.forward`, `Decoder.run`): one
  scope around the whole forward keeps every block's intermediates alive together; fixture-sized oracles never show it.
- The stages read their results on the host, so a Metal failure inside them hands back zeros with no error: the
  host refuses a job whose MLX latch is set after `step`. The CPU stages run on a worker (`sse.offload`).
- The DiT runs the published recipe only (RoPE, one shared modulation, QK norms on both attentions); the input,
  time and output layers stay f32, the blocks bf16, the decoders f16 (as the reference converts them).

## Proofs

- Tiny pack (`conformanceFixture`): every model at the smallest width its code takes; the job runs the 512 pipeline
  to a GLB. Golden (`MLX_SERVE_GOLDEN_DIR=<dir> zig build test -Dtest-filter=trellis2`): the decoded voxels
  (coords, dual-grid head, PBR attributes) at two steps, saved on the old code, bit-equal after the sdk port
  (gelu, modulation, timestep sinusoid, the flow schedule).
- Reference oracles (env-gated `TRELLIS2_*`, `tests/dump_trellis2_fixtures.py`), the end-to-end run
  (`TRELLIS2_PACK`, `TRELLIS2_IMAGE`, `TRELLIS2_OUT`) and `tests/test_trellis2_3d.sh`.

## Known issues

- The golden covers the GPU half; the CPU half (no sdk change) is covered by its hermetic tests.

## The row from the root architecture table

microsoft/TRELLIS.2-4B (MIT): a `pipeline.json`-only pack (`pack_claims`), `.mesh` modality, always textured. Gated
DINOv3 rides in the pack; no background remover (RMBG-2.0 is non-commercial), a cutout's alpha is the subject.
512/1024/1536 cascade
