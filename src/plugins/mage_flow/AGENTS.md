# mage_flow — Microsoft Mage-Flow (Turbo text to image, Edit multi-reference editor), ported from mflux

A Qwen3-VL text encoder (36 layers, prompt template with the first 34 tokens dropped), a native-resolution
double-stream flow DiT with its own centred 3-axis RoPE, a DiCo VAE (16x, 128-channel latent) decoded once at t=0,
FlowMatchEuler with a static shift. The Edit checkpoint adds the VAE encoder and the Qwen3-VL vision tower
(DeepStack into LM layers 0-2). The official layout has no root config.json: `model_index.json` names it.

## Files

| File | Owns |
|---|---|
| `config.zig` | MLX-free: claims (`mage_flow*`/`mageflow`, or `model_index.json`), caps, bills |
| `plugin.zig` | the media table: the job (body via `sdk/media/image_request`, edit geometry), PNG, the tiny pack |
| `mage_flow.zig` | `MfLinear` (shared with Qwen-Image and MiniMax-H3), the encoder (`teLayerForward` on `sdk/blocks/text_encoders/qwen3`), the vision tower, the DiT, the VAE, the engine |

## Rules

- DiT and encoder run bf16, the VAE f32, and both are load-bearing: Turbo was distilled in bf16 (an f32 DiT washes
  to a grey blob), and it memorized a bf16 timestep, so the timestep and its frequency table are rounded to bf16
  (`roundBf16`) and every scalar is made in its operand's dtype (`scalarLike`).
- DiT geometry (hidden, heads, depth, MLP ratio, RoPE axes and theta) comes from `transformer/config.json`; the
  linears' input widths are load-bearing (`MfLinear.load` solves a packed weight's bits/group from them).
- An editor keeps its input's geometry: the target is the primary reference's aspect at the requested budget or at
  its own size (`resolveEditTargetSize`), and every reference is resized onto that grid.
- Edit capability is the pack NAME (`mage-flow-edit`): both releases carry the same class and the tower weights.
  `caps.edit` says only that the pack may edit; the job refuses an edit on the Turbo pack by name.
- The DiT blocks keep the two streams as separate arrays (no `sdk/bases/dit` skeleton): a joint [text, image]
  sequence would add a concat and a split per block for nothing.
- The VAE attention attends inside 32x32 patches with edge padding: not the sdk mid attention; it stays here.

## Proofs

- Tiny pack (`conformanceFixture`): the 36-layer encoder at hidden 64 (layers alias one buffer), a one-block DiT
  (hidden 64, two heads, a 4-bit MLP), the DiCo decoder at 32 channels; the conformance suite runs a 256x256 job.
- Blocks moved onto the sdk bit-equal: `MLX_SERVE_GOLDEN_DIR=<dir> zig build test -Dtest-filter=mage_flow` saved a
  Turbo run on the old code; the port compares bit for bit.
- Reference oracles (env-gated, `MAGEFLOW_*_FIXTURE`, `tests/dump_mageflow_*`): VAE decode and encode, the tower,
  the encoder (text and edit), the DiT, the edit loop, the w8a8 bar.

## Known issues

- The golden covers Turbo text to image only: the edit path (VAE encoder, tower, DeepStack) is proven by the sdk
  blocks' own bit-equal tests and the env-gated oracles, not by a before/after run (a tiny tower needs the tower's
  geometry read from `vision_config`, as Qwen-Image does).
- `mage_flow.zig` is past the 1500-line cap and loads weights through the engine's `model.zig`.
