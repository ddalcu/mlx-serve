# ming_image — Ming-Image-0.2-Design / 0.1-Design (inclusionAI, MIT)

Text to image and one-picture edits, with native alpha: a 16B-A1.4B Bailing-MoE-v2 MLLM reads the prompt (and an
edit's picture, through a Qwen2.5-VL tower) plus 256 learnable query tokens,
a Qwen2-1.5B connector (non-causal) turns the query rows into caption features, the tapped prompt rows are a second
caption stream; a 6B single-stream DiT (Z-Image's S3-DiT shape) denoises 16-channel latents that the Qwen-Image VAE
decodes to RGBA. Port of github.com/inclusionAI/Ming-Image (`modeling_bailingmm2.py`, `diffusion/`).

## Files

| File | Owns |
|---|---|
| `config.zig` | MLX-free: claims, caps, the aspect buckets, bills |
| `plugin.zig` | the media table: the job (body via `sdk/media/image_request`), RGBA PNG, the tiny pack |
| `llm.zig` | the MLLM (MoE on `sdk/bases/blocks/moe`, two routers), video-RoPE tables, connector (`sdk/blocks/text_encoders/qwen3`), the two caption streams |
| `dit.zig` | the DiT on `sdk/bases/dit` (`block` hook), its 3-axis RoPE, patchify |
| `vision.zig` | an edit's picture: the release processor's prep, the Qwen2.5-VL tower, `linear_proj`, Pillow's RGBA resize |
| `ming_image.zig` | the schedule, the pipeline (text to image and edit), the env-gated reference oracles |

## Rules

- The root config's `model_type` (`bailingmm_moe_v2_lite`) is the Ming omni chat models' too: the claim also needs
  `transformer/` (`DiffusionTransformer`), `mlp/` and `connector/`.
- The MLLM's query rows route through `image_gate`, every other row through `gate` (`MultiRouter`); one expert bank.
  The per-expert tensors (release and mlx-community trees alike) are stacked into banks at load.
- The schedule is a STATIC shift of `scheduler_config.json`'s `shift` over linspace(1, 0, n): the release's
  `config['use_dynamic_shifting'] = True` writes a FrozenDict item the scheduler never reads. Its last step is
  zero-length and is not run (12 steps = 11 forwards).
- CFG runs only past 1 (`pos + s·(pos − neg)`, the negative = both caption streams zeroed); default 1, one forward
  per step. `negative_prompt` is a named 400 (the release has none).
- Both caption streams are cast to bf16 at the encoder's end: a connector quantized as stored (f32 scales) made them
  f32, and the caption concat promoted all 30 DiT blocks to f32 (1.6x slower). Guard: `ming_image: both caption
  streams reach the DiT in bf16 from a connector quantized with f32 scales`.
- The quantized DiT linears stay on stock qmm: dequant + GEMM measured slower on the M5 Ultra.
- A request runs at the release's nearest aspect bucket (`config.bucket`); the model trained on those sizes only. A
  snapped size is logged (`asked WxH -> bucket WxH`); the PNG carries the real size (4096x4096 renders 2048x2048).
- An edit is one picture (`ref_images` and `size` are named 400s): it runs at the picture's bucket in the release's
  1024 table (`config.editSizes`) and is LANCZOS-resized back to its aspect. The picture's 576-ish tokens sit in the
  prompt at the centred video-RoPE positions and route through `image_gate` with the query block; its VAE latents
  ride the DiT as frame 1 (kept in CFG's negative branch), and the output is frame 0.
- An RGBA picture resizes the way Pillow does (`vision.resizeRgba`: premultiplied, divided back): a plain resize
  bled the colour of transparent pixels into the reference (cos 0.71).
- A pack edits only with the MLLM's `vision.*` + `linear_proj.*` and `mllm/preprocessor_config.json` (`caps.edit`).
- A chat-formatted prompt past 4096 tokens is a named 400, never cut: the release truncates nowhere, and a cut drops
  the assistant turn both caption streams are read after.
- Both sequences' pad-to-32 rows are masked in the release (`zero_masked`), so they are dropped; only the caption's
  padded length survives, as the image tokens' first RoPE position.
- Packs: the release (bf16) and mlx-community's layout (the same tree, MLX affine triples beside each weight);
  `tests/convert_ming_image_weights.py` writes the latter (MLLM 4/8-bit, connector and DiT linears 8-bit).

## Proofs

- Tiny pack (`conformanceFixture`): 3-layer MoE MLLM with 4-bit experts, 2-layer connector stored f32 (its down
  projections 4-bit with f32 scales), a 2-block DiT, a 16-channel VAE; the conformance suite runs a 512x512 job.
- `ming_image llm: a stacked expert bank holds no reference to the per-expert tensors` (active memory flat).
- Reference oracle: `tests/dump_ming_image_fixtures.py` builds a tiny random pack with the release's classes;
  `MING_IMAGE_TEST_MODEL` + `MING_IMAGE_FIXTURE` compare ids, both caption streams, a DiT forward at a canvas both
  sequences pad, a VAE decode and 3-step pipelines with and without CFG; and the edit path: picture prep, the tower
  (a windowed and a full block over partial windows), the edit's ids and streams, the bucket sizes, the reference's
  resize and VAE encode, a DiT forward with the reference frame and a 3-step edit (cosine and rms ratio).

## Known issues

- The layer-decomposition checkpoint (`learned` pads, multi-frame output) is refused by name.
- Text to image returns the bucket size; the release resizes to the size asked for (and never uses the 512 table).
- The MLLM stays resident through the denoise (it is idle there); staging it per request is not done yet.
