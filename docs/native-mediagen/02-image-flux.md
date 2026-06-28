# Native Image Generation (FLUX) — Zig + mlx-c Port Plan

**Goal:** Replace the `mflux` Python venv path for text-to-image with a native Zig + mlx-c
implementation served from `mlx-serve`. No Python. Exposed as an OpenAI-compatible
`/v1/images/generations` endpoint.

**Status:** 🟢 **Engine done + validated** (FLUX.2-klein-4B, `src/flux.zig`). **Modality 2 of 3.**
Branch `feature/Any2Any`. End-to-end vs the Python reference (seed 42, "a red apple on a wooden
table, studio lighting", 4 steps, 1024×1024): text encoder **corr 0.99973**, DiT **corr 0.99990**,
VAE **corr 0.99998**, **end-to-end corr 0.99950 / PSNR 37.9 dB** — a real PNG essentially identical
to Python, seed-exact noise via `mlx_random_key`+`mlx_random_normal`.

Gotchas surfaced (worth CLAUDE.md notes):
1. `Flux2Modulation` applies `silu(temb)` **before** its linear — skipping it tanks the DiT to corr 0.29.
2. Strided/lazy mlx arrays + conv/readback (same class as the lazy-slice→gather_qmm gotcha): `mlx_conv2d`
   miscomputes on strided input, and `mlx_array_data_float32` on a lazy *transpose* returns source memory
   order — `mlx_contiguous` before conv and before reading output.
3. Qwen hidden states have massive-activation outliers (|x|≈15000) — validate the text encoder by
   **correlation**, not absolute rmse.

VAE convs are bf16 OHWI (direct `mlx_conv2d`, no transpose); transformer/text Linears are affine-4bit g64
(existing quant path); seed-exact noise is free.

*Remaining for full feature*: Qwen3 chat-template tokenization (shared with audio's tokenizer work),
`/v1/images/generations` endpoint + dispatch, real PNG encoding (engine emits PPM today — vendor
`stb_image_write.h`), `ImageGenService` app swap, FLUX.1-schnell fast-follow (T5+CLIP).

---

## Scope & first-slice choice

`mflux` ships two unrelated families: **FLUX.1** (`Flux1`: MMDiT + T5-XXL + CLIP-L) and
**FLUX.2-klein** (`Flux2Klein`: parallel-attn DiT + **Qwen3** text encoder). They share almost
nothing at the tensor level → two model trees, one shared affine-quant loader + conv2d/group-norm.

**First slice = FLUX.2-klein-4B**, not schnell — deliberately diverging from the generic
recommendation, because for *us*:
- Its text encoder is a **standard causal Qwen3 dense LLM, which we already run natively** — vs
  schnell needing **two new encoders ported from scratch** (T5-XXL with relative-position bias +
  gated GELU, and CLIP-L). Reusing Qwen3 is the big win (mirrors the LTX "reuse our Gemma" play).
- **klein-4B is already downloaded** (`~/.cache/huggingface/hub/models--Runpod--FLUX.2-klein-4B-mflux-4bit`);
  schnell needs a fresh `dhairyashil/FLUX.1-schnell-mflux-4bit` pull.
- The Qwen3 layer-hidden capture (layers 9/18/27) reuses the same "capture intermediate hidden
  states" hook needed by the LTX Gemma encoder → shared infrastructure.

**Fast-follow = FLUX.1-schnell** (Apache-2.0, no guidance embedder, plainest scheduler) once T5-XXL
+ CLIP-L are ported. Both VAE decoders share the same conv2d/group-norm/nearest-upsample primitives.

| In scope (slice 1) | Out / later |
|---|---|
| FLUX.2-klein-4B txt2img → PNG | FLUX.1 schnell/dev (fast-follow), klein-9B |
| `/v1/images/generations` (OpenAI shape) | img2img, image_strength, Fill/Depth variants |
| Reuse native Qwen3 encoder (layer 9/18/27 capture) | Negative prompt (no-op in app's call path anyway) |
| Seed-exact noise via MLX's own RNG | VAE encoder (txt2img doesn't need it), tiling |

---

## Architecture (from installed `mflux` 0.17.5, real klein-4B checkpoint)

### Control flow — `Flux2Klein.generate_image(seed, prompt, num_inference_steps=4, height, width, guidance=1.0)`
1. **Prompt encode** (Qwen3): apply Qwen3 chat template (`enable_thinking=False, add_generation_prompt=True`)
   → tokenize (max_length 512, pad) → Qwen3 forward with causal+pad mask → **concat raw hidden states
   from layers 9/18/27** → `[B, seq, 3·hidden = 7680]` (`joint_attention_dim`). NOT final-norm output.
   Negatives only if `guidance>1.0` (default 1.0 → no CFG).
2. **Latents**: `mx.random.normal((1,128,H//16,W//16), key=mx.random.key(seed))` → `pack_latents` →
   `[1, (H//16)(W//16), 128]`; `latent_ids` grid `[1, seq, 4]` (t,h,w,layer).
3. **Denoise loop** (FlowMatchEulerDiscrete, `requires_sigma_shift` → empirical `mu(image_seq_len, steps)`,
   `sigmas = exp(mu)/(exp(mu)+(1/σ−1))`, `timesteps = σ·1000`): per step
   `noise = transformer(...)`; if CFG `noise = neg + guidance·(pos−neg)`; `latents += (σ[t+1]−σ[t])·noise`.
4. **Decode**: reshape packed → `[1,128,latH,latW]`; `vae.decode_packed_latents`: **batch-norm de-normalize
   `packed·√(running_var+1e-4)+running_mean` → unpatchify 128→32 → post_quant_conv → decoder** → `[1,3,H,W]`.
5. **To image**: `clip(x/2+0.5, 0,1)·255` → uint8 → PNG.

### Text encoder — Qwen3 dense (REUSE) + layer capture
klein-4B: hidden 2560, 36 layers, 32 q-heads / 8 kv (GQA), head_dim 128, intermediate 9728, vocab
151936, rope_theta 1e6, rms_eps 1e-6, per-head q/k RMSNorm(128), SwiGLU. **= our `qwen3` arch.**
- New bit: capture hidden states **after** layers 9, 18, 27 (list index where [0]=embeddings, [k]=after
  layer k−1), stack+reshape → `[B,seq,7680]`. Build a generic "capture at layer set" hook (also serves LTX).
- Chat template via existing Jinja engine; tokenize via existing BPE (Qwen2 tokenizer).

### Diffusion transformer — Flux2 (parallel-attn, shared modulation)
klein-4B: inner_dim 3072 (24×128), **5 double + 20 single** blocks, `joint_attention_dim 7680`,
in/out_channels 128, mlp_ratio 3.0, rope_theta **2000**, axes_dims_rope (32,32,32,32), no guidance embed.
- **Shared/global modulation** (`Flux2Modulation`): `double_stream_modulation_{img,txt}` (sets=2) and
  `single_stream_modulation` (sets=1) computed **once** from `silu(temb)→Linear`, **reused every block**
  (no per-block AdaLN linears — don't allocate them).
- `x_embedder 128→inner (no bias)`, `context_embedder 7680→inner (no bias)`,
  `time_guidance_embed`: sinusoidal(256)→Linear→silu→Linear (no guidance branch).
- **4-axis RoPE** (t,h,w,layer), each 32 dims, returns (cos,sin) width 64, applied interleaved;
  text+image ids each roped then concatenated.
- **Double block**: LayerNorm(affine=False) modulated `(1+scale)·norm+shift`; `Flux2Attention`
  (to_q/k/v + add_q/k/v_proj + to_out + to_add_out, all no-bias; per-head RMSNorm(128) q/k); joint
  `[text,image]` attention; gated residual. FF = `Flux2FeedForward(mult=3)`: `linear_in→2·3·inner`,
  **SwiGLU** (silu(x1)·x2), `linear_out 3·inner→inner`; gated.
- **Single block**: LayerNorm(affine=False) + shared single mod; `Flux2ParallelSelfAttention`: **fused**
  `to_qkv_mlp_proj inner→3·inner + 2·3·inner`, split qkv+mlp; RMSNorm q/k; rope; SDPA;
  `concat(attn(inner), SwiGLU_mlp(3·inner)) → to_out(4·inner→inner)`; gated.
- After: slice off text tokens, `norm_out` (AdaLayerNormContinuous), `proj_out inner→128`.

### VAE decoder — Flux2 (`vae/`)
`latent_channels 32, scaling 1.0, shift 0.0`; `Flux2BatchNormStats(128, eps 1e-4)` for packed-latent
denorm; `quant_conv Conv2d(64,64,k1)`, `post_quant_conv Conv2d(32,32,k1)`. Decoder:
`block_out (128,256,512,512), 2 layers/block (3 resnets), norm_groups 32, eps 1e-6`:
conv_in(32→512,k3,p1) → mid (resnet512 + attention512 + resnet512) → up_blocks (512→512+up, 512→512+up,
512→256+up w/ conv_shortcut, 256→128 no up) → GroupNorm(32,128)→silu→conv_out(128→3,k3).
- **ResnetBlock2D**: GroupNorm32(fp32)→silu→Conv3×3→GroupNorm32(fp32)→silu→Conv3×3 + (1×1 shortcut if ch≠).
- **Upsampler**: nearest 2× (broadcast+reshape, **no conv_transpose**) + Conv3×3.
- **Attention**: GroupNorm(32,512)→to_q/k/v Linear→single-head SDPA over H·W (scale 1/√512)→to_out, residual.
- Conv weights stored **OHWI** (mflux already transposed at save); BF16 (not quantized). The only new VAE
  primitive vs FLUX.1 is the latent BatchNorm denorm (`·std+mean` with stored running stats).

---

## Required MLX ops / FFI
| Op | Status | Where |
|---|---|---|
| `mlx_conv2d` | ✅ **added this session** | VAE convs (k3 p1, k1) |
| `mlx_random_normal` (takes `key`) | ⬜ **bind next** — gives **seed-exact noise via `mlx_random_key(seed)`**, no Threefry reimpl | latent init |
| group_norm (32 groups, fp32, pytorch-compat) | ⬜ **hand-roll** (reshape→mean/var→normalize→affine); not in mlx-c headers | every VAE resnet/attn/norm_out |
| nearest 2× upsample | ✅ compose (broadcast+reshape) | VAE upsamplers |
| SDPA / RMSNorm / LayerNorm(affine=false) / SwiGLU / silu | ✅ existing | encoder + DiT + VAE attn |
| affine quant matmul (g64, U32+bf16 scales/biases) | ✅ existing loader | all Linears + embeddings (NOT convs) |
| layer-hidden capture hook | ⬜ **add** (also serves LTX Gemma) | Qwen3 encoder layers 9/18/27 |
| `conv_transpose2d` | ❌ **NOT needed** | — |

### Seed-exact noise (the de-risked item)
mflux calls MLX's `mx.random.normal(shape, key=mx.random.key(seed))`. We call the **same** mlx-c ops
(`mlx_random_key` + `mlx_random_normal`), so the Threefry2x32 stream is **bit-identical** — seed-exact
parity for free. No custom RNG. (Residual pixel deltas come only from bf16 reduction-order, like our
existing INT4 long-greedy tail.)

---

## Integration points
- **New files** `src/flux.zig` (Flux2 DiT + sampling + pipeline), `src/flux_vae.zig` (VAE decoder),
  reuse Qwen3 forward from `transformer.zig` with the new layer-capture hook.
- **Affine split-safetensors loader** — shared with video: mflux metadata-gated (`mflux_version`/
  `quantization_level`), per-subdir shards (`transformer/`, `text_encoder/`, `vae/`), internal naming,
  OHWI convs, "quantized iff `.scales` sibling" rule. Read pre-quantized directly (skip diffusers mapping).
- **Model dispatch** — `model.zig`: recognize the mflux repo layout (config `model_type` + subdir shards);
  request→file path via the inference thread (`runImageGen`); no token slot machinery.
- **HTTP** — `server.zig`: `POST /v1/images/generations` (OpenAI: `{model, prompt, n, size, ...}`) →
  returns `{data:[{b64_json}]}` or a URL/path; PNG bytes.
- **PNG writer** — vendor `stb_image_write.h` into `lib/` (or a minimal PNG encoder); `src/png.zig` wrapper.
- **App** — `ImageGenService.swift`: replace `python.runScript(mfluxScript)` with a `/v1/images/generations`
  call; keep `Phase`/recent/log UI.

---

## Implementation checklist

### Foundation
- [x] Bind `mlx_conv2d` in `src/mlx.zig`
- [ ] Bind `mlx_random_normal` (+ confirm `mlx_random_key` binding) in `src/mlx.zig`
- [ ] conv2d helper (NHWC in, OHWI weight, k3 p1 / k1) + hermetic shape test
- [ ] group_norm helper (32 groups, fp32, affine) + test vs a tiny Python `nn.GroupNorm` reference
- [ ] nearest-2×-upsample helper (broadcast+reshape) + test
- [ ] Vendor `stb_image_write.h` + `src/png.zig` writer + round-trip test
- [ ] Affine split-safetensors mflux loader (metadata gate, subdir shards, OHWI) — shared w/ video
- [ ] Qwen3 layer-hidden capture hook (capture after specified layer indices, raw pre-final-norm)

### Text encoder (reuse Qwen3)
- [ ] Chat template render + tokenize (max_length 512, pad) + causal/pad mask
- [ ] Capture layers 9/18/27 → stack/reshape → `[B,seq,7680]`
- [ ] **Oracle:** match reference `prompt_embeds` for a fixed prompt (dump `.npy`, L∞)

### DiT
- [ ] x_embedder / context_embedder / time_guidance_embed; shared global modulation (computed once)
- [ ] 4-axis theta-2000 RoPE; per-head q/k RMSNorm; joint `[text,image]` attention
- [ ] 5 double blocks (Flux2Attention + Flux2FeedForward SwiGLU, gated)
- [ ] 20 single blocks (fused qkv-mlp parallel attention, gated)
- [ ] norm_out + proj_out→128
- [ ] **Oracle:** single-forward `noise` match vs reference (fixed latents+temb, `.npy` L∞)

### Sampling + VAE + output
- [ ] FlowMatchEuler `mu`/sigma schedule + Euler step; pack/unpack latents; latent grid ids
- [ ] **Oracle:** pre-VAE latents match reference after N steps (seed-exact noise → tight match)
- [ ] `flux_vae.zig`: batchnorm denorm → unpatchify → post_quant_conv → decoder (conv2d/groupnorm/upsample/attn)
- [ ] `clip(x/2+0.5)·255` → uint8 → PNG
- [ ] **Oracle:** decoded PNG vs reference `ref_klein4b_s42.png` (per-pixel max-abs / PSNR within tol)

### Endpoint + app + ship
- [ ] `POST /v1/images/generations` + `model.zig` dispatch + capability
- [ ] `tests/test_image_gen.sh` (curl → PNG; dims, non-blank, decodes)
- [ ] `ImageGenService.swift` → server; remove mflux Python script
- [ ] `zig build test` + `swift build` green; CLAUDE.md arch table + gotchas (OHWI conv, shared modulation, layer capture)
- [ ] **Commit locally** (modality 2)
- [ ] *(Fast-follow)* FLUX.1-schnell: T5-XXL encoder, CLIP-L encoder, MMDiT (19+38), Flux1 VAE; schnell oracle

---

## Equivalence-test oracle

**FLUX.2-klein-4B (cached — runs now):**
```bash
~/.mlx-serve/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
local = snapshot_download("Runpod/FLUX.2-klein-4B-mflux-4bit")
flux = Flux2Klein(model_config=ModelConfig.flux2_klein_4b(), model_path=local, quantize=None)
img = flux.generate_image(seed=42, prompt="a red apple on a wooden table, studio lighting",
                          num_inference_steps=4, height=1024, width=1024, guidance=1.0)
img.save(path="/tmp/ref_klein4b_s42.png", overwrite=True); print("saved")
PY
```
- Stage taps for tight bisection: dump `prompt_embeds`, post-step-0 latents, pre-VAE latents to `.npy`
  at the §1 call sites and compare per-stage — a whole-PNG diff hides where divergence starts.
- Seed-exact noise (same MLX RNG) → the deterministic stages should match tightly; only bf16
  reduction-order produces small pixel deltas (expected, like our INT4 tail).

**FLUX.1-schnell (fast-follow, needs download):** `ModelConfig.schnell()` + `Flux1`, seed 42, 4 steps,
guidance 0.0, 1024² → `/tmp/ref_schnell_s42.png`.

---

## Risks / notes
1. **group_norm hand-roll must be fp32 + pytorch-compatible** (eps inside sqrt, per-group over (C/g)·H·W).
2. **Flux2 modulation is global** — computed once from temb, reused; the checkpoint has no per-block AdaLN.
3. **Qwen3 chat template + layers 9/18/27 raw hidden** (not final-norm) — match exactly.
4. **OHWI conv weights** already transposed in mflux-4bit repos — read as-is; convs are BF16 (not quantized).
5. Token **embedding is quantized** (U32) in these repos — handle quantized embedding lookup.
6. PNG writer: prefer vendored `stb_image_write.h` (public domain, single header) for correctness.
