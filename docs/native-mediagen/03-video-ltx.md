# Native Video Generation (LTX-Video 2.3) — Zig + mlx-c Port Plan

**Goal:** Replace the `ltx-2-mlx` Python venv path for text-to-video with a native
Zig + mlx-c implementation served from `mlx-serve`. No Python. Exposed as a custom
`/v1/video/generations` endpoint.

**Status:** 🟡 **Foundation laid + validated** (`src/ltx_video.zig`). **Modality 3 of 3** (hardest).
Branch `feature/Any2Any`. The full pipeline is a large remaining effort (scoped at weeks); the
foundation de-risks the load + the new primitives.

Done + validated vs the real `dgrauet/ltx-2.3-mlx-q4` checkpoint (loads 262 connector + 86 vae_decoder
tensors, no GPU forward → no OOM):
- `LtxConfig` from config.json; **single-component q4/bf16 loader** (`loadComponent`) — the 41 GB model
  stores each sub-model as a separate file, so the dir-scanning `loadWeights` would OOM; this loads one
  component and classifies tensors by sibling `.scales`.
- New primitives: `conv3d`, null-weight `pixelNorm`.
- Pinned findings (accelerate the rest): **VAE conv weights are already MLX-layout** `[C_out,kD,kH,kW,C_in]`
  (NO transpose, unlike the PyTorch-layout audio convs); DiT = 48 blocks × 154 tensors, six q4 attention
  sub-modules/block; adaLN tables F32 with `scale_shift_table [9,4096]` and AV-cross `[5,*]` **scale-first**
  (porting trap); connector bf16, two 8-block `Embeddings1DConnector`s + 128 registers + gated attn.

### Progress (all validated against the reference, per-stage `.npy` oracle taps)

| Stage | Status | Validation |
|---|---|---|
| Single-component q4/bf16 loader (`loadComponent`) | ✅ | loads 262 connector + 86 vae tensors |
| `conv3d` + `decoderConv3d` (causal 3D conv) | ✅ | corr 1.0 vs reference conv_in |
| **3D VAE decoder** (`vaeDecode`, full) | ✅ | corr 1.000000, mse 0.0 |
| **Connector projection** (`connectorProject`) | ✅ | video/audio corr 0.999994 |
| **Connector transformers** (`connectorTransform`, 8-block ×2) | ✅ | video 0.999910 / audio 0.999679 |
| **Gemma 49-layer capture** (`gemmaCapture`) | ✅ | layers 0/1/24/48 corr 0.999999…0.999963 |
| **DiT conditioning** (timestep + `AdaLayerNormSingle`) | ✅ | params + embedded corr 1.000000 |
| **DiT 48-block attention forward** + output head | ⬜ **remaining (the last big piece)** | — |
| Guided Euler sampler + dynamic_shift_schedule | ⬜ remaining | — |
| End-to-end wiring (text→Gemma→connector→DiT-loop→VAE→frames) + AVFoundation mux + `/v1/video/generations` | ⬜ remaining | — |

**The entire conditioning path (Gemma → connector) and the 3D VAE decode are done and validated.** The
one remaining big component is the DiT's 48× `BasicAVTransformerBlock` attention forward — and it is fully
scaffolded: the split-RoPE (`applyRopeSplit`), q4 matmul (`gQLin`), and all the modulation params
(`ditAdaLNSingle`, corr 1.0) it consumes are done. Stage it block-0-first (catches the adaLN-table-ordering
[per-block 9-param shift-first; AV-cross 5-param scale-first], per-head gating `2·sigmoid`, and affine-QK-norm
traps cheaply), then the full 48 → x0 = x_t − sigma·v. Then the Euler sampler + wiring are comparatively small.

> This is the heaviest modality. The plan front-loads the two pleasant surprises (the q4
> weights load with our existing affine-4bit loader; the text encoder is **our own
> Gemma-3-12B**) and isolates the two hard parts (the **conv3d 3D-VAE decoder** and the
> **48-block joint AV DiT**). Weights repo `dgrauet/ltx-2.3-mlx-q4` is **already on disk**
> (`~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q4/…`) → the equivalence oracle runs now.

---

## Scope

| In scope (slice 1) | Out of scope (later / never) |
|---|---|
| Text→video, **one-stage** pipeline (`TI2VidOneStagePipeline`) | Two-stage / two-stage-HQ, distilled LoRA, latent upsampler, stage-2 refine |
| **Silent** mp4 (skip audio decode) | Audio VAE + HiFi-GAN vocoder + BWE (rfft/ConvTranspose1d/STFT) |
| Reuse native Gemma-3-12B (49-layer capture) | Second mlx-lm Gemma copy |
| `conv3d` 3D-VAE decoder, single-pass (small res) | VAE temporal tiling, TeaCache, block streaming, res2s sampler |
| AVFoundation mp4 mux (Swift) | ffmpeg runtime dependency |
| `/v1/video/generations` endpoint | I2V (image conditioning → VAE encoder), `enhance_*` prompt rewrite |

**Faithfulness caveat:** the DiT couples video and audio via bidirectional AV cross-attention,
so faithful **video** requires running the **audio latent stream through the DiT** even though we
never decode it to a waveform. We skip only the audio *decoder*, not the audio *stream*.

---

## Architecture (from installed `ltx_core_mlx` / `ltx_pipelines_mlx` v0.14.9)

### Component → weight-file map (q4 repo)
| Component | Weight file | Quantized |
|---|---|---|
| Text connector | `connector.safetensors` (5.9 GB) | no (bf16) |
| Video DiT (one-stage) | `transformer-dev.safetensors` (11 GB) | **q4, blocks only** |
| Video VAE decoder | `vae_decoder.safetensors` (777 MB) | no |
| Audio VAE / vocoder | `audio_vae.safetensors` / `vocoder.safetensors` | no — **skipped slice 1** |
| Latent upsampler | `spatial_upscaler_x2_v1_1.safetensors` | no — skipped |

### 1. Text encoder (reuse our Gemma-3-12B) + connector
- Gemma conditioning is **raw text, NO chat template**: `tokenizer.encode(text.strip())`,
  **left-pad to 1024** with pad token, `attention_mask` 1=valid/0=pad.
- Extract **all 49 hidden states**: index 0 = embedding output **scaled by √3840** (Gemma embed
  scale); indices 1..48 = each decoder layer's **residual-stream output BEFORE the final norm**
  (do **not** apply `model.norm`). Each `(1,1024,3840)`. Causal + left-pad mask.
- **Contract to implement in our Gemma:** an all-49-layers capture over a 1024-token left-padded
  raw-tokenized prompt + the padding mask. (Today we return only the post-final-norm last hidden.)
- **Connector** (`connector.safetensors`, port verbatim): stack→`[B,T,3840,49]`, per-token RMSNorm
  over the 3840 axis (eps 1e-6), reshape to `[B,T,188160]` (D-interleaved), zero pads → two Linears
  `188160→4096` (video) / `188160→2048` (audio) with input pre-scale `√(target/3840)` → two
  `Embeddings1DConnector` transformers (8 blocks each, 128 registers, 32 heads, head_dim 128/64,
  gated attention, RoPE max_pos 4096, output RMSNorm). → `video_embeds(1,?,4096)`, `audio_embeds(1,?,2048)`.

### 2. Video DiT (`transformer-dev.safetensors`) — the core
`num_layers 48, video_dim 4096 (32 heads × 128), audio_dim 2048 (32 × 64), av_cross 32 × 64,
patch_channels 128, ff_mult 4 (gelu-approx), timestep_embed_dim 256, timestep_scale 1000,
rope_theta 10000 type "split", video max_pos (20,2048,2048), audio max_pos (20,), norm_eps 1e-6,
affine-free norms, rms QK-norm.`
- **Block** `BasicAVTransformerBlock`, **adaLN-zero**, 8 sublayers in order: video self-attn →
  audio self-attn → video→text x-attn → audio→text x-attn → audio→video x-attn → video→audio
  x-attn → video FFN → audio FFN. Each: `normed = rms(x)*(1+scale)+shift; x += sublayer(normed)*gate`.
- **Attention**: full (non-causal) `scaled_dot_product_attention(scale=head_dim**-0.5, mask=None)`;
  affine QK-norm over inner_dim; **per-head gating** `out *= 2·sigmoid(to_gate_logits(x))`; single `to_out`.
- **RoPE**: custom **log-spaced fractional 3D RoPE** (NOT `1/θ^k`): `indices = θ**linspace(0,1,nf)·π/2`,
  `frac = pos/max_pos`, `scaled = indices·(frac·2−1)`, SPLIT rotate-half. Video num_freqs `4096//6=682`.
- **Patchify**: pure reshape (patch=1). Video `(B,128,F,H,W)→(B,F·H·W,128)`; audio `(B,8,T,16)→(B,T,128)`.
  Then `patchify_proj 128→{4096,2048}`; reverse `proj_out {4096,2048}→128`.
- **Timestep**: sinusoidal(256) → MLP(silu) → `AdaLayerNormSingle` shift/scale/gate triples; `t·1000`.
- **Velocity → x0**: `x0 = x_t − sigma·v` in f32.
- **No KV cache / no AR** — full bidirectional forward over all tokens each step. `mx.eval` every ~8
  blocks to dodge the GPU watchdog.
- adaLN table ordering traps: per-block `scale_shift_table (9,4096)` is **shift,scale,gate** order;
  `scale_shift_table_a2v_ca_* (5,*)` is **scale-first**. Get these wrong → silent garbage.

### 3. 3D video VAE decoder (`vae_decoder.safetensors`) — heaviest new kernel work
Compression temporal 8× / spatial 32× / 128 latent ch. Decoder `causal=False`, zero spatial pad.
No attention, no GroupNorm. Decode: `(B,128,F,H,W)` → BFHWC → `denormalize x·std+mean` (per-channel
128-vec from `vae_decoder.per_channel_statistics`) → `conv_in 128→1024 (k3)` → 9 up-blocks (ResStage
+ `DepthToSpaceUpsample`) → pixelnorm→silu→`conv_out 128→48` → `unpatchify_spatial(4)` 48→3 → BCFHW
`(B,3,8F−7,32H,32W)` in [−1,1].
- **Causal 3D conv** = `Conv3d(padding=0)` + manual padding (decoder: symmetric temporal **replicate**
  of first/last frame; spatial **zero-pad** via `mx.pad`). Weight layout `(O,D,H,W,I)`, data BDHWC.
- **Upsampling = manual depth-to-space** (`pixel_shuffle_3d`: channel-expanding conv then reshape/
  transpose). **No conv_transpose3d, no interpolation.** Final RGB uses `unpatchify_spatial` (different
  channel order — mixing with pixel_shuffle swaps H↔W; porting trap).
- **PixelNorm** = null-weight `rms_norm(eps 1e-8)` over channel axis (the mlx-c null-weight gotcha →
  pass ones / compute `x/√(mean(x²)+eps)`).
- Tiling (temporal-only, 8 GB budget) — **skip at small res**, needed for production res.

### 4. Sampling / guidance
- Sigmas: `ltx2_schedule = dynamic_shift_schedule(num_steps, num_tokens)` — token-adaptive flow-matching,
  descending 1→0, length `steps+1` (`base_shift 0.95, max_shift 2.05, base 1024, max 4096, terminal 0.1`).
- `euler_step(x,x0,σ,σ_next) = x + (σ_next−σ)(x−x0)/σ`.
- **Guided loop**: up to 4 DiT forwards/step — `cond`, `uncond` (CFG, fires cfg≠1), `ptb` (STG: skip
  video/audio self-attn in `stg_blocks=[28]`, fires stg≠0), `mod` (modality-isolation: skip both AV
  x-attns, fires modality_scale≠1). Combine: `pred = cond + (cfg−1)(cond−uncond) + stg(cond−ptb) +
  (mod−1)(cond−mod)`, variance-rescale by `rescale_scale 0.7`. Video guider `cfg 3, stg 1, mod 3`;
  audio `cfg 7`.

### 5. Output / mux
Frames `(x+1)·127.5 → uint8` HWC. Reference pipes rawvideo to **ffmpeg** (`libx264, yuv420p, crf 18`,
optional AAC audio). **Replace with AVFoundation** (`AVAssetWriter` + `AVAssetWriterInputPixelBufferAdaptor`,
H.264) in the Swift app — removes ffmpeg runtime dep, matches the Zig-emits-frames / Swift-muxes split.
For slice 1: silent (drop the audio input args).

---

## Required MLX ops / FFI
| Op | Status | Where |
|---|---|---|
| `mlx_conv3d` (via conv3d binding) | ✅ **added this session** | video VAE — the one genuinely new heavy kernel |
| null-weight `mlx_fast_rms_norm` (pass ones) | ✅ existing pattern | pixelnorm + DiT affine-free norms |
| `mlx_pad` (per-axis) | ✅ bound | conv padding |
| `mlx_repeat_axis` | ✅ bound | causal temporal replicate, meshgrid |
| non-causal `mlx_fast_scaled_dot_product_attention` | ✅ bound (mask=null) | DiT attention |
| depth-to-space (reshape+transpose) | ✅ compose | VAE upsampling |
| `linspace`/`clip`/`mean`/`var`/`rsqrt`/`gelu`/`silu`/`sigmoid` | ✅ have/compose | various |
| affine-4bit dequant (`mlx_dequantize` g64/b4) | ✅ existing loader | DiT blocks |

**Not needed slice 1:** conv_transpose3d, interpolation, GroupNorm, FFT (audio only).

---

## Integration points
- **New files** `src/ltx_video.zig` (DiT + connector + sampling + guided loop), `src/video_vae.zig`
  (3D-conv decoder), and Gemma 49-layer capture hook in the existing Gemma forward (`transformer.zig`
  / `forwardCaptureHidden` extension — capture *all* layers' pre-norm residual, not just last).
- **Model dispatch** — `model.zig`: `model_type`/dir recognition for the LTX repo layout (multiple
  per-component safetensors). A video request is request→file; route via the inference thread with a
  dedicated `runVideoGen` path (no token slot machinery, no PLD/drafter/MTP/batching).
- **Affine-4bit loader** — reuse; add split-file prefix stripping (`transformer.`, `connector.`,
  `vae_decoder.`, …) and the "q4 iff sibling `.scales` exists, else stored dtype" rule. Distinguish
  `.biases` (quant) from `.bias` (Linear).
- **HTTP** — `server.zig`: `POST /v1/video/generations` `{model, prompt, height, width, num_frames,
  frame_rate, steps, cfg_scale, stg_scale, seed}` → streams progress (SSE) → returns mp4 path/bytes.
- **App** — `VideoGenService.swift`: replace Python script with server call; do the AVFoundation mux
  app-side from frames the server returns (or have the server emit a frames blob + the app encodes).
- **`low_memory` staging** — reproduce the load→free→load discipline (Gemma → connector → DiT →
  decoder) or OOM the Metal heap on ≤48 GB Macs. Reuse `computeMaxSafeContext`-style guards.

---

## Implementation checklist

### Foundation (shared with image)
- [x] Bind `mlx_conv3d` + transpose convs in `src/mlx.zig`
- [ ] Affine-4bit split-safetensors loader: prefix strip + per-component key maps + `.scales`-presence rule
- [ ] conv3d helper (BDHWC, weight `(O,D,H,W,I)`, manual causal/zero pad) + hermetic shape test vs a tiny Python conv3d reference
- [ ] null-weight pixelnorm helper + test

### Gemma encoder reuse
- [ ] All-49-layers capture hook in Gemma forward (embed×√3840 + 48 pre-final-norm residuals)
- [ ] Raw tokenization + left-pad-to-1024 + padding mask path (no chat template)
- [ ] **Oracle:** compare our 49-stack to the reference `get_all_hidden_states` for a fixed prompt (L∞)

### Connector
- [ ] Load `connector.safetensors`; per-token RMS over layer stack → 188160 reshape → 2 projections (√ rescale)
- [ ] Two `Embeddings1DConnector` 8-block transformers (registers, gated attn, RoPE) → video/audio embeds
- [ ] **Oracle:** match reference connector output for the captured Gemma states

### DiT
- [ ] Patchify + `patchify_proj`; timestep sinusoidal+MLP; `AdaLayerNormSingle` triples
- [ ] Custom 3D split-RoPE (log-spaced fractional); per-head gating; affine QK-norm
- [ ] `BasicAVTransformerBlock` 8 sublayers (video+audio streams, 4 attentions, adaLN-zero) — exact table orderings
- [ ] 48-block stack with `mx.eval` flush cadence; `proj_out`; velocity→x0
- [ ] **Oracle:** single-forward latent match vs reference DiT (fixed seed, tap `video_latent`)

### Sampling
- [ ] `dynamic_shift_schedule` + `euler_step`
- [ ] Guided loop: cond/uncond/ptb/mod forwards + combine + variance-rescale
- [ ] **Oracle:** denoised `video_latent` matches reference (latent-level, the robust check)

### 3D VAE decode + mux
- [ ] `video_vae.zig`: denorm → conv_in → 9 up-blocks (ResStage + depth-to-space) → pixelnorm/silu → conv_out → unpatchify_spatial
- [ ] Frames `(x+1)·127.5→uint8`; verify per-frame PSNR vs reference frames (PNG extract)
- [ ] AVFoundation silent mp4 mux (Swift) from frames
- [ ] **Oracle:** end-to-end silent mp4; per-frame compare to reference (video track only)

### Endpoint + app + ship
- [ ] `POST /v1/video/generations` + `model.zig` dispatch + `low_memory` staging
- [ ] `tests/test_video_gen.sh` (curl → mp4; frame count, dims, non-blank)
- [ ] `VideoGenService.swift` → server + AVFoundation mux; remove Python path
- [ ] `zig build test` + `swift build` green; CLAUDE.md arch table + gotchas (conv3d layout, adaLN orderings, Gemma contract)
- [ ] **Commit locally** (modality 3 — final)

---

## Equivalence-test oracle (weights already present)
Check: `ls -lh ~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q4/snapshots/*/transformer-dev.safetensors` → present (11 GB).

```bash
~/.mlx-serve/venv/bin/python - <<'PY'
from ltx_pipelines_mlx import TI2VidOneStagePipeline
pipe = TI2VidOneStagePipeline(model_dir="dgrauet/ltx-2.3-mlx-q4",
    gemma_model_id="mlx-community/gemma-3-12b-it-4bit", low_memory=True)
pipe.generate_and_save(prompt="a red apple on a wooden table, soft daylight",
    output_path="/tmp/ltx_ref_onestage.mp4", height=256, width=384, num_frames=9,
    frame_rate=24.0, seed=1234, num_steps=8, cfg_scale=3.0, stg_scale=1.0)
PY
```
- **Compare at the latent, not pixels** (INT4 + VAE conv numerics drift). Tap `video_latent` before
  decode for the robust check; then per-frame PSNR on decoded frames; then the muxed file.
- Reference mp4 carries an AAC track — compare **video frames only** for the silent slice
  (`ffmpeg -i ref.mp4 -pix_fmt rgb24 ref_%03d.png` then per-frame max-abs/PSNR).
- For faster wiring of VAE+mux, the `--distilled` path (8 steps, 1 forward/step) is a quicker smoke ref.

---

## Risks / open questions (honest)
1. **conv3d kernel correctness** — `(O,D,H,W,I)` layout + manual causal/zero padding must match exactly
   or the VAE outputs garbage (cf. the project's existing lazy-slice / reduction-order VAE gotchas).
2. **Custom 3D RoPE + adaLN table orderings** — subtle; bit-check against reference per-step taps.
3. **Audio↔video coupling** — can't drop the audio stream from the DiT and still match the reference video.
4. **3D VAE memory** at production res — needs the temporal tiler (slice 1 dodges via small res).
5. **41/56 GB load + GPU watchdog** — `low_memory` staging + `mx.eval` flush cadence are load-bearing.
6. **Gemma contract bit-match** — raw tokenization + left-pad-1024 + all-49-layers + embed-scale + no-final-norm.
7. **AVFoundation mux fidelity** — confirm H.264/yuv420p output is acceptable vs ffmpeg crf18.
