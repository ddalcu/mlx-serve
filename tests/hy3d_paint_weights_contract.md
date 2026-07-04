# Hunyuan3D-2.1 PAINT (texture) weight-conversion contract

BINDING converted-tensor-name contract for the Hunyuan3D-2.1 **paint / texture** pipeline
(image → multiview albedo + metallic-roughness maps → baked PBR texture). Implemented by
`tests/convert_hunyuan3d_paint_weights.py` and consumed by the native Zig paint engine
(`src/hunyuan3d_paint.zig`, in progress) plus the fixture/oracle dumps.

Sibling of the SHAPE contract implemented by `tests/convert_hunyuan3d_weights.py`. **Read that
script first** — this stage reuses its `Source` strict-accounting pattern, its `should_quantize`
predicate, and its `mx.save_safetensors` + `--self-test` + `--bits {8,16}` structure. The one
big structural difference: **the paint models need NO per-head QKV de-interleave.** Every
attention here (base SD, MDA, RA, MA, DINO-x-attn, VAE mid-block) stores `to_q`/`to_k`/`to_v`
(or `query`/`key`/`value`) as SEPARATE `nn.Linear`s, so there is no fused-QKV interleave to
undo. The paint mapping is therefore **strictly 1:1**: no tensor is fused, split, stacked, or
de-interleaved — only NAMES, CONV LAYOUT, and DTYPE change (plus optional per-linear quant).

All facts below were verified 2026-07-04 against the on-disk checkpoints (key inventories dumped
by torch mmap-load) AND the reference sources
(`reference/hy3dpaint/hunyuanpaintpbr/unet/{modules.py,attn_processor.py,model.py}`,
`textureGenPipeline.py`, `pipeline.py`, the shipped `unet/`, `vae/`, `scheduler/` configs, and the
`facebook/dinov2-giant` HF snapshot config).

---

## 1. Output layout

```
~/.mlx-serve/models/local/hunyuan3d-2-1-paint-8bit      (--bits 8, default; shipping build)
~/.mlx-serve/models/local/hunyuan3d-2-1-paint-fp16      (--bits 16; parity-debug build the HY3DP_* oracles target)
├── config.json          {"model_type":"hunyuan3d_2_1_paint", ...}  (§3)
├── unet.safetensors      main 2.5D UNet: SD2.1 UNet + MDA/RA/MA+PoseRoPE/DINO additions
│                          + learned text tokens + ImageProjModel   (1061 logical tensors)   (§4)
├── unet_dual.safetensors reference-stream UNet: a plain SD2.1 UNet copy (686 tensors)         (§5)
├── vae.safetensors       standard SD2.x AutoencoderKL, ENCODER + DECODER (248 tensors)        (§6)
├── dino.safetensors      DINOv2-giant image encoder (726 tensors; 727 − mask_token)           (§7)
├── LICENSE, NOTICE       copied from the paint checkpoint / reference tree
```

Four weight files, one per source state-dict, so the convert accounting is a clean 1:1 per file.
Total logical tensors: **1061 + 686 + 248 + 726 = 2721** (before quant triples eligible linears).

Source checkpoints (verified present):

| Output file | Source |
|---|---|
| `unet.safetensors`, `unet_dual.safetensors` | `hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin` (3.93 GB, fp16 pickle; top-level prefixes `unet.` 1061 + `unet_dual.` 686 = 1747 keys) |
| `vae.safetensors` | `hunyuan3d-paintpbr-v2-1/vae/diffusion_pytorch_model.bin` (335 MB, **fp32** pickle; 248 keys) |
| `dino.safetensors` | `dinov2-giant/model.safetensors` (4.5 GB, fp32; 727 keys) |
| scheduler facts (§3) | `hunyuan3d-paintpbr-v2-1/scheduler/scheduler_config.json` |

`image_encoder/`, `text_encoder/`, `tokenizer/`, `feature_extractor/` are **DEAD at inference**
(learned prompt embeds replace CLIP text; DINOv2 replaces the CLIP image encoder) — NOT converted.

---

## 2. Global transform rules

Applied uniformly; the mapping is otherwise a 1:1 rename. `[O,I,kH,kW]` = PyTorch conv weight.

| # | Rule | Applies to | Detail |
|---|---|---|---|
| T1 | **Conv NCHW→NHWC (OHWI)** | every ndim-4 conv `.weight` | `transpose(0,2,3,1)` → `[O,kH,kW,I]`, contiguous. Matches the mlx-serve conv convention (flux/krea VAE `conv2d` helper: NHWC data, OHWI weight). Conv `.bias` unchanged (1-D). |
| T2 | **1×1 convs stay 4-D OHWI** | `conv_shortcut`, VAE `quant_conv`/`post_quant_conv` | `[O,I,1,1]` → `[O,1,1,I]`. **NOT flattened to Linear** — see the ⚠ flag in §9. The engine runs them through the same `mlx_conv2d` (pad 0) as flux/krea. |
| T3 | **Drop `.transformer.`** | main+dual UNet transformer-block subkeys | The `Basic2p5DTransformerBlock` wrapper nests the base `BasicTransformerBlock` under `.transformer.`; remove that segment so the base fields (`attn1/attn2/ff/norm1-3`) sit at the SAME level as the 2.5D siblings (`attn_multiview/attn_refview/attn_dino`). |
| T4 | **Flatten `.processor.`** | MDA + RA extra material projections | `attn1.processor.to_q_mr` → `attn1.to_q_mr`; `attn_refview.processor.to_v_mr` → `attn_refview.to_v_mr`. |
| T5 | **Normalize `.to_out.0`→`.to_out`** | every attention output proj | Drop the `nn.ModuleList` index 0 (index 1 is a param-less Dropout). Applies to `to_out.0` AND `to_out_mr.0`. |
| T6 | **fp32→fp16 cast** | VAE (fp32 source), DINO (fp32 source) | UNet is already fp16. bf16 (none here) would also cast down. |

**GeGLU / SwiGLU are kept WHOLE (not physically split)** — the split is a documented forward-time
`chunk`, encoded in the canonical name + these semantics (T7, a spec, not a tensor transform):

- **UNet FFN = GeGLU** (`ff.net.0.proj` [2·inner, dim]): `v, g = proj(x).chunk(2, dim=-1)`,
  out `= v * gelu(g)`. **value = FIRST half of the output rows `[0:inner]`, gate = SECOND half
  `[inner:2·inner]`.** (diffusers `GEGLU`.)
- **DINO FFN = SwiGLU** (`mlp.w_in` [2·hidden, dim]): `x1, x2 = w_in(x).chunk(2, dim=-1)`,
  out `= silu(x1) * x2`. **x1 = FIRST half (gets SiLU), x2 = SECOND half (multiplied).**
  (HF `Dinov2SwiGLUFFN` — note the activation is on the FIRST half, the OPPOSITE of GeGLU where
  the activation is on the SECOND half. Do not swap.)

**NOT needed here** (present in the SHAPE contract, absent here): per-head QKV de-interleave
(`deinterleave_qkv`), MoE expert stacking, fused-QKV/KV splits. All attentions are unfused.

---

## 3. `config.json`

Everything the loader/scheduler/engine needs that is not derivable from tensor shapes. Scheduler
block copied VERBATIM from `scheduler/scheduler_config.json` (the SHIPPED inference scheduler is
`DDIMScheduler`, per `model_index.json`).

```json
{
  "model_type": "hunyuan3d_2_1_paint",
  "quant": "8bit",                       // or "fp16"
  "pbr_settings": ["albedo", "mr"],       // N_pbr = 2; the UNet emits 2 material streams
  "num_views": 6,                         // max_selected_view_num (demo default; 6..9 supported)
  "view_resolution": 512,                 // multiview diffusion H=W (demo default; 512 or 768)
  "guidance_scale": 3.0,                  // CFG scale used by the paint pipeline
  "num_inference_steps": 30,              // DDIM steps (validation/default)
  "primary_camera_azims":   [0, 90, 180, 270, 0, 180],
  "primary_camera_elevs":   [0, 0, 0, 0, 90, -90],
  "primary_view_weights":   [1, 0.1, 0.5, 0.1, 0.05, 0.05],

  "unet": {                               // diffusers UNet2DConditionModel, SD2.1 shape
    "in_channels": 12,                    // ⚠ 4 (latent) + 4 (normal embeds) + 4 (position embeds); the shipped unet/config.json LIES with in_channels:4
    "out_channels": 4,
    "cross_attention_dim": 1024,
    "block_out_channels": [320, 640, 1280, 1280],
    "layers_per_block": 2,
    "attention_head_dim": [5, 10, 20, 20],  // diffusers "heads per block"; head_dim = channels/heads = 64 everywhere
    "down_block_types": ["CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"],
    "up_block_types":   ["UpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D"],
    "sample_size": 64,
    "norm_num_groups": 32,
    "norm_eps": 1e-05,
    "act_fn": "silu",
    "use_linear_projection": true,        // proj_in/proj_out are Linear (2-D), NOT 1x1 conv
    "transformer_layers_per_block": 1
  },
  "unet_dual": { "in_channels": 4 },       // otherwise identical to unet (base SD2.1, no 12-ch conv_in)

  "vae": {                                 // diffusers AutoencoderKL, standard SD2.x
    "in_channels": 3, "out_channels": 3, "latent_channels": 4,
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2, "norm_num_groups": 32, "sample_size": 768,
    "scaling_factor": 0.18215             // ⚠ NOT in the shipped vae/config.json; the diffusers SD default. Used as vae.config.scaling_factor by encode_images.
  },

  "dino": {                                // facebook/dinov2-giant
    "hidden_size": 1536, "num_layers": 40, "num_heads": 24, "head_dim": 64,
    "patch_size": 14, "image_size": 518, "num_tokens": 1370,   // (518/14)^2 + 1 = 37^2 + 1
    "mlp": "swiglu", "intermediate_size": 4096, "layerscale_init": 1.0,
    "layer_norm_eps": 1e-06, "qkv_bias": true
  },
  "image_proj": {                          // ImageProjModel (lives in unet.safetensors, §4)
    "clip_embeddings_dim": 1536, "cross_attention_dim": 1024, "num_context_tokens": 4
  },

  "scheduler": {                           // verbatim from scheduler/scheduler_config.json
    "_class_name": "DDIMScheduler",
    "num_train_timesteps": 1000,
    "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear",
    "prediction_type": "v_prediction",
    "set_alpha_to_one": true, "steps_offset": 1,
    "timestep_spacing": "trailing", "rescale_betas_zero_snr": true,
    "clip_sample": false, "trained_betas": null
  }
}
```

---

## 4. `unet.safetensors` — main 2.5D UNet (1061 tensors)

A diffusers `UNet2DConditionModel` (SD2.1 topology) whose every `BasicTransformerBlock` is wrapped
by `Basic2p5DTransformerBlock`, PLUS learned text tokens and the DINO ImageProjModel. Source keys
carry the `unet.` prefix (stripped on output).

**Topology** (verified counts): down_blocks 0-2 = `CrossAttnDownBlock2D` (2 resnets + 2 attention
modules each) + block 3 `DownBlock2D` (2 resnets, no attention); mid_block (2 resnets + 1 attention);
up_blocks 0 = `UpBlock2D` (3 resnets, no attention) + blocks 1-3 = `CrossAttnUpBlock2D` (3 resnets +
3 attention modules each). **16 transformer blocks total** (6 down + 1 mid + 9 up). Each attention
module is a diffusers `Transformer2DModel` with a GroupNorm + linear `proj_in`/`proj_out` (because
`use_linear_projection:true`) wrapping ONE transformer block.

### 4a. Top-level (17)

| Source (after `unet.`) | Canonical | Shape (example) | Transform | Quant? |
|---|---|---|---|---|
| `conv_in.weight` | `conv_in.weight` | `[320,12,3,3]`→`[320,3,3,12]` | T1 (12-ch!) | no (conv) |
| `conv_in.bias` | `conv_in.bias` | `[320]` | — | no |
| `conv_norm_out.{weight,bias}` | same | `[320]` | — | no (1-D GroupNorm) |
| `conv_out.weight` | `conv_out.weight` | `[4,320,3,3]`→`[4,3,3,320]` | T1 | no (conv) |
| `conv_out.bias` | `conv_out.bias` | `[4]` | — | no |
| `time_embedding.linear_1.{weight,bias}` | same | `[1280,320]`,`[1280]` | — | no (min<512) |
| `time_embedding.linear_2.{weight,bias}` | same | `[1280,1280]`,`[1280]` | — | **yes** (weight) |
| `image_proj_model_dino.proj.{weight,bias}` | same | `[4096,1536]`,`[4096]` | — | **yes** (weight) |
| `image_proj_model_dino.norm.{weight,bias}` | same | `[1024]` | — | no (1-D LayerNorm) |
| `learned_text_clip_albedo` | same | `[77,1024]` | — | no (not `.weight`) |
| `learned_text_clip_mr` | same | `[77,1024]` | — | no |
| `learned_text_clip_ref` | same | `[77,1024]` | — | no |

`image_proj_model_dino` = the DINO `ImageProjModel` (Linear 1536→4·1024 + LayerNorm 1024). It lives
PHYSICALLY in the unet `.bin` (registered on `self.unet`), so it is written to `unet.safetensors`,
NOT `dino.safetensors`. The DINO/ImageProjModel engine task (P2-5) reads it from here. The three
`learned_text_clip_*` `[77,1024]` are the material prompt embeddings that replace CLIP text
(`albedo`/`mr` for the two generated streams, `ref` for the reference stream).

### 4b. Resnets / downsamplers / upsamplers (conv-only, per T1)

| Source (after `unet.`) | Canonical | Transform | Quant? |
|---|---|---|---|
| `{down,mid,up}_blocks.N.resnets.N.norm1.{weight,bias}` | same | — | no (GroupNorm) |
| `…resnets.N.conv1.weight` / `.conv2.weight` | same | T1 (`[O,I,3,3]`→`[O,3,3,I]`) | no |
| `…resnets.N.conv1.bias` / `.conv2.bias` / `.norm2.*` | same | — | no |
| `…resnets.N.time_emb_proj.{weight,bias}` | same | — | no (Linear, but in=1280 out∈{320,640,1280}; min<512 except 1280×1280 → those quantize) |
| `…resnets.N.conv_shortcut.weight` | same | T2 (`[O,I,1,1]`→`[O,1,1,I]`) | no |
| `…resnets.N.conv_shortcut.bias` | same | — | no |
| `down_blocks.N.downsamplers.0.conv.{weight,bias}` | same | T1 (weight) | no |
| `up_blocks.N.upsamplers.0.conv.{weight,bias}` | same | T1 (weight) | no |

### 4c. Attention module wrapper (per `Transformer2DModel`)

| Source (after `unet.`) | Canonical | Transform | Quant? |
|---|---|---|---|
| `…attentions.N.norm.{weight,bias}` | same | — | no (GroupNorm) |
| `…attentions.N.proj_in.{weight,bias}` | same | — | Linear; quant iff min-dim≥512 (1280 blocks yes, 320 no) |
| `…attentions.N.proj_out.{weight,bias}` | same | — | same |

### 4d. Transformer block (base `BasicTransformerBlock` — the `.transformer.*` keys, T3 strips it)

Prefix `…attentions.N.transformer_blocks.N.transformer.` → canonical `…transformer_blocks.N.`.

| Source subkey (under `.transformer.`) | Canonical subkey | Transform | Quant? |
|---|---|---|---|
| `norm1.{weight,bias}` / `norm2.*` / `norm3.*` | `norm1/2/3.*` | — | no (LayerNorm) |
| `attn1.to_q.weight` / `to_k` / `to_v` | `attn1.to_q/to_k/to_v.weight` | T3 | yes iff ≥512 (self-attn, no bias) |
| `attn1.to_out.0.{weight,bias}` | `attn1.to_out.{weight,bias}` | T3,T5 | weight quant iff ≥512 |
| `attn1.processor.to_q_mr.weight` / `to_k_mr` / `to_v_mr` | `attn1.to_q_mr/to_k_mr/to_v_mr.weight` | T3,T4 | yes iff ≥512 (MDA) |
| `attn1.processor.to_out_mr.0.{weight,bias}` | `attn1.to_out_mr.{weight,bias}` | T3,T4,T5 | weight iff ≥512 |
| `attn2.to_q.weight` / `to_k` / `to_v` | `attn2.to_q/to_k/to_v.weight` | T3 | to_q/out iff ≥512; to_k/to_v are `[C,1024]` (cross to text) |
| `attn2.to_out.0.{weight,bias}` | `attn2.to_out.{weight,bias}` | T3,T5 | weight iff ≥512 |
| `ff.net.0.proj.{weight,bias}` | `ff.net.0.proj.{weight,bias}` | T3 | **GeGLU** (T7); weight `[2·inner,dim]`; quant iff ≥512 |
| `ff.net.2.{weight,bias}` | `ff.net.2.{weight,bias}` | T3 | output Linear `[dim,inner]`; quant iff ≥512 |

### 4e. Transformer block 2.5D additions (siblings of `.transformer`, on the wrapper)

Prefix `…attentions.N.transformer_blocks.N.` (no `.transformer.`).

| Source subkey | Canonical | Meaning | Transform | Quant? |
|---|---|---|---|---|
| `attn_multiview.to_q/to_k/to_v.weight` | same | MA (cross-view, `PoseRoPEAttnProcessor2_0`) | T5 on to_out | yes iff ≥512 |
| `attn_multiview.to_out.0.{weight,bias}` | `attn_multiview.to_out.*` | | T5 | weight iff ≥512 |
| `attn_refview.to_q/to_k/to_v.weight` | same | RA (reference, `RefAttnProcessor2_0`) | | yes iff ≥512 |
| `attn_refview.to_out.0.{weight,bias}` | `attn_refview.to_out.*` | | T5 | weight iff ≥512 |
| `attn_refview.processor.to_v_mr.weight` | `attn_refview.to_v_mr.weight` | RA per-material V | T4 | yes iff ≥512 |
| `attn_refview.processor.to_out_mr.0.{weight,bias}` | `attn_refview.to_out_mr.*` | RA per-material out | T4,T5 | weight iff ≥512 |
| `attn_dino.to_q.weight` | same | DINO cross-attn (Q from latent) | | yes iff ≥512 |
| `attn_dino.to_k/to_v.weight` | same | K/V from projected DINO tokens `[C,1024]` | | yes iff ≥512 |
| `attn_dino.to_out.0.{weight,bias}` | `attn_dino.to_out.*` | | T5 | weight iff ≥512 |

RA note: reference attention shares Q/K with the base self-attn but has a per-material `to_v_mr` +
`to_out_mr` (only `mr`, not `albedo` — `albedo` reuses base `to_v`/`to_out`). MA/RA/DINO output
projections are zero-initialized residual paths in training but are FULL trained tensors in the
shipped checkpoint — convert them all.

---

## 5. `unet_dual.safetensors` — reference-stream UNet (686 tensors)

`copy.deepcopy(unet)` taken BEFORE the 12-ch `conv_in` swap and BEFORE the 2.5D additions, then
wrapped with `Basic2p5DTransformerBlock(use_ma=use_ra=use_mda=use_dino=False)`. So it is a **plain
SD2.1 UNet**: identical to §4 MINUS every 2.5D addition (§4e), MINUS the MDA `to_*_mr` (§4d),
MINUS `image_proj_model_dino`/`learned_text_clip_*` — AND with a **4-channel `conv_in`**
(`[320,4,3,3]`→`[320,3,3,4]`), because the dual stream consumes clean reference latents (4 ch), not
the normal/position-concatenated 12-ch input.

Its transformer-block base keys still carry `.transformer.` (T3 strips it), so after conversion the
`unet_dual` base block namespace is **byte-identical** to the `unet` base block namespace — the
P2-7 "base SD2.1 UNet" loader can read both with one code path. Same transforms (T1/T3/T5/T6),
same quant rule. Only `conv_in.weight` differs in channel count. Prefix `unet_dual.` stripped.

---

## 6. `vae.safetensors` — SD2.x AutoencoderKL (248 tensors, ENCODER + DECODER)

Standard diffusers `AutoencoderKL`, source dtype **fp32** → cast fp16 (T6). Convert BOTH encoder and
decoder (the paint pipeline VAE-encodes reference/normal/position images AND decodes generated
latents). **VAE linears are NOT quantized** (stays fp16 regardless of `--bits`; the VAE is 335 MB
and quant hurts reconstruction). Mid-block self-attention uses the LEGACY diffusers naming
(`query`/`key`/`value`/`proj_attn` + `group_norm`) — preserved verbatim, NOT renamed to `to_q` etc.

| Source key group | Count | Canonical | Transform | Quant? |
|---|---|---|---|---|
| `encoder.conv_in.weight` `[128,3,3,3]` | 1 | same | T1 | no |
| `encoder.down_blocks.N.resnets.N.{conv1,conv2}.weight` | 16 | same | T1 | no |
| `encoder.down_blocks.N.resnets.N.conv_shortcut.weight` `[…,1,1]` | 2 | same | T2 | no |
| `encoder.down_blocks.N.downsamplers.0.conv.weight` | 3 | same | T1 | no |
| `encoder.mid_block.resnets.N.{conv1,conv2}.weight` | 4 | same | T1 | no |
| `encoder.mid_block.attentions.0.{query,key,value,proj_attn}.{weight,bias}` `[512,512]`,`[512]` | 8 | same | — | **no** (VAE never quant) |
| `encoder.mid_block.attentions.0.group_norm.{weight,bias}` | 2 | same | — | no |
| `encoder.conv_out.weight` `[8,512,3,3]` | 1 | same | T1 | no |
| all `encoder.*.{bias, norm*.weight/bias}` (1-D) | rest | same | — | no |
| `decoder.conv_in.weight` `[512,4,3,3]` | 1 | same | T1 | no |
| `decoder.up_blocks.N.resnets.N.{conv1,conv2}.weight` | 12 | same | T1 | no |
| `decoder.up_blocks.N.resnets.N.conv_shortcut.weight` `[…,1,1]` | 2 | same | T2 | no |
| `decoder.up_blocks.N.upsamplers.0.conv.weight` | 3 | same | T1 | no |
| `decoder.mid_block.resnets.N.{conv1,conv2}.weight` | 4 | same | T1 | no |
| `decoder.mid_block.attentions.0.{query,key,value,proj_attn,group_norm}.*` | 10 | same | — | no |
| `decoder.conv_out.weight` `[3,128,3,3]` | 1 | same | T1 | no |
| `quant_conv.weight` `[8,8,1,1]`→`[8,1,1,8]` / `.bias` | 2 | same | T2 | no |
| `post_quant_conv.weight` `[4,4,1,1]`→`[4,1,1,4]` / `.bias` | 2 | same | T2 | no |
| all `decoder.*.{bias, norm*.weight/bias}` (1-D) | rest | same | — | no |

Group counts (verified): encoder 106, decoder 138, quant_conv 2, post_quant_conv 2 = 248.

---

## 7. `dino.safetensors` — DINOv2-giant (726 tensors; 727 − 1 dropped)

HF `Dinov2Model` (`facebook/dinov2-giant`): 40 layers, hidden 1536, 24 heads (head_dim 64), patch
14, image 518 → 1370 tokens, **SwiGLU FFN**, LayerScale, qkv-bias. Source fp32 → fp16 (T6).
Re-canonicalized to the SHAPE-stage conditioner's short scheme (so P2-5 can share helpers with the
P1 DINO port) — **except the MLP**, which is SwiGLU here (giant) vs gelu-MLP in the P1 large model,
so it uses distinct `w_in`/`w_out` names to signal the SwiGLU split (T7).

| Source key | Canonical | Shape | Transform | Quant? |
|---|---|---|---|---|
| `embeddings.cls_token` | `cls_token` | `[1,1,1536]` | — | no |
| `embeddings.position_embeddings` | `pos_embed` | `[1,1370,1536]` | — | no |
| `embeddings.patch_embeddings.projection.weight` | `patch_embed.weight` | `[1536,3,14,14]`→`[1536,14,14,3]` | T1 | no (conv) |
| `embeddings.patch_embeddings.projection.bias` | `patch_embed.bias` | `[1536]` | — | no |
| `embeddings.mask_token` | — | `[1,1536]` | **DROPPED** (masked-pretraining token, unused at inference) | — |
| `encoder.layer.N.norm1.{weight,bias}` | `layers.N.norm1.{weight,bias}` | `[1536]` | — | no |
| `encoder.layer.N.attention.attention.query.{weight,bias}` | `layers.N.attn.q.{weight,bias}` | `[1536,1536]`,`[1536]` | — | weight **yes** |
| `encoder.layer.N.attention.attention.key.{weight,bias}` | `layers.N.attn.k.{weight,bias}` | | — | weight yes |
| `encoder.layer.N.attention.attention.value.{weight,bias}` | `layers.N.attn.v.{weight,bias}` | | — | weight yes |
| `encoder.layer.N.attention.output.dense.{weight,bias}` | `layers.N.attn.out.{weight,bias}` | `[1536,1536]` | — | weight yes |
| `encoder.layer.N.layer_scale1.lambda1` | `layers.N.ls1` | `[1536]` | — | no (not `.weight`) |
| `encoder.layer.N.norm2.{weight,bias}` | `layers.N.norm2.{weight,bias}` | `[1536]` | — | no |
| `encoder.layer.N.mlp.weights_in.{weight,bias}` | `layers.N.mlp.w_in.{weight,bias}` | `[8192,1536]`,`[8192]` | — | weight **yes** (SwiGLU, T7) |
| `encoder.layer.N.mlp.weights_out.{weight,bias}` | `layers.N.mlp.w_out.{weight,bias}` | `[1536,4096]`,`[1536]` | — | weight yes |
| `encoder.layer.N.layer_scale2.lambda1` | `layers.N.ls2` | `[1536]` | — | no |
| `layernorm.{weight,bias}` | `norm.{weight,bias}` | `[1536]` | — | no |

Per-layer (×40): 18 tensors. Embeddings 5 (−1 dropped = 4 kept). Final norm 2. → 4 + 720 + 2 = 726.

---

## 8. Quantization (`--bits`, reuses the SHAPE `should_quantize` predicate)

- `--bits 8` (default, shipping): quantize eligible linear `.weight`s with mlx affine
  `group_size=64`, packed uint32 `.weight` + fp16 `.scales`/`.biases` (each eligible weight → 3
  tensors). **Scope: `unet`, `unet_dual`, `dino` only.** VAE is never quantized. Convs stay fp16.
- `--bits 16`: everything fp16, no quant (the parity-debug build the `HY3DP_*` oracles target).

`should_quantize(name, shape, bits)` (identical predicate to `convert_hunyuan3d_weights.py`):
quantize iff `bits==8` AND name ends `.weight` AND `ndim ∈ {2,3}` (excludes 1-D norms/biases and
4-D convs) AND `last_dim % 64 == 0` AND `min(last_two_dims) ≥ 512`. Consequences here:

- Quantized: all 1280-dim UNet attention/proj/ff linears, `time_embedding.linear_2`,
  `image_proj_model_dino.proj`, every DINO attn + SwiGLU linear.
- **Left fp16** (min-dim < 512): all 320-dim down-block linears (`to_q/k/v`, `proj_in/out`,
  `ff.net.0.proj` `[2560,320]`, `ff.net.2` `[320,1280]`, `attn2.to_k/v` `[320,1024]`),
  `time_embedding.linear_1`, `image_proj_model_dino.norm`.
- Never: convs (ndim 4, incl. OHWI + 1×1), `learned_text_clip_*` / `cls_token` / `pos_embed` /
  `ls1` / `ls2` (not `.weight`), all 1-D norms/biases, entire VAE.

All quantized weights satisfy `in % 64 == 0` (verified: in ∈ {1024,1280,1536,4096,5120} for the
eligible set) so mlx affine quant is valid.

---

## 9. Flags / contradictions (per the "flag loudly" instruction)

1. **⚠ `unet/config.json` `in_channels:4` is WRONG for the runtime UNet.** `from_pretrained`
   (`modules.py:818`) replaces `conv_in` with a 12-channel conv (latent 4 + normal-embeds 4 +
   position-embeds 4, concatenated in `UNet2p5DConditionModel.forward`). The checkpoint tensor is
   `[320,12,3,3]` — the config lies. `config.json.unet.in_channels` is set to **12** here. (The
   dual UNet keeps 4 — verified `unet_dual.conv_in.weight` is `[320,4,3,3]`.)

2. **⚠ Divergence from the crib's "1×1 convs → linear" rule (T2).** The design-review note said
   flatten 1×1 convs to Linear. This contract instead keeps them 4-D OHWI `[O,1,1,I]`, matching the
   **established mlx-serve convention**: both `flux.zig` and `krea.zig` VAEs run `conv_shortcut` /
   `quant_conv` / `post_quant_conv` through the same `mlx_conv2d` (pad 0) helper as 3×3 convs — none
   flatten to Linear. Keeping 4-D lets the P2-4 SD-VAE engine reuse that helper verbatim; flattening
   would force a special-cased matmul path. **If the P2 design doc truly wants Linear**, this is the
   one rule to change (in `T2` + `conv_to_ohwi` skip for 1×1) — please confirm.

3. **⚠ `vae/config.json` omits `scaling_factor`.** The shipped VAE config has no `scaling_factor`
   key; `encode_images` reads `vae.config.scaling_factor`, so diffusers falls back to the SD default
   **0.18215**, written into `config.json.vae.scaling_factor`. Confirm this is the intended value
   (it is the standard SD2.x latent scale; no evidence of an override in the paint repo).

4. **Two DINOs, two variants.** The SHAPE stage uses DINOv2-**large** (24 layers, hidden 1024,
   gelu-MLP); the PAINT stage uses DINOv2-**giant** (40 layers, hidden 1536, SwiGLU). The paint
   DINO is a SEPARATE HF download (`facebook/dinov2-giant`), not embedded in the paint checkpoint
   (contrast the shape conditioner, which rides inside the DiT ckpt). Do not cross-wire them.

5. **`image_proj_model_dino` file placement is a judgment call.** It is the DINO projection but
   lives physically in the unet `.bin`, so it is written to `unet.safetensors` (§4a). Documented so
   P2-5 (DINO/ImageProjModel) knows to read it from the UNet file, not `dino.safetensors`.

6. **No contradictions found** in the attention structure vs the stated facts: MDA/RA/MA/DINO
   additions, learned text tokens (`albedo`/`mr`/`ref` all `[77,1024]`), and the ImageProjModel
   (`proj` `[4096,1536]`, `norm` `[1024]`) match the reference `modules.py` exactly. All attentions
   are UNFUSED (no de-interleave needed).
