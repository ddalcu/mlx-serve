# SeedVR2 — transcribed architecture spec

**TRANSCRIPTION FIXTURE. This document is NOT executable and was NOT run against the
reference.** It is a hand transcription of ByteDance-Seed/SeedVR (Apache-2.0) at
`main`, files `models/dit_v2/*`, `models/video_vae_v3/*`, `projects/video_diffusion_sr/infer.py`,
`projects/inference_seedvr2_3b.py`, `configs_3b/main.yaml`. Every numeric claim here is
worth exactly what the parity fixture that pins it is worth — see
`tests/dump_seedvr2_fixtures.py`. When this file and a dumped fixture disagree, the
fixture wins and this file is the bug.

Upstream also ships `models/dit/` (v1, SeedVR1). **SeedVR2 3B uses `models/dit_v2`;
SeedVR2 7B's config points at `models.dit.nadit` (v1).** Do not read v1 for the 3B port.

## 0. What runs at inference

Three artifacts, no text encoder:

| File | Role | Size |
|---|---|---|
| `seedvr2_ema_3b.pth` (or `_fp16.safetensors`) | NaDiT | 13.5 GB fp32 / 6.8 GB fp16 |
| `ema_vae.pth` (or `ema_vae_fp16.safetensors`) | causal 3D video VAE | 1.0 GB / 0.5 GB |
| `pos_emb.pt`, `neg_emb.pt` | precomputed text embeddings, `[L, 5120]` | 0.6 MB |

`neg_emb` is only read when `cfg_scale != 1.0`. The shipped SeedVR2 inference entry
(`inference_seedvr2_3b.py`) defaults `cfg_scale=1.0` and `sample_steps=1`, so the
one-step path never runs the unconditional forward. **Ship the pos embedding only;
gate neg on a cfg_scale request field that we do not need to expose in v1.**

## 1. NaDiT (3B) — `configs_3b/main.yaml`

```
vid_in_channels 33   vid_out_channels 16   vid_dim 2560
txt_in_dim 5120      txt_dim 2560          emb_dim 15360 (= 6*vid_dim)
heads 20             head_dim 128          expand_ratio 4
norm fusedrms        norm_eps 1e-5         qk_norm fusedrms   qk_bias False
ada single           patch_size (1,2,2)    num_layers 32      mm_layers 10
mlp_type swiglu      block_type mmdit_sr   rope mmrope3d      rope_dim 128
vid_out_norm fusedrms
txt_in_norm fusedln  (txt_in_norm_scale_factor 0.01 — INIT only, checkpoint carries it)
window (4,3,3) every layer
window_method alternating ["720pwin_by_size_bysize", "720pswin_by_size_bysize"]
              i.e. EVEN layers unshifted, ODD layers shifted
```

`emb_dim == 6 * dim` is asserted by `AdaSingle`. 6 = 2 layers (attn, mlp) × 3
(shift, scale, gate).

### 1.1 Weight naming — `shared_weights` flips at layer 10

`MMModule` (models/dit_v2/mm.py) stores EITHER `.vid`+`.txt` submodules
(`shared_weights=False`) OR a single `.all` (`shared_weights=True`).

`shared_weights = not (i < mm_layers)` with `mm_layers=10`:

- **layers 0–9**: `shared_weights=False` → separate `.vid` / `.txt` weights.
- **layers 10–31**: `shared_weights=True` → one `.all` weight serving both streams.

So `blocks.0.attn.proj_qkv.vid.weight` but `blocks.10.attn.proj_qkv.all.weight`.
A loader that assumes one naming for all 32 layers loads 22 layers of garbage
**without erroring** — every shape matches. Pin the layer-10 boundary with a
weight-name assertion at load, not a shape check.

#### 1.1.1 The last block treats text strangely, in three separate ways

`is_last_layer` (i == 31) sets `vid_only=True` on `mlp_norm`, `mlp` **and `ada`**.
This is a FORWARD-PATH fact, not a naming one — by layer 31 the branch is already
`.shared`, so every tensor is named `.all` exactly like layers 10–30 and nothing is
missing from the checkpoint. What changes is what RUNS:

1. **No txt MLP.** `mlp_norm` and `mlp` are vid-only, so text skips both.
2. **No txt modulation at all — including attn.** `self.ada` is ONE `MMModule`
   covering both the `attn` and `mlp` layers, so `vid_only` strips the text's
   `attn` shift/scale AND its `attn` gate, not just the mlp ones. Only `attn_norm`
   still applies to text — that module is built without `vid_only`. Modulating text
   here anyway keeps magnitudes plausible and bends the direction: cos 0.9998 where
   every other block scores 1.000000.
3. **The text output is DOUBLED.** `MMModule` with `vid_only` *passes txt through
   unchanged* rather than skipping the stage, and the residual add is unconditional:
   `txt_mlp, ... = (txt_mlp + txt_attn)`. Since `txt_mlp` **is** `txt_attn` by then,
   text is added to itself. Returning `t1` instead of `2*t1` shows up as a max
   relative error of exactly 0.5.

Nothing downstream consumes the final block's text, so none of this changes the
model's output. Reproduce it anyway: a port that quietly disagrees with the
reference somewhere harmless is a port you cannot use to localise a bug somewhere
harmful.

#### 1.1.2 Attention runs in bf16, and so does its OUTPUT

`mmattn.py` casts q/k/v with `.bfloat16()` before `flash_attn_varlen_func` (flash-attn
takes no f32), and the result comes back **bf16** — flash-attn returns the dtype it
was given. The trailing `.type_as(vid_q)` upcasts an already-rounded tensor and
recovers nothing.

So the reference's post-attention activations sit on the bf16 grid, and no
implementation can agree with them to better than one bf16 ulp, `2^-8 = 3.9e-3`.
Computing attention in f32 is *more* accurate and lands ~3.9e-3 away, which reads
exactly like a formula bug; rounding only the inputs still leaves ~3.3e-3. Set
parity bars downstream of attention at bf16 resolution and let the COSINE carry the
structural signal.

### 1.2 Block forward — `NaMMSRTransformerBlock`

```
hid_len   = (vid_len, txt_len)
a,b       = attn_norm(vid, txt)                 # RMS, elementwise_affine=FALSE
a,b       = ada(a, b, layer="attn", mode="in")
a,b       = NaSwinAttention(a, b, vid_shape, txt_shape)
a,b       = ada(a, b, layer="attn", mode="out")
vid,txt   = a+vid, b+txt
m,n       = mlp_norm(vid, txt)                  # RMS, affine=FALSE
m,n       = ada(m, n, layer="mlp", mode="in")
m,n       = mlp(m, n)                           # SwiGLU
m,n       = ada(m, n, layer="mlp", mode="out")
vid,txt   = m+vid, n+txt
```

Both norms are **affine-free** RMSNorm; all the per-channel affine lives in `AdaSingle`.

### 1.3 AdaSingle (`models/dit_v2/modulation.py`)

`emb` is `[b, 6*dim]` reshaped `"b (d l g) -> b d l g"` with `l=2, g=3`. **`d` is the
slowest axis**: element `(d, l, g)` sits at flat index `d*6 + l*3 + g`. A naive
`[3, 2, dim]` view is transposed and silently produces plausible-looking garbage.

`layers = ["attn", "mlp"]`, so `idx = 0` for attn, `1` for mlp. After indexing:
`shiftA, scaleA, gateA = emb[..., idx, :].unbind(-1)`.

Learned per-block parameters add to the timestep-derived ones:

```
mode "in" : hid * (scaleA + attn_scale) + (shiftA + attn_shift)
mode "out": hid * (gateA  + attn_gate)
```

`{layer}_scale` is initialised centred at **1**, `_shift`/`_gate` centred at 0 —
irrelevant for loading, but it means a checkpoint's `attn_scale` near 1.0 and
`attn_gate` near 0.0 is the *expected* magnitude. Values near 0 for `_scale` mean
you loaded the wrong tensor.

`vid_out_ada` uses `layers=["out"], modes=["in"]` — one `out_shift`/`out_scale` pair,
**no gate**, applied after `vid_out_norm` and before `vid_out`.

#### 1.3.1 `vid_out_ada` is a CACHE-KEY COLLISION, and the collision is the spec

Read literally, `vid_out_ada` cannot run. Its `layers` list has ONE entry, so
`rearrange(emb, "b (d l g) -> b d l g", l=1, g=3)` yields `d = emb_dim/3 = 2*dim`,
which cannot broadcast against a `dim`-wide hidden or a `dim`-wide `out_scale`.
Verified empirically — it raises.

What saves it is `AdaSingle`'s memo key: `f"emb_repeat_{idx}_{branch_tag}"`.
`self.layers.index("out")` is **0**, and the branch tag is **"vid"** — precisely the
key every block's `attn` ada already populated with a correctly-shaped
`(L, dim, 3)` tensor. So the cache HITS, the malformed lambda is never evaluated,
and `vid_out_ada` silently reuses the **attn** modulation slice:

```
vid = vid_out_norm(vid)
vid = vid * (scale_attn_slot + out_scale) + (shift_attn_slot + out_shift)
```

where `scale_attn_slot` / `shift_attn_slot` come from `emb`'s layer-0 (attn) slot
under the **2-layer** `(d, l=2, g=3)` decomposition — not from any 1-layer view.

This is identical in all three implementations (ByteDance reference, numz's
ComfyUI port, ComfyUI core), and ComfyUI core explicitly enables it for the 3B
(`dit_config["vid_out_norm"] = True` in `model_detection.py`), so it is what the
trained weights expect. It is not a bug to route around.

Two consequences for anyone touching this:

- **`disable_cache=True` breaks the reference.** The dump script must run with the
  cache ON; that flag is load-bearing semantics, not a performance switch.
- A port that "fixes" the 1-layer slice by taking `emb[..., :3*dim]` gets a
  DIFFERENT slice and is wrong. Take the attn slot of the 2-layer decomposition.

### 1.4 SwiGLU MLP

```
hidden = ceil_to_256( int(2 * dim * expand_ratio / 3) )
       = ceil_to_256( int(2*2560*4/3) = 6826 ) = 6912
proj_out( silu(proj_in_gate(x)) * proj_in(x) )    # all bias=False
```

Note the naming: **`proj_in_gate` is the SiLU'd branch, `proj_in` is the linear
branch.** Swapping them is a valid-shaped, wrong-output load.

### 1.5 TimeEmbedding

`get_timestep_embedding(t, 256, flip_sin_to_cos=False, downscale_freq_shift=0)` →
`proj_in(256→2560)` → SiLU → `proj_hid(2560→2560)` → SiLU → `proj_out(2560→15360)`.

`flip_sin_to_cos=False` + `downscale_freq_shift=0` is diffusers' **sin-first**
ordering with `exp(-log(10000) * arange(128) / 128)`. Diffusers' default is
`flip_sin_to_cos=True`; getting this backwards swaps the halves of the 256-vector.
There is no `scale` argument, so scale=1 and the input timestep is in **[0,1000]**,
not [0,1].

### 1.6 Patchify — `patch_size (1,2,2)`

`t == 1`, so the `t > 1` frame-replication branches in `NaPatchIn`/`NaPatchOut` are
**dead code for this config**. Do not port them.

```
in : (T, H, W, 33) -> (T, H/2, W/2, 2*2*33=132) -> Linear(132, 2560)
out: (T,H/2,W/2,2560) -> Linear(2560, 2*2*16=64) -> (T, H, W, 16)
```

Channel order inside the patch is `(h w c)` — `rearrange("(H h) (W w) c -> H W (h w c)")`,
so c is fastest, then w, then h.

### 1.7 mmrope3d (`models/dit_v2/rope.py`)

`NaMMRotaryEmbedding3d(dim=rope_dim=128)` → `RotaryEmbedding(dim=128//3=42,
freqs_for="lang", theta=10000)`.

- per-axis freqs: `1 / 10000^(arange(0,42,2)[:21] / 42)` → **21 freqs**, then each
  repeated twice (`'... n -> ... (n r)', r=2`) → **42 values per axis**.
- axial over 3 axes → `cat` → **126 values**. `head_dim` is 128, so **dims 126:128
  are NOT rotated**. This is not a bug to fix — it is the checkpoint's contract.
- `apply_rotary_emb` uses the `rotate_half` convention **on the interleaved-pairs
  layout** (`r=2` repeat means freq j covers dims 2j, 2j+1).

Position assignment (`get_freqs`), with `l = txt_len`:

```
vid_freqs = axial(1024, 128, 128);  vid uses [l : l+f, :h, :w]
txt_freqs = axial(1024)          ;  txt uses [:l] tiled 3x → 126
```

**The video temporal axis is offset by the text length `l`.** Text occupies temporal
positions `0..l-1`, video `l..l+f-1`. Under windowing, `self.rope` is called with
`window_shape` — the window's own `(t,h,w)` extent — so **RoPE is window-LOCAL**:
every window restarts at `(l, 0, 0)`. There are no global window offsets.

## 2. Adaptive window attention — `models/dit_v2/window.py`

This is the paper's contribution and the highest-risk piece of the port.

```python
scale        = sqrt((45*80) / (h*w))          # 45x80 = 720p at /8 VAE /2 patch
resized_h,w  = round(h*scale), round(w*scale)
wh, ww       = ceil(resized_h/3), ceil(resized_w/3)     # num_windows = (4,3,3)
wt           = ceil(min(t,30) / 4)
```

The window **size** is computed from the 720p-equivalent grid; the window **count**
`ceil(h/wh)` is then taken over the *actual* grid. So window size is fixed in 720p
tokens (15×27 for the 3×3 split) and the number of windows grows with resolution.
That is what "adaptive" means here — it is the opposite of the usual fixed-count
split, and getting it backwards still runs.

Unshifted (`720pwin_by_size_bysize`): windows tile `[i*w, min((i+1)*w, extent))`,
empty ones dropped.

Shifted (`720pswin_by_size_bysize`): `s = 0.5 if window < extent else 0` per axis;
window `i` spans `[max(int((i-s)*w), 0), min(int((i-s+1)*w), extent))`; count is
`ceil((extent-s)/w) + 1` when `s>0` else `1`; empty windows dropped. Note the
`int()` truncation, not `round`.

**Emission order is `for iw: for ih: for it:`** — w-major, t fastest. The flattened
window sequence and therefore every downstream `cumsum` depends on it.

### 2.1 What windowing does to the sequence

Each window becomes its **own attention sequence**, and the **full text sequence is
concatenated into every window** (`repeat_concat_idx`). With `n` windows the text
tokens are attended `n` times. On the way out, `unconcat_coalesce` **averages the
`n` copies of each text token** (`reshape(-1, n, ...).mean(1)`) back to one.

So: video tokens are partitioned; text tokens are broadcast then mean-pooled. A port
that concatenates text into only the first window, or that takes the last copy
instead of the mean, produces subtly-wrong-but-plausible output.

## 3. Video VAE — `s8_c16_t4_inflation_sd3.yaml`

SD3's 2D VAE inflated to causal 3D (`inflation_mode: pad`).

```
in/out_channels 3        latent_channels 16      block_out_channels [128,256,512,512]
layers_per_block 2       norm_num_groups 32      act silu
spatial_downsample 8     temporal_downsample 4   temporal_scale_num 2
use_quant_conv False     use_post_quant_conv False
slicing_sample_min_size 4
scaling_factor 0.9152    (latent = (raw - shift) * scale;  shift absent → 0)
```

No quant/post-quant conv — the encoder's final conv emits `2*16` channels
(mean, logvar) directly.

### 3.1 Causality is head-replication, not masking

`InflatedCausalConv3d` (causal_inflation_lib.py) does NOT pad in time. It sets
`padding[0] = 0` and instead replicates the **first frame** `2 * temporal_padding`
times onto the head of the sequence (`extend_head`), then convolves with zero
temporal padding. For a `k=3, p=1` conv that is: duplicate frame 0 twice, convolve,
land back on the original `T`.

The decoder's mirror is `remove_head(tensor, times)`, which keeps frame 0 and drops
the next `times` frames — `cat(tensor[:, :, :1], tensor[:, :, times+1:])`.

There is no attention mask anywhere in the VAE's temporal handling. A port that
implements causality as masking produces the right shapes and the wrong content.

`inflation_mode: pad` in the yaml only steers 2D→3D weight *inflation at init*
(`inflate_weight`). We load already-3D weights, so it is **inert for us** — do not
try to honour it at load.

### 3.2 `time_receptive_field` is "full", and the inner default lies

**CORRECTED 2026-08-10 against the real checkpoint.** An earlier revision of this
document said the resnets were `(1,3,3)` and mixed no time. That was wrong, and the
way it was wrong is the lesson.

`ResnetBlock3D`, `DownEncoderBlock3D`, `UNetMidBlock3D`, `Encoder3D` and `Decoder3D`
*all* declare `time_receptive_field: _receptive_field_t = "half"` in their own
signatures. **`VideoAutoencoderKL.__init__` declares `"full"` and passes it down to
the encoder and decoder it constructs** (attn_video_vae.py:1072), so the inner
defaults are never used. The yaml sets nothing, so "full" is what this checkpoint is.

Reading `Encoder3D`'s signature and taking its default gives you `"half"` and the
wrong kernel everywhere. Observed shapes from `ema_vae_fp16.safetensors`:

| Tensor | Shape | Kernel |
|---|---|---|
| `encoder.down_blocks.{0..3}.resnets.{0,1}.conv1` | `[C,C,3,3,3]` | `(3,3,3)` — mixes time |
| `encoder.down_blocks.0.downsamplers.0.conv` | `[128,128,1,3,3]` | `(1,3,3)` — spatial only |
| `encoder.down_blocks.{1,2}.downsamplers.0.conv` | `[C,C,3,3,3]` | `(3,3,3)` — temporal + spatial |
| `encoder.down_blocks.3.downsamplers` | *absent* | no downsampler |

So under "full": `kernel=(3,3,3)`, `padding=(1,1,1)` on every resnet conv, and time is
mixed almost everywhere rather than in three places.

This one fails LOUDLY — `(1,3,3)` and `(3,3,3)` are different tensor sizes, so a
wrong port dies at load rather than producing bad pixels. That makes it the *cheap*
kind of error. The ladder in §3.3 is the expensive kind, and it was confirmed
correct by the same dump.

`conv_shortcut` (kernel `(1,1,1)`) exists only where a block changes channel count —
blocks 1 (128→256) and 2 (256→512). Blocks 0 and 3 keep their width and have none.

### 3.3 The down ladder — which blocks stride, and in which axis

Four blocks, `block_out_channels [128, 256, 512, 512]`, and the two strides are
governed by *different* predicates:

```
add_downsample       = not (i == 3)                      -> i in {0,1,2}   spatial /2 each = /8
is_temporal_down     = i >= len(channels) - temporal_down_num - 1 = i >= 1 -> i in {1,2,3}
```

Block 3 is flagged `temporal_down=True` but has **no downsampler at all**, so its
temporal flag is dead. Effective temporal striding happens at blocks **1 and 2 only**
→ `/4` in time, matching `temporal_downsample_factor: 4`.

That off-by-one is deliberate upstream and easy to "fix" into `/8` temporal. The
`temporal_down_num` arithmetic and the `add_downsample` guard must both be ported,
not collapsed into one loop condition.

`Downsample3D` geometry: `kernel = (3 if temporal_down else 1, 3, 3)`,
`stride = (2 if temporal_down else 1, 2, 2)`,
`padding = (1 if temporal_down else 0, downsample_padding, downsample_padding)`.
The encoder passes `downsample_padding=0`, so **spatial padding is 0** and the
asymmetric `(0,1,0,1)` pad is applied manually in `forward` — diffusers'
`Downsample2D` convention. A symmetric `p=1` here shifts the whole feature map by
half a pixel per block.

### 3.4 Mid block

`UNetMidBlock3D` with `attention_head_dim = block_out_channels[-1] = 512`, i.e.
**single-head** attention over the 512 channels, `resnet_groups=32`, `eps=1e-6`,
`output_scale_factor=1`, `temb_channels=None`.

## 4. Sampling — one step

```
schedule   lerp, T=1000            x_t = (1-t/T)*x_0 + (t/T)*noise
sampler    euler, v_lerp           prediction is v = x_1 - x_0
timesteps  uniform_trailing, transform=True
```

`timestep_transform` (infer.py) applies a resolution-dependent shift:
`t' = shift*t / (1 + (shift-1)*t)` on `t` normalised by `T`, then rescaled by `T`.

Conditioning (`get_condition`, task `"sr"`):

```
cond = zeros[t, h, w, 17]
cond[..., :16] = latent_blur        # VAE-encoded LOW-RES input
cond[...,  16] = 1.0
vid_in = cat([x_t (16ch), cond (17ch)], dim=-1)    # 33 channels
```

`cond_noise_scale = 0.0` in the SeedVR2 entry — the condition latent is **clean**.

With `sample_steps=1` and `cfg_scale=1.0`: one Euler step from pure noise at the
single transformed timestep, conditional forward only.

## 5. Post-process

`wavelet_reconstruction` (`projects/video_diffusion_sr/color_fix.py`) transfers the
low-frequency wavelet bands of the *input* onto the output to kill colour drift.
Pure numeric, no weights — port it as a hermetic unit.

## 6. Traps, ranked by how quietly they fail

1. **`shared_weights` layer-10 boundary** (§1.1) — shapes match either way.
2. **AdaSingle `(d l g)` ordering** (§1.3) — a transposed view is the same size.
3. **Text mean-pooling across windows** (§2.1) — plausible output if you take one copy.
4. **Window size from the 720p grid, count from the real grid** (§2) — runs either way.
5. **`proj_in_gate` vs `proj_in`** (§1.4) — symmetric shapes.
6. **`flip_sin_to_cos=False`** (§1.5) — diffusers' non-default.
7. **RoPE leaves dims 126:128 unrotated, video temporal offset by `txt_len`** (§1.7).
8. **Last block has no txt MLP** (§1.1).
