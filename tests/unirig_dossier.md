# UniRig — implementation dossier (source-verified 2026-07-04)

Feeds **Phase 3** of the 3D pipeline: Hunyuan3D GLB → auto-rig → SceneKit skeletal animation.
Port target = Zig + mlx-c (Apple MLX), sibling of `src/hunyuan3d.zig`; must run on a 16 GB Mac.

All `file:line` citations are into a shallow clone at `/Volumes/Sandisk_1TB/hy3d-scratch/unirig`
(`github.com/VAST-AI-Research/UniRig`, SIGGRAPH '25). Every fact below was read from source.
`INFERRED` marks anything not literally in the code; `⚠` flags a load-bearing ambiguity or porting risk.

UniRig is **two independent models run as two sequential stages** (`README.md:39-42`):
**(1) Skeleton Prediction** — an autoregressive OPT transformer that emits a *skeleton-tree token
sequence*; **(2) Skinning** — a PTv3 point encoder + bone-point cross-attention that predicts
per-vertex per-bone weights. They do NOT share weights and are separate `.ckpt` files. There is no
single end-to-end forward — stage 2 consumes stage 1's exported skeleton `.npz`.

> ⚠ **The tractable neural port is stage 1 (skeleton).** Stage 2 (skinning) is dominated by
> PointTransformerV3 which needs space-filling-curve serialization + **submanifold sparse 3D conv** +
> windowed attention — none of which are plain transformer ops. There is a **non-neural geodesic
> skinning already in the pipeline** (`voxel_skin`, the model's own input prior) that can bootstrap
> animation without porting PTv3 at all. See §4, §8, §9.

---

## 1. End-to-end pipeline (mesh in → rigged out)

Driver `run.py` (Lightning `predict`); shell wrappers `launch/inference/generate_skeleton.sh`,
`generate_skin.sh`, `merge.sh`.

**Stage 0 — extract** (`src/data/extract.py`, needs Blender `bpy`): load mesh, decimate to
`faces_target_count=50000` via `fast_simplification` (`extract.py:346`), recompute trimesh vertex/face
normals, write `raw_data.npz` (`extract.py:save_raw_data`). ⚠ **This is a `bpy`/trimesh preprocess we
must replace.** Our Hunyuan3D output is already a clean indexed watertight mesh from
`marching_cubes.zig` — feed it directly; decimation optional (skeleton is robust to face count, skin
resamples anyway).

**Stage 1 — skeleton** (`configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml`):
1. `transform_asset` (`src/data/transform.py:55`) in fixed order: process tails → arrange bone order →
   augments (**inference: only `AugmentAffine` normalize-into-cube, all random p=0**,
   `configs/transform/inference_ar_transform.yaml`) → vertex groups (none for AR) → **sample point
   cloud** (`SamplerMix`, `num_samples=65536`, `vertex_samples=8192`).
2. `UniRigAR.generate` (`src/model/unirig_ar.py:129`): encode point cloud → 1024 mesh tokens (§3) →
   prepend `[mesh…, bos, cls]` as `inputs_embeds` → HF `transformer.generate(...)` with a grammar
   logits mask → detokenize to joints/bones/parents/tails (§2).
3. Writer (`ARWriter`, `system/ar.py:282`) inverse-transforms back to original scale and exports
   `predict_skeleton.npz` (+ `.fbx`/`.obj`).

**Stage 2 — skin** (`configs/task/quick_inference_unirig_skin.yaml`, `data_name=predict_skeleton.npz`):
re-extract the SAME mesh, load stage-1 skeleton, sample `num_samples=32768`, compute the `voxel_skin`
geodesic prior (§4), run `UniRigSkin.predict_step` → `(N,J)` weights, `reskin` back onto original
vertices (`system/skin.py:312`), export `predict_skin.npz`.

**Stage 3 — merge** (`src/inference/merge.py`, `bpy`): inject skeleton+skin into the ORIGINAL file,
brute-force axis/sign orientation search vs the original mesh (`get_correct_orientation_kdtree`,
`merge.py:153`), denormalize, top-4 weights per vertex, export rigged `.glb/.fbx/.vrm`. ⚠ We replace
this with `glb.zig` skin authoring (§9) + SceneKit — no `bpy`.

---

## 2. Skeleton tokenization (`src/tokenizer/tokenizer_part.py`, `configs/tokenizer/tokenizer_parts_articulationxl_256.yaml`)

Coordinate discretization: `num_discrete=256`, `continuous_range=[-1,1]`.
`discretize` (`tokenizer_part.py:343`): `round(clip((t-lo)/(hi-lo)*256, 0, 255))`.
`undiscretize` (`:354`): `(t+0.5)/256*(hi-lo)+lo` (bin-center dequant).

**Vocabulary (267 tokens)** — offsets assigned in `TokenizerPart.__init__` (`:22-45`):

| id range | meaning |
|---|---|
| `0..255` | discrete coordinate values (`num_discrete`) |
| `256` | `branch` (topology: next joint reparents to an explicit parent, not the previous joint) |
| `257` | `bos` |
| `258` | `eos` |
| `259` | `pad` |
| `260` | `spring` (a "part" separator with no name) |
| `261` | part `body` |
| `262` | part `hand` |
| `263` | `cls_none` |
| `264` | cls `vroid` |
| `265` | cls `mixamo` (⚠ config comment: "currently untrained, do not use") |
| `266` | cls `articulationxl` (the inference default, `assign_cls: articulationxl`) |

`vocab_size = 267` = the OPT `vocab_size` override (`unirig_ar.py:50`, `tokenizer.vocab_size`).

**Sequence layout** (`tokenize`, `:186`): `bos, cls, [ per-bone: optional part-token, optional
branch(256), coords…], eos`. Each bone emits **3 coords** (`x,y,z` of its head/joint), OR **6 coords**
when `branch` is set (`parent_xyz` then `joint_xyz`) — `:209-220`. Root is bone 0 (its parent==itself).
`branch[i] = (parent[i] != i-1)` (`asset.py:362`): a bone whose parent is NOT the immediately
preceding bone is prefixed with token 256 + the explicit parent coords. So topology is encoded
positionally (chain continuation) with `branch` markers for tree jumps. Parts (`body`/`hand`) are
inserted before the first bone of that named group (`order.py:arrange_names`).

**Max lengths**: `n_positions = max_position_embeddings = 3076` (`configs/model/…ar_350m…yaml`).
Budget = 1024 mesh prefix + 2 (bos+cls) + skeleton tokens; `generate` caps `max_new_tokens=2048`
(`configs/system/ar_inference_articulationxl.yaml`). README `:267`: "`n_positions` must > cond length
+ max skeleton tokens". Typical skeleton ≈ tens–low-hundreds of bones → a few hundred tokens.

**Detokenize** (`tokenizer_part.py:225`): strip bos + trailing pad + eos, walk coords rebuilding
`joints`/`p_joints`, then `make_skeleton` (`spec.py:207`) reconstructs parents by nearest-previous-joint,
extrudes tails for leaves/branches (`extrude_tail_for_leaf=True`, `extrude_tail_for_branch=True`,
`extrude_scale=0.5`), and `order.make_names` assigns human bone names by cls part templates
(`configs/skeleton/{vroid,mixamo}.yaml` — the fixed 22 body + 30 hand Mixamo/VRoid bone-name lists).

---

## 3. Autoregressive model (skeleton) — `src/model/unirig_ar.py`

### 3a. Transformer backbone
`AutoModelForCausalLM.from_config(AutoConfig.from_pretrained("facebook/opt-350m", **overrides))`
(`unirig_ar.py:44-56`). **It is a plain HF OPT decoder** (`OPTForCausalLM`), NOT a custom arch.
Base `facebook/opt-350m` config (fetched from HF) + these overrides (`configs/model/unirig_ar_350m_1024_81920_float32.yaml` + code):

| field | value | note |
|---|---|---|
| `num_hidden_layers` | 24 | OPT-350m base |
| `hidden_size` | 1024 | |
| `num_attention_heads` | 16 | head_dim 64 |
| `ffn_dim` | 4096 | |
| `activation_function` | **relu** | OPT default (not GELU!) |
| `word_embed_proj_dim` | **1024** (override; base 512) | removes OPT `project_in/project_out` |
| `do_layer_norm_before` | **True** (override; base False) | ⇒ pre-norm blocks + `final_layer_norm` present |
| `max_position_embeddings`/`n_positions` | **3076** (override; base 2048) | learned abs pos embed, **padding_idx offset 2** (OPT `OPTLearnedPositionalEmbedding`) |
| `vocab_size` | **267** (override; = tokenizer) | tiny embedding table |
| `torch_dtype` | float32 (forced `:53`) | ⚠ but trainer runs `precision: bf16-mixed` (autocast) at inference |
| `pre_norm` | True (forced `:55`) | ⚠ not a real OPTConfig field; `do_layer_norm_before` is the operative pre-norm switch — treat `pre_norm` as a no-op |

Positional encoding = **OPT learned absolute** (embedding table `[3076+2, 1024]`, positions derived
from the attention mask, +2 padding offset). ⚠ Because generation feeds `inputs_embeds` (not
`input_ids`), positions for the mesh-prefix + generated tokens are the running cumulative mask index.

### 3b. Conditioning = PREFIX (not cross-attention)
`generate` (`:140-163`): `cond = encode_mesh_cond(vertices,normals)` → `[1, 1024, 1024]`, then
`cond = cat([cond, embed([bos, cls])])` → `[1, 1026, 1024]` fed as `inputs_embeds`. Mesh tokens are a
**soft prefix** in the same sequence; generated tokens attend to them via normal causal self-attention.
`attention_mask` is padded with 1s over the cond span (`training_step:96`). Logits are sliced to the
post-cond span (`:104`). **This maps cleanly onto our existing decode machinery** (a learned prefix, no
new cross-attn plumbing).

### 3c. Point-cloud encoder (the "shape encoder")
`mesh_encoder = michelangelo_encoder` → `ShapeAsLatentPerceiverEncoder`
(`parse_encoder.py:5,14`; `michelangelo/models/tsal/sal_perceiver.py:532`). ⚠ **GPLv3 source** (§10).
Config (`…ar_350m…yaml` `mesh_encoder`): `width=512, heads=8, num_encoder_layers=16, num_freqs=8,
include_pi=False, point_feats=3, num_latents=512, token_num=1024, use_ln_post=True, qkv_bias=False,
init_scale=0.25, no_query=True`. Perceiver forward (`CrossAttentionEncoder._forward` else-branch,
`sal_perceiver.py:135-193`):

1. `FourierEmbedder` (`embedder.py:31`): `num_freqs=8, include_pi=False, include_input=True` ⇒
   `out_dim = 3*(8*2+1) = 51`; concat `point_feats=3` (normals) ⇒ 54 → `input_proj` Linear → 512.
2. **Subsample**: deterministic `np.random.default_rng(seed=0).choice(N, token_num*4=4096)`
   (`:145-148`) → 4096 points. ⚠ Must reproduce this RNG exactly for parity (seed 0, `replace` only if
   4096>N; here N=65536 so no replace).
3. **FPS**: `torch_cluster.fps(pre_pc, ratio=1/4, random_start=False)` (`:160`) → 1024 farthest points.
   ⚠ **Farthest-point sampling is a porting item** (not a transformer op) — need a deterministic Zig FPS
   with `random_start=False` (starts at index 0).
4. `data = input_proj(fourier(full 65536 pc + normals))` (KV); `sampled_data = input_proj(fourier(1024
   pts + normals))` (Q). `latents = cross_attn(sampled_data, data)` (1 `ResidualCrossAttentionBlock`)
   → `self_attn` (16-layer `Transformer`) → `ln_post`. Output **1024 latents × 512**.
5. `encode_latents` (`sal_perceiver.py:609`): returns ALL 1024 tokens (`latents = x`, `shape_embed=x[:,0]`
   unused here).
6. `output_proj` Linear 512→1024 (`unirig_ar.py:66,81`).

**Attention math** (`michelangelo/models/modules/transformer_blocks.py`): pre-LN residual blocks;
`c_qkv`/`c_kv` packed Linears; scale = `1/sqrt(sqrt(head_ch))` applied to BOTH q and k (`:90,201`) ⇒
net `1/sqrt(head_ch)`; **softmax in float32** then cast back (`:102,214`); MLP = Linear→GELU→Linear,
hidden ×4. `qkv_bias=False` for the AR encoder. `flash=True` in config selects an SDPA path
(`flash_attention`, `:37`) that is numerically the plain path (fp32 softmax) — port as one attention.

### 3d. Parameter/tensor map for conversion
Prefixes in the ckpt state_dict (`INFERRED` from module names):
`transformer.model.decoder.*` (OPT: `embed_tokens`, `embed_positions`, `layers.{0..23}.{self_attn.
{q,k,v,out}_proj, self_attn_layer_norm, fc1, fc2, final_layer_norm}`, `final_layer_norm`),
`transformer.lm_head`, `mesh_encoder.encoder.{input_proj, cross_attn.*, self_attn.resblocks.{0..15}.*,
ln_post}`, `mesh_encoder.fourier_embedder.frequencies` (buffer, non-persistent — recompute), `output_proj`.

---

## 4. Skinning model — `src/model/unirig_skin.py`, `configs/model/unirig_skin.yaml`

### 4a. Inputs (`_get_predict`, `unirig_skin.py:310`)
Per mesh: `vertices (B,N,3)`, `normals (B,N,3)` (N=`num_samples=32768`, `SamplerMix`),
`joints (B,J,3)`, `tails (B,J,3)`, `parents (B,J)` (-1=root), `num_bones (B)`, and the **`voxel_skin`
prior `(B,N,J)`** (§4d). All from stage-1 skeleton + the sampler. Processes vertices in **chunks of
`num_train_vertex=512`** at inference (`:347`).

### 4b. Three encoders (config dims: `feat_dim=768, num_heads=16, bone_embed_dim=1024, mlp_dim=512,
num_bone_attn=8, num_mesh_bone_attn=16, grid_size=0.005, voxel_mask=3.0`)

- **`global_encoder`** = `michelangelo_encoder` (same class/dims as §3c, `width=512, 16 layers,
  token_num=1024`) → 1024 global latents → `out_proj` 512→768 (`:258-263`). ⚠ GPLv3 source.
- **`mesh_encoder`** = `ptv3obj` = `PointTransformerV3Object` (`pointcept/models/PTv3Object.py:552`;
  **pointcept = MIT**). Input feature = `cat([vertices, normals, zeros(N,3)], -1)` = **9 channels**
  (`:361`), `coord=vertices`, `grid_size=0.005`. Config: `enc_depths=[3,3,3,6,16]`,
  `enc_channels=[32,64,128,256,384]`, `enc_num_head=[2,4,8,16,24]`, `enable_qknorm=True`,
  `layer_norm=False` (⇒ **BatchNorm1d**, `PTv3Object.py:590`), `res_linear=True`, `enc_patch_size`
  default `[1024]*5`. Output N×384 → `feat_map` Linear 384→768 (`:251,372`).
  ⚠ **run at float32 at inference** (`torch.autocast(float32)`, `:370`) "to avoid sparse-conv precision
  bugs".
- **`bone_encoder`** (`BoneEncoder`, `unirig_skin.py:132`): each joint `(base_bone-min_coord)` →
  `FrequencyPositionalEmbedding(input_dim=3)` (same 51-dim log-freq embed as §3c) → MLP
  (Linear→LN→GELU stack to 768) → then `num_bone_attn=8` `ResidualCrossAttn` layers cross-attending
  bone features into `cat([bone_feats, global_latents])` (`:165-182`). Output `(B,J,768)`.

`ResidualCrossAttn` (`unirig_skin.py:110`): `MHA(cross_attn)` (flash_attn's `MHA`) → `norm1(res+attn)`
→ `norm2(x+FFN)`, FFN = Linear(768→3072)→GELU→Linear(3072→768). ⚠ post-LN (unlike michelangelo's
pre-LN). `flash_attn.modules.mha.MHA` → port as standard MHA cross-attention.

### 4c. Bone-point cross-attention + weight head (`unirig_skin.py:380-442`)
1. `mesh_feat` refined by `num_mesh_bone_attn=16` `ResidualCrossAttn` blocks, q=mesh_feat,
   kv=`cat([bone_feat, global_latents])` (`:383`).
2. `bone_feat = kmesh(bone_feat)` → `(B, heads=16, J, 768)`; per 512-vertex chunk `cur_mesh_feat =
   qmesh(mesh_feat)` → `(B,16,N,768)`.
3. **attention weights** `softmax((q·kᵀ)/sqrt(768), dim=J)` → `(B,16,N,J)` → permute `(B,N,J,16)` →
   `attn_skin_norm` LayerNorm (`:408-414`).
4. `voxel_skin` embedded: `voxel_skin_embed` Linear(1→16) → `voxel_skin_norm` LN; concat with attn →
   `(B,N,J,32)` → `downscale` Linear(32→16)+LN+GELU (`:416-420`).
5. **`SkinweightPred`** (`:184`): 4× (Linear(16→512)→LN→GELU) → Linear(512→1). Per bone → `(N,J)` →
   `softmax over J` (`:431/434`).
6. **Post-processing** (`:437-441`, inference only): additive parent-mask accumulate of `voxel_skin`
   (`skin_mask`, `:394-401`), then `skin_pred *= skin_mask^voxel_mask(=3.0)`, renormalize over J.
   So `voxel_skin` is BOTH a network input AND a multiplicative gate on the output.

Output per vertex: a full `(J,)` softmax weight vector (dense over all bones), NOT top-k inside the
model. Top-4 selection happens at export (`merge.py:278`, `group_per_vertex=4`).

### 4d. `voxel_skin` prior (`src/data/vertex_group.py:130`, `VertexGroupVoxelSkin`)
Non-neural geodesic skinning. Config (skin transform): `grid=196, alpha=0.5, link_dis=1e-5,
grid_query=7, vertex_query=1, grid_weight=3.0, mode='square'(default), backend='pyrender'`.
1. Normalize verts/joints into unit cube by bbox (`:147-155`).
2. **Voxelize** (`voxelization`, `:282`): 6-view orthographic depth carving (pyrender) OR open3d voxel
   fill → occupied grid coords. ⚠ pyrender/open3d + OpenGL — heavy geometry to replace.
3. **`voxel_skin`** (`:434`): build a graph over `[mesh vertices ∪ grid voxels]` (mesh edges +
   grid-neighbor edges + grid↔vertex edges), attach each joint to nearest combined node,
   **scipy Dijkstra `shortest_path`** (`:507`) → per-(joint,vertex) geodesic distance →
   `weight = (1/((1-α)d + α d²))²`, normalize over joints → `(N,J)`. ⚠ **Dijkstra over ~40k+ nodes**.

> ⚠ This `voxel_skin` heuristic IS a usable, topology-aware skinning on its own (it's the model's own
> prior; the net only refines it). Recommended stage-2 fallback (§9): implement voxelize+Dijkstra in
> Zig (we already have mesh/marching-cubes geometry code) and ship `voxel_skin` directly, deferring the
> full PTv3 neural refiner.

---

## 5. Sampling / inference procedure (skeleton decode)

`generate` (`unirig_ar.py:156`) → HF `transformer.generate(inputs_embeds=cond, logits_processor=[
VocabSwitchingLogitsProcessor], **generate_kwargs)`. Kwargs
(`configs/system/ar_inference_articulationxl.yaml`):

```
max_new_tokens=2048, num_return_sequences=1, num_beams=15, do_sample=True,
top_k=5, top_p=0.95, repetition_penalty=3.0, temperature=1.5, assign_cls=articulationxl
```

⚠ **This is HF "beam-search multinomial sampling" (num_beams=15 AND do_sample=True).** Porting risk:
we currently have greedy/top-k/top-p single-stream sampling but **no beam search**. A faithful port
needs 15-beam search with per-beam temperature=1.5 / top_k=5 / top_p=0.95 / repetition_penalty=3.0,
selecting the best beam. INFERRED-acceptable simplification for v1: single-beam sampled decode
(num_beams=1) — will change outputs but still yields valid skeletons; flag as a fidelity gap.

**Grammar constraint** (`VocabSwitchingLogitsProcessor`, `unirig_ar.py:14`): every step masks logits to
`tokenizer.next_posible_token(bos ++ generated)` (`tokenizer_part.py:65`). This is a hard finite-state
grammar over the token classes (expect_bos → expect_cls_or_part_or_joint → expect_joint_2 →
expect_joint_3 → expect_branch_or_part_or_joint → …) that guarantees a topologically valid, parseable
skeleton (only legal next tokens get logit 0, all else −inf). ⚠ **Must port this state machine** — it's
what makes decode robust; it is deterministic and hermetically testable (no weights). `bones_in_sequence`
(`:146`) counts completed bones.

Repetition penalty 3.0 is strong (discourages coordinate repeats). Temperature 1.5 > 1 (more diverse).
`seed` (default 12345/123) → `L.seed_everything`, so a fixed seed makes the FPS + sampling reproducible.

---

## 6. Checkpoints (HF `VAST-AI/UniRig`, queried 2026-07-04)

| file | bytes | ~size | role |
|---|---|---|---|
| `skeleton/articulation-xl_quantization_256/model.ckpt` | 1,439,617,174 | 1.44 GB | **stage-1 AR** (this dossier's target); ≈360M params fp32, weights-only |
| `skin/articulation-xl/model.ckpt` | 4,375,464,854 | 4.38 GB | **stage-2 skin**; ≈360–400M params but ckpt is ~3× → **⚠ almost certainly includes Adam optimizer state** |
| `skeleton/rignet/model.ckpt` | 621,660,032 | 0.62 GB | RigNet-benchmark AR variant (hidden 512, 8 enc layers, `unirig_rignet.yaml`) — NOT needed |
| `data/rigxl/processed.7z`, `data/rignet/processed.zip` | — | — | training data, ignore |

- Format: **PyTorch Lightning `.ckpt`** (pickle), weights under `state_dict` key, keys prefixed
  `model.` (Lightning wraps the `ModelSpec` as `self.model`). ⚠ `download.py` maps
  `experiments/.../model.ckpt` → HF `repo_id='VAST-AI/UniRig'`. Skinning + Rig-XL/VRoid variants "released
  separately at a later date" (HF card) — only Articulation-XL skeleton+skin are public now.
- Encoders are trained end-to-end (`pretrained_path: ~`, `freeze_encoder: False`) — **all weights are in
  the two ckpts**; no separate Michelangelo/PTv3 download.
- **dtype**: model built float32; ckpt stores fp32 (`RawData` uses fp16 for *mesh data*, not weights).
- **Conversion** (`tests/convert_unirig_weights.py`, mirror `convert_hunyuan3d_weights.py`): load ckpt,
  take `state_dict`, **strip optimizer/EMA/non-`model.` keys**, drop non-persistent `frequencies`
  buffers (recompute), emit safetensors. ⚠ verify skin ckpt actually carries optimizer state (extract
  only `model.*`) before assuming 4.38 GB of weights.
- **Memory** (16 GB Mac, INFERRED from param counts): AR ≈360M → fp16 ~0.7 GB / 8-bit ~0.36 GB;
  skin ≈0.4B → fp16 ~0.8 GB / 8-bit ~0.4 GB. Both comfortable. Runtime KV: mesh prefix is 1024 tokens
  × 24 layers × 1024 dim — modest. README `:394`: reference needs ≥8 GB VRAM for generation.
  ⚠ The 65536-point encoder cross-attn (1024 Q × 65536 KV × 8 heads) is the AR memory spike — chunk KV
  if needed. PTv3 at N=32768 with patch attention is the skin spike (reference training took ≥60 GB but
  that's batch-2 training; single-mesh inference is far smaller).

---

## 7. Normalization / axis / scale conventions (parity-critical)

**Forward normalize** (`AugmentAffine.transform`, `augment.py:580`): bbox over `vertices ∪ joints`
→ center `(min+max)/2` → **isotropic** scale `1/max(extent_xyz / (hi-lo))` into `normalize_into=[-1,1]`
→ shift to cube center (bias 0 for [-1,1]). Preserves aspect ratio (single max-extent scale). Normals
NOT transformed (pure translate+uniform-scale) (`:616`). This is the transform applied before both AR
and skin sampling. ⚠ **`voxel_skin` re-normalizes independently** with `scale=max_extent/2` (radius, not
diameter) (`vertex_group.py:153`) — different convention; reproduce both exactly.

**Inverse / back-to-original scale**: `AugmentAffine.inverse` (`:624`) applies `inv(trans)`. The writer
inverse-transforms predicted joints to original coords. `merge.denormalize_vertices` (`merge.py:180`)
independently rescales by `scale=max_extent/2, center=bbox_center` of the ORIGINAL mesh. ⚠ Since our
pipeline knows the exact Hunyuan3D→cube transform, we invert it directly — skip UniRig's KDTree
orientation search (`get_correct_orientation_kdtree`, `merge.py:153`), which exists only because `bpy`
re-imports lose the transform.

**Root/bind conventions**: root = bone 0, `parents[0]=None` (`raw_data.py:110`). Bones are
`(parent_joint_xyz, joint_xyz)` pairs; `joints = bones[:,3:]`, `p_joints = bones[:,:3]`
(`tokenizer/spec.py:102`). Tails via `make_skeleton` extrusion (§2). `matrix_local` (Blender bone local
axes, **Y-up**, `README.md:164`) is produced by the `bpy` extractor and used only for LBS training loss
(`unirig_skin.py:473`) — **not needed for our inference**. ⚠ **Inverse-bind matrices for glTF skinning
must be computed by us** (INFERRED): `inverseBindMatrix[j] = inverse(worldBindTransform(joint_j))`;
with only joint positions + parents (no per-bone rotation), use identity-rotation bind poses at
joint world positions (translation-only bind), or derive bone frames from parent→child direction.
This is a design decision for the SceneKit/glTF authoring, not something UniRig hands us.

**Coordinate frame**: UniRig/Blender is **Y-up** (`matrix_local` "aligned to Y-up"). ⚠ glTF is Y-up too,
but Hunyuan3D/our marching-cubes frame must be reconciled — verify axis agreement in an oracle
(a wrong axis = a skeleton rotated 90°). The merge stage's 48-way axis/sign brute force
(`merge.py:161-176`) exists precisely because axis frames drift between tools.

---

## 8. Non-transformer dependencies to replace in Zig (porting-risk register)

| dep | where | risk | port note |
|---|---|---|---|
| **`torch_cluster.fps`** (farthest-point sampling) | AR + skin michelangelo encoder (`sal_perceiver.py:160`) | MED | deterministic FPS, `random_start=False` (start idx 0), ratio 1/4 of 4096; new `src/fps.zig` |
| `np.random.default_rng(seed=0).choice` presample | `sal_perceiver.py:145` | LOW | reproduce PCG64 seed-0 choice, or accept a different fixed subsample (fidelity gap) |
| **HF beam-search multinomial `generate`** (num_beams=15) | `unirig_ar.py:156` | **HIGH** | port beam search OR simplify to sampled single-beam (fidelity gap) |
| grammar logits mask (FSM) | `tokenizer_part.py:65` | LOW | pure state machine, hermetic; port verbatim |
| **`spconv.SubMConv3d`** (submanifold sparse 3D conv, k5 stem + k3 CPE/block) | PTv3 `PTv3Object.py:299,523` | **HIGH** | no MLX sparse conv; implement via voxel-hash gather/scatter matmul, or skip PTv3 (§9) |
| **space-filling-curve serialization** (z-order, Hilbert, +transposed) | PTv3 `Point.serialization` (`utils/serialization/*`) | **HIGH** | needed to group points into attention patches; port z-order/Hilbert encoders |
| windowed "serialized attention", patch_size 1024, `flash_attn_varlen` | PTv3 `SerializedAttention` (`PTv3Object.py:63`) | MED | patch = 1024 contiguous serialized points; plain attention per patch |
| BatchNorm1d (`layer_norm=False`), QK-norm | PTv3 | LOW | fold BN into affine at convert; QK-norm = LN over head_dim |
| **pyrender/open3d voxelization** (6-view depth carve) | `voxel_skin` (`vertex_group.py:282`) | **HIGH** | replace with our own voxelizer (we have marching-cubes geometry) |
| **scipy `shortest_path` Dijkstra** over vertex∪grid graph | `voxel_skin` (`vertex_group.py:507`) + `reskin` | MED | Zig Dijkstra/graph; also used in geodesic diffusion post-proc |
| `torch_scatter.segment_csr` (min/mean reduces) | skin `min_coord`, PTv3 pooling | LOW | trivial segment reduce |
| `bpy` mesh extract + fbx/vrm/glb export | extract/merge | replaced | we feed our own mesh + author glTF skins (§9) |
| `fast_simplification` decimation | `extract.py:347` | LOW | optional; skip or use a quadric decimator |
| `reskin` kNN-median + geodesic diffusion + threshold | `system/skin.py:312` | MED | maps 32768-sample weights → original N verts; pure numpy/scipy, portable |

PTv3 note: ⚠ **`PointTransformerV3Object` has NO spatial pooling** — the `SerializedPooling`/`Unpooling`
classes exist but are unused; the encoder is a *flat* 31-block transformer that keeps all N points and
grows channels via `nn.Linear` between stages (`PTv3Object.py:608-642`). That removes octree
pooling/unpooling from the port, but serialization + sparse conv + windowed attention remain.

---

## 9. Porting plan sketch (mirror the HY3D workstream style; disjoint files)

**Recommended staging** (phase 3 split):

- **P3a — Skeleton (neural, do first).** New `src/unirig_skeleton.zig` = OPT-1024/24L decoder (reuse our
  transformer primitives; ReLU FFN, learned abs pos +2, final_layer_norm) + prefix conditioning + the
  michelangelo perceiver encoder (⚠ **clean-room from the standard perceiver / our `hunyuan3d.zig`
  cross-attn code, NOT a line port of GPLv3 `sal_perceiver.py`**, §10) + `src/fps.zig` (shared) +
  `src/unirig_tokenizer.zig` (discretize/undiscretize + FSM grammar + `make_skeleton` + bone-name
  order tables). Decode = grammar-masked sampled/beam loop. Output: joints, parents, tails, names.
- **P3b — Skin (start non-neural).** New `src/voxel_skin.zig` = our voxelizer + Dijkstra geodesic
  weights (`vertex_group.VertexGroupVoxelSkin` math) → ship as the skinning directly. This is a
  complete rig without PTv3. Later fidelity upgrade `src/unirig_skin.zig` = PTv3 (needs the §8 HIGH
  items) + bone-point cross-attn refiner, gated behind an oracle.
- **glTF skin authoring** — extend `src/glb.zig` (additive, like the texture phase): `JOINTS_0`
  (u8/u16 vec4) + `WEIGHTS_0` (f32 vec4, top-4 normalized) accessors, a `skins` array with
  `inverseBindMatrices`, joint node hierarchy from `parents`, and `mesh.skin` binding. Mirror the
  existing "u32 indices always, POSITION min/max" discipline; hermetic parse-back test + a Swift
  ModelIO/SceneKit load test (like `GLBMeshLoaderTests`/`hy3d_sphere.glb`).
- **Engine seam** — a `MeshEngine`-adjacent rig path, or a post-`generateGlb` rig hook in `gen.zig`
  (`handleMesh`): image→GLB (Hunyuan3D) → rig (UniRig) → skinned GLB. Weights local-only initially
  (`tests/convert_unirig_weights.py`), `~/.mlx-serve/models/local/unirig-*`.
- **Conversion + fixtures** — `tests/convert_unirig_weights.py` (ckpt `state_dict`→safetensors, strip
  optimizer state, fp32 debug + 8-bit ship builds) and `tests/dump_unirig_fixtures.py`
  (mirror `dump_hunyuan3d_fixtures.py`).

**Oracle taps** (mirror `HY3D_*`; float32 debug build, cos>0.99 on converged tensors, exact on
deterministic pieces):
- `UNIRIG_SKEL_ENC` — michelangelo encoder latents `[1,1024,512]` for a fixed pc+normals+**fixed FPS
  index set** (pin FPS + seed-0 presample determinism first, else the oracle is unstable).
- `UNIRIG_SKEL_PREFIX` — `output_proj` + `[mesh,bos,cls]` assembly `[1,1026,1024]`.
- `UNIRIG_SKEL_STEP` — one OPT forward: next-token logits at a fixed prefix (pins the decoder + pos
  embed + grammar mask application).
- `UNIRIG_SKEL_E2E` — full greedy (num_beams=1, do_sample=False, temp 0) token sequence for a fixed
  mesh — the integration oracle; also the hermetic FSM-grammar test (no weights).
- `UNIRIG_SKIN_GLOBAL` — michelangelo global latents `[1,1024,768]`.
- `UNIRIG_SKIN_PTV3` — PTv3 `mesh_feat [1,N,768]` (the hard oracle; only meaningful once serialization
  + sparse conv are in).
- `UNIRIG_SKIN_BONE` — `bone_encoder` output `[1,J,768]`.
- `UNIRIG_SKIN_OUT` — final `skin_pred [1,N,J]` after voxel-mask renorm.
- `UNIRIG_VOXELSKIN` — geometry-only fixture for `src/voxel_skin.zig` (no weights; validates
  voxelize+Dijkstra against the numpy reference).

⚠ **Numeric parity caveat**: model is built float32 but the reference trainer runs `bf16-mixed`
autocast at inference (`configs/task/…yaml precision: bf16-mixed`, `system/ar.py validate_cast
bfloat16`), and beam-search sampling is stochastic. So end-to-end token parity is not bit-exact; pin
the deterministic sub-pieces (grammar FSM, discretize, FPS indices, `make_skeleton`) exactly and use
cosine thresholds on encoder/decoder tensors — same discipline as the HY3D router-tie caveat.

---

## 10. License

- **Repo code**: MIT (`LICENSE`, "Copyright (c) 2025 VAST-AI-Research"). `pointcept/LICENSE` = MIT.
- ⚠ **Michelangelo shape encoder = GPLv3.** `src/model/michelangelo/LICENSE` is GNU GPL v3, and every
  encoder source (`sal_perceiver.py`, `transformer_blocks.py`, `embedder.py`, `get_model.py`,
  `tsal_base.py`, `checkpoint.py`, `__init__.py`) carries a GPLv3 header ("derived from
  NeuralCarver/Michelangelo"). The perceiver is architecturally central to BOTH stage models.
  **Do not line-by-line translate these files into `mlx-serve`** (would create a GPLv3 derivative and
  is incompatible with shipping). The encoder is a bog-standard Fourier-embed + cross-attention
  perceiver — **re-derive it clean-room from the Michelangelo paper and our own `hunyuan3d.zig`
  cross-attention primitives**; only the numeric *weights* cross the boundary.
- **Weights**: the HF model card `VAST-AI/UniRig` is tagged **`license: mit`** (both card `cardData` and
  the README front-matter). So the converted weights are shippable under MIT. ⚠ Trained on
  `Seed3D/Articulation-XL2.0` — check that dataset's terms if that matters downstream; the released
  weights themselves are MIT.
- Skinning + Rig-XL/VRoid checkpoints are "released separately at a later date" (HF card) — only the
  Articulation-XL skeleton and skin ckpts exist publicly today.

---

## Appendix — exact config values (quick reference)

**AR** (`unirig_ar_350m_1024_81920_float32.yaml`, `ar_inference_articulationxl.yaml`,
`tokenizer_parts_articulationxl_256.yaml`, `inference_ar_transform.yaml`): OPT 24L/1024/16h/4096ffn/relu,
n_positions 3076, vocab 267; encoder width 512/8h/16L/num_freqs 8/include_pi False/point_feats 3/
token_num 1024/qkv_bias False/use_ln_post True; sampler mix 65536 (vertex 8192); num_discrete 256,
range [-1,1]; generate beams 15/top_k 5/top_p 0.95/rep 3.0/temp 1.5/max_new 2048/cls articulationxl.

**Skin** (`unirig_skin.yaml`, `skin.yaml`, `inference_skin_transform.yaml`): feat_dim 768, heads 16,
bone_embed_dim 1024, mlp_dim 512, num_bone_attn 8, num_mesh_bone_attn 16, grid_size 0.005,
voxel_mask 3.0, num_train_vertex 512; PTv3 in9/depths[3,3,3,6,16]/ch[32,64,128,256,384]/
heads[2,4,8,16,24]/patch1024/qknorm True/BN/res_linear True; global encoder = AR encoder dims →
out_proj 768; sampler mix 32768 (vertex 8192); voxel_skin grid 196/alpha 0.5/grid_query 7/vertex_query
1/grid_weight 3.0/mode square/backend pyrender; export group_per_vertex 4; reskin k7 median/alpha 2.0/
threshold 0.03.
