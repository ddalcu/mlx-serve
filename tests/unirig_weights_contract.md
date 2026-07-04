# UniRig stage-1 SKELETON weight-conversion contract

BINDING converted-tensor-name contract for the UniRig **stage-1 skeleton** model
(mesh point cloud → autoregressive skeleton-token sequence → joints/parents/tails).
Implemented by `tests/convert_unirig_weights.py` and consumed by the native Zig
skeleton engine (`src/unirig_skeleton.zig`, Phase 3, in progress) plus the
fixture/oracle dumps (`tests/dump_unirig_fixtures.py`).

Sibling of the Hunyuan3D SHAPE contract (`tests/hy3d_weights_contract.md` /
`convert_hunyuan3d_weights.py`) — **read that converter first**. This stage reuses
its `Source` strict pop/leftover accounting, its `deinterleave_qkv` per-head
un-interleave, its `should_quantize` predicate, and its `mx.save_safetensors` +
`--self-test` + `--bits {8,16}` structure. The architecture dossier this contract
was built against is `tests/unirig_dossier.md`; every fact below was additionally
verified 2026-07-04 against the on-disk checkpoint (585-tensor `state_dict`
inventoried by torch mmap-load) AND the reference sources
(`src/model/unirig_ar.py`, `src/model/michelangelo/models/{tsal/sal_perceiver.py,
modules/transformer_blocks.py,modules/embedder.py}`, `src/tokenizer/tokenizer_part.py`,
`configs/model/unirig_ar_350m_1024_81920_float32.yaml`,
`configs/system/ar_inference_articulationxl.yaml`,
`configs/tokenizer/tokenizer_parts_articulationxl_256.yaml`,
`configs/transform/inference_ar_transform.yaml`) in a clone at
`/Volumes/Sandisk_1TB/hy3d-scratch/UniRig`.

**Two structural facts drive the mapping:**

1. **Lightning `model.` prefix.** The ckpt is a PyTorch-Lightning `.ckpt` (pickle);
   weights live under the top-level `state_dict` key and every tensor name is
   prefixed `model.` (Lightning wraps the `UniRigAR` `ModelSpec` as `self.model`).
   The prefix is stripped first; there is NO optimizer / EMA / momentum state in
   this checkpoint (verified: `state_dict` holds exactly the 585 model tensors;
   the only other top-level key is the string `pytorch-lightning_version`).

2. **The michelangelo point-cloud encoder has the SAME per-head QKV interleave as
   the Hunyuan3D ShapeVAE michelangelo attention** — it IS the same
   `MultiheadAttention`/`MultiheadCrossAttention` classes. Its `c_qkv`/`c_kv`
   forward does `qkv.view(bs, n, heads, -1).split(head_dim, dim=-1)` (a per-head
   interleave, NOT a `[q|k|v]` block concat). So a plain row split silently
   produces garbage. We bake the un-interleave at convert time
   (`deinterleave_qkv`, mirrored from `convert_hunyuan3d_weights.py`) so the Zig
   side does a STANDARD head reshape. The OPT decoder attention (`q_proj`/`k_proj`/
   `v_proj`) is a plain HF layout and is NOT de-interleaved.

The OPT decoder itself is a stock HuggingFace `OPTForCausalLM` — its converted
namespace is a clean rename of the HF `model.decoder.*` / `lm_head` keys, so the
Zig side can reuse mlx-serve's transformer primitives (ReLU FFN, learned absolute
positions with +2 offset, pre-norm blocks + top-level `final_layer_norm`).

---

## 1. Output layout

```
~/.mlx-serve/models/local/unirig-skeleton-8bit    (--bits 8, default; shipping build)
~/.mlx-serve/models/local/unirig-skeleton-fp16     (--bits 16; parity-debug build the UNIRIG_* oracles target)
├── config.json          {"model_type":"unirig_skeleton", ...}   (§3)
├── skeleton.safetensors  ONE file, three namespaces:            (§4-§6)
│                          ar.*          OPT-350m decoder (389 logical tensors)
│                          enc.*         michelangelo perceiver encoder (227 logical tensors)
│                          output_proj.* 512→1024 bridge (2 tensors)
├── LICENSE               UniRig repo MIT license (weights are MIT — HF card `license: mit`)
├── NOTICE               records the GPLv3-code / MIT-weights split (§7)
```

One weight file (the task's contract) with three top-level prefixes; the convert
accounting is strict per namespace. **Total logical tensors: 389 + 227 + 2 = 618**
(fp16 / `--bits 16`), before quant triples the 247 eligible linears at `--bits 8`
(→ 618 + 247·2 = 1112 physical tensors). The 618 vs the 585-key source differ by
the +33 tensors the per-head de-interleave adds (§2 T2): the single `c_kv` per
cross-attn splits into k+v (+1) and each of the 16 self-attn `c_qkv` splits into
q+k+v (+2·16 = +32).

Source checkpoint (verified present):

| Item | Source |
|---|---|
| all weights | `skeleton/articulation-xl_quantization_256/model.ckpt` (1.44 GB Lightning ckpt; `state_dict` = 585 fp32 tensors, `model.`-prefixed; ≈360.1M params) |
| download | `hf download VAST-AI/UniRig --include "skeleton/articulation-xl_quantization_256/*"` |

The RigNet skeleton variant (`skeleton/rignet/model.ckpt`, hidden 512 / 8 enc
layers) and the stage-2 SKIN ckpt (`skin/articulation-xl/model.ckpt`) are **NOT**
converted here (skin = PTv3 + non-neural geodesic `voxel_skin`, deferred to a
separate contract — dossier §4/§8/§9).

---

## 2. Global transform rules

Applied uniformly; the mapping is otherwise a 1:1 rename. `[O,I]` = PyTorch Linear
weight (`y = x @ W.T`, stored `[out,in]`), kept as-is (no transpose — mlx `QLinear`
and dense matmul both consume the `[out,in]` layout, same as every other mlx-serve
converter).

| # | Rule | Applies to | Detail |
|---|---|---|---|
| T1 | **Strip `model.`** | every key | Lightning wrapper prefix. Then strip the per-component sub-prefix (`transformer.` / `mesh_encoder.encoder.` / `output_proj`). |
| T2 | **Per-head QKV de-interleave** | michelangelo `c_qkv` (M=3) + `c_kv` (M=2) ONLY | `deinterleave_qkv(w, heads=8, head_dim=64, n_members)` — identical to the HY3D SHAPE converter. Undoes the reference `cat.view(heads, M·head_dim).split(head_dim)` so a STANDARD per-head reshape reproduces the reference. Row map: `Wm[h·head_dim+j] = w[h·(M·head_dim) + m·head_dim + j]`. The michelangelo `c_q` (cross-attn query) and ALL OPT `*_proj` are plain — NOT de-interleaved. |
| T3 | **fp32 → fp16** | every tensor (source is fp32) | `t2np` casts down (bf16 would too; none here). The `--bits 16` build is fp16, matching the HY3D "parity-debug" build; the oracle cosine thresholds absorb the rounding vs the fp32-CPU reference. |
| T4 | **Embedding tables lose `.weight`** | `ar.embed_tokens`, `ar.embed_positions` | Named WITHOUT `.weight` (like the HY3D `cls_token`/`pos_embed`) to signal "gathered lookup table, keep dense" — so `should_quantize` (§8) never packs a table the Zig side gathers rows from. `lm_head.weight` KEEPS `.weight` (it is a real hidden→vocab matmul; it stays fp16 anyway — 267 rows < 512). |

**NOT needed here** (present in other contracts, absent here): conv NCHW→OHWI
(this model has zero convs — the point cloud is Fourier-embedded, not conv'd), MoE
expert stacking, GeGLU/SwiGLU (both FFNs are plain `Linear→act→Linear`; OPT = ReLU,
michelangelo = GELU), the OPT `project_in`/`project_out` (absent because
`word_embed_proj_dim == hidden_size == 1024`).

Two **spec** facts (not tensor transforms, but the Zig engine needs them; recorded
so the converter's `--self-test` and the oracles pin them):

- **T5 — michelangelo attention scale.** `QKVMultiheadAttention` applies
  `1/sqrt(sqrt(head_dim))` to BOTH q and k (net `1/sqrt(head_dim)` = `1/8` for
  head_dim 64), then softmax IN FLOAT32, cast back. The `flash=True` config flag
  selects an SDPA path that is numerically the plain path — port as one attention.
  No learned q/k norm (unlike the HY3D ShapeVAE michelangelo, which added qk-norm;
  this AR/skin encoder has none — verified: no `q_norm`/`k_norm` keys).
- **T6 — Fourier embedder.** `num_freqs=8, include_pi=False, include_input=True`,
  `logspace` frequencies `2^[0..7] = [1,2,4,…,128]` (NO ·π). For a coord vector
  `x∈R^3`: out = `concat([x (3), sin(x⊗f) (24), cos(x⊗f) (24)])` = 51, coord-major
  in the ⊗ (x0·f0..f7, x1·f0..f7, x2·f0..f7). Concatenated with the 3 normal feats
  → 54 = `input_proj` in-dim. The `frequencies` buffer is `persistent=False`, so it
  is ABSENT from the ckpt (recompute in Zig; the converter has no key to map).

---

## 3. `config.json`

Everything the Zig loader / tokenizer / decoder / sampler needs that is not
derivable from tensor shapes. Values cite their source config.

```json
{
  "model_type": "unirig_skeleton",
  "quant": "8bit",                          // or "fp16"

  "tokenizer": {                            // tokenizer_parts_articulationxl_256.yaml + tokenizer_part.py:22-45
    "num_discrete": 256,                    // coordinate bins; ids 0..255
    "continuous_range": [-1.0, 1.0],        // discretize/undiscretize domain (bin-center dequant: (t+0.5)/256*(hi-lo)+lo)
    "vocab_size": 267,
    "token_branch": 256,                    // topology: next joint reparents to an explicit parent
    "token_bos": 257,
    "token_eos": 258,
    "token_pad": 259,
    "token_spring": 260,                    // unnamed part separator
    "parts": { "body": 261, "hand": 262 },
    "token_cls_none": 263,
    "cls": { "vroid": 264, "mixamo": 265, "articulationxl": 266 },   // mixamo currently untrained
    "default_cls": "articulationxl",        // ar_inference_articulationxl.yaml assign_cls
    "skeleton_name_templates": {            // configs/skeleton/{vroid,mixamo}.yaml — the Zig make_names bone-name tables
      "vroid": "configs/skeleton/vroid.yaml",
      "mixamo": "configs/skeleton/mixamo.yaml"
    }
  },

  "ar": {                                   // OPT decoder — unirig_ar_350m_1024_81920_float32.yaml + facebook/opt-350m base
    "arch": "opt",
    "num_hidden_layers": 24,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "head_dim": 64,
    "ffn_dim": 4096,
    "activation_function": "relu",          // OPT default (NOT gelu)
    "word_embed_proj_dim": 1024,            // override (base 512) → drops project_in/project_out
    "do_layer_norm_before": true,           // override (base false) → pre-norm blocks + top-level final_layer_norm
    "max_position_embeddings": 3076,        // override (base 2048)
    "n_positions": 3076,
    "position_offset": 2,                   // OPT OPTLearnedPositionalEmbedding padding offset
    "pos_embed_rows": 3078,                 // 3076 + 2 (matches embed_positions shape)
    "layer_norm_eps": 1e-05,                // OPT default
    "vocab_size": 267,
    "tie_word_embeddings": true             // embed_tokens IS bit-identical to lm_head (verified torch.equal)
  },

  "encoder": {                              // michelangelo perceiver — mesh_encoder block of the AR yaml
    "arch": "michelangelo_perceiver",
    "width": 512,
    "heads": 8,
    "head_dim": 64,
    "num_encoder_layers": 16,               // self_attn.resblocks count
    "num_freqs": 8,
    "include_pi": false,
    "include_input": true,
    "fourier_out_dim": 51,                  // 3*(8*2+1)
    "point_feats": 3,                       // normals
    "input_proj_in": 54,                    // 51 + 3
    "num_latents": 512,                     // constructor arg; UNUSED (no_query=True) — token_num governs output length
    "token_num": 1024,                      // #latents out; FPS keeps ratio 1/4 of the presample
    "presample": 4096,                      // token_num * 4 (seed-0 np.random.default_rng choice, replace=False when N>4096)
    "presample_seed": 0,
    "fps_ratio": 0.25,
    "fps_random_start": false,              // deterministic: start index 0
    "use_full_input": true,                 // cross-attn KV = FULL point cloud; Q = the 1024 FPS points
    "use_ln_post": true,
    "qkv_bias": false,                      // c_q/c_kv/c_qkv have NO bias; only c_proj has bias
    "mlp_ratio": 4,                         // michelangelo MLP hidden = 4*width
    "mlp_act": "gelu"
  },

  "cond": {                                 // prefix conditioning — unirig_ar.generate
    "num_mesh_tokens": 1024,                // soft-prefix length (output_proj(latents))
    "output_proj_in": 512,
    "output_proj_out": 1024,
    "start_tokens": ["bos", "cls"],         // full prefix = [mesh_1024 ..., embed(bos), embed(cls)] → length 1026
    "logits_slice_from": "post_cond"        // logits taken over the post-cond span
  },

  "sampling": {                             // ar_inference_articulationxl.yaml generate_kwargs
    "max_new_tokens": 2048,
    "num_beams": 15,                        // HF beam-search multinomial (do_sample AND beams) — see §7 flag
    "do_sample": true,
    "top_k": 5,
    "top_p": 0.95,
    "repetition_penalty": 3.0,
    "temperature": 1.5,
    "assign_cls": "articulationxl",
    "seed": 12345
  },

  "sampler": {                              // inference_ar_transform.yaml (SamplerMix) + AugmentAffine
    "num_samples": 65536,
    "vertex_samples": 8192,
    "method": "mix",
    "normalize_into": [-1.0, 1.0],          // AugmentAffine isotropic normalize-into-cube; all random augments p=0 at inference
    "normals_transformed": false            // normalize is translate + uniform-scale only; normals unchanged
  }
}
```

---

## 4. `ar.*` — OPT-350m decoder (389 tensors)

Stock HF `OPTForCausalLM`. Source keys after stripping `model.transformer.` are the
HF `model.decoder.*` + `lm_head.weight` namespace. Canonical rename below. 24 layers
(0..23). `embed_tokens` == `lm_head` (tied, bit-identical) — both stored & converted
faithfully.

### 4a. Top-level (5 tensors)

| Source (after `model.transformer.`) | Canonical | Shape | Transform | Quant? |
|---|---|---|---|---|
| `model.decoder.embed_tokens.weight` | `ar.embed_tokens` | `[267,1024]` | T4 (drop `.weight`) | no (gathered table) |
| `model.decoder.embed_positions.weight` | `ar.embed_positions` | `[3078,1024]` | T4 (drop `.weight`) | no (gathered table) |
| `model.decoder.final_layer_norm.{weight,bias}` | `ar.final_norm.{weight,bias}` | `[1024]` | — | no (1-D) |
| `lm_head.weight` | `ar.lm_head.weight` | `[267,1024]` | — | no (min 267 < 512) |

### 4b. Per decoder layer ×24 (16 tensors each → 384)

Prefix `model.decoder.layers.{i}.` → `ar.layers.{i}.`.

| Source subkey | Canonical subkey | Shape | Quant? |
|---|---|---|---|
| `self_attn.q_proj.{weight,bias}` | `attn.q.{weight,bias}` | `[1024,1024]`,`[1024]` | weight yes |
| `self_attn.k_proj.{weight,bias}` | `attn.k.{weight,bias}` | `[1024,1024]`,`[1024]` | weight yes |
| `self_attn.v_proj.{weight,bias}` | `attn.v.{weight,bias}` | `[1024,1024]`,`[1024]` | weight yes |
| `self_attn.out_proj.{weight,bias}` | `attn.out.{weight,bias}` | `[1024,1024]`,`[1024]` | weight yes |
| `self_attn_layer_norm.{weight,bias}` | `attn_norm.{weight,bias}` | `[1024]` | no |
| `fc1.{weight,bias}` | `mlp.fc1.{weight,bias}` | `[4096,1024]`,`[4096]` | weight yes |
| `fc2.{weight,bias}` | `mlp.fc2.{weight,bias}` | `[1024,4096]`,`[1024]` | weight yes |
| `final_layer_norm.{weight,bias}` | `mlp_norm.{weight,bias}` | `[1024]` | no |

`attn_norm` is the pre-attention norm (`self_attn_layer_norm`); `mlp_norm` is the
pre-FFN norm (the OPT per-layer `final_layer_norm` — do NOT confuse with the
top-level `ar.final_norm`). Both are pre-norms because `do_layer_norm_before=true`.
The attention has bias on all four projections (OPT default). FFN = `fc2(relu(fc1(x)))`.

---

## 5. `enc.*` — michelangelo perceiver encoder (227 tensors)

A `ShapeAsLatentPerceiverEncoder` (the simplified, `no_query=True` variant — no
`pre_kl`/`post_kl`/`transformer`/`geo_decoder`, only the presample→FPS→cross-attn→
self-attn stack). Source keys after stripping `model.mesh_encoder.encoder.`. All
attention `c_*` linears are BIAS-FREE (`qkv_bias=False`); only `c_proj` (→ `out`)
carries a bias. **Clean-room caveat (dossier §10): the Zig engine re-derives this
perceiver from the paper + mlx-serve's own cross-attention primitives — only the
converted numeric weights cross the GPLv3 boundary.**

### 5a. Top-level (4 tensors)

| Source (after `…encoder.`) | Canonical | Shape | Transform | Quant? |
|---|---|---|---|---|
| `input_proj.{weight,bias}` | `enc.input_proj.{weight,bias}` | `[512,54]`,`[512]` | — | no (in 54 < 512, 54%64≠0) |
| `ln_post.{weight,bias}` | `enc.ln_post.{weight,bias}` | `[512]` | — | no (1-D) |

### 5b. Cross-attention block (`ResidualCrossAttentionBlock`, pre-LN) — 15 tensors

Reference forward: `x = x + attn(ln1(x_query), ln2(kv_data)); x = x + mlp(ln3(x))`.
Query = the 1024 FPS points; KV = the full point cloud. Prefix
`cross_attn.` → `enc.cross_attn.`.

| Source subkey | Canonical | Shape | Transform | Quant? |
|---|---|---|---|---|
| `ln_1.{weight,bias}` | `ln1.{weight,bias}` | `[512]` | — | no |
| `ln_2.{weight,bias}` | `ln2.{weight,bias}` | `[512]` | — | no |
| `attn.c_q.weight` | `attn.q.weight` | `[512,512]` | — (standard) | yes |
| `attn.c_kv.weight` | `attn.k.weight`, `attn.v.weight` | `[1024,512]`→ 2×`[512,512]` | **T2 de-interleave M=2** (heads 8, hd 64) | yes (each) |
| `attn.c_proj.{weight,bias}` | `attn.out.{weight,bias}` | `[512,512]`,`[512]` | — | weight yes |
| `ln_3.{weight,bias}` | `ln3.{weight,bias}` | `[512]` | — | no |
| `mlp.c_fc.{weight,bias}` | `mlp.fc1.{weight,bias}` | `[2048,512]`,`[2048]` | — | weight yes |
| `mlp.c_proj.{weight,bias}` | `mlp.fc2.{weight,bias}` | `[512,2048]`,`[512]` | — | weight yes |

### 5c. Self-attention blocks ×16 (`ResidualAttentionBlock`, pre-LN) — 13 tensors each → 208

Reference forward: `x = x + attn(ln1(x)); x = x + mlp(ln2(x))`. Prefix
`self_attn.resblocks.{i}.` → `enc.blocks.{i}.`.

| Source subkey | Canonical | Shape | Transform | Quant? |
|---|---|---|---|---|
| `ln_1.{weight,bias}` | `ln1.{weight,bias}` | `[512]` | — | no |
| `attn.c_qkv.weight` | `attn.q.weight`, `attn.k.weight`, `attn.v.weight` | `[1536,512]`→ 3×`[512,512]` | **T2 de-interleave M=3** (heads 8, hd 64) | yes (each) |
| `attn.c_proj.{weight,bias}` | `attn.out.{weight,bias}` | `[512,512]`,`[512]` | — | weight yes |
| `ln_2.{weight,bias}` | `ln2.{weight,bias}` | `[512]` | — | no |
| `mlp.c_fc.{weight,bias}` | `mlp.fc1.{weight,bias}` | `[2048,512]`,`[2048]` | — | weight yes |
| `mlp.c_proj.{weight,bias}` | `mlp.fc2.{weight,bias}` | `[512,2048]`,`[512]` | — | weight yes |

Counts: top-level 4 + cross-attn 15 + 16·13 self-attn 208 = **227**.

---

## 6. `output_proj.*` — 512→1024 bridge (2 tensors)

| Source (after `model.`) | Canonical | Shape | Quant? |
|---|---|---|---|
| `output_proj.weight` | `output_proj.weight` | `[1024,512]` | yes (min 512, 512%64==0) |
| `output_proj.bias` | `output_proj.bias` | `[1024]` | no |

`nn.Linear(512, 1024)` applied to every one of the 1024 encoder latents to lift
them to the OPT hidden dim before they enter the decode sequence as the soft prefix.

---

## 7. Quantization (`--bits`, reuses the SHAPE `should_quantize` predicate)

- `--bits 8` (default, shipping): quantize eligible linear `.weight`s with mlx
  affine `group_size=64`, packed uint32 `.weight` + fp16 `.scales`/`.biases` (each
  eligible weight → 3 tensors). **247 weights quantize** (ar 144 = 24·(4 attn + 2
  ffn); enc 102 = cross-attn 6 + 16·6 blocks; output_proj 1).
- `--bits 16`: everything fp16, no quant (the parity-debug build the `UNIRIG_*`
  oracles target).

`should_quantize(name, shape, bits)` (identical predicate to
`convert_hunyuan3d_weights.py`): quantize iff `bits==8` AND name ends `.weight`
AND `ndim == 2` AND `last_dim % 64 == 0` AND `min(last_two_dims) ≥ 512`.
Consequences here:

- **Quantized**: every 1024/4096-dim OPT attn/ffn linear, every 512/2048-dim
  michelangelo attn/ffn linear (post de-interleave the split members are
  `[512,512]`, min 512 → eligible), `output_proj.weight` `[1024,512]`.
- **Left fp16** (min-dim < 512 or non-`.weight`): `enc.input_proj.weight`
  `[512,54]` (in 54 < 512 AND 54%64≠0), `ar.lm_head.weight` `[267,1024]`
  (min 267 < 512), every 1-D norm/bias, the two `ar.embed_*` tables (T4 dropped
  their `.weight`), all `.bias`.
- All quantized in-dims ∈ {512, 1024, 4096} are % 64 == 0, so mlx affine quant is
  valid for the whole eligible set.

---

## 8. Flags / contradictions (per the "flag loudly" instruction)

1. **⚠ The michelangelo AR/skin encoder attention DOES need per-head
   de-interleave — the dossier §3d does not spell this out but it is load-bearing.**
   `QKVMultiheadAttention.forward` does `qkv.view(bs, n, heads, -1)` then
   `split(attn_ch, dim=-1)` — the exact interleave the HY3D SHAPE converter's
   `deinterleave_qkv` undoes for the ShapeVAE `c_qkv`/`c_kv`. These are literally
   the same `MultiheadAttention`/`MultiheadCrossAttention` classes. A plain row
   split of `c_qkv[1536,512]`/`c_kv[1024,512]` would load garbage that passes a
   value-equality test but fails a forward. This contract de-interleaves them (§5).
   The OPT `q/k/v_proj` are NOT interleaved (plain HF) — do not de-interleave them.

2. **⚠ Sampling is HF beam-search multinomial (`num_beams=15` AND `do_sample=True`)
   — a HIGH porting risk, not a conversion issue.** The converter emits the full
   `sampling` block verbatim; the Zig decoder must either port 15-beam search
   (temp 1.5 / top_k 5 / top_p 0.95 / rep-penalty 3.0) or accept an INFERRED v1
   simplification (single-beam sampled decode) as a documented fidelity gap
   (dossier §5/§8). The `UNIRIG_SKEL_E2E` oracle uses GREEDY (num_beams=1,
   do_sample=False) for determinism — it validates the grammar-masked decode path,
   not the production stochastic sampler.

3. **⚠ End-to-end token parity is NOT bit-exact; pin deterministic sub-pieces,
   cosine on tensors.** The model is built float32 but the reference trainer runs
   `bf16-mixed` autocast at inference, and production sampling is stochastic. Per
   the CLAUDE.md gotcha "Parity fixtures for fp16-fragile giants", the oracle
   reference is dumped **fp32 on CPU**; the fp16 MLX engine (fp32-accumulating
   matmuls) is compared at cosine > 0.99 on encoder/prefix/step tensors, and the
   greedy E2E oracle compares a FIRST-N-token prefix (INT4/fp16 long-greedy
   argmax-tie divergence is expected past the first tens of tokens — same class as
   the HY3D router-tie / PLD long-greedy caveats). The deterministic pieces
   (discretize/undiscretize, the grammar FSM, `make_skeleton`) are exact.

4. **⚠ `embed_tokens` is bit-identical to `lm_head` (tied).** Both `[267,1024]`
   tensors are stored; verified `torch.equal`. Converted faithfully (both written);
   `config.ar.tie_word_embeddings=true` records it so the Zig loader may share one.

5. **No optimizer/EMA state in this ckpt.** Unlike the stage-2 SKIN ckpt (4.38 GB,
   ~3× the params → almost certainly carries Adam state, per dossier §6), the
   skeleton `state_dict` is exactly the 585 model tensors. The converter's strict
   `Source` leftover check FATALs on any unmapped key, so a future ckpt that DID
   ship optimizer state would be caught (add a `DROP_RULES` entry for it, mirroring
   the HY3D VAE-encoder drops).

6. **No convs, no MoE, no GeGLU/SwiGLU, no q/k-norm, no `project_in/out`, no
   `frequencies` buffer.** All four are present in sibling contracts and absent
   here — enumerated so a reader does not go looking for them. The Fourier
   `frequencies` buffer is `persistent=False` (recompute in Zig; §2 T6).

---

## 9. License

- **Repo code + weights**: MIT. The UniRig repo `LICENSE` is MIT
  ("Copyright (c) 2025 VAST-AI-Research and contributors."), and the HF model card
  `VAST-AI/UniRig` is tagged `license: mit`. **The converted weights are shippable
  under MIT.** The converter copies the repo `LICENSE` into the output dir.
- **⚠ The michelangelo encoder SOURCE is GPLv3** (`src/model/michelangelo/LICENSE`
  = GNU GPL v3; every encoder `.py` carries a GPLv3 header, "derived from
  NeuralCarver/Michelangelo"). **Do not line-by-line translate those files into
  `mlx-serve`** — re-derive the perceiver clean-room from the paper + our own
  cross-attention primitives (dossier §10). Only the numeric *weights* (MIT) cross
  the boundary. The converter writes a `NOTICE` recording this split so a reader of
  the output dir understands why the weights are MIT while the reference code is
  GPLv3. (The fixture dump `tests/dump_unirig_fixtures.py` may IMPORT the reference
  GPLv3 encoder at runtime from the user's clone — same established pattern as
  `dump_hunyuan3d_fixtures.py` importing the reference `hy3dshape`; that is a
  user-run reference, not shipped mlx-serve source.)
- Trained on `Seed3D/Articulation-XL2.0` — the released weights are MIT; check the
  dataset terms only if that matters downstream.
