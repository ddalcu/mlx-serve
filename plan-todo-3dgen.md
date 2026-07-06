# 3DGen roadmap — handoff plan (written 2026-07-04)

> **2026-07-05 DESCOPE:** the avatar experiment (P3 rig + P4 avatar window + the follow-up
> mouth-articulation work) was stripped — the talking-avatar results were not realistic enough
> to ship. What SHIPS from this plan is P1 + P1.x + P2: photo → shape → full-PBR textured GLB,
> turntable preview, history shelf, one-click combined download. The voice-clone piece of P4
> survives in a different form: a global "Voice clone clip" in Settings ▸ Voice that hands-free
> voice mode speaks with (`ClonedVoiceSynthesizer`, Qwen3-TTS `ref_audio`). Sections on rig /
> avatar / mouth below are HISTORICAL — the code they describe was removed from the tree
> (`unirig_*`, `voxel_skin`, `fps`, `face_anchor`, `mouth_*`, `AvatarEngine`/`AvatarView`/
> `SkeletalAnimator`); the published HF repo still carries the `unirig/` subdir, which the app
> now simply ignores.

This document is the complete, self-contained TODO for finishing the 3DGen → virtual-avatar
feature. It assumes NO context beyond this file + the repo's `CLAUDE.md`. Phase 1 is DONE and
validated; everything else is specified here in execution order. Read §3 (house rules) before
writing any code — every rule there was paid for in Phase 1.

## 1. Mission & phase map

End goal: **a virtual avatar** — upload a photo of a person → realistic 3D model → it animates,
emotes, and eventually talks using the app's existing persona system prompt + DocumentIndex RAG +
Qwen3-TTS voice cloning + voice input.

| Phase | What | Status |
|---|---|---|
| P1 | Shape: photo → untextured GLB (`.mesh` modality, Hunyuan3D-2.1 shape stage) | **DONE, validated 2026-07-03/04** |
| P1.x | Loose ends: FlashVDM speedup, HF weight publish, pane polish | **DONE** — FlashVDM (6× volume decode, default res 384), history thumbnails, and HF publish (combined `ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit`, uploaded + verified 2026-07-04) |
| P2 | Texture: full PBR (albedo + metallic-roughness) — **user chose option B** (full PBR in one go, not albedo-first) | **DONE** — all `HY3DP_*` oracles pass at fp16 (full UNet step cos 1.00000), live 8-bit e2e green; §5.8.14 follow-ups (UniPC-15, RealESRGAN, >6-view) still optional/open |
| P3 | Rig & animate: auto-rig + skeletal idle/emote + TTS-driven jaw | **DESCOPED 2026-07-05** — was done (UniRig + voxel skinning, oracles exact) but stripped with the avatar: not realistic enough to ship |
| P4 | Avatar loop: persona + RAG + voice clone + speech pipelining + avatar window | **DESCOPED 2026-07-05** — window/engine/persona removed; the voice-clone + sentence-pipelining piece lives on as voice mode's `ClonedVoiceSynthesizer` (Settings ▸ Voice clip) |

**Status 2026-07-04 (128 GB Mac):** Zig 722 pass / 0 fail; Swift 1111 / 0 fail; chat 37/37.
Weights re-converted HERE from source ckpts (`~/hy3d-scratch/` sources, venv there too) into
`~/.mlx-serve/models/local/{hunyuan3d-2-1-8bit,hunyuan3d-2-1-paint-8bit,unirig-skeleton-8bit}`;
`test_3d_{gen,paint,rig}.sh` ALL PASS live (shape verts 72,128 == dev-Mac record). §4.2 done up
to the upload: COMBINED single repo staged at `~/hf-staging/Hunyuan3D-2.1-MLX-Serve-8bit`
(shape root + `paint/` + `unirig/` + README/licenses), server resolves stage subdirs
(`gen.findStageModelDir`, subdir → sibling → env), Swift preset/bundle flipped to
`ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit` (recursive pull, all-stage ready markers). Both paint+rig
integration tests ALL PASS against the combined layout. **PUBLISHED 2026-07-04** — user
uploaded; the repo was re-downloaded fresh from HF and all three integration tests pass
against the published artifact (discovery lists it exactly once, no subdir leakage). Also fixed en route: ready mesh models reported
`capabilities: []` (missing mesh arm in `readyCapsJson`) + test_3d_gen.sh check 1 was non-fatal.
Remaining: optional §5.8.14 follow-ups; live §8 avatar demo; `plan-tts-8bit.md` separate side-plan.

## 2. Current state (verified)

### Shipped code (all on `main`, uncommitted as of writing — check `git status`)

**Zig server** — fourth gen modality `.mesh`, endpoint `POST /v1/3d/generations`:
- `src/hunyuan3d.zig` (~2.4k LOC) — the shape engine: DINOv2-**Large** conditioner (@518), 3.3B MoE
  flow-match DiT (timestep-as-token-0, U-ViT LIFO skips, softmax-before-top-2 router NO renorm +
  always-on shared expert), ShapeVAE decoder + geo-decoder SDF queries (per-mesh precomputed
  cross-attn KV), chunked volume decode. Dense dtype **fp16** (not bf16). 7 hermetic tests + 5
  env-gated `HY3D_*` cos oracles. **`Engine.generateMeshRaw` (src/hunyuan3d.zig:1959) is the P2
  texture seam** — returns a raw `marching_cubes.Mesh` before GLB encoding.
- `src/marching_cubes.zig` — pure Zig, watertight indexed mesh, CCW-outward along −∇field,
  hermetic manifold/Euler/normal tests.
- `src/glb.zig` — minimal glTF 2.0 binary writer (u32 indices, POSITION min/max). **Deliberately
  append-only/structured so P2 can add TEXCOORD_0 + images + PBR material additively.**
- Plumbing: `gen.zig` (`Modality.mesh`, `MeshEngine`, `handleMesh`), `model_registry.zig`
  (`mesh_engine` slot + free blocks), `scheduler.zig` (`.mesh` load arm), `server.zig` (route +
  `genJobRun`/`handleGen` arms), `model_discovery.zig` (`hunyuan3d*` prefix — documented
  duplication with `gen.modalityFromType`, keep in sync).

**Swift app** (fourth pane clone): `Services/Model3DGenService.swift`, `Views/Model3DGenView.swift`
(contains `GLBMeshLoader` — ModelIO/SceneKit GLB load — and the procedural turntable/breathing
"animate v1"), `Services/SubjectCutout.swift` (Vision `VNGenerateForegroundInstanceMaskRequest`
cutout → composite on white), presets/bundle/settings/storage additions in `MediaGen.swift`,
`MediaBundle.swift`, `MediaGenSettings.swift`, `MediaStorage.swift` (`models3dRoot`), tile in
`StatusMenuView.swift`, window in `MLXServeApp.swift`. Tests in `app/Tests/MLXCoreTests/`
(`Model3DGenServiceTests`, `GLBMeshLoaderTests` against the Zig-written fixture
`Fixtures/hy3d_sphere.glb`, `SubjectCutoutTests`, bundle/settings/tile tests).

**Scripts + tests**: `tests/convert_hunyuan3d_weights.py` (torch ckpt → converted safetensors,
`--bits 8|16`, `--self-test` runs hermetic mapping tests), `tests/dump_hunyuan3d_fixtures.py`
(reference-pipeline fixture dump, prints the `HY3D_*` env block), `tests/test_3d_gen.sh`
(headless boot → load-by-path → `"3d"` capability → gen → GLB validity → 400s → SSE streaming →
coexistence → unload; SKIPs cleanly without the converted model).

### Validation record (all reproduced live on the dev Mac, 16 GB)

| Check | Result | Threshold |
|---|---|---|
| Oracle 1 DINO features | cos 0.999996 | 0.999 |
| Oracle 2 DiT one-step velocity | cos 0.999993 | 0.995 |
| Oracle 3 latent→SDF | cos 1.000000 | 0.995 |
| Oracle 4 10-step CFG denoise | cos 0.997363 | 0.99 |
| Oracle 5 e2e SDF grid 129³ | cos 0.999896 (fp16), 0.999758 (8-bit) | 0.98 / 0.97 |
| e2e mesh vertices | 76,012 (fp16) / 75,980 (8-bit) vs reference 76,006 | ±10% |
| Live `/v1/3d/generations` | 3.4 MB GLB, 72,128 verts from a photo; streaming = 77 progress + complete | — |
| Regression | `zig build test` 652 pass/56 skip; full `swift test`; 37/37 chat integration | — |

### Asset locations (KEEP — oracle reruns and P2 need them)

- Shipping 8-bit build: `~/.mlx-serve/models/local/hunyuan3d-2-1-8bit` (3.5 GB on disk).
- `/Volumes/Sandisk_1TB/hy3d-scratch/` (~16 GB): `snapshot/` (source ckpts: DiT 7.37 GB + VAE
  656 MB), `hunyuan3d-2-1-fp16/` (parity-debug build — oracles target THIS), `reference/`
  (Tencent-Hunyuan/Hunyuan3D-2.1 clone), `venv/` (uv Python 3.12: torch 2.12, mlx, diffusers,
  transformers, timm, torchvision, scikit-image, opencv-headless, trimesh, pymeshlab, omegaconf,
  einops), `fixtures/` (the `HY3D_*` .raw files). The paint ckpts (§5) download into `snapshot/`
  too (`hf download tencent/Hunyuan3D-2.1 --include "hunyuan3d-paintpbr-v2-1/*" --local-dir ./snapshot`).

### Endpoint contract (P1, live today)

Request: `{"model","image" (b64 PNG/JPEG, REQUIRED),"steps" (30),"guidance_scale" (5.0),
"octree_resolution" (256, [64,512]),"seed","stream"}` →
`{"created":0,"format":"glb","data":"<b64 GLB>"}`; stream = SSE progress + complete (same event
shape as image/video/audio; the app's `APIClient.streamGeneration` is shared).

## 3. House rules (each one bit us in P1 — do not relearn them)

1. **Builds**: `zig build test -Doptimize=ReleaseFast` always (never bare `zig build`);
   `SKIP_NOTARIZE=1 bash app/build.sh` for the app; `swift build/test` always with
   `-Xswiftc -swift-version -Xswiftc 5`. TDD order per repo `CLAUDE.md` (failing test first).
2. **Contract-first for parallel agents**: before spawning workstreams, write a BINDING
   converted-tensor-name contract file that both the convert script AND the Zig loader implement.
   In P1 this caught the killer divergence: the DiT stores q/k/v separately but its forward
   re-fuses them with a per-head interleave — the convert script bakes the permutation out, and
   the engine uses plain standard head reshapes. Expect the same class in P2 (see §5 traps).
3. **Oracle pattern**: user-run `tests/dump_*_fixtures.py` against the PyTorch reference dumps
   f32 `.raw` fixtures + prints an env block; env-gated Zig tests compare at cos thresholds.
   Rules: oracle inputs are taken POST-preprocess (decouples resize-kernel drift); e2e oracles
   REUSE their parent oracle's injected noise/trajectory (a fresh trajectory fails spuriously);
   dump deterministic values (VAE posterior MEAN, not `.sample()`).
4. **Reference-runner memory (16 GB Mac)**: torch reference loaders build fp32 modules from a
   fully-materialized ckpt (~22 GB transient → swap-thrash). Fix in the dump script, not by
   suffering: `torch.load(..., mmap=True)` + `torch.set_default_dtype(torch.float16)` around
   pipeline construction (reset after), and retire each sub-model to CPU (`model.to("cpu")` +
   `torch.mps.empty_cache()`) the moment its oracle is dumped.
5. **MPS buffer caps**: chunked GPU work must keep the reference chunk sizes. P1 example: 50k-point
   geo-decoder chunks → a single 13.1 GB f32 attention buffer → hard Metal assertion; the
   reference's 8k default is sized for this. P2's multiview attention (24,576-token sequences)
   MUST use `mlx_fast_scaled_dot_product_attention` — a materialized score matrix is ≥7 GB.
6. **Shell tests**: base64 request bodies go in FILES (`-d @file`), never inline
   `$( … )` argv substitution (~150 KB argv blobs silently fail).
7. **Parallel decomposition that worked in P1** (repeat it): 4 concurrent agents on disjoint
   files — (a) pure-CPU Zig modules with hermetic tests, (b) Python convert+fixture scripts,
   (c) the engine file, (d) the Swift pane — with the integrator owning all shared files
   (gen.zig/server/scheduler/registry/tests.zig) and applying them only after (a)+(c) land.
8. Never commit to git (user does that). Fold CHANGELOG bullets into the topmost UNSHIPPED entry
   (check `gh release list --limit 1` first).

## 4. Phase 1 loose ends (small, independent — good warm-up tasks)

1. **FlashVDM-class hierarchical volume decode** — coarse-to-fine octree (start ~63³, refine only
   sign-change cells) instead of the dense (R+1)³ sweep; 10–30× fewer geo-decoder queries. The
   engine keeps the volume decoder behind a small seam (`decodeVolume` in `src/hunyuan3d.zig`).
   Guard: oracle 5 must still pass (grid values at surviving cells identical; compare extracted
   mesh instead if the sparse grid changes the fixture shape). Then bump the pane default res to 384.
2. **Publish converted weights to HF** (after user sign-off on quality): upload the 8-bit dir as
   e.g. `ddalcu/hunyuan3d-2-1-shape-8bit-mlx-serve` with the Tencent LICENSE/NOTICE; flip the
   preset repo in `MediaGen.swift` from `local/hunyuan3d-2-1-8bit` to the HF id (one line) and the
   pane's "convert locally" hint becomes the standard `BundleDownloadBar` download. Mirror FLUX's
   license-acceptance posture.
3. Pane polish: generation history thumbnails; surface `octree_resolution` 384 once (1) lands.

## 5. Phase 2 — Full-PBR texture (option B). Design verified against reference sources 2026-07-04

Everything below was verified by reading `Tencent-Hunyuan/Hunyuan3D-2.1@main` `hy3dpaint/`
(files cited inline), the HF `tencent/Hunyuan3D-2.1` tree, and `ZimengXiong/Hunyuan3D-MLX@main`.
Trust these facts but re-verify anything that smells stale before coding against it.

### 5.1 Pipeline (what the reference actually does)

Entry: `hy3dpaint/textureGenPipeline.py` (`Hunyuan3DPaintPipeline`), config
`Hunyuan3DPaintConfig`: `render_size=2048`, `texture_size=4096`, `bake_exp=4`,
`bake_mode="back_sample"`, `max_num_view=6`, view `resolution=512`.

- **A. Mesh prep (CPU)**: optional remesh via pymeshlab (SKIP — GPL, and our meshes come from our
  own shape stage; `use_remesh=False` is a supported reference path). UV unwrap =
  `xatlas.parametrize` with ALL defaults (`utils/uvwrap_utils.py`). `MeshRender.set_mesh`
  (`DifferentiableRenderer/MeshRender.py` ~:665) does an **axis remap** (negate x,y then swap
  y/z), **UV v-flip**, auto-center + normalize to `scale_factor=1.15`. `extract_textiles` (~:923)
  rasterizes the mesh in UV space at 4096² → per-texel world position/normal + texel coords (the
  bake's lookup tables).
- **B. View selection** (`utils/pipeline_utils.py::bake_view_selection`): with the default
  `max_num_view=6` the set is EXACTLY 6 fixed views — azims `[0,90,180,270,0,180]`, elevs
  `[0,0,0,0,90,-90]`, bake weights `[1,0.1,0.5,0.1,0.05,0.05]`. (Greedy coverage extension only
  engages for >6 views — defer.)
- **C. Geometry renders**: **orthographic** camera, `ortho_scale=1.2`, distance 1.45, z-up
  look-at with `elev=-elev, azim+=90` (`DifferentiableRenderer/camera_utils.py::get_mv_matrix`).
  Per view at 512²: world-space FACE-normal map `(n+1)/2` bg white, and position map
  `0.5 − vtx_pos/scale_factor` bg white. **Background is exactly 1.0** — the UNet detects
  background texels via `position != 1`. Rasterizer = the reference's OWN CPU C++ kernel
  (`custom_rasterizer/lib/custom_rasterizer_kernel/rasterizer.cpp`, ~150 lines): pixel centers
  +0.5, screen map `(x/w·0.5+0.5)·(width−1)+0.5`, int64 z-token min-compare
  (`z_quant·MAXINT + faceid+1`), perspective-corrected barycentric. Port THIS to Zig verbatim.
  nvdiffrast is never used in 2.1.
- **D. Neural conditioning**: reference photo 512² RGBA→white → SD-VAE encode `(x−0.5)·2` →
  **posterior MEAN** ·0.18215 → `ref_latents [1,1,4,64,64]`. **DINOv2-giant** (40 layers, hidden
  1536, 24 heads, **SwiGLU FFN**, patch 14; preprocess resize-256 → center-crop **224** →
  ImageNet norm) → `[1,257,1536]`. Normal/position maps per view → VAE → `embeds_normal/position
  [1,6,4,64,64]`. **CLIP text encoder is NOT run**: prompt embeds are learned parameters
  `learned_text_clip_albedo/mr [77,1024]`; negative embeds are the SAME tokens (not zeros). The
  shipped CLIP text/image encoders + tokenizer are dead weight — do not convert.
- **E. CFG**: triple batch `ref_scale=[0,1,1]`, dino `[zeros,zeros,real]` — but `camera_azims` is
  never passed by the 2.1 wrapper so the middle batch CANCELS algebraically; a port may run
  standard 2-batch CFG (33% cheaper) — validate final-latent cos vs the 3-batch reference.
- **F. Denoise** (`hunyuanpaintpbr/pipeline.py::denoise` :590–697): runtime scheduler =
  **UniPCMultistep from the shipped DDIM config** → **v-prediction + rescale_betas_zero_snr +
  trailing spacing carry over**; 15 steps, guidance 3.0, seed 0. Latents `[12,4,64,64]`
  (2 PBR materials × 6 views). UNet input = channel-concat `sample⊕embeds_normal⊕embeds_position
  = 12ch` — the shipped `unet/config.json` still says `in_channels: 4`; runtime replaces
  `conv_in` with a 12-channel conv (`unet/modules.py::from_pretrained` :818).
  Per transformer block (16 blocks), `Basic2p5DTransformerBlock` (:472–707) runs IN ORDER:
  1. **MDA** material self-attn — albedo uses base `attn1.to_*`, "mr" uses
     `attn1.processor.to_*_mr` clones.
  2. **RA** reference attention — query = ALBEDO-slice only; K from base `to_k`; V = concat of
     `to_v` and `to_v_mr`; per-material `to_out`/`to_out_mr`; residual × `ref_scale`.
     The reference features come from a **dual UNet** (full SD2.1 copy, `unet_dual`) run ONCE per
     generation (not per step!) on `ref_latents` at **timestep 0**, caching each block's
     post-norm1 hiddens (`kwargs["cache"]` created before the loop, pipeline.py :501).
  3. **MA** multiview attention — `[3·2, 6·L, C]` full cross-view self-attn with **3D RoPE from
     quantized position voxels**: grids `[64,32,16,8]` / voxel res `[512,256,128,64]` keyed by
     seq-len; axis dims `3/8,3/8,2/8` of head_dim; **interleaved-pair** rotation
     (`repeat_interleave(2)`), f32.
  4. attn2 cross-attn over the 77 learned tokens.
  5. **DINO cross-attn** — K/V from `ImageProjModel` (per-token Linear 1536→4·1024 + LN →
     `[·,1028,1024]`), zero-init out.
  6. GeGLU FF.
  Scheduler steps on `latents[:, :4]` only (:696).
- **G. Decode**: VAE decode /0.18215 → 12 images 512² → albedo = first 6, mr = last 6.
- **H. Super-res**: RealESRGAN x4plus (RRDBNet 23 blocks, ~16.7M params, BSD-3 code) → 2048².
  **Ship v1 WITHOUT it** (bilinear upscale, quality-flagged); add as a follow-up task.
- **I. Bake** (`MeshRender.back_project` :1113–1315 + `fast_bake_texture`): per texel, project
  through the view; frustum + depth test (`|Δz| < 3e-3`); cos map = dot(face normal, view dir),
  zeroed below cos 75°; boundary erosion with a **resolution-dependent kernel**
  (`int(2/512·render_size)` → 17×17 at 2048); Canny 30/80 depth-edge dilation; blend
  `Σ tex·w_view·cos⁴ / Σw` into the 4096² atlas (skip views >99% overlap).
- **J. Inpaint**: vertex-graph color propagation along mesh edges
  (`DifferentiableRenderer/mesh_inpaint_processor.cpp`, ~400 lines, port to Zig) + hole fill
  (reference uses `cv2.INPAINT_NS`; replace with iterative diffusion fill — visual-only
  deviation, document it).
- **K. Export**: textures ÷2 → 2048²; generated "mr" image: **metallic = R, roughness = G**;
  glTF metallicRoughness texture packs **G=roughness, B=metallic** — two remaps that can
  silently cancel; albedo → baseColorTexture.

### 5.2 Weights (HF `tencent/Hunyuan3D-2.1`, dir `hunyuan3d-paintpbr-v2-1/`)

| File | Size | Convert? |
|---|---|---|
| `unet/diffusion_pytorch_model.bin` | 3.93 GB fp16 pickle, ≈1.96B params (main UNet ≈1.10B incl. MDA/RA/DINO additions + learned tokens, `unet_dual` ≈0.87B) | YES |
| `vae/diffusion_pytorch_model.bin` | 335 MB fp32, standard SD 2.x AutoencoderKL | YES (enc+dec, cast fp16) |
| `image_encoder/`, `text_encoder/`, `tokenizer/`, `feature_extractor/` | ~2.6 GB | NO — dead at inference |
| `scheduler/scheduler_config.json` | — | copy values into synthesized config.json |

External: `facebook/dinov2-giant` (1.14B; fp16 ≈2.27 GB) — REQUIRED; `RealESRGAN_x4plus.pth`
(67 MB) — optional stage H.

Memory: fp16 weights ≈6.4 GB, peak ≈8–9 GB; 8-bit-quantize all UNet + DINO linears (convs stay
fp16) → ≈4.5 GB peak. Same `--bits 8` default / `--bits 16` parity-build split as P1.

### 5.3 New modules (house pattern)

- `src/hunyuan3d_paint.zig` — the engine: SD-VAE enc/dec, UNet2p5D (main + dual), DINOv2-giant
  (CLONE the DINOv2-Large block code from `src/hunyuan3d.zig`, add SwiGLU FFN + 224 preprocess),
  scheduler, denoise orchestration, memory staging (DINO → retire; UNet → retire before bake).
  Consumes `hunyuan3d.Engine.generateMeshRaw` output; lives behind `gen.MeshEngine`.
- `lib/xatlas/` — vendor the MIT single-amalgamation `xatlas.cpp/.h` (same pattern as
  `lib/jinja_cpp`; add to build.zig) + `src/uvwrap.zig` wrapper. Hermetic: unwrap a cube →
  chart count / UV bounds / no-overlap sampling.
- `src/rasterize.zig` — port of `rasterizer.cpp` (pixel-center, z-token, barycentric). Hermetic:
  analytic triangle coverage, z-ordering, perspective-correct barycentric on a known quad.
- `src/bake.zig` — camera matrices (golden-value tests vs `get_mv_matrix`), normal/position/alpha
  renders, view selection, `back_project` + cos⁴ blend. Hermetic: axis-colored cube → assert
  atlas face colors.
- `src/texinpaint.zig` — vertex-graph inpaint + diffusion hole fill. Hermetic: checkerboard-mask
  propagation invariants.
- `src/glb.zig` — EXTEND (additively): TEXCOORD_0 attribute, `images`/`textures`/`samplers`,
  PBR material (baseColor + metallicRoughness, embedded PNGs via `src/png.zig`). Hermetic
  parse-back + a GLBMeshLoader (Swift) fixture round-trip with texture.
- `tests/convert_hunyuan3d_paint_weights.py` + `tests/dump_hunyuan3d_paint_fixtures.py` —
  mirrors of the P1 scripts (`HY3DP_*` env prefix). Write the TENSOR-NAME CONTRACT FIRST (§3.2).
  The MLX port's `texgen/mlx/convert_weights.py` is the crib for the transforms:
  conv NCHW→NHWC, 1×1 convs → linear, **GeGLU split (value = FIRST half, gate = SECOND half)**,
  `.processor.to_*` key flattening, `to_out.0` normalization.

### 5.4 Oracle taps (`HY3DP_*`, fp16 build; thresholds ~P1)

1. `HY3DP_VAE_ENC`: image [1,3,512,512] → posterior MEAN·0.18215 [1,4,64,64] (cos > 0.999)
2. `HY3DP_DINO`: [1,3,224,224] → [1,257,1536] AND post-ImageProjModel [1,1028,1024] (cos > 0.999)
3. `HY3DP_REF`: ref latent → dual-UNet cached hiddens for 3 canary layers (down_0/mid/up_3) (cos > 0.995)
4. `HY3DP_UNET`: ONE full denoise step, fixed latents [1,2,6,4,64,64] + all conditioning →
   noise_pred [12,4,64,64] (cos > 0.995) — the big one; exercises MDA/RA/MA-RoPE/DINO jointly
5. `HY3DP_SCHED`: f32 scheduler trajectory (timesteps, zero-SNR-rescaled alphas, x_{t−1} for
   synthetic outputs) — pins v-pred + trailing + step math (near-exact)
6. `HY3DP_VAE_DEC`: latent → image [1,3,512,512] (cos > 0.999)
7. (optional) `HY3DP_E2E_ATLAS`: reference numpy dump of `bake_from_multiview` albedo atlas vs
   ours (cos > 0.98) — pins the whole geometry stack
Scheduler choice: implement DDIM (v-pred, zero-SNR, trailing, ~30 steps) FIRST — simple and
matches the shipped config; UniPC-bh2 15-step is a follow-up perf task. Dump fixtures with
whichever the Zig side implements.

### 5.5 Parity traps (verified in source — the P2 analogue of P1's list)

1. `unet/config.json` `in_channels: 4` LIES — runtime 12-ch conv_in (modules.py :818).
2. Ckpt is fp16 inside a pickle `.bin` — convert via torch `weights_only=True`, never upcast.
3. **GeGLU split order**: value first half, gate second half — wrong half still runs, output garbage.
4. VAE encode: reference uses `.sample()` under seed 0 — port + fixtures use the MEAN.
5. Negative prompt embeds = the SAME learned tokens, NOT zeros.
6. Triple-batch CFG collapses to 2-point (camera_azims never passed → view_scale=1) — exploit it,
   but validate against the 3-batch reference output.
7. Dual-UNet ref features cached ONCE per generation at constant timestep 0 — re-running per step
   is 15× waste; passing the CURRENT timestep is silently wrong.
8. RA asymmetry: query = albedo slice only; "mr" projections live on the PROCESSOR
   (`attn.processor.to_v_mr`) — key remap must flatten `.processor.`.
9. PoseRoPE: axis dims 3/8,3/8,2/8 of head_dim; interleaved-pair rotation; voxel grids keyed by
   seq-len; background = `position == 1` exactly; f32 tables.
10. MA attention: 24,576-token sequences — fused SDPA mandatory (materialized scores ≥7 GB).
11. Scheduler: v-prediction + zero-SNR rescale (compute in f32/f64) + trailing spacing.
12. Scheduler steps on `latents[:, :4]` while the UNet consumes 12ch.
13. Rasterizer bit-details (pixel centers, (width−1) scaling, z-token, barycentric renorm) must
    match — view selection consumes raw face IDs.
14. Axis/camera conventions: set_mesh negate-x/y + y/z swap + UV v-flip; `elev=-elev, azim+=90`;
    ortho ±0.6 — one sign error rotates every conditioning render.
15. MR channels: generated image R=metallic, G=roughness; glTF packs G=roughness, B=metallic.
16. f32 discipline: cos maps, cos⁴ weights, depth compare 3e-3, zero-SNR alphas, RoPE tables,
    bake accumulators (fp16 accumulation at 4096² visibly bands).
17. Bake boundary kernel is resolution-dependent: `int(2/512·render_size)` (17×17 at 2048).

### 5.6 What the existing MLX port (ZimengXiong/Hunyuan3D-MLX) is good for

- Proves the paint UNet runs fast in MLX fp16 on Apple Silicon (2.0-paint: 114s vs 302s MPS, M4 Max).
- Its `texgen/mlx/convert_weights.py` documents every weight transform; its
  `texgen/mlx/{attention,unet,hybrid_unet,pipeline}.py` is a complete second implementation of
  MDA/RA/PoseRoPE/DINO — excellent DIFF MATERIAL when oracle 4 fails.
- But: 2.1 paint is marked WIP there; renderer/bake/inpaint/xatlas stay PyTorch/C++; its scheduler
  deviates (DDIM-10 vs UniPC-15). NEVER use it as an oracle — dump fixtures from the Tencent
  reference only.

### 5.7 API + app surface

- `handleMesh` (src/gen.zig) gains `"texture": true|false` (default false initially, flip after
  validation) + optional `"texture_size"`; texture path = shape gen → `generateMeshRaw` → paint
  engine → textured `glb.writeGlb`. SSE progress spans both stages (label the phase in progress
  events like video's two-stage).
- `MeshEngine` grows a lazily-loaded paint engine slot; the load→gen→unload default flow must
  stage memory: shape engine (3.5 GB) frees before paint UNet+DINO load on 16 GB Macs.
- Model dir: paint weights convert into a SIBLING dir `~/.mlx-serve/models/local/hunyuan3d-2-1-paint-8bit`;
  the bundle/preset gains it as a dependency (MediaBundle dependency-repo pattern, like LTX+gemma).
- Swift: "Texture (PBR)" toggle in the 3D pane; SceneKit renders the textured GLB automatically
  (ModelIO reads glTF PBR); update `Model3DGenServiceTests` request contract + a textured-fixture
  GLBMeshLoader test.

### 5.8 Task order (sizes S/M/L; ⇉ = parallelizable workstreams per §3.7)

1. Tensor-name contract file + `convert_hunyuan3d_paint_weights.py` (--self-test) ⇉ (M)
2. `lib/xatlas` vendoring + `src/uvwrap.zig` ⇉ (M)
3. `src/rasterize.zig` + camera math ⇉ (M)
4. SD-VAE enc/dec in `hunyuan3d_paint.zig` + oracles 1/6 (M)
5. DINOv2-giant (+ImageProjModel) + oracle 2 (S/M)
6. DDIM v-pred/zero-SNR/trailing scheduler + oracle 5 (S)
7. Base SD2.1 UNet (resnets, linear-proj transformer blocks, GeGLU) (L)
8. 2.5D additions: MDA → RA (+dual-UNet once-per-gen cache) → MA+PoseRoPE → DINO attn +
   oracles 3/4 (L — THE core risk item; budget debugging time against the MLX port)
9. `src/bake.zig` (renders, selection, back_project, blend) ⇉ with 7-8 (M/L)
10. `src/texinpaint.zig` ⇉ (M)
11. `src/glb.zig` textures/material + Swift textured-fixture test (S/M)
12. Engine orchestration + memory staging + `handleMesh` texture wiring + `tests/test_3d_paint.sh`
    (clone test_3d_gen.sh; assert textured GLB has TEXCOORD_0/material/images) (M)
13. Fixture dump run + oracle validation at fp16 → 8-bit build → live e2e (M)
14. Follow-ups: UniPC-bh2 15-step; RealESRGAN RRDBNet; >6-view greedy selection (S/M each)

Overall: comparable to all of P1. Licensing: xatlas MIT ✓, RealESRGAN BSD-3 ✓, pymeshlab GPL ✗
(skip), Tencent community license rides with converted weights (same as shape).

## 6. Phase 3 — Rig & animate

1. **UniRig port** (github.com/VAST-AI-Research/UniRig, SIGGRAPH 2025): autoregressive skeleton
   prediction + cross-attn skinning weights; 1–5 s inference; MLX-portable transformer. Start with
   the same dossier-first workflow: spawn a design agent on the UniRig sources to pin the exact
   architecture + tokenization of skeletons before writing the plan for the engine.
2. `src/glb.zig` grows skins: joints/weights vertex attributes, inverse bind matrices, node
   hierarchy (additive, as designed).
3. Swift viewer: skeletal idle/emote clips (SceneKit `SCNSkinner` comes free from ModelIO when
   the GLB has a skin); replace the procedural turntable with idle animation; emote triggers.
4. TTS-driven speech motion: drive jaw/head bones from the Qwen3-TTS audio envelope (the audio
   pipeline exists in `AudioGenService`); no visemes in v1 — amplitude → jaw open is convincing
   at avatar scale.

## 7. Phase 4 — Avatar loop (wiring only, no new models)

1. Avatar window: SceneKit view + mic button + persona picker; chat history behind it.
2. Persona = system prompt + DocumentIndex RAG (both exist).
3. Voice = Qwen3-TTS `ref_audio` cloning (exists; clone from the same person as the photo).
4. Input = existing voice input.
5. **Sentence-level pipelining** on the single inference thread: stream LLM sentences → TTS each
   → play + drive jaw while the next sentence decodes. This is the only new engineering: a small
   scheduler-level queue discipline so TTS jobs interleave with chat decode without starving either.

## 8. Per-phase verification gates

- **Every phase**: `zig build test -Doptimize=ReleaseFast` (652+ pass baseline) + `cd app && swift
  test -Xswiftc -swift-version -Xswiftc 5` + `./tests/integration_test.sh <chat model>` (37/37)
  — chat must never regress.
- **P1.x FlashVDM**: oracle 5 still ≥ thresholds; `test_3d_gen.sh` ALL PASS; measure the volume-
  decode stage speedup and record it in `BenchmarkLog.md`.
- **P2**: `HY3DP_*` oracles 1-6 at fp16 (dump via the scratch venv:
  `source /Volumes/Sandisk_1TB/hy3d-scratch/venv/bin/activate`); 8-bit e2e relaxed threshold;
  `tests/test_3d_paint.sh` ALL PASS; live app smoke = textured model of a real photo looks like
  the person; RAM stays sane on the 16 GB dev Mac (paint peak ≤ ~5 GB at 8-bit).
- **P3**: rigged-GLB parse-back hermetic tests; SceneKit renders the skinned mesh without
  distortion at bind pose; idle clip plays; UniRig oracles vs its reference.
- **P4**: end-to-end conversation demo — speak → answer spoken in cloned voice with jaw motion,
  first audio within a few seconds of the LLM's first sentence.

## Appendix: quick-start for the next agent

```
# sanity: everything P1 still green
zig build test -Doptimize=ReleaseFast
./tests/test_3d_gen.sh                       # needs ~/.mlx-serve/models/local/hunyuan3d-2-1-8bit

# rerun P1 oracles (fixtures + fp16 build live on the Sandisk)
source /Volumes/Sandisk_1TB/hy3d-scratch/venv/bin/activate
python tests/dump_hunyuan3d_fixtures.py --repo /Volumes/Sandisk_1TB/hy3d-scratch/reference \
  --model /Volumes/Sandisk_1TB/hy3d-scratch/snapshot --out /Volumes/Sandisk_1TB/hy3d-scratch/fixtures \
  --test-model /Volumes/Sandisk_1TB/hy3d-scratch/hunyuan3d-2-1-fp16   # prints the HY3D_* env block

# P2 starts here
hf download tencent/Hunyuan3D-2.1 --include "hunyuan3d-paintpbr-v2-1/*" \
  --local-dir /Volumes/Sandisk_1TB/hy3d-scratch/snapshot
```
