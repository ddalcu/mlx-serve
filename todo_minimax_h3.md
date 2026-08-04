# MiniMax-H3 — follow-up work

Brief for a long-running agent. The port is COMPLETE and shipping: it generates
video with native stereo audio, is wired through `gen.zig` and VideoGenView, and
both suites are green (branch `feature/minmax-h3`).

**Read first:** `src/minimax_h3.zig`'s header, the `minimax_h3` rows in
`CLAUDE.md` (Layout + Supported architectures), and the three H3 rules under
`## Rules ▸ Model loading`. The reference implementation is ComfyUI's
`comfy/ldm/minimax/*` + `comfy_extras/nodes_minimax_h3.py`, mirrored at
`~/claude-tmp/h3-ref/`. It is the spec; read it before changing behaviour.

**Measured baselines (M4 Max, 128 GB), so you can tell a regression from
noise. Two eras — the fast recipe (velocity cache + attention broadcast +
dq-gemm) is DEFAULT-ON since 2026-08-03:**

| config | weights | pre-session | current defaults |
|---|---|---|---|
| 256x256, 56f, 30 steps | 8-bit | ~4 min | ~1.5 min sampling |
| 864x480, 73f, 30 steps | 8-bit | ~22 min | ~9 min |
| 1344x768, 124f, 30 steps | 8-bit | 2 h 19 m | **49 min** (measured, same seed) |
| 1344x768, 209f, 30 steps | 8-bit | ~6 h (projected) | 1 h 57 m |

`"fast": false` on the request (or `MINIMAX_H3_STEP_CACHE=0` +
`MINIMAX_H3_ATTN_BCAST=0`) restores the dense per-step arms for A/Bs.

Run one with:

```bash
MINIMAX_H3_DIR=~/claude-tmp/h3-build \
MINIMAX_H3_MODEL=~/.mlx-serve/models/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit \
MINIMAX_H3_OUT=~/claude-tmp/h3out MINIMAX_H3_SIZE=256 MINIMAX_H3_FRAMES=56 MINIMAX_H3_STEPS=30 \
zig build test -Doptimize=ReleaseFast -Dtest-filter="minimax h3 live"
```

Note `zig build` CACHES test runs and does not track env vars — after the first
run, invoke the built binary directly (`ls -t .zig-cache/o/*/test | head -1`) or
you will silently re-read a stale result. Output under `nohup` is fully
buffered, so absence of step lines is not absence of progress; check GPU busy
with `ioreg -r -d 1 -w 0 -c IOAccelerator | grep "Device Utilization"` instead.

---

## Phase 0 — MEASURE. Gates everything below.

**DONE 2026-08-03** (in-run ablation ladder `MINIMAX_H3_ABLATE=ladder` — arms
remove one component from the step graph with shapes intact, so attribution
needs no inserted syncs; logs in `~/claude-tmp/h3-prof/queue.log`).

Results at 864x480 / 73f, 8-bit pack, per step (stock qmm baseline 41.3 s):

| component | s/step | share |
|---|---|---|
| attention total (qkv+SDPA+out) | 22.4 | 54% |
| — of which SDPA alone | 10.1 | 24% |
| MLP (fc1+SwiGLU+fc2) | 18.6 | 45% |
| AdaLN | ~0 | 0% (as predicted — never fuse it) |

- [x] **Profile one 480p generation.** Weight-load ~36 s cold (attributed by
      the forced `evalWeights`), TE ~15 s cold / ~1 s warm, VAE decode ~1-2 s,
      audio ~0.05 s. The step loop is everything.
- [x] **bf16 vs 8-bit control at 864x480**: bf16 arms 41-48 s/step and JITTERY
      (62 GB resident sits near the working-set edge); 8-bit stock qmm 42 s;
      **8-bit + wide-M dq-gemm 36.6 s** (2/2 adjacent pairs, output
      byte-identical at u8). The 8-bit pack is the right serving config and
      `MLX_SERVE_MF_DQ_GEMM` is now DEFAULT-ON for the H3 path at the 2048-row
      floor. Corollary: a dequant-once bf16 weight cache is a DEAD END — it
      recreates the bf16 residency regime that measured slower.
- [x] **SDPA "anomaly" RESOLVED — there is none.** The scare was a
      per-block-vs-per-step FLOP confusion: one SDPA at [1,56,9266,128] is
      ~2.5 TFLOP and ubenches at **186 ms = 13.3 TFLOPS effective** (pip mlx
      AND our libmlx agree); the step has FIFTY of them, 50 x 186 ms = 9.3 s
      ≈ exactly the ladder's 10.1 s (the plain `sdpa` arm also pruned the
      dead q/k branches, ~0.6 s — the `sdpa_dep` dependency-preserving arm
      measures 27.0 vs 36.6 baseline = 9.6 s honest). MLX's full-attention
      kernel is NEAR-ROOFLINE at this shape; a custom kernel's ceiling is
      ~2% of the step. Do not write one.

Linear GEMMs measured ~13.6 TFLOPS effective with dq-gemm and SDPA at 13.3 —
the whole DiT step is at the compute roofline, exactly as the arithmetic
predicted. Wall-clock now only moves by doing FEWER forwards (steps,
step-cache) or less math per forward (sparse attention, still withheld
upstream).

---

## Phase 1 — Correctness and completeness gaps

**These are real defects, not polish. Do them before any perf work.**

- [x] **`tests/test_minimax_h3.sh`** written (capability + staged-preflight
      engagement + LoRA/canvas/chat named 400s + rgb8/pcm length checks + SSE
      progress→complete with 40→56 snapping + fl2va first-frame adherence +
      bad-keyframe 400). In `tests/CLAUDE.md`'s matrix.
- [x] **`docs/reference.md` section** added (MiniMax-H3 bullet block under
      Unified media generation).
- [x] **VAE encode + fl2va first-frame conditioning** (2026-08-03): the
      single-frame (T==1) conv encoder is `minimax_h3_vae.Encoder` — for one
      frame every CausalConv3d collapses exactly to a 2D conv on its LAST
      temporal tap (the causal front pads are zeros), so the full-T encoder
      waits for ref2va. Tiled moments encode mirrors `tiled_encode` (blend at
      latent granularity vs RAW neighbours). Parity is against the EXECUTED
      reference (`tests/dump_minimax_h3_vae_encoder_fixture.py` — plain torch,
      unlike the DiT), single-tile AND tiled cases. `generate` composes
      `[cond rows | target]` per step with the 0.999 noise aug (mlx RNG — seed
      streams are not portable, same class as the initial latents);
      `first_frame_image` (stretch) / `last_frame_image` (center-cover) on the
      request; undecodable → named 400 (the a2vid rule). KNOWN DEVIATION: the
      reference also splices keyframes into the Qwen encoding as
      `<Picture i>: <vision block>` (tag-0 spans); we condition through VAE
      latents only until the vision tower lands with ref2va.
- [x] **Spatial tiling parity**: seam-energy live test for the tiled DECODE
      (smooth synthetic latent at 384px → no gradient spike at the known seam
      vs ambient median; measured seam/median 1.59x col / 2.31x row against a
      3x bound — tiles are semantically different decodes, so a modest bump is
      the REFERENCE's behavior too; a hard concat reads ~10x), and the tiled
      ENCODE is pinned against the reference executing its own `tiled_encode`
      (cos 0.999998 single-tile AND tiled, 2026-08-03).
- [x] **The memory preflight over-bills staged-residency models.** Fixed:
      `gen.h3PeakBytes` (max(TE,DiT)+VAEs) via `estimatePeakResidentBytes`,
      wired as a media preflight in `doLoadGenOnInferenceThread` (media loads
      previously had NO preflight at all); other media types keep the
      sum-of-safetensors bill. Direction-safe by construction; unit tests in
      gen.zig. ORIGINAL TEXT:
      `modelDiskBytes` sums every safetensors in the dir, so H3 is billed
      64.5 GB (`[preflight] weights ~64.53 GB` on a real boot). But
      `minimax_h3.generate` loads the text encoder, frees it, THEN loads the
      DiT, so true peak is ~35 GB plus VAEs and activations, not their sum.
      Over-billing fails safe, but on a 48 GB Mac it refuses a load that would
      have worked. The engine knows its own residency plan and the preflight
      does not; either give the media path a per-backend estimate or have
      `VideoEngine` declare a peak. Whatever shape it takes, the fix must not
      let a backend UNDER-bill — an MLX OOM is uncatchable, so the direction of
      the error matters more than its size.
- [x] **On-spec acceptance run DONE** 2026-08-03: 124f @ 1344x768, 30 steps,
      defaults (precompute + dq-gemm, no step cache). Sampling 2.31 h at a
      flat 275.7 s/step (zero drift step 1→30), tiled VAE decode 2.4 min,
      audio 0.7 s, in-sync stereo (5.175 s audio vs 5.167 s video). DiT
      resident 20.46 GB THROUGHOUT — the precompute + weights-map-scope fixes
      hold at the trained scale. Artifacts: `~/claude-tmp/h3-prof/out_accept/`
      + `h3_accept_1344x768_124f.mp4` (sent to David). Harness note: an EMPTY
      env var (`VAR=`) reaches getenv as "" not null — fixture-gated tests
      must len-check or load_safetensors("") kills the binary (fixed).

---

## Phase 2 — Footprint (do regardless of the profile)

- [x] **Engine-side AdaLN precompute LANDED** (2026-08-03, default ON, kill
      `MINIMAX_H3_ADALN_PRECOMPUTE=0`): `Model.precomputeAdaln` materializes
      the whole schedule's modulations (union of every step's unique t via
      `collectScheduleTs` — bit-identical f64s to the loop's own lookups,
      `buildTimestepPlanGlobal` remaps runs into the global table) while the
      trunk is still LAZY, freeing each block's 260M-param AdaLN right after
      its table evals — transient peak one block, resident win ~13 GB on the
      8-bit pack. NOT bit-identical to per-step modulation (M=n_ts vs M=2
      kernel selection, the sanctioned class); PSNR-checked in postqueue.
      The converter-side PRUNED PACK (ship tables/basis instead of weights,
      69 -> ~55 GB on disk) remains open — note Comfy's curve-form checkpoints
      (`use_adaln_curves` in the reference model.py) are the shape to match if
      we publish one. ORIGINAL:
      Modulation depends only on the timestep and there are <=4 unique ones per
      step, so the whole schedule can be materialized up front (~1.2 GB for 30
      steps) and the 13.04B AdaLN parameters never loaded. This is **39% of the
      model**: the pack goes 69 -> ~55 GB and DiT residency 35 -> ~21 GB, which
      is what makes a 36/48 GB Mac viable. Comfy ships exactly this as its
      "pruned" variant. NOTE: this is a MEMORY win, not a speed one (the FLOPs
      and the per-step read are both ~1%); do not sell it as perf.
      When it lands, add a pruned variant to the converter and re-publish.

---

## Phase 3 — Perf, ONLY if Phase 0 shows headroom

Ordered by expected value given what is already known.

- [~] **Fewer steps.** Ladder RUN 2026-08-03 (256px, same seed): 30 vs 20 vs
      16 clips sent (`h3_s{20,16}.mp4` vs `h3_new.mp4`); no eyeball ruling
      yet. LOWER PRIORITY now the fast recipe landed — the step cache already
      harvests the adjacent-step redundancy that fewer steps would, so the
      incremental win is smaller than it looked pre-recipe; if judged, it
      stacks (would compose to ~4x).
- [x] **Step cache (TeaCache-style velocity reuse)**: reuse the previous
      velocity when the accumulated relative input change stays under a
      threshold (audio velocity re-scaled to the current sigma's slope).
      0.08 skipped 15/30 forwards = 2.0x sampling at 256px. DEFAULT-ON at
      thresh 0.05 since the capstone eyeball verdict (see below), via
      `resolveSpeed`; `MINIMAX_H3_STEP_CACHE` overrides both ways.
- [x] **dequant -> bf16 GEMM at M>=2048**: LANDED, default-on for the H3 path
      (−13%/step at 480p, 2/2 adjacent pairs, u8-identical output; see
      Phase 0). MageFlow keeps its own default until its own A/B.
- [x] **Attention broadcast (PAB-style)** 2026-08-03, DEFAULT-ON at k=2 via
      `resolveSpeed` since the capstone verdict (env overrides): per-block
      attention OUTPUTS are cached and the whole branch (norm1/mod/qkv/SDPA/
      out — ~70% of a 768p step) skipped on non-refresh steps; gate re-applied
      at the current timestep. `MINIMAX_H3_ATTN_BCAST=<k>` (warmup 4 + last 2
      always refresh). Measured: broadcast step 15.5 s vs 35.7 full at 480p
      (the ladder's attention-total removed exactly); 256px k=2 PSNR 26.6 dB
      vs baseline = INSIDE the sanctioned numeric-drift band (26.8), k=3
      22.6 dB. Extrapolated 768p: k=2 ~1.4x, k=3 ~1.6x. Cache cost one
      [S,hidden] bf16 per block (~20 GB at 768p/124f, ~34 GB at 209f —
      measured fine on 128 GB; budget it before raising k or frames on
      smaller Macs). Clips: h3_pab2/h3_pab3.mp4.
- [~] **Sparse video attention (training-free, SVG/STA-style), NEW lever**:
      per-layer spatial (within-frame) / temporal (same-patch stripe)
      attention with the GLOBAL STRIP (text/cond/audio rows) concatenated into
      every tile's key set; strip queries keep full attention; dense anchor
      layers at ends+middle under `MINIMAX_H3_SPARSE_ATTN=mix`. Pure mlx ops
      (batched SDPA with the pattern axis folded into batch — no custom
      kernel needed for v1). Subset selection PROVEN by the uniform-score
      closed-form test (q=k=0 ⇒ each output row is the mean of exactly its
      subset). MEASURED 2026-08-03: 480p step 29.4 vs 36.6 s dense (−20%;
      projects ~1.9x at 768p where attention is 3.5x the share). QUALITY
      CAUTION on v1's `mix` policy: PSNR vs same-seed dense is 9.8 dB and the
      sparse clip x264-compresses at 10x the control's bitrate — the classic
      added-noise tell; eyeball pair h3_sp480q vs h3_d480q. If degraded (as
      expected), the tuning ladder is: denser anchor cadence (every 4th layer
      dense), spatial-only sparse layers (temporal stripes are the likelier
      harm), then SVG-style per-head online classification. The MACHINERY is
      landed and subset-proven; only the layer policy needs iteration.
- [x] **4-bit pack (custom quant)**: built via `--bits 4 --cpu` (40.3 GB vs
      69; DiT resident 11.11 GB measured; compute-neutral). EYEBALL VERDICT
      (David): "q4 looks good, worth offering as an additional option,
      especially for people with not a lot of ram" → shipped as a SECOND app
      preset (`minimaxH3Q4`, low-RAM copy, approxRAMGB 26) beside the 8-bit
      default; both ride one factory so they cannot drift. Card/NOTICE/
      MODIFICATIONS say 4-bit (converter now interpolates BITS). Upload is
      David's: `hf upload ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit <dir> .` —
      the preset 404s on download until that runs.
- [x] **Capstone MEASURED** 2026-08-03: the acceptance config (768p/124f/30
      steps) with step-cache 0.05 + broadcast k=2 sampled in **49.0 min vs
      2 h 19 min = 2.83x**, same seed (14 velocity-cached + 7 broadcast + 9
      full steps; cadence full ~280 s / broadcast ~63 s / cached ~0.02 s).
      EYEBALL VERDICT (David, 2026-08-03): "both look really good, non-fast
      just a smidge better" → the fast recipe is DEFAULT-ON. `resolveSpeed`
      wires it: request `"fast": false` is the off switch (app: "Max quality
      (slower)" toggle, H3 pane only), env vars stay the strongest knob for
      A/Bs. Integration pins BOTH directions by engagement count.
- [ ] **Do NOT bother fusing the AdaLN modulation.** It looks like weak code and
      it is, but the arithmetic says ~105,000 dispatches (~0.16 s) plus ~10 s of
      data movement in a 1320 s run: under 1%. This repo has burned this exact
      lesson twice already — `fusedAttnGate` measured NEUTRAL and
      `fusedAddRmsNorm` measured ~1% NEGATIVE, because at these widths the GPU
      already overlaps small ops behind big GEMMs. Left here explicitly so
      nobody re-derives it as a good idea.
- [ ] **Sparse attention is the only large FLOP saving** (attention is 25% of
      cost at 480p, 59% at 768p) and MiniMax deliberately withheld it from the
      release. Watch for it; nothing to build meanwhile.
- [ ] **NAX** is M5-only. Nothing to do on an M4.

---

## Phase 4 — Features

- [ ] **Ref2VA** (the second checkpoint, 66 GB). Needs: the Qwen3-VL vision
      tower (mage_flow.zig has it, currently unused by H3), the reference
      presentation format (`<Picture i>: ` / `<Video k>: ` + `<T.T seconds>`
      blocks / `<Audio j>: `, exact wording in `comfy/text_encoders/minimax.py`),
      the ref-block branch of `PackedLayout` (deliberately deferred — the
      hermetic layout fixture covers t2va/fl2va only, so EXTEND
      `tests/dump_minimax_h3_layout.py` with ref cases first), the audio VAE
      ENCODE side, and a `Ref2VA` preset arm.
- [ ] **2K output** needs `H3-Regenerate-2K`, which is API-only and not in the
      open release. Not portable.

---

## Traps already paid for — do not rediscover

- The ComfyUI **int8/nvfp4 packs are unreadable** on Metal (`convrot` is a
  comfy_kitchen CUDA tensor-core layout, and W4A4 so activations are quantized
  too). Use the bf16 files and our own converter.
- **safetensors' numpy framework cannot represent bfloat16.** The converter must
  read through `mx.load`'s lazy mmap; this is the only path that works, not just
  the faster one.
- **Spatial tiling is SEMANTIC.** `create_token_ids` normalizes coordinates over
  the extent it is handed, so a tile is not a slice of an untiled pass. Same for
  temporal chunking (the VAE was trained on 17-frame clips).
- **Quantize only what a matmul reads.** `NEVER_QUANTIZE` in the converter is an
  explicit list because shape cannot distinguish a gathered table
  (`embed_tokens`, `pos_embed`) from a linear, and packing one yields silent
  garbage through `take_axis`.
- **Frame counts are 17k+5, not LTX's 8N+1**, and positions are AREA-NORMALIZED
  (aspect ratio moves every coordinate). Rope is PARTIAL: 96 of head_dim 128,
  top 32 pass through.
- **`modalityFromType` and `model_discovery.isMediaModelType` are duplicated on
  purpose** (discovery must not import mlx) and drifted once already, producing
  a 400 for a model the server could serve. `gen.media_model_types` plus its
  bidirectional test is the guard; extend the list, not one predicate.
- **Hiding a control is not the same as not sending its field.** The pane
  gated H3's CFG/LoRA/pipeline controls, but `VideoGenService.requestBody`
  still put `pipeline` in the body unconditionally, so every H3 generation
  carried a field that backend has no concept of and the server's named 400
  fired on all of them. The request builder now gates on the same declared
  capabilities the view does. Corollary learned the hard way: a server-side
  400 on the mere PRESENCE of a field is brittle when a shared client always
  sends it — reject only what cannot be honored in any form (LoRA), and ignore
  what is merely inapplicable (CFG on a distilled model).
- **A residency ESTIMATE belongs to a BACKEND too** (second bite of the rule
  below, 2026-08-03): the media-load stub config's `model_type` is a MODALITY
  static ("AudioVideo" for every video backend), so the staged preflight keyed
  on it billed H3 the 64.5 GB sum while its log said "staged". The scheduler
  now re-peeks the dir's REAL type (`gen.peekModelType` — the engine
  dispatch's own authority) and `tests/test_minimax_h3.sh` asserts the NUMBER,
  never the log line's presence.
- **A readiness marker belongs to a BACKEND, never to a modality.**
  `detectModality` gated the whole `.video` modality on LTX's
  `connector.safetensors`. H3 has no connector, so detection returned null and
  `preloadCpuState` fell through to the MLX TEXT path — it globbed all four H3
  safetensors into one weight map and died on `model.norm.weight`, which reads
  as a tensor bug rather than a routing bug. `requiredMarkerFor(model_type)` is
  the guard, and the failure now names the missing file instead of silently
  degrading to a text load. Any per-modality condition is suspect the moment
  that modality has two backends; the same shape is worth auditing in the
  image/audio paths, which have had unions for longer.
- A **permutation-invariant checksum cannot see a permutation** — the layout
  fixture pairs each column sum with a row-index-weighted one after a stereo
  channel swap slipped through.

## License

The pack redistributes modified weights under the MiniMax H3 Community License:
`LICENSE`, `NOTICE` and `MODIFICATIONS.md` ship in the model dir and the
converter treats a missing LICENSE as fatal. **The Agreement's Applicable
Territory excludes the EU, UK, South Korea and the USA** (Section V.4 covers
use, reproduction, modification, distribution and display). Anything that
changes how this is published should re-read Section III.
