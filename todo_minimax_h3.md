# MiniMax-H3 — follow-up work

Brief for a long-running agent. The port is COMPLETE and shipping: it generates
video with native stereo audio, is wired through `gen.zig` and VideoGenView, and
both suites are green (branch `feature/minmax-h3`).

**Read first:** `src/minimax_h3.zig`'s header, the `minimax_h3` rows in
`CLAUDE.md` (Layout + Supported architectures), and the three H3 rules under
`## Rules ▸ Model loading`. The reference implementation is ComfyUI's
`comfy/ldm/minimax/*` + `comfy_extras/nodes_minimax_h3.py`, mirrored at
`~/claude-tmp/h3-ref/`. It is the spec; read it before changing behaviour.

**Measured baseline (M4 Max, 128 GB), so you can tell a regression from noise:**

| config | weights | wall clock |
|---|---|---|
| 256x256, 56f, 30 steps | bf16 | ~4 min |
| 864x480, 73f, 30 steps | 8-bit | ~22 min |

Run one with:

```bash
MINIMAX_H3_DIR=~/claude-tmp/h3-build \
MINIMAX_H3_MODEL=~/.mlx-serve/models/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit \
MINIMAX_H3_OUT=/tmp/h3out MINIMAX_H3_SIZE=256 MINIMAX_H3_FRAMES=56 MINIMAX_H3_STEPS=30 \
zig build test -Doptimize=ReleaseFast -Dtest-filter="minimax h3 live"
```

Note `zig build` CACHES test runs and does not track env vars — after the first
run, invoke the built binary directly (`ls -t .zig-cache/o/*/test | head -1`) or
you will silently re-read a stale result. Output under `nohup` is fully
buffered, so absence of step lines is not absence of progress; check GPU busy
with `ioreg -r -d 1 -w 0 -c IOAccelerator | grep "Device Utilization"` instead.

---

## Phase 0 — MEASURE. Gates everything below.

Do not write a kernel before this. Arithmetic says the DiT is running at roughly
13-14 TFLOPS effective against a ~14.4 TFLOPS dense GEMM roofline, i.e. close to
saturated, but that is a FLOP count divided by one wall-clock number, not a
profile. If it holds, most of the perf list below is not worth doing.

- [ ] **Profile one 480p generation** split into DiT-linear / DiT-attention /
      AdaLN / VAE-decode / weight-load. Beware `MLX_SERVE_DECODE_PROFILE`'s
      lesson: inserting per-phase evals destroys pipelining and inflates what it
      measures. Phase laps are honest only where the path already syncs.
- [ ] **bf16 vs 8-bit control at 864x480.** Never run. The 22 min number
      conflates resolution with quantization, so "8-bit is slower at prefill
      widths" is still a PREDICTION. Same prompt, seed, steps; alternate the
      arms rather than running one then the other (thermal drift).

Expected FLOP split at 864x480 / 30 steps, for sanity-checking the profile:
linear ~10.7 PF, attention ~3.7 PF, AdaLN ~0.002 PF, VAE ~0.6 PF.

---

## Phase 1 — Correctness and completeness gaps

**These are real defects, not polish. Do them before any perf work.**

- [ ] **No SSE progress.** `handleVideoH3` ignores `stream` and answers with one
      `sendBytesJson` at the end. A 22-minute generation therefore shows the app
      a dead progress card, which is exactly the "a media generation shows a
      meter or it reads as a hang" rule in `app/CLAUDE.md`. Wire `sse.Progress`
      through `minimax_h3.generate`'s step loop (it already logs per step) and
      emit the same event shape the LTX path does. **Highest-value item in this
      file** — it is user-visible on every single run.
- [ ] **No `tests/test_minimax_h3.sh`.** Every other media backend has one. It
      should boot a server over the converted dir and assert: `/v1/models`
      advertises `video`, a small generation returns rgb8 + pcm_s16le of the
      right lengths, the four named 400s fire (`lora_path`, `cfg_scale`,
      `stg_scale`, `pipeline`), a non-32-multiple size 400s, and a chat request
      against it 400s with the media-modality message. Add it to
      `tests/CLAUDE.md`'s matrix.
- [ ] **No `docs/reference.md` section.** The growth policy requires one per
      subsystem (Layout row + reference section); only the Layout row exists.
- [ ] **VAE encode is unimplemented**, so first/last-frame conditioning (the
      "fl2va" half of the checkpoint's own name) does not work even though the
      DiT and `PackedLayout` already support keyframes end to end. Needs the
      conv encoder from `vae.py`. `VideoGenView` still shows its First-frame
      picker for H3 — either implement this or gate the control.
- [ ] **Spatial tiling has no output parity test.** `splitTiles` is pinned
      hermetically, but `decodeSpatial`'s blend/trim path is only validated by
      "the 480p clip looked right." A tiled decode of a >256 px canvas should be
      compared against a reference dump, or at minimum a seam-energy check.
- [ ] **The memory preflight over-bills staged-residency models.**
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
- [ ] **On-spec acceptance run never done**: 124 frames at 1344x768. Expect
      several hours. This is the only config inside the model's TRAINED range
      (~124-362 frames at 768 short edge); everything shipped so far is
      deliberately off-distribution.

---

## Phase 2 — Footprint (do regardless of the profile)

- [ ] **Precompute AdaLN across the sampling schedule and drop the weights.**
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

- [ ] **Fewer steps.** Perfectly linear and completely untested for quality.
      30 vs 20 vs 16 at 256x256, same seed, eyeball the three. If 20 holds this
      is a 33% cut for one afternoon, larger than anything else here.
- [ ] **dequant -> bf16 GEMM at M>=2048** for the quantized path, mirroring the
      existing `prefillDqGemm` rule. Only if the Phase 0 control shows 8-bit is
      actually slower. Cheap if so.
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
