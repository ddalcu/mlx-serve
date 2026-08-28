# Anima image-gen support — implementation spec

Working spec for adding Anima (circlestone-labs/Anima) as a native image backend.
Anima = NVIDIA **Cosmos-Predict2 2B** DiT (`MiniTrainDIT`) + an **Anima-specific
`LLMAdapter`**, conditioned by a **Qwen3-0.6B** encoder, decoded by the
**Qwen-Image VAE**, sampled with **Cosmos rectified flow**. Anime/illustration model.

Reference source (ComfyUI is the ground truth — Anima ships as ComfyUI single-file):
- DiT: `comfy/ldm/cosmos/predict2.py` (`MiniTrainDIT`), `comfy/ldm/cosmos/position_embedding.py`
- Anima wrapper: `comfy/ldm/anima/model.py` (`Anima`, `LLMAdapter`, `TransformerBlock`, `RotaryEmbedding`)
- Text: `comfy/text_encoders/anima.py` (`AnimaTokenizer`, `Qwen3_06BModel`), `comfy/text_encoders/llama.py` (`Qwen3_06B`)
- Sampler: `comfy/model_sampling.py` (`COSMOS_RFLOW`, `ModelSamplingCosmosRFlow`)
- Model wiring: `comfy/model_base.py` (`Anima`), `comfy/supported_models.py` (`Anima`), `comfy/model_detection.py` (image_model "anima"), `comfy/latent_formats.py` (`Wan21`)
- VAE: diffusers `AutoencoderKLQwenImage` (or ComfyUI equivalent). z_dim 16.
Copies saved under scratchpad `ref/` during development.

## Checkpoint layout (HF circlestone-labs/Anima, ComfyUI split_files — NO config.json)

```
split_files/diffusion_models/anima-<variant>.safetensors   # DiT + llm_adapter, ~2GB bf16
split_files/text_encoders/qwen_3_06b_base.safetensors      # Qwen3-0.6B, ~1.2GB
split_files/vae/qwen_image_vae.safetensors                 # Qwen-Image VAE, ~250MB
```
Variants: `anima-base-v1.0`, `anima-aesthetic-v1.0/1.0b/1.1`, `anima-turbo-v1.0/1.1`,
`anima-preview/preview2/preview3-base`. Turbo = 8-12 steps CFG 1; others 30-50 steps CFG 4-5.

Weight-name prefixes in the DiT file (from model_detection): `blocks.<i>.mlp.layer1.weight`,
`x_embedder.proj.1.weight`, `llm_adapter.blocks.0.cross_attn.q_proj.weight`.

## DiT config (2B) — from model_detection.py cosmos_predict2 + image_model=="anima"

```
in_channels=16, out_channels=16 (17 in-proj after concat_padding_mask)
patch_spatial=2, patch_temporal=1
model_channels=2048, num_heads=16, num_blocks=28 (count from state dict), mlp_ratio=4.0
crossattn_emb_channels=1024
pos_emb_cls="rope3d", extra_per_block_abs_pos_emb=False  (in_channels==16 branch)
rope_h/w_extrapolation_ratio=4.0, rope_t=1.0, rope_enable_fps_modulation=False
use_adaln_lora=True, adaln_lora_dim=256
max_img_h=240, max_img_w=240, max_frames=128, min_fps=1, max_fps=30
concat_padding_mask=True
```
Head dim = 2048/16 = 128. RoPE 3D over (T,H,W) with T=1 for images; dim split
dim_h=dim_w=head_dim//6*2=42, dim_t=head_dim-2*dim_h=44. h/w theta=10000*4^(42/40),
t theta=10000 (ratio 1). fps modulation OFF (image). Extra abs pos emb NOT built (16ch).

### Block (predict2.py `Block`), residual stream in fp32:
1. self-attn: LN(no affine,eps1e-6) → AdaLN(shift,scale,gate from SiLU+Linear(→adaln_lora_dim→3D)+adaln_lora)
   Attention: q/k/v proj (no bias), per-head RMSNorm q_norm/k_norm (eps1e-6, head_dim), v_norm=Identity,
   RoPE applied to q,k (self-attn only), out_proj. addcmul gate.
2. cross-attn: same modulation shape; context = adapter output (512x1024); NO rope on cross; k/v from context.
3. mlp: GPT2FeedForward = Linear(2048→8192) GELU Linear(8192→2048). addcmul gate.
t_embedder: Timesteps(2048, sincos cos||sin base1e4) → TimestepEmbedding(SiLU, linear_2→3*D since adaln_lora)
  → returns (emb, adaln_lora_B_T_3D). Then t_embedding_norm = RMSNorm(2048,eps1e-6).
PatchEmbed: Rearrange b c (t) (h m)(w n) -> b t h w (c m n) then Linear(16*1*2*2 +pad*4 =68 → 2048) no bias.
  (concat padding mask adds 1 channel BEFORE patchify → in 17ch → 17*4=68 in-features.)
FinalLayer: LN(no affine) → AdaLN(2 chunks shift,scale via SiLU+Linear→adaln_lora_dim→2*D + adaln_lora[:2D])
  → Linear(2048 → 2*2*1*16=64). unpatchify → B,16,T,H,W. Crop to orig latent H,W.

## LLMAdapter (anima/model.py) — runs BEFORE the DiT, once per prompt

```
embed: Embedding(32128, 1024)         # T5 vocab; indexed by t5xxl_ids -> query tokens x
in_proj: Identity (model_dim==target_dim==1024)
rotary_emb: RotaryEmbedding(head_dim=1024/16=64, theta 10000)  # Llama-style rope, cos||cos
blocks: 6 x TransformerBlock(source_dim=1024, model_dim=1024, num_heads=16, use_self_attn=True, layer_norm=False)
out_proj: Linear(1024,1024), norm: RMSNorm(1024,eps1e-6)
```
TransformerBlock (RMSNorm since layer_norm=False, eps1e-6, pre-norm, residual add — NO adaln/gate):
  x += self_attn(norm_self_attn(x), rope on q AND k with x-positions)
  x += cross_attn(norm_cross_attn(x), context=qwen_hidden, rope on q(x-pos) and k(context-pos))
  x += mlp(norm_mlp(x)); mlp = Linear(1024→4096) GELU Linear(4096→1024)
Attention here: q_proj/k_proj/v_proj/o_proj no bias, per-head RMSNorm q_norm/k_norm(eps1e-6, head_dim=64),
  NO v_norm, SDPA. rope: rotate_half convention (Llama), cos/sin duplicated (cat(freqs,freqs)).
preprocess_text_embeds(qwen_hidden, t5xxl_ids, t5xxl_weights):
  out = adapter(qwen_hidden, t5xxl_ids); out *= t5xxl_weights (per-token, unsqueezed);
  if out.shape[1] < 512: pad seq to 512 with zeros. -> DiT context [1,512,1024].

## Text pipeline (anima.py) — TWO tokenizers on the same prompt string

1. Qwen3 tokenizer: Qwen2Tokenizer (qwen25_tokenizer), no start/end token, pad=151643,
   weights forced 1.0. -> qwen ids.
2. Run Qwen3-0.6B (SDClipModel, layer="last", layer_norm_hidden_state=False, attention_mask=True)
   -> last hidden states [1,Lq,1024] = source_hidden_states (cross_attn). Qwen3-0.6B: hidden 1024,
   heads 16, kv-heads 8, head_dim 128, 28 layers, RMSNorm, SwiGLU, qk-norm, rope theta 1e6 (verify from TE).
3. T5 tokenizer: T5TokenizerFast (t5_tokenizer, SentencePiece **unigram**), no start token, has end token,
   -> t5xxl_ids + t5xxl_weights (weights from prompt weighting syntax; 1.0 default).
   >>> NEW DEPENDENCY: mlx-serve tokenizer.zig is BPE only. Need a UNIGRAM (Viterbi) tokenizer
   >>> + the T5 spiece vocab. Bundle t5 tokenizer vocab as a fixture / in the pack.

## Sampler — rectified flow (CONST + ModelSamplingDiscreteFlow) — PINNED, no live dump needed

**Correction (was wrong earlier in this doc):** `comfy/model_base.py Anima.__init__` never
passes `model_type` to `BaseModel.__init__`, so it defaults to `ModelType.FLOW` — NOT
`ModelType.FLOW_COSMOS`. `model_base.model_sampling()` maps `FLOW -> (CONST, ModelSamplingDiscreteFlow)`,
the same standard flow-matching mixin Flux/SD3 use. `COSMOS_RFLOW`/`ModelSamplingCosmosRFlow`
are NOT used by Anima despite the DiT trunk being named "Cosmos" — those classes belong to
NVIDIA's own CosmosPredict2 image/video checkpoints. Confirmed independently: `ModelSamplingCosmosRFlow.sigma/.timestep`
never reference `self.shift` at all, so `shift: 3.0` in `Anima.sampling_settings` would be
inert garbage under that mixin — it only does anything under `ModelSamplingDiscreteFlow`.

`sampling_settings = {"multiplier": 1.0, "shift": 3.0}`, `latent_format = Wan21` (16ch).

- `time_snr_shift(alpha, t) = alpha*t/(1+(alpha-1)*t)` (model_sampling.py) — pure function, alpha=3.0.
- `ModelSamplingDiscreteFlow` discrete sigma buffer (1000 entries): `sigmas[i] = time_snr_shift(shift, (i+1)/1000)`
  for i=0..999 (ASCENDING, since multiplier=1.0 makes `timestep/multiplier` == the raw fraction).
- `timestep(sigma) = sigma * multiplier = sigma` (multiplier 1.0 -> **identity**): the DiT's
  `t_embedder` receives the raw sigma value directly, in [0,1]-ish domain. Confirmed against the
  DiT parity fixture, which forwards `timesteps=[0.7]` unmodified — this is exactly the sigma
  domain the schedule produces, not a *1000-scaled discrete step index.
- "simple" scheduler (`comfy/samplers.py simple_scheduler`, the ComfyUI default for flow models):
  `ss = 1000/steps`; for x in 0..steps-1: `sig[x] = sigmas[-(1+int(x*ss))]` (reads the ascending
  buffer backwards); append `sig[steps] = 0.0`. Descending schedule from ~1.0 to 0.0.
- `CONST.calculate_input(sigma, noise) = noise` (**identity** — no input rescale, unlike COSMOS_RFLOW).
- `CONST.calculate_denoised(sigma, out, x) = x - out*sigma` (`out` is a velocity prediction).
- `CONST.noise_scaling(sigma, noise, lat) = sigma*noise + (1-sigma)*lat` (lat=0 for pure txt2img).
- Euler step (comfy `sampling_function`/`cfg_function`, confirmed from source): CFG combines in
  **denoised (x0) space**: `denoised_cfg = uncond_denoised + cfg*(cond_denoised - uncond_denoised)`.
  For CONST math this is algebraically identical to combining the raw velocities directly
  (`d = uncond_out + cfg*(cond_out-uncond_out)`) — verified by substitution, see anima.zig
  `cfgCombine` doc comment. Step: `x = x + d*(sigma_next - sigma)`; at the final step
  (`sigma_next=0`) this reduces exactly to the last `denoised` estimate. `cfg==1.0` (Turbo) skips
  the uncond forward entirely (comfy's own `disable_cfg1_optimization` path) — free 2x speedup.
- This whole pipeline mirrors `src/flux.zig`'s `computeSchedule`/`generateFromCondWithOpts` Euler
  loop (Flux is ALSO `CONST`-mixin, `ModelType.FLUX`; only the shift/schedule function differs —
  Flux's resolution-dependent `flux_time_shift` vs Anima's fixed `time_snr_shift(3.0, ·)`).
- All of the above is exact, pure-math, and requires NO live ComfyUI dump — it's read straight
  from `comfy/model_sampling.py` + `comfy/samplers.py`, not reverse-engineered from behavior.
Wan21 latent process_in/out: (lat-mean)/std and lat*std+mean; per-channel mean/std tables (16 values,
in this file). Applied on VAE encode(out)/decode(in) boundary, NOT inside RF (RF sees normalized latents).

## Qwen-Image VAE (diffusers AutoencoderKLQwenImage) — z_dim 16

WAN2.1-derived causal 3D VAE; for images use single frame (T=1). 8x spatial compression.
Decoder: latent[1,16,1,H/8,W/8] -> RGB[1,3,1,H,W]. Encoder needed for img2img/edit.
process latents with Wan21 mean/std BEFORE decode (process_out) / AFTER encode (process_in).
Read structure from ref/qwen_image_vae.py during VAE phase. Likely reuses conv3d/resnet/attn.

## mlx-serve integration points

- Classification: NO config.json. Add `isAnimaRepo` predicate (model_discovery.zig, fs-only) keyed on
  `split_files/diffusion_models/anima-*.safetensors` + `split_files/vae/qwen_image_vae.safetensors`
  presence (mirror peekMfluxFlux2 / isMageFlowRepo). `gen.peekModelType` synthesizes "anima".
  Add "anima" to gen.media_model_types, modalityFromType (->.image), discovery.isMediaModelType,
  requiredMediaMarker (require the vae file). Keep Swift LocalModel classifier in sync.
- Backend: new `src/anima.zig` (DiT + LLMAdapter + RFlow sampler + Qwen3-0.6B TE glue) and
  `src/qwen_image_vae.zig` (decode + encode). T5 unigram tok: extend tokenizer.zig or new small module.
- gen.zig: new `ImageBackend` arm (image_engine currently flux/krea/mageflow). load + generate + residency
  (stagedPeakBytes plan: DiT bf16 + TE + VAE + adapter; declare its own plan) + LoRA (parseLoraFields, stacked).
  img2img/edit via VAE encode + denoise strength; OpenAI edits multipart already plumbed.
- App: ImageGenView surfaces via /v1/models automatically once classified; verify model card + Use button.
- Tests (mandatory TDD): parity oracles dumped fp32-on-CPU from real diffusers/ComfyUI where possible —
  RoPE3D, adapter, one DiT block, final layer, VAE decode, RF step, T5 tokenization vs HF. Classification
  guards (tests/test_model_rescan.sh, modalityFromType test). Integration tests/test_unified_gen.sh + anima script.
  Residency test. Env-gated cos/rms oracles like tests/dump_*_fixtures.py.
- Docs/licensing: CLAUDE.md Layout row + rules; docs/reference.md media section; NOTICE port entry
  (Cosmos Apache-2.0, ComfyUI GPL-ish — port math not code; Qwen-Image VAE). CHANGELOG via /release.

## Open items to verify against a reference dump (blocked on disk space here)
- ~~Exact shift-3.0 application + default scheduler for the RF sigma schedule.~~ RESOLVED from
  source (no live dump needed) — see the Sampler section above; Anima uses `ModelType.FLOW`
  (CONST + ModelSamplingDiscreteFlow), not FLOW_COSMOS.
- ~~CFG order (before/after calculate_denoised) and whether turbo (CFG 1) skips uncond.~~ RESOLVED
  from source: CFG combines in denoised/x0 space in comfy, algebraically == combining raw
  velocities for CONST math; cfg==1.0 skips the uncond forward (comfy `disable_cfg1_optimization`).
- Qwen3-0.6B exact config (rope theta, kv heads, norm eps) from the TE safetensors header. DONE —
  confirmed via te_ref.py + parity test: theta 1e6, 16 q / 8 kv heads, head_dim 128, eps 1e-6.
- ~~T5 tokenizer: does the pad/eos handling append eos? weights semantics.~~ RESOLVED — see item 8.
- Qwen-Image VAE decoder graph + any tiling; single-frame image path. DONE — ported + parity-exact.
- padding_mask channel value (zeros) and its effect; unpatch crop offsets.

## Build order (bottom-up, each with a failing test first)
1. Classification/discovery — hermetic. ✅ DONE (green).
2. Numeric core (RFlow math, RoPE3D geom, Wan21 norm, timestep) — ✅ DONE (8 hermetic tests).
3. LLMAdapter (src/anima.zig `Adapter`) — ✅ DONE, parity cosine 1.000000.
4. Cosmos MiniTrainDIT (src/anima.zig `Dit`) — ✅ DONE, parity cosine 1.000000.
5. Qwen-Image VAE decode (src/anima.zig `Vae`) — ✅ DONE, parity cosine 1.000000 (all stages).
6. Qwen3-0.6B TE (src/anima.zig `TextEncoder`) — ✅ DONE, parity cosine 1.000000.
7. RFlow sampler schedule (shift 3.0) + CFG loop — ✅ DONE, pinned from source (`buildSimpleSchedule`,
   `calculateDenoised`, `noiseScaling`, `cfgCombine` in anima.zig), hermetic tests green. No Euler
   *loop* wired into a `generate()` entry point yet — that lands with the gen.zig engine (item 9).
8. T5 unigram tokenizer (+ HF fixture) for the adapter's t5xxl_ids — ✅ DONE (`src/t5_tokenizer.zig`,
   new module; not folded into tokenizer.zig's TokenizerType enum since it's Anima-only and every
   switch there is exhaustive). Parity-exact (10/10 prompts) vs the real `t5_tokenizer/tokenizer.json`
   read through the HF `tokenizers` lib (not `transformers`, which has a broken `regex` version
   check in this env) — see tests/dump_anima_t5_fixtures.py. Documented gap: the SentencePiece
   `Precompiled` normalizer (charsmap NFKC-ish) is not implemented; ASCII text is unaffected
   (verified), exotic composed Unicode may diverge slightly. Prompt-weighting syntax unsupported
   (weights always 1.0, matching the model's own default).
9. gen.zig Engine wiring (load DiT+adapter+TE+VAE; generate loop; PNG) — ✅ DONE, LIVE END-TO-END
   VALIDATED (not just component parity): converted a real Turbo v1.1 pack with
   `scripts/convert_anima_weights.py`, booted the actual server, POSTed to
   `/v1/images/generations`, got a coherent PNG back in ~4-5s at 512x512/6-10 steps
   for two different prompts/seeds ("a red apple on a wooden table", "a blue
   butterfly on a sunflower") — both correct subject, composition, and style.
   No LoRA (Anima has none trained), no img2img/edit (no VAE encoder ported).
   Residency/gate estimator NOT written (Anima has no `stagedPeakBytes` plan;
   uses whatever generic fallback the admission gate applies) — see the new
   "Known follow-ups" section below, this is a real gap surfaced by testing.
10. App surfacing + docs (CLAUDE.md Layout row + rules; docs/reference.md) + NOTICE + CHANGELOG.

## Validated parity harnesses (env-gated Zig tests + tests/dump_anima_*.py)
- Adapter: ANIMA_DIT + ANIMA_ADAPTER_FIXTURE   (tests/dump_anima_fixtures.py)
- DiT:     ANIMA_DIT + ANIMA_DIT_FIXTURE        (tests/dump_anima_dit_fixtures.py; uses comfy_kitchen)
- VAE:     ANIMA_VAE + ANIMA_VAE_FIXTURE        (tests/dump_anima_vae_fixtures.py)
- TE:      ANIMA_TE  + ANIMA_TE_FIXTURE         (tests/dump_anima_te_fixtures.py)
- T5 tok:  ANIMA_T5_TOKENIZER + ANIMA_T5_FIXTURE (tests/dump_anima_t5_fixtures.py; needs only the
  `tokenizers` pip package, not `transformers` — 10/10 prompts exact, incl. multi-space/punct/empty)
Reference env: uv venv + torch(cpu)+safetensors+einops+comfy_kitchen (+diffusers optional).
NOTE the adapter/DiT parity tests load the full 3.9 GB DiT file → need ~6 GB free RAM
(they OOM under memory pressure; VAE needs only ~250 MB). Code is correct regardless.

## Known follow-ups (surfaced by live testing, not yet done)

- **Max resolution was initially borrowed from Krea (2048) — corrected to 1920.**
  `comfy/model_detection.py`'s cosmos_predict2/anima branch hardcodes
  `dit_config["max_img_h"] = dit_config["max_img_w"] = 240` for every
  checkpoint (not read from the state dict). `predict2.py build_pos_embed`
  divides that by `patch_spatial` (2) to get the RoPE table's token-grid
  length, and it's passed as `len_h`/`len_w` to `VideoRopePosition3DEmb` —
  so 240 is in LATENT pixels (pre-patchify), giving 240 * 8 (the Qwen-Image
  VAE's downsample factor) = **1920 pixels** as the resolution the DiT's 3D
  RoPE NTK extrapolation (`h/w_extrapolation_ratio 4.0`) was calibrated
  against — not Krea's unrelated 2048. Checked `position_embedding.py`
  directly: the RoPE forward actually computes its position range from the
  REAL input shape (`torch.arange(max(H,W,T))`), never from this stored
  length, so `len_h`/`len_w` are effectively dead for the "rope3d" class —
  meaning there's no hard architectural crash past 1920 either, only
  unverified extrapolation quality. `gen.clampAnimaDim` now floors/ceilings
  independently of Krea's `clampKreaDim` (still 16-multiple, [256,1920]).
- **No residency/memory estimator.** `gen.zig`'s admission gate has no
  `stagedPeakBytes`-style plan for Anima, so it falls back to a generic
  estimate that does NOT account for `cfg != 1.0` running the DiT TWICE per
  step (cond + uncond forward). Forcing `recommended_cfg` to 3.0 on the
  Turbo pack (same weights, just to exercise the dual-forward branch) crashed
  the server outright on this machine's already-tight free RAM — consistent
  with "Metal OOM is UNCATCHABLE" (CLAUDE.md). `cfg == 1.0` (Turbo, the only
  live-tested config) is single-forward and has run cleanly multiple times.
  Before shipping a base/aesthetic pack (which needs cfg ~4.5, i.e. the
  dual-forward path, by default): give Anima its own plan in `gen.zig`
  (`gateEstimateBytes`/`stagedPeakBytes`) that doubles the DiT activation
  term when `recommended_cfg != 1.0`.
- **bf16 not validated.** The whole pipeline runs at float32 (see the module
  doc on `Engine` in `anima.zig`) — 2x memory/compute vs the checkpoint's
  native bf16, traded for certainty on a brand-new backend. A bf16 A/B needs
  its own parity pass before flipping the default.
- **img2img and LoRA landed since the bullet above was written.** The
  Qwen-Image VAE encoder is ported (`Vae.encode`/`hasEncoder`), img2img mixes
  `x=(1-t)·z0+t·noise` at the request's `strength` like krea/flux, and
  runtime LoRA attaches to the DiT's per-block attn/MLP linears
  (`attachLora`/`detachLora`, `lora.Arch.generic` — no alias table, since no
  established community Anima LoRA convention exists yet). No instruction-
  edit (in-context reference) training. See docs/reference.md for the full
  writeup and live-validation notes (SDEdit strength gradient, zero-B LoRA
  byte-transparency).
- **`meta.architecture` in `/v1/models` shows `"flux2"` for an Anima pack** —
  NOT anima-specific: `gen.zig`'s image-modality stub config hardcodes
  `model_type = "flux2"` regardless of backend (there's an existing test,
  `gen.zig:4712`, pinning exactly this for the stub path), so Krea and
  MageFlow packs show the same wrong string. Pre-existing, out of scope here.
  Doesn't affect `capabilities: ["image"]` (correct) or generation itself.
- **Only Turbo v1.1 has been live-tested** (the base/aesthetic DiT weights
  were never downloaded in this environment — disk space was tight enough
  that only one ~4GB variant was fetched). The conversion script
  (`scripts/convert_anima_weights.py`) and the `Engine`/config.json format
  are architecture-generic across variants (same DiT/adapter/TE/VAE, only
  `recommended_steps`/`recommended_cfg` differ per pack) — untested claim,
  not a proven one, until a base/aesthetic pack is actually run.

## Gotchas found while porting (mlx)
- mlx matmul/conv MISCOMPUTE on strided/lazy-view inputs → materialize (contig) before feeding.
  Bit us on the VAE input (transpose→strided) and on slice-born conv weights.
- mlx fast-SDPA has a head-dim ≤ 256 wall AND its "causal" mask mode aborted on the TE's
  shapes here → manual softmax(qkᵀ/√d + causal_mask)·v (VAE hd up to 384; TE causal).
- WAN RMS_norm (F.normalize over channels ×√C) == mlx_fast_rms_norm over the channel axis;
  gamma ships as [C,1,1,1] → flatten to [C] (loadVaeGamma).
- DiT self-attn fused `rms_rope_split_half` == RMSNorm(q/k) then rotate-half rope with cos/sin
  taken from the 2×2 rotation matrices (cos=em[…,0,0], sin=em[…,1,0]); verified == comfy_kitchen.
- A test reader (evalToF32) must materialize before a linear data read, or a transposed
  output scrambles (this masked the VAE as "0.68" when it was actually exact).
- **`mlx_fast_scaled_dot_product_attention` fails at REAL request sizes even with
  a genuine GPU stream**: `"force_fused=True but no fused kernel is available: the
  fused kernels require a GPU (Metal) stream"`, thrown from the adapter's M=512
  self-attn and the DiT's L~1000+ self-attn on an actual live request — the tiny
  synthetic parity fixtures (L=7, M=5) never exercised these sizes, so the bug was
  invisible until an end-to-end HTTP generation was attempted. Fourth instance of
  the same class as the VAE head-dim wall and TE causal-abort: `sdpa()` now ALWAYS
  does manual `softmax(qkᵀ·scale)·v` (no fast-kernel path left in this file at
  all). Re-verified parity cosine 1.000000 for both Adapter and DiT after the
  change. Lesson: an isolated-component parity fixture proves the MATH, not that
  the fast kernel survives production shapes — needed a live request to catch this.
```
