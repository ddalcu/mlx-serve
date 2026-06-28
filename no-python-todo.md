# No-Python Media Generation — Handoff TODO

Goal: native Zig+mlx-c audio/image/video generation in mlx-serve, replacing the
Python venv (mflux / ltx-2-mlx / mlx-audio). Branch: **`feature/Any2Any`** (NOT main).
All work is **local commits only — never push** (user rule).

## TL;DR status

| Modality | Engine | HTTP API | macOS app | State |
|---|---|---|---|---|
| **Audio** (Qwen3-TTS) | ✅ bit-exact | ✅ `/v1/audio/speech` | ✅ native server | **DONE** |
| **Image** (FLUX.2-klein-4B) | ✅ corr 0.9995 | ✅ `/v1/images/generations` | ✅ native server | **DONE** |
| **Video** (LTX-Video 2.3) | ✅ corr ~1.0 | ✅ `/v1/video/generations` | 🟡 service TODO | **DONE (server)** |

`zig build test` → ~539/559 pass (rest are env-gated live tests). 23 commits.
Plans + checklists: `docs/native-mediagen/{01-audio-tts,02-image-flux,03-video-ltx}.md`.

**ALL THREE modalities now generate natively, no Python.** Video text-to-video is
live end-to-end: `POST /v1/video/generations` on a Zig+mlx-c server produced a
coherent 9-frame 256×384 mp4 from "a red fox running through a snowy forest" (4 steps,
cfg-only). Every stage is validated against the `ltx_core_mlx` reference by correlation:
- Gemma 49-layer capture 0.99999 · connector project 0.999994 / transform 0.9999
- **DiT** (the big piece): `ditNormedSA` 0.999995 · block-0 forward out_v 1.0 / out_a 0.999998
  · full 48-block `ditForward` velocity 0.999998 · rope/schedule/positions exact
- **Sampler** (CFG-only Euler): final latent 0.999939 · `encodeTextLtx` 0.9999/0.9966
- **Frames** (unpatchify + VAE + uint8): 78.6 dB vs reference, 99.9% exact

Remaining: the macOS app `VideoGenService.swift` (mirror `ImageGenService`/`AudioGenService`)
+ AVFoundation mp4 mux from the returned RGB frames. STG/modality guidance, the audio
branch (audio VAE + vocoder + BWE), and multi-resolution are deliberate slice-1 omissions.

---

## DONE — Audio (don't touch unless extending)
- `src/tts.zig`: `Synthesizer` (talker Qwen3 + 5-layer code predictor + codec decoder).
  Talker frame-0 codes BIT-EXACT vs Python greedy oracle; codec decoder corr 1.0 / rms 0.0.
  Temperature sampling → correct, naturally-terminating 3.92s utterance.
- `src/wav.zig`: WAV encoder. `src/tokenizer.zig`: `loadTokenizerSlow`/`loadTokenizerAny`
  (Qwen3-TTS ships vocab.json+merges.txt, not tokenizer.json).
- `src/tts_server.zig`: `runTtsServe` → `/v1/audio/speech`. `main.zig` `isTtsModel` peek.
- App: `NativeGenServer.swift` + `AudioGenService.swift` POST to the native server.
- Test: `tests/test_tts.sh` (curl→WAV).

## DONE — Image (don't touch unless extending)
- `src/flux.zig` (~1500 lines): Qwen3 text encoder (layers 9/18/27 capture), Flux2 DiT
  (5 double + 20 single, shared modulation, 4-axis RoPE, parallel attn), VAE decoder.
  End-to-end corr 0.9995 / PSNR 37.9 dB. **Fixed 1024×1024** (latent grid hardcoded).
- `src/png.zig` + `lib/stb_image_write.h`: PNG encoder. `src/flux_server.zig`:
  `runFluxServe` → `/v1/images/generations`. `main.zig` `isFluxModel` peek.
- App: `ImageGenService.swift` POST to native server. Test: `tests/test_image_gen.sh`.
- **Gotchas (worth CLAUDE.md notes):** (1) `Flux2Modulation` applies `silu(temb)` BEFORE
  its linear — skip it → DiT corr 0.29. (2) `mlx_contiguous` before conv AND before
  `mlx_array_data_*` on a lazy transpose (else source-memory-order garbage). (3) Qwen
  hiddens have |x|≈15000 outliers — validate by CORRELATION, not rmse.
- **Fast-follow (optional):** FLUX.1-schnell needs T5-XXL + CLIP-L encoders + MMDiT
  (19 double + 38 single) + FLUX.1 VAE — see `docs/native-mediagen/02-image-flux.md`.
  Multi-resolution image: unhardcode the DiT latent grid in flux.zig.

---

## VIDEO — what's DONE (all in `src/ltx_video.zig`, all validated)

| Component | Function(s) | Validation |
|---|---|---|
| q4/bf16 single-component loader | `loadComponent`, `Component.{get,isQuantized}` | loads ok |
| 3D conv (causal) | `conv3d`, `decoderConv3d` | corr 1.0 vs conv_in |
| **Full 3D VAE decoder** | `vaeDecode` (+ resBlock3d, pixelShuffle3d, unpatchifySpatial, pixelNormFast) | **corr 1.000000, mse 0.0** |
| Connector projection | `connectorProject`, `projOne` | corr 0.999994 |
| Connector transformers | `connectorTransform` (+ linBias, rmsAF, rmsW, geluApprox, applyRopeSplit, connectorRope) | 0.99991 / 0.99968 |
| **Gemma 49-layer capture** | `gemmaCapture` (+ gQLin, gRms, gLayer, clipResidual) | layers 0/1/24/48: 0.999999…0.999963 |
| DiT timestep + AdaLayerNormSingle | `ditTimestepSinusoid`, `ditAdaLNSingle` (→ `AdaLNOut`) | params + embedded corr 1.0 |

**The entire text→DiT-conditioning path (Gemma → connector) and the 3D-VAE decode are
native + validated.** Reusable helpers the DiT needs are ALL present and validated:
- `gQLin(weights, …)` — q4 affine matmul: `mlx_quantized_matmul(x, wq, sc, bi, transpose=true, gs=64, bits=4, "affine")`. **Need a `Component`-based variant** (`dQLin(comp, …)`) — trivial: mirror gQLin but read `comp.get("<base>.{weight,scales,biases}")`.
- `applyRopeSplit(x, cos, sin)` + `connectorRope(…)` — **the DiT uses the SAME split-RoPE**.
- `ditAdaLNSingle` → the modulation params every block + the head consume.
- `loadComponent("…/transformer-dev.safetensors")` for the q4 DiT.

---

## VIDEO — what's REMAINING (the next agent's job)

### 1. The 48-block DiT attention forward — `BasicAVTransformerBlock` (the last big piece)

Source: `~/.mlx-serve/venv/lib/python3.14/site-packages/ltx_core_mlx/model/transformer/{model.py,transformer.py,attention.py}`. Config in `model.py` `LTXModelConfig`: num_layers 48, video_dim 4096 (32 heads × 128), audio_dim 2048 (32 × 64), av_cross 32 × 64, patch_channels 128, ff_mult 4 (gelu-approx), timestep_embedding_dim 256, t_scale 1000, norm_eps 1e-6, affine-free norms, rms QK-norm.

**`LTXModel.__call__` (model.py:224) order** (reproduce in a `ditForward` in ltx_video.zig):
1. `patchify_proj` video (B,Nv,128)→(B,Nv,4096); `audio_patchify_proj` (B,Na,128)→(B,Na,2048). (bf16)
2. Timestep embeds → MANY adaLN param sets via `ditAdaLNSingle` on the right tables (already built): `video_adaln_emb` (adaln_single, 9p), `av_ca_video_emb` (av_ca_video_scale_shift, 4p), `video_prompt_emb` (prompt_adaln_single, 2p), `av_ca_a2v_gate_emb` (av_ca_a2v_gate, 1p) — and the audio analogues. **AV-cross gate uses a DIFFERENT timestep scale**: `t_emb_av_gate = get_timestep_embedding(sigma * av_ca_timestep_scale_multiplier)` (NOT ×1000). See model.py:300-306.
3. RoPE freqs: `_compute_rope_freqs(video_positions, 32 heads, 128 hd)`; audio analogous with `audio_positional_embedding_max_pos`; cross-modal RoPE uses **temporal axis only** (`positions[:,:,0:1]`), av_cross dim, `max_pos=[max(video_max_pos[0], audio_max_pos[0])]`. Reuse `applyRopeSplit`; port `_compute_rope_freqs` (log-spaced fractional grid — `rope.py`).
4. **48 × block** (each returns new video_hidden, audio_hidden). Then the head.
5. Head: slice off text tokens; `norm_out` = affine-free LayerNorm modulated by `(scale_shift_table[2,dim] + embedded_ts)`; `proj_out` {4096,2048}→128; unpatchify. **Velocity→x0: `x0 = x_t − sigma·v` in float32.**

**`BasicAVTransformerBlock.__call__` (transformer.py:207) — 8 sublayers IN ORDER:**
video self-attn → audio self-attn → video→text x-attn → audio→text x-attn → a2v cross-attn → v2a cross-attn → video FFN → audio FFN. Each: `normed = rms(x, weight=None)*(1+scale)+shift; x += sublayer(normed)*gate`.

**adaLN tables (per block) + UNPACK ORDERINGS (the traps — transformer.py:142-151, 263-274):**
- `scale_shift_table [9, video_dim]` → `_unpack_adaln(video_adaln_params, table, 9)` →
  `(shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff, shift_ca, scale_ca, gate_ca)` — **shift,scale,gate order**.
- `audio_scale_shift_table [9, audio_dim]` → same for audio.
- `prompt_scale_shift_table [2, video_dim]` → (shift, scale) for video→text x-attn (no gate). `audio_prompt_scale_shift_table [2, audio_dim]`.
- `scale_shift_table_a2v_ca_video [5, video_dim]` → **SCALE-FIRST**: `[0]=scale_a2v, [1]=shift_a2v, [2]=scale_v2a, [3]=shift_v2a, [4]=gate`. `scale_shift_table_a2v_ca_audio [5, audio_dim]`.
- `_unpack_adaln(params, table, n)` = chunk `(table[None] + params)` into n along the param axis. (params come from the global `*_adaln_emb`; the per-block table is added.)
- Tappable isolated helper: `compute_video_normed_sa(video_hidden, video_adaln_params)` (transformer.py:182) = the modulated normed SA input — **validate THIS first** to nail the table-ordering + modulation.

**Attention (`attention.py`, `apply_gated_attention=True`):** q/k/v proj (q4 → `dQLin`); reshape to heads; **affine QK-norm** `RMSNorm(inner_dim)` on q,k (q_norm/k_norm.weight) BEFORE head split; self-attn applies RoPE, **text x-attn does NOT** (use_rope=False), AV-cross uses temporal RoPE; `mlx_fast_scaled_dot_product_attention(scale=head_dim**-0.5, mask=null)`; **per-head gating: `out *= 2*sigmoid(to_gate_logits(normed_x))`**; single `to_out`. FFN: `Linear(dim→4·dim)→gelu_approx→Linear(4·dim→dim)`.

Per-block keys: `transformer.transformer_blocks.{0..47}.{attn1,attn2,audio_attn1,audio_attn2,audio_to_video_attn,video_to_audio_attn,ff,audio_ff}.{to_q,to_k,to_v,to_out,to_gate_logits,q_norm,k_norm}.{weight,scales,biases}` (attn weights q4 g64/b4; q_norm/k_norm/gate bf16). adaLN tables top-level bf16/F32. `mx.eval` every ~8 blocks (GPU watchdog). Load on CPU stream (Load has no GPU eval), then GPU.

**Oracle (block-0 first, then full):** instantiate the reference `LTXModel(config)`, load `transformer-dev.safetensors` (11 GB q4, heavy but tractable; load on CPU stream), build small synthetic inputs (patchified video [1,192,128] for 256×384×9 → latent F=2,H=8,W=12; audio; text embeds [1,1024,{4096,2048}]; positions; timestep). **Monkeypatch `model.transformer_blocks[0].__call__`** to dump its inputs (video_hidden, audio_hidden, all adaln params, text embeds, rope freqs) + its output. Validate `compute_video_normed_sa` → the SA sublayer → the full block-0 output (corr ≥0.99 — INT4+bf16 isn't bit-exact, corr is the bar like the connector/Gemma stages). Then scale to 48 blocks → match `video_velocity`.

### 2. Guided Euler sampler (small)
`ltx_pipelines_mlx/.../scheduler.py` `ltx2_schedule`=`dynamic_shift_schedule` (token-adaptive flow-matching sigmas) + `euler_step(x,x0,σ,σ_next)=x+(σ_next−σ)(x−x0)/σ`. Guided loop: up to 4 DiT forwards/step (cond/uncond[CFG]/ptb[STG]/mod[modality]) combined `pred = cond + (cfg−1)(cond−uncond) + stg(cond−ptb) + (mod−1)(cond−mod)`, variance-rescale 0.7. Video guider cfg 3 / stg 1 / mod 3; audio cfg 7. **Slice 1 can start with cfg-only (drop ptb/mod) for simplicity.** Validate denoised latent (the robust check) vs the reference.

### 3. End-to-end wiring + endpoint + app
Wire: text → `gemmaCapture` → `connectorProject`+`connectorTransform` → init noise latents (video [1,128,F,H,W] + audio; `mlx_random_normal` seed-exact) → sampler loop calling `ditForward` → `vaeDecode` → frames `(x+1)*127.5→uint8`. Then `src/ltx_server.zig` `runLtxServe` + `/v1/video/generations` (mirror `src/flux_server.zig`/`tts_server.zig`; `main.zig` `isLtxModel` peek). Output: frames → AVFoundation mp4 mux in the Swift app (drop audio decode → silent mp4; the audio VAE+vocoder+BWE is a separate large piece you can skip — see `docs/native-mediagen/03-video-ltx.md` §"audio branch SKIP"). `VideoGenService.swift` → native server. The AUDIO decode (audio_vae+vocoder+BWE) is **optional** — slice-1 is silent video.

### 4. Notes / pinned gotchas (from the forks)
- **Strided/lazy mlx arrays:** `mlx_contiguous` before conv/matmul/SDPA AND before `mlx_array_data_*` readback (lazy transpose returns source memory order). Bit me on the VAE final transpose (corr 0.40 → 1.0).
- **Safetensors load on CPU stream** (`mlx_default_cpu_stream_new`), `mlx_array_eval`, THEN run the graph on the GPU stream. The Load op has no GPU eval.
- **Validate valid-token rows only** for left-padded sequences (padded rows attend an all-masked row → garbage in both impls).
- **PixelNorm / affine-free RMS** must use fused `mlx_fast_rms_norm` (ones-weight) to match the reference's fp32-internal kernel.
- Gemma sliding-window: the reference passes ONE full causal+pad mask to every layer, so sliding vs full layers differ ONLY in RoPE base (local 1e4 / global 1e6, full when `(i+1)%6==0`).

---

## Recommended order for the next agent (each a stage-validated commit)
1. `dQLin` (Component q4 matmul) + `compute_video_normed_sa` equivalent → validate (catches adaLN-table-ordering trap cheaply).
2. Full `BasicAVTransformerBlock` block-0 forward → validate block-0 output corr ≥0.99.
3. 48-block `ditForward` + head → validate `video_velocity` corr ≥0.99.
4. Euler sampler (cfg-only first) → validate denoised latent.
5. End-to-end wiring (`gemmaCapture`→connector→sampler→`vaeDecode`→frames) → produce a short silent mp4, compare frames to the reference (PSNR).
6. `ltx_server.zig` + `/v1/video/generations` + `main.zig` peek; `VideoGenService.swift` → native server (AVFoundation mux).

## Build / test
- Zig: `zig build -Doptimize=ReleaseFast` (NEVER bare `zig build` for the binary). `zig build test` for unit/live tests.
- Live video tests are env-gated: `LTX_TEST_MODEL=<dgrauet/ltx-2.3-mlx-q4 snapshot> LTX_*=<oracle raw files> zig build test -Dtest-filter="ltx ..."`. Models: LTX q4 + gemma-3-12b-it-4bit are in `~/.cache/huggingface/hub/`.
- Swift app: `cd app && swift build -c release -Xswiftc -swift-version -Xswiftc 5` (or `SKIP_NOTARIZE=1 bash app/build.sh`).
- **Forks worked well** for this (one stage-validated component per fork). Each fork: symlink `lib/{llama,ds4,jinja_cpp}` (+libjinja.a) from the parent worktree to build, then remove to leave git clean. Validate every stage against a `.npy`/`.raw` oracle by CORRELATION before committing.
</content>
