# Model loading, configs, converters, media parity — war stories (moved out of CLAUDE.md)

Full histories: live failures, measurements, diagnosis ladders, dead ends. The distilled RULES live in the root CLAUDE.md "Rules" section — when a rule changes, update the story here too. New gotchas in this domain: add the 1-3 line rule to root, the full story here.

### A reference-image editor that honors the requested size distorts every non-square edit (MageFlow Edit)
MageFlow's edit path is in-context: each reference is VAE-encoded AT THE TARGET SIZE and its latent tokens are concatenated into the DiT image stream beside the denoising target, so target and references share one (lh, lw) grid. The port took (W,H) straight from the request, which defaults to 1024×1024 — and the app ALWAYS sends an explicit size from its resolution picker, so in practice every edit of a phone photo went through a 3:2 → 1:1 bicubic squash before the model ever saw it. Nothing errors; the output is a competent edit of a distorted picture, which reads as "the model is bad at faces" rather than as a preprocessing bug. The reference pipeline never has this problem because it resolves the target from the primary reference (`resolve_target_size(refs[0], …)`, /16 floor) instead of from a request field. Fix (`gen.fitAspect`, applied on the `editUsesRawBytes` path only): the primary reference's ASPECT wins, the requested size is reinterpreted as the pixel BUDGET it's fitted to, result rounded to /16 and logged (`edit: target 1024x1024 -> 1360x768 (primary reference is 1536x864)`). The FLUX edit path is untouched — its references keep their own aspect and its output grid is independent, which is the same principle reached a different way. Two robustness guards landed with it, both in the same "two independent paths must agree" shape: `encodeEdit` asserts the `<|image_pad|>` count equals the merged vision-feature rows (prompt templating and the ViT grid are computed separately; a mismatch dies inside `mlx_put_along_axis`, which is an uncatchable server kill, not an error), and `buildEditPromptIds` caps the templated prompt at `EDIT_DROP_TOKENS + TE_MAX_COND` exactly like the txt2img path — the LM's causal mask is materialized DENSE on the host, so a client prompt of 120k tokens asked for a ~57 GB allocation before a single matmul (live: that request now returns a normal image in 7 s).

### MageFlow's per-generation memory leak lives in engine-mlx.md
MageFlow leaked ~2.2 GB per megapixel per generation and survived unload — the cause was an mlx ownership idiom (a `mlx_slice` handle wrapped in `contiguous` and never freed, which pins the parent buffer), not anything model- or precision-specific. Full story, diagnosis ladder and guards: `docs/gotchas/engine-mlx.md`, "A slice handle wrapped in `contiguous` and never freed pins its PARENT's buffer".

### A distilled few-step diffusion model can REQUIRE bf16 — f32 washes the output (MageFlow Turbo)
Porting Microsoft Mage-Flow Turbo (`src/mage_flow.zig`), every component passed bit-parity at cosine 1.0 in f32 (VAE decode, DiT forward masked+unmasked, Qwen3-VL text encoder), the scheduler/Euler/pack glue was hermetically tested, the tokenizer produced byte-identical HF ids, and a step-by-step e2e replay against the f32 reference matched cosine 1.0 at every step — yet the first live 4-step generation was a grey blob with a faint correctly-colored fox (conditioning right, denoising incomplete), and MORE steps made it NOISIER (grid artifacts at VAE-patch scale), not sharper. The reference at the identical seed/size/steps produced a crisp photorealistic fox. The whole f32 pipeline was the bug: **`ModelConfig.precision = bf16`, and the distilled 4-step model washes in f32**. Isolation ladder (each an inline-decoded reference image): watermark-vs-plain noise → identical N(0,1) stats, both wash in f32; f32-model + bf16-rounded freqs → still washes; **f32 → washed, bf16 → crisp at the same 256px**; bf16-DiT + f32-VAE → crisp (only the DiT/encoder need bf16, the VAE upcasts its latent internally and stays on its f32 parity path). Converting the DiT to bf16 exposed a SECOND layer: my bf16 DiT still washed across all seeds while the reference bf16 was crisp. A per-stage tap comparison (img_in/txt_in/temb/rope/block-0/v0 all cosine ≥0.9999) proved the forward was faithful, but the e2e loop diverged from cosine 0.9999 at step 1 to 0.55 at step 4 — the divergence started at step 2, the first step consuming a NON-integer sigma. Root cause: the reference casts the timestep to the model dtype (`timesteps.astype(img.dtype)`) so the model sees a **bf16-rounded** sigma; I fed the full-precision f32 sigma. A distilled few-step model memorizes the exact bf16 timestep→denoising mapping, so an f32 timestep is out-of-distribution → wash. Fix (`roundBf16` in `timeTextEmbed`): round both the timestep value and its sinusoidal frequency table to bf16 on the bf16 path (f32 path leaves them full). Also required for the bf16 port: RoPE rotates in f32 then casts to the compute dtype (`applyRope` — a bf16 rotation drifts over 12 blocks × N steps), and mlx promotes `bf16 ⊕ f32-scalar → f32` so every scalar constant goes through `scalarLike(dtype)` and the param-free LayerNorm returns its input's dtype (else the modulation chain silently upcasts to f32). Class lessons: (1) component bit-parity in ONE precision does NOT prove the multi-step loop — a tiny per-step bias compounds; debug the LOOP by dumping the reference in its NATIVE dtype and comparing per-step latents, and DECODE a suspect latent to tell "washed" from "different-but-valid" (cosine alone can't). (2) f32 is not universally safer than the training precision. Parity workflow: `tests/dump_mageflow_{te,e2e}_fixture.py` (e2e dumps native bf16 for the loop test; TE/DiT/VAE component fixtures cast f32), Zig oracles env-gated on `MAGEFLOW_{TE,E2E_BF16,DIT,DIT_MASKED,VAE}_FIXTURE`.

### DiffusionGemma numerics: garbage canvases amplify chaos; parity-test on converged canvases
Logit/argmax comparisons against the reference on RANDOM-token canvases are meaningless for correctness: garbage input puts the sigma-MoE router on knife-edge ties, and a single bf16 kernel-order difference flips expert sets, ballooning into multi-unit logit deltas while both outputs are "valid". The meaningful invariant is SELF-CONSISTENCY on a converged canvas (real prompt + the canvas the reference committed → one decoder forward must reproduce it as argmax; reference 63/64 no-sc / 64/64 with-sc). Related: sliding-layer prefill now takes the `"causal"` fast path whenever `total_kv <= sliding_window` (same kernel + reduction order the mlx-lm/mlx-vlm reference picks) — the `"array"`-mask kernel's different reduction order alone degraded diffusion convergence via router-tie flips. Debug aids: `MLX_SERVE_DIFFUSION_TRACE=1` prints per-step mean entropy + acceptance counts and decoder layer-0/self-conditioning tensor heads.

### Per-token AdaLN (LTX I2V): the t2v path must stay byte-identical; tripwire is uniform-mask equivalence
LTX image-to-video conditions on a reference image two ways (`src/ltx_video.zig`): (1) VAE-encode the image (`vaeEncode`, the mirror of `vaeDecode`) → clean tokens pinned into latent frame 0 via `applyDenoiseMask` each Euler step; (2) **per-token timestep AdaLN** — the DiT is told frame-0 tokens are at `t=0` (clean) while the rest are at `t=sigma`. The native DiT was scalar-AdaLN (one param set broadcast over all tokens); the per-token path runs the VIDEO stream's 9-param self/ff/text-ca AdaLN, the 4-param AV-cross-video AdaLN, the output-head embedded timestep, AND the x0 sigma subtraction PER-TOKEN. Audio, prompt, and AV gates ALWAYS stay scalar (image-only conditioning never sets `audio_timesteps`). This is a CLASS: an optional per-token modulation path bolted onto a scalar one. Two rules: (a) gate the whole thing behind an optional `cond_mask`/`nv_pt` so `nv_pt==0` hits the EXACT, unchanged scalar code (`adalnUnpack`/`adalnRowN`/`outputBlock` all dispatch `nv_pt==0 → scalar helper`), giving zero t2v overhead and zero regression; (b) the regression tripwire is **uniform-mask equivalence** — with `cond_mask` all-ones every token is at the same timestep, so per-token MUST reproduce scalar (the test `per-token AdaLN uniform-mask equals scalar` runs `ditForward` both ways, cos > 0.9999 on video AND audio — verified 1.0/1.0). An equality test against the reference can't catch a scalar-path regression; the uniform-mask self-equivalence test can. Encoder parity (`vaeEncode`, cos 1.0) is decoupled from image preprocessing by feeding the fixture's dumped PIXELS (not a file), so CRF/resize choices never confound the encoder oracle. Live e2e (`tests/test_video_gen.sh` I2V case): a high-contrast split image pinned as frame 0 reconstructs near-exactly (input 20/235 → decoded 20.2/232.8). Fixtures: `tests/dump_ltx_vae_encoder_fixtures.py`. The encoder (`vae_encoder.safetensors`, ~0.6 GB) is a SEPARATE download from the decoder-only q4 bundle — allowlisted in `MediaBundle.swift` but not a ready marker; missing → graceful t2v fallback (logged), mirroring `ref_audio`.

### Anything reading a model's config must read `text_config` FIRST, then the root (nested-config class)
A multimodal checkpoint (every Gemma 3/4, every Qwen-VL, i.e. MOST of them) puts **every text dim under `text_config`** and leaves only `model_type` / `vision_config` / `quantization` at the root. Two parsers have now shipped this bug, in opposite directions — treat any new config reader as guilty until it handles the nesting:
- **`model_discovery.parseStubMeta` (the `/v1/models` stub) read the ROOT ONLY**, so every UNLOADED multimodal model advertised `hidden_size: 0, num_layers: 0, context_length: 0, is_moe: false`. The `is_moe` lie is the dangerous one: gemma-4-26B-A4B (128 experts, declared in `text_config`) and Ornith-1.0-35B (256) both reported dense, so a client could not tell a MoE from a dense model without cold-loading 16 GB of weights — and MoE-ness is exactly what gates MTP/drafter defaults. Fixed 2026-07-14: read the nested block first, fall back to the root PER FIELD (a minimal `text_config` may omit a field the root still carries). Symptom signature: a `/v1/models` entry with `loaded: false` whose dims are all `0` while `quantization` (root-level) is correct. Guarded by the multimodal + nested-MoE + root-fallback cases in the `parseStubMeta` test (`model_discovery.zig`).
- **`model.zig parseConfigFromJson` reads `text_config` but OMITTED fields silently keep struct defaults** — see below.

### Config fields omitted by nested `text_config` fall back to 12b-shaped struct defaults
`ModelConfig`'s field defaults (`src/model.zig`) are the Gemma-3/4 **12b/27b** shape — `num_attention_heads=16`, `num_key_value_heads=8`, `hidden_size=3840`, 48 layers. A multimodal checkpoint's nested `text_config` is read at `parseConfigFromJson` (lines 458-460), but **omitted** fields silently keep those struct defaults instead of the model family's HF defaults. `gemma-3-4b-it-4bit`'s `text_config` omits `num_attention_heads`/`num_key_value_heads`/`head_dim` and leans on HF Gemma3TextConfig defaults (8/4/256) — so it loaded with 16/8, and the Q projection (`8*256=2048`) crashed reshaping into `(1,1,16,256)` at warmup (issue #43). Symptom signature: **`MLX error: [reshape] Cannot reshape array of size N into shape (1,1,H,D)` during warmup, where N = (real heads)×head_dim and H = the wrong (default) head count.** Rule: when an arm relies on HF per-arch defaults for fields a minimal `text_config` may omit, fill them explicitly in that arm gated on `cfg_obj.get(field) == null` (see the gemma3 arm). Guarded by the two `ModelConfig … gemma3 … head counts` tests in `model.zig` (omitted → HF defaults; explicit → preserved).

### Text-only `*_text` model_types + flat weight prefix (Gemma3ForCausalLM)
The arm dispatch in `model.zig` keys on the **top-level** `model_type`, and each multimodal family also ships a flat text-only checkpoint with a sibling tag the arm must accept: `gemma3` ↔ `gemma3_text` (Gemma3ForCausalLM, e.g. the abliterated `-lm-` builds), mirroring the existing `gemma4_text`, `qwen3_5_text`, `qwen3_moe_text` pairs. A flat text-only checkpoint has **no `vision_config`/`text_config`**, so its weights live under `model.*`, NOT the multimodal `language_model.model.*`. Two failure modes if the `_text` tag isn't routed: (1) it falls through to the llama-family else arm → `model_type=unknown`, no QK-norm/embedding-scale/pre-FF-norm → loads-but-incoherent; (2) `tie_word_embeddings` defaults false and Gemma ships no `lm_head` tensor (always tied) → **`MISSING WEIGHT: lm_head.weight` crash at load**. Rule: accept the `_text` sibling in the arm, collapse `config.model_type` onto the base tag, pick the prefix by `root.get("text_config") != null ? "language_model.model" : "model"` (mirrors the LFM2 VL split), and force `tie_word_embeddings = true` for Gemma. Also add the tag to BOTH allowlists that gate visibility — `model_discovery.supported_model_types` (Zig; else it's skipped from `/v1/models`) AND `supportedModelTypes` in `HFModels.swift` (Swift; else the model browser flags it "Unsupported architecture"). Guarded by the flat-`gemma3_text` routing + multimodal-prefix tests in `model.zig`, the discovery test in `model_discovery.zig`, and `testGemma3TextIsSupportedArchitecture` in `DownloadManagerLayoutTests.swift`.

### hy_v3 MoE expert container name varies by converter (`mlp.experts` vs `mlp.switch_mlp`)
The stacked hy_v3 experts ship under **two different tensor-container names** depending on who converted the checkpoint: ox-ox-style MLX builds use `model.layers.N.mlp.experts.{gate,up,down}_proj.*`, while **mlx-lm's converter** (every `mlx-community/Hy3-oQ2*` and `pipenetwork/Hy3-REAP*` build) names the SAME `[E, out, in]` tensors `model.layers.N.mlp.switch_mlp.{gate,up,down}_proj.*`. The `is_hy3` binding arm (`transformer.zig`) hardcoded `mlp.experts.*`, so every mlx-lm-converted hy_v3 checkpoint died at load with **`MISSING WEIGHT: model.layers.1.mlp.experts.gate_proj.weight`** during "Precomputing MoE layer weights…" (layer 1 = first MoE layer after `first_k_dense_replace`). Live 2026-07-16 (both the default `Hy3-oQ2e` and a REAP-pruned build). Fix: `hy3ExpertContainer` probes `mlp.experts.gate_proj.weight` once per layer and threads the resolved container (`mlp.experts` if present, else `mlp.switch_mlp`) through the 9 expert fetches via `moeExpertSuffix` — same layout, so an ox-ox checkpoint is byte-identical (probe prefers `experts`). Router (`mlp.router.gate.*`, quantized on REAP / bf16 on oQ2e — optional scales handle both), shared expert (`mlp.shared_mlp.*`), and `mlp.expert_bias` names are unchanged across converters. Rule: never hardcode ONE MoE-expert container name for an arch whose checkpoints come from multiple converters — resolve it by probe. Guarded by the `hy3ExpertContainer resolves both mlx-lm switch_mlp and ox-ox experts naming` test (transformer.zig); the ox-ox path stays the default-preferred branch. NOTE: this is a load-path naming alias only — mlx-lm's `switch_mlp` name matches what the qwen3_5_moe arm already uses, so the forward (`moeMLP2`, sigmoid routing) is name-agnostic once bound.

### Gemma `<end_of_turn>` (106) leaks when config declares a scalar eos
Gemma's chat template always ends turns with `<end_of_turn>` (id 106), but `<end_of_turn>` is NOT the tokenizer's `eos_token` (that's `<eos>`/1) and is only a stop token if it's in `ModelConfig.eos_token_ids`. The full Gemma checkpoints ship `eos_token_id: [1, 106, 50]` (array → all added by the common parse), but minimal checkpoints (the abliterated text-only build) ship a **scalar `eos_token_id: 1`**. Gating the 106 add on `num_eos_tokens == 0` was then defeated by the scalar `1`, leaving 106 out of the stop set → generation never halts at the turn end and **leaks repeated `<end_of_turn>` into the visible content** (`finish_reason` still eventually `stop`). Same class as the Qwen2.5-Coder `<|im_end|>` leak handled by the load-path merge (`main.zig`/`scheduler.zig doLoad`). Rule: never gate a known chat-terminator on "config provided no eos"; merge it ADDITIVELY + dedup-guarded. All three Gemma arms call `config.ensureGemmaTerminators()` (adds 1 + 106 if absent); guarded by the scalar-eos parse test + the `ensureGemmaTerminators` idempotence test in `model.zig`.

### Quantization modes (nvfp4 / mxfp4 / mxfp8)
`config.json`'s `quantization.mode` lands on `ModelConfig.quant_mode` (default `affine`; unknown → `error.UnsupportedQuantMode` at parse). Non-affine modes store NO `.biases` tensors and use uint8 fp8-encoded scales — but mixed QAT checkpoints (gemma-4 `*-qat-nvfp4`) override some layers to affine 8-bit/gs64 WITH bf16 scales + biases, so resolution is PER WEIGHT (`transformer.computeQuantParams`, cached by scales ctx): uint8 scales → config's fp8 mode; float scales → affine, with (bits, group_size) solved from the activation inner dim (`w_cols*32/s_cols` alone only pins bits×gs). Two loader rules: `.biases` fetches are mandatory under affine, OPTIONAL under non-affine modes (`getLayerBias`) — never skipped, or the affine-override matmuls break; and never gate fp8-vs-affine on `biases.ctx == null` alone (that's the legacy mxfp8-inside-affine heuristic in `qmatmulBits`, kept for NVIDIA Nemotron mxfp8 layers).

### Parity fixtures for fp16-fragile giants must be dumped fp32-CPU (MPS-fp16 fixture-corruption class)
A cos oracle that fails around ~0.3 while every per-layer bisect of the port is clean means the FIXTURE is corrupt, not the engine. Live bite (2026-07-04, paint DINOv2-giant): the fixture was dumped with torch fp16 on MPS; dinov2-giant's hidden magnitudes grow to ~400 by the deep layers and the MPS fp16 run collapses to corr 0.2985 vs fp32-CPU — while the MLX fp16 engine (fp32-accumulating matmuls) hits 0.999995 against the fp32 truth. No NaN/inf in the bad fixture — it just quietly decorrelates, so nothing "fails" at dump time. Diagnostic that found it (reusable): re-run the SPEC in numpy from the CONVERTED safetensors; if numpy matches the Zig output but not the fixture, bisect fixture-vs-fp32-CPU per layer (`output_hidden_states=True`) — embeddings clean + all layers clean + final mismatch ⇒ the dump dtype/device is the bug. Rules: (1) dump reference fixtures for deep ViTs/DiTs in **fp32 on CPU** unless deliberately reproducing the reference's production dtype (SD-class UNets/VAEs are fp16-safe — the paint VAE fixture at fp16-MPS matched at 0.9999); (2) when an oracle misses badly, suspect the fixture BEFORE rewriting the port — the numpy-spec-repro takes minutes and localizes the fault to spec vs mechanics vs fixture. See `tests/dump_hunyuan3d_paint_fixtures.py` (the fp32 DINO dump carries the war-story comment).

### transformers ≥5.x zeroes custom-model rotary buffers → identity-RoPE reference (fixture-corruption class)
A SECOND way reference fixtures silently corrupt (beside the MPS-fp16 dtype one): loading a custom-code (`trust_remote_code`-style) checkpoint under transformers 5.x leaves every `Qwen3RotaryEmbedding.inv_freq` NON-PERSISTENT buffer as ZEROS — the meta-device init materializes it empty, the checkpoint doesn't contain it, and the re-init hook only covers transformers' OWN model classes. cos=1/sin=0 → the whole reference runs IDENTITY RoPE and dumps decorrelated-but-plausible fixtures. Live bite (2026-07-05, ACE-Step): the Zig DiT oracle sat at 0.66 while a no-rope variant "passed" at 0.99996 — the tell that the FIXTURE, not the port, was unrotated (my rope was already proven by the AutoModel-based text-encoder oracle, which uses transformers' own class and was unaffected). Diagnostic ladder that found it (reusable): stage-bisect fixtures (timestep/proj_in/layer0) → sub-block bisect (norm/self-attn/cross/mlp, each fed REFERENCE inputs) → a no-rope attention variant → probe `named_modules()` for all-zero `inv_freq`. Fix in `tests/dump_acestep_fixtures.py`: after `from_pretrained`, rebuild each zeroed rotary from a FRESH instance of its own class (`type(mod)(config=mod.config)`) and copy `inv_freq`/`attention_scaling` over. Rules: (1) any fixture-dump script that loads custom modeling code must assert no `*.rotary_emb.inv_freq` is all-zero; (2) when an oracle improves by REMOVING a transform, suspect the fixture's transform was silently disabled before touching the port; (3) to localize a DiT parity miss, dump torch intermediates (timestep/proj_in/per-layer, then per-sub-block fed REFERENCE inputs) and cos-compare stage-by-stage — a throwaway pair of dump script + env-gated Zig test, deleted once the bug is found (the 6 committed `ACESTEP_*` oracles are the permanent gate).

### Voice cloning (Qwen3-TTS Base) = ECAPA-TDNN speaker embedding, and a `TimeDelayNetBlock` is conv + ReLU
Zero-shot voice cloning on the `-Base` checkpoints runs the reference clip (24 kHz mono) through `mel_spectrogram` (slaney filterbank + symmetric-Hann STFT via `mlx_fft_rfft`) → an **ECAPA-TDNN speaker encoder** (`SpeakerEncoder` in `tts.zig`, weights `speaker_encoder.*` in the main safetensors, stored MLX conv layout `[out,k,in]` — load WITHOUT transpose, unlike the codec) → a `[1, t_hidden]` embedding, spliced as ONE extra position in the talker's codec prefix (mirrors `_prepare_generation_inputs`). API: `/v1/audio/speech` accepts `ref_audio` (base64 24 kHz-mono WAV); the app's `AudioGenService` sends the recorded/picked clip. With-transcript "ICL" cloning (codec encoder + Whisper) is NOT implemented — `ref_text` is ignored; the speaker-embedding path needs no transcript. There are **no predefined voices** on Base models (`spk_id: {}`); named voices are a `custom_voice`-model feature we don't ship. **GOTCHA that cost a fidelity gap:** ECAPA's `AttentiveStatisticsPooling.tdnn` is a `TimeDelayNetBlock`, which is `conv → ReLU` (every `TimeDelayNetBlock.__call__` ends in `nn.relu`). Porting it as bare `conv` (then tanh) — missing the ReLU — gave a deterministic embedding cos 0.9989 / norm +3% (clone voice-sim 0.977 vs the reference's 0.986). Adding the ReLU made the embedding BIT-EXACT vs the Python oracle (cos 1.000000) and lifted clone fidelity. Validated by `tts.zig`'s "speaker encoder (ECAPA-TDNN) matches the Python oracle" test (`TTS_TEST_MODEL`/`TTS_SPK_REF`/`TTS_SPK_EMB`; cos>0.998) + the mel sub-check (`TTS_SPK_MEL`, f32-exact). Rule: in an ECAPA/SpeechBrain port, every TDNN block carries an implicit ReLU — never drop it when fusing with a following activation.

### Embedding lookup must reshape to the table's NATIVE width, not a config-derived `hidden` (Qwen3-TTS 0.6B crash)
`tts.embed(table, ids, hidden)` gathers rows with `mlx_take_axis` (always the table's native width) then reshapes to `[1, N, width]`. It used to reshape to the PASSED `hidden`, which is only correct when the table's native width equals it. On the Qwen3-TTS **0.6B-Base** checkpoint the `text_embedding` table is 2048-wide but the talker `t_hidden` is 1024 (a `text_projection` fc1/fc2 bridges 2048→1024 right after) — so `projectText` passed `t_hidden`=1024 and the reshape of a `[N,2048]` gather into `[1,N,1024]` threw `MLX error: [reshape] Cannot reshape array of size 24576 into shape (1,12,1024)`, an UNCATCHABLE C++ exception that crashed the whole unified server. The 1.7B-Base worked only because its `text_embedding` happens to be `t_hidden`-wide (2048). Fix: `embed` now reads the width from `getShape(table)` and ignores `hidden` (the gather is always native-width, so this is correct for every caller — text/codec/cp/rvq tables). Rule: an embedding reshape's width comes from the WEIGHT, never from a sibling config dim — they diverge whenever a projection sits between the lookup and the residual stream. Guarded by the `embed uses the table's native width` hermetic test in `tts.zig` (red-on-revert: reshape size-mismatch). Audio panel only offers `qwen3_tts` models — the native engine dispatches on `model_type`, so the old gpt2-based MOSS-TTS can't load (removed from the catalog).

### A GGUF folder is a SHELF OF MODELS, not a model (per-quant identity class)
An MLX checkpoint is a directory; a GGUF repo is a directory of independently-loadable quants (`Qwen3.5-4B-Q4_K_M.gguf`, `-IQ4_NL.gguf`, `-Q8_0.gguf`, …) PLUS non-LLM sidecars. Every layer that assumed "one folder = one model" broke a different way (all fixed 2026-07-13):
- **Discovery emitted one model per folder** (`DownloadManager.makeLocalModel` picked the alphabetically-smallest `.gguf`), so the tray picker could only ever offer ONE quant of a repo you had two of — the others were invisible to the entire app. Now `makeLocalModels` (plural) returns one `LocalModel` per quant, `path` = the FILE (which is what the server already loads, so picking a quant needs zero server change), `id` carries a `#<file>` suffix (same id ⇒ SwiftUI collapses the rows), and `LocalModel.displayLabel` = `repo · QUANT` (`name` stays the repo — filters/grouping key off it). `duplicateNames` is computed on `displayLabel`, NOT `name`: macOS `.menu` Pickers key the checkmark by item TITLE, so two same-titled rows both render selected.
- **`existingModelDir` gated on `config.json`**, which a GGUF download never writes — so it returned nil for every GGUF folder, `isReady`'s GGUF fast-path *underneath that guard* was dead code, and a downloaded quant stopped resolving the moment the in-memory download row went away (i.e. after relaunch). It now accepts config.json OR a servable `.gguf`.
- **Delete/cancel were folder-scoped.** Deleting one quant (or cancelling a SECOND quant's download — `finalizeIfCancelled` wipes the whole download dir) took every sibling quant with it. `removeGgufQuant` deletes one file and only drops the folder when the last servable quant goes; `startGguf` records the in-flight filename so a cancel is scoped to it.
- **A repo never reaches a terminal "on disk" state.** Owning Q4_K_M says nothing about wanting Q8_0, so the Discover action cell stays a menu forever (`GgufQuantMenu` + pure `GgufQuantMenuModel`): ✓ on-disk quants (click = Use), the rest downloadable, per-quant Delete. The old `.onDisk` collapse to "✓ On disk" + trash left NO way back to the quant picker.
- **Not every `.gguf` is a quant — sidecars are the trap.** `mmproj-*.gguf` (CLIP) was known; a SPEECH TOKENIZER is the other (live: `qwen3-tts-tokenizer-f16.gguf`, 341 MB, beside `qwen3-tts-0.6b-f16.gguf`). Enumerating every file in a folder EXPOSES this — the old alphabetical pick dodged the tokenizer only by luck of the name. `DownloadManager.isGgufSidecar` / Zig `model_discovery.isGgufSidecarBasename` (mmproj prefix + `tokenizer` substring) are the filter, and they MUST stay in sync: the app lists every quant as selectable, so a file the client offers and the server can't load is a user-visible dead end (the server's `scanLlmGguf` picks alphabetically and would have loaded a tokenizer as the LLM in any repo whose sidecar sorts first). **Rule: adding a new `.gguf` companion convention means updating BOTH classifiers.** Guards: `GgufQuantTests` (Swift, incl. the tokenizer case) + `isGgufSidecarBasename` (Zig).

### A checkpoint declaring an affine bit-width MLX has no kernels for kills the server at warmup — reject at config parse (2026-07-20)
MLX ships affine quant Metal kernels only for bits {2,3,4,5,6,8}; `ops.cpp` rejects anything else — but only inside `quantize()`. An ALREADY-quantized checkpoint (config.json `"quantization": {"bits": 1, ...}`) never calls `quantize()`, so it loads all weights, precomputes, compiles kernels, and then dies during warmup with `MLX error: [metal::Device] Unable to load kernel affine_dequantize_bfloat16_t_gs_128_b_1` — an uncatchable process kill (same class as Metal OOM: prevention only). Live bite: `prism-ml/Bonsai-27B-mlx-1bit` (a 1-bit qwen3_5 conversion presumably made against a newer/patched runtime). With `--model` at boot the whole server died; via cold-load it would have killed it mid-traffic.
- **Fix**: `parseConfigFromJson` validates `quant_bits` against {2,3,4,5,6,8} when `quant_mode == .affine` (non-affine modes have their own bit semantics; bits 0 = dense) → `error.UnsupportedQuantBits` with a message naming the supported set. Headless cold-load now returns a clean 500 `model_load_failed` and the server keeps serving other models (verified live); boot with `--model` exits with the clear message instead of the Metal abort.
- **Defense in depth already present**: `affineParamsFromGeometry` (the per-weight solver) only ever returns bits in the supported set, so a mixed checkpoint can't sneak an unsupported width past the config guard — the config-level `quant_bits` fallback was the ONLY path to an unsupported dispatch.
- **Rule**: when bumping the mlx submodule, if upstream adds a bit-width (e.g. 1-bit), extend the switch in `parseConfigFromJson` AND the one in `affineParamsFromGeometry` together. Guard: `parseConfigFromJson rejects affine bits MLX has no kernels for` (model.zig, red pre-fix).

### MageFlow 8-bit: the two things shape alone gets wrong, and what the fixtures actually measure (2026-07-24)

Porting krea's `MixedLinear` into `src/mage_flow.zig` as `MfLinear` (dense bf16 OR affine-quantized, `(bits, group_size)` solved per tensor from packed geometry) was the mechanical half: 43 `linearT` sites, ~32 of which became `MfLinear.forward`, VAE untouched. All ten env-gated bf16 parity fixtures stayed byte-identical afterwards, which is what a `maxAbsDiff == 0.0` unit test on the dense path predicts.

**Lookup tables are invisible to a shape rule.** The converter's `should_quantize` is shape-driven (rank 2, `in % 64 == 0`, `min(out,in) >= 512`) and that is right for every real linear. It is also true of `model.language_model.embed_tokens.weight` [151936, 2560] and `model.visual.pos_embed.weight` [2304, 1024], both of which the engine reads with `mlx_take_axis`. Packing them yields a gather over uint32 words — not an error, just wrong rows. No shape test can distinguish them, so `NEVER_QUANTIZE` is an explicit list. Cost of getting it right: 0.78 GB stays bf16 (the biggest single "keep" in the manifest).

**Sizes, measured.** 17.45 GB of weights → 9.16 GB (txt2img) / 9.74 GB (edit). The DiT is 99.97% 2-D linears so it quantizes almost completely (8.23 → 4.38 GB). Pruning is worth ~1 GB on top: `pipeline.y_embedder.encoder.*` (69 MB) is the VAE's training-time encoder half and is dead in BOTH modes, while `model.visual.*` (830 MB) and `student.dconv_encoder.*` (132 MB) are loaded only when `is_edit`. A wrong prune fails loudly (`ownWeight` errors on a missing key), which is what makes aggressive pruning safe.

**A fixture cosine on an 8×8 latent is a sensitivity probe, not a quality measure.** 8-bit vs the PyTorch reference: DiT forward 0.9956 (bf16: 1.0000), e2e 4-step final 0.9902 (bf16: 0.9998), edit loop 0.9515 (bf16: 0.9942). The edit number looks alarming and is not — same seed, same prompt, the 8-bit edit is indistinguishable from bf16 (same snow placement, same identity, same rug), and multi-reference composition differs only by a slight neon tint. Judge images; the fixtures tell you whether the code path is intact.

**The AdaLN modulation projections are the biggest target AND the riskiest one.** `img_mod.1` + `txt_mod.1` are [18432, 3072] × 12 blocks = 2.72 GB, 33% of the DiT, and they sit on the timestep conditioning path this checkpoint is known to be picky about. Holding them back (`--keep-bf16 img_mod.1,txt_mod.1`) improves DiT forward 0.9956 → 0.9980 and e2e step-4 0.9902 → 0.9953, but barely moves the edit loop (0.9515 → 0.9576) and costs +1.27 GB. Not worth it at 8-bit — nothing shows in the output. Re-measure before assuming that at 4-bit. `MfLinear` reads bits per tensor precisely so this stays a converter flag, never a code change.

### A configless repo shape has to be taught to every "is this a model?" answer (2026-07-24)

MageFlow ships no root `config.json` — `model_index.json` is the only signal, and the weights live in `transformer/`, `text_encoder/`, `vae/`. `peekConfig` knew that (its `peekMageFlowIndex` fallback fires when `config.json` won't open). It was dead code: `tryAddModel` ran `statFile("config.json") catch return false` several lines EARLIER, so the fallback was never reached. Two more facets of the same miss: the byte scan only summed `*.safetensors` at the top level (so a discovered MageFlow model would report a null size), and `cli.zig` carried a third private `isModelDir` testing only config.json-or-gguf.

Net effect: every MageFlow checkpoint — including the two official ones, downloaded and working — was invisible to `mlx-serve list`, `/v1/models` and the app's picker, for as long as the backend has existed. It went unnoticed because the feature was developed and validated with `--model <path>`, which bypasses discovery entirely.

- **Fix**: `tryAddModel` falls through to `peekConfig` when either config.json OR a MageFlow index is present; `sumComponentWeights` sums one level down, but only when the flat scan found nothing (so the common MLX layout pays nothing); `peekMageFlowIndex` is now `pub` and `cli.zig` calls it instead of keeping its own answer.
- **Rule**: a new repo LAYOUT (not just a new model_type) has to be taught to every classifier that answers "is this a model?" — currently `model_discovery.tryAddModel`, `model_discovery.peekConfig`, `cli.isModelDir`, and Swift's `existingModelDir`. A layout that only `--model` can load is a layout the app cannot see.
- **Guards**: `discoverModels finds a MageFlow repo (model_index.json, no root config.json)` (asserts the size too — null was the second facet) and `cli: isModelDir accepts a MageFlow repo`, which also pins that an unrelated diffusers pipeline (`StableDiffusionPipeline`) stays hidden.

#### Second instance: an mflux conversion with no index either (2026-07-29)

`mlx-community/flux2-klein-9b-4bit` is the only MLX build of FLUX.2-klein 9B, and its layout is the 4B's minus one file: `transformer/`, `text_encoder/`, `vae/`, `tokenizer/`, no root `config.json`, and no `model_index.json` to fall back to. All three classifiers above had to learn a shape for the second time.

- **Signal**: the DiT's own weight name, `double_stream_modulation_img`, not a directory shape. `transformer/` + `vae/` + `text_encoder/` describes most of diffusers, so a shape test adopts every future MLX conversion of anything; a weight name IS the architecture. `peekMfluxFlux2` reads the shard index when there is one and the first shard's safetensors HEADER when there isn't (a single-file conversion has no index) — in both cases a bounded 1 MB prefix via `readSliceShort`, never `allocRemaining`, which on a 2 GB shard either eats the RAM or (with a cap) errors `StreamTooLong` and silently declines the model.
- **No third copy**: `model_discovery.zig` imports only `std` + `log`, so `gen.zig` can import IT. `gen.peekModelType` delegates rather than duplicating (the MageFlow round has two copies kept in sync by comment). `cli.isModelDir` calls the same predicate.
- **App side**: `MediaBundle.flux(...)` listed `config.json` as a ready marker, so a complete 10 GB download of this repo would read as permanently incomplete and the pane would offer Download forever — the Kokoro-reusing-`tts()` bug exactly. `hasRootConfig: false` is a fact about the repo, declared per preset, not a preference.
- **Guards**: `discoverModels finds an mflux FLUX.2 repo (no root config.json)`, `flux2 detects from an mflux conversion with no root config.json`, `cli: isModelDir accepts an mflux FLUX.2 repo (no config.json at all)` — each also pinning that a same-shaped repo with different weight names stays hidden — plus `testKlein9BBundleDoesNotRequireARootConfigItNeverShips`.

### A model family's geometry belongs to the checkpoint the moment a second size exists (2026-07-29)

Adding klein 9B to the app is a five-line preset. Running it was not: `FluxConfig`'s struct defaults WERE the 4B (5 double / 20 single blocks, inner 3072, 24 heads, joint 7680, encoder 2560/9728), `loadDit` did `d.cfg = .{}` and allocated `cfg.double_layers` blocks from it. Against a 9B checkpoint that loads 5 of its 8 double blocks and 20 of its 24 single blocks, raises NO error (the extra blocks are simply never asked for), and generates a plausible, wrong image.

Reading mflux's `AVAILABLE_MODELS` shows the entire 4B→9B delta is six numbers — `num_layers` 5→8, `num_single_layers` 20→24, `num_attention_heads` 24→32, `joint_attention_dim` 7680→12288, encoder `hidden_size` 2560→4096, `intermediate_size` 9728→12288 — while head_dim 128, in_channels 128, mlp_ratio 3, RoPE theta 2000, the 36 encoder layers and the 9/18/27 taps are shared. The VAE is byte-for-byte the same tensor set (266 tensors, identical shapes), and so is the tokenizer.

- **Source of truth**: the WEIGHT SHAPES. Neither checkpoint carries the numbers in json (the 9B has no config.json at all; the 4B's records only its quantization), and shapes are the one source that cannot disagree with the tensors about to be loaded. `ditConfigFrom`/`teConfigFrom` are pure functions over a probe struct, so they unit-test without mlx; the loaders fill the struct with `rowsOf` / `logicalInDim` / `countIndexed`.
- **Trap**: encoder head counts come from `q_proj`/`k_proj` ROWS, never `hidden ÷ head_dim`. The latter reads 32 on the 9B (right, by luck) and 20 on the 4B (wrong — it has 32 q heads over a 2560 hidden).
- **Trap**: `ff.linear_in` emits gate+up fused, so `mlp_ratio` is its rows ÷ 2 ÷ inner. Reading it as rows ÷ inner gives 6.
- **A zero probe leaves the field alone** rather than writing 0 — an unreadable shape must not produce a `alloc(0 blocks)` that "succeeds". Whatever is actually missing gets named by the weight load a few lines later.
- **Verified no-op on the 4B**: the derived values log identically to the old hardcoded ones (`inner=3072 heads=24x128 double=5 single=20 joint=7680`, `hidden=2560 layers=36 heads=32/8 inter=9728`) and a live 512² generation is unchanged.

### `fitAspect` preserved the aspect; `normalizeSize` threw it away one step later
Found by pre-merge review of the MageFlow branch (2026-07-25). The edit path already had the right idea: MageFlow edits AT the target grid, so a square request would squash a 3:2 photo, and `fitAspect` was added to reshape the requested size to the primary reference's aspect at the same pixel budget. Its unit test passed. The integration test passed. The geometry was still wrong for most real inputs.

`fitAspect`'s result goes straight into `engine.normalizeSize`, which clamps width and height **independently** through `clampKreaDim` (`[256, 2048]`). Independent clamping is not aspect-preserving. Verified against the real functions:

```
source    4032x3024  aspect 1.3333      (a 12 MP 4:3 phone photo)
fitAspect 4032x3024                     (correct — matches the source)
clamped   2048x2048  aspect 1.0000      (squared off)
```

So `fitAspect` did its job and the very next call undid it, producing the exact failure it was written to prevent. Reachable through the most ordinary call there is: `client.images.edit(image=…, prompt=…)` with no `size`, on any photo whose long edge exceeds 2048 — which is every modern phone camera.

Why no test caught it: **every fixture was under the cap.** The unit test's "match source" round-trip used `{1152,768}, {2048,512}, {640,1600}, {512,512}`; `tests/test_mageflow_edit.sh` generates a 1152x768 source. Below 2048 the clamp is a no-op, so the composition looks correct from every angle the suite could see. A test suite that only samples inside a threshold cannot see a bug that lives above it.

- **Fix**: `resolveEditTargetSize` = `fitAspect` (aspect at the budget) → `fitWithinCap` (scale BOTH dimensions by one factor until neither exceeds the cap, /16). The cap comes from `ImageEngine.maxDim()`, so the edit path gets the ceiling as a NUMBER to scale toward rather than a clamp to be squared by.
- **Drift guard**: `maxDimFor` and the clamps encode the same ceiling in two places, so `maxDim matches what normalizeSize actually clamps to` asserts `clampFluxDim(99999) == maxDimFor(.flux)` and the Krea/MageFlow pair. Widening a clamp without touching `maxDim` reintroduces the squash silently, so the drift test is the actual long-term guard.
- **Rule**: when one step establishes an invariant and a later step re-derives geometry, test the COMPOSITION, not the step. And pick fixtures that straddle every threshold in the pipeline — an all-under-the-cap corpus proves nothing about the cap.
- **Guards**: `resolveEditTargetSize keeps the reference's aspect ABOVE the backend cap` (4:3 and 3:2 in both orientations, plus a huge square that must stay square, plus the under-cap round-trip so the existing sizeless behavior is pinned unchanged), and `tests/test_mageflow_edit.sh` now also drives a 3024x2016 source and asserts the output is within the cap AND still 3:2. Verified red on revert.

### Laguna YaRN: the checkpoint's `attention_factor` is not the mscale the model was trained with

Found 2026-07-28 before a single Laguna-XS number was recorded, by reading poolside's own
MLX implementations rather than trusting the config.

Laguna's full-attention layers use YaRN with `factor: 32.0`. MLX's YaRN computes its
default mscale as `0.1 * ln(factor) + 1 = 1.3465735…` when `mscale`/`mscale_all_dim` are
left at their defaults. **Both vendored MLX Laguna implementations deliberately do not
forward the HF `attention_factor` field** (`LagunaRuntimeModel.swift:42-48`), and
poolside's fused RoPE kernel hardcodes the computed result, `1.3465735912322998f`.

`model.zig` read the field verbatim into `config.yarn_attention_factor`, which
`transformer.zig` applies as the mscale on the rotated dims of full-attention layers:

```
Laguna-S-2.1  config: "attention_factor": 1.3465735902799727   ← equals the computed value
Laguna-XS-2.1 config: "attention_factor": 1.0                  ← does NOT
```

So S matched **by luck** and XS would have silently run 10 of its 40 layers with unscaled
YaRN RoPE. Nothing crashes; the model just attends slightly wrong at long range, and every
benchmark taken on it would be measuring a subtly different model than the reference.

- **Fix**: the `laguna` arm stops reading `attention_factor` and computes
  `0.1 * @log(yarn_factor) + 1` (guarded on `yarn_factor > 1.0`). The generic nested-rope
  YaRN reader is untouched — other architectures legitimately honour the field.
- **Why S stays provably unchanged**: the computed value equals the value S ships, so the
  pre-existing S config test passes without modification. That is the regression guard.
- **Rule**: when an arch's reference implementation deliberately IGNORES a config field,
  the field is not the source of truth for that arch — pin the value the checkpoint was
  trained with, and make the two configs agree by construction rather than by luck. A
  config field that happens to hold the right number on the model you tested is the most
  expensive kind of coincidence.
- **Guard**: `ModelConfig: laguna YaRN mscale is COMPUTED, never read from
  attention_factor (Laguna-XS ships 1.0)` — a config carrying `1.0` must still parse to
  `1.3465735…`; verified red before the fix.

## An imatrix is only valid for the weights it was collected on (2026-08-01/02)

The imx mirror shipped 2026-08-01 was calibrated with the only published DSV4
imatrix — antirez's `DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat`,
committed 2026-05-12, i.e. collected on the PREVIEW checkpoint. 0731 retrained
the weights. Live symptom that kicked off the investigation: the o-for-0 digit
class (`1o` for `10`, `#ff0o0o`) making a pi-driven Three.js build loop on
self-repair (5 of 8 initial file writes corrupted; the model twice
re-introduced the corruption while rewriting, then got confused by its own
grep; run cut by `[length]` at 26K ctx — full record in the session
post-mortem).

Collection: `lib/ds4/ds4 --imatrix-dataset gguf-tools/imatrix/dataset/
rendered_prompts.txt --imatrix-out … --ctx 32768` over antirez's official
imatrix-fixed-0731 GGUF (97.6 GB donor, resident on 128 GB; prebuilt Jul-15
binary no longer matched the checked-in Metal sources — `make ds4` first). Full
corpus 2.9M tokens / 4,692 prompts / 747.6M routed observations in ~3.2 h at
~250 tok/s prefill. The known upstream delimiter bug (fixed on the mxfp4
branch) split exactly 2 prompts that quote the collector's own source —
negligible, but cherry-pick `0a62b39` on the next ds4 bump.

Numbers that justify the class rule:
- preview vs our 0731 collection: median Pearson 0.66, top-5%-channel Jaccard
  0.33; per-expert slices ~85% of experts decorrelated (r<0.9).
- our collection vs ox-ox's independent 1.5M-token 0731 collection: 0.85 —
  two same-weights collections agree far better than either agrees with
  preview, so the movement is the retrain, not donor/sampling noise. (ox-ox
  published the first 0731 `.dat`; ours doubles the token count. Their README's
  structural analysis holds here: post-RMSNorm gate/up statistics are near-flat
  across channels, so an imatrix mostly informs `down`.)
- pilot on real 0731 shards (L3/21/40 × w1/w2/w3, today's recipe, old-cal vs
  new-cal, scored under the new statistics): w2 (3b) −5.5..−10.3% weighted
  error, w1/w3 (2b) −0.4..−0.7%. 7 wins / 2 ties / 0 losses.
- v2 vs preview-cal build, live: char precision 35 vs 38 slips; task set 4/5
  both; o-for-0 collapse rate UNCHANGED (1/12 sampled arms each — the collapse
  is a 2-bit near-tie property: `o`≈`0` in numeric contexts everywhere in
  training text, quant noise flips the nearest tie first, then the model
  copies its own slip → all-or-nothing corruption; greedy oracle parity
  exonerates the engine). test_dsv4.sh 14/14, template 17/17, greedy pins
  UNCHANGED, DSpark acceptance 3.79/round at 53.3 tok/s sampled.

Two operational traps caught in the same round: (1) re-converting the mtp
group is NOT byte-stable across sessions (59/149 tensors drifted at rounding
level in scales+packed words with identical headers) — the stage byte-identical
rule is enforced by copying the proven shard, never re-converting; (2) the
memory preflight legitimately refuses the ~102 GB mirror on a box with a
browser open (~105 GB "available") — `DSV4_TEST_SKIP_PREFLIGHT=1` in
test_dsv4.sh mirrors how the mirror is actually served, default stays strict.

## The DSV4 echo-precision saga was a TOKENIZER bug (2026-08-02, iq2 week)

Symptom history: DSV4 mirrors slipped single characters in verbatim material
(`1o` for `10`, invalid hex, split digits, the sampled o-for-0 collapse) at a
rate that survived the imatrix recalibration (38→35 char-precision slips).
During iq2-codebook bring-up the
new mirror slipped 38-40/70 on the char-precision battery while the SAME GGUF
weights served through the embedded ds4 engine scored 0/70.

Elimination ladder (each step a real discriminator, all on live boots):
MV kernels off → identical; f32 expert activations → identical; AR fused
kernels off → identical; affine canonical on our
engine → 36/70 (same level); the python reference oracle on the canonical →
uuid echo PERFECT. Per-case table: the ONE clean case everywhere was the one
without digit-dense payload. The three clean systems (oracle, ds4) tokenize
with their OWN encoders; every slipping config used tokenizer.zig.

The probe that ended it (no GPU): `/tokenize` "100" → ours [1,0,0] per-digit;
canonical HF `tokenizers` → [1457] = "100" one token. DSV4's tokenizer.json
pre_tokenizer rule 1 is `Split(Regex: \p{N}{1,3}, Isolated)` — greedy 3-digit
groups — while our hand-rolled gpt2PreTokenize implemented the QWEN reference
(bare `\p{N}`), correct for Qwen3.6 (which declares single digits — surveyed)
and wrong for DSV4. Every number the model ever read through our server was
per-digit segmented — off-distribution for a model trained on {1,3} groups —
so verbatim digit recall degraded exactly like a quantization-quality bug,
which is what everyone (including the quality batteries) attributed it to.

Fix: `Tokenizer.digit_group` parsed from the Split rule (exact match on
`\p{N}{1,3}`, conservative default 1), pattern 3 consumes up to that many
digits. Char-precision went 38-40 → **0/70** on the iq2 mirror. Guards:
gpt2PreTokenize group tests (["104","857","6"] etc.), digitGroupFromPreTokenizer
parse tests, Qwen single-digit behavior pinned unchanged.

Meta-lesson: when a quality defect is INVARIANT across engine configs and
quant recipes, stop ablating the engine — diff the PROMPT ENCODING against an
independent tokenizer first. It is a two-minute check that would have saved
days. (Also: the iq2 bring-up's ds4-vs-native cross-check is what surfaced
this — an independent-implementation oracle earns its keep.)

## Uniform ≤2-bit experts to the last layer cause TURN-LEVEL agent loops (2026-08-02, tail4)

The shipped DSV4 mirror (`ddalcu/DeepSeek-V4-Flash-0731-MLX-Serve-mixed-2-3-8bit`,
115.4 GB, HF commit `be89f88`) is the **tail4** build: imatrix-calibrated
2b/g128 gate/up + 3b/g128 down experts on layers 0-38, but AFFINE 4b/gs64
experts on the last 4 layers (39-42). The 4-bit tail exists because the
uniform low-bit build had a failure mode no token-level gate could see:
**turn-level agent repetition loops**.

Mechanism: the same near-tie physics as the o-for-0 collapse, lifted from the
token level to the DECISION level. At a turn boundary the model weighs "am I
done" against "verify once more"; 2-bit expert noise in the late layers flips
that near-tie, and once the transcript contains one redundant verification
round the model copies its own pattern (error-echo class) and orbits. Every
INDIVIDUAL response is short, fluent, and correct — the SEQUENCE is what
loops — so:

- the token-level degenerate-tail guard (`loopStopReason`, even the 9..64
  long-period tier) never fires — no token window repeats;
- the char-precision / o0 batteries score the arms identical;
- task-success scoring is blind too — the looping runs eventually finish
  with the SAME 50/50 correctness, just 3-5x the tokens.

bench4 (agent tasks via the pi harness, DSpark on): the uniform-2-bit
canonical looped 3 of 4 task-B runs, burning 16-28K tokens each; tail4
looped 0 of 3 at 5.4-7.6K tokens. Correctness identical (50/50), throughput
identical (~53 tok/s DSpark both arms) — the tail bits buy BEHAVIOR, not
speed or single-token quality.

Rule distilled to CLAUDE.md: quality gates for low-bit mirrors need an
AGENT-LOOP cell. The loop metric is **max consecutive identical tool calls**
across a multi-turn agent run; reusable harness:
`~/claude-tmp/iq2-week/bench3/loop_stats.py` + the bench4.sh pattern
(paired arms, same tasks, token budgets + loop stats per run).

## The weights MAP pins everything a staged loader frees (MiniMax-H3 AdaLN precompute, 2026-08-03)

H3's AdaLN precompute tables the whole schedule's modulations and frees each
block's 260M-param AdaLN weight right after its table evals — a designed
~13 GB residency win on the 8-bit pack. The first live run showed `dit
resident: 33.36 GB` where ~20 was expected, and the OFF-arm control read the
same 32.8 GB, proving the free was a no-op. Cause: `generate` kept the
safetensors `Weights` map alive (`defer dw.deinit()` at scope end), and the
map holds +1 refs on every raw file-backed array — the model's frees only
dropped the model's handles. mlx lazy graphs keep their inputs alive
internally, so the map can be dropped the moment `Model.load` returns; scoping
it inside a blk took residency to 19.95 GB measured. Two durable lessons: any
STAGED-residency loader owes the same scoping, and the fix was only visible
because the load path logs `mlx_get_active_memory` — a memory claim without a
resident log line is a hope, not a design.

## The stub model_type is a MODALITY static — the per-backend preflight must re-peek (2026-08-03)

`buildStubCpuState(modality)` stamps `modality.modelType()` ("AudioVideo" for
every video backend) on the stub config, so the new media preflight keyed on
`params.config.model_type` never matched "minimax_h3" and billed the 64.5 GB
sum-of-safetensors — while printing a log line that CLAIMED staged billing,
and while the unit tests for the estimator (called directly with the right
type) stayed green. Second bite of "a marker belongs to a BACKEND, never a
modality". Fix: `doLoadGenOnInferenceThread` re-peeks the dir's real type via
`gen.peekModelType`, the same authority `VideoEngine.load` dispatches on. The
guard lesson: `tests/test_minimax_h3.sh` now asserts the printed NUMBER sits
in the staged band — the line's presence proved nothing, which is the same
class as spec-decode engagement counts vs output equality.

## The weight prefix was a config GUESS, and LFM2.5 nests without saying so (2026-08-04)

`mlx-community/LFM2.5-2.6B-8bit` (and the nvfp4 sibling) refused to load:

```
Model: lfm2 (30 layers, 2048-dim, head_dim=64, 32h/8kv, 8-bit affine quant)
Loaded 600 weights from 1 file(s)
MISSING WEIGHT: model.embed_tokens.weight
```

All 600 weights were there. They just lived under `language_model.model.*`.

`parseConfigFromJson`'s lfm2 arm picks the prefix the same way the gemma4 arm
does — `if (root.get("text_config") != null) "language_model.model" else
"model"` — and this config has NO `text_config`. It declares
`architectures: ["Lfm2ForCausalLM"]`, a text-only 2.6B, and the only hint that
anything is nested is an EMPTY `"vision_config": {}`. Reading nesting out of
that is a coin flip: an empty vision_config is exactly as plausible on a flat
checkpoint, and the same class already shipped in the opposite direction
(a nested guess against flat weights).

So the guess stops being the authority. `model.resolveWeightPrefix` runs at
both trunk load sites (main.zig offline, scheduler.zig serve), right after
`loadWeights` and before `Transformer.init`, and re-points the config when the
configured prefix holds NOTHING and the other nesting has weights. Three things
keep it from being able to break a checkpoint that loads today:

- it only ever alternates between `model` and `language_model.model` — an arch
  with its own prefix (`backbone`, `model.llm`, `""`) returns immediately;
- it fires only when the configured prefix matches ZERO keys, so a real VL
  checkpoint carrying both keeps what the config said;
- the prefix match requires a `.` boundary, so `model_extra.*` is not a hit for
  `model`.

Same shape as `hy3ExpertContainer`: when two converters disagree about a name
and config.json cannot settle it, probe the tensors.

Both variants load and generate after the fix (8-bit 148 tok/s, nvfp4 247
tok/s on an M-series 128 GB), thinking splits correctly into
`reasoning_content` under `reasoning_effort`.

Still open, and a SEPARATE gap: LFM2.5 tool calling. Its template emits
Python-call syntax — `<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>` —
which no arm of the parse chain knows, so a tools request comes back with
empty content and no `tool_calls`. Its template also `raise_exception`s on
tool-call `arguments` passed as a JSON STRING, which is what
`serializeMessagesJson` emits — the Inkling rule in reverse, and a raise is the
silent-fallback class.

## The hybrid layer-init path demanded scales, so it could not load its own arch dense (2026-08-04)

`mlx-community/LFM2.5-2.6B-bf16` refused to load, one line past the prefix
probe that fixed its 8-bit sibling:

```
MISSING WEIGHT: language_model.model.layers.0.conv.in_proj.scales
```

The checkpoint is dense bf16 — no `.scales`, no `.biases`, anywhere. 266
tensors, all present.

`initHybridLayers` fetched all 17 of its scales tensors with `getLayerWeight`,
the MANDATORY getter whose miss is `log.err` + `unreachable`. The other two
layer-init paths never did: `initStandardLayers` and `initMoeLayers` both read
scales with `getLayerWeightOpt(...) orelse mlx_array_new()`, and `qmatmulBits`
has had a plain-`mlx_matmul` arm on null-ctx scales since the Unsloth Dynamic
mixed checkpoints (which leave a SUBSET of layers unquantized — so "the config
declares no quantization" is not the right gate either; absence is per-tensor).
The hybrid path was simply written against quantized checkpoints and never
revisited, which is invisible for as long as every published checkpoint of
that arch is quantized.

Two halves to the fix, and the second is the one that would have produced
silent garbage instead of an honest abort:

- scales fetch through `getLayerScaleOpt` (absent ⇒ null-ctx);
- every weight a matmul CONTRACTS gets `maybeTransposeForBf16`, because the
  dense arm expects `[in, out]` while the checkpoint stores `[out, in]`. In a
  hybrid layer that is in_proj/out_proj (gated conv), q/k/v/o, mamba2's
  in_proj/out_proj, the simple MLP's up/down and the LFM2 SwiGLU's w1/w2/w3 —
  and NOT `conv.conv.weight` (depthwise kernel, read by `conv1dWithCache`) or
  any SSM state tensor (`A_log`, `D`, `dt_bias`). Transposing those would
  either crash in the conv or quietly compute the wrong thing.

`initHybridLayers` now returns `owned_bf16` like its two siblings, wired into
the same `moe_owned_bf16` list `Transformer.deinit` frees — a transposed array
is a NEW array, and the load path is the only place that can hand it to the
teardown.

Guards, both proven red-on-revert:

- a synthetic 2-layer dense lfm2 `Weights` map through `initHybridLayers`,
  asserting null-ctx scales, `[in, out]` on the matmul operands and an
  UNTOUCHED conv kernel (instance);
- a source scan over every `.scales")` fetch in transformer.zig rejecting the
  mandatory getter (class) — a new arch arm that reaches for scales the same
  way breaks dense checkpoints of that arch, and a quantized checkpoint of the
  same arch would pass either way.

Live after the fix: bf16 82.6 tok/s decode, 8-bit 148, nvfp4 247 on a 128 GB
M-series; `test_hybrid_reuse_equivalence.sh` green on the bf16 (warm reuse
13.3x, byte-identical) and unchanged on the 8-bit.

---


## An fp override INSIDE the fp family is invisible to a scales-dtype rule (2026-08-05)

`LiquidAI/LFM2.5-2.6B-MLX/mxfp4` killed the server on its first request:

```
MLX error: [dequantize] Shape of scales does not match the matrix given the
quantization parameters. Provided matrix of shape (1,512) and scales of shape (1,64).
```

Its config declares a model-wide mode AND a per-tensor override:

```json
"quantization": { "group_size": 32, "bits": 4, "mode": "mxfp4",
                  "model.embed_tokens": { "group_size": 32, "bits": 8, "mode": "mxfp8" } }
```

`computeQuantParams` resolves per weight off the SCALES DTYPE — uint8 ⇒ the
config's fp mode, float ⇒ affine with (bits, gs) solved from the activation's
inner dim. That rule was written for the mixed shape that existed at the time:
an *affine* override inside an *fp* model, where the two are told apart by
their scales (bf16 + biases vs uint8, bias-less). Here BOTH sides are fp —
MXFP4 and MXFP8 scales are the same uint8 E8M0 byte — so the dtype says
nothing, the embedding resolved at the model-wide 4 bits, and `mlx_dequantize`
rejected the shapes. An MLX error is not a Zig error: one request, whole
process gone. The shape in the message is the GATHERED row (`embedding()` takes
rows before dequantizing), which is why it reads (1,512) rather than the
table's [128000, 512].

The geometry states the answer exactly, so `fpParamsFromGeometry` solves it the
same way the affine arm already solves off-config sidecar quants: a row holds
`w_cols * 32` packed bits over `in_dim` values ⇒ bits; `s_cols` scales cover
them ⇒ group size. For the embedding that is `512*32/2048 = 8` bits and
`2048/64 = 32`; for every linear in the same file, `256*32/2048 = 4` and 32.

Two things keep it conservative:

- it returns **null when the solve AGREES with the config**, so a consistent
  checkpoint resolves byte-for-byte as it did before this existed (nvfp4 and
  affine checkpoints re-smoked: unchanged);
- the mode comes from `fpQuantModeFor`, and each fp format DEFINES its block
  size (MXFP8/MXFP4 = 32, NVFP4 = 16), so the solved pair names exactly one of
  them. A pair matching none returns null — the checkpoint's own declaration
  stands and a genuinely broken file still fails honestly, rather than being
  relabelled into a mode we invented for it.

Live: mxfp4 serves at 239 tok/s, coherent, alongside the 4-bit variant of the
same repo (234). Guard: the `computeQuantParams` fp-override test, which pins
the override, the untouched sibling linear, an nvfp4 model, an unmappable
geometry and the no-hint path.

## The H3 `<Picture i>` splice: a faithful port that made the model worse (2026-08-05)

MiniMax-H3's reference (`comfy/text_encoders/minimax.py`) splices every fl2va
keyframe into the Qwen3-VL conditioning as `<Picture i>: <vision block>`, on top
of conditioning through its VAE latent. Our port had shipped with only the VAE
path and a `KNOWN DEVIATION` comment. Closing that deviation was supposed to
improve keyframe fidelity on a checkpoint already on disk.

It inverted it.

### The measurement

Same seed, 256px, 12 steps, `fast: false`, conditioned on a hard 20/235
left/right split; the numbers are frame 0's left and right half means.

| arm | frame 0 | delta |
|---|---|---|
| vision blocks OFF (pre-change) | 16.1 / 231.4 | **+215** |
| ON, plain 1-D rope | 43.0 / 119.5 | +77 |
| ON, interleaved mRoPE, no DeepStack | 48.5 / 105.2 | +57 |
| ON, mRoPE + DeepStack | 150.6 / 53.0 | **-98, INVERTED** |

Turning off EITHER contributor restores the correct polarity. That pattern is
the whole diagnosis: if one component were broken, disabling the OTHER would not
help. It is interference.

### The port is not the bug, and proving that took the right instrument

Everything was pinned against ComfyUI's own output before concluding anything:
patch order (`cos 1.000000`), the tower (`cos 0.999987` merged, ~0.99999 on all
three DeepStack taps), the mRoPE position ids (exact, including row-index-
weighted checksums), the interleaved-mRoPE axis map, and the vision delimiter
ids against H3's actual vocab.

**The instrument that was missing was scale.** A cosine is scale-invariant, and
a tower whose features came out 40x too large would have scored a perfect 1.0
while destroying every sequence it was spliced into. Adding `rms_ratio` to the
parity test answered the real question: `1.0000`.

Which surfaced the actual mechanism. The reference's own merged vision rows
carry a norm of **61.0** against **1.49** for a token embedding — 41x. Qwen's
per-layer RMSNorm absorbs most of that on the way into attention, but the
RESIDUAL stream keeps it, so 64 vision rows dominate a 9-token prompt's
contribution to what the DiT receives as context. On fl2va that conditioning has
nothing to add: the keyframe is ALREADY pinned by its VAE cond rows, which is
why the un-spliced arm reproduces the input almost exactly (16/231 vs 20/235).
The second path is not extra information, it is a competing one.

### What shipped

Default OFF for fl2va (`MINIMAX_H3_VISION_BLOCKS=1` re-enables for A/Bs), ON for
ref2va — where a reference image has no cond row to compete with and the prompt
ADDRESSES it as `<Picture i>`, so the tower is the entire mechanism rather than a
second opinion.

**The rule: a reference implementation is a spec for what to build, not a
promise that turning it on helps your checkpoint.** The deviation comment was
right to exist; what it needed was a measurement, not a fix.

Bisect arms are kept in-tree (`MINIMAX_H3_VISION_BLOCKS`,
`MINIMAX_H3_VISION_MROPE`, `MINIMAX_H3_VISION_DEEPSTACK`) because the three-way
split above is what made the interference legible, and any future attempt to
turn this back on will need exactly the same three.

---

## H3 ref2va: two partitions, one file layout, and a cond stream ordered by the layout (2026-08-05)

Ref2VA is MiniMax-H3's reference-conditioning checkpoint: up to 9 images, 3
clips and 3 audio references that the generation follows for character, style
and scene continuity. Landing it turned up three classes worth writing down.

### The two partitions are indistinguishable from their files

FL2VA and REF2VA share the text encoder, both VAEs, the tokenizer and every
geometry number. The Comfy-Org repo ships one text-encoder file that serves
both. Only the DiT weights differ — and a DiT does not announce what it was
trained for. So an FL2VA pack handed `ref_images` would load, run, generate a
perfectly good clip, and ignore every reference the user attached.

The only honest discriminator is the task list our own converter writes into
`config.json` (`["t2va","ref2va"]` vs `["t2va","fl2va"]`), read once at engine
load (`gen.h3ConfigDeclaresRef2va`). Absent, malformed or wrong-typed reads as
NOT ref2va: a pack that cannot say it supports references must not be handed
them. The server 400s by name; the app's preset carries `supportsReferences`
through the shared `minimaxH3Preset` factory so the two FL2VA presets cannot
drift into claiming it; and `requestBody` gates the FIELDS on it rather than
only the controls — hiding a control is not the same as not sending its field,
which is the class that once made every H3 request carry `pipeline` and 400.

The integration script reads the same list to pick which half of its assertions
to run. Asserting fl2va keyframe adherence against a REF2VA DiT would be a
checkpoint expectation in a server test's clothes.

### The cond stream is ordered by the LAYOUT, not by the request

`PackedLayout` fills its `img_update` / `audio_update` false rows in SEGMENT
order — keyframe conds, then each reference block in resolver order, then the
target — and the forward drops the concatenated condition rows straight into
them. Assemble those rows in REQUEST order instead and every reference lands on
the wrong block's positions: the model conditions on a smear, the generation
succeeds, and nothing reports it.

The ordinals are a contract with the checkpoint for the same reason. The prompt
refers to `<Picture 2>`, and a video's soundtrack consumes an `<Audio j>`
ordinal emitted BEFORE its own `<Video k>` — so one soundtracked clip plus one
standalone audio makes the standalone `<Audio 2>`, not `<Audio 1>`.

One function decides order and ordinals (`resolveRefs`); the DiT block is
DERIVED from a resolved reference through the same function `resolveRefs` itself
uses (`refBlockFor`), so a second hand-rolled mapping cannot drift from it —
the class that put the dsv4 spec exclusion in three places.

And the ENGINE consumes that decision rather than re-deriving it. That is forced
rather than chosen: a reference's canvas comes OUT of the resolver, and the
server has no image resampler — it resizes by decoding AT a size. So the caller
must resolve before it can decode, which makes the caller the only place the
decision can live. It is also why each visual reference carries two decodes: one
at the VAE canvas for the DiT payload, one at `h3v.fitCanvas` of it for the
vision tower.

Two details that look like oversights and are not: audio condition rows ride in
CLEAN (`AUDIO_COND_TIMESTEP` is 1.0 and the reference only augments below 1.0,
so no noise is ever drawn), and vision blocks are ON for references while they
are OFF for fl2va keyframes — a reference has no cond row competing with the
tower, and the prompt addresses it by name.

### The upstream layout was not what the driver assumed

`tests/fetch_convert_minimax_h3_ref2va.py` had been written ahead of the
release. Two of its assumptions were wrong, and both are the kind that read as
"upstream did not publish this":

* the Comfy-Org repo has **no `split_files/` prefix** — the paths are
  `diffusion_models/`, `text_encoders/`, `vae/`;
* MiniMaxAI's tree is spelled **`Ref2VA/`**, not `REF2VA/`.

A repo listing settles both in seconds and is the right first move before
concluding a file is unpublished — **but the listing has to be read
case-INSENSITIVELY.** A `f.startswith("REF2VA")` filter over the same listing
returned nothing and produced a confident "upstream publishes no REF2VA tree,
so the partitions must share FL2VA's tokenizer". The conclusion happened to be
right — `Ref2VA/processor/tokenizer.json` and `chat_template.json` are
byte-identical to FL2VA's (sha256 a5d85b6d… / 5c72a170…) — which is exactly
what makes this the dangerous shape: a wrong premise that a correct-looking
result never contradicts. Hash the two files; do not infer identity from an
absence you did not actually observe. The driver now also takes `--reuse-from
<pack>` and CLONES the shared text encoder and both VAEs from an
already-converted pack (APFS `cp -c`: instant, no extra space), so a second
partition is one 66 GB download and one conversion instead of two — 21 minutes
wall clock instead of hours. Copying a proven shard is also the CORRECT move
rather than merely the fast one: reconversion is not byte-stable across
converter sessions (the DSV4 mirror round measured 59/149 tensors drifting at
rounding level).

One ordering bug fell out of the same work: the converter checked its SOURCE
before its destination, so every resumed run failed on the shard it had already
finished — the driver deletes each source once converted, by design.

## H3 length planning: a stale cap, a borrowed memory model, and what actually bounds a clip (2026-08-05)

Asked whether the app exposes what MiniMax-H3 supports, three answers came back
and only one of them was "yes".

**Resolutions: complete.** The card states short side 768 and ratios "21:9,
16:9, 4:3, 1:1, 3:4, 9:16"; `h3Resolutions` carries all six plus our own
sub-native 960x544 long-form canvas. The only missing tier is 2K, which needs
H3-Regenerate-2K — a checkpoint we have not converted.

**Steps: MiniMax publishes nothing.** No default, no range, no maximum; the
reference Comfy node leaves it to the KSampler. So the range was ours and had
never been said out loud: the slider was `4...50` with LTX's help text ("LTX runs
well from ~8") rendered on H3, where 8 steps is below anything we have a verdict
on and — worse — below the point where the fast recipe's warmup and tail windows
cover the whole schedule, so those runs pay full price per step while looking
like the cheap option. Now `stepsRange` / `stepsHelp` are per preset, 16...50 on
H3.

**Frames: we were short of the model.** MiniMax states 4-15 s at 24 fps; the
ladder is 17k+5, so 15 s is 362 frames. The picker stopped at 209 — the rap
demo's length, the longest clip we had shipped a verdict on — behind:

```
// Trained range is ~124-362 (reference tooltip); 209 is our own
// validated ceiling (the rap demo), not the untested-by-us 362.
```

which `MediaGenServiceTests` had already disproved in a comment of its own:
*"Field measurement on an M5 Max (2026-08-04, our engine): 362 frames — the top
of the 17k+5 ladder — in ONE generation at 139.6 s/step"*. A cap standing in for
"we have not measured this" goes stale silently, because nothing fails when the
measurement finally happens.

### What actually bounds a clip

`RAMChecker.safeFrameCap` was LTX's formula — a fixed load cost plus ~12 GB per
megapixel per 100 frames of VAE decode staging — applied to every video backend.
On H3 that is not an approximation but a different model, and it produced two
useless readings: 677 frames on a 128 GB Mac (so the warning never fired) and 32
on a 48 GB one (below H3's own 124 floor, so it fired always).

H3's frame-dependent term is the fast recipe's attention-broadcast cache: one
`[S, hidden]` bf16 per block, held for the whole run.

```
latent_t = ((frames - 5) / 17) * 5 + 2
S        = latent_t * (w/32) * (h/32) + 2 * round(frames * 5/3) + text
cache    = S * 5376 * 2 * 50 blocks
```

Audio is STEREO — `n_audio_rows = audio_t * 2` — which is easy to drop and worth
~0.4 GB at 768p. The cache is LINEAR in rows: SDPA is fused, so the `[S, S]`
score matrix is never materialized, which is the only reason a 108k-row sequence
exists at all (it would be 1.3 PB). The formula reproduces both measured points
— 20.4 and 34.1 decimal GB at 124 and 209 frames, 1344x768 — which is what makes
it worth trusting at 362, where nobody has measured.

Three things fell out of writing it down:

- **"Max quality (slower)" is also the LOW-MEMORY mode.** `bcast_k = 1` never
  allocates the cache, so 1344x768 x 362f goes from ~85 GiB to the 38 GiB staged
  load peak, in exchange for roughly 4x the runtime. That is backwards from
  every other quality toggle in the app and impossible to guess, so the
  over-budget warning now offers it by name.
- **The load floor is the pack's MEASURED staged peak, not `approxRAMGB`.**
  `approxRAMGB` is that number rounded up (26 vs 22.82 GiB for the 4-bit pack)
  and drives the coarse "does this Mac have enough RAM at all" alert, where
  erring high is free. In the frame model it is not free: 26 against a 32 GB
  Mac's 0.8 budget of 25.6 refuses the pack that exists for 32 GB Macs.
- **A cap with a FLOOR cannot answer "does anything fit".** `frameCap` never
  returns below the ladder's 124 (below it the model is off-distribution, so a
  smaller number is not a usable answer), which makes `cap == 124` mean both
  "124 frames fits" and "nothing fits" — and a 24 GB Mac, which cannot load the
  8-bit pack at all, saw no warning at exactly 124 frames. `H3Plan.fits` answers
  the configuration, and the warning asks it.

Ladder reach and quality-TIER defaults also had to be separated. Raising `cap`
to 362 silently moved `.quality` and `.superQuality` there too, i.e. picking
"Quality" would have started a five-hour job; `tierMax` stays 209 and the slider
reaches 362.

### Time

`c + a*S + b*S^2`, fitted on two M4 Max dense measurements (36.6 s/step at
864x480 x 73f, 275.7 s/step at 1344x768 x 124f). The square term is load-bearing:
at a matched 124 frames, 1344x768 measured **2.9x** the time of 960x544 for
**1.98x** the pixels, so a linear-in-pixels model under-promises the wide canvas
by about half.

The fast recipe is priced from the engine's OWN broadcast schedule
(`attnBroadcastRefresh`, mirrored) rather than a flat 1/2.83, because those
windows SCALE with the schedule — at <=6 steps they cover the whole run and the
recipe is a no-op, so a flat factor would promise a short run 3x faster than it
can go. Velocity caching then removes a further share of what is left; 0.55
reproduces the capstone's measured 14-of-30 cached steps.

Reproduces the anchors end to end (measured / modelled): 49 / 51 min at 124f
fast, 1 h 57 / 1 h 58 at 209f, 2 h 19 / 2 h 21 dense.

Three provenance tiers, because a number extrapolated from someone else's Mac
and a number measured on yours are different claims: `H3Hardware.current` scans
`gpu-core-count` off the IORegistry and the chip name off sysctl (the step is at
the compute roofline — SDPA and the linears both ~13 TFLOPS against ~15 — so
cores is the figure that tracks); `H3RunHistory` fits ONE scalar (this machine
vs the anchor) from completed runs, MEDIAN so a thermally-throttled run does not
own every later estimate, keeping 20; and `H3StepClock` gives a live ETA from
the run's own cadence, discarding step 0 (graph build + Metal JIT) and taking a
median rather than the last lap, since a velocity-cached step takes ~0.02 s and
would make the remaining run look instant.

Also fixed while in here: the card caps ref2va at **12 files across all types**,
while the three per-type caps (9/3/3) sum to 15 — so a set could clear every
per-type cap and still be one the model was never given. `MAX_REF_TOTAL` is a
named 400 server-side and `H3RefLimits` stops the picker offering past it,
including on an empty list whose own cap has room.



## `chat_template` has two legal shapes, and reading the wrong one is a PANIC (2026-08-07)

Found during 26.8.3 pre-release validation, on a family nobody had exercised
recently: **Mistral-7B-Instruct-v0.3-4bit could not be loaded at all.**

```
thread 9501346 panic: access of union field 'string' while field 'array' is active
src/chat.zig:115:33 in loadChatConfig
```

`loadChatConfig` did `try allocator.dupe(u8, v.string)` on
`tokenizer_config.json`'s `chat_template`. HF permits that key to be either:

- a bare template string (what almost everything ships), or
- a LIST of `{"name": ..., "template": ...}` objects.

Mistral v0.3 ships the list form, with two entries — `default` and `tool_use`:

```python
[{"name": "default",  "template": "..."},   # no tools
 {"name": "tool_use", "template": "..."}]   # tools
```

A Zig tagged-union field access on the wrong active field is not a recoverable
error, so this was not "the template failed to load and we fell back" — it was
the process dying during model load. One of eight local checkpoints with a
`chat_template` uses this shape, which is exactly why it went unnoticed.

**Fix.** `chatTemplateFromValue` handles both shapes and selects the entry named
`"default"`, matching transformers (`apply_chat_template` uses `default` unless
the caller names another template). Falls back to the first usable entry when
none is named `default`, and returns null for any other shape so the caller
drops to its `chat_template.jinja` / family fallback rather than panicking.

**On not selecting `tool_use`.** Only that variant references `tools`, so the
obvious worry is that tool calling breaks. It does not: our pipeline passes
`tools_json` into the render and has its own parse/repair chain, and llmprobe
scores Mistral **8/8** on the tool checks (serialization, streamed argument
reassembly, results accepted back, parallel calls, `tool_choice: "none"`,
`parallel_tool_calls: false`, typed arguments). Per-request template selection
by name is a feature worth having; it is not what a load-time panic needed.

Result: Mistral went from *cannot load* to **99.5% engine conformance**
(chat 64/64, responses 62/62, messages 57/58, embeddings 4/4, completions 3/3).

**The general rule this leaves:** in a `std.json.Value` reader, `.string` on a
field you have not shape-checked is a crash, not a graceful miss. Any new
tokenizer_config/config reader owes both shapes — or an explicit switch whose
`else` returns null.

## An adapter's alpha lives wherever its exporter put it (`lora.fileAlphaScale`, 2026-08-06)

A LoRA's net strength is alpha/rank, and four real community files ship four conventions — a kohya per-module `.alpha` TENSOR, a PEFT JSON document inside one `lora_adapter_metadata` metadata string (`transformer.lora_alpha`/`transformer.r`), flat `lora_alpha`+`lora_rank`|`r` pairs, kohya's `ss_network_alpha`/`ss_network_dim` — plus files declaring none at all (alpha baked into the weights, 1.0 is correct). Reading only the tensor ran every PEFT export at 1.0 = rank/alpha = **8x too strong** for the common 4/32 pairing, which renders PURE STATIC. Resolution order is nested-JSON > flat > `ss_*` > none, per-module tensor still WINS (more specific), and a non-finite/non-positive alpha or rank is DECLINED — a scale we had to guess at is the bug. The resolved scale is logged per file (`[lora] <file>: scale …`) so a wrong one is visible instead of silent; `lora_scales` stays a MULTIPLIER on top of it.

**The test lesson is the bigger half**: `test_multi_lora.sh`'s 56 checks were all green through this — its exact stacking maths (`d+d == 2d` byte-for-byte, zero-B transparency, order independence) is satisfied by noise, and a second real adapter passed only by being weak enough to survive 8x. Renders now go through `tests/lora_noise.py` (mean abs difference of horizontally adjacent pixels, bar 20: real 4-8, static 43-50) and `tests/test_real_loras.sh` runs published adapters on all four backends asserting scale + attach counts + usable output. A guard that asks "did the bytes change?" must be paired with one that LOOKS.

## A sampler must never draw a RESERVED special (`tokenizer.reservedOutputIds`)

`<|fim_hole|>`-class FIM markers can be EMITTED at temp 0 by a checkpoint whose distribution collapses. Per-model mask = `special: true` added tokens MINUS EOS ids MINUS specials whose text appears in the chat template source (thinking/tool/role markers ride each model's own template; no template ⇒ off, so fallback-formatted models are untouched). The legit-output set is DERIVED, never hardcoded. Built once at load (`generate.installSuppressMask`, `[suppress]` engagement log, `MLX_SERVE_SUPPRESS_RESERVED=0`), sized to the LOGITS dim (`unpadded_vocab_size` when set — inkling), riding `SamplingParams.suppress_mask` through the `initWithOptions` chokepoint into BOTH samplers + both stochastic-verify filters (`mlx_where` + -inf, fully lazy, no host sync; the batched tick reads `gen.sampling`, not `slot.sampling`). Logprobs stay the RAW distribution — under suppression rank 1 may differ from the emitted token BY DESIGN (the field reports the model; the mask is policy).

## The degenerate-tail loop guard needs a LONG-period tier (dsv4, 2026-08-02)

A two-sentence ~58-token cycle repeated 26x sailed through the 8-token scan; `loopStopReason` also fires on periods 9..64 at 10 exact reps (`isDegenerateTailLoopRange` — long periods demand fewer reps: identical long lines in real code repeat a handful of times, not ten).

## A tiled decoder's tiles are not slices of an untiled pass (H3 visual VAE)

`create_token_ids` maps each axis to `(arange(0.5,n)/n)*2-1`, so a 256-px tile's coordinates differ from that region's coordinates in a full-canvas pass — tiling is SEMANTIC, not a memory optimization. `minimax_h3_vae.decode` refuses above the 256-px tile extent (`fitsSingleTile` → `error.TilingUnsupported`) rather than silently decoding untiled and producing off-distribution output that looks like a quality problem. Same reason its TEMPORAL chunking (5-token chunks, 2 overlap, 3-frame pre-pad drop, 5-frame cross-fade) cannot be collapsed into one pass: the VAE was trained on 17-frame clips.

## An oracle that cannot execute the reference must say so (H3 DiT parity)

The reference block calls comfy_kitchen CUDA kernels that do not run on a Mac, so `tests/dump_minimax_h3_fixtures.py` is a TRANSCRIPTION and its header states that a green test proves agreement with an independently written implementation, not with the reference. Where the reference CAN run (`dump_minimax_h3_layout.py` — pure layout/schedule math) it is executed directly, with import stubs that RAISE on attribute access so no golden value can be produced by a mock. Prefer executing the reference; when you cannot, say which one you built.

## A permutation-invariant checksum cannot see a permutation (H3 layout fixture)

The per-axis position SUM passes unchanged when the two stereo channels' pinned `w` coordinates are swapped. Pair any column checksum with a row-index-WEIGHTED one (`pos_weighted`) — found by doing red-on-revert properly rather than by inspection.

## An incomplete media pack reads as a model everywhere and dies in the text loader (live 2026-08-08)

The Video pane's Turbo-adapter fetch wrote to the DESTINATION root while the 4-bit H3 pack lived in another owned root, creating `models-dl/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit/` holding only config.json + tokenizer files + `turbo_lora.safetensors` (0.7 GB). That fragment carries a valid `minimax_h3` config.json, so: (1) server discovery registered it as a model, and first-wins across `--model-dir` roots SHADOWED both complete copies in later roots; (2) the app's `resolveModelDir` walks the same order and resolved it too, so a plain Generate — Turbo unchecked — loaded it; (3) `gen.detectModality` correctly declined it (missing `transformer.safetensors`) but the loader then fell through to the MLX TEXT path, globbed the lora as an LM, and hit `MISSING WEIGHT: model.embed_tokens.weight` → `log.err` + `unreachable` → dead process. Nothing heals the fragment: readiness is checked across ALL owned roots, the pack reads complete elsewhere, so no download is ever offered for the fragment dir.

The class is bigger than the fragment: EVERY H3/LTX pack download holds a valid config.json for the tens of minutes its big weights are still `.partial`, and an interrupted pull stays that way — a boot (or `/v1/models/rescan`) during that window registers a landmine.

Four-part fix, each half load-bearing:
- `model_discovery.requiredMediaMarker` is the ONE completeness table (`gen.requiredMarkerFor` DELEGATES — gen can import discovery, not the reverse); discovery skips marker-missing media dirs like any half-pulled download, and `probeModelDir` (register-by-path) answers `error.IncompleteMediaPack` → a named 400.
- The scheduler's preload refuses a media-typed dir that failed its marker BY NAME instead of falling through to the text loader — by-path loads never see discovery, and `unreachable` on a missing weight means one stale dir kills the server.
- `startTurboLora` downloads INTO the pack's resolved dir (`download(destDirOverride:)`) — an adapter's whole point is to sit beside weights that already exist.
- The Turbo toggle's off-flip cancels an in-flight fetch via `cancelTurboLora`, which is surgical (drops the one file's `.partial` + sidecar) and a NO-OP with nothing running — the generic `cancel(_:)` no-task fallback wipes the repo's whole download dir, which for an adapter fetch is a live 40-69 GB pack.

Guards: `discovery skips an incomplete media pack` + `probeModelDir refuses…` (model_discovery.zig), `incompleteMediaDir` (gen.zig), `tests/test_model_rescan.sh` (incl. server-survives-the-refused-load), `TurboLoraFetchTargetTests`.

## A directory-entry filter that skips symlinks measures an HF-cache model at ZERO (live 2026-08-09)

A first-launch user loaded `DeepSeek-V4-Flash-0731-iQ-MLX-3.3bpw` (121 GB) straight out of the HuggingFace hub cache and the machine went 34 GB into swap within a minute. The log had already said why nothing stopped it: `[preflight] weights ~0.00 GB, available 71.19 GB`. A hub-cache snapshot dir holds SYMLINKS into `../../blobs`; entry iteration reports them as `.sym_link`, and every server-side size sum filtered `if (entry.kind != .file) continue;` — so every weight file measured zero, and `memInsufficientForLoad` deliberately treats 0 as "unknown → allow" (a failed memory query must never block a load).

The class was every size sum, not just the preflight: `scheduler.modelDiskBytes` (the preflight), discovery's `bytes_on_disk` loops in `tryAddModel`/`sumComponentWeights`/`probeModelDir` (feeding `/v1/models`, the app's RAM column, and `scheduler.gateEstimateBytes` — the #126 admission/eviction gate, which also passed at ~0), `gen.sumSafetensorsIn` (media residency), and cli's `list` size loop + GGUF classification. The app was INNOCENT: `DownloadManager` already resolves symlinks for sizes and deliberately scans the hub cache as a read-only root — it is what hands the server these snapshot paths.

Fix: every entry filter also accepts `.sym_link` and stats through it (`Dir.statFile` FOLLOWS symlinks by default — all stat-RESULT checks already worked), plus a post-stat `st.kind == .file` check so a symlink-to-directory can't be summed; dangling links hit the existing `catch continue`. Zero now means a genuinely weightless dir, so the allow-on-zero semantic stands. Guards: `modelDiskBytes follows HF-cache symlinks` (scheduler.zig), `discovery measures a SYMLINKED (HF hub cache) model dir's real bytes` (model_discovery.zig).

Same report, second item: the CLI bound `0.0.0.0` by default, putting a first-launch server on whatever network the laptop joins. Kept (the app and Agent Sandbox need wide binds they set EXPLICITLY) but serve mode now warns when the bind is non-loopback and nobody chose it — no `--host`, no `--lan-share` (`server.shouldWarnOpenBind`); the default flips to 127.0.0.1 in a future release, at which point the helper's host check silences the warning without a code change.

## A combined Split regex hides the digit rule inside an alternation (muse "8 4" echo, 2026-08-10)

First live boot of Muse-Glimmer-30B echoed "What is 8 4 * 3 / 2?" for a prompt
containing "84" — the DSV4 echo-precision class on a new spelling. The
tokenizer ships the Llama-3-style COMBINED Split regex: case-classed word
branches with an attached `(?i:'s|'t|'re|'ve|'m|'ll|'d)` contraction group,
`\p{N}{1,3}`, ` ?[^\s\p{L}\p{N}]+[\r\n/]*`, and the usual whitespace tail —
all alternatives of ONE pattern. `digitGroupFromPreTokenizer`'s deliberate
exact-match (`regex == "\p{N}{1,3}"`) never fires on it, so the model was
served per-digit numbers.

The fix is a grammar, not a wider digit match: `Tokenizer.pretok_style =
.llama3` (selected by `llama3StyleFromSplitRegex` — contraction group AND the
{1,3} digit branch present) runs `llama3PreTokenize`: greedy upper-class run
then greedy lower-class run per word (reproduces the regex's backtracking for
disjoint ASCII classes; non-ASCII letters are treated caseless — both classes
— which coincides on every reachable match END except mid-word case
transitions in non-Latin cased scripts, a documented gap), attached (?i)
contractions, {1,3} digit groups, marks riding with punct, and `/` joining
the punct tail.

Two verification traps burned time:

- HF ENCODE output is not pre-token boundaries. "cat's" encodes as
  `cat` + `'s` and "don't" as one token — both have the contraction ATTACHED
  at the pre-token level; the visible splits are BPE-internal merges. Pin the
  pre-tokenizer against the raw regex `findall` (python `regex` module), and
  end-to-end ids against HF `tokenizers`.
- The live tell was subtle: generation was coherent (the model still solved
  the math) — only the echo spelled the number with spaces. Cross-check
  `/tokenize` vs HF at bring-up, per the standing rule; it found 8/8 diverse
  cases byte-identical after the fix.

## A config default only ONE family wants is a silent per-arch trap: muse rope base (first-turn repetition loops, 2026-08-11)

pi on Muse-Glimmer-30B looped in the thinking channel on its FIRST turn — restated the user's
request ~11 times until `[loop-stop]` cut it (54 cuts in one day's app-server log). The
discrimination matrix (`~/claude-tmp/muse-loop/matrix.tsv`, fixture = the exact captured pi
request) exonerated everything else one arm at a time: loops at temp 0 DETERMINISTICALLY (so
not sampling), with `--no-drafter` (PLD-only — not spec), on 4bit AND 8bit AND the upstream
bf16 under `--no-decode-attn-quant` (not quant), on the byte-identical HF-rendered prompt fed
raw through `/v1/completions` (not our chat render), with 0/1673 token-id diffs vs HF
tokenizers (not the pretokenizer). The decisive arm: the SAME upstream bf16 trunk under
mlx-lm (PR #1710, key-surgery remap) continued CLEAN where our runtime looped — greedy
divergence at ~token 13, ours doubling a phrase inside the request echo.

Root cause: every forward path picks the RoPE base as
`if (is_global) cfg.rope_theta else cfg.rope_local_base_freq`, and `rope_local_base_freq`
defaults to Gemma-3's 10000. Muse ships ONE theta (500000) for all roped layers — as
`text_config.rope_parameters.rope_theta`, which the generic parse lands in `rope_theta`
only — and its global layers are NoPE, so **every rotated layer ran at base 10000 instead of
500000**. Wrong positional geometry: short-range copying survives (the echo), mid-range
coherence collapses into restart loops. Fix: the muse arch block sets
`rope_local_base_freq = rope_theta` when the key is absent (model.zig). The bf16 greedy
continuation then tracks mlx-lm's (319-char shared prefix, wording-level bf16 noise after),
and the incident config went 0/6.

Class lessons: (1) a per-layer-TYPE config knob with a family-flavored default must be pinned
by the arch's config test for EVERY layer type the forward distinguishes — the muse test
asserted `rope_theta` and never `rope_local_base_freq`, exactly the value the sliding layers
read; (2) "coherent for N tokens then degenerates into loops, deterministic at greedy" is a
POSITIONAL-ENCODING symptom signature, not a sampling one; (3) an unmerged reference port
(mlx-lm PR) is still a usable oracle once weights + token ids are proven identical — behavior
divergence then isolates the runtime. Guards: the `rope_local_base_freq == 500000` assertion
in the muse config test (red on revert) + `tests/test_muse_repetition.sh` (replays the real
capture; exits 1 on any `finish_details.type == "repetition_loop"`).

## A converter that keeps the HF module layout renames the router (laguna oQ, issue #169)

`mlx-community/Laguna-S-2.1-oQ4e-fast` and `-oQ5e` died at load with
`MISSING WEIGHT: language_model.model.layers.1.mlp.gate.weight`. poolside's own
NVFP4-mlx build flattens the router module, so the [E, hidden] router matrix is
`mlp.gate.weight` and the aux-loss-free selection bias sits beside it as
`mlp.gate.e_score_correction_bias`. The oQ converter keeps HF's module tree, where
the router is a `proj` Linear INSIDE the gate module: same tensor, same dtype
(bf16, unquantized in both builds), at `mlp.gate.proj.weight`. The bias is a
buffer on the gate module in both layouts, so it never moves — only the
projection does. `lagunaRouterBase` probes for the flat name first (so the
poolside pack binds byte-identically) and falls back to the nested one, exactly
like `hy3ExpertContainer` resolves `mlp.experts` vs `mlp.switch_mlp`. Everything
else about the pack loads generically: the experts are mlx-lm's stacked
`mlp.switch_mlp.*`, and the mixed-bit affine quantization (4/5/6/8 bits, gs
64/128, 214 per-tensor overrides) solves from packed geometry per weight.

Two things about that pack that are NOT bugs: the `language_model.` weight
prefix is resolved from the checkpoint (`resolveWeightPrefix`, logged), and its
config declares YaRN `factor: 128` / `max_position_embeddings: 1048576` where
poolside declares 32 / 262144 — a 1M-context rope the loader honours as
declared (the computed mscale 1.4852 matches the field the pack ships).

Class lesson: a checkpoint's module tree is the CONVERTER's choice, not the
architecture's. Any name a converter could have flattened or nested gets a probe
the first time a second converter ships the arch — one probe, resolved once per
layer, flat name preferred so the reference build cannot shift.
Guard: transformer.zig `lagunaRouterBase resolves both the flat mlx-lm gate and
the HF-native gate.proj naming`.

## LTX-2.5 looked soft: three causes, one of them ours (2026-08-13)

Report: "videos generate and look generally good, just not at the bar I see in
other examples." Three separate things, measured rather than guessed.

**1. The canvas.** The clips were 768×512 / 704×480 — the app's LTX ladder
topped out there. LTX's own `PipelineParams` default is 1920×1088 (2.09 MP vs
0.39), and the published examples are 1080p+. Worse, the two-stage "Quality"
tiers denoise at HALF the requested size and upscale, so picking Quality at
768×512 ran the dev DiT at 384×256 — softer than the one-stage tier sitting
above it in the same menu. No amount of steps, guidance or quantization width
fixes a canvas the model was never asked to fill.

**2. The quantization.** Our pack was affine 4-bit group-64 on all 1632
transformer linears (the recipe inherited from the community 2.3 pack, where
q4 is documented as the "fits 16 GB" option and q8 is what the reference's own
examples use). Measured off the packs themselves — quantizer step over
per-group weight std, sampled across blocks 0/24/47:

| pack | median injected weight noise |
|---|---|
| q4 (affine g64) | **9.87%** (p90 ~12%) |
| q8 (same recipe) | **0.584%** |

~10% noise on every linear, compounded over 48 blocks × 8 distilled steps.
Same prompt, same seed, one-stage 97f at 768×512: the 4-bit fox has no visible
eye, blurred legs and flat fur; the 8-bit one has a defined muzzle, individual
fur strands and real legs. Cost: 175 s vs 169 s (+3.5%) — the DiT is
compute-bound at these token counts, so the wider weights are nearly free.
Upstream's own quants are int8-convrot / fp8 / nvfp4, and every community
*int4* pack uses ConvRot / W4A8 — plain affine int4 is nobody's shipping
recipe for this DiT.

Objective per-frame metrics could NOT settle it: a 10% weight perturbation in
an 8-step schedule moves the trajectory, so the two arms fork CONTENT
(luma correlation 0.44–0.55) and per-frame detail deltas ran from −20% to +40%
with a +5.9% clip mean. Same class as the spec-decode rule — when the arms
fork content, per-frame numbers are variance. The stills decided it.

**3. The decoder we didn't ship.** Upstream ships
`vae_diffusion_decoder.safetensors` (417M, "DiffVAE 1-step x0 decoder") and
decodes its demos with it; we used the plain conv decoder, and so does the MLX
reference. Ported 2026-08-13 — see below.

### The bug the 8-bit pack found

The first 8-bit load died at its first matmul:

```
MLX error: [quantized_matmul] The shapes of the weight and scales are
incompatible based on bits and group_size. w.shape() == (4096,1024) and
scales.shape() == (4096,64) with group_size=64 and bits=4
```

`ltx_video.zig` passed `mlx_optional_int.some(4)` literally at all three
quantized reads. The pack's config said `bits: 8`, the geometry said 8, and
nothing asked either. The error names the shapes, not the assumption, so it
reads like a corrupt download — which is what makes this class expensive:
every other quant path in the tree (mtp, dflash, drafter, deepseek_v4) already
solves `(bits, group_size)` from packed geometry via
`transformer.affineParamsFromGeometry`. The LTX engine now does too
(`quantGeom`), pinned by a source scan over its own call sites.

### Building the 8-bit pack

`scripts/quantize_ltx25.py <src> <dst> --bits 8`. Only the two transformers and
the text encoder need the bf16 source (~100 GB); the connector, both VAEs, the
vocoder and both upscalers are passthrough and byte-identical to the ones
already in the 4-bit pack (verified against upstream blob sizes), so they are
symlinked in rather than re-downloaded. Result: 20.60 GB per transformer,
12.65 GB encoder, ~59 GB total.

Watch the CLI: `hf download REPO --include a b c` eats `a` as the option's
value and warns `Ignoring --include since filenames have been explicitly set`
— which is how `transformer-dev.safetensors` silently didn't download on the
first pass. Pass filenames POSITIONALLY.


## The DiffVAE decoder: a faithful port that decoded to static (2026-08-13)

The third cause above, closed. `vae_diffusion_decoder.safetensors` is four
deterministic neighborhood-attention stages that upsample the latent into a
full-resolution context volume, then eight diffusion blocks that denoise
patchified pixel tokens against it. Architecture, geometry and the tiling plan
live in `src/ltx_diffvae.zig`; the fused NA Metal kernel in
`src/ltx_diffvae_kernel.zig`; the MLX pass in `src/ltx_diffvae_forward.zig`.

### The kernel came first, and its window semantics are the whole risk

NATTEN SHIFTS its window inward at a boundary — it does not clamp-and-mask.
Porting the clamp reading gives a decoder that looks right everywhere except a
`kernel/2`-wide frame around each edge of every tile, which is exactly the kind
of error a whole-frame cosine cannot see. `naWindowStart` is the one definition
and the kernel's parity test covers boundary-only volumes (`T=3` against
`K_t=3`, axes shorter than their kernel) as well as the interior. Verified
red-on-revert: swapping in a clamp turns both cases red.

One thread per (query, head), 64-float accumulator and an online softmax in
registers, K/V read THROUGH the cache. Staging K/V in threadgroup memory was
declined by arithmetic, not by taste: a (1,4,32) query tile under (3,7,7) needs
(3,10,38) keys = 145 KB of K+V, against Metal's 32 KB and the house occupancy
rule of ~10 KiB. Neighbouring queries share ~95% of their window, so threads
walk W first and a simdgroup's 32 windows overlap in L1.

### The failure: a faithful port of a guess

Per-stage parity against a PyTorch transcription of the reference came out at
cos 0.99997-0.99999 with `rms_ratio` 0.999 on every stage, the context volume,
the model prediction and the final pixels. Then the first real generation
decoded to **pure static**.

The port was right. The CONTRACT was wrong. `DiffusionVideoDecoder.__init__`
takes `model_output_type`, `default_num_inference_steps` and
`timestep_scale_multiplier` as constructor arguments, read by
`_build_diffusion_video_decoder` from a `vae` config section — and no LTX pack
ships one. The class defaults are v-prediction, 2 steps, timestep x1. Run that
way the model's output is small against the noise it is subtracted from, so
`x - 0.5*v` twice returns approximately the noise.

The transcription inherited the same defaults, so parity was green against a
reference that would also have produced static. **A faithful-port oracle proves
your port; it cannot prove the reference's own unstated arguments.** The tell
was in the file's own name all along: Lightricks call it a *1-step x0 decoder*.

Measured, one lever at a time, on the same prompt/seed — the metric is
`tests/lora_noise.py`'s mean absolute difference between horizontally adjacent
pixels, which is the static detector:

| arm | gradient | verdict |
|---|---|---|
| conv decoder (baseline) | 2.2-2.9 | picture |
| x0, 1 step, t x1000 | **2.4** | picture |
| x0, 2 steps, t x1000 | 17.7 | static |
| x0, 1 step, t x1 | 29.5 | static |
| v, 2 steps, t x1 (reference defaults) | 44.8 | static |

All three matter, and each is provable. `Sampler` owns them with a kill switch
each (`MLX_SERVE_DIFFVAE_{OUTPUT,STEPS,TSCALE}`), a unit test pins the values,
and `tests/test_video_gen.sh` asserts the gradient RELATIVELY against the conv
arm's own number on the same clip — the absolute value is a property of the
canvas and step count (13.9 on a 4-step 9-frame 384x256 render, 2.2 on a
25-frame 8-step one), so an absolute bar false-fails a small canvas. Measured
ratios: 1.04-1.09 healthy, 7.7-15.4 on each broken arm.

### What it buys, and what it costs

Same prompt, same seed, 97f at 768x512 one-stage: the DiffVAE resolves the
fox's ear, eye and individual fur strands where the conv decoder smears them,
and the background branches separate. Cost: the decode itself measures 21.2 s
and peaks at 20.7 GiB. End to end it does not show: 165.7 s (conv) against
163.8 s (diffusion), same session, same prompt and seed — the conv decoder over
97x512x768 is not free either, and the DiT dominates. Do not quote a per-clip
penalty from these two runs; quote the decode stage, which is what was
isolated.

Per-frame metrics do NOT settle this either — the two decoders share the latent
so the content does not fork, but detail deltas are still the wrong instrument
at this size. The stills decided it, as with the quant round above.

### Memory: a per-request decision, not a load-gate term

The diffusion stage carries context + x + q/k/v at 256 wide over the whole
tile: measured 10 KiB per stage-5 token. Stages 1-3 run on the FULL volume
(1.6M tokens at dim 512 even for a 1920x1088 clip); only stage 4 and the
diffusion stage are tiled, cut until one tile fits a budget that
`tileTokensForMemory` derives from currently-free RAM. Billing that into the
per-MODEL load gate would newly refuse packs that only ever use the conv
decoder — the same reason `ltxPeakBytes` carries no activation term. The
decoder FILE is already in the directory sum.

Tiles overlap by `max(stage-4 halo, diffusion halo)` and blend with trapezoid
ramps whose two sides sum to exactly 1 (half-offset: `(j+0.5)/L` against
`(L-j-0.5)/L`), so no weight buffer and no division. Whole-volume against
forced-tiled on the same seed: cos 0.9999972 overall AND on a band centred on
the seam — the seam crop is asserted separately because a whole-frame cosine
will happily hide a visible band down the middle.

Cutting the budget is a real trade: 3M tokens/tile = 21 s and 20.7 GiB, 1.2M =
46 s and 6.4 GiB (the tiles overlap, and stage 4 re-runs per tile).

## MiniMax Music 3 (`minimax_music3`, src/music3.zig) — the traps behind the port (2026-08-13)

Shipped against the diffusers `minimax-music3-integration` reference (dafe3733)
with per-component parity oracles (`tests/dump_music3_fixtures.py`, a
TRANSCRIPTION oracle in plain torch on our dequantized weights — the branch is
not installable standalone). Everything below is a place where the "obvious"
implementation is wrong and the oracle or a table test caught or pinned it.

### It looks like ACE-Step; almost every detail differs

Same endpoint, same modality slot, entirely different machine. The DiT's
timestep is a PREPENDED TOKEN removed after the blocks — `norm1`/`norm2` are
plain LayerNorm with weight AND bias, there is no `scale_shift_table` anywhere
in the checkpoint, so reaching for acestep's AdaLN helpers produces a model
that runs and generates plausible noise. The block FF is REVERSED SwiGLU:
`ff_in` is fused 2×8192 and the output is `first_chunk * silu(second_chunk)` —
the usual `silu(first) * second` also runs fine and also produces garbage. The
vocoder Snake is `x + (α+1e-9)⁻¹·sin²(αx)` — alpha only, NO beta, NO exp;
ACE-Step's has both. DiT RoPE rotates only the first 32 of 64 head dims at
theta 10000 — hardcoded in the module default, NOT the LLM's 1e6 and NOT in
config.json.

### The AR loop's three silent corrupters (reference `encoders.py:282`)

1. **Frame 0 is not emitted.** The loop runs `max_frames + 1` iterations; the
   first only advances state past `<|audio_start|>` — but its codes STILL feed
   back through `_embed_audio_frame`. The greedy-replay fixture therefore dumps
   frame 0's codes too, or the replay desyncs immediately.
2. **The vocab mask applies three times** in the reference because guiding two
   `-inf` logits makes NaN. Our engine sidesteps the whole class with a FINITE
   additive mask (-1e9): CFG arithmetic stays finite, one application, no
   nan_to_num needed.
3. **The top-k threshold comes from the CONDITIONAL branch**, strictly-less,
   ties kept — thresholding the guided distribution instead is a one-line slip
   the logits oracle sees as a different sampled trajectory.

The per-frame feedback embedding is `(semantic row + Σ residual bank rows) ×
8^-0.5` — the scale is load-bearing; the depth banks are codebook-major
(`code + (index-1)*1024`).

### Weight-norm fusion: the bias-length heuristic fails on square convs

The plan said "which conv kind a tensor is can be read off the bias length
(`bias.len == v.shape[1]` ⇒ transpose)". The vocoder's res-unit convs are
768×768 — `shape[0] == shape[1]`, the heuristic is ambiguous exactly there.
`conv_t1` is the only ConvTranspose1d in the stack, so the kind is keyed BY
NAME, with the load test pinning the fused/swapped shapes
(`[1536,768,16] → [768,16,1536]`).

### A bare f32 scalar promotes a bf16 stream (`scalarLike`)

`mlx_array_new_float` makes a STRONG f32 rank-0 array; `bf16_tensor * f32_scalar`
promotes the whole product to f32 and every downstream quantized matmul then
runs off-dtype. Every scalar op in music3.zig goes through `scalarLike`
(cast-to-operand-dtype) — the same rule MageFlow established.

### The prompt is a byte contract with cross-line regex semantics

`_clean_caption`'s markdown-rule regex `^\s*[-*_]{3,}\s*$` (MULTILINE) lets
`\s` cross newlines: a whitespace-only line IMMEDIATELY before a rule is
absorbed into the match, and the trailing `\s*$` greedily ends at the LAST
newline inside its run. A line-based reimplementation diverges on exactly
those captions. The Zig scanners replicate regex retry semantics too (a failed
`<|…|>` match advances ONE char, so `<|a<|b|>` → `<|ab`), and the whole
surface is table-pinned against outputs generated by executing the reference's
own Python regexes.

### Numbers (2026-08-13, M-series 128 GB, 8-bit pack)

AR stage 29.4 ms/frame ReleaseFast after the perf round (43.8 before it;
60.3 Debug pre-round; probe-armed ladder 30.2/31.4 across two reps, both-off
arm on the new build 43.7 ≈ the old-code baseline) — a 60 s song is ~45 s of
AR plus the DiT/vocoder stage. The 4-bit depth requant experiment measured
26.3 ms/frame (depth 11.7 → 8.0) at a real quality cost (replay agreement
0.942 → 0.913, ar_hiddens cos 0.9988) — env-only, default OFF. Per-stage attribution (MUSIC3_COST_PROBE laps, eval barriers symmetric
across arms): lm decode 17.5 ms in EVERY arm (batch-2 q8 8B weight-read
floor), depth 23.3 → 11.7 ms (KV cache), head+sample 2.6 → 0.5 ms (lm_head
prune), feed+book ~0.2. Parity after the round: prefill cos 0.99992, lm_head
logits 0.99998, greedy AR replay agreement 0.942 with prune AND depth-kv
engaged (identical to pre-round), ar_hiddens cos 0.9999, condition encoder
0.999999, DiT velocity 0.9990–0.9999, vocoder 1.000000. The QLin preresolve
is byte-identical (fixed-seed 8 s WAV vs the pre-round build). Multi-window
stitch (3 windows, 14 s) has no silent seams; the 86/258-latent crops tile
the song sample-exact (350 frames → exactly 14.00 s).

### A depth KV cache CAN beat a "weight-read floor" — when the naive path was never at it

The bandwidth model priced the depth decoder's 7 full re-forwards at ~11 ms
of weight reads (7 × 0.65 GB q8), so a KV cache — which still reads every
weight once per step — was estimated worth 1–3 ms. Measured: 23.3 ms naive,
11.7 ms cached. The naive path sat 2× ABOVE its own read floor: re-computed
positions widen every kernel, the per-step concat rebuild and per-forward
graph-build add CPU, and none of that appears in a bytes-read table. The
cache took the stage TO the floor — the floor is where a cache stops paying,
not a reason it can't start. The corollary cuts the other way too: the lm
decode stage measured 17.5 ms in every arm of the ladder — that one really
is the read floor, and no cache, prune, or handle preresolve moved it by a
tenth. Attribute with per-stage laps (MUSIC3_COST_PROBE; barriers symmetric
across arms so A/B deltas stay honest) before pricing a lever off bandwidth
arithmetic. Same round, same old trap: one zsh ladder arm passed two kill
switches via `env $extra` (no word-split), silently dropping the second —
caught immediately by the engagement line in the arm's own log, which is
exactly what that rule is for.

### LFM2-VL: a forward ARM with no vision splice answers "a completely black rectangle"

The tower was correct before the model saw an image. Position resample cos
0.999999 across six grids, tower hidden cos 0.99998, projected features cos
0.9997 against transformers' own `Siglip2VisionModel` on our pack's weights.
First live request: *"The image appears to be a completely black rectangle with
no visible content or details."*

Nothing errored. The encoder logged its boot line, the decode logged
`grid 26x38 (247 tokens)`, the encode logged `[1,247,2048]`, the prompt logged
`Inserted 247 image tokens`, `prompt_tokens` came back 265. Every number was
right. `forwardWith` fans out to one arm per architecture family and the vision
splice lives inside each arm — `forwardStandardWith` and `forwardMoeWith` have
it, `forwardHybridWith` never did, because until LFM2-VL no hybrid arch (lfm2,
nemotron_h, qwen3_next) had a tower. So 247 identical `<image>` placeholder
embeddings went through the trunk and the model described them faithfully.

The failure mode is the point: a missing splice presents as a bad IMAGE, not as
a bug. Every observable except the answer is indistinguishable from working, so
the guard cannot be an output assertion. It is a source scan —
`every generative forward arm splices vision embeddings` — over every
`forward*With` the dispatcher reaches, with the encoder-only arms
(`forwardBertWith`, `forwardGemma3EncoderWith`, which serve `/v1/embeddings` and
have no `image_token_id`) exempt BY NAME. A new arch is required to splice
rather than inheriting the hole.

### The reference for a position resample is `bilinear` + antialias, and mlx-vlm's bicubic is a SCALE error

SigLIP2-NaFlex stores one learned 16x16 position table and resamples it onto
each image's patch grid. transformers uses
`F.interpolate(mode="bilinear", align_corners=False, antialias=True)`. mlx-vlm
0.6.3 uses `bicubic_interpolate` there instead. Measured on the real table:
cos 0.99 and **rms_ratio 1.13** — the resampled table comes out 13% hot, which
is a scale error added to every patch embedding before block 0, not rounding.

That made mlx-vlm unusable as the oracle even though it is the obvious one
(installed, has `lfm2_vl`, and the packs are LiquidAI's own MLX conversions).
The oracle became torch + transformers run on the pack's weights instead, and
the Zig side reuses the Pillow-convention filter already in `qwen_vision.zig`
(`resampleWeightMatrix`, new `.bilinear` arm): its footprint widens by
`input/output` on downscale, which is precisely what torch's antialias does and
what a fixed 2-tap bilinear does not. Antialias is a no-op when upsampling, so
this only bites when a grid axis is SHORTER than 16 — which happens whenever an
extreme aspect ratio fits the token budget (a 200x50 source lands on 32x8).
Measured after: cos 0.999999 on every grid, both directions.

Two smaller traps in the same file. The patch feature order is `[py, px, c]`
with channel INNERMOST (`convert_image_to_patches`), where Qwen and Muse both
put channel outermost — same shapes, same token count, scrambled image, and a
cosine test on flat colour cannot see it. And the checkpoint carries TWO gelus:
the encoder MLP is `gelu_pytorch_tanh`, the projector is plain erf `gelu`.

### For a NaFlex model, tiling IS the resolution

`smart_resize` caps a single view at `max_image_tokens` (256 merged tokens =
a 32x32 patch grid). Anything past `max_image_tokens * 32² *
max_pixels_tolerance` is instead resized onto a `cols x rows` canvas of 512px
tiles, cut up, and encoded tile by tile with a thumbnail of the whole image
appended. A 1800x1400 photo is 2x3 tiles + a 252-token thumbnail = 1788 tokens
against 252 for the single view.

Skipping tiling is not a small quality delta. Same picture, same prompt, 15px
print: tiled reads `Serial: QX-88231-KLM`, `Batch code: 7741-ZZ`,
`Checksum: 0xBEEF42` exactly; the untiled geometry returns
`Social OS #0001 HLM`, `Batch code: 7941-02`, `Chemist: 100%F1F2` — confident,
well-formed, wrong. That is why `tests/test_lfm2_vision.sh` runs BOTH arms and
compares: this checkpoint reads large text fine at thumbnail resolution, so an
absolute bar on the tiled arm alone would pass a build where tiling silently
stopped happening.

Two implementation notes. The canvas is resampled ONCE and each tile is a
WINDOW into it (`buildPixelValuesRegion`) — resizing each tile region on its own
resamples different pixels and is not what `split_to_tiles` does. And the prompt
block is not a flat pad run: `lfm2ImageSegment` labels every tile
`<|img_row_R_col_C|>` and the thumbnail `<|img_thumbnail|>`, so the model knows
where each tile sits. A flat run still splices — the counts match — and hands
the model six unordered crops. Marker order has to match the order the encoder
concatenated the pieces, because the splice is positional. None of those ids
live in config.json; `populateLfm2ImageTokens` resolves them by string from the
tokenizer at load, the same way the user-turn marker is.
## A routing chain that returns f32 by design makes the CALLER responsible for the dtype (2026-08-11)

`groupLimitedRouting` keeps its expert weights in f32 on purpose, and says so: a bf16 round trip AFTER `routed_scaling_factor` (2.5) lands ~0.4% of 2.5 on every token, measured 2.6e-3 max error on the reference MoE block, 500x the tolerance the rest of the dsv4 port holds. Its doc comment ends "The caller casts to the activation dtype" — dsv4's caller does.

Wiring that chain into `moeMLP2` for `bailing_hybrid` inherited the obligation and not the cast. The f32 weights multiplied the routed expert sum, the sum went back into the residual stream, and from the first MoE layer (layer 1 of 24) the whole forward ran in f32 — which promotes every bf16 weight on read for the remaining 23 layers. Nothing is WRONG in the output; it is a uniform slowdown with no single slow op to find, which is precisely why `[dtype-trace]` exists. The whole diagnosis was one line:

```
[dtype-trace] moe: residual widened at layer 1: bfloat16 -> float32
```

The fix is a dtype-conditional cast at the ONE place the scores enter the expert combine, so every current and future caller of the grouped chain is covered rather than each remembering. Two rules restated: a helper that deliberately returns a wider dtype than the stream states the caller's obligation IN the doc comment (this one did), and the caller's failure to honor it is invisible except in the dtype trace — so read the trace on every new arch's first forward, not only when something looks slow.

## Chunked vision prefill: the splice row index is per-PREFILL, not per-forward (issue #197, 2026-08-16)

**Live failure.** A coding agent taking browser screenshots worked early in a session and died once the conversation passed ~10k tokens: any request with an image got a 400 from the memory guard (`prompt 39113 tokens needs ~63155MB … rejecting`) while the same conversation text-only prefilled fine. Vision prompts ran the WHOLE prompt as one forward — `generate.zig` set `default_chunk = loop_end` for `has_vision` — and the guard correctly billed that full width (`unchunked_prefill`), which at 39k tokens on a 48 GB Mac is more memory than the machine has. No flag could fix it: `--prefill-chunk` was bypassed on this path, and `--skip-mem-preflight` would admit it into the uncatchable Metal OOM.

**Why vision was unchunked.** `spliceVisionEmbeddings` finds each placeholder's source row via `cumsum(mask) - 1` over the CURRENT forward's tokens. Under chunking every chunk restarts at row 0 and re-splices the first image rows — coherent output, wrong image content. The failure is invisible from the outside: nothing errors, the model just answers about a corrupted image (in the quadrant repro, the bottom quadrants take the top quadrants' colors).

**Fix.** `ForwardCtx.vision_splice_offset` carries the placeholder rows earlier chunks consumed; the prefill loop counts them host-side per chunk (`countSpliceRows`, no GPU sync) and sets the offset before every prefill forward INCLUDING the final-span forward (an image ending there would otherwise re-splice from row 0). The splice adds `offset × hidden` to the cumsum indices. All forward arms are covered for free because they share `applyVisionEmbeddingsWith` (pinned by the existing every-arm-splices scan). The memory guard flips with it: `checkAttentionMemory` now bills the chunk, and both sides read `generate.visionPrefillUnchunked` so the guard and the loop cannot disagree. Kill switch `MLX_SERVE_VISION_CHUNKED=0` restores the whole-prompt forward AND the full-width bill together. SSM checkpointing stays off for vision (prefix reuse excludes image prompts — that exclusion is unchanged).

**Repro traps found while validating (both cost a debugging round):**

1. **`nextChunkEnd`'s TAIL_MERGE_MAX (512) silently un-chunks short repros.** A 90-token vision prompt at `MLX_SERVE_PREFILL_CHUNK=32` runs ONE chunk (the "remainder < 512 merges" rule absorbs everything), so a boundary-inside-the-image A/B that "passed" proved nothing — the trace line `chunks=1` is the tell. A real boundary repro needs prompt length > chunk + 512. This also means a 1-token chunk cannot exist, so the splice's `seq_len <= 1` decode gate can never skip a placeholder.
2. **Byte-equality between the chunked and unchunked arms is an unsound bar.** Chunk width changes GEMM shapes; a 4-bit checkpoint flips near-tie argmaxes across widths (measured text-only: chunk 512 vs 16384 diverge mid-answer on LFM2.5-VL-1.6B-4bit) — same class as MTP verify-width divergence. The A/B bar is the perceived-content invariant: both arms must name the SAME color set for a quadrant image (checkpoint-agnostic — gemma-4-qat calls dark yellow "brown" in BOTH arms), the off arm must read the top row at all, and the class bug shows as the chunked arm's set collapsing.

**Validated live** on LFM2-VL (hybrid arm, image split across chunks 1–3 at chunk 32: identical answers), gemma-4-12B-qat (standard arm + PLE), and Muse-Glimmer-30B-4bit (muse arm, window-attention ViT). Qwen3.5-VL (M-RoPE) rides the same shared splice but had no local checkpoint in the loop — M-RoPE indexes its position table by absolute offset and rebuilds cos/sin per forward, so a chunk boundary inside an image is positionally safe there by construction. Guards: `tests/test_vision_chunked_prefill.sh` (two-arm engagement + color-set invariant), `spliceVisionRows` chunk-equivalence + audio-ordering tests (transformer.zig), `countSpliceRows` + wiring scan (generate.zig), guard-billing scan (server.zig).

---

## A renamed vision tower, and a Conv3d read in the wrong axis order (Alis, 2026-08-19)

`avlp12/Qwen3.8-27B-Alis-MLX-*` stores the Qwen3-VL tower under `model.visual.*` where
every pack before it said `vision_tower.*` — 333 tensors, substructure identical, a
pure rename. `src/qwen_vision.zig` hardcoded `vision_tower.` in ~10 `must(...)` keys
plus `fmtLayer`, so the tower loaded nothing and the boot printed
`MISSING QWEN VISION WEIGHT: vision_tower.patch_embed.proj.weight` → "vision disabled".
It degrades gracefully, which is why it is easy to miss: the pack serves text
perfectly. `shouldKeepWeightKey` did not list the prefix either, so `--no-vision` could
not drop it and the load carried 0.92 GB it could never read. The prefix is now probed
once in `QwenVision.init` (`resolveVisionPrefix`, the `model.resolveWeightPrefix`
pattern) and threaded through every key including `fmtLayer`, and `model.visual.` joins
the `--no-vision` list. Muse / LFM2 / Gemma already probe; Qwen was the last hardcoded
one.

**The second half is the one that fails silently.** With the prefix fixed the tower
ran, the token accounting was right, and the model answered a red square with "black"
and a 256px red square with "a black and white vertical striped pattern" — the
signature of scrambled patch features, not of a bad checkpoint. `patch_embed.proj` is a
Conv3d whose STORED axis order is the converter's choice: mlx_lm writes channels LAST,
`[out, kT, ps, ps, Cin]` (mlx-community's Qwen3.5-0.8B: `(768, 2, 16, 16, 3)`), while a
straight torch export keeps channels FIRST, `[out, Cin, kT, ps, ps]` (Alis:
`(1152, 3, 2, 16, 16)`). Our loader permuted `(0,4,1,2,3)` unconditionally, which is
correct for the first spelling and a scramble for the second. `patchProjLayout` now
derives it from the shape against `qv_temporal_patch`/`qv_patch` (the two dims we
already parse), logs which spelling won, and refuses by name if neither matches; the
channels-first arm flattens with no transpose at all.

**Probe with three colours.** Red → "black", green → "black", blue → "blue" was the
broken state: one colour alone cannot tell "saw it" from "guessed", and blue was a
coin-flip pass. After the fix: red/green/blue all correct, and `--no-vision` drops the
capability from `/v1/models` entirely. Guards: the `patchProjLayout` + prefix unit
tests (both spellings, plus the `--no-vision` key gate in `shouldKeepWeightKey`) and
the live three-colour probe.

**The layout trap is a CLASS, found by fact-checking a table (2026-08-19).** Three
divergence rows in the Qwen3.8 shootout — `mlx-community/Qwen3.8-27B-8bit`,
`avlp12/…-Alis-MLX-8bit` and `Youssofal/…-MTPLX-Optimized-Quality` — agreed to 15
decimal places, which looks like a harness bug. It is not: a byte-level scan (dtype,
shape, size, head+tail SHA over every tensor, plus full SHA-256 over 20 whole tensors)
shows all 1847 shared language-model tensors are identical — same deterministic
`mlx_lm` 8-bit/gs-64 affine quantization of the same bf16 source. The scan surfaced
exactly ONE differing tensor between mlx-community and MTPLX:
`vision_tower.patch_embed.proj.weight`, `[1152, 3, 2, 16, 16]` in MTPLX against
`[1152, 2, 16, 16, 3]` in mlx-community's — and byte-identical to the Alis conv. So
MTPLX-Optimized-Quality shipped the same scrambled vision under the prefix we DID
support, where nothing pointed at it: the tower loaded, the token accounting was right,
and only the answers were wrong. Both packs name colours correctly on the fixed build.
Same scan explains the `ddalcu-8bit` row sitting 0.8 points apart: exactly one shared
tensor differs (`embed_tokens`, dense bf16 vs 8-bit) plus its MTP head.

---

## Standardizing our own Qwen3.8 packs: embeddings, head norms, DWQ (2026-08-19)

Three changes, each driven by a measurement rather than by "what everyone else does",
applied by `tests/restandardize_qwen38_pack.py` (source pack in, new pack out; every
transform is a no-op when the pack is already in the target state).

**1. The embedding is now quantized.** Our 4-bit and 8-bit packs kept `embed_tokens`
dense bf16 while every other vendor packs it; our own 6-bit and iQ packs already packed
it, so we were not even self-consistent. The comparison is unusually clean because
`ddalcu-4bit` and `mlx-community/Qwen3.8-27B-4bit` differ in **exactly one shared
tensor** (verified by a full-key scan: 2178 common keys, 1 mismatch), as do the two
8-bit packs. Dense vs quantized: 83.6% / KL 0.322 vs 84.0% / 0.342 at 4-bit, 95.5% /
0.0136 vs 96.3% / 0.0158 at 8-bit — top-1 nominally favours quantized, KL nominally
favours dense, all inside ±2se. Cost of dense: 1.8 GB at 4-bit, 1.2 GB at 8-bit. A
trap worth naming: `mx.quantize` on the raw stored bf16 bytes reproduces mlx_lm's
output **byte-for-byte**, but routing the same values through an f32 round trip does
not — and byte-identical codes are what makes a DWQ graft legal, so the repack
quantizes from the raw bytes.

**2. Head norms are published FOLDED.** Qwen's own release stores them delta-encoded
(zero-centered, the layer computes `1 + w`) and ddalcu packs were byte-faithful to it.
The fold is not a numerical improvement: our loader's `+1` (upcast f32, add, cast back
to bf16) reproduces the folded packs' stored values bit-for-bit on all 7 tensors, so
both conventions are the same bytes at serve time. It is an interop choice — a loader
that does not detect the convention multiplies by gamma ≈ 0.03 and drafts at ~0%
acceptance while looking healthy.

**Do not assume the convention fixes a downstream loader.** Simulating oMLX's actual
two stages against the real tensors — `sanitize`'s per-key `mean < 0.5 → +1`, then
`norm_repair`'s backbone-anchor pass — both delta (ddalcu) and folded (jundot) land
7/7 correct on Qwen3.8-27B. The pack that breaks there is `alis-4bit`, and not because
of its head: its BACKBONE `post_attention_layernorm` norms sit 0.8 higher than
mlx-community's, which moves the anchor and makes oMLX double-shift the head's copy —
the same false-repair class we fixed on our own side the same day.

**3. DWQ is graftable at 4-bit, and only at 4-bit.** `WaveCut/Qwen3.8-27B-MLX-4bit-DWQ`
is `mlx-community/Qwen3.8-27B-4bit` with learned dequantization: identical packed codes
(0 of 40 sampled differ), every `.scales` different, some `.biases`, and all the
un-quantized GDN `conv1d` weights. Our 4-bit codes are byte-identical to that same
base, so the learned tensors transfer as a file copy — 1044 tensors grafted, the tool
verifying the underlying codes match PER TENSOR and refusing where they do not.
Measured on our repacked pack: 83.6% → **84.2%** top-1, KL 0.322 → **0.289** (the best
4-bit KL in the shootout bar `jundot-oQ4e`), resident 17.67 → **15.85 GB**, decode and
MTP acceptance unchanged (36 engagement lines, 74–95% per-draft). At 6 or 8 bits the
scales are meaningless — the codes are a different alphabet — and the shapes still
MATCH (group size decides scale geometry, not width), so a wrong graft would load and
produce garbage rather than erroring.

## Reference audio is a conditioning SWAP, and FLF is a mask (#259 / #260, 2026-08-22)

**ACE-Step reference audio.** The Music pane was text-only while every 1.5 DiT (XL-Turbo included) takes "Refer audio". The engine already had the VAE encoder (`vaeEncodeMean`, oracle 6) and the condition encoder's timbre branch — which `generateWav` always fed the SILENCE latent. The reference (`conditioning_embed.infer_refer_latent`) feeds `refer_audio_acoustic_hidden_states_packed` = the clip's VAE latent, and the silence latent `[:, :750]` is exactly what it substitutes when no clip is given. So the feature is one slot with two sources, never both. Reference prep (`io_audio.process_reference_audio`): stereo 48 kHz, clamp [-1,1], TILE a clip shorter than 30 s, then take one 10 s segment from each third at a RANDOM offset and concatenate — always 30 s, and `tiled_encode` SAMPLES the posterior. Ours (`acestep.referenceWindow`) takes the start of each third and the MEAN so a seed is reproducible. The timbre window is the reference's own number (`timbre_fix_frame` 750 = 30 s x 25 Hz), which is why the engagement line reads `750 latent frames` whatever the clip length. Cover / Vocal2BGM landed the same day as a second story below. Parity: oracle 7 (`ACESTEP_REFCOND`, the oracle-6 audio's mean in the slot) measured 2026-08-22 on the fp32 HF checkpoints: whole pack cos 0.99995, timbre row cos 0.99989 — pin the ROW, a whole-tensor cosine over 86 rows cannot see one wrong row (the silence and reference fixtures agree byte-for-byte on the other 85). Live A/B on a real clip (same seed) moves the track but does not make it a cover — that is the feature's ceiling upstream too, one pooled token. Guards assert the log line — a silently ignored clip still yields a perfectly good track.

**LTX first/last frame (#260).** I2V pins one image as latent frame 0 via `cond_mask` + clean latent (`VideoConditionByLatentIndex`). The first cut of last-frame support did the same at `latent_idx = F_lat-1`, reasoning that the last latent frame IS the last pixel frame on the causal VAE. A user's FLF run (start == end image, 97 frames) came back with exactly the last 8 pixel frames decoding as noise and the target image sitting ~9th from last. The cause: the causal VAE encodes a lone image as a 1-pixel-frame "frame 0" latent; every later latent slot decodes as an 8-frame group, so a frame-0-style latent in slot `F_lat-1` is out-of-distribution for the decoder — and the DiT, pinned to it, converged the neighbouring group to the image instead. Upstream (`ltx_pipelines/utils/helpers.py`) only ever replaces at `frame_idx == 0`; any other index goes through `VideoConditionByKeyframeIndex`: HW extra tokens APPENDED to the sequence, clean, denoise mask 0, RoPE positions from `get_pixel_coords(causal_fix=False)` shifted by `frame_idx` with the temporal end narrowed to one frame (midpoint `(num_frames-1+0.5)/fps`), no attention mask for a single guide, trimmed before decode. That is what `keyframeMask`/`keyframePositions`/`buildKeyframeCond` do now on both pipelines (stage 1 trims before the upsampler, stage 2 before the VAE). Second deviation fixed in the same pass: we added the 2.5 `keyframes_abs_pos_embedding` to every `cond_mask == 0` token; the reference marks only GENERATED keyframe slots (`extend_keyframes_mask(marked=False)` for image guidance), so it is never added here (the checkpoint's row is rms 8e-4, so i2v output barely moves). The earlier claim that latent replacement was "Lightricks' older FLF recipe" was wrong: their 0.9.x pipeline also appended non-zero-frame conditioning tokens. `F_lat == 1` still refuses a last anchor (`error.KeyframeCanvasTooShort` → 400). Sidecar now records `last_frame:` too.

## An ACE-Step task is the context stream plus the instruction line (cover / vocal2bgm, 2026-08-22)

**What the modes are.** Read from `modeling_acestep_v15_xl_turbo.py`, not the README: `cover` and `complete` are the SAME 32-layer DiT forward text2music runs. Two things change. (1) The `context_latents` the DiT reads through `proj_in` — `[src_latents | chunk_mask]`, 64+64 channels — which text2music fills with the silence latent and all-ones. `complete` (the gradio "Add stem" button, the vocal2bgm use) puts the source clip's raw VAE latent there. `cover` puts the latent through `tokenize` (pad T to x5 with silence, `audio_acoustic_proj`, a CLS token prepended to every 5-frame window, two `AceStepEncoderLayer`s over each 6-token window, RMSNorm, CLS row → `ResidualFSQ`) and `detokenize` (embed, x5 + per-slot special tokens, two encoder layers over each 5-token window, RMSNorm, `proj_out` 2048→64) — 25 Hz "LM hints" that carry the melody and structure but not the timbre. (2) The instruction line of the SFT prompt: `TASK_INSTRUCTIONS` in `constants.py`, with `complete` taking the instrument list upper-cased and ` | `-joined (`task_utils.generate_instruction`). The target length IS the source length (`conditioning_target.py`). So `acestep.zig` grew a `Task` enum, `taskInstruction`, `contextLatents(src)`, and the cover helpers; the request's `duration_seconds` is ignored with a log line.

**The FSQ grid is not the one the FSQ paper describes, and not the one the class docstring describes either.** `vector-quantize-pytorch`'s `ResidualFSQ` (1.31.1, the version upstream pins to) SOFT-CLAMPS its input first — `soft_clamp_input_value` defaults from the levels to `c = L/(L-1)` per dim and `forward` applies `c * tanh(x / c)` — and then constructs its `FSQ` with `preserve_symmetry=True` and `bound_hard_clamp=True`, so the quantizer is `2/(L-1) * floor((L-1)(clamp(c*tanh(x/c),-1,1)+1)/2 + 0.5) - 1`. Neither the tanh `bound()` a bare `FSQ` uses (the plan was written from it) nor the hard clamp alone (my first reading of the source, which skipped the constructor's `default(...)`) — the second one survived two rounds of parity work because it is right on every value except those the soft clamp pulls across a floor boundary. Indices are `codes_to_indices` with the symmetric `_scale_and_shift` and basis `cumprod([1] ++ L[:-1])`. One quantizer → scale 1. The reference forces the quantizer to fp32; ours runs the whole tokenizer/detokenizer in f32 activations (they are tiny) with the two FSQ projections stored f32 in the pack (`DENSE_F32_PREFIXES`). Don't read a quantizer's formula off its `quantize` method: instantiate it and print its flags (`~/claude-tmp/acestep-cover/tok_diag.py` shape).

**The pooler's windows are a batch.** `encoderStack` was written for `[1,T,D]`; the pooler wants `(b t) p c` — every 5-frame window as its own sequence with RoPE positions 0..5 and a band mask that never bites. Feeding it `[T/5, 6, 2048]` gives exactly that for free (`mlx_fast_rope` restarts positions per batch row, the `[1,1,T,T]` mask broadcasts), so the stack only needed to take its weight map as a parameter (the FSQ weights live in their own map).

**A grid code is a near-tie, and the oracle bar says so — which is also what found the soft clamp.** Oracle 8 first demanded index EXACTNESS and failed 3/10 on the 8-bit pack and 2/10 on a dense-bf16 FSQ build. The bar became the MTP near-tie rule: the dump saves the reference's pre-quantization `z` (`acestep_tok_z.raw`), and a flipped digit is acquitted only where the reference put that dim within 0.05 cells of a floor boundary (`fsqBoundaryDistance`); a wrong RoPE, mask or window layout flips dims 0.3-0.5 cells deep and still fails. Under that bar ONE flip stayed deep at 0.073 cells across every arm — 8-bit weights, dense bf16, fp32 weights, f32 activations — with our `z` equal to the reference's to 2e-5. A value that is bit-right and a level that is wrong is the FORMULA: the soft clamp above. With it, dense-bf16 weights + f32 activations are EXACT (0/10, hints cos 1.000000); an 8-bit pooler moves `z` by up to 0.039 and flips 3/10 codes, all at ≤ 0.012 cells (acquitted by the bar). Detokenizer: cos 1.000000 + rms_ratio 1.0000 in f32. Cover DiT velocity: cos 0.99719.

**Acquitted is not harmless: the FSQ file ships DENSE.** The first Billie Jean cover "sounded nothing like the original". Per-stage oracles were all green, so oracle 11 runs the WHOLE cover pipeline on a real 15.8 s clip against the reference's own 8-step cover (`--cover-clip`): with the 8-bit FSQ file our final latents correlate 0.64 with the reference's; with dense bf16 FSQ weights 0.985 (the text2music e2e on the same 8-bit DiT reads 0.954, so 0.985 is the pack's ceiling). Three near-tie flips on a 2 s chirp become dozens over 79 codes, and every flipped code re-steers five frames of context. `FSQ_BITS = 16` in the converter whatever `--bits` says: 420 MB instead of 223, and it is the only setting under which cover is the reference's cover. Note also that the reference's own cover at strength 1.0 sits at latent cos 0.11 to its source (detok context 0.25) — a cover keeps melody/structure, not the latent, so never grade a cover by latent similarity to the source; grade it against the reference's cover.

**Weights ship as a SEPARATE file, and the engine stats for it.** The converter used to DROP `tokenizer.*`/`detokenizer.*` (60 tensors, 223 MB at 8-bit). Re-uploading a 5 GB pack to add them is not an answer, so they go to `fsq.safetensors` (`--fsq-only` writes just that file into an existing pack) and the engine loads it lazily on the first cover request (`ensureFsq`). `fsqAvailable` is a STAT, not a load-time flag: the app drops the file into a pack that predates it while the engine may already be resident (`DownloadManager.startPackFile`, the Turbo-adapter pattern — TEMPORARY migration code, to go once installs have re-downloaded), and the handler's named 400 for a missing file has to stop being true the moment the file lands. `complete` needs no new weights and must work on the OLD pack — that is why it went first.

**The source clip is encoded in windows.** `vaeEncodeMean` on 600 s would hold 128 channels x 28.8M samples f32 (15 GB) per activation. `vaeEncodeMeanChunked` encodes 30 s windows with a 64-frame halo, cores starting on 1920-sample (one-latent) boundaries so every strided conv lands where the untiled pass would — the decode's own overlap strategy, mirrored; the encoder's receptive field is ~12 frames. And a 10 s WAV's base64 is past ARG_MAX: the integration script passes clips by FILE (`mkreq`), which the #259 `ref_audio` arm had silently been failing on.

**Sampler extras.** `cover_strength < 1` builds a SECOND condition set (text2music instruction re-encoded while the text encoder is still resident — before the low-mem release —, same lyrics/timbre, silence context, its own cross K/V) and switches at `int(steps * strength)`, resetting nothing else (the reference rebuilds its cross KV cache; ours is per set). `cover_noise_strength` starts from `t*noise + (1-t)*RAW source latent` at the schedule entry nearest `1 - strength` (`nearestStep`, earlier step on ties like Python's `min`) and runs the remaining steps — the blend is the raw latent, never the hints. Both are integration-pinned by their log lines.


## Qwen3.8-Flash-Next (`qwen4_exp`) port (2026-08-26)

Qwen/Qwen3.8-Flash-Next arrived as `model_type: qwen4_exp` — the Qwen4 preview
arch, not a qwen3_5 pack: 125B-A6B GDN+MoE trunk (48 layers, 3 GDN : 1
attention) wrapped in FOUR hyper-connection residual streams, a 51B hashed
n-gram embedding injected at layer 1 (PLE), Qwen Sparse Attention (QSA:
indexer-selected 4-token blocks past 2048 tokens) on the attention layers, and a
1-layer MTP head. HF transformers main (`models/qwen4_exp`) is the trunk oracle
and ignores `mtp.*`; the MTP forward lives only in the vLLM (#53896) / SGLang
(#36497) PR branches (see memory notes; not served yet).

What the port reuses: the qwen3_5 split-projection GatedDeltaNet with the
`kda_sigmoid_out_gate` output gate, the qwen3_5 MoE (softmax top-10 + sigmoid
shared expert), `gatedFullAttnWith` (q-gate, QK norm, partial rotary 64/256,
KV cache). What is new: `hcRead`/`hcWrite` (grouped `(1+w)` norm per stream,
sigmoid mix, `2σ(inject/hc)` write gates), `pleForward` (host n-gram gather →
key/value → gated → dilated depthwise conv), `qsaMask` (bool mask threaded
through `ForwardCtx.qsa_mask`), and the final `hyper_connection_mixer` in place
of `model.norm`.

Traps that cost time:

- **HF `output_hidden_states` records layer INPUTS**: `hidden_states[0]` is the
  tiled embedding, `hidden_states[i]` the input of layer i. Comparing our layer
  output with `stream_i` read as "layer 0 already broken" while every sub-module
  matched at 0.9999.
- **QSA tail is per query**: the reference concatenates each query's own
  incomplete block (`visible[num_complete_blocks*ratio:]`), so token j is visible
  iff selected-block OR `j >= ratio*floor((p+1)/ratio)`. A global `kv % ratio`
  tail left rows 0..2 with NO visible tokens.
- **Indexer scores in f32, ties to the lower index**: relu zeroes most blocks
  early in a sequence; `torch.topk` keeps the lower index among equal scores and
  `argpartition` does not. Bit-faithful selection needs f32 scores plus a tiny
  index-descending bias.
- **The n-gram eos is the text config's**: `_shift_right_ignore_eos` resets on
  `config.eos_token_id[0]` (248044); the root config has none and
  `generation_config.json` lists several. Read it into `ngram_eos`.
- **A freed-but-not-nulled `aux_state`**: `resetCache` re-creates conv/ssm
  handles; the new field must be nulled with them or the warmup's second forward
  SIGBUSes on a dead handle.
- **Toy geometry has to clear the kernels' floors**: the GDN Metal kernel refuses
  dk 16 (`float state[n_per_t]` zero-length) — the tiny oracle uses dk/dv 64.

Bring-up flow: `dump_qwen4_exp_fixtures.py build` (random tiny model in the
real naming) → `convert_qwen38_flash_next.py --src` → `dump` (reference on OUR
dequantized weights, so the fixture measures the engine, not the quantizer) →
`QWEN4_TEST_MODEL=… QWEN4_FIXTURE=… zig build test -Dtest-filter="qwen4 fixture"`
(prints a per-layer bisect ladder + layer-0/1/3 sub-module cosines). Final:
full prefill cos 0.99996, 20/20 argmax; chunked prefill + stepwise decode past
the budget 6/6.

MTP (phase 2, same day): the head is one more hyper-connected QSA+MoE layer
fed `fc_hidden(rms_10240(pre-mixer stream)) + fc_embedding(rms_2560(embed))`
per stream, then its own mixer and the trunk lm_head — vLLM/SGLang agree, HF
ignores `mtp.*`, so the oracle is a torch rendering of that math on a synthetic
head (layer-3 tensors + random fc). Wired as `MtpHeadRef.qwen4` / `MtpCacheRef.qwen4`
(state on `Transformer.qwen4_mtp`, single-flight), so the whole EV controller /
acceptance / rollback loop is reused. Rollback extends `ssmRollbackFromCapture`:
the PLE layer captures its conv input + token history during a capturing
verify (`spec_ple_input/tokens`), attention layers truncate their positional
QSA key history, the head truncates its own KV + keys. Two traps: draft ids
arrive as lazy graphs of the sampler's dtype (cast + contiguous before the host
n-gram read — `TokenIdsUnreadable` otherwise), and the head's key row 0 sits at
position 1, so `qsaMask` takes a `pos_base` (explicit angle table for the pooled
keys; a scaled `fast_rope` cannot express a non-multiple base). Bars: head
parity cos 0.99998 vs the reference, verify→rollback(accept 1)→continue exact
(cos 1.00000) vs a fresh forward, server MTP-on == MTP-off greedy at 0 accepts.

Known v1 limits: serial (module-owned); PLD/DFlash off; batch 1;
QSA scores/mask are `[S, kv]`-sized transients per attention layer (long-context
prefill wants a smaller `--prefill-chunk`); MTP default-off (acceptance audit
pending, see engine-mlx.md round 3); pooled block keys recomputed per step.

Prefix cache (round 3, same day): `SSMCacheEntrySnapshot` now carries
`aux_state` (QSA raw-key history), `qsa_pooled`, `qsa_ratio` and `ple_prev`
(the PLE window tokens) beside conv/ssm; the disk tier writes them as
`l{d}.aux`/`l{d}.pooled`/`l{d}.ple` with `qsa_ratio` in the manifest, and the
`HotPrefixCache.shouldUse` refusal is gone. Live two-turn: reused 4385/4502,
then 4471/4502. The parity bar on this arch is not the hybrid 0.047 nats:
restored-vs-cold top-5 logprobs differ by 0.14–0.30 nats, while a COLD
chunk-1024 prefill vs a cold chunk-4096 one differs by 0.34 — the restore
resumes at a different chunk boundary, so the gap is the chunking-rounding
class, not a state bug. A restore that drifted past 0.34 would be one.

### Image input (vision tower + M-RoPE + QSA), same day

The checkpoint is `Qwen4ExpForConditionalGeneration`: a 27-layer Qwen3-VL-style
ViT (hidden 1152, 16 heads, patch 16, temporal 2, merge 2, learned 48x48 pos
table bilinear-resampled, 2D rope, `out_hidden_size` 2560, no deepstack) whose
weight names are Qwen3-VL's under `model.visual.` — `qwen_vision.zig` loads it
unchanged (prefix probe, `patchProjLayout` channels-first). Our pack had dropped
the tower; `convert_qwen38_flash_next.py --add-vision` fetches the ONE HF shard
that holds all 333 tensors, passes them through bf16 into
`model-vision.safetensors`, merges the index and restores `vision_config`
(idempotent, `--src` for a local tree). The tower is dense bf16 (~0.9 GB); the
Zig tower is dense-only anyway.

Three engine seams, none of them new code paths for text:

1. **Splice before the tile.** The reference `masked_scatter`s the image rows
   into the 2560-wide `inputs_embeds` and only THEN `repeat`s them into the 4
   hyper-connection streams. Our `forwardQwen4With` called
   `applyVisionEmbeddingsWith` after `mlx_tile` (a no-op on text) — the first
   real embedding would have met 10240-wide rows. Moved above the tile.
2. **One M-RoPE preamble.** `forwardMoeWith` built the per-chunk cos/sin inline;
   `beginMropeChunk`/`endMropeChunk` now serve both forwards, in the stream's
   dtype (bf16 served, f32 under the fixture oracle), so `gatedFullAttnWith`'s
   existing mrope branch just works on this arch.
3. **The QSA indexer reads the same table.** `Qwen4ExpTextQSAIndexer.forward`
   ropes q with `current_cos/sin` and the pooled block keys with
   `full_cos[group_starts]` — 3-D positions inside an image, `abs + delta` past
   the prompt, i.e. exactly `PositionContext.axisPosition`. `qsaMask` takes
   `ctx`: q via `applyMrope(mrope_cos_cur)` on prefill chunks, scalar
   `pos_base + offset + delta` past the table; new pooled rows via
   `mropeCosSinAt(start = pos_base + nb_cached*ratio, stride = ratio)` (pure
   host fill in `mrope.fillCosSin`, hermetically pinned as "strided rows ==
   every stride-th row"). Cached pooled rows keep the angles they were built
   with. The MTP head's ctx carries no table, so its call is unchanged — and
   image turns decline MTP (`specInitWiring` image_request) rather than draft
   with a mis-roped head.

Oracle: `dump_qwen4_exp_fixtures.py build|dump --vision` (tiny tower: depth 2,
hidden 64, 4x4 pos grid; one 6x8-patch image = 12 tokens in a 28-token prompt
crossing the tiny budget 8). `qwen4 fixture vision` asserts tower rows vs
`pooler_output` (0.99999), `getRopeIndex` == the reference 3-D table + delta,
our M-RoPE cos table vs the reference's (7e-8), full prefill cos/argmax +
per-layer streams, the layer-3 QSA mask EXACT (the direct guard for the
pooled-key angles: scalar-roping them red-probes at 32 mismatches), a two-chunk
prefill split INSIDE the image span (`vision_splice_offset` + pooled blocks
built across chunks; bar = our own full forward, 1.00000) and stepwise decode
past the table. Live arm: `test_qwen4_exp.sh [7]` + the `qwen4-exp` row of
`test_vision_all.sh`.

The oracle's own trap, found on the way: a random tiny MoE routes NEAR TIES
everywhere. The first vision run matched every stream to 0.99999 except ONE
row at layer 3 — a top-2-vs-3rd router gap of 1.1e-4 that our bf16-class
kernel noise (~0.5%/layer) flipped. Sharpening the router does not help: the
tie rate is scale-invariant (noise scales with the logits), 224 routing
decisions always carry one, and no seed in 3..120 clears a 2% floor. Then the
same thing one level down: layer 7 row 20's second and third QSA block scores
were BOTH exactly 0.0 (relu), so a near-zero q·k flipping sign selects a
different block — a tie the lower-index rule cannot resolve. Design that
survives: every expert active in the tiny config (`num_experts_per_tok =
num_experts` — a wrong expert mapping still shows, a pick cannot flip), and the
dump records the reference's OWN margins per row (`qsa_gap` = relative margin
between the last selected and first rejected block score, min over QSA layers;
`logit_margin` = top-2 logit gap) so `Qwen4Ties` acquits rows under 3% by the
reference's numbers, never ours (28-row vision prompt: 8 QSA + 5 argmax
acquittals, 17 decisive rows; 20-row text prompt: 2 + 0). Same machinery on
the text fixture, which had been passing by luck.

The live sanity then caught a SHARED-path hole: a 4-frame clip through the
video path logged `Decoded 4 frames → qwen video grid_t=2 … 690 tokens`,
`Inserted 0 image + 690 video … soft tokens`, `M-RoPE: 0 images, 1 videos` —
every engagement line green — and the model described "a person in a white
shirt standing in front of a plain background". `spliceVisionRows` built its
mask from `image_token_id | audio_token_id`; `<|video_pad|>` is a third id, so
the 690 encoded rows were never scattered and the trunk read 690 raw pad
embeddings. transformer.zig had never mentioned `video_token_id`. The qwen3_5
video e2e (`test_qwen_video_input.sh`) asks for >= 2 of {house, robot,
street/sign/road, app/phone} — words a model hallucinates freely — so it
passed. Fix: the mask and `countSpliceRows` (chunked-prefill row accounting)
take `video_token_id`; guard = `spliceVisionRows: video placeholders are
scattered like image rows` (red on the old mask). Rule: an engagement line
proves the PATH ran, only content the pixels alone could supply proves the
rows reached the prompt.


## A k < E tiny MoE fixture covers selection only through the MTP head (2026-08-26)

`dump_qwen4_exp_fixtures.py build --topk 2` builds the tiny qwen4_exp with 2 of 8 experts. The reference is constructed from `TINY`, so `dump` now reads `num_experts_per_tok` back from the checkpoint (the first dump ran k = 8 against our k = 2 pack: MTP head cos 0.894 with ZERO acquittals — a config mismatch looks exactly like a kernel bug). With k honored, the head matched at cos 1.0000 on every decisive row except one at a 3.5% softmax margin, hence `QWEN4_TIE_REL_ROUTE` = 5% (the router reads a bf16 residual; the QSA scores are f32 and keep 3%). Margins come from forward hooks on every `Qwen4ExpTextTopKRouter` (`route_gap` = min over MoE layers of `(p[k-1] - p[k]) / p[0]`; `mtp_route_gap` from the head's own router).

The trunk cannot be covered this way on a random model: 8 MoE layers x k = 2 leaves 18 of 20 prefill rows and all 6 decode rows under the bar, so the `decisive >= T/2` floor only applies when no routing table is present. The head is one MoE layer and keeps 12/19 decisive rows — that is where top-k selection is hermetically pinned. The k = E fixture remains the CI oracle; the k = 2 pack lives in `~/claude-tmp/qwen4-tiny-k2/`.

Also from this round: the video e2e bar (`tests/test_qwen_video_input.sh`) now demands frame 3's sign text; the informational bf16 logits readout in `qwen4 fixture bf16` reads cos 0.9997 / argmax 20/20 on the k = E pack.

## A latent-to-RGB preview is a FIT, and a hue wheel is not one (#208, PR #308 review, 2026-08-30)

The first cut of `preview.zig` projected latent channel `i` to a colour by walking the hue circle at the golden angle (`cos(i * 2.4)`, `cos(i * 2.4 - 2π/3)`, …), then mean/std-stretched the result per frame so it filled the 0–255 range. Every part of that is defensible in isolation and the whole thing is wrong: the golden angle spreads the channels EVENLY around the wheel, which is precisely the assumption a fit refutes — on MiniMax-H3's published table the largest channel weight is 30x the smallest, so a projection that treats all 24 equally is dominated by channels the decoder barely reads. The output was a colourful blob whose only relationship to the video was that it changed when the video did. The per-frame stretch made it worse in a way that is invisible in a single still: it renormalizes against each step's own statistics, so brightness and contrast jump between consecutive previews even where the latent is converging.

The shell guard passed throughout, because it asserted `FF D8` and a non-zero length. That is the class: an image codec test cannot tell a picture from noise, and there was nothing else.

What ships instead is the projection ComfyUI publishes for each of the two latent spaces — `LTXAV` (128 channels, and NOT `LTXV`, which is LTX-0.9's table at the same geometry over a different latent space) and `MiniMaxH3Video` (24). ComfyUI is GPL-3.0 and this repo is MIT, so its code is a SPECIFICATION here, the same standing every other ported reference has: `tests/dump_latent_rgb_factors.py` imports the reference, EXECUTES `Latent2RGBPreviewer` on a pinned latent, and emits both the coefficient tables (`src/latent_rgb.zig`) and the reference's own outputs (`src/fixtures/latent_rgb.json`). Nothing is transcribed by hand.

Three things the oracle needed before it could fail for the right reason:

- **The stub has to explode, but not everywhere.** ComfyUI's `latent_formats` reaches into `comfy.model_management` and `comfy.taesd`, neither of which is part of the projection. An `_Exploding` module that raises on ANY attribute access fails during import; registering real `types.ModuleType` shells whose *members* explode lets the projection bind and still makes a golden value derived from a stubbed path impossible.
- **A case must have more pixels than the map has channels.** With 128 channels and 64 pixels a wrong coefficient can be cancelled by the others across the case, so the LTX case is generated large enough to over-determine the fit.
- **`preview_to_image` clamps and TRUNCATES.** `(x + 1) / 2` clamped, `* 255`, then torch's `.to(uint8)`, which truncates. Rounding instead disagrees with the reference on roughly half of all pixels — which is what the mutation check now proves, since flipping `toU8` back to `+ 0.5` fails the fixture test.

`pinnedLatent` is the shared input: an integer LCG, then `(x >> 8) / 2^24 * 3 - 1.5` in f32 only. A 24-bit numerator over 2^24 is exact in single precision, so Python and Zig land on identical bits and the float comparison is a real tolerance (1e-5, for BLAS reduction order) rather than a fudge factor hiding a disagreement.

**An oracle proves fidelity to the reference, never resemblance to the video.** Byte-equality with ComfyUI would be just as green if ComfyUI's own table were a hue wheel, so the fixture test cannot answer the review's actual question. The perceptual bar is a second, live pair of tests — `minimax h3 vae live: the Latent2RGB preview resembles the decoded frame` and its LTX twin — which encode a real image with the real VAE, project the resulting latent through the shipping map, box-average the real decode down to the latent grid and take a centered Pearson correlation. Three details are load-bearing:

- **The control arm is the retired projection.** A bar that only says "correlation > 0.6" is worthless until something fails it, so both tests also score `preview.goldenAngleControlMap(C)` — the golden-angle table, through the same code path — and require the fit to beat it by 0.3. Measured 2026-08-30: H3 (FL2VA 8-bit, 256 px → a 24x16x16 latent) **fit 0.740, control 0.029**; LTX (2.5 8-bit VAE pair, 512 px → 128x16x16) **fit 0.898, control 0.196**. The absolute floors are sanity bounds; the control MARGIN is the discriminator, and it lands near 0.7 against a bar of 0.3.
- **Correlation is CENTERED, which is why the control needs no dead code.** The retired projection's per-frame mean/std stretch is an affine map, and Pearson correlation is invariant to it, so scoring a golden-angle `Map` through today's `latentSliceToRgb` measures exactly what the retired code would have measured. Nothing is resurrected to be tested.
- **The decode is AVERAGED down, never point-sampled.** A preview pixel is one latent position covering 16x16 (H3) or 32x32 (LTX) pixels; sampling one texel of that block scores the fit against whichever detail the sampler landed on.
- **Content chosen for "the compression can carry it" made the bar meaningless, and only the control arm revealed it.** The first version fed smooth per-channel ramps, and on LTX the golden-angle control scored chroma **0.79** against the fit's 0.94 — a 0.15 margin. A monotone gradient is nearly rank-1, and summing 128 channels at uniform magnitude reproduces one about as well as a fit does, so that arm was passing on the CONTENT rather than on being right. `preview.perceptualTestFrame` now paints one pseudo-random colour per LATENT CELL — 3 independent predictions per cell — and the control falls to 0.26 (LTX) / 0.04 (H3). Note which number moved: the fit's own score barely changed, so nothing except the control could have caught it.
- **Full RGB is mostly luminance, so chroma is a SEPARATE assertion** (`rgbChromaCorrelation`: each channel minus that pixel's own mean, needing no centering because a pixel's residuals sum to zero). Any sum of latent channels tracks brightness; WHICH COLOUR is what a fit knows and a hue wheel cannot. It is also where the fit scores highest — H3 chroma 0.880 vs 0.740 full-RGB, the gap being a brightness offset the preview never promised to match.

Each test SKIPS on a missing pack and needs only the VAE, not the transformer: `MINIMAX_H3_MODEL=<pack>` (probes `video_vae.safetensors`) and `LTX_TEST_MODEL=<dir with vae_encoder + vae_decoder>` — 1.45 GB of the LTX pack is enough to run it. `tests/test_video_preview.sh` drives both and FAILS when a named pack produced no correlation line, because in a test summary a skipped arm looks exactly like a passing one.

The second half of the review was cost. The app sent `preview: true` on every video generation, so every user paid an x0 solve plus a host copy per step whether or not anyone was watching — and on LTX the copy was the WHOLE `[1,128,F,H,W]` volume, ~25 MB of f32 per step for a 480p 121-frame clip when the strip shows one 0.8 MB frame. The temporal pick moved onto the GPU (`unpatchifyVideoFrames` gathers the wanted frames before the transpose, so the materialized array is n frames wide, not F — H3 already sliced per frame), and the pane owns a toggle that defaults OFF.
