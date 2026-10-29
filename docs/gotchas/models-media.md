# Model loading, configs, converters and media parity: gotchas

One story per gotcha: the defect, the cause, the fix, the guard. The rules live in the root `AGENTS.md` (`## Rules`)
and in each plugin's `AGENTS.md` (`src/plugins/<name>/`); what is still open lives in `docs/known-issues.md`. An env
lever a story names that `src/env.zig` no longer lists was retired once its arm was decided.

## Configs, prefixes, layouts

### A per-layer table and the Gemma vision projection ignored a per-tensor group size (Gemma 4 E4B mixed 4/6-bit)

`zecanard/gemma-4-E4B-it-ultra-uncensored-heretic-MLX-4bit-mixed_4_6` declares gs 32 model-wide but stores every language-model tensor and both media projections at gs 64. Warmup died with `Cannot reshape array of size 5376 into shape (1,1,42,256)`, every text turn 500'd and every image turn 500'd. Cause: the plugin loaded the per-layer embedding table with width hint 0, and the vision `quantLinear` took the config gs, so each fell back to the model-wide scheme (a 6-bit gs-64 table read as gs 32 gives half-width rows; the 4-bit projection read as 8-bit gs 32). Fix: the table is loaded at its real row width (`layers x ple_dim`) and `quantLinear` solves each weight from its input width (`quantParamsFromGeometry`), like the other towers. Guard: `quantLinear reads a projection at its own group size inside a model with another one`, `tests/test_gemma4_mixed_quant.sh`.

### Read `text_config` first, then the root, per field
Defect: every unloaded multimodal model advertised `hidden_size: 0` and `is_moe: false` (a 128-expert Gemma read as
dense, which gates MTP/drafter defaults). Cause: `discovery.parseStubMeta` read the root only. Fix: the nested block
first, the root per field. Guard: the multimodal + nested-MoE + root-fallback cases of the `parseStubMeta` test.

### Fields a minimal `text_config` omits fell back to another family's defaults
Defect: `gemma-3-4b-it-4bit` crashed at warmup reshaping into 16 heads. Cause: `ModelConfig` defaults are the Gemma
12b shape; the pack omits head counts and leans on HF's Gemma3 defaults (8/4/256). Fix: an arm fills HF per-arch
defaults explicitly for omitted fields. Tell: `[reshape] Cannot reshape array of size N into shape (1,1,H,D)` at
warmup. Guard: the gemma3 head-count tests.

### `*_text` siblings and the flat weight prefix
Defect: `gemma3_text` packs loaded incoherent or crashed with `MISSING WEIGHT: lm_head.weight`. Cause: the tag fell to
the llama arm; a flat pack keeps weights under `model.*` and is always tied. Fix: accept the sibling, collapse to the
base type, pick the prefix by `text_config` presence, force tied embeddings for Gemma, and add the tag to both
visibility allowlists (discovery and the app). Guard: routing + prefix tests, `testGemma3TextIsSupportedArchitecture`.

### The weight prefix was a config guess (LFM2.5)
Defect: `LFM2.5-2.6B-8bit` died with `MISSING WEIGHT: model.embed_tokens.weight`; all 600 weights lived under
`language_model.model.*`. Cause: the arm picked the prefix from `text_config` presence, and this config has none (only
an empty `vision_config`). Fix: `resolveWeightPrefix` re-points the config when the configured prefix holds nothing
and the other nesting has weights (only between `model` and `language_model.model`, `.`-boundary match). When
converters disagree about a name and config.json cannot settle it, probe the tensors.

### A converter's container name: `mlp.experts` vs `mlp.switch_mlp` (hy_v3)
Defect: every mlx-lm-converted Hy3 pack died with `MISSING WEIGHT: model.layers.1.mlp.experts.gate_proj.weight`.
Cause: the binding hardcoded one converter's name. Fix: `hy3ExpertContainer` probes per layer. Guard:
`hy3ExpertContainer resolves both mlx-lm switch_mlp and ox-ox experts naming`.

### A converter that keeps the HF module layout renames the router (Laguna oQ, #169)
Defect: oQ Laguna packs died with `MISSING WEIGHT: …mlp.gate.weight`. Cause: poolside's build flattens the router to
`mlp.gate.weight`; the oQ converter keeps HF's nested `mlp.gate.proj.weight` (the bias buffer never moves). Fix:
`lagunaRouterBase` probes the flat name first, then the nested one. A module tree is the converter's choice. Guard:
the router-base test.

### `chat_template` has two legal shapes, and reading the wrong one panicked
Defect: Mistral-7B-Instruct-v0.3 could not load: `access of union field 'string' while field 'array' is active`.
Cause: HF allows `chat_template` as a string or a list of `{name, template}`; Mistral v0.3 ships the list. Fix:
`chatTemplateFromValue` reads both and picks `default` (as transformers does), else the first usable entry, else null.
`.string` on an unchecked JSON value is a crash, not a miss. Guard: `chat_template accepts HF's
list-of-named-templates shape`.

### Gemma `<end_of_turn>` leaked when the config declared a scalar eos
Defect: a minimal Gemma pack with `eos_token_id: 1` never stopped at turn end and leaked `<end_of_turn>`. Cause: id
106 was added only when the config gave no eos. Fix: `ensureGemmaTerminators` merges 1 and 106 additively;
`generation_config.json`'s `eos_token_id` joins the stop set (`mergeEosTokens`). Guard: the scalar-eos and idempotence
tests.

### Llama 3.x answered empty past 1k tokens
Defect: Llama-3.2-3B answered a 1k prompt and returned whitespace at 1.8k and 3.8k, every build back to 26.9.2. Cause:
`has_sliding_window` defaulted to Gemma's layout (window 1024) and the llama branch never cleared it; `rope_type:
"llama3"` was read as nothing, so low frequencies ran 32x fast. Fix: the llama family is full attention in every
layer, as in mlx-lm; `Llama3Rope` feeds HF's frequencies; llama3 packs get their own SSD root. Every short check had
missed it. Guard: the llama3 rope parse test, `computeLlama3Freqs` goldens, smoke `long prompt`.

### A config default only one family wants: Muse's rope base
Defect: pi on Muse looped in its thinking channel on the first turn (54 cuts a day), deterministically at greedy, on
every quant, on HF-identical prompt and ids; mlx-lm continued clean on the same weights. Cause: sliding layers read
`rope_local_base_freq`, defaulting to Gemma-3's 10000; Muse ships one theta (500000). Fix: the muse arch sets
`rope_local_base_freq = rope_theta` when absent. "Coherent for N tokens then loops, deterministic at greedy" is a
positional-encoding signature. Guard: the config test asserts both values; `tests/test_muse_repetition.sh` replays the
capture.

### Laguna YaRN: the checkpoint's `attention_factor` is not the trained mscale
Defect: Laguna-XS ships `attention_factor: 1.0` and would have run 10 of 40 layers with unscaled YaRN RoPE (S matched
by luck). Cause: poolside's own implementations ignore the field and compute `0.1·ln(factor) + 1`. Fix: the laguna arm
computes it. When a reference ignores a field, the field is not the truth for that arch. Guard: `laguna YaRN mscale is
COMPUTED, never read from attention_factor`.

### Malformed model configs reached unchecked reads and casts
Defect: wrong-typed or out-of-range config fields hit illegal behaviour in ReleaseFast. Fix: checked readers: optional
nulls are absent, explicit disables (`sliding_window: null`, negative BOS sentinels) keep their meaning, narrowing and
derived dimensions return named errors; discovery checks root types and metadata ranges. Tensor shapes need their own
validation. Guard: `parseConfigFromJson rejects invalid field types and ranges`, `preserves optional nulls and skipped
fields`. Same class: qwen4_exp's n-gram bounds sat behind a `std.debug.assert` (compiled out of ReleaseFast) and wrote
past fixed arrays; `validateQwen4Config` and `NgramTable.parse` now refuse by name, and a PLE the layer loop never
installed is a load error.

### An affine bit-width MLX has no kernels for killed the server at warmup
Defect: `Bonsai-27B-mlx-1bit` loaded everything, then died in warmup with `Unable to load kernel
affine_dequantize_…_b_1`. Fix: affine bits outside {2,3,4,5,6,8} are refused at config parse
(`error.UnsupportedQuantBits`); `affineParamsFromGeometry` only returns supported widths. Extend both together when
MLX adds a width. Guard: `parseConfigFromJson rejects affine bits MLX has no kernels for`.

### Quant resolves per weight (nvfp4 / mxfp4 / mxfp8)
Defect: mixed QAT packs override some layers to affine 8-bit inside an fp8-mode config; a single per-config rule
mis-read them. Fix: `computeQuantParams` per weight: uint8 scales take the config's fp mode, float scales are affine
with bits and group size solved from geometry; `.biases` mandatory under affine, optional otherwise. Guard: the
quant-resolution tests.

### An fp override inside the fp family is invisible to a scales-dtype rule
Defect: `LFM2.5-2.6B-MLX/mxfp4` killed the server on its first request (`[dequantize] Shape of scales does not
match`). Cause: its embedding overrides to mxfp8 inside an mxfp4 model; both store uint8 E8M0 scales, so the
scales-dtype rule resolved the embedding at 4 bits. Fix: `fpParamsFromGeometry` solves bits and group size from the
packed geometry, returns null when it agrees with the config, and names a mode only when the pair matches one format's
block size. Guard: the `computeQuantParams` fp-override test.

### The hybrid layer-init path demanded scales
Defect: `LFM2.5-2.6B-bf16` died on `MISSING WEIGHT: …conv.in_proj.scales`. Cause: the hybrid layer init fetched scales
with the mandatory getter; the other paths already treated scales as optional per tensor. Fix: `getLayerScaleOpt`, and
every contracted dense weight gets `maybeTransposeForBf16`, never the depthwise conv kernel or SSM state (`A_log`,
`D`, `dt_bias`), which would silently compute the wrong thing. Guard: a synthetic dense 2-layer lfm2 through the
hybrid init.

### ModelOpt's NVFP4 global scale is not MLX's
Defect: NVIDIA's Flash Next FP4 pack read every image as "a blank white canvas". Cause: ModelOpt stores
`weight_scale_2 = amax / (448·6)`; passed raw as MLX's `global_scale`, every tower weight came out 2688x too small.
Fix: `modelopt_pack.dequantDense` multiplies by 448·6 first. Guard: `ModelOpt NVFP4 dense dequant`.

### A GGUF folder is a shelf of models, not a model
Defect: the app could offer one quant per GGUF repo; a downloaded quant stopped resolving after relaunch; deleting or
cancelling one quant took its siblings; a speech tokenizer GGUF could load as the LLM. Fix: one entry per quant
(`id#<file>`), `existingModelDir` accepts a servable `.gguf`, per-quant delete and cancel, and sidecar classifiers
(`isGgufSidecar` / `discovery.isGgufSidecarBasename`) kept in sync. Guard: `GgufQuantTests`, `isGgufSidecarBasename`.

### A configless repo shape has to be taught to every "is this a model?" answer
Defect: every MageFlow checkpoint (no root config.json) and the mflux FLUX.2 klein 9B (no index either) were invisible
to `list`, `/v1/models` and the app, though `--model` loaded them. Fix: one predicate per shape, the media plugin's
`claims` (reading a bounded header prefix), read by every classifier through `served.claimIn`; the
app's bundle markers declare `hasRootConfig: false`. Signal on weight names, not directory shapes. Guard: the
discovery and `isModelDir` tests for each shape.

### An incomplete media pack read as a model and died in the text loader
Defect: a Turbo-adapter fetch wrote config.json + tokenizer + LoRA into a fresh dir, which shadowed the complete pack
in a later root; loading it fell through to the text loader and `unreachable` killed the process. Every H3/LTX
download holds a valid config.json while its weights are still `.partial`. Fix: one completeness table
(the plugin's `required_file`, read through `discovery.requiredMediaMarker`) skips incomplete packs; `probeModelDir` answers
`error.IncompleteMediaPack`; the preload refuses a media dir that failed its marker by name; adapters download into
the pack's resolved dir and cancel surgically. Guard: `tests/test_model_rescan.sh`, `TurboLoraFetchTargetTests`.

### A directory filter that skips symlinks measured an HF-cache model at zero
Defect: a 121 GB model loaded from the HF hub cache sent the Mac 34 GB into swap: `[preflight] weights ~0.00 GB`.
Cause: snapshot dirs hold symlinks and every size sum skipped non-`.file` entries; 0 means "unknown, allow". Fix:
every entry filter accepts `.sym_link` and stats through it, then checks the result is a file. Same report: an open
default bind now warns (`shouldWarnOpenBind`). Guard: `modelDiskBytes follows HF-cache symlinks`, the discovery
symlink test.

### The stub `model_type` is a modality static
Defect: H3's media preflight billed the 64.5 GB file sum while logging a staged bill. Cause: a media stub's config
carries the modality (`"AudioVideo"`), never the backend type. Fix: re-peek the directory's real type
(`gen.peekModelType`). A log line's presence proved nothing: `tests/test_minimax_h3.sh` asserts the printed number
sits in the staged band.

### A companion pack under its base's `model_type` (Qwen3-ForcedAligner)
Defect: the forced aligner was listed as a transcription model, loaded with no tower and became the headless default
chat model, as the ASR itself did. Cause: both roots say `qwen3_asr`; only `thinker_config.model_type` tells them
apart; any qwen3 trunk behind an audio tower counted as chat-capable. Fix: discovery reports `qwen3_forced_aligner`
(capability `alignment`, a named 400 on text routes), and an audio-tower trunk is never chat-capable. Japanese and
Korean go one character per unit (the reference's splitters are Python; word times agree within one 80 ms step).
Guard: the aligner discovery and parse tests, `tests/test_transcriptions.sh` 6.

### A download bundle that fetches fewer files than the loader names (Pixal3D)
Defect: the app's Pixal3D bundle skipped the `_mv` weights but fetched `pipeline_mv.json`, and the engine required
what the JSON named; TRELLIS.2's `pipeline.json` always names the optional 512 texture flow. Cause: a loader reading
"what exists" from a JSON instead of the disk. Fix: an optional part counts only when its weights are on disk
(`pixal3d.mvInstalled`, `onDiskOrEmpty`); a multi-view request is then refused by name. Guard: `a pack with
pipeline_mv.json but no _mv weights serves one view`.

## Tokenizers

### Metaspace `prepend_scheme: "first"` read as "always" (Mistral v0.3)
Defect: every Mistral prompt grew tokens: `[INST]Use` became `▁Use`. Cause: `first` was treated as `always`. Fix:
`MetaspacePrepend` (three-valued), `encode` passes `at_start` per segment. Diff `/tokenize` against HF on a prompt
with specials. Guard: the `"first"` case of `encodeSentencePiece: Metaspace`.

### A combined Split regex hides the digit rule (Muse "8 4")
Defect: Muse-Glimmer echoed "8 4" for "84". Cause: its Llama-3-style combined Split regex holds the `\p{N}{1,3}`
branch inside one alternation, so the exact-match digit rule never fired. Fix: a grammar, not a wider match:
`pretok_style = .llama3` runs `llama3PreTokenize` (cased word runs, attached contractions, {1,3} digits). HF encode
output is not pre-token boundaries: pin the pre-tokenizer against the regex `findall`, ids against HF `tokenizers`.
Guard: `tests/test_tokenizer_hf_parity.sh`.

### Pre-tokenizer grammars served to the wrong family (Llama 3, Flint)
Defect: GLM-5.3, Llama-3.2, LFM2.5 and K2 split camelCase and `//!\n` (+0.3% tokens on code, every agent prompt
off-distribution); Flint split `Python's` and `can't`. Cause: the cased-grammar detector keyed on the contraction
group + `\p{N}{1,3}`, shared by the plain Llama-3 regex, and required that digit branch, which Flint isolates in its
own Split entry; BPE `ignore_merges` was never read; `isDigit` was ASCII-only. Fix: the cased signature is `\p{Lu}` +
attached contractions; `.gpt2` + 3-digit groups for plain Llama-3, `.llama3` for Muse, `.cased` for Flint's
punctuation branch; isolated runs feed either grammar; `ignore_merges` emits a vocab word whole; `\p{N}` is a
generated table. Guard: `tests/test_tokenizer_hf_parity.sh` (zero diff vs HF per family), the Flint pre-tokenizer
test.

### Whitespace runs split where HF's regex keeps them whole
Defect: a blank line holding spaces cost a token on most families; U+3000/NBSP read as punctuation; on DeepSeek-V4,
Hy3 and Spark a whitespace run before a number or CJK was split, so every indented number in code was
off-distribution. Cause: `\s*[\r\n]+` stopped at the first newline where Oniguruma backtracks to the last; `\s` was
ASCII-only; those families isolate digits and CJK in their own Split entries ahead of the main regex. Fix:
`whitespaceLen`, the White_Space set, `isolatedRunsPreTokenize` (`Tokenizer.isolated_runs`). Guard: `pre-tokenize:`
tests, `tokenizer parity` over `tests/fixtures/tokenizer_parity/`, `tests/test_tokenizer_hf_parity.sh`.

### The DSV4 echo-precision saga was a tokenizer bug
Defect: DeepSeek-V4 mirrors slipped characters in verbatim digits (`1o` for `10`), surviving an imatrix recalibration;
the same GGUF through ds4 scored 0/70. Cause: DSV4's pre-tokenizer splits digits in greedy groups of three
(`\p{N}{1,3}`); ours split per digit (correct for Qwen), so every number was off-distribution. Fix:
`Tokenizer.digit_group` parsed from the Split rule: 38-40 slips → 0/70. When a quality defect is invariant across
engine configs and quant recipes, diff the prompt encoding against an independent tokenizer first. Guard: the
digit-group tests.

### A sampler never draws a reserved special
Defect: a collapsing checkpoint emitted `<|fim_hole|>`-class markers at temp 0. Fix: a per-model mask of `special:
true` tokens minus EOS ids minus specials the chat template uses (derived, never hardcoded; no template = off), sized
to the logits (`unpadded_vocab_size`), riding `SamplingParams.suppress_mask` into both samplers and the stochastic
verify; logprobs stay raw, so rank 1 may differ from the emitted token by design. `[suppress]` logs engagement.

### The degenerate-tail loop guard needs a long-period tier
Defect: a two-sentence 58-token cycle repeated 26 times passed the period-≤8 scan. Fix: `isDegenerateTailLoopRange`
also convicts periods 9..64 at 10 exact reps (long lines in real code repeat a handful of times, not ten), on span.
Guard: the degenerate-tail tests.

## Quantized packs and calibration

### An imatrix is only valid for the weights it was collected on
Defect: the DSV4 0731 mirror was calibrated with an imatrix collected on the preview checkpoint. Cause: the retrain
decorrelated ~85% of experts' statistics (median Pearson 0.66 vs preview; two independent 0731 collections agree at
0.85). Fix: collect on the weights being quantized (`ds4 --imatrix-dataset …`); the recollection won 7 of 9 shard
pilots, mostly on `down`. Trap: re-converting a shard is not byte-stable across sessions; copy proven shards.

### Uniform ≤2-bit experts to the last layer cause turn-level agent loops
Defect: a uniform 2-bit DSV4 build looped at the decision level ("verify once more"), 3 of 4 agent runs at 16-28K
tokens, while every single response was fluent and no token window repeated. Fix: a 4-bit tail on the last four layers
(tail4: 0 of 3 loops). Quality gates for low-bit packs need an agent-loop cell (max consecutive identical tool calls).

## Media backends and parity

### A faithful port that made the model worse (H3 `<Picture i>` splice)
Defect: splicing fl2va keyframes into the text conditioning as the reference does inverted frame 0 (+215 → -98
contrast). Cause: the tower's rows carry 41x a token embedding's norm in the residual stream, competing with the
keyframe already pinned by its VAE cond rows. Cosine could not see it; an `rms_ratio` did. Fix: vision blocks off for
fl2va keyframes, on for ref2va references (the prompt addresses them and no cond row competes). A reference is a spec
for what to build, not a promise that turning it on helps your checkpoint.

### H3 ref2va: two partitions, one file layout, a cond stream ordered by the layout
Defect: an fl2va pack handed `ref_images` would silently ignore them; references assembled in request order landed on
the wrong blocks' positions. Fix: our converter writes `tasks` (`["t2va","ref2va"]`) into config.json, the only
discriminator, read at load (absent = not ref2va); one resolver (`resolveRefs`) decides order and `<Picture
i>`/`<Audio j>` ordinals and the DiT block derives from it (`refBlockFor`); the card's 12-file total cap is a named
400. Read repo listings case-insensitively and hash files before inferring identity; clone proven shards instead of
reconverting.

### H3 length planning: a stale cap and a borrowed memory model
Defect: the app capped H3 at 209 frames (an unmeasured cap that went stale) and priced H3 with LTX's VAE formula (677
frames on 128 GB, 32 on 48 GB, both useless). Fix: H3's frame term is the fast recipe's per-block attention cache,
linear in rows (`latent_t·(w/32)·(h/32) + stereo audio + text`), reproducing both measured points; "Max quality" never
allocates it, so it is also the low-memory mode; the load floor is the measured staged peak; `H3Plan.fits` answers
"does anything fit". Time is `c + a·S + b·S²` from two anchors, scaled per machine by run history.

### H3's two fast levers compounded into garbled speech
Defect: a 10 s REF2VA clip said "deepseek ddeepiki emeng minimax" with the step cache and attention broadcast both on.
Cause: the broadcast schedule went by step index, so after cached steps attention was fresh only every sixth step.
Fix: a forward right after a cached step recomputes attention (`attnBroadcastRefresh(…, after_skip)`). Guard: `a
forward after a step-cache skip recomputes attention`; the bar was word errors over six seeds.

### The weights map pins everything a staged loader frees (H3 AdaLN precompute)
Defect: the DiT stayed at 33 GB resident where ~20 was designed. Cause: the safetensors `Weights` map outlived
`Model.load` and held +1 refs on every raw array. Fix: scope the map inside the load. A memory claim without a
resident log line is a hope.

### A tiled decoder's tiles are not slices of an untiled pass (H3 visual VAE)
Defect: risk of off-distribution output that looks like a quality problem. Cause: token coordinates normalize over the
extent, so tiling changes semantics; temporal chunking is how the VAE was trained. Fix: refuse above the tile extent
(`error.TilingUnsupported`) rather than decode untiled.

### A LoRA's alpha lives wherever its exporter put it
Defect: PEFT exports ran 8x too strong (pure static) while 56 stacking checks stayed green. Cause: only the kohya
per-module `.alpha` tensor was read. Fix: `lora.fileAlphaScale` reads nested PEFT JSON > flat pairs > `ss_*` > none,
the per-module tensor wins, a non-finite or non-positive alpha is declined, the scale is logged per file. A guard that
asks "did the bytes change?" needs one that looks: `tests/lora_noise.py` + `tests/test_real_loras.sh`.

### A distilled few-step diffusion model can require bf16 (MageFlow Turbo)
Defect: every component passed f32 parity at cosine 1.0, yet the 4-step generation was a grey wash, noisier with more
steps. Cause: the distilled model memorized bf16 timesteps; the reference casts the timestep to the model dtype. Fix:
`roundBf16` on the timestep and its frequency table on the bf16 path; RoPE rotates in f32 and casts; scalars through
`scalarLike`. Component parity in one precision does not prove the multi-step loop: compare per-step latents in the
reference's native dtype and decode suspects. Fixtures: `tests/dump_mageflow_{te,e2e}_fixture.py`.

### MageFlow 8-bit: lookup tables are invisible to a shape rule
Defect: a shape-driven converter would pack `embed_tokens` and `pos_embed`, which the engine reads by gather. Fix: an
explicit `NEVER_QUANTIZE` list (media); LM `embed_tokens` quantize only where a quantized-gather path reads them.
Fixture cosines on an 8x8 latent are sensitivity probes, not quality: judge images.

### A reference-image editor keeps its input's geometry (MageFlow Edit)
Defect: every phone-photo edit was squashed to the requested square first; then `fitAspect`'s aspect was undone by
per-dimension clamping above 2048 (every modern photo). Fix: `resolveEditTargetSize` = `fitAspect` (aspect at the
budget) → `fitWithinCap` (one factor for both dims); a drift test pins `maxDim` to the clamps; `encodeEdit` asserts
placeholder count == vision rows; the edit prompt is capped like txt2img (a dense host-side causal mask). Test the
composition with fixtures that straddle every threshold. Guard: `resolveEditTargetSize keeps the reference's aspect
ABOVE the backend cap`.

### The `--no-vision` prefix filter ate MageFlow Edit's tower
Defect: every Edit load failed `MissingMageFlowWeight model.visual.patch_embed.proj.weight`. Cause: the global
`--no-vision` drop list gained `model.visual.` and the tower read through `loadWeights`. Fix:
`VisionTower.openWeights` reads through `loadWeightsWithVision`. Guard: `VisionTower.openWeights keeps the
model.visual tower keys`.

### A family's geometry belongs to the checkpoint once a second size exists (FLUX.2 klein 9B)
Defect: the 9B loaded 5 of 8 double and 20 of 24 single blocks from the 4B's struct defaults with no error. Fix:
derive geometry from weight shapes (`ditConfigFrom`, `teConfigFrom`): head counts from `q_proj`/`k_proj` rows,
`mlp_ratio` from the fused `linear_in` / 2; a zero probe leaves the field alone. Verified a no-op on the 4B.

### Per-token AdaLN (LTX I2V) must leave the t2v path byte-identical
Defect: risk: an optional per-token modulation path regressing the scalar one. Fix: gate it so `nv_pt == 0` runs the
unchanged scalar helpers; the tripwire is uniform-mask equivalence (an all-ones mask must reproduce scalar, cos 1.0).
Encoder parity is fed the fixture's pixels so preprocessing never confounds it. Guard: `per-token AdaLN uniform-mask
equals scalar`.

### Voice cloning: an ECAPA `TimeDelayNetBlock` is conv + ReLU (Qwen3-TTS)
Defect: clone fidelity fell short (embedding cos 0.9989). Cause: the pooling block was ported as a bare conv. Fix: the
implicit ReLU: bit-exact. Guard: "speaker encoder (ECAPA-TDNN) matches the Python oracle".

### An embedding lookup reshapes to the table's width (Qwen3-TTS 0.6B)
Defect: the 0.6B-Base crashed the server (`[reshape] Cannot reshape array of size 24576 into shape (1,12,1024)`).
Cause: the reshape used the talker hidden size; the 0.6B text table is 2048 wide with a projection after it. Fix: the
width comes from the weight. Guard: `embed uses the table's native width`.

### An oracle that cannot execute the reference must say so (H3 DiT)
The reference block calls CUDA kernels that do not run on a Mac, so `tests/dump_minimax_h3_fixtures.py` is a
transcription and says so; where the reference can run (layout math) it is executed directly, with import stubs that
raise on attribute access so no golden value comes from a mock.

### A permutation-invariant checksum cannot see a permutation (H3 layout)
A per-axis position sum passed with the two stereo channels' `w` coordinates swapped. Pair any column checksum with a
row-index-weighted one (`pos_weighted`).

### LTX-2.5 looked soft: canvas, quantization, decoder
Defect: LTX-2.5 clips looked below the published examples. Cause: three measured causes: the app's ladder topped out
at 768x512 and two-stage tiers denoise at half size; plain affine 4-bit injected ~10% weight noise per linear (8-bit:
0.6%, +3.5% time); we used the plain conv decoder where upstream decodes with DiffVAE. The 8-bit pack then crashed:
`video.zig` hardcoded `bits = 4`. Fix: a wider canvas ladder, an 8-bit mirror, geometry-solved quant params
(`quantGeom`), the DiffVAE decoder. When arms fork content, per-frame metrics are variance: stills decide. Trap: `hf
download REPO --include a b c` eats `a`; pass filenames positionally.

### The ANE seam is fp16 end to end, and H3's DiT rendered black
Defect: the ANE MLP offload on H3 gave 1.24x and an all-zero video; per-block parity read cos 0.99999. Cause: the
program computes in fp16 and H3's fc2 partial peaks at ~1.7M (fp16 max 65504): INF, then NaN. Fix: divide the `up`
half of the ANE weight copy by `OUT_PLANE_SCALE` (256) and multiply the read-back: exact, since `up` is linear into
the product and a power of two is exact in fp16 (scaling `gate`, under a nonlinearity, or fc2 alone is wrong). Krea
had been silently losing precision. A correct-looking image is not evidence of range headroom.

### A metric taken after a global normalization measures the normalizer (ACE-Step)
Defect: ANE vs GPU audio showed a ±1 dB RMS "loudness lift" that changed sign across seeds. Cause: `peakNormalize`
sets the whole track's gain from one sample. Fix: measure trimmed per-frame RMS and log-STFT cosine, never a
whole-track scalar after a normalizer.

### ANE media offload: the tile tail, the share, the calibration dtype
Defect: the fixed-tile media seam trailed the whole-request tile by 3-4%; the share re-derived per request recompiled
program sets (shares straddled a rounding line); the calibration aimed low. Cause: a GPU tail built after the blocking
ANE wait is serial launch latency; a parameter that keys a compiled artifact was re-solved per request; the complement
was timed in bf16 where ACE's stream is f32. Fix: async-eval every GPU piece before the tile loop and pad a large tail
onto the ANE (`mediaTilePlan`); probe at fixed `MEDIA_PROBE_ROWS` and reuse the compiled set's share (`cachedShare`);
each calibrator names its MLP's dtype (`AneCalib.dtype`). Guard: `mediaTilePlan`, `shareVariant`, `calibratedShare`
tests.

### The DiffVAE decoder: a faithful port that decoded to static (LTX-2.5)
Defect: per-stage parity read cos 0.99997+ and `rms_ratio` 0.999 everywhere, and the first real generation decoded to
pure static. Cause: the decoder's output type, steps and timestep scale are constructor arguments no pack ships; the
class defaults (v-prediction, 2 steps, x1) return the noise, and the transcription inherited them. A faithful-port
oracle cannot prove the reference's unstated arguments; the file's name said "1-step x0". Fix: `Sampler` pins x0, 1
step, timesteps x1000; NATTEN shifts its window inward at a boundary (never clamp-and-mask), pinned on boundary-only
volumes; the tile budget is a per-request decision from free RAM (`tileTokensForMemory`), with blending ramps that sum
to 1 and a seam-crop assertion. Guard: the `Sampler` test, `tests/test_video_gen.sh` (gradient relative to the conv
arm).

### MiniMax Music 3 is not an ACE-Step variant
Defect: the obvious ACE-Step reading runs and produces plausible noise. Cause: the DiT's timestep is a prepended token
(no AdaLN), the FF is reversed SwiGLU (`first * silu(second)`), the vocoder Snake is alpha-only, the DiT RoPE rotates
32 of 64 dims at a hardcoded 10000; the AR loop feeds frame 0 back though it is never emitted, top-k thresholds from
the conditional branch, the feedback embedding scales by 8^-0.5; square convs defeat the bias-length weight-norm
heuristic (key the transpose by name); `_clean_caption`'s regex crosses lines. Fix: each pinned by a transcription
oracle (`tests/dump_music3_fixtures.py`) or a table test; a finite additive vocab mask keeps CFG finite. Attribute
stages with laps (`MLX_SERVE_DIAG=music3-cost-probe`) before pricing a lever off bandwidth arithmetic: a depth KV cache halved a
stage the model said was at its read floor.

### A bare f32 scalar promotes a bf16 stream
Defect: `bf16 * f32_scalar` promoted products to f32 and every downstream quantized matmul ran off-dtype. Fix: scalars
through `scalarLike` (MageFlow, Music3) / `scalarOf(v, dtype)`; a chain that returns f32 by design makes the caller
cast back. Tell: `[dtype-trace] residual widened`.

### LFM2-VL: a forward arm with no vision splice answered "a completely black rectangle"
Defect: tower parity was perfect and every log number right, and the model described a black rectangle. Cause: the
hybrid forward arm had no splice; 247 placeholder embeddings went through the trunk. A missing splice presents as a
bad image. Fix: every generative forward arm splices vision. Guard: lint pin (formerly a source scan) over the
forwards; encoder-only arms exempt by name.

### A position resample is `bilinear` + antialias; mlx-vlm's bicubic is a scale error (LFM2-VL)
Defect: mlx-vlm's resampled position table came out 13% hot (`rms_ratio` 1.13). Fix: torch + transformers as the
oracle; the Pillow-convention filter's `.bilinear` arm widens its footprint on downscale as antialias does. Patch
features are `[py, px, c]` with channel innermost; the checkpoint carries two GELUs (tanh in the encoder MLP, erf in
the projector).

### For a NaFlex model, tiling is the resolution (LFM2-VL)
Defect: untiled, 15px print was read confidently and wrong. Fix: past `max_image_tokens x 2.0` the image is resized
once onto a canvas of 512px tiles, each a window into it, plus a thumbnail, labelled `<|img_row_R_col_C|>` /
`<|img_thumbnail|>` in encoder order. Guard: `tests/test_lfm2_vision.sh` compares the tiled and untiled arms.

### A routing chain that returns f32 by design widened the residual stream (Ling)
Defect: Ling's forward ran in f32 from its first MoE layer: a uniform slowdown with no slow op. Cause:
`groupLimitedRouting` returns f32 weights on purpose and the new caller skipped the cast. Fix: a dtype-conditional
cast where the scores enter the expert combine. Read the dtype trace on every new arch's first forward.

## Vision

### Chunked vision prefill: the splice row index is per prefill, not per forward (#197)
Defect: an agent's screenshot requests were refused past ~10k tokens (63 GB billed on a 48 GB Mac). Cause: vision
prompts prefilled whole because the splice counted placeholder rows per forward, so a chunk restarted at row 0 and
re-spliced the first image rows (coherent answers about the wrong pixels). Fix: `ForwardCtx.vision_splice_offset`,
counted host-side per chunk (`countSpliceRows`), set on every prefill forward including the final span; the guard
bills the chunk. Repro traps: the 512-token tail merge un-chunks short repros (check `chunks=1`), and byte equality
across chunk widths is unsound at 4 bits (assert the perceived color set). Guard:
`tests/test_vision_chunked_prefill.sh`.

### A renamed vision tower and a Conv3d read in the wrong axis order (Qwen3.8 Alis)
Defect: the Alis packs disabled vision (`vision_tower.*` renamed `model.visual.*`), and once loaded answered a red
square "black". Cause: the prefix was hardcoded; `patch_embed.proj` is stored channels-last by mlx_lm and
channels-first by a straight torch export, and the loader always permuted. Fix: `resolveVisionPrefix` +
`patchProjLayout` (derive from the shape, refuse by name); `model.visual.` joins the `--no-vision` drop list.
MTPLX-Optimized-Quality shipped the same scrambled conv under the supported prefix. Probe with three colours: one
cannot tell seeing from guessing. Guard: the layout and prefix tests.

### An engagement line proves the path ran, not that the rows reached the prompt (Qwen3-VL video)
Defect: a 4-frame clip logged every engagement line green and the model described a stranger in a white shirt. Cause:
the splice mask covered image and audio ids, never `video_token_id`; the video e2e accepted words a model hallucinates
freely. Fix: the mask and `countSpliceRows` take `video_token_id`; the e2e demands frame 3's sign text. Guard:
`spliceVisionRows: video placeholders are scattered like image rows`.

### A placeholder id of 0 is unset, not token 0 (Qwen3-ASR)
Defect: Qwen3-ASR's first transcription answered "no speech" with the tower matching mlx-audio at cos 1.0. Cause:
three splice sites treated `image_token_id` as always set: an audio-only model has 0, so its rows were never spliced,
and a text `!` (id 0) would have taken an audio row. Fix: all three skip a 0 id. Guard: `spliceVisionRows: an unset
placeholder id (0) never claims token 0`, `tests/test_transcriptions.sh`.

## Qwen packs

### Standardizing our own Qwen3.8 packs: embeddings, head norms, DWQ
Defect: our 4/8-bit packs kept `embed_tokens` dense (1.8 / 1.2 GB) where every other pack, including our own 6-bit and
iQ, quantized it. Fix: `tests/restandardize_qwen38_pack.py`: quantize the embedding from the raw stored bf16 bytes (an
f32 round trip breaks byte identity with mlx_lm), publish head norms folded (`--fold-mtp-norms`; a loader that misses
the delta convention drafts at ~0% while looking healthy), and graft DWQ scales at 4-bit only, verified per tensor (at
other widths the shapes still match and a wrong graft loads garbage).

### Qwen3.8-Flash-Next (`qwen4_exp`) port
Defect: traps that cost time on the port: HF `hidden_states[i]` is layer i's INPUT; the QSA tail is per query (a
global tail left rows with nothing visible); indexer scores need f32 and `torch.topk`'s lower-index tie break; the
n-gram eos is the text config's; a new per-request field must be nulled with the conv/ssm handles; tiny oracle
geometry must clear the kernels' floors (GDN refuses dk 16); image rows splice before the hyper-connection tile; the
indexer reads the same M-RoPE table as attention, with one YaRN mscale. Tiny random MoE oracles tie everywhere: the
dump records the reference's own margins and `Qwen4Ties` acquits by those, never ours. A k < E fixture covers top-k
selection only through the MTP head's one MoE layer (`--topk 2`, read k back from the checkpoint). Restore parity on
this arch is the chunking class (~0.3 nats top-5), not the hybrid 0.047. Detail: `src/plugins/qwen4_exp/AGENTS.md`.

## More model ports

### Spark-X2.5: a later `Digits` rule re-splits the groups
Defect: `digitGroupFromPreTokenizer` stopped at DeepSeek's `\p{N}{1,3}` and would have served 3-digit groups; a later
`Digits(individual_digits)` rule makes the checkpoint per-digit. Fix: read the whole pre-tokenizer sequence. The arch
otherwise adds exact-erf GELU, a fused `q_k_v_proj` split at load and a per-head sigmoid output gate. Bar: HF
`tokenizers` byte-identical ids; the bf16 HF oracle.

### K2-Horizon: markers, a generation-config EOS, and a thinking default
Defect: the 51 KB template hit two jinja.cpp gaps (`is sameas true` with a bare argument); `<ifm|…>` markers needed a
fourth think spelling; `<|ifm|im_end|>` is declared only in `generation_config.json`, so the model never stopped;
thinking defaulted off on a template that always opens a think marker, so every llmprobe cell ended `length`; JSON
schema + thinking matched a literal `</think>` inside the reasoning; a parameterless GLM call vanished. Fix: the jinja
gaps; `installMarkerAliases` decodes markers to canonical bytes (only the rendered prompt keeps the pack's spelling,
`k2ThinkOpenerAt`); `mergeEosTokens` for every model; thinking ON by default; `Tokenizer.markerCloserFor` pairs opener
and closer ids; a bare-identifier body routes to the GLM parser. Also found here, for every model: `top_k: 1` and
`top_p` near 0 still sampled because the filters masked by value and bf16 logits tie (filters now cut by rank).

### Bonsai 2 served in bf16
Defect: Prism's Hadamard packs ship f16 scales and f32 GDN tables, and Prism runs f16 with an f32 GDN state; we
narrowed to bf16 (KL 1.7e-4). Fix: `actDtype` f16 and `ssmStateDtype` f32 for Hadamard packs, the loader keeps f16
(`LoadOpts.keep_f16`), every kernel on the path takes f16; two bf16 constants surfaced only under f16 (the batched
decode mask, the M-RoPE cast). KL 2.9e-6, same speed. Guard: `tests/test_hadamard_fidelity.sh`.

### Qwen3.8 Flash Next's GatedDeltaNet decayed in bf16
Defect: bf16's spacing below 1 rounded a slow-forgetting head's decay to exactly 1.0: 206 of 1728 heads never forgot,
and the state re-rounded every token. The packs declare `mamba_ssm_dtype: float32`. Fix: `ssmStateDtype` f32 on
qwen4_exp and every gate producer computes at the state's dtype (`gdn_decode.gateDtype`); f32 roots get their own SSD
namespace. Guard: `qwen4_exp GDN: a slow-decay head forgets at its own rate on every arm`.

### Qwen-Image-2.1: an 18 GB VAE decode on a 5 GB engine
Defect: a 1024² generation peaked at 18 GB of process footprint while MLX's counter read 4.9 GB. Cause: MLX's 3x3 conv
holds an unfolded input copy outside its counters (11 GB at 288 channels), and a stage keeps several full-resolution
intermediates alive. Fix: exact row strips for big 3x3 convs (`Conv.forwardStrips`) and banded stages with a cropped
halo (`Stage.banded`): everything but the 3x3 convs is per pixel. 18.1 → 9.5 GB. Guard: the VAE parity oracle with
every stage forced into bands.

### Nemotron-H: two layer-pattern spellings, and an MoE arm that was `unreachable`
Defect: Nemotron 3.5 Lightning failed with `MISSING WEIGHT: backbone.layers.0.mixer.q_proj.weight`. Cause: only
`hybrid_override_pattern` (a string) was parsed, not `layers_block_type` (a list); the MoE layer-init arm was an
`unreachable` TODO that ReleaseFast folded into the MLP arm. Fix: both spellings; a real `nemotron_moe` op (grouped
sigmoid routing, ReLU² experts, shared expert). Guard: the parse tests and the plugin's recorded proofs.

### Nemotron-H: attention ran with RoPE the model never had
Defect: needle lookups in 3k-28k token logs came back wrong; the MTP head accepted ~1.1 per round. Cause: the shared
hybrid attention roped both LFM2 and Nemotron-H; Nemotron-H attention is NoPE whatever `rope_theta` its config
carries. Fix: `layer_no_rope` set at parse for every layer and the MTP head; `cacheLayoutNamespace` gives it a fresh
SSD root so RoPE-era keys never restore. Guard: `nemotron_h served: a NoPE attention layer ignores the order of
earlier tokens`.

### Nemotron-H: the MTP head's input and depth on a MoE trunk
Defect: auto-depth MTP ran at depth 6, slower than serial. Cause: the head expects the residual before `norm_f`; each
verify row routes to K experts, so weight traffic grows with depth. Fix: the hybrid path captures before the final
norm; depth is 1 or 2; verify windows ≤ 8 rows ride the in-place expert path. Guard: `nextMtp on a Nemotron-H trunk
emits the serial greedy stream`.

### Nemotron-H: the fused add+norm path dropped the shared expert
Defect: after the decode fusions, text degenerated after ~30 tokens with every kernel test green. Cause: the fused
MoE-combine kept only the routed slots once the shared expert stopped riding the banks. Fix: the kernel takes the
shared expert's output and mirrors MLX's reduction bit for bit. A fused path carries every term of the chain it
replaces, and a forward-level test compares the two. Guard: `the fused add+norm decode path equals the unfused
blocks`.

### Nemotron-H: a single routing group ran the composed router chain
Defect: ~30 µs per MoE layer of routing against a ~5 µs matvec. Cause: the grouped kernel declines `n_group <= 1` and
Nemotron 3.5 sets `n_group: 1`. Fix: a single group selects the ungrouped `.sigmoid_bias` kernel.

### Nemotron-H: the Mamba2 output widened the residual stream to f32
Defect: `[dtype-trace] hybrid: residual widened at layer 0`; decode at half speed. Cause: the f32 SSM state's `y` was
not cast back as mlx-lm does. Fix: cast `y` to the input dtype before the gated norm.

## Media backends (continued)

### A pinned library's per-op transient is invisible to every residency bill (#321, #424)
Defect: after an mlx bump, LTX-2.5 died at "Decoding video" with a green preflight
(`kIOGPUCommandBufferCallbackErrorOutOfMemory`): 66.8 GB peak where 36 GB sufficed. Cause: mlx decomposed small-depth
3D convs into per-tap 2D convs, each tap a full output held until the op completes, each Winograd call budgeting 75%
of the working set; H3's encoder built one lazy graph over 48 tiles. Fix: `conv3dDepthChunked` (exact overlapping
depth windows, each evaluated); H3 `encodeMoments` evaluates per tile pass. No env lever for an exact, free fix. Diff
conv dispatch on every mlx bump. `/props` `peak_bytes` reproduces a peak on a box that survives it.

### A latent-to-RGB preview is a fit, not a hue wheel (#208)
Defect: the preview projected channels around the golden angle and stretched each frame: a colourful blob unrelated to
the video, while the test asserted `FF D8`. Fix: ComfyUI's published table per latent space (`LTXAV`, not `LTXV`;
`MiniMaxH3Video`), generated by executing the reference (`tests/dump_latent_rgb_factors.py`), with live perceptual
tests whose control arm is the retired projection (the fit must beat it by 0.3; chroma asserted separately; content of
one random colour per latent cell). Previews default off in the app, and only the shown frames are copied.

### Hunyuan3D paint unwrapped the raw marching-cubes mesh
Defect: no textured res-320 job finished (xatlas at 900k faces). Cause: upstream decimates to 40,000 faces before the
unwrap. Fix: vendored Fast-Quadric-Mesh-Simplification (`lib/fqms`), `mesh_simplify.decimate` before the unwrap.
Guard: `tests/test_3d_paint.sh` at res 320.

### A resident H3 text encoder crashed the second keyframe request
Defect: the second keyframe request on a resident server killed it. Cause: the vision load unwrapped a weights map
that was only opened when something had to be read. Fix: load the tower only when the encoder has none. Guard:
`tests/test_h3_resident.sh` [4].

### A runtime LoRA bypass cost H3 Turbo half its step
Defect: a Turbo step took 18.7 s against 10.1 s without the adapter. Cause: `lora.delta` multiplied the bf16 product
by an f32 scalar array, promoting a 1.3-1.8 GB tensor per adapted linear. Fix: `lora.addTo`: one skinny GEMM per
adapter and `mlx_addmm` onto the base output in its own dtype. Guard: `addTo equals y + deltaSum in y's dtype`.

### A LoRA at scale 0 still changed the image
Defect: `lora_scales: [0]` rendered differently from no adapter. Cause: MLX's addmm at `alpha == 0` does not return
`beta*c` at DiT shapes in bf16. Fix: `addTo` skips a zero-scale adapter. Guard: `addTo with a zero or tiny scale
returns y`, `tests/test_qwen_image_lora.sh` check 2.

### Stable Audio 3: a fractional length came out as clipped noise
Defect: any non-whole length (0.5, 0.9, 1.5 s) produced full-scale noise; parity fixtures ran at 5 s. Cause: training
conditions on `ceil(n_samples / sample_rate)`, whole seconds ≥ 1; the reference also samples 6 s past the request and
trims. Fix: `trainedSeconds` and `latentCount`. A condition works only as training fed it. Guard: the two sa3 tests.

### A Sushi MiMo pack: a stale bf16 copy, and MTP heads in the source layout
Defect: `MiMo-V2.6-Flash-Sushi-2.3bpw` died in warmup on a bf16 `o_proj`; later every boot logged MTP off. Cause: the
source shard still carries bf16 copies beside the affine triples and the loader kept the last listed; the heads were
stored like the trunk (fused e4m3 QKV) and never converted. Fix: duplicated tensors load from the shard the index
names (`indexOwners`); embed/head are dense by dtype; the heads get the trunk's QKV split and MLP decode. Guard:
`indexOwners: the index names the shard…`, the smoke `mimo_v2_sushi.mtp` cell.

### MiMo's own DFlash drafter loaded, drafted garbage and outranked MTP
Defect: Xiaomi's drafter accepted 2-4%, the gate turned it off, and the request ran serial at 32 tok/s against 67 with
the MTP heads. Cause: its attention needs partial rotary, a value scale and sink biases the block forward lacks. Fix:
such a drafter is refused by name (`UnsupportedDflashAttention`). Guard: `dflash: a drafter whose attention needs
unported arms is refused by name`.

### EmbeddingGemma 2: a padded row can NaN a whole batch
Defect: every row of a mixed-length batch came back NaN. Cause: a padded query whose whole band is padding softmaxes
over `-inf`, and the NaN reaches real rows through V; only NAX kernels happened to stay finite. Fix: the summed mask
is floored at -1e9; the head returns f32. Ported facts not in the config: the band radius is inclusive, per-layer
inputs are projection-only, the projection runs after pooling, `pooling_mode` is a string in the sidecar.
EmbeddingGemma 1 still lacks the floor (`docs/known-issues.md`). Guard: the tiny-checkpoint reference test,
`tests/test_embeddings.sh` [4d].

### EmbeddingGemma 2 video frames embedded at 70 tokens, not 140
Defect: frames embedded at half the soft tokens the release asks for. Cause: the processor config was parsed with a 16
KB fixed buffer, the `OutOfMemory` swallowed and defaults kept; a toy JSON fit in the test. Fix: the parser takes an
allocator and the real file is its test input. Gemma 4's own image processor (aspect-preserving, multiples of 48),
alpha dropped as PIL does, each frame in its own `<boi>…<eoi>`. Guard: the processor table tests,
`tests/dump_embeddinggemma2_media.py`.

### YuE2: a Scope held 25 GB of decoder temporaries; three oracle bars were wrong
Defect: a 40 s song peaked at 29 GB; the oracles showed a 0.4% VAE gain and a logit shortfall, and the nucleus kept
the wrong tokens. Cause: a stage-wide `Scope` kept ~70 half-gigabyte temporaries alive; the gain is TF32
(`MLX_ENABLE_TF32=0` matches); the shortfall is one bf16 ulp a softmax cannot see; the nucleus overwrote its own
pivot. Fix: per-op scopes (weights + 5.4 GB); bars on what matters (argmax + KL, cosine + a 1% scale band,
`rms_ratio`). Guard: the sampler test, the `YUE2_*` oracles, `tests/test_yue2_gen.sh`.

### A speaker cache keeps one of many equal frames (Nemotron 3 Diarization)
Defect: agreement with mlx-audio drifted from 0.04 to 0.43 after the first 27.2 s chunk. Cause: confidently spoken
frames tie in the speaker cache's ranking, so compression keeps one of many near-equal frames (the references disagree
the same way); underneath, `Scope.geluErf` added an f32 scalar to a bf16 stream. Fix: `geluErf` keeps its dtype;
compression runs on the host (`aoscKeep`), checked on its own. A cached-selection model is judged by agreement and
accuracy (99.43% vs 99.48/99.49%), never frame-exact output past the first chunk.

### The int8 copy sat beside the packed weights (`w8a8`)
Defect: `--w8a8` left packed weights resident (H3 held 77 GB) and its int8 attention ran on M1-M4, where the kernel
cannot. Cause: four paths still read the originals, loaders held their weight maps, only the linears checked for NAX.
Fix: fused q/k/v as row views, `Weights.dropBlock` per converted block, the memo dropped per job, `w8a8.nax()` gates
every kernel, default-on only with NAX + 64 GiB. Guard: the decline and default tests, every engine arm's cosine + rms
vs packed and resident bytes vs the bill.

### A ModelOpt GLM answered with its own prompt: the `up` scale landed after the clamp
Defect: GLM-5.3 FP4 echoed the question back. Cause: each expert's `up` global scale was folded into the routing
weight (exact for Flash Next), but GLM clamps `up` at ±10 before that point. A scale folds past an activation only if
the activation is linear in that operand. Fix: with a `swiglu_limit` the up scale stays apart (`up_expert_scale`).
Guard: `ModelOpt banks: a clamped SwiGLU keeps the up scale apart`, `tests/test_glm5_next.sh`.

### `F_NOCACHE` is ignored for an unaligned read
Defect: loading GLM FP4 froze a 256 GB Mac and ended in SIGKILL. Cause: `F_NOCACHE` bypasses the cache only when
offset, length and buffer are page-aligned; expert tensors sit anywhere. Fix: read the covering aligned span through
an aligned stage, 6 banks at a time; `memoryPressureLevel` stops the load with `InsufficientMemory`; the bill adds
decoded scales. Tell: `File-backed pages` climbing as fast as the load. Guard: the `ModelOpt reads` tests.

### An f32 norm weight widened a bf16 model's MoE input (GLM FP4)
Defect: GLM FP4 decoded far slower than the 5.1bpw pack. Cause: NVIDIA keeps trunk layer norms in f32 and `rms_norm`
promotes to the weight's dtype, so the fused reads and fp4 kernels declined. Fix: `narrowsLoadedF32Norm` casts trunk
f32 norms to bf16 for ModelOpt packs only (vision towers keep theirs); `moe_fp4` takes the up scale and clamp. Guard:
the predicate test, `moe fp4 decode: a clamped SwiGLU …` vs fp32 truth.

### TRELLIS.2's dual grid is not a manifold, and its remesh sits off the colours' surface
Defect: unwraps of 80-110 s into ~14k charts, then black stripes after a remesh. Cause: the dual grid is non-manifold
by design; the remesh skin sits a voxel or two off the decoded surface. Fix: `trellis2_remesh` (unsigned-distance
offset iso-surface) before xatlas, and every texel snapped to the raw mesh (`snapToSurface`) before the volume read.
Guard: the remesh tests and the pipeline oracle.

### A flow-DiT forward over tens of thousands of tokens needs a scope per block
Defect: one scope over a whole forward keeps every block's intermediates alive together; fixture oracles never show
it. Fix: `Dit.forward` and `Decoder.run` evaluate and free per block (1024 request: 21 GiB); Pixal3D's NAF encoders
got a scope per conv (49.9 → 21.9 GiB). Guard: the peaks pinned by `trellis2PeakBytes`.

### A fixture script's `.float()` turned a complex buffer real (Pixal3D)
Defect: one structure-flow oracle diverged (output cos 0.79). Cause: `.float()` dropped the imaginary half of a
complex64 `rope_phases` buffer in the reference. Fix: keep complex tensors as they are. An oracle that improves when a
transform is removed had it disabled.

### xatlas segments one connected surface on one thread
Defect: a 1M-face car spent 446 s unwrapping with 35 idle cores. Cause: xatlas segments a chart group on one thread
and a car is one group. Fix: `uvwrap.parametrizeParallel` labels equal spatial regions as materials (median split,
`REGION_FACES` 8192): 3.4 s, same coverage. Hunyuan paint keeps `parametrize`. Guard: `a connected surface unwraps as
one chart group per region`.

### TRELLIS.2 roughness is flat, so shininess is one global finish
Defect: a car rendered as satin plastic. Cause: the decoded roughness is nearly constant and the GLB shipped
`roughnessFactor` 1. Fix: `roughness_scale` and `clearcoat` request fields (Hunyuan3D refuses them by name); gloss
shows every bump, so the seed matters. Guard: the finish tests.

### `gen.mjs glb --max-tris` grew the file and crumpled a hood
Defect: simplifying a 1M-face car returned more triangles in a larger file. Cause: the old index buffer and
unreferenced vertices stayed in the BIN; fixed UV seams put every collapse on the smooth panels. Fix: `rebuildGlb`
repacks the BIN; hard-surface budgets come from the server's seam-free `decimation_target`. Guard: the `gen.mjs`
repack tests.

### A doc that spells the image token failed every DeepSeek-V4.1 request
Defect: an agent working in this repo carried AGENTS.md (which spelled `deepseek_image`) in its prompt, and every
request failed `Dsv41VisionRowsMissing`. Cause: the added token splits out of any text and became an image position
with no rows. Fix: without the request's rows the image token is text (no splice, no VL routing bias, its own Engram
id); a literal beside a real image stays a named 400. Guard: `dsv41 image token: without image rows it is text`.

## Parity method

### Fixtures for fp16-fragile giants are dumped fp32 on CPU
Defect: a paint DINOv2-giant oracle sat at cos 0.30 while every per-layer bisect of the port was clean. Cause: the
fixture was dumped in fp16 on MPS; hidden magnitudes reach ~400 and the run silently decorrelated. Fix: dump deep
ViTs/DiTs in fp32 on CPU; when an oracle misses badly, re-run the spec in numpy from the converted weights before
rewriting the port.

### transformers ≥ 5 zeroes custom-model rotary buffers
Defect: the ACE-Step DiT oracle sat at 0.66 while a no-RoPE variant passed at 0.99996. Cause: a `trust_remote_code`
model's non-persistent `inv_freq` is materialized as zeros, so the reference ran identity RoPE. Fix: rebuild each
zeroed rotary from a fresh instance of its class; dump scripts assert no `*.rotary_emb.inv_freq` is all-zero. An
oracle that improves when a transform is removed had it disabled.

### Cosine cannot see scale
Defect: a tower whose features came out 40x too large would score cos 1.0. Fix: concatenated tensors also assert
`rms_ratio`; pin a single row where a whole-tensor cosine averages over many (ACE-Step's timbre row).

## Nemotron 3.5 Super and RADIO

### RADIO configuration and numerics follow the checkpoint
Defect: native omni packs loaded text-only. Cause: vision keyed on a private flag instead of
`vision_config.model_type: "radio"`. Fix: loader and discovery share `discovery.hasRadioVision`; preprocessing follows
the official f32 antialiased bicubic and f32 CPE (PIL's uint8 step moved pixels). Guard: the RADIO config tests and
the executed fp32 fixtures (cosine + RMS ratio).

### Native Nemotron wrappers retain more than an architecture tag
Defect: Super's `nemotron_h_omni` pack was hidden in the app and failed to load. Cause: exact type lists and flat
`backbone`/`mlp1` names. Fix: the language wrapper and `mlp1.layers` projector resolve from the checkpoint's names,
quantization per linear; the native config's final vision LayerNorm before pixel shuffle is required. Guard:
picker/type parity tests, the Super RADIO fixtures.

### A LiLiCorr drafter can own its output head
Defect: Super's drafter proposed through the target's head, and `/v1/messages` dropped DFlash. Cause: DFlash ignored
`has_own_lm_head`; the messages surface checked only the Gemma slot. Fix: a declared head is loaded, validated and
used for proposals only (verify keeps the target head); all four surfaces share `requestEnableDrafter`. Guard: native
config corpus, head forwards, the HTTP defaults on all four APIs.

### A plugin's copy of the MoE dropped the arm only ModelOpt packs reach
Defect: Qwen3.8 Flash Next FP4 (NVIDIA ModelOpt) decoded garbage from the first token after the plugin moved off the
legacy `Transformer`; the iQ pack was fine. Cause: the trunk's copy of the MoE dropped the `per_expert_scale` routing
arm (each expert's up x down `weight_scale_2`) as unused, and every proof replayed tapes of a fixture pack that carries
no such scale. Fix: the arm restored, plus a load-time `unservedLayout` that refuses, by name, any layout an arm the
fork dropped would read. Guard: `a per-expert scale of 2 doubles the routed experts exactly`, the layout-guard test,
the real-pack greedy gate.
