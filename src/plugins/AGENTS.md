# plugins/ — what every plugin owes

Shared rules for configs, tokenizers, weights, quant, vision, media and parity. A model's own rules: its folder's `AGENTS.md`. Stories: `docs/gotchas/models-media.md`.

## Rules

Configs, templates, tokenizers:
- **Config JSON is untrusted**: use the checked readers in `model.zig`; optional nulls retain defaults, explicit disable sentinels and guarded skips remain valid, and unsafe narrowing or dimensions return named errors.
- **A marker family that is GLM/`<think>` under another spelling is ALIASED at decode, never re-parsed** (K2 `<ifm|…>` → `Tokenizer.marker_aliases`); only the rendered prompt keeps the pack's spelling (`k2ThinkOpenerAt`). JSON-schema + thinking on K2 keeps thinking off (known).
- **`generation_config.json` `eos_token_id` joins the stop set** (`mergeEosTokens`, additive); Gemma terminators merge additively (`ensureGemmaTerminators`). jinja.cpp: `is sameas true` / `is divisibleby 3` parse a BARE test argument.
- **A struct default is ONE family's**: Gemma's 1024 sliding default hit every Llama 3.x (no `sliding_window` key) past 1k tokens; `rope_type: llama3` is read. Guard: smoke `long prompt`.
- **Config reads**: when the reference IGNORES a field, the field is not the truth (laguna YaRN mscale); a field HF allows in two SHAPES is read as both (`chat_template`; Nemotron-H's `hybrid_override_pattern` string vs `layers_block_type` list); `text_config` FIRST, then root, PER FIELD; a default only ONE family wants is pinned PER LAYER TYPE (muse `rope_local_base_freq`, `tests/test_muse_repetition.sh`).
- **A shared attention helper follows EACH arch's position scheme**: Nemotron-H is NoPE whatever `rope_theta` its config carries (`layer_no_rope` set at parse, read by `HybridAttention` AND the MTP head); LFM2 ropes. Guard: `nemotron_h served: a NoPE attention layer ignores the order of earlier tokens`.
- **`*_text` siblings**: accept the tag, collapse to base type, prefix by `text_config` presence, force `tie_word_embeddings` for Gemma, add to BOTH visibility allowlists.
- **A sampler never draws a RESERVED special or a PADDING row** (`reservedOutputIds` + `definedVocabSize` → `installSuppressMask`; `unpadded_vocab_size` = ONE trim); logprobs stay RAW.
- **WordPiece honours `max_input_chars_per_word`** (`Tokenizer.wordpiece_max_chars`, HF: a longer word is one `[UNK]`): the greedy longest-match is cubic in the word, and a 200k-char word pinned an embeddings connection thread for good.
- **Metaspace `prepend_scheme` is THREE-valued** (`MetaspacePrepend`): `first` prepends ▁ only at offset 0, never after a special token (Mistral `[INST]Use`); `always` prepends per segment (laya). Diff `/tokenize` vs HF on a prompt WITH specials.
- **Digit GROUPING is per-model** (`Tokenizer.digit_group`; a COMBINED Split regex hides it); the CASED grammar (`\p{Lu}` + attached contractions) is `.llama3` (Muse, `[\r\n/]*` tail) or `.cased` (Flint: `[\p{P}\p{S}]` punct, newline-only tail, an unmatched `\p{C}` run kept whole), the plain Llama-3 regex is `.gpt2` + 3-digit groups; `ignore_merges` emits a vocab word whole; `\s` is Unicode White_Space (U+3000, NBSP), `\s*[\r\n]+` BACKTRACKS to a run's last newline, and a Split entry of its OWN (DeepSeek's and Flint's digits, CJK) cuts the text first under EITHER grammar (`isolated_runs`). Guard: `tests/test_tokenizer_hf_parity.sh`. The degenerate-tail guard has a LONG-period tier (`isDegenerateTailLoopRange`).

Weights, quant, loading:
- **Tree, prefix and axis order are CONVERTER choices — probe** (`resolveWeightPrefix`, `lagunaRouterBase`, `hy3ExpertContainer`, `resolveVisionPrefix` + `patchProjLayout`). A family's geometry comes from the CHECKPOINT once a second size exists (FLUX klein 4B/9B).
- **Quant resolves PER WEIGHT** (`computeQuantParams`; scales dtype decides fp8 vs affine; overrides can hide inside the fp family, `fpParamsFromGeometry`); affine bits outside {2,3,4,5,6,8} rejected at PARSE; no engine hardcodes a width (`affineParamsFromGeometry`, scan-pinned). A table or projection loaded WITHOUT its input width (hint 0, config gs) reads another group size: Gemma 4 E4B mixed 4/6-bit declares gs 32, stores gs 64 (`tests/test_gemma4_mixed_quant.sh`).
- **Dense checkpoints**: scales absence PER-TENSOR (`getLayerScaleOpt`); dense contracted weights owe `maybeTransposeForBf16` — never depthwise conv or SSM state.
- **ModelOpt's NVFP4 `weight_scale_2` is amax / (448 x 6); MLX's nvfp4 `global_scale` is the amax** (`modelopt_pack.dequantDense`): passed through raw, every weight shrank 2688x and the vision tower saw a blank image.
- **A pack runs in its reference's numerics** (`actDtype`, `LoadOpts.keep_f16`; Bonsai 2 f16 + f32 GDN state; qwen4 f32 GDN state + gate, `MLX_SERVE_GDN_STATE_F32=0` = bf16); constants take the activation dtype. Guard: `test_hadamard_fidelity.sh`.
- **A plugin's fork of an SDK function keeps every arm its loader can bind**, else refuses that layout at load BY NAME (`qwen4_exp` `unservedLayout`): the trunk copy dropped `per_expert_scale`; ModelOpt FP4 decoded garbage, tapes stayed green.
- **Every new LM or media port supports a bf16 pack AND the quantized ones**: loader, converter presets and app catalog carry both; quantized is the default download, bf16 the quality reference.
- **A gather-read table is quantized only where the READER has a quantized-gather path** (media `NEVER_QUANTIZE`; LM `embed_tokens` via `gatherQuantizedRows`).
- **Calibrated quant**: weights per-input-channel and per-expert; bit width beats group granularity ≤3 bits; round (s,b) to the STORED dtype first; an imatrix is valid only for the WEIGHTS it was collected on; uniform ≤2-bit experts to the LAST layer cause agent loops (4-bit tail fixes it).
- **Publish MTP head norms FOLDED** (`--fold-mtp-norms`).
- **Discovery**: a configless repo SHAPE is its media plugin's claim (`served.claimIn`; discovery, `list` and `gen.peekModelType` read it); size sums stat THROUGH symlinks; an incomplete media pack is invisible until its marker lands (`requiredMediaMarker`, `error.IncompleteMediaPack`); a GGUF folder is a SHELF (`id#<file>`). Guard: `tests/test_model_rescan.sh`.
- **The `--no-vision` drop list is a GLOBAL prefix filter**: a tower loads through `loadWeightsWithVision`, never `loadWeights`.

Vision:
- **`x-mlx-pixels` is a GEMMA format** (`wantsServerPreprocess`; server refuses when `vp.mode != .gemma`); `VisionEncoder.forward` returns named errors.
- **A MiniCPM-V image is SEVERAL encoder calls** (thumbnail + crops, `tile_*` on `ImageData`) laid out as `<image_id>N</image_id><image>…</image><slice>…</slice>` rows; the canvas M-RoPE table is built from the EXPANDED ids (`minicpm_canvas.canvasPositions`), so a changed placeholder run shifts every crop silently. Guard: the `MiniCPM-V 4.7` test in server.zig.
- **A placeholder id the splice mask does not name never reaches the prompt** (`spliceVisionRows` takes `video_token_id`; an id of 0 is UNSET, never token 0: audio-only Qwen3-ASR heard silence); a chunked splice resumes its ROW INDEX (#197, `vision_splice_offset`, `tests/test_vision_chunked_prefill.sh`); every generative forward arm splices vision (scan-pinned).
- **Per-arch preprocessing**: patch feature order (LFM2-VL channel INNERMOST); two GELUs in one checkpoint (LFM2-VL); NaFlex resample is `bilinear` WITH antialias and TILING is the resolution; muse = aspect-closest grid, Lanczos, CHANNEL-major shuffle (`VisionPreproc.mode`). Guards: `tests/test_lfm2_vision.sh`, `tests/test_muse_vision.sh`.
- **A processor pixel bound is not the ENGINE's** (`effectivePixelBounds`, `ENGINE_MAX_PIXELS` 1536²).
- **Nemotron omni's native MTP config requires a final vision LayerNorm BEFORE pixel shuffle**, even if a conversion omitted MTP weights; projector quantization is per linear. Guards: Super RADIO reference fixtures and format corpus.

Media backends:
- **A condition works only as TRAINING fed it**: `instrumental` as TEXT (`[Instrumental]`; + lyrics = NAMED 400, `instrumentalConflicts`); Stable Audio 3 `seconds_total` as whole seconds >= 1 (`trainedSeconds`; a fraction is noise, reference too).
- **xatlas segments one connected surface on ONE thread** (`uvwrap.parametrizeParallel`: median-split regions as `faceMaterialData` = one chart group per core; 1M faces 424 s → 3.4 s). Hunyuan paint keeps `parametrize`.
- **A flow-DiT forward over tens of thousands of tokens needs a scope PER BLOCK** (`Dit.forward`, `Decoder.run`): one scope around the whole forward keeps every block's intermediates alive together; fixture-sized oracles never show it.
- **A full-resolution f32 VAE stage is BANDED, exact because everything but its 3x3 convs is per-pixel** (`qwen_image.Stage.banded` + `Conv.forwardStrips`); MLX's peak counter does not see the whole-stage peak. Bar = the fixture oracle with bands forced.
- **A pinned library's op TRANSIENT is invisible to residency bills** (#321, #424): `conv3dDepthChunked` windows decoder convs; H3 `encodeMoments` EVALS per tile pass. Diff conv DISPATCH per mlx bump.
- **Canvas**: `recommendedResolution(totalGB:)` per Mac, capped by `autoCanvasCapPixels`; two-stage denoises at HALF; `maxFramePayloadBytes` is a TRANSPORT limit. A tiled decoder whose positions normalize over the EXTENT refuses (`error.TilingUnsupported`).
- **LoRA alpha lives wherever the exporter put it** (`lora.fileAlphaScale`; per-module tensor WINS); test with `tests/lora_noise.py` + `tests/test_real_loras.sh`.
- **A reference-image editor keeps its input's geometry** (`resolveEditTargetSize`); a few-step distilled model can REQUIRE bf16 (MageFlow Turbo: `roundBf16`, `scalarLike`).
- **Latent→RGB previews are GENERATED fits per latent SPACE** (#208, `tests/dump_latent_rgb_factors.py`; `LTXAV` ≠ `LTXV`).

Parity method:
- **Cosine cannot see SCALE** — concatenated tensors assert `rms_ratio`; a permutation-invariant checksum cannot see a permutation (`pos_weighted`); an oracle that cannot execute the reference must say so; a faithful port can make the checkpoint WORSE (H3 splice default-off).
- **Deep ViT/DiT fixtures dump fp32 on CPU**; transformers ≥5 zeroes custom rotary buffers (assert non-zero), `.float()` drops a complex buffer's sine half; an oracle that IMPROVES when a transform is removed had it disabled.
