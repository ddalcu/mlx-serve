# Native Audio Generation (TTS) — Zig + mlx-c Port Plan

**Goal:** Replace the `mlx_audio` Python venv path for text-to-speech with a native
Zig + mlx-c implementation served over HTTP from `mlx-serve`. No Python. Exposed as
an OpenAI-compatible `/v1/audio/speech` endpoint.

**Status:** 🟡 In progress. This is **Modality 1 of 3** (audio → image → video).
Branch: `feature/Any2Any`. Commit locally when feature-complete + tested.

---

## Scope

| In scope (slice 1) | Out of scope (later / never) |
|---|---|
| Plain TTS (text → WAV), greedy + sampled | Zero-shot voice cloning (`ref_audio`) |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` (first), then 1.7B | Whisper auto-transcription (cloning only) |
| `/v1/audio/speech` endpoint (OpenAI shape) | Speaker encoder (ECAPA-TDNN, cloning only) |
| 24 kHz mono WAV output | MOSS-TTS-Nano (different backbone: GPT-2 + separate 48 kHz LFQ codec) |
| App `AudioGenService` switched from Python → server | Streaming SSE audio (non-stream first) |

**Why Qwen3-TTS-0.6B first:** the talker backbone is literally **Qwen3** (RMSNorm,
QK-norm, SwiGLU, GQA) which we already run natively; at 0.6B `talker_hidden == 1024 ==
code_predictor_hidden`, so the `small_to_mtp_projection` Linear is *absent* (one fewer
component vs 1.7B). Greedy decode (temp 0) is fully deterministic → byte-exact testing.

---

## Architecture (from installed `mlx_audio` source)

Two transformer levels you mostly have + one new codec decoder you don't.

### A. Talker (Qwen3 transformer — reuse existing forward)
- `talker.model.text_embedding.weight` `[151936, 2048]` (Qwen2 BPE vocab)
- `talker.text_projection.linear_fc1/fc2.{weight,bias}` — ResizeMLP, SiLU, **has bias**, 2048→hidden
- `talker.model.codec_embedding.weight` `[3072, hidden]` (audio/control-token embed)
- 28× `talker.model.layers.N`: `self_attn.{q,k,v,o}_proj.weight` (no bias),
  `self_attn.{q,k}_norm.weight` (RMSNorm over head_dim=128, **QK-norm**),
  `input_layernorm.weight`, `post_attention_layernorm.weight`,
  `mlp.{gate,up,down}_proj.weight` (SwiGLU)
- `talker.model.norm.weight`; `talker.codec_head.weight` `[3072, hidden]` (codebook-0 head)
- **RoPE simplification:** base TTS passes no `position_ids` → interleaved MRoPE
  (`mrope_section [24,20,20]`) collapses to **plain RoPE**. Implement plain RoPE.
- 0.6B dims to confirm on download: hidden 1024, intermediate 3072, 28 layers, 16/8 heads,
  head_dim 128, rms_eps 1e-6, rope_theta 1e6. (1.7B: hidden 2048, intermediate 6144.)

### B. Code predictor (5-layer Qwen3 mini, plain RoPE)
- `talker.code_predictor.model.layers.0..4` (Qwen3 layer shape; hidden 1024, inter 3072, QK-norm)
- `talker.code_predictor.model.codec_embedding.0..14.weight` `[2048, hidden]` (codes 1..15)
- `talker.code_predictor.lm_head.0..14.weight` `[2048, 1024]` (15 heads)
- `talker.code_predictor.model.norm.weight`
- `talker.code_predictor.small_to_mtp_projection.{weight,bias}` — **1.7B only** (hidden≠1024)

### C. Codec decoder = the NEW component (`speech_tokenizer/`, separate safetensors)
Decoder config: `latent_dim 1024, codebook_dim 512, decoder_dim 1536, 8 transformer layers
(16 heads, head_dim 64, RoPE θ 10000), num_quantizers 16 (1 semantic + 15 acoustic),
upsample_rates [8,5,4,3], upsampling_ratios [2,2]`. **Total upsample 8·5·4·3·2·2 = 1920
samples/frame → 24000 Hz at 12.5 Hz frames.** Only the **decoder** path is needed.

Decode pipeline (`Qwen3TTSSpeechTokenizerDecoder.__call__`, NLC `[B,T,C]`):
1. **SplitResidualVectorQuantizer.decode** — codebook = `embedding_sum / max(cluster_usage, eps)`
   computed at load (keys `decoder.quantizer.rvq_first.vq.layers.0._codebook.{embedding_sum
   [2048,256], cluster_usage [2048]}` + `rvq_rest.vq.layers.0..14`). Per RVQ: Embedding lookup
   → transpose → Σ → `output_proj` (1×1 Conv1d). `rvq_first`(1) + `rvq_rest`(15) → `[B,512,T]`.
2. transpose → `[B,T,512]`.
3. **pre_conv** CausalConv1d(512→1024, k=3). Causal = left-pad `(k-1)*dil`, then conv pad=0.
4. **pre_transformer** DecoderTransformer: input_proj 1024→512, 8 layers (RMSNorm, SDPA 16×64,
   RoPE θ 10000, **LayerScale** on attn+mlp residuals, SwiGLU), output_proj 512→1024.
5. **upsample** ×2 (ratios [2,2]): each = CausalTransposeConv1d(1024,1024,k=2,stride=2) + ConvNeXtBlock
   (depthwise conv k=7 groups=1024 → LayerNorm → Linear→gelu→Linear → `gamma*` → residual). Time ×4.
6. **decoder** `decoder.decoder.0..6`:
   - `[0]` DecoderInitialConv Conv1d(1024→1536, k=7) causal
   - `[1..4]` DecoderBlock: chans 1536→768→384→192→96, up_rates [8,5,4,3]. Each = SnakeBeta →
     ConvTranspose1d(in,out,k=2·up,stride=up) (trim_right=up) → 3× ResidualUnit(dil∈{1,3,9})
     (SnakeBeta→CausalConv1d k7→SnakeBeta→CausalConv1d k1 + residual)
   - `[5]` DecoderOutputSnake(96); `[6]` DecoderOutputConv(96→1, k=7) causal
7. transpose → `[B,1,samples]`, `clip(-1,1)`.

`SnakeBeta(x) = x + (1/(exp(beta)+1e-9)) · sin(x·exp(alpha))²` (alpha/beta stored as logs).
**Conv weight sanitize:** PyTorch `[out,in,k]` → MLX `[out,k,in]`.

### Per-frame generation loop (12.5 Hz)
1. Talker forward → `codec_head` → sample codebook-0 (suppress `[vocab-1024, vocab)` except EOS;
   EOS = `codec_eos_token_id 2150`).
2. Code predictor inner loop ×15: codebooks 1..15. Step 0 input = `concat([talker_hidden_last,
   embed(code0)])` (fresh KV cache); steps 1..14 = `codec_embedding[i-1](prev_code)`; head `lm_head[step]`.
3. 16 codes → `all_codes [1,16]`, append.
4. Next input = `text_embed_next` (or `tts_pad_embed`) **+** Σ codec embeds of all 16 codes
   (code0 via `talker.codec_embedding`; codes1..15 via `code_predictor.codec_embedding[i]`).
5. Stop on codebook-0 == 2150.
6. After loop: `speech_tokenizer.decode(codes [1,T,16])` → waveform; trim to `(codes[...,0]>0).sum()*1920`.

### Input assembly (`_prepare_generation_inputs`)
- Text tokenizer = Qwen2 BPE. Literal chat template:
  `"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"`.
- `text_embed = text_projection(text_embedding(ids))`.
- Codec control prefix (no language for `lang_code` not in `codec_language_id`):
  `codec_nothink_id 2155, codec_think_bos_id 2156, codec_think_eos_id 2157, codec_pad_id 2148,
  codec_bos_id 2149`. Text + codec streams **summed element-wise** as model advances.

---

## Required MLX ops / FFI

| Op | Status |
|---|---|
| `mlx_conv1d` (incl. grouped/depthwise via `groups`) | ✅ already bound |
| `mlx_conv_transpose1d` | ✅ **added this session** |
| `mlx_pad` | ✅ already bound |
| SnakeBeta (`exp`,`sin`,`square`,recip,add,mul) | ✅ compose from bound ops |
| RMSNorm / LayerNorm / SDPA / RoPE / SwiGLU / silu / gelu | ✅ existing helpers |
| LayerScale (elementwise `scale *`) | ✅ trivial |

**No FFT/STFT, no GroupNorm, no weight-norm, no interpolation in the decode path.** Snake is the
only nonstandard activation. Codebook `embedding_sum/cluster_usage` is a one-time load-side compute.

---

## Integration points

- **New file** `src/tts.zig` — `Qwen3TtsModel` (talker + code predictor + codec decoder structs,
  weight load by safetensors key, forward + per-frame loop + `decode`). Mirror `vision.zig`/`drafter.zig`
  struct-of-`mlx_array` + helper-method conventions.
- **Model dispatch** — `model.zig` parse `model_type == "qwen3_tts"`; a TTS model is request→file,
  not token streaming. Route through the scheduler's inference thread (serial GPU) but a dedicated
  `runTtsSynthesis` path (no KV-stream slot machinery).
- **HTTP** — `server.zig` route table (`~:950–1130`): add `POST /v1/audio/speech`
  (OpenAI: `{model, input, voice, response_format, speed}`) → returns `audio/wav` bytes.
- **WAV writer** — `src/wav.zig` (trivial: 44-byte header + int16/float32 PCM). 24 kHz mono.
- **App** — `AudioGenService.swift`: replace `python.runScript` with an HTTP call to
  `/v1/audio/speech`; keep the same `Phase`/recent/log UI. `PythonManager` stays only for image/video
  until those land too.
- **`/v1/models` + discovery** — register `qwen3_tts` as a supported arch with `audio` capability;
  add to `supported_model_types` (Zig) + `supportedModelTypes` (Swift).

---

## Implementation checklist

### Foundation
- [x] Bind `mlx_conv2d/conv3d/conv_transpose1d/2d/3d` in `src/mlx.zig`
- [x] Verify FFI compiles (ReleaseFast build green)
- [x] `src/wav.zig` — WAV encoder (mono f32→int16 PCM) + unit tests (header bytes, clamp, stereo) — 3 tests green
- [x] Conv1d/ConvTranspose1d/causal-pad helper wrappers + SnakeBeta/gelu helpers (in `src/tts.zig`)
- [x] `loadWeights` follows symlinks (HF-cache snapshots) — `src/model.zig`

### Model load
- [x] Use cached `Qwen3-TTS-12Hz-1.7B-Base-bf16` (0.6B downloading as lighter default); confirmed dims
- [x] `src/tts.zig` config parse (`qwen3_tts`) + talker/code-predictor/codec key wiring (against real keys)
- [x] Codebook precompute (`embedding_sum / clip(cluster_usage, 1e-5)`) at load
- [x] Conv weight sanitize at load: regular `(0,2,1)`, transpose-conv `(1,2,0)` — verified vs reference shapes

### Forward — talker + code predictor
- [x] Talker Qwen3 forward (plain RoPE, QK-norm, GQA) with KV cache (prefill + incremental step)
- [x] Input assembly: chat template embeds, control-token prefix, text+codec stream sum
- [x] Codebook-0 sampling (suppress mask + EOS 2150) — greedy + repetition penalty + temp/top-k
- [x] Code predictor 15-codebook inner loop (full re-forward, ≤16 pos)
- [x] Per-frame loop + next-embed reconstruction + EOS stop
- [x] **Oracle A:** Zig reproduces **frame-0 codes BIT-EXACT** (16/16) vs the greedy Python oracle; beyond frame 0 the greedy bf16 path drifts by near-tie flips (documented), so temperature sampling is the production path

### Forward — codec decoder
- [x] SplitRVQ dequant (semantic + 15 acoustic) → `[1,T,512]` (NLC)
- [x] pre_conv + 8-layer pre_transformer (RoPE, LayerScale, SwiGLU)
- [x] 2 upsample blocks (CausalTransposeConv1d + ConvNeXt)
- [x] 4 DecoderBlocks (SnakeBeta + ConvTranspose1d + residual units, dil 1/3/9)
- [x] output snake + conv + clip → waveform
- [x] **Oracle B:** Zig codec on the reference codes reproduces the reference waveform **BIT-EXACT** (corr 1.00000, rms_err 0.00000, exact 94080 samples)

### End-to-end (engine)
- [x] text ids → talker+predictor → codec → WAV: **49 frames → 3.92s, terminates at EOS, non-silent (peak 0.45)**, same length as the Python reference; temperature sampling avoids the greedy degenerate loop
- [x] 6/6 audio tests green (`zig build test` full suite 533/541, 0 failures)

### Endpoint + app (serving layer — REMAINING)
- [ ] Native tokenization of the chat template (build ids from fixed special tokens + BPE-encode plain text — sidesteps missing `tokenizer.json`)
- [ ] `Synthesizer` (TtsModel + CodecDecoder + Tokenizer) `synthesize(text) → []f32`
- [ ] `POST /v1/audio/speech` handler → WAV bytes; `model.zig` dispatch + `audio` capability
- [ ] Integration test `tests/test_tts.sh` (curl → WAV; sample count, sample rate, non-silent)
- [ ] App `AudioGenService` → HTTP; remove its Python script path; `swift build` green
- [ ] CLAUDE.md: add `qwen3_tts` to arch table + the conv-transpose-layout + frame-0-exact gotchas
- [ ] **Commit locally** (modality 1 fully complete)

> **Status:** the native generation **engine** is complete and validated bit-exactly (codec) /
> frame-0-exact (talker). What remains is the serving layer (HTTP endpoint + app), checkpointed
> separately because the engine is the novel, high-risk 90%.

---

## Equivalence-test oracle

Deterministic reference (greedy, temp 0) — dumps both raw floats and the intermediate codes for
two-layer bisection (talker/code-predictor vs codec decoder):

```bash
~/.mlx-serve/venv/bin/python -c '
import numpy as np, mlx.core as mx
from mlx_audio.tts.utils import load
text = "Hello world, this is a test of the local text to speech engine."
m = load("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")   # or 0.6B
mx.random.seed(0)
chunks=[]
for r in m.generate(text=text, voice="af_heart", temperature=0.0, speed=1.0,
                    lang_code="en", max_tokens=1200, verbose=True,
                    top_k=50, top_p=1.0, repetition_penalty=1.05):
    chunks.append(np.array(r.audio, dtype=np.float32))
audio = np.concatenate(chunks) if len(chunks)>1 else chunks[0]
np.save("/tmp/qwen3tts_ref.npy", audio); print("samples", audio.shape, "sr", m.sample_rate)
'
```
- `lang_code="en"` is **not** in `codec_language_id` (which uses `"english"`) → no-language `nothink`
  prefix. Match this exact value in Zig.
- **Oracle A (codes):** insert `np.save("/tmp/qwen3tts_codes.npy", np.array(mx.stack(generated_codes,
  axis=1)))` before `speech_tokenizer.decode` (qwen3_tts.py ~L1442). Zig talker+predictor must reproduce
  the integer `[T,16]` matrix exactly under greedy.
- **Oracle B (decoder):** feed that codes `.npy` through `m.speech_tokenizer.decode(mx.array(codes))`;
  compare to Zig codec output per-sample. Pins conv/snake/transpose independent of sampling.

---

## Risks / notes
- 0.6B repo not yet downloaded — confirm exact dims on first fetch (defaults imply hidden 1024).
- `mx.compile` on the decode path in Python is irrelevant to the port (we eval directly).
- Conv layout: MLX is NLC + weight `[out,k,in]`; the Python `sanitize` already does the transpose —
  replicate at load.
- Sampling parity beyond greedy (top_k/top_p/repetition_penalty) — match the Leviathan-style order
  used elsewhere in `generate.zig`; greedy is the testable contract.
