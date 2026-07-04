# Plan: 8-bit Qwen3-TTS support

## Why

`AudioModelPreset` (`app/Sources/MLXServe/Services/MediaGen.swift:374-397`) only offers
bf16 repos today:

- `.qwen3TTS06B` → `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` (~4.5 GB incl. codec)
- `.qwen3TTS17B` → `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` (~4.5 GB incl. codec)

mlx-community already publishes 8-bit (and 4/5/6-bit) quantized variants of both, with
the exact file layout our loader expects (`config.json`, `model.safetensors[.index.json]`,
`speech_tokenizer/`), confirmed via the HF API:

- `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` — `model_type: qwen3_tts`, `quantization_config.bits: 8`
- `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` — same, main weights 2.42 GB vs bf16's
  3.86 GB (`speech_tokenizer/model.safetensors`, 682 MB, is identical/unquantized in both —
  same blob hash). Total download ~3.1 GB vs ~4.5 GB.

**No need to quantize ourselves or upload to `ddalcu`** — mlx-community's repos are the
right artifact. The work is entirely on the loader side.

## The blocker

`src/tts.zig` only implements a dense bf16 forward path. Its own header says so:

> "This module reuses the codebase's bf16-Linear idioms (pre-transposed weights +
> `mlx_matmul`) rather than the quantized path: Qwen3-TTS bf16 checkpoints store dense
> weights."

Confirmed: zero `QLinear`/`MixedLinear`/`gather_qmm`/`quantized_matmul` in `tts.zig` today.
Every linear goes through `matmul(x, w_t, s)` (`tts.zig:162-166`) against a weight pulled
straight out of the safetensors dict via `ownWeight`/`ownConvW` (`tts.zig:267+`). Pointing
a preset at an `-8bit` repo as-is would hand these helpers packed uint32 tensors + separate
`scales`/`biases` — wrong shape, wrong dtype, silent garbage or a hard crash.

The talker and code predictor are themselves plain Qwen3 transformers (QK-norm, GQA,
SwiGLU — `tts.zig:10-12`), so this is a scoped, well-understood gap, not a redesign.

## Existing precedent to reuse

`src/krea.zig:181-260` already has exactly this abstraction: `MixedLinear` — bf16 OR
affine-quantized, with `(bits, group_size)` inferred from tensor geometry at load time
(no config field needed), same weight dict, transparent `.forward()`. This is the
established codebase pattern (see CLAUDE.md "Image-backend seam" / "Quantization modes")
for "one engine loads dense or quantized transparently." Port the same shape into
`tts.zig` rather than inventing a new one:

- `load(w, allocator, prefix, in_features, s)`: if `<prefix>.scales` exists in the
  weights dict → quantized branch (own weight+scales+biases, solve bits/group_size from
  `w_cols = in*bits/32` and `s_cols = in/group_size`); else → today's bf16 branch
  unchanged (pre-transpose + `mlx_contiguous` + cast bf16).
- `forward(x, s)`: quantized → `mlx_quantized_matmul(..., "affine", s)`; dense →
  today's `matmul`.
- `deinit`: free `w` (+ `scales`/`biases` if quantized) (+ `add_bias` if present).

Apply it to every talker + code-predictor linear currently built via `ownWeight` +
`matmul`/`linearBias` in the Qwen3 layer loader (`loadQwenLayer`, `tts.zig:1092`) and
the surrounding attention/MLP/lm-head projections. The codec decoder (RVQ + conv/
transformer/snake, `ownConvW` call sites ~`tts.zig:1852-2397`) stays dense — mlx-community
does not quantize it (same blob hash in both bf16 and 8bit repos), so no change needed
there.

## TDD plan (per CLAUDE.md: test first, red → green)

1. **Red**: add a hermetic hidden test in `tts.zig` gated on a new env var
   (`TTS_QUANT_TEST_MODEL`, mirroring `TTS_TEST_MODEL`) pointed at a downloaded
   `Qwen3-TTS-12Hz-0.6B-Base-8bit` dir: load the model, run one forward step, assert
   it doesn't crash and produces finite (non-NaN) logits. This fails today (dense-only
   loader chokes on the quantized tensors).
2. **Green**: port `MixedLinear` into `tts.zig`, wire it into the talker + code-predictor
   linear sites, make the test above pass.
3. **Parity**: a coherence/equivalence check — same reference clip + text, compare
   8-bit output against the bf16 reference qualitatively (cosine sim on mel or a
   temp-0 first-N-token/frame agreement check, same spirit as the existing
   `test_kv_quant_equivalence.sh` style thresholds) — not bit-exact (quantization
   changes numerics), but coherent speech.
4. Extend `tests/test_tts.sh` (or a new `tests/test_tts_quant.sh`) to boot the server
   with the 8-bit dir, hit `/v1/audio/speech`, and sanity-check the returned WAV
   (non-empty, plausible duration) — mirrors the existing TTS integration test shape.

## App-side wiring (after the engine change lands)

- Add 8-bit variants to `AudioModelPreset.all` (`MediaGen.swift:374-398`) — likely as
  new cases (`qwen3TTS06B8bit`, `qwen3TTS17B8bit`) shown alongside the bf16 ones in the
  picker (`AudioGenView.swift:108-118` already just iterates `AudioModelPreset.all`), or
  swap the existing two presets to point at `-8bit` repos outright and drop bf16 from
  the picker. Bias toward **adding** rather than replacing, since bf16 has zero
  numerical surprises — but that's a judgment call to make once the engine change is
  proven out and the fidelity delta is heard by ear.
- Update `approxDownloadGB`/`approxRAMGB` per preset to match the real numbers above
  (~3.1 GB / ~4-5 GB RAM for 1.7B at 8-bit, vs current 3.5/8).
- No `MediaBundle`/`FileSelection` changes needed — the 8-bit repos have the identical
  `config.json` + `speech_tokenizer/` layout the existing `.tts(...)` bundle factory
  (`MediaBundle.swift:76-91`) already handles.

## Open questions (decide before/while implementing)

- Keep both bf16 and 8-bit presets, or replace bf16 outright? (Recommend: add 8-bit
  alongside for now; revisit once real fidelity is heard.)
- Worth quantizing group_size/bits choice validation against what mlx-community shipped
  (need to confirm group_size — likely 64, matching the rest of the codebase's default)
  before committing the `MixedLinear` port's assumptions.
