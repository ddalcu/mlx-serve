# Plan: cache the Qwen3-TTS speaker embedding (voice-clone path)

Status: **Option A implemented 2026-07-30** (`SpkEmbCache` in `src/tts.zig`,
content-keyed 4-entry LRU on the Synthesizer, kill switch
`MLX_SERVE_TTS_SPK_CACHE=0`; guards: 4 unit tests + the TTS_TEST_MODEL-gated
bit-identity test + `tests/test_tts_clone_cache.sh`). Stage 0 measured on the
1.7B-8bit with a real 11.5 s clip: ref decode ~0.6 ms, speaker embed
~10.5 ms warm / ~80 ms cold, generate ~1770 ms (70 frames), codec decode
~88 ms per sentence. After: hits cost ~0.03 ms, output byte-identical to the
pre-cache binary. The transport half (~0.6 ms) is negligible, so the cache
keys on the decoded samples and **Option B is closed** — not worth the API
surface.

**Pre-warm SHIPPED 2026-07-30** (the first-sentence lever): server side,
`POST /v1/audio/speech` accepts `{"warm_only":true,"ref_audio":...}` — embeds
+ caches WITHOUT synthesizing, replies `{"warmed":true,"cache":"hit"|"miss"}`;
misuse is a named 400 (no ref, non-cloning checkpoint, Kokoro). App side,
`SpeechSynthesizing.prewarm()` (default no-op) fires on voice-mode activation:
`ClonedVoiceSynthesizer.prewarm` → `VoiceCloneTTS.prewarm(voice:)` loads the
TTS model and POSTs `warmBody` (pure/static, tested — `warm_only` + `ref_audio`,
never `input`). Only the clone arm warms. Guards: `warmSpeaker` gated unit
test, `tests/test_tts_clone_cache.sh` [5/5], `ClonedVoiceSynthesizerTests`
warm-body + routing cases.

## The problem, precisely

Voice mode synthesizes **one sentence per HTTP request** (that's deliberate —
`ClonedVoiceSynthesizer` pipelines synthesis against playback, so sentence N+1 is
being made while N is sounding). Every one of those requests re-does the whole
reference-voice pipeline for the *same clip*:

| Step | Where | Per sentence |
|---|---|---|
| Base64 the clip into the JSON body | `VoiceCloneTTS.requestBody` (`app/Sources/MLXServe/Services/ClonedVoiceSynthesizer.swift`) | ~512 KB for an 8 s 24 kHz mono 16-bit wav |
| `jsonUnescape` + `base64DecodeAlloc` + `decodeWavToF32` | `gen.handleAudio` (`src/gen.zig` ~1765) | again |
| mel spectrogram + full ECAPA-TDNN forward | `tts.SpeakerEncoder.embed` (`src/tts.zig:1677`), called from `tts.Synthesizer.synthesize` (`src/tts.zig:1790`) | again |

The output of all that is **one `[1, enc_dim]` array**, spliced as a single
codec-prefix position in `TtsModel.generateCodes` (`src/tts.zig:755`). It is a
pure function of the clip bytes, so it is cacheable in the strictest sense: same
clip ⇒ same embedding ⇒ **bit-identical** wav. That is the property every test
below leans on.

What is *already* cached and must not be re-solved: the checkpoint itself stays
resident (app: `VoiceCloneTTS.loadedModelId`/`loadedDir`; server: the model
registry). Kokoro has none of this cost — named voices are a table lookup — so
everything here is clone-path only.

## Expected size of the win — read this before committing to the work

Nobody has measured it. The honest prior is **modest**: ECAPA over ~800 mel
frames at 512 channels is small next to the autoregressive codec LM plus
`codec.decode`, which dominate a sentence. Best guess is single-digit to
low-tens of milliseconds per sentence against hundreds, i.e. real in aggregate
over a 10-sentence answer but not a step change. The ~0.5 MB base64 round trip
per sentence is plausibly the same order as the ECAPA math itself.

**Measure first (Stage 0). If the saving is under ~5 ms per sentence, close this
plan and do the pre-warm in "Related, probably bigger" instead.** Bench rules
apply (`/bench` skill): same-boot A/B, never quote a win without naming what it
is over, and a µbench win can lose in the live path.

## Stage 0 — measure, before writing any cache

1. `zig build -Doptimize=ReleaseFast` (never Debug for timing).
2. Add a temporary `log.info` timing around the three costs in one request:
   base64+WAV decode, `SpeakerEncoder.embed`, and `generateCodes`+`codec.decode`.
3. Drive it with a realistic clip and a realistic sentence:
   `tests/test_tts.sh` is the smoke path; for clone timing send a real
   `ref_audio` (base64 of `~/.mlx-serve/voice-clips/*.wav`).
4. Record: embed ms, decode ms, generate ms, total ms, clip seconds, sentence
   chars. Put the numbers in the PR/commit message — they are the justification
   for the whole change and the baseline the acceptance A/B compares against.

## Option A — server-side cache keyed by clip CONTENT (recommended first)

One hash lookup. No API change, no app change, no new lifetime to reason about.

### Design

- Cache lives on the **loaded model**, not in a global: add it to
  `tts.Synthesizer` (`src/tts.zig:1742`) so `Synthesizer.deinit` — reached from
  `gen.AudioEngine.deinit` (`src/gen.zig:719`) — frees it on unload. A global
  would outlive the model whose stream produced the arrays.
- Key on the **base64 string** the request carried (stable encoding of the same
  bytes) or on the decoded samples; hashing the base64 lets a hit skip
  `base64DecodeAlloc` + `decodeWavToF32` too, which is part of the saving.
  Content-keyed means **invalidation is free** — a re-recorded clip is a
  different key, never a stale hit.
- Capacity: tiny. 2–4 entries is plenty (one agent's voice, maybe a switch
  mid-session); an `enc_dim`-sized f32 array is a few KB. LRU by last use.
- Shape: `embedCached(self: *Synthesizer, key: []const u8, samples: []const f32)
  !mlx.mlx_array` returning a **borrowed** handle owned by the cache, with
  `synthesize` no longer freeing it. That ownership flip is the one real hazard —
  see below.

### Traps this must not walk into (all are existing repo rules)

- **Ownership.** `synthesize` currently `defer`-frees `spk_emb`. A cached array
  must NOT be freed per request. Either return an owned copy per call (cheap,
  keeps the existing defer, loses nothing measurable) or make the borrow
  explicit and delete the defer. Pick one and say which in the code comment;
  double-free and use-after-free both live here.
- **Materialize what you cache.** A cached mlx array must be a materialized
  owned buffer, not a view of a per-request tensor — the slice-born-view class
  (`materializedOwnedCopy` in the engine rules). A view would pin the whole
  parent mel/activation buffer for the life of the cache.
- **Inference thread is the sole mlx caller**, including frees. The cache is
  touched only from the gen queue's thread; do not add a mutex and call it from
  the HTTP thread.
- **`mlx_clear_cache` doesn't help and isn't the fix** — this is a handle-lifetime
  question, not allocator-cache growth.

### Tests (TDD order, `zig build test`)

1. **Failing first:** a unit test in `src/tts.zig` that calls the cached path
   twice with the same key and asserts (a) the second call does not re-run the
   encoder (a counter incremented in `embed`), and (b) the returned embedding is
   **bit-identical** to the uncached one.
2. Different key ⇒ miss ⇒ different embedding (guards a key collision).
3. Eviction: N+1 distinct keys, the oldest is gone, no leak (run under the
   existing test allocator so a leaked handle fails the test).
4. `deinit` with a populated cache frees everything (debug-allocator `leaked`
   lines in the server log are the tell in production).
5. **Output equivalence, end to end:** extend `tests/test_tts.sh` (or a new
   `tests/test_tts_clone_cache.sh`) to POST the same `ref_audio` + text twice and
   assert the two wavs are **byte-identical**, then a third with a different clip
   and assert it differs. This is the guard that a cache can never change what a
   voice sounds like.
6. Kill switch: `MLX_SERVE_TTS_SPK_CACHE=0` restores the uncached path, and the
   equivalence test runs both ways (every engine lever in this repo ships with
   one, bit-identical to its off state).

### Acceptance

- Stage 0's numbers, re-measured on the same boot with the switch on and off.
- Report the per-sentence delta and the delta over a 10-sentence answer. If the
  live A/B doesn't show what the µbench promised, believe the live one.
- Optional: a `mlx_serve:tts_speaker_cache_hits/misses` counter in
  `src/metrics.zig` (zero cost when `--metrics` is off).

## Option B — reference the clip by id (only if A's transport half matters)

Removes the ~0.5 MB per sentence as well as the encoder work, at the cost of API
surface and a lifetime.

### Design

- New optional request field on `/v1/audio/speech`: `ref_audio_id` (a hash the
  app computes over the clip). `ref_audio` stays and still works.
- Server keeps the Option A cache but keyed by that id. **Miss must be a named
  400, not silence**: `{"error":"unknown ref_audio_id; send ref_audio once"}` —
  the app then retries with the bytes. Silently synthesizing in the plain voice
  is the failure mode the media-gen rules exist to prevent.
- App side: `VoiceCloneTTS.requestBody` (pure and already unit-tested — extend
  those tests) sends `ref_audio_id` alone when it believes the server has the
  clip, and both fields on the first request of a session or after a 400.
  `VoiceClipLibrary` already gives clips stable paths, so the id can be a hash of
  the file (cache it in memory alongside the path; the file is immutable once
  installed).
- Lifetime: entries die with the loaded model (same as A). The app must therefore
  treat "unknown id" as normal, not exceptional — a model unload between
  sentences is ordinary.

### Why this is second

It adds a stateful handshake to a body that is currently pure OpenAI-shaped
data, and the failure path (id known to the app, forgotten by the server) has to
be exercised or it will bite in exactly the situation nobody tests: unload
mid-answer. Do it only if Stage 0 shows the transport is a material share of the
per-sentence cost.

### Extra tests beyond A's

- `requestBody` sends id-only / both / bytes-only in the three states (pure, in
  `ClonedVoiceSynthesizerTests`).
- A 400 on unknown id triggers exactly one retry WITH bytes, and that retry
  succeeds (drive `VoiceCloneTTS` with a stubbed transport).
- Unload between sentences: the next sentence recovers rather than falling back
  to the system voice. This is the one that justifies the option's complexity.

## 2026-07-30 round 2: the frame loop itself (GPU chain + fusions)

Shipped, all byte-identical to the composed path and pinned by
`tests/test_tts_fastpath_equivalence.sh` + the `gpu predict` unit tests:

- **GPU-chained code predictor** (`MLX_SERVE_TTS_GPU_PREDICT=0` kills): the host
  loop paid 16 CPU↔GPU syncs per frame (code0 + 15 codebooks); the chain keeps
  every sample lazy (sampling was ALREADY pure mlx ops keyed by a host counter,
  so laziness changes no draw) and reads all 16 codes in ONE sync. On the EOS
  frame the predictor build is discarded; the extra rng advances feed no kept
  token, so output stays byte-identical.
- **Fused kernels reused from transformer.zig** (talker + predictor):
  `fusedSwiGLU` (exact sig-table), `fusedQkNormRope` at talker decode L==1
  (hd 128; angles probed once per frame via `ropeAngleRow`; eps MUST be a
  0-dim scalar — a `[1]` array binds `constant float*` and the kernel fails to
  compile at dispatch), `fusedAddRmsNormUngated` behind
  `MLX_SERVE_TTS_ADD_RMSNORM=0` (tts default-ON; laguna's opt-in unchanged).
- **`MLX_SERVE_TTS_ASYNC_STRIDE`** (default 1): async-eval kick per sampled
  codebook + one after code0, overlapping CPU encode with GPU exec. Stride 0
  measured ~4% worse in the one alternating-boot window.

**Round 3 (same day): banded levers + the corrected numbers.** Two more
predictor modes, both reduction-order deviations (near-tie flips at temp>0
change the rendition; greedy agreement measured 1.0000 over 48 frames):
- **KV-cached predictor steps** (`MLX_SERVE_TTS_CP_CACHE=0` kills; DEFAULT):
  step 0 prefills [talker_hidden, code0-embed], every later step is ONE
  position through the cp stack — ~30% fewer graph nodes per step, and the
  fused QK-norm+RoPE kernel engages at L==1 (cp head_dim is 128).
- **Compiled full-re-forward chain** (`MLX_SERVE_TTS_COMPILE=0` kills): the
  whole 15-step chain as ONE `mlx_compile` closure (fixed signature: hidden,
  code0, 15 keys), traced once, replayed per frame. Only −3%/frame — replay
  still schedules every op; compile fuses elementwise, not the matmul-class
  nodes that dominate encode. Kept as the second-preference path.

**Final same-window alternating A/B (engagement-verified per arm):** composed
25.1 ms/frame (~1758 ms generate) vs full default **12.9 ms/frame** (~1031 ms)
— **−49%/frame, 1.95x**. Sentence with the real clip: ~1.86 s → ~1.12 s,
~5.9x realtime. Ladder: bit-exact set (GPU chain + fusions + stride-1 kicks)
25.1 → 20.9; compiled 20.3; KV-cached 12.9.

**Measurement trap that cost half a day (now a bench rule):** an earlier A/B
concluded "wall-clock neutral" because the composed arm read ~1473 — but the
harness passed multi-switch env prefixes as `env $VAR` under ZSH, which does
NOT word-split, so `env "A=0 B=0"` set A to the string `"0 B=0"`, every kill
switch stayed ON, and the "composed" arm silently ran the fast path. Single-
switch arms were valid, which made the numbers look internally consistent. An
A/B arm is proven by ENGAGEMENT / no-engagement lines in its own log, never by
its launch env; the fastpath guard asserts both directions.

**Remaining roadmap (after the KV-cache round landed −49%/frame):** node count
is still the currency; the frame is now ~12.9 ms with the talker stack
(~420 nodes) and the cached predictor (~75 nodes × 15 steps) sharing it.
(1) Talker static-KV + fixed-shape decode (preallocated cache + slice_update +
mask input) so its step can also shrink; (2) mega-kernels — one custom kernel
per L==1 predictor layer (quantized GEMVs + norms + rope + sdpa in one
dispatch) would collapse the remaining ~1,100 nodes to ~150/frame, the true
2x-again; (3) cross-frame pipelining (read frame N's codes after building
frame N+1; EOS one frame late). (2) is a multi-session kernel project.

## Related, probably bigger than either option

Neither option helps the **first** sentence, which is always a miss — and
time-to-first-sentence is what a user actually perceives. Pre-warming the
embedding when voice mode starts (or when an agent with a clone voice is
selected — the clip is known at that moment, see `ActiveAgentVoice` /
`AppState.applyAgentSelection`) moves that cost off the critical path entirely.
It needs the cache from Option A to land anywhere, so: **A, then pre-warm, then B
only if measured.**

## Files a future agent will touch

- `src/tts.zig` — `Synthesizer` (cache field, `deinit`), `SpeakerEncoder.embed`
  call site in `synthesize`, tests at the bottom of the file.
- `src/gen.zig` — `handleAudio` (~1765) passes the key through; `AudioEngine`
  needs no change beyond what `Synthesizer.deinit` already does.
- `tests/test_tts.sh` or a new `tests/test_tts_clone_cache.sh` — byte-identity
  guard.
- Option B only: `app/Sources/MLXServe/Services/ClonedVoiceSynthesizer.swift`
  (`VoiceCloneTTS.requestBody` + retry), `app/Tests/MLXCoreTests/ClonedVoiceSynthesizerTests.swift`.
- `CLAUDE.md` `## Rules` (1–3 lines) + the full story in
  `docs/gotchas/engine-mlx.md` if anything surprising turns up, per the growth
  policy.
