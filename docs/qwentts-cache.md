# Plan: cache the Qwen3-TTS speaker embedding (voice-clone path)

Status: **not started**. Written 2026-07-26 for a future agent. Nothing here has
been implemented or measured.

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
