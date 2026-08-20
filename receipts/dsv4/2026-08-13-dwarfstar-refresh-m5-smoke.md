# DeepSeek V4 Flash DwarfStar refresh — M5 Max smoke

Date: 2026-08-13 (amended 2026-08-14)

This is an integration and fixed-work smoke result, not a release-quality
benchmark or a claim of parity with the original FP8 model.

## Frozen inputs

- Historical smoke lineage base: `d9c7986de5c8be9dc02640e7c017bdfd8b1da483`
- Superseded direct parent base: `603b4c865d0682c8de96e6cbbf804e01fce6d632`
- Current merge-ready upstream base:
  `1607911d9d5725cf66d9960b72cd8bdf2f8d197c` (`origin/main`, mlx-serve
  26.8.6).
- Historical pre-`34e30e7` parent `HEAD`:
  `b06b01de63c8f2dc7545abd1982c03e9c476ad6b` on
  `codex/dsv4-dwarfstar-merge-ready`. It is the clean cherry-pick of
  `118dd8a` (backend integration) and `b06b01d` (reviewed DS4 pin and
  evidence) onto the current upstream base.
- Current parent integration candidate: frozen parent `HEAD`
  `c70db2718cab696dfdb6d03cc0fbbfba9721c188` on the same branch, with
  merge-base `1607911d9d5725cf66d9960b72cd8bdf2f8d197c`. The `34e30e7`
  gitlink, live pin references, and current evidence are uncommitted local
  changes on that candidate; no future parent commit SHA is claimed here.
- DwarfStar base: `84cc882352757baf628a1776badf7cc54d584e28`
- Historical first published DwarfStar integration commit:
  `0c654079b8fd78aa3440107b5db7158c115cb93e`. The superseded reviewed pin is
  `69b376268f52e44cc42077efcecfd7c4990ab6ae` on
  `PhilipJohnBasile/ds4:codex/dsv4-ssd-dspark-m5`. Its raw smoke artifacts
  below are immutable historical evidence; they do not establish the later
  `34e30e7` wrapper result and must not be overwritten.
- DwarfStar local integration patch: quality-mode SSD decode uses bounded
  full-layer maps instead of the compact selected-expert static map; an exact
  DSpark support GGUF can use a separate persistent read-only Metal view while
  target-layer views rotate under SSD streaming
- Built `mlx-serve` binary SHA-256:
  `ad5175302ed6f972ae29da19f9e3d34b94891bff580890b80a087d7cb15806cc`
  (11,202,664 bytes; historical smoke binary, superseded below)
- Historical `69b3762` merge-ready ReleaseFast binary SHA-256:
  `0ab608666fdcc1deda0b49b143608acc56c329ac53e07a29226f21994d411e96`
  (11,526,568 bytes; `mlx-serve 26.8.6`, `mlx 0.32.0`, `llama.cpp b10034`,
  `ds4 69b376268f52`). This is historical only; the current rebuilt binary is
  recorded in the `34e30e7` section below.
- Standalone DwarfStar server used for the bounded-memory mixed-model quality
  follow-up SHA-256:
  `6f82a4bf286abaea4c178de80f9bb82dc8463e3beab173083327acecf5ece1c1`
  (2,331,272 bytes)
- Target GGUF SHA-256:
  `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`
  (86,720,111,488 bytes)
- DSpark support GGUF SHA-256:
  `8b3adf5942bec22ae2ea867cd7079cf13530ba83ffcffaf00f5de48664a1a34e`
  (5,989,114,272 bytes)
- Host: Apple M5 Max, 128 GiB unified memory
- OS: macOS 27.0 build 26A5406e
- Context: 8,192 tokens
- Prompt lookup: disabled
- Sampling: temperature 0

The refresh required adding DwarfStar's new
`placement_session_count_hint` field to the Zig FFI mirror. The layout guard
failed before the fix (C size 272, Zig size 264) and passes after it. A new
integration test also checks that all 19 upstream Metal-loader entries have
matching embedded sources.

## Verification

- Focused root `-Dtest-filter=ds4` suite: PASS in Debug, ReleaseSafe, and
  ReleaseFast (14 case-sensitive matching source test names).
- ReleaseFast executable build: PASS.
- Embedded Metal source inventory against upstream loader: PASS (19/19).
- The final audit also ran the broad suite in all three modes. Each run had
  1,486 passed, 122 skipped, and two pre-existing numerical failures in the
  fused decode-chain and MiniMax paths. This refresh does not suppress or
  relabel them.
- The focused no-model checks for this remediation are recorded separately
  from those historical full-suite failures.

## Post-review repair validation — no model or GPU run

- Root focused compile-time `-Dtest-filter=ds4`: PASS in Debug, ReleaseSafe,
  and ReleaseFast (14 case-sensitive matching source test names).
- All five scheduler `LoadParams` factories are source-guarded to forward the
  same SSD-streaming, MTP, and DSpark launch policy; the focused suite fails if
  any constructor omits one of those fields.
- DwarfStar provenance helper: PASS in a hermetic temporary Git repository for
  a clean checkout, a tracked edit, an untracked file, and a partial Git
  failure where revision lookup succeeds but status fails. Dirty checkouts
  append `-dirty`; status failure returns no identity instead of false-clean.
  The release workflow derives that helper value once and passes it as explicit
  `-Dds4-commit`; plain Zig builds deliberately report `unknown` unless given
  that explicit pin, so cached Git state cannot be advertised as provenance. A
  same-isolated-cache ReleaseFast regression covers clean → broken-index status
  failure (`unknown`) → clean recovery, plus an explicit helper-derived pin.
- Release-workflow static gates: PASS after the app-build provenance change.
- Standalone no-model admission ledger: PASS for exact boundary,
  insufficient-headroom rejection, and overflow rejection.
- Standalone no-model explicit-DSpark readiness fixture: PASS for a complete
  support summary plus missing-tensor, invalid-tensor, metadata, stage, block,
  and target-layer rejection cases. The same production startup predicate
  rejects legacy-MTP content under explicit DSpark with SSD both off and on,
  while preserving ordinary non-SSD legacy MTP.
- Standalone persistent-support lifecycle source-order guard: PASS; it pins
  synchronization and release before the support mmap is closed, and the
  explicit-DSpark readiness rejection before persistent registration.
- Standalone Metal `ds4-server` syntax/link build and CPU build: PASS. The
  Metal compile emitted 27 existing macOS-27 `didModifyRange:` deprecation
  warnings.
- The pre-publication ReleaseFast wrapper build reported
  `ds4 84cc88235275-dirty`, proving a local dirty checkout is no longer
  misreported as unknown. The first published nested candidate then reported
  `ds4 0c654079b8fd`; its historical binary SHA-256 was
  `6a86d136e562bd430765f6c76483758772fbcadc6240db4ade54f16d6e39f86f`
  (11,205,144 bytes).
- After the admission/lifecycle hardening was published and repinned, the
  clean-submodule ReleaseFast wrapper build reported `ds4 69b376268f52`. Its
  binary SHA-256 is
  `db90948aa1f12d511bec24af610e86764bbf50732325b9ed1b39ff8f66968063`
  (11,238,376 bytes). The DS4 FFI layout guard is among the 14 source-test
  names selected by the focused `ds4` filter, which passed in Debug,
  ReleaseSafe, and ReleaseFast. The historical full 1,617-test run recorded
  1,493 passes, 122 skips, and two unrelated GPU numerical failures in the
  DSV4 fused decode-chain and MiniMax-H3 sparse-attention tests; those failures
  are retained, not counted as post-repin passing evidence.

No model, GPU inference, public endpoint, or network benchmark was run during
the post-review repairs recorded above. The next section records the current
`34e30e7` re-pin; the following serialized `69b3762` gate remains explicitly
historical evidence rather than post-repair performance approval.

## `34e30e7` lifecycle remediation re-pin and M5 functional smoke

- Nested DwarfStar pin: `34e30e7a6635e910d38b1726f468dd4732ff5dca`, public
  on `PhilipJohnBasile/ds4:codex/dsv4-ssd-dspark-m5`. The lifecycle repair
  makes session-owned host-buffer release and DSpark admission release precede
  the final shared-prefill-workspace borrow that permits engine close. It adds
  deterministic concurrent close/free and creation-failure coverage; it is not
  a DSpark enablement or default-on change.
- Parent gates: PASS — `make test-dspark-final-remediation` (SSD admission,
  DSpark readiness, shared-workspace admission, startup guards, persistent
  support lifecycle, and final wiring); root `-Dtest-filter=ds4` in Debug,
  ReleaseSafe, and ReleaseFast (including the DS4 FFI layout guard and 19/19
  embedded-Metal inventory); the isolated-cache provenance regression
  (plain clean → broken index → clean all `unknown`, then the explicit helper
  pin); release-workflow static gates; and the staged-MLX/NAX static gate.
- Rebuilt exact wrapper: SHA-256
  `361a21f31e5c0df8559e488ad932f414b992b62e59f593b3371623d48ad48e91`,
  11,527,672 bytes, `mlx-serve 26.8.6`, `mlx 0.32.0`, `llama.cpp b10034`,
  and `ds4 34e30e7a6635`.
- Public closure at the live check: `git ls-remote` resolved the public fork
  branch to `34e30e7a6635e910d38b1726f468dd4732ff5dca`; `antirez/ds4#798`
  was open, draft, and clean with that exact head. Parent `ddalcu/mlx-serve#175`
  was open draft at public head `c70db2718cab696dfdb6d03cc0fbbfba9721c188`
  with merge state `UNSTABLE`; this uncommitted candidate is not represented as
  already published in that PR.
- One exclusive M5 target-only SSD smoke: PASS. The current wrapper, exact
  97,591,747,456-byte mixed-0731 target
  (`659e22fbd01c9e13ea37a57c8d9c41e0a8819dffa3473d3c5286ee44b2d3398f`),
  loopback-only launch, admission, health check, HTTP 200 request, response
  `11`, memory observation, and clean teardown are recorded with exact argv,
  environment, and raw artifact hashes in
  `receipts/dsv4/2026-08-14-dwarfstar-34e30e7-m5-provenance.md`.

This current smoke is a one-token compatibility gate only. Its 1.421339-second
client wall time and one-token decode timing do not support a throughput claim;
DSpark remains experimental and default-off.

## Historical `69b3762` serialized post-review M5 target-only gate

After rebasing the integration onto current `origin/main`, one exclusive local
functional smoke was run against the exact merge-ready ReleaseFast binary and
mixed-0731 target. This was a target-only SSD-streaming compatibility gate,
not a benchmark and not DSpark evidence. The earlier `db90948a...` smoke on
the superseded parent candidate also passed, but the evidence below replaces
it for merge decisions.

- The checked-in server log proves this run identified itself as `mlx-serve
  26.8.6` using the DS4 GGUF backend. The binary hash, size, and detailed
  component-version command were contemporaneous separate-command observations;
  they are not persisted in the checked-in raw smoke evidence.
- Target path:
  `/Users/pjb/git/ds4/gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf`
  (97,591,747,456 bytes). The full-file SHA-256 command that matched
  `659e22fbd01c9e13ea37a57c8d9c41e0a8819dffa3473d3c5286ee44b2d3398f` was
  likewise a contemporaneous separate-command observation, not persisted in
  the checked-in raw evidence.
- Launch policy: `--ssd-streaming --no-ds4-mtp --ctx-size 8192 --temp 0`,
  loopback-only on port 18074. The 93%-free memory reading and process review
  were contemporaneous PTY observations; neither is persisted in the checked-in
  raw evidence.
- Admission: PASS. The checked-in server log proves `has_mtp=false` and the
  bounded 8.00 GiB SSD expert budget. The skipped full-residency/warmup state
  and the detailed 9.47 GiB plan (0.42 GiB KV, 0.06 GiB buffers, 0.99 GiB
  resident model, 4.62 GiB dynamic cache, 3.38 GiB prefill reserve) were
  contemporaneous PTY observations, not persisted raw evidence.
- Public OpenAI-compatible request: PASS. For “Reply with exactly the integer
  that is one more than 10. No punctuation.” with `max_tokens=1`, the server
  returned exactly `11` (20 prompt + one completion token). The checked-in
  response JSON proves the content, token counts, 3.876 seconds prompt work
  (5.160 tok/s), and 0.551 seconds for the single completion token (1.817
  tok/s); the checked-in server log records 4.426 seconds total. The client
  HTTP status and client-wall observation were not persisted. This one-token
  length-limited result is intentionally adverse compatibility evidence, not a
  representative speed claim.
- Teardown: PASS. The checked-in server log records graceful shutdown. Port
  closure, no remaining `mlx-serve`/DwarfStar process, and the post-run memory
  reading were contemporaneous PTY observations, not persisted raw evidence.
- Post-run identity: binary and target hash commands were contemporaneous
  separate-command observations, not persisted raw evidence.
- Checked-in raw evidence:
  `receipts/dsv4/2026-08-13-dwarfstar-merge-ready-m5-smoke.log` and
  `receipts/dsv4/2026-08-13-dwarfstar-merge-ready-m5-response.json`.
  Their SHA-256 identities are respectively
  `76c713d6dd47f7ca2da8c33391e0a9b833d1c73f805f3de6b641d25412f4073c`
  and
  `008e86001729c7a95a642aa49be0aa7d757df1a178143a4fc55409e726a6c9c5`.

This historical result closed the then-required `69b3762` target-only SSD
functional gate. It does not establish the `34e30e7` result above, approve
DSpark as a default, establish representative throughput/long-context
stability, or establish parity with the original FP8 model.

## Historical `69b3762` publication topology

The reviewed nested patch is published at immutable commit
`69b376268f52e44cc42077efcecfd7c4990ab6ae` in the public
`PhilipJohnBasile/ds4` fork, and the parent submodule pin targets that reachable
commit. At the frozen check, upstream PR `antirez/ds4#798` was open, non-draft,
and mergeable with this exact head; it preserves the path back to the canonical
repository. A later upstream merge would require a separate canonical-URL and
gitlink bump, not a reinterpretation of this candidate's public-fork source
closure. The parent is rebased onto upstream `1607911`; its current ReleaseFast
build, DS4 ABI guard, and focused DS4 tests pass. The two unrelated full-suite
GPU numerical failures are recorded above. A dirty or locally unreachable
submodule is never accepted as release evidence.

## Standalone DwarfStar evidence

The following historical measurements were taken through the standalone
DwarfStar route. They are not measurements of the public `mlx-serve` wrapper.

| Arm | Request | Result |
| --- | --- | --- |
| Target only | 33 prompt + 256 generated tokens, non-streaming | 25.169 prefill tok/s; **45.159 decode tok/s** |
| DSpark | Same 33 + 256 request | 32.699 prefill tok/s; **63.080 decode tok/s** |
| DSpark | 33 + 128, non-streaming | 30.1 prefill tok/s; **61.2 decode tok/s** |
| DSpark | Same 33 + 128, streaming | 21.4 prefill tok/s; **55.8 decode tok/s** |

The 256-token output was the requested ordered integer sequence and was
identical between target-only and DSpark through the length stop. DSpark's
fixed setup cost made one-token answers slower, so it is a long-generation
optimization rather than an unconditional default win.

## Compact quality result

The existing 50-task objective local suite was run once against each arm:

- Target only: **43/50**
- DSpark: **43/50**, with the same seven failed tasks and same outputs
- Pinned OpenRouter CoreWeave FP8 reference from the prior receipt: 50/50
- Superseded MLX target-only artifact from the prior receipt: 37/50

The seven target-only misses were rerun three times each and reproduced
deterministically. They were `json_square`, `math_mod`, `math_sqrt`,
`reason_tom`, `reason_half`, `reason_square`, and `choice_vowels`.

Conclusion: the refreshed DwarfStar route is the fastest locally verified
DeepSeek V4 Flash path in this workspace and materially improves on the MLX
artifact, but the current 2-bit GGUF is not quality-equivalent to the original
FP8 service.

Raw local log identities (not checked in):

- target-only log SHA-256:
  `d00003f5cb0e63a0e3c966fe37aece834b8351dd030da6d76fb4a102914b00bc`
- DSpark log SHA-256:
  `76cae6a3288fce3efb18edac62a996f48035a510fb17cc1468cd2dc675233e25`

## Exact 0731 mixed-quant follow-up

The Hub contains two same-size mixed 2+4-bit files with different content.
The unsuffixed file was added on 2026-05-18; the file selected by current
DwarfStar was added on 2026-07-31 and ends in `-0731.gguf`. An initial transfer
of the unsuffixed file was stopped after this mismatch was discovered. The
Hugging Face client removed its own incomplete transfer; no completed artifact
was deleted.

The exact current files are downloaded and byte-verified:

- Mixed 0731 target GGUF SHA-256:
  `659e22fbd01c9e13ea37a57c8d9c41e0a8819dffa3473d3c5286ee44b2d3398f`
  (97,591,747,456 bytes)
- DSpark 0731 support GGUF SHA-256:
  `7e319924541db3f7a163ed7e11d7532a70d48228ab59d36cb81e1d4511885360`
  (5,989,114,272 bytes)

The earlier speed result used the same 0731 q2 target but the superseded
2026-07-16 DSpark support file (`8b3adf...`). It therefore remains historical
local evidence, not the final exact-0731 DSpark measurement.

`mlx-serve` previously returned the first DSpark/MTP sidecar encountered in a
directory. With both support versions present that could pair a 0731 target
with the older sidecar. Ordinary legacy MTP lookup now ranks matching 0731
lineage first, requested DSpark versus legacy MTP second, then a lexical
tie-break. Explicit `--dspark` is stricter: it accepts only a DSpark support
file with the exact same 0731 lineage as its target and otherwise returns no
sidecar, which fails closed at engine open. Temporary-directory regressions
cover both matching candidates, legacy-MTP rejection, and mismatched-only
DSpark in both target/support directions. The DS4-focused suite and
ReleaseFast executable build also pass.

An unrelated live MTPLX Qwen server remained healthy and resident, so the
103.58 GB target-plus-sidecar pair was not loaded fully resident and no speed
claim was attempted. The service was preserved. The mixed target was instead
loaded with DwarfStar SSD streaming and an 8 GB expert-cache budget. DwarfStar
reported 9.47 GiB total planned memory: 0.42 GiB KV, 0.06 GiB buffers,
0.99 GiB resident model, 4.62 GiB dynamic expert cache, and 3.38 GiB prefill
expert reserve.

The complete 50-task objective suite was run once through that bounded-memory
target-only profile:

- Mixed 0731 target-only: **46/50**
- Previously passing q2 cases retained: **43/43**
- Recovered q2 misses: `json_square`, `math_sqrt`, and `reason_half`
- Remaining misses: `math_mod`, `reason_tom`, `reason_square`, and
  `choice_vowels`

This is quality evidence only. SSD streaming and the still-live Qwen service
make it unsuitable as a throughput measurement.

The remaining four chat misses were also tested through the explicit
`deepseek-reasoner` profile with thinking enabled and a 512-token budget. All
four passed: `math_mod`, `reason_tom`, `reason_square`, and `choice_vowels`.
This does not make reasoner a universal default. A complete 50-task reasoning
run at a 256-token budget scored 40/50 and took 433.064 seconds; longer JSON and
code cases often exhausted the reasoning budget before emitting a final
answer. Retesting those ten at 512 tokens recovered six, while `json_square`,
`code_cap`, and `code_reverse` still exhausted the budget. `code_max3` produced
a semantically correct implementation that the deliberately narrow objective
grader did not accept. The supported product conclusion is therefore:

- `deepseek-chat` is the best verified general default: **46/50**.
- `deepseek-reasoner` is the stronger explicit mode for math and word
  reasoning, not an automatic benchmark-specific router.

The first `--quality` plus SSD-streaming launch exposed a real mapping defect:
exact quality kernels consumed whole routed tensors while the compact static
decode map contained only selected-expert views. The local DwarfStar patch now
disables that compact map in quality mode and maps one complete layer at a
time. A real 10-prompt-token request then completed correctly with no uncovered
range or `metal decode failed` error. The bounded-memory cost is prohibitive:
prefill took 11.007 seconds and the single generated token took 21.854 seconds
(0.05 tok/s). The same post-patch request on the normal selected-expert path
returned the exact requested answer; prefill took 2.887 seconds and the one
decode token took 0.050 seconds (19.83 tok/s).

Conclusion: retain the mapping fix for correctness, but do not ship `--quality`
as the SSD-streaming default. The mixed chat profile remains the practical M5
route.

The exact 0731 DSpark sidecar was then enabled narrowly under SSD streaming.
The first experimental launch loaded all 81 expected tensors, but its support
view was displaced whenever a target layer was remapped; repeated 1.82–1.87
GiB uncovered-range errors showed that inference was silently falling back to
the target. DwarfStar now keeps the exact sidecar in its own persistent 5.58
GiB Metal view. Its 15.04 GiB line was a component plan, not proof of an
admission decision. This remediation adds a fail-closed Metal startup ledger
before persistent registration; it counts the target map, full support map,
expert cache, prefill reserve, KV, scratch, and safety headroom against host
capacity. It is intentionally not a cross-process safety claim. No new
real-model DSpark measurement was run for this repair.

The result is a measured negative result for this target/sidecar pair:

| Bounded-memory arm | Work | Prefill | Decode | End-to-end |
| --- | --- | ---: | ---: | ---: |
| Exact 0731 DSpark | 44 prompt + 127 generated | 2.796 s | 11.38 tok/s | 13.961 s |
| Target only | identical request | 2.323 s | 16.00 tok/s | 10.261 s |

DSpark was 36.1% slower end-to-end and 28.9% slower in decode. Its session
reported 118 speculative cycles, 19 proposed tokens, zero accepted draft
tokens, 99 no-draft cycles, and 1,858.121 ms of proposal work with no saved
target time. Both arms produced the same 127-token reasoning prefix and
stopped at the same length limit. This is a controlled local A/B under the
still-resident Qwen service, not a representative release benchmark, but it is
sufficient to reject exact DSpark as the default for this artifact. The
persistent-view fix remains useful experimental plumbing; the product route is
target-only SSD streaming.

## Public `mlx-serve` wrapper evidence — target-only only

No public-wrapper DSpark launch or measurement was run. The wrapper evidence
below validates the target-only SSD route and must not be used to claim
DSpark speed, quality, or admission behavior through `mlx-serve`.

The first integrated `mlx-serve --ssd-streaming` smoke caught a separate
wrapper bug: its generic engine defaults still set `warm_weights=true`, so the
embedded engine began faulting all 90.88 GiB of GGUF pages before configuring
the streaming map. The process was interrupted immediately. The wrapper now
enforces `warm_weights=false` whenever SSD streaming is enabled, regardless of
which caller retained the general warmup default. A focused ReleaseFast unit
test covers all requested/streaming combinations.

The rebuilt `mlx-serve` then logged `full model residency and warmup are
skipped`, reached the public OpenAI-compatible endpoint, and returned exactly
`11` for a one-token deterministic smoke. The server reported 10 prompt tokens,
one completion token, 3.620 seconds total, 2.8 prefill tok/s, and 14.4 decode
tok/s. This is a cold functional smoke under competing residency, not a speed
benchmark. The temporary server was shut down cleanly; the unrelated MTPLX
Qwen service and model pull were not stopped or modified.

A later historical wrapper-binary smoke caught one more wrapper-level capacity issue before
inference: leaving DwarfStar's cache fields at zero selected a 62.44 GiB expert
budget from the device's static recommended working-set limit, which cannot
see unrelated resident processes. That server was stopped before a request.
`mlx-serve` now substitutes the measured-safe 8 GiB budget whenever SSD
streaming is requested without an explicit internal cache override. That
then-current historical wrapper binary planned the expected 8.00 GiB expert budget, skipped complete
GGUF residency/warmup, reached the public endpoint, and returned exactly `11`
for 10 prompt + one completion token in 1.665 seconds. This confirms the
bounded default and public serve path; it is not a throughput claim.
