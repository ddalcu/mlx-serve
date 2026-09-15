# Qwen3.8 MTP verifier port (issue #419)

The existing exact verifier remains the default. Two opt-in, mutually exclusive
sampled-decode modes use the same sampler-filtered target rows and the same
sampled draft distribution as the existing batched verifier. Neither mode is
distribution-exact relative to the target model.

* **Typical** (`--mtp-typical 0.2`): for each drafted token `x`, compute the
  target-row entropy `H(p)` and accept iff `p(x) > min(eps, delta*exp(-H(p)))`,
  with `eps=1`. This is deterministic and consumes no acceptance uniform. On
  the first rejection, sample directly from `p` at that position. The full
  accept bonus samples from the final target row.
* **TokenV3** (`--mtp-tokenv3 0.95`): let
  `Top = {v: p(v) >= max(p)*(1-alpha)}` and
  `eta = sum_{v outside Top} q(v)`. The effective target for the verifier is
  `pi(v) = q(v)*1[v in Top] + p(v)*eta` for **every** vocabulary item. A draft
  in `Top` is kept with no acceptance uniform. Otherwise use the exact
  `min(1,pi(x)/q(x))` coin and sample a correction from
  `normalize(max(pi-q,0))` if it rejects. The full accept bonus still samples
  from `p`. Computing only `p(x)` or using `p` for the deferred coin would be
  wrong for sampled `q`.

The modes are installed at request construction. The batched verifier builds
all acceptance values and all possible correction samples in one lazy graph,
then performs one async evaluation per round. An experimental mode cannot
silently fall back to the exact per-position path. Greedy temperature retains
the existing argmax verifier; neither lossy mode claims engagement there.

Validation gates: pure toy-distribution math tests, GPU graph tests for the
one-hot and sampled-proposal layouts, unchanged exact tests, live sampled MTP
engagement, then matched llmprobe runs for both engines with
`--sampling creative`, thinking explicitly off, the same probe version,
prompt set and run count. The published llmprobe 0.6.7 CLI does not explicitly
disable thinking on its timed or evaluation chat requests, so use an audited
disposable bundle and capture outgoing request bodies. `--reasoning off` alone
only omits `reasoning_effort` on evaluation requests.

On the Apple M5 Max, full llmprobe 0.6.7 runs with creative sampling,
thinking explicitly off, three timed samples, sampled drafts, and MTP depth 3
measured these decode rates (short median / approximately 16K median, tokens/s):

| Verifier | mlx-serve mixed pack, KV8 | MTPLX Optimized-Speed pack, BF16 KV |
| --- | ---: | ---: |
| Exact | 92.5 / 82.4 | 102.0 / 90.9 (PR #475) |
| Typical 0.2 | 106.3 / 86.7 | 115.5 / 99.8 (PR #478) |
| TokenV3 0.95 | 106.5 / 89.9 | 119.3 / 103.1 (PR #485) |

The comparison uses the same probe prompts and timed request fields, but
different packs, KV formats, sampler filter orders, and unpaired random
streams. In one seeded, 100-question llmprobe creative-thinking-off JavaScript
code eval on the same mlx-serve binary, exact passed 76, Typical passed 78, and
TokenV3 passed 77. These are sanity results, with 5–7 token-cap stops per arm;
the 164-task Python HumanEval quality gate remains open. llmprobe's standard
conformance/capability/fidelity scores are largely outside its creative-sampling
benchmark path.

## Paired routed gate/up kernel

`MLX_SERVE_MOE_VERIFY_PAIRED_GU=1` installs an opt-in physical-S=4 routed
gate/up kernel at model load. It is limited to the Qwen3.8 Flash-Next pack with
48 MoE layers, hidden width 2560, expert width 640, 512 experts, top-10 routing,
separate contiguous affine q4/group-64 gate and up banks, and an M5-class GPU
(`verifySharedHardware`). A pack or chip outside that contract declines with a
`[mtp-verify] paired routed gate/up declined:` line and loads on the stock
route. A matching pack is bit-compared per layer against the stock sorted gather
plus fused SwiGLU path before generation; a self-check failure stops the load.

The kernel adapts MTPLX's paired routed producer to this checkpoint's split
banks and group size. In a two-order, six-pair 16K serving A/B on the same
binary, with exact MTP acceptance, sampled depth-3 drafts, KV8, creative
sampling, thinking xhigh, and identical request seeds, mean decode speed rose
from 70.377 to 70.914 tok/s (+0.76%). Total wall time fell from 109.077 to
108.675 seconds (0.37% shorter). All paired completion lengths, answer
digests, and reasoning digests matched. This is a measured 16K-cell result;
long-context and full state/cache parity remain separate gates.

The complete unchanged-control comparison, quality sanity screen, and 1K-1M
timing and memory charts are in the
[benchmark record](benchmarks/qwen38-mtp-port/README.md). The high-context cells are
explicitly statistics-only and should not be read as long-context quality or
recurrent-state parity evidence.
