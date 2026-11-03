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
prompt set and run count. llmprobe 0.6.8+ turns thinking off on every request
under `--reasoning off` (falling back to `enable_thinking: false` on mlx-serve)
and says so in the banner; the 0.6.7 runs recorded here used a patched bundle
for the same effect.

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

## Greedy tail (`--mtp-greedy-tail`)

Opt-in, off by default; a model's `mtp_greedy_tail` in model-settings.json
outranks the flag. A sampled request draws depth 0 from the draft sampler and
takes the head's argmax at every later depth, and the verify reads a one-hot q
for those depths, so `exact` stays distribution-exact and `typical` judges the
argmax against its entropy floor. It pays with `--mtp-typical` only: under exact
acceptance an argmax draft is accepted with its target probability. Apple M5 Max,
Vontra Flash-Next 4-bit pack, T=1.0, code and prose at 1K/4K/16K, two seeds,
A B B A on one binary: typical 0.2 decoded 86.7 tok/s, typical 0.2 plus the tail
99.4 tok/s (2.96 to 3.36 tokens per round). Greedy output is byte-identical with
the tail on.
