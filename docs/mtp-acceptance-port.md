# Qwen3.8 MTP verifier port (issue #419)

The existing exact verifier remains the default. Two opt-in, mutually exclusive
sampled-decode modes use the same sampler-filtered target rows and the same
sampled draft distribution as the existing batched verifier. Neither mode is
distribution-exact relative to the target model.

* **Typical** (`--mtp-typical <delta>`, `--mtp-typical-eps <eps>`): for each
  drafted token `x`, compute the target-row entropy `H(p)` and accept iff
  `p(x) > min(eps, delta*exp(-H(p)))`. The default `eps` is 1, no cap (the
  rule Qwen and MiMo were measured on; `--mtp-typical 0.2` for the Qwen3.8
  matched comparison, and the model setting `typical` is delta 0.2 uncapped).
  The cap is opt-in: `--mtp-typical-eps` sets eps, which is Medusa's
  posterior_threshold (`delta` is its posterior_alpha). Alone it takes
  delta = sqrt(eps), Medusa's coupling; with `--mtp-typical` both apply as
  given. Medusa's standard pair is eps 0.09, delta 0.3. This is deterministic and consumes no acceptance uniform. On
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

## Lossless and lossy acceptance

Every speculative mode is one of two classes, and a lossy one is never on by default.

* **Lossless** (the default everywhere): the output distribution is the serial decoder's. Greedy requests accept a
  draft only where it equals the target's argmax (the same tokens as serial greedy decoding). Sampled requests accept
  draft k with probability min(1, p_k / q_k) and draw the first reject from max(0, p - q): the MTP exact verifier, and
  DSpark's stochastic arm.
* **Lossy** (opt-in, each logs `NOT distribution-exact`): Typical (`--mtp-typical <delta>` / `--mtp-typical-eps <eps>`, or a model's
  `mtp_acceptance` setting) and TokenV3 (`--mtp-tokenv3 <alpha>`, MTP only). They apply to sampled requests only;
  greedy requests keep the argmax verifier.

## DSpark typical acceptance

DeepSeek-V4/V4.1's DSpark stochastic arm reads the same mode. Under typical a sampled round keeps draft k iff
`p(x) > min(eps, delta*exp(-H(p)))`, with no acceptance uniform, and draws the first reject from p
(`mtpBatchedTypicalGraph`, the MTP head's graph; `dspark.verdictOf`). The server logs
`[dspark-accept] typical delta=… eps=… NOT distribution-exact (copied drafts exact)`, and `[spec-stats] dspark typical`
counts the rounds the rule decided.

Drafts copied from the context (DSpark's copy rounds, as a prompt lookup's) keep the exact rule (`dspark.draftRule`).
Such a draft is a point mass the model did not propose. Typical acceptance would keep it whenever the target clears
its floor, so an echo of the context would feed itself and self-echo would become certain. A plugin's native drafter,
whose round does not say which drafts it copied, keeps the exact rule too. TokenV3 is not applied to DSpark.

A smaller delta or eps lowers the floor and accepts more: near 0 it keeps any draft with mass, the lossy end. The cap
eps bounds the floor on confident rows (low entropy), where delta * e^-H is near delta. With eps 1 and a delta large
enough that the floor reaches 1, nothing is kept, so every token is a draw from p (serial sampling's distribution,
with no speedup).
