---
name: bench
description: mlx-serve benchmarking methodology — bench.sh/llmprobe usage, comparison-trap rules (same-methodology CSVs only, spec-decode variance, thermal lies, engine naming), perf-claim etiquette. Use before running benchmarks or making any performance claim.
---

## Benchmarking

**llmprobe is the measurement layer.** `tests/bench.sh` drives engines (boot, warm, kill, settle) and llmprobe takes every number: `--bench-only` per engine per model, saved as JSON, converted by `tests/bench_csv.py` into ONE CSV that both charts render from. We do not hand-roll timing loops any more — llmprobe discards a warmup per scenario, reports median-of-3 as `median (min-max)`, refuses to fabricate a number when usage is missing, records the machine, and applies the identical protocol to every engine.

```
./tests/bench.sh                                   # mlx-serve only, all models (~did we regress)
./tests/bench.sh --only qwen36-27b                 # one model row
./tests/bench.sh --lmstudio --omlx --mtplx --full  # the release / marketing run
```

Artifacts, all from ONE run: `docs/perf-csvs/probe-<tag>.csv`, `docs/perf-pngs/perf-vs-lmstudio-omlx-all-<tag>.png` (headline bars), `docs/perf-pngs/perf-mtp-ladder-<tag>.png` (context ladder). The ladder is llmprobe's `contextScaling` block out of the same run — there is no second protocol to keep in sync.

**One boot = one cell, on shipping defaults.** llmprobe measures the server that is running, so there is no spec sweep and no "best config" collapse. mlx-serve picks its own speculative mode per checkpoint; the bar is what a user gets out of the box, and llmprobe's speculative probe reports what that turned out to be (ratio + tokens/step land in the CSV as `spec_ratio` / `tok_per_step`). To A/B a flag, boot mlx-serve yourself and compare cells — a kill-switch A/B beats a cross-version absolute diff, always.

### Comparison traps (these cost real days)

- **Diff only against same-methodology CSVs.** `probe-*.csv` (llmprobe) and `all-*.csv` / `mtp-ladder-*.csv` (the pre-2026-08 hand-rolled bench.sh) are DIFFERENT methodologies — different prompts, different warmup, different rate math. Never diff across the two families; the old CSVs are frozen history, kept for their charts.
- **"Reproducible ≠ not variance"** for spec-decode cells — sample across runs and boot orders before any regression claim. A cell that reads the same twice can still be variance.
- **Attribute before believing.** Check whether the change could physically reach the cell that moved; reachability is faster to check than another bench and is what makes a repeat a confirmation.
- **Never quote a win without naming the engine it is over** — vs LM-GGUF a row reads +33%; vs oMLX the same row is +1.6%.
- **Thermal soak lies harder than drift** — same-session ratios only. llmprobe's own sustained-load check catches drift WITHIN a cell; bench.sh runs mlx-serve first on every row so drift that builds across a row lands on the comparison engines, not on us.
- **An A/B arm is proven by ENGAGEMENT lines in its own log, never by its launch env.** zsh does not word-split `env $VAR`, so a multi-switch arm's first switch swallowed the rest as its value and the "composed" arm silently ran the fast path — reading a 2x win as "neutral" for half a session (live 2026-07-30, story in docs/qwentts-cache.md).
- **A bench's port wait-list must equal its kill-list.** `stop_all_engines` waits only on ports it kills. LM Studio's server is a persistent daemon we deliberately never kill (its MODEL is freed by `lms unload --all`), so LMS_PORT in the wait list burned the full 30 s timeout on every stop — measured 11 of 20 min on a `--family all --lmstudio` run, one row 233.5 s → 55.5 s when dropped. Symptom: wall clock dominated by fixed ~40 s stalls between cells regardless of model size, and fast when comparison engines are off.

The record IS the artifacts: CSVs → `docs/perf-csvs/`, charts → `docs/perf-pngs/`. `BenchmarkLog.md` is retired as a hand-maintained narrative — don't add entries.
