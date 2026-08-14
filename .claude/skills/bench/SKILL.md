---
name: bench
description: mlx-serve benchmarking methodology — bench.sh/llmprobe usage, comparison-trap rules (same-methodology cells only, spec-decode variance, thermal lies, engine naming), perf-claim etiquette. Use before running benchmarks or making any performance claim.
---

## Benchmarking

**llmprobe is the measurement layer.** `tests/bench.sh` boots mlx-serve (one model at a time: boot, probe, kill, settle) and llmprobe takes every number via `--bench-only`. We do not hand-roll timing loops — llmprobe discards a warmup per scenario, reports median-of-3 as `median (min-max)`, refuses to fabricate a number when usage is missing, records the machine it ran on, and applies the same protocol to every engine.

```
./tests/bench.sh                                # every model (~did we regress)
./tests/bench.sh --only qwen36-27b              # one row
./tests/bench.sh --url 127.0.0.1:1234 -m <id>   # a server someone else started
./tests/bench.sh --full                         # median of 3 per rung, to 64k
```

**Each cell is mlx-serve at its FASTEST.** `--mtp` is forced wherever the checkpoint ships an MTP head, because it is default-OFF on MoE targets and that is where it pays most (35B-A3B reads 157 without and 191 with). Everything else is already on by default. The mode that actually engaged is read off the server's own `[spec-stats] mode=` lines and named beside the number — a mode that silently stops engaging shows up as a bare cell, which is the regression signal.

**Another engine = another URL.** Start LM Studio / oMLX / MTPLX / llama-server yourself, then `--url host:port -m <id>`. Same script, same probe, nothing about their binaries, ports or version strings lives in the bench.

The only artifacts: the paste-ready rows bench.sh prints at the end, which go into `benchmarks.md` (one column per release in the history table, plus the cross-engine table rewritten when a comparison is run), and the saved llmprobe reports + server logs under `~/claude-tmp/bench-<tag>/`.

### Comparison traps (these cost real days)

- **Only diff same-methodology cells.** Columns through 26.7.12 are the old in-repo harness; 26.8 on is llmprobe — different prompts, different warmup, different rate math. Never diff across that boundary. Same rule inside one column: a forced-spec cell and a shipping-defaults cell are not the same measurement.
- **"Reproducible ≠ not variance"** for spec-decode cells — sample across runs and boot orders before any regression claim. A cell that reads the same twice can still be variance.
- **Attribute before believing.** Check whether the change could physically reach the cell that moved; reachability is faster to check than another bench and is what makes a repeat a confirmation.
- **Never quote a win without naming the engine it is over** — vs LM-GGUF a row reads +33%; vs oMLX the same row is +1.6%.
- **Thermal soak lies harder than drift** — same-session ratios only. llmprobe's own sustained-load check catches drift WITHIN a cell; run the comparison engine right after ours, not hours later, or say so beside the number.
- **An A/B arm is proven by ENGAGEMENT lines in its own log, never by its launch env.** zsh does not word-split `env $VAR`, so a multi-switch arm's first switch swallowed the rest as its value and the "composed" arm silently ran the fast path — reading a 2x win as "neutral" for half a session (live 2026-07-30, story in docs/qwentts-cache.md).
- **A bench's port wait-list must equal its kill-list.** LM Studio's server is a persistent daemon you never kill (its MODEL is freed by `lms unload --all`), so waiting on its port burns the full timeout on every stop — measured 11 of 20 min on one run. This is why bench.sh no longer manages other engines at all.
