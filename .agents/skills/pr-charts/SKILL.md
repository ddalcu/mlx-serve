---
name: pr-charts
description: Turn llmprobe reports into the charts a perf PR embeds (one line per arm across context size, every point labelled, machine and model in the caption) and host them so the PR body renders them. Use when a PR makes a speed claim, or when asked for a chart of a bench run.
---

## PR charts

The chart goes above the table in "How I verified it". A reviewer sees in two seconds whether a win holds at every context size or only at one.

1. **Measure each arm with `tests/bench.sh`, alternating.** Run the ladder once per arm, then again in reverse order (A B B A), all in one session. Tag each run by arm and round (`--tag main-1`, `--tag pr-1`, `--tag pr-2`, `--tag main-2`) so the reports land in `~/claude-tmp/bench-<tag>/`. Serve arms from one binary with an env switch where you can, and use `--url` for a server you started yourself. `--full` gives a median of 3 per rung, out to 64k.
2. **Chart them.**
   ```
   uv run tests/bench_chart.py --out charts/ main='~/claude-tmp/bench-main-*/*.json' pr='~/claude-tmp/bench-pr-*/*.json'
   ```
   The script writes `decode.png`, `prefill.png` and `ttft.png`. Each point is the median across that arm's reports, and the caption names the machine, model, engine version and report count. It reads `bench.contextScaling` and times nothing itself.
3. **Look at the PNGs before posting.** A flat line at zero, a missing arm, or two arms that match to the digit means a bad run, not a result. Check each arm's server log for the engagement line (`.agents/skills/bench/SKILL.md`).
4. **Host them.** `gh` can't attach images. Commit the PNGs to a branch of your public fork, for example `pr-charts` with a directory per PR, and link the raw URL. A private repo's raw URL renders for nobody else.
   ```markdown
   | Prompt processing | Generation |
   |---|---|
   | ![prefill](https://raw.githubusercontent.com/<you>/mlx-serve/pr-charts/<pr>/prefill.png) | ![decode](https://raw.githubusercontent.com/<you>/mlx-serve/pr-charts/<pr>/decode.png) |
   ```
5. **One line of method under the charts:** chip, macOS, model, flags, runs per arm, arm order.

Chart what the PR claims. A multi-stream change charts aggregate tok/s against stream count, and a latency change charts TTFT; if the ladder doesn't show the claim, build the one chart that does from measured numbers. Keep the points that got slower.
