---
name: release
description: mlx-serve pre-release validation checklist, CalVer versioning, release steps, and CHANGELOG style. Use when preparing or cutting a release, running pre-release validation, or writing CHANGELOG entries.
---

## Pre-release validation — ALWAYS run this, same process every time

Timings measured 2026-07-16 on the M4 Max 128 GB, AFTER the `stop_all_engines` port-wait fix (before it, everything below was ~2.2× slower — see the gotcha in Benchmarking).

| # | Step | Command | Time |
|---|---|---|---|
| 1 | Hermetic suite | `zig build test` (**must** be 6/6 steps, 0 fail) + `cd app && swift test` | ~1 min |
| 2 | ReleaseFast binary | `zig build -Doptimize=ReleaseFast` → `du -h zig-out/bin/mlx-serve` ≈ **7 MB** (Debug ≈ 2× = fake regression) | ~10 s |
| 3 | **Perf gate** (did WE regress?) | `./tests/bench.sh --family all` (mlx-serve only, llmprobe) → diff vs `docs/perf-csvs/probe-<prev>.csv` → append this release's column to `benchmarks.md` | ~15 min |
| 4 | Tool-call correctness | `zig build test -Dtest-filter="format corpus"` + `-Dtest-filter="tool traffic"`; live: `./tests/test_tool_matrix_small.sh` | ~3 min |
| 5 | API conformance | `npx llmprobe@latest http://127.0.0.1:<port>/v1 --quick` → expect **100%** engine conformance | ~10 s/model |
| 6 | Regression scripts | `integration_test.sh`, `test_anthropic_api.sh`, `test_ollama_api.sh`, `test_stream_keepalive.sh`, `test_disconnect_cancel.sh`, `test_pld_equivalence.sh`, `test_mtp_equivalence.sh` | ~15 min |
| 7 | Soak (bigger releases) | `SOAK_DURATION_HOURS=1 ./tests/test_soak_24h.sh` — RSS drift < 10% | 1 h |
| 8 | **Marketing chart** (only when an ENGINE version changed, or before a public claim) | `./tests/bench.sh --family all --lmstudio --omlx --mtplx --full` | ~90 min |
| 9 | Bundle | `SKIP_NOTARIZE=1 bash app/build.sh` (both binaries move together) | ~2 min |

**Rules:**
- **Steps 3 and 8 are different questions.** 3 = "did our code regress" — mlx-serve only, the ONLY one needed every release. 8 = the public chart; LM Studio/oMLX/MTPLX numbers cannot move when only OUR code changes, so re-run 8 only when an engine version bumps.
- **Diff step 3 against `probe-<version>.csv` only.** The `all-*.csv` / `mtp-ladder-*.csv` families are the pre-2026-08 hand-rolled bench and a DIFFERENT methodology — frozen history, never a diff target. See /bench.
- **`--only <substr>`** runs a single model row for tight dev loops.
- **Depth**: default `--bench-only` is one run per ladder rung to 16k. `--full` takes median-of-3 per rung and climbs to 32k/64k — that's the release artifact depth (step 8). For a regression CLAIM on a spec-decode cell, sample across runs and boot orders regardless of depth: "reproducible ≠ not variance".
- **Never quote a win without naming the engine it is over** — vs LM-GGUF the 26B-A4B row reads +33%; vs oMLX it is +1.6%.
- **`benchmarks.md` gets one new COLUMN per release, from step 3's headline rows** (Laguna's raw number comes from its live A/B harness, not the bench matrix). Obey the file's own header rules: results into the tables only, no text; **Apple M4 Max 128 GB only** — skip the update entirely when releasing from any other machine (the M4 mini), a mixed column poisons the history.

## Release benchmark artifacts

Each release ships **ONE CSV + TWO PNGs**, all from a single bench run on the FINAL release tree (a chart generated mid-cycle is stale the moment another perf round lands). llmprobe measures both the headline numbers and the ladder in one pass, so there is no second protocol:

```
./tests/bench.sh --family all --lmstudio --omlx --mtplx --full --tag <ver>
```

writes `docs/perf-csvs/probe-<ver>.csv`, `docs/perf-pngs/perf-vs-lmstudio-omlx-all-<ver>.png` and `docs/perf-pngs/perf-mtp-ladder-<ver>.png`. The CSV's `#` header line carries date + depth and becomes both charts' subtitle, so the methodology travels with the picture.

Rules:
- **The ladder chart is per model** (`--ladder-model`, default `qwen36-27b`) — a multi-model CSV without one is refused rather than blended.
- **Delete the cycle's untracked dev CSVs/PNGs before cutting** (fwd-probes, timestamped `probe-2026*.csv` dev runs, scratch files) — only the three artifacts above land; committed history stays.
- **Never re-render an old CSV with new styling or lane specs** — a chart and its CSV travel together; if the numbers are being kept, the chart is too. The `all-*` / `mtp-ladder-*` CSVs predate llmprobe and their charts are final.
- Both charts must be visually consistent with the previous release's (same lane order, names, colors, panels) so releases compare at a glance.

## Versioning & Releases

CalVer `YY.M.N` (e.g., `v26.4.25` = 2026, April, 25th release). `N` auto-increments from the last GitHub release for that `YY.M` prefix; `build.sh` computes via `gh release list`.

**Version sources**: `app/Info.plist` (`CFBundleVersion`/`CFBundleShortVersionString`), Zig `-Dversion` build option (`build_options.version`), git tag (`gh release create v{version}`). CI derives ONE version and stamps all three — the bundle plist is stamped from it, never shipped as committed (v26.8.1's DMG reported 26.7.12 and nagged forever; `docs/gotchas/app.md`).

**`YY.M` is TZ-pinned** (`America/New_York`, in release.yml + app/build.sh): runners are UTC, so a `workflow_dispatch` cut after ~20:00 local otherwise rolls into next month. **Prefer a tag push over a dispatch** when the version is already decided — the tag-push path takes `version=${GITHUB_REF_NAME#v}` and never consults the clock.

**Release**:
1. Update `CHANGELOG.md` with NEXT version (check `gh release list --limit 1` first — never reuse an existing tag)
2. Dont commit or push

### CHANGELOG style

**One entry per shipped release. No new entries for unshipped work — fold it into the next pending entry.** Always run `gh release list --limit 1` first; if the topmost CHANGELOG entry is newer than the latest GitHub release, that entry is unshipped and any new bullets get merged into it. Never bump version numbers ahead of an actual release.

Tone: high-level executive bullets, marketing-style. The audience is users/integrators, not contributors reading the diff.

- Lead each bullet with **what changed for the user** (capability, speed, model support), not the implementation.
- Quantify where impressive — concrete tok/s percentages, model names, the workload it applies to.
- Avoid: file paths, function names, internal symbol renames, line-count diffs, "we discovered that…", PR/issue numbers.
- 4–7 bullets per release. If you need more, the release is too big and should ship sooner.

Template:

```markdown

## vYY.M.N — Two-to-five-word headline

- **<User-visible thing>**: one or two sentences on the impact. Numbers if you have them.
- **<New model / API / behavior>**: what unlocks, when it kicks in, what stays the same.
- **<Speed or reliability win>**: workload + measured gain.
- **<Removed / deprecated thing, if any>**: why, and what users should do instead.

---
```

When in doubt, look at the existing entries (v26.5.4 and earlier) — keep the same density and tone.
