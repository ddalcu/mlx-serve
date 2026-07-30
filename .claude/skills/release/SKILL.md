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
| 3 | **Perf gate** (did WE regress?) | `./tests/bench.sh --family all` (mlx-serve only) → diff vs `docs/perf-csvs/all-<prev>.csv` → append this release's column to `benchmarks.md` | **~4 min** |
| 4 | Tool-call correctness | `zig build test -Dtest-filter="format corpus"` + `-Dtest-filter="tool traffic"`; live: `./tests/test_tool_matrix_small.sh` | ~3 min |
| 5 | API conformance | `npx llmprobe@latest http://127.0.0.1:<port>/v1 --quick` → expect **100%** engine conformance | ~10 s/model |
| 6 | Regression scripts | `integration_test.sh`, `test_anthropic_api.sh`, `test_ollama_api.sh`, `test_stream_keepalive.sh`, `test_disconnect_cancel.sh`, `test_pld_equivalence.sh`, `test_mtp_equivalence.sh` | ~15 min |
| 7 | Soak (bigger releases) | `SOAK_DURATION_HOURS=1 ./tests/test_soak_24h.sh` — RSS drift < 10% | 1 h |
| 8 | **Marketing chart** (only when an ENGINE version changed, or before a public claim) | `./tests/bench.sh --family all --lmstudio --omlx --mtplx` | **~12.5 min** |
| 9 | Bundle | `SKIP_NOTARIZE=1 bash app/build.sh` (both binaries move together) | ~2 min |

**Rules:**
- **Steps 3 and 8 are different questions.** 3 = "did our code regress" — mlx-serve only, the ONLY one needed every release. 8 = the public chart; LM Studio/oMLX/MTPLX numbers cannot move when only OUR code changes, so re-run 8 only when an engine version bumps. 22 of 56 cells measure other engines.
- **Diff step 3 against `all-<version>.csv`, never `{gemma,qwen36}-26.7.6.csv`** (different methodology AND pre-`verifyQmm` — see BenchmarkLog rules).
- **`--only <substr>`** runs a single model row (~30 s) for tight dev loops.
- **`--runs N`**: default 2 = run 1 dropped as warmup ⇒ **one measured sample per cell**. For any regression CLAIM use `--runs 3`+ and sample across runs — see the "reproducible ≠ not variance" rule in BenchmarkLog.
- **Never quote a win without naming the engine it is over** — vs LM-GGUF the 26B-A4B row reads +33%; vs oMLX it is +1.6%.
- **`benchmarks.md` gets one new COLUMN per release, from step 3's cells** (best-config table on top, raw decode below; Laguna's raw number comes from its live A/B harness, not the bench matrix). Obey the file's own header rules: results into the tables only, no text; **Apple M4 Max 128 GB only** — skip the update entirely when releasing from any other machine (the M4 mini), a mixed column poisons the history.

## Release benchmark artifacts

Each release ships exactly **TWO CSVs + TWO PNGs**, all measured on the FINAL release tree (a chart generated mid-cycle is stale the moment another perf round lands):

1. `docs/perf-csvs/all-<ver>.csv` + `docs/perf-pngs/perf-vs-lmstudio-omlx-all-<ver>.png` — one bench.sh run serves both (the mlx-serve cells are the next release's step-3 gate-diff target, the engine cells are the chart):
   ```
   ./tests/bench.sh --family all --lmstudio --omlx --mtplx \
     --out docs/perf-pngs/perf-vs-lmstudio-omlx-all-<ver>.png \
     --keep-csv docs/perf-csvs/all-<ver>.csv
   ```
2. `docs/perf-csvs/mtp-ladder-<ver>.csv` + `docs/perf-pngs/perf-mtp-ladder-<ver>.png` — **WARM protocol** via `tests/mtp_ladder_pair.sh` (one boot per engine, discarded warmup, COLD prompts, prefix caches off, ours first): one default run (→ `oursjundot` + `omlx` lanes) + one `OURS_ONLY=1 MODEL=<27B-MTPLX-Optimized-Speed path>` run (→ `oursmtplx` lane). **MTPLX lanes are CARRIED from the previous release's ladder CSV unless MTPLX shipped a new version** (their numbers cannot move when our release changes); the subtitle must disclose the carry. Pinned plot invocation (26.7.12 wording — bump versions/subtitle per release):
   ```
   python3 tests/plot_mtp_ladder.py docs/perf-csvs/mtp-ladder-<ver>.csv \
     docs/perf-pngs/perf-mtp-ladder-<ver>.png \
     --engines "mtplx:MTPLX 2.3.0 · 27B-mtplx-speed:#a78bfa,omlx:oMLX 0.5.2 Lightning · 27B-oQ4e:#38bdf8,oursjundot:mlx-serve · 27B-oQ4e:#f59e0b,oursmtplx:mlx-serve · 27B-mtplx-speed:#ea580c" \
     --delta oursjundot:omlx \
     --title "Native MTP context ladder — MLX-serve vs oMLX (27B-oQ4e) & MTPLX (27B-mtplx-speed), 0.5K–64K" \
     --subtitle "Qwen3.6-27B · WARM: one boot/engine, discarded warmup, COLD prompts, prefix-cache off, ours-first, idle M4 Max 128GB · MLX-serve <ver> · MTPLX 2.3.0 lanes carried from the <prev> run · temp 0.6, thinking off"
   ```

Rules:
- **Delete the cycle's untracked dev CSVs/PNGs before cutting** (fwd-probes, timestamped bench-* runs, medians scratch files) — only the four artifacts above land; committed history stays.
- **Never re-render an old CSV with new styling or lane specs** — a chart and its CSV travel together; if the numbers are being kept, the chart is too.
- Both charts must be visually consistent with the previous release's (same lane order, names pattern, colors, panels) so releases compare at a glance.

## Versioning & Releases

CalVer `YY.M.N` (e.g., `v26.4.25` = 2026, April, 25th release). `N` auto-increments from the last GitHub release for that `YY.M` prefix; `build.sh` computes via `gh release list`.

**Version sources**: `app/Info.plist` (`CFBundleVersion`/`CFBundleShortVersionString`), Zig `-Dversion` build option (`build_options.version`), git tag (`gh release create v{version}`).

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
