# Benchmark Log

Current benchmark state, one page: the final numbers, how to reproduce them, and the
rules that keep them honest. The full historical narrative (2026-04 → 2026-07 session
entries, bisects, retractions) lives in this file's **git history**.

Hardware: Apple M4 Max, 128 GB, AC power, idle machine. Engines: LM Studio 0.4.15+2,
MTPLX 2.0.2, mlx-lm 0.31.3. Identical weights within every row. Last refreshed
**2026-07-12** (v26.7.6-pending).

## Native-MTP context ladder — Qwen3.6-27B, identical checkpoint on all 3 engines

Model `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` (4-bit trunk + calibrated MTP
adapter; LM Studio has no MTP support and decodes plain AR). Coding-agent prompts,
temp 0.6, max_tokens 128, fresh loads, **cold prompts**, best-of-N per cell.
CSV: `docs/perf-csvs/mtp-ladder-26.7.6.csv` · chart: `docs/perf-mtp-ladder-26.7.6.png`.

| prefill tok/s | 0.5k | 1k | 2k | 4k | 8k | 16k | 32k | 64k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **mlx-serve** | 243.8 | 248.1 | 239.3 | 242.4 | 235.0 | 224.4 | 209.3 | 186.0 |
| LM Studio (cold) | 229.2 | 243.5 | 245.2 | 239.7 | 239.3 | 226.5 | 214.7 | 189.5 |
| MTPLX | 237.9 | 244.7 | 235.1 | 236.8 | 231.3 | 216.4 | 205.1 | 180.3 |

| decode tok/s | 0.5k | 1k | 2k | 4k | 8k | 16k | 32k | 64k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **mlx-serve** (MTP d3) | 55.5 | 50.8 | 50.6 | 50.2 | 47.5 | 45.9 | 41.3 | 33.5 |
| LM Studio (AR) | 29.4 | 29.1 | 29.4 | 28.0 | 27.6 | 26.3 | 25.0 | 21.7 |
| MTPLX (MTP) | 46.6 | 40.4 | 43.4 | 38.5 | 40.3 | 37.6 | 37.2 | 29.9 |

Verdict: decode +11–30% vs MTPLX and +54–89% vs LM Studio at every rung; prefill ahead
of MTPLX at all 8 (+1.4–3.7%) and within ±2.5% of LM Studio everywhere.

## vs LM Studio — family matrix (ctx 4096, temp 0, best cell vs best cell)

CSVs: `docs/perf-csvs/{gemma,qwen36}-26.7.6.csv` · chart:
`docs/perf-vs-lmstudio-qwen36-26.7.6.png`. Geomean over 18 cells: **+47.7%**
(**+38.0%** with native-MTP cells excluded).

| Model | Echo | Code | Free-form |
|---|---:|---:|---:|
| Gemma 4 E2B | +111% | +25% | +2% |
| Gemma 4 E4B | +113% | +64% | 0% |
| Gemma 4 31B | +98% | +24% | −1% |
| Gemma 4 26B-A4B-MoE (QAT) | +62% | +5% | 0% |
| Qwen 3.6 27B | +112% | +98% | +32% |
| Qwen 3.6 35B-A3B-MoE | +132% | +70% | +32% |

## Long-context prefill (hd-256)

| Case | Result |
|---|---|
| Gemma 26B-A4B, 99K prompt (band kernel v2) | 317 (composed) → 652.6 (v1) → **715.4 tok/s** (2.4×), MLX peak 39.2 GB unchanged, identical output |
| Gemma E4B, 851-token bench cell | 2045.7 → **2135.7 tok/s** (kernel v2), decode flat |
| Qwen 27B, 8K prompt (2048-chunk cap) | 225.0 → **235.8 tok/s**, peak phys 28.9 → **19.8 GB** |
| Qwen 27B, 32K prompt (2048-chunk cap) | 205.4 → **209.3 tok/s** |

## Restart TTFT — SSD prefix-cache tier (~11.2K-token prompt, temp 0)

CSVs: `docs/perf-csvs/ttft-ssd-kv-cache-{baseline-20260706,final-20260707}.csv`.

| Model | Restart TTFT before → after |
|---|---|
| gemma4-e4b-4bit | 5.9 s → 0.66 s (9×) |
| gemma3-12b-4bit | 40.6 s → 2.7 s (15×) |
| gemma4-26B-A4B-moe-4bit | 10.5 s → 2.3 s (4.6×) |
| qwen36-27b (hybrid SSM) | 47.8 s → 0.55 s (87×) |

## Methodology rules (hard-won — violate these and the numbers lie)

- **Cold prompts only** for prefill: engines prompt-cache across requests; nested or
  repeated prompts inflate client-side pp. `bench.sh` salts every run's prompt at
  position 0 for exactly this reason, and its prompts (<2048 tokens) sit below LM
  Studio's cache-chunk granularity — the family matrix was never contaminated.
- **Same-boot A/Bs only** for kernel/dispatch decisions: warm-GPU rungs read 2–4% below
  cold single-cell boots (thermal), and an isolated-kernel µbench win can still be a
  live loss. Treat sub-2% single-run deltas as noise.
- **Always `zig build -Doptimize=ReleaseFast`** — a Debug binary is 2–4× slower and
  reads as a fake regression.
- One 27B in memory at a time (`pkill -f mlx-serve` before any Python mlx run;
  `lms unload --all` after LM Studio measurements).

## How to reproduce

- Family matrix: `./tests/bench.sh --family gemma|qwen36 [--lmstudio --omlx]`
- MTP ladder: manual harness (`~/mlx-bench-assets/harness.py` + nested prompts;
  regeneration recipe in the memory notes) — server flags
  `--no-pld --mtp-depth 3 --prefix-cache-entries 0 --ctx-size 140000`.
- Chart re-render: `python3 tests/plot_mtp_ladder.py docs/perf-csvs/mtp-ladder-26.7.6.csv out.png`
  and `python3 tests/plot_vs_lmstudio_omlx.py docs/perf-csvs/<family>-26.7.6.csv out.png --family <family>`.
