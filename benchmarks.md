# Benchmarks — mlx-serve decode by release

**Update rules — read before editing:**
- Results go into the tables ONLY. No text, no commentary, no per-release notes — `docs/gotchas/` carries the stories.
- **Apple M4 Max 128 GB ONLY.** Do not update these tables from any other machine (e.g. the M4 mini) — numbers across hardware are not comparable and one mixed column poisons the whole history.
- A cell is `./tests/bench.sh` decode tok/s (llmprobe `--bench-only`: warmup discarded, median of 3, its own code-completion prompt), mlx-serve ReleaseFast at its FASTEST config, with the speculative mode that engaged named beside the number. `·` = not measured that release.
- Every column is llmprobe (26.8 on). The pre-26.8 columns from the old in-repo harness were dropped in 26.9.2 — they were never comparable cell to cell. `speedup` = first measured column vs the latest.

## Decode tok/s by release

| Model | 26.8.6 | 26.8.11 | 26.9.1 | 26.9.2 | speedup |
|---|---|---|---|---|---|
| Gemma 4 E4B 4b | 115 | 117 | 114 | 116 | +1% |
| Gemma 4 26B-A4B 4b | 116 | 120 | 120 | 120 | +3% |
| Qwen3.6 35B-A3B 4b (MTP) | 191 mtp | · | · | 259 mtp | +36% |
| Qwen3.8 27B 4b (ddalcu MTP) | · | 70 mtp | 71 mtp | 68 mtp | -3% |
| Qwen3.8 Flash-Next 4b (MTP) | · | 85 mtp | 83 mtp | 93 mtp | +9% |
| Qwen3.8 27B oQ4e (Jundot MTP) | · | · | 67 mtp | 61 mtp | -9% |
