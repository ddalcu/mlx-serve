# Benchmarks — mlx-serve decode by release

**Update rules — read before editing:**
- Results go into the tables ONLY. No text, no commentary, no per-release notes — `docs/gotchas/` carries the stories.
- **One table per machine** (Apple M4 Max 128 GB, Apple M5 Ultra 80c 256 GB, the x86 Linux box named in its heading). Do not update a table from any other machine (e.g. the M4 mini) — numbers across hardware are not comparable and one mixed column poisons the whole history.
- A cell is `./tests/bench.sh` decode tok/s (llmprobe `--bench-only`: warmup discarded, median of 3, its own code-completion prompt), mlx-serve ReleaseFast at its FASTEST config, with the speculative mode that engaged named beside the number. `·` = not measured that release.
- A table keeps its last 6 releases: a new column drops the oldest, and `speedup` is recomputed over what is left.
- Every column is llmprobe (26.8 on). The pre-26.8 columns from the old in-repo harness were dropped in 26.9.2 — they were never comparable cell to cell. `speedup` = first measured column vs the latest.

## Decode tok/s by release — Apple M4 Max 128 GB

| Model | 26.10.1 | 26.10.2 | 26.10.3 | 26.11.2 | 26.11.3 | 26.11.4 | speedup |
|---|---|---|---|---|---|---|---|
| Gemma 4 E4B 4b | 129 pld | 129 | 127 pld | 129 pld | 129 pld | 128 pld | -1% |
| Gemma 4 26B-A4B 4b | 126 pld | · | 128 pld | · | 128 pld | 128 pld | +2% |
| Qwen3.6 35B-A3B 4b (MTP) | 238 mtp | · | 239 mtp | · | 236 mtp | · | -1% |
| Qwen3.8 27B 4b (ddalcu) | 92 dflash | 109 dflash | 106 dflash | · | 103 dflash | · | +12% |
| Qwen3.8 27B iQ-MLX 3.8bpw | · | · | 62 mtp | 60 mtp | · | · | -3% |
| Qwen3.8 Flash-Next mixed 4-8b | 88 mtp | 91 mtp | · | · | · | · | +3% |
| Qwen3.8 Flash-Next iQ-MLX 4.7bpw | · | · | · | 89 mtp | 88 mtp | 86 mtp | -3% |

## Decode tok/s by release — Apple M5 Ultra 80c 256 GB

| Model | 26.10.2 | 26.10.3 | 26.11.1 | 26.11.2 | 26.11.3 | 26.11.4 |
|---|---|---|---|---|---|---|
| Gemma 4 E4B 4b | 196 pld | 199 pld | 200 pld | 199 pld | 197 pld | 197 pld |
| Gemma 4 26B-A4B 4b | 187 pld | 186 pld | 184 pld | 189 pld | 184 pld | 186 pld |
| Qwen3.6 35B-A3B 4b (MTP) | 416 mtp | 424 mtp | 421 mtp | 439 mtp | 439 mtp | 436 mtp |
| Qwen3.8 27B 4b (ddalcu) | 242 dflash | 242 dflash | 243 dflash | 243 dflash | 244 dflash | 250 dflash |
| Qwen3.8 27B iQ-MLX 3.8bpw | 134 mtp | 136 mtp | · | · | · | · |
| Qwen3.8 Flash-Next mixed 4-8b | · | 129 mtp | · | · | · | · |
| Qwen3.8 Flash-Next iQ-MLX 4.7bpw | 169 mtp | 175 mtp | 180 mtp | 168 mtp | 168 mtp | 164 mtp |
| MiMo-V2.6-Flash MXFP4-Q8 | 94 mtp | · | · | · | · | · |
| GLM-5.3-Flash oQ4-MTP | 82 mtp | · | · | · | · | · |

## Decode tok/s by release — x86 Linux, RTX 5060 Ti 16 GB (CUDA 13), i5-12600K, 31 GB RAM

| Model | 26.10.2 | 26.10.3 |
|---|---|---|
| Gemma 4 E4B 4b | 105 pld | 108 pld |
| Qwen3.8 27B iQ-MLX 3.8bpw | 24 pld | 27 mtp |
