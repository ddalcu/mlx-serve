# Benchmarks — mlx-serve decode by release

**Update rules — read before editing:**
- Results go into the tables ONLY. No text, no commentary, no per-release notes — `docs/gotchas/` carries the stories.
- **Apple M4 Max 128 GB ONLY.** Do not update these tables from any other machine (e.g. the M4 mini) — numbers across hardware are not comparable and one mixed column poisons the whole history.
- A cell is `./tests/bench.sh` decode tok/s (llmprobe `--bench-only`: warmup discarded, median of 3, its own code-completion prompt), mlx-serve ReleaseFast at its FASTEST config, with the speculative mode that engaged named beside the number. `·` = not measured that release.
- Columns through 26.7.12 come from the old in-repo harness (temp 0, max_tokens 128, ctx 4096, thinking off, per-cell medians); 26.8 on comes from llmprobe. **The two are not comparable cell to cell** — a `speedup` spanning the switch measures the harness as much as the engine.

## Decode tok/s by release

| Model | 26.5.5 | 26.5.6 | 26.6.10 | 26.7.6 | 26.7.7 | 26.7.9 | 26.7.10 | 26.7.12 | 26.8.6 | 26.8.11 | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma 4 E2B 4b | 206 drafter | 202 drafter | 231 drafter | 239 drafter | · | · | · | · | · | · | +16% |
| Gemma 4 E4B 8b | 136 drafter | 131 drafter | 154 drafter | 194 drafter | 189 drafter | 174 drafter | 167 drafter | 177 drafter | · | · | +30% |
| Gemma 4 E4B 4b | · | · | · | · | · | · | · | · | 115 | 115 | · |
| Gemma 4 26B-A4B 4b | · | · | · | 124 pld | 126 pld | 127 pld | 126 pld | 125 pld | 116 | 119 | -6% |
| Gemma 4 31B 4b | 19 drafter | 20 | 24 | 31 drafter | 31 drafter | 32 drafter | 32 drafter | 33 drafter | 25 | 25 | +32% |
| Qwen3.6 27B 4b | 24 | 24 | 29 | 58 mtp | 74 mtp | 76 mtp | 76 mtp | 76 mtp | 69 mtp | · | +188% |
| Qwen3.6 27B MTPLX-opt | · | · | · | · | 80 mtp | 78 mtp | 80 mtp | 79 mtp | 73 mtp | · | -9% |
| Qwen3.6 35B-A3B 4b | 104 | 106 | 128 | 175 mtp | 210 mtp | 215 mtp | 227 mtp | 237 mtp | 191 mtp | · | +84% |
| Laguna XS 2.1 NVFP4 | · | · | · | · | · | · | 25 | 121 | · | · | +384% |
| Qwen3.8 27B 4b (ddalcu MTP) | · | · | · | · | · | · | · | · | · | 69 mtp | · |
| Qwen3.8 Flash-Next 4b (MTP) | · | · | · | · | · | · | · | · | · | 78 mtp | · |

(Note, testing harness changed in last release, so numbers differ)

## vs other engines

Latest release only — overwritten each run, decode tok/s, identical weights per row.

| Model | mlx-serve 26.8.3 | LM Studio 0.4.19+2 | oMLX 0.5.2 | MTPLX 2.5.3 |
|---|---|---|---|---|
| Gemma 4 E4B 4b | 118 | 119 | 114 | 115 | · |
| Gemma 4 26B-A4B 4b | 118 | 112 | 112 | 119 | · |
| Gemma 4 31B 4b | 25 | 26 | 25 | 25 | · |
| Qwen3.6 27B 4b (oQ4e+MTP) | 71 mtp | · | 58 mtp | · | · |
| Qwen3.6 27B MTPLX-opt | 73 mtp | 30 | 30 | · | 66 mtp |
| Qwen3.6 35B-A3B 4b | 157 | · | 138 | · | · |

2026-08-07 run. mlx-serve here is on shipping defaults, not the forced-MTP config of the table above; LM Studio ran in a later, cooler session. Cells are `·` where the engine has no copy of that checkpoint or cannot load it.
