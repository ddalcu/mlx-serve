# Benchmarks — mlx-serve decode by release

**Update rules — read before editing:**
- Results go into the tables ONLY. No text, no commentary, no per-release notes — the CSVs in `docs/perf-csvs/` and `docs/gotchas/` carry the stories.
- **Apple M4 Max 128 GB ONLY.** Do not update these tables from any other machine (e.g. the M4 mini) — numbers across hardware are not comparable and one mixed column poisons the whole history.
- Methodology: `tests/bench.sh` code cell — decode tok/s, temp 0, max_tokens 128, ctx 4096, thinking off, mlx-serve ReleaseFast. Per-cell medians where repeated. `·` = not measured / not in the matrix that release. The `speedup` column is first measured column vs latest; recompute it when adding a release column.

## Best config (MTP / drafter / PLD — speculative decoding on where it wins)

| Model | 26.5.5 | 26.5.6 | 26.6.10 | 26.7.6 | 26.7.7 | 26.7.9 | 26.7.10 | 26.7.12 | speedup |
|---|---|---|---|---|---|---|---|---|---|
| Gemma 4 E2B 4b | 206 drafter | 202 drafter | 231 drafter | 239 drafter | · | · | · | · | +16% |
| Gemma 4 E4B 8b | 136 drafter | 131 drafter | 154 drafter | 194 drafter | 189 drafter | 174 drafter | 167 drafter | 177 drafter | +30% |
| Gemma 4 26B-A4B 4b | · | · | · | 124 pld | 126 pld | 127 pld | 126 pld | 125 pld | +1% |
| Gemma 4 31B 4b | 19 drafter | 20 | 24 | 31 drafter | 31 drafter | 32 drafter | 32 drafter | 33 drafter | +74% |
| Qwen3.6 27B 4b | 24 | 24 | 29 | 58 mtp | 74 mtp | 76 mtp | 76 mtp | 76 mtp | +217% |
| Qwen3.6 27B MTPLX-opt | · | · | · | · | 80 mtp | 78 mtp | 80 mtp | 79 mtp | -1% |
| Qwen3.6 35B-A3B 4b | 104 | 106 | 128 | 175 mtp | 210 mtp | 215 mtp | 227 mtp | 237 mtp | +128% |

## Raw decode (no speculation)

| Model | 26.5.5 | 26.5.6 | 26.6.10 | 26.7.6 | 26.7.7 | 26.7.9 | 26.7.10 | 26.7.12 | speedup |
|---|---|---|---|---|---|---|---|---|---|
| Gemma 4 E2B 4b | 168 | 170 | 185 | 191 | · | · | · | · | +14% |
| Gemma 4 E4B 8b | 102 | 101 | 115 | 118 | 117 | 116 | 113 | 115 | +13% |
| Gemma 4 26B-A4B 4b | · | · | · | 116 | 114 | 116 | 115 | 114 | -2% |
| Gemma 4 31B 4b | 17 | 20 | 24 | 25 | 25 | 25 | 25 | 25 | +47% |
| Qwen3.6 27B 4b | 24 | 24 | 29 | 29 | 29 | 29 | 28 | 28 | +17% |
| Qwen3.6 35B-A3B 4b | 104 | 106 | 128 | 131 | 129 | 130 | 129 | 155 | +49% |
| Laguna XS 2.1 NVFP4 | · | · | · | · | · | · | 25 | 121 | +384% |
