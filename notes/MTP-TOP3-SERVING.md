# Top-three batch-MTP serving measurement

Measured the tree now committed as local main `597d539` on M5 Max with binary SHA-256
`358f1e3f1e2d3330c824e15eda44b6e07b233198df17375b6cb9df58917d1714`.
The complete committed verifier stack was enabled, including dense NV6 tiles,
joined hyper-connection reads and fused stable route packing. All switches remain
opt-in. Nothing was pushed or changed in production source during measurement.

## Result

Four concurrent clients, two 300-token requests per client, code-heavy prompt,
greedy generation, affine-8 KV and warm-prefix reuse. Two model boots reverse both
context and request-arm order. Aggregate decode tok/s is measured from first output
to final output across all eight timed requests in a cell.

| Context | Plain tok/s | Adaptive batch MTP tok/s | Median paired change | Pair changes | Plain / MTP p95 gap |
|---:|---:|---:|---:|---|---:|
| 4k | 99.3 | 113.3 | +14.0% | +3.5%, +24.6% | 41.3 / 63.9 ms |
| 16k | 109.2 | 136.2 | +24.6% | +28.3%, +20.9% | 37.4 / 74.3 ms |
| 64k | 101.0 | 104.4 | +3.3% | +7.2%, −0.5% | 40.7 / 62.4 ms |

Raw paired aggregate rates:

- 4k: 98.5→101.9 and 100.1→124.7 tok/s.
- 16k: 112.3→144.1 and 106.1→128.3 tok/s.
- 64k: 102.8→110.2 and 99.1→98.6 tok/s.

Wall tok/s, which includes each cell's first-token span, is available in
`summary.json`. Median MTP wall rates across the two boots are 107.3, 128.0 and
99.2 tok/s at 4k, 16k and 64k respectively.

## Planner interpretation

The first tested context after each cold model load remained calibration-only:
all eight 4k requests in boot 1 and all eight 64k requests in boot 2 ran exactly
12 speculative probe rounds and did not sustain MTP. When those contexts ran last,
all eight 4k requests sustained speculation; four of eight 64k requests sustained.
Both boots sustained all eight 16k requests.

This explains the wide 4k pair spread and the near-neutral 64k median. It also
shows that verifier speed is no longer the only limiter: cold planner evidence
coverage and transfer across context buckets still determine whether the faster
path is used. The 16k result is the strongest and most consistent measurement.

## Method and checks

The prompt is a long mlx-serve source prefix followed by a request for a complete
Python streaming JSONL reader and tests. Prompt sizes are 4,088, 16,376 and 65,528
tokens. Minimum timed cached fraction was 99.27%. The two boots contain 12 cells,
96 timed requests, six per-arm warmups plus one initial calibration request: 109
successful requests, zero cancellations.

Every required path was verified from its own server log: grouped planner/verifier,
batched MTP head, shared/GDN/attention/vocabulary projections, paired expert down,
expert reduction, indexed inputs, coarse head, shared-expert gate tail, joined HC,
dense tiles and route packing. No decode failures or position gaps occurred.

All 12 power windows were Nominal. Median GPU frequency was 1603–1607 MHz in every
cell, with paired arms within 1–3 MHz. The first boot ran contexts 4k→16k→64k and
arms plain→MTP; the second reversed both. Fans were set to max before each cell,
with 10-second idle at 4k/16k and 30 seconds at 64k, then returned to AUTO. Each
boot held the exclusive Fleeter GPU lease; all owned processes stopped afterward.

Artifacts: `batch-mtp-chain-results/mtp-chain/top3-serving-20260912/`.
The measurement preceded the history-only squash from `57c0bce`; both commits have
the same tree. `summary.json` and `power-summary.json` contain the derived results;
`summarize.py` and `power-summary.py` reproduce them from the raw CSV, request JSONL,
server logs and power samples. The frozen executable and prompt source are stored
beside them.
