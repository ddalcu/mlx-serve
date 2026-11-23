#!/usr/bin/env python3
"""Dump Hy4-flash parity fixtures for the Zig oracle tests in `src/plugins/hy_v4/`.

USER-RUN (needs torch + transformers >= 5.10 with `hy_v4_flash` support — the modeling PR
(huggingface/transformers#50903) may need `--revision` until it merges). Runs the REFERENCE
PyTorch modeling code over the HF checkpoint (tencent/Hy4-flash) and writes byte-exact golden
tensors the tape tests compare the Zig port against (cos + rms_ratio per plugins/AGENTS.md),
mirroring `tests/dump_dsv41_fixtures.py`.

MEMORY: the trunk is 220B-A12B; the HF bf16 checkpoint is ~440 GB (trunk ~313 GB + Engram tables
~127 GB), so it needs the 512 GiB host with `--device cpu` in bf16 and nothing else resident; the Engram
tables stream from disk in our 4-bit pack, not from the HF set.

Oracle taps (env prefix HYV4_) land with the plans that read them:

  1 EMBED  : fixed token ids -> embedding rows                       (plan 03)
  2 HC     : per-layer iHC streams + post gates, layers 0..1         (plan 03)
  3 MLA    : one DSA layer's q/kv projections + gated o_proj         (plan 03)
  4 IDX    : compressed_indexer pooled groups + top-512 group ids    (plan 04)
  5 ENG    : engram hash rows + gate/signal residual, layer 1        (plan 05)
  6 MOE    : router scores + 8-expert combine, layer 12              (plan 03)
  7 DFLY   : DFly2 draft round + fusion logits                       (plan 06)
  8 VIT    : RADIO-style tower rows for a fixed image                (plan 07)
  9 E2E    : greedy 32 tokens for two prompts                        (plan 08)

STATUS: stub — plan 03 fills in the first taps when its tape test lands.
"""

import sys

sys.exit("tests/dump_hyv4_fixtures.py is a stub until plan 03; see the docstring.")
