# PLAN 05 — Engram n-gram injection (layers 1/11/21/31)

Phase: 01-hy4-flash-port | Wave: 2 | Depends on: 01, 02, 03
Requirements: HY4-05
Files: `src/plugins/hy_v4/blocks/engram.zig`, gather kernel + contract, fixture dumps, `tests/test_hyv4_engram.sh` (add)

## Objective

The 64B Engram: hash bigram/trigram ids of the recent token window, gather 16 head embeddings from the streamed tables, normalize, attend against the hidden state, inject per-stream-scaled output — at the exact position the reference specifies, on layers 1/11/21/31, at parity.

## Reuse bill (copy `//! copied_from:`)

- `qwen4_exp/qwen4_exp.zig` `NgramHash{init, multipliers, rowIds}` — reusable as-is (also used by `deepseek_v41/engram.zig`).
- `qwen4_exp/ple_table.zig` `EmbeddedTable`/`openEmbedded`/`preadPart` — the mmap/pread streaming-table reader, reusable as-is (this is the Plan 02 stream-bank arm for engram tables).
- `qwen4_exp/ple_gpu.zig` (`Table`/`wrap`/`embed` + `ple_gather*.metal`) copy-in as the GPU gather arm (re-target its quant arm to affine g32); `ple.zig` slot-history/spec bookkeeping pattern copy-in; load wiring from `qwen4_exp/trunk_load.zig` `NgramTable.open`.
- Precedent for cross-arch reuse of this exact mechanism: `deepseek_v41/engram.zig` + `decode/engram_decode.zig` (LRU) + `prompt/engram_direct.zig`.
- New: the attention-style head combination (16 norms + key/value projs + unified gamma/per-hc scales) per SEMANTICS §5 — the dsv41 engram's "gated add" is the shape precedent, not the math.

## Tasks

### Task 1: hash + lookup + injection math (tests first)
- Token→slot: `lookup_table[token_id]` (I32[120818], pad `engram_pad_id=120002` maps to 0 = no lookup); bigram slot = base2 + t1·120832 + t0 (validate `% 10,312,000 == slot−2` for fixture vocab ranges — if non-injective in-range, this is the bug class; unit-test it), trigram slot = base3 + ((t2·120832 + t1)·120832 + t0) mod-space check; row = order·(8·10,312,000) + head·10,312,000 + slot.
- Per head: gather 96-d row (dequant g32), `lookup_per_head_norms.h` → `key_projs.n` → `norm_key/norm_query` RMS → 16 heads attend `query = hidden` (post-what-norm per §5 citation) with softmax over 16 head-scores → concat 768 → `value_proj` (3072).
- Injection: `unified_pre_gamma` on the stream pre-state, `unified_per_hc_scales[k]` per iHC stream (k=4 → streams), add at the position/point (pre vs post attn/mlp) pinned by §5 line cites.
- Lookup order + rolling n-gram state across decode steps + prefill chunking; prefix-cache: n-gram state keyed to token suffix like KV.
- **Verify:** fixture test — for 8 fixture token windows, gathered rows + final engram output cos ≥ 0.999 vs HF dump; hash function unit test vs config head-vocab sizes; contract for the gather-dequant kernel.
- **Done:** lookup parity on fixtures; state across steps correct (a decode-step table vs step-by-step HF dump matches).

### Task 2: stream-bank integration + perf guard
- Gather pages through the Plan 02 stream bank: pin hot pages policy, measure gather bytes/token, reject if a request's peak (billed via `GenRequest.peak_bytes`) exceeds admission; batch all 4 layers' gathers per forward to one bank pass.
- Perf guard: engram on/off decode tok/s delta recorded on the 512 GB host (no acceptance bar yet — record + document; target beats llama.cpp/hf python).
- **Verify:** `tests/test_hyv4_engram.sh`: greedy parity with engram ON at full pack (cos/token match), RAM-stress arm with tight budget stays within admission, unload frees pages; green.
- **Done:** engram active in forward at parity, memory-behaved.

## Verification (plan-level)
Engram parity test in CI (fixture arm) + full-pack script; hash non-injectivity guard; lint 0.
