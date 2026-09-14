# Consolidated Qwen4 prefill candidate

PR [#408](https://github.com/ddalcu/mlx-serve/pull/408) is the single review target for this investigation. [The final report](https://github.com/nikolai-vysotskyi/mlx-serve/blob/ca4e691/research/followups/20260913-final-prefill.md) contains the full measurement table, raw evidence, reviewer answers, related-thread dispositions and rejected hypotheses.

## Measured result

M5 Max128GB, mixed4/8-bit Flash-Next, identical uncached64,947-token HTTP requests: existing QSA+HC+GDN2173.8tok/s, plus bounded PLE-ahead, counted MoE grouping and HC-upmix2347.1tok/s (+7.97% observed). This is one passively cooled pair on087c210, not a sustained serving or llmprobe acceptance result. A later snapshot with optional MoE MPP recorded2299.9tok/s under temporary Full blast fans; that was not a matched fan-speed comparison. Fans were restored to Automatic.

The older full llmprobe ladder on core head d23d9df measured1517→1761 at65.8K and1447→1771 at131.1K, three runs/rung. Its different workload cannot be used as a denominator for2347. No>1.5× or2400 claim is made. [Raw cooled pair](https://github.com/nikolai-vysotskyi/mlx-serve/tree/ca4e691/research/followups/cooled-prefill-20260913), [later snapshot and validation](https://github.com/nikolai-vysotskyi/mlx-serve/tree/ca4e691/research/followups/final-prefill-20260913).

## Run the candidate

Build ReleaseFast with the pinned Zig setup and an isolated cache. The measured local MLX was0.32.3 source1f8e74e3f12f31365464a6867c6579f0e9b29d85; inspect the installed library instead of trusting the stale0.32.2 startup label. Use the mixed pack, not the withdrawn pure4-bit checkpoint. Do not mix this branch with a newer main binary and reuse its measurements.

```sh
MLX_SERVE_PREFILL_CHUNK=8192 \
MLX_SERVE_QSA_PAIR=1 \
MLX_SERVE_HC_PREFILL=1 \
MLX_SERVE_GDN_PREFILL_FUSED=1 \
MLX_SERVE_PLE_PACKED=1 MLX_SERVE_PLE_AHEAD=1 \
MLX_SERVE_MOE_PREFILL_GROUP=1 MLX_SERVE_HC_UPMIX=1 \
MLX_SERVE_MOE_PREFILL_MPP=0 \
zig-out/bin/mlx-serve serve \
  --model ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit \
  --host 127.0.0.1 --port 11234 \
  --prefill-chunk 8192 --ctx-size 131072 --prefix-cache-entries 0 \
  --kv-quant off --no-mtp --no-pld --no-drafter
```

The environment pin above is part of the measured configuration. In this revision, `--prefill-chunk 8192` still passes through `boundedPrefillChunk`, which caps non-sliding hd256 MoE at4096 under the default fused-causal mode. `MLX_SERVE_PREFILL_CHUNK=8192` takes precedence before that cap. Both published benchmark drivers already set it; the earlier manual example omitted it. Check `[prefill-trace] chunk_size` and `chunk_widths`, not just the CLI argument.

MTP changes this comparison too: `mtp_active` currently declines PLE-ahead, even with its flag enabled, and committed-history capture/head forwards run during prefill. The2347 HTTP screen explicitly disabled MTP. An MTP-on result requires its own engagement/trace metadata and cannot be called the identical feature combination.

The new paths have narrow geometry guards and may fall back. Confirm `[qsa-pair]`, `[hc-prefill]`, `[gdn-prefill]`, `[ple-packed]`, `[ple-ahead]`, `[moe-prefill-group]` and `[hc-upmix]` engagement in the relevant request. MPP is an additional experiment, enabled separately by `MLX_SERVE_MOE_PREFILL_MPP=1`; it is not needed for the2347 cell and its incremental whole-model gain remains weak.

For matched HTTP screening, run `research/followups/bench-prefill-combined.py` from repository root. Its `--help` and the final report specify the GPU lock, optional read-only macmon telemetry, cooldown and warmup policy. Hex sequence0 retains core fusions,7 adds PLE/group/upmix,f adds MPP. Leave `QWEN4_PREFILL_ARM_SEQUENCE`, `QWEN4_MOE_LIVE_AB`, `QWEN4_MOE_CAPTURE_PATH`, `QWEN4_HC_UPMIX_VERIFY` and other profiling controls unset for normal serving. Replay/capture adds synchronization and invalidates normal-throughput comparisons.

## Validation and readiness

Consolidation commit b7d9033 has runtime/build inputs byte-identical to the pre-wide32 research snapshot: ReleaseFast passed, full Zig suite2333 passed/154 skipped, actual model HC/MoE replay parity and uncached HTTP recall passed before packaging. [Source and binary provenance](https://github.com/nikolai-vysotskyi/mlx-serve/blob/ca4e691/research/followups/final-prefill-20260913/source-manifest.json). Packaging is not a new full-model measurement. No model/GPU rerun occurred during finalization while the owner used the GPU for another task.

HC/GDN core fusions retain default-on behavior with `=0` kill switches and production compiled-reference tests. Paired QSA changes reduction order and remains opt-in with its float64 bar. New bounded PLE, grouping, HC-upmix and MPP remain opt-in pending general rollout/quality/pressure review; finite fixture parity does not prove universal bit identity or quality.

This is a consolidated **draft**. Outstanding: corrected-head M4 HC/GDN greedy-divergence isolation; QSA broad quality; startup cooperative-layout probe/fallback for paired QSA, HC-upmix and MPP; final-combination llmprobe acceptance; new buffer admission/pressure and research-diagnostic review; final upstream integration validation. NAX availability alone is not layout validation, and unsupported-layout diagnostics in the experimental MPP kernels can produce NaN. Do not enable them indiscriminately. The failed32K prefill changes and whole-table GPU PLE mapping are excluded.

Related threads: #366 research ledger and #368 PLE follow-up are consolidated here; #375 is superseded as a separate proposal, with its pressure acceptance explicitly unproven; #385 was already integrated via#388; #365 was already fixed in26.9.2. Details and exact requested pressure protocol remain in the final report.

## Final upstream integration

The final PR integrates upstream0814cf3ce2 after the measured snapshot. Conflict resolution preserves upstream per-slot PLE speculative capture, batched deferred gather and its pending-state error check. The experimental packed path declines batched slots. Both upstream batched GDN tests and the prefill-width test are retained. This adaptation has separate build/test status in the final PR comment; no new model throughput is claimed for the integrated head.

## September14 upstream refresh

The PR now incorporates upstream1075630. Native/grouped deferred PLE retains its single grouped ID synchronization; packed PLE shares the evaluated-ID history transition. Wide prefill grouping stays outside the joined-verifier path, whose route packing, indexed input, paired down projection and fused reduction are retained. ReleaseFast7/7 and2512 Zig tests passed/169 skipped onM5. No full-model speed measurement was performed for this integration. Native-MTP PLE-ahead remains a separately documented, unintegrated proposal.
