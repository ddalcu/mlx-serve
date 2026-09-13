# Long-context group MTP: serving results and review validation

The earlier 64k collapse was primarily a loss of speculation: 68.69% of output
came from ordinary ticks, versus 11.42% at 16k. The logged depth-two candidate
had adequate timing samples and fit the latency budget, but narrowly failed a
profit test that compared noisy speculative upper costs with optimistic ordinary
lower costs. Different restored head histories also confounded the two boots.
See the historical [serving measurement](MTP-TOP3-SERVING.md).

Throughput now uses confidence bounds on the learned mean cost; individual-round
variance and observed gaps still constrain latency. Runtime shape prices distinguish
cache formats, widths, sampling and head history, while acceptance stays private
to each request. Bounded recovery can refresh narrow-depth evidence after 48
ordinary tokens: at most eight recovery rounds, with at least 96 tokens remaining.
The three-sample requirement, 5% profit margin and 100 ms admission budget remain.

## Four concurrent streams: three paired boots

Both arms enable native MTP. OFF runs the existing main/#391 MTP scheduling policy;
ON additionally enables the new group planner. At four qwen4 streams, the legacy
policy normally demotes crowded requests to ordinary batched ticks. These numbers
compare the two serving policies, not two continuously speculating verifiers.

| Context | Legacy MTP median tok/s | Group MTP median tok/s | Median paired gain | Paired gains | Median p95 output gap, OFF / ON |
|---|---:|---:|---:|---|---:|
| 4k | 102.2 | 133.0 | +30.1% | +28.1%, +30.1%, +30.4% | 38.6 / 74.6 ms |
| 16k | 110.5 | 139.6 | +26.8% | +23.9%, +26.8%, +26.8% | 35.8 / 72.8 ms |
| 64k | 105.5 | 125.0 | +18.9% | +18.5%, +18.9%, +26.3% | 38.9 / 78.6 ms |

All 24 group-MTP requests per context sustained speculation beyond their probes.
At 64k, ordinary output fell to 5.2–16.9% across boots. The collapse is substantially
reduced, but cold calibration still makes the four-stream 64k result less consistent
than 16k. Wider output bursts are a throughput/latency tradeoff; the 100 ms setting
is a round-admission budget, not an end-to-end gap cap.

## Two streams against actively speculating legacy MTP

A separate paired boot verifies the comparison where legacy MTP remains active.
All four ON requests per context sustained speculation. This is one boot, not a
three-boot confidence estimate.

| Context | Legacy MTP tok/s | Group MTP tok/s | Gain | p95 gap, OFF / ON |
|---|---:|---:|---:|---:|
| 4k | 90.6 | 114.8 | +26.7% | 80.7 / 45.4 ms |
| 16k | 95.7 | 121.7 | +27.2% | 77.1 / 43.4 ms |
| 64k | 90.7 | 114.9 | +26.7% | 77.0 / 44.4 ms |

## Method and scope

M5 Max; Qwen3.8-Flash-Next mixed-4/8-bit; affine-8 target and head KV; full head
history; 4 GB hot-prefix budget; greedy code generation; two 300-token requests
per client. Token-calibrated warm prompts contain 4,088 / 16,376 / 65,528 tokens.
Each pair shares a model boot and request texts. Boots 1 and 3 run 64k→16k→4k,
ON then OFF; boot 2 reverses both orders. Question nonces are 0, 1 and 2.
The two-stream boot uses nonce 3, 64k→16k→4k, OFF then ON.

Every cell uses maximum fans, Nominal pressure, and a subsequent idle period of
10 seconds at 4k/16k or 30 seconds at 64k, followed by another Nominal check.
Fans return to AUTO after each cell. All timing windows remained Nominal;
four-stream paired GPU medians were 1,604–1,606 MHz. Two-stream paired medians
were 1,602–1,617 MHz, with the ON arm slightly lower. No clock correction is applied.

The batch driver measures aggregate completion tokens over the interval from first
output to last completion, including calibration and recovery. llmprobe 0.6.7 keeps
benchmark timing serial, so this concurrency study uses the established batch
harness. The public driver supports a frozen source, saved calibration, nonce,
and optional external control hooks; it owns no server or machine-specific controls.

## Feature boundary and validation

Group planning defaults on; `MLX_SERVE_MTP_GROUP_PLANNER=0` disables the feature.
`enable_batch_mtp:false` retains legacy MTP per request. Existing v3 round-cost
files are unchanged; the group has one runtime-only shape table. Rollback keeps
KV offsets and SSM state without retaining KV backing-buffer references.
Pending legacy drafts are consumed before grouping, and a surviving single row
continues without applying a multi-row padding guard.

ReleaseFast validation on the final merged tree: 2,503 passed, 173 gated skips, zero failures; Swift suite
green. The real-model gate passed 54 exact draft/logit/accepted/state comparisons.
Live global-OFF and request-OFF paths each matched seven current-main streams at
N=1/2/4 byte-for-byte, including finish reasons and completion counts. EOS,
cancellation, survivor speculation, acceptance-failure restoration, cold-head
resume and re-entry after 48 ordinary ticks passed.

Numbers are M5 Max only. On M4, M5-specific projection/reduction kernels decline;
the group scheduler, head batching and routed-expert grouping retain their generic
MLX/Metal paths, with solo adapters for unsupported shapes. No M4 speedup is claimed.

Measured executable SHA-256:
`195c55568ef1280bfd4ce6e22a2968316d84c22786fbd3743b67f9d5fc93c60e`.
Source is pinned in `validated-source.json` at local commit `a9a4629`.
The later main merge `0814cf3` changes tool-argument coercion, outside these requests.

Artifacts: `.zig-cache/batch-mtp-chain-results/mtp-chain/long-context-tuning-20260913/`.
Use `final-confirm-r1/r2/r3`, `final-two-streams`, `final-summary.json` and
`two-stream-summary.json` for these results. Each boot saves its driver, source,
calibration, manifest, request records, CSV, server log and power samples.
Earlier experiments, including the discarded narrow-calibration trial, are excluded.
