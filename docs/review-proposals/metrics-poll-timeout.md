# TensorFold review proposal: metrics-poll-timeout

Review baseline: `3129a772b4a5a46678e44902b82db4bed42fa737`.

This patch was previously applied outside the user's review-only request. It is now preserved separately for explicit review; it is not merged or enabled in the review checkout.

## Triage

- Priority: **P3 — low reliability**.
- Disposition: **Draft; review browser support and timeout policy**.
- Finding: setInterval starts another asynchronous poll every second even if the preceding request has not settled.

## Observed verification

Actual Monitor exercised in Chromium against deliberately stalled HTTP/1.1 metrics responses. Baseline accumulated at least five outstanding requests and stayed connecting; proposal surfaced an error after its deadline and returned to live after the fixture resumed. Rate-math tests previously passed but do not prove this network behavior.

## Limits and risks

AbortSignal.timeout requires a supported browser. The fixed five-second policy is a design choice. The preserved patch comment overstates universal 6–8 connection exhaustion; HTTP/2 and other protocols differ. No production-wide chat freeze is claimed.

No full server/app build or model inference benchmark is claimed.

## Publication boundary

Fork-only migration review, against the unchanged reviewed baseline. Upstream's `CONTRIBUTING.md:63` full-tree build and `zig build test` gate has not been met. Zig is absent from PATH and the MLX submodules are unstaged. This proposal is not upstream-ready; no merge is authorized.
