# TensorFold review proposal: hf-startup-timeout

Review baseline: `3129a772b4a5a46678e44902b82db4bed42fa737`.

This patch was previously applied outside the user's review-only request. It is now preserved separately for explicit review; it is not merged or enabled in the review checkout.

## Triage

- Priority: **P3 — low website availability**.
- Disposition: **Draft; review timeout policy**.
- Finding: A stale/missing HF cache causes boot to await two requests without a deadline before dynamic initialization and fallback.

## Observed verification

Actual tier-list module exercised in Chromium with HF fetch URLs redirected to a local hanging endpoint, preserving fetch options and AbortSignal. At 6.5 seconds baseline had zero dynamic table rows; proposal rendered six seed-backed rows. Visual smoke fixture did not serve the full website asset set. Fresh-cache, successful live HF, and production Firebase paths were not exercised in this migration.

## Limits and risks

Shared AbortSignal.timeout(5000) bounds discovery and body reads. Modern browser support and the five-second policy need review. Static HTML was present even before the correction; this is not a wholly blank-page claim.

No full server/app build or model inference benchmark is claimed.
