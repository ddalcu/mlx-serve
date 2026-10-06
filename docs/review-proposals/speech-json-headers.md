# TensorFold review proposal: speech-json-headers

Review baseline: `3129a772b4a5a46678e44902b82db4bed42fa737`.

This patch was previously applied outside the user's review-only request. It is now preserved separately for explicit review; it is not merged or enabled in the review checkout.

## Triage

- Priority: **P2 — normal**.
- Disposition: **Draft; viable narrow correction**.
- Finding: The voice reply path passes a header object to authHeaders(key), producing Bearer [object Object] and omitting JSON content type.

## Observed verification

Actual console exercised in Chromium with a controlled recognizer and HTTP fixture. Before: Authorization Bearer [object Object], Content-Type text/plain;charset=UTF-8, fixture status 401. After: Authorization Bearer smoke-key, Content-Type application/json, fixture status 200. Actual model speech synthesis and production authentication middleware were not exercised.

## Limits and risks

Uses the existing jsonHeaders helper; changes only the voice reply request. No retry/error-reporting scope added.

No full server/app build or model inference benchmark is claimed.

## Publication boundary

Fork-only migration review, against the unchanged reviewed baseline. Upstream's `CONTRIBUTING.md:63` full-tree build and `zig build test` gate has not been met. Zig is absent from PATH and the MLX submodules are unstaged. This proposal is not upstream-ready; no merge is authorized.
