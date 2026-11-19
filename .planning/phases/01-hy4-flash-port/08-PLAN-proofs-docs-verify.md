# PLAN 08 — Proofs, docs, live verification

Phase: 01-hy4-flash-port | Wave: 4 | Depends on: 03, 04, 05, 06, 07
Requirements: HY4-08, HY4-10
Files: `src/plugins/hy_v4/AGENTS.md`, `docs/gotchas/models.md` (entries), `tests/test_hyv4_*.sh` (final suite wiring), root `AGENTS.md` row (WIP→live), `src/host/chat/format_corpus_test.zig` (template entries), CHANGELOG (release-time)

## Objective

The port is proven, documented, and honest: every subsystem has a green test arm, the pack's real failure modes are captured as gotchas, the docs and registry table reflect what actually ships, and the issue's reproduction case is verified closed.

## Tasks

### Task 1: proof suite + gotchas
- Wire `tests/test_hy4.sh` umbrella (loader/stream/engram parity arms hermetic; DSA/DFly/vision/full-pack parity env-gated to the 512 GB host like other cos-oracles; CI arm on `test-org-copilot/HYV4-0.7B` tiny pack + synthetic fixtures).
- Chat-template corpus entry (pack's jinja) + format corpus; replay a captured tool-traffic session through the model if template/tool claims survive.
- Gotchas entries (one story each, <20 lines): the pack-vs-config layer-index offset; the padded nope halves in q_b/linear_gate/o; embed+lm_head quantized despite `enable_lm_head_fp32`; the two 33 GB `biases` tensors' true role; DFly not in the HF reference (weights-ignored key class); any class-guard corpus entry born during the port.
- **Verify:** full `zig build test -Dslow-tests` green (incl. format corpus + replay); `tests/test_hy4.sh` green; lint 0.
- **Done:** every Plan 01–07 verify-criterion has a named, green test somewhere.

### Task 2: docs + closure verification
- `src/plugins/hy_v4/AGENTS.md` full rules/quirks/packs/oracle per plugin conventions; root arch-table row finalized (model_type `hy_v4_flash` → `hy_v4` plugin); `docs/reference.md` section if the subsystem needs a surface beyond the plugin folder (keep one section); STATE/benchmarks per repo norms — a first-run decode number recorded in `benchmarks.md` at release time only (per /release style).
- Live closure check (the issue's own repro): `mlx-serve list` claims the pack; launch on the 512 GB host; text + image + long-context + spec-ON chat all serve; log greps clean (`[admission]`, `[spec-stats]` engagement).
- **Verify (checkpoint:human-verify):** the user (or nunez) confirms the repro case: "it loads and it's good" — checklist: 5-prompt smoke, 32k context, one image, spec ON vs OFF identical.
- **Done:** issue #1241 commentable as fixed.

## Verification (plan-level)
All green; docs exist; human checkpoint cleared; commit per repo norms (no Co-Authored-By).
