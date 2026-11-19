# PLAN 01 — Claim + config parse

Phase: 01-hy4-flash-port | Wave: 1 | Depends on: —
Requirements: HY4-01, HY4-09
Files: `src/plugins/hy_v4/config.zig`, `src/plugins/hy_v4/plugin.zig`, `src/fixtures/model-configs/hy-v4-flash.json`, `src/plugins/hy_v4/AGENTS.md` (stub), root `AGENTS.md` (arch table row)

## Objective

`hy_v4_flash` claims `model_type: hy_v4_flash` (both root and `text_config`), parses the pack config with checked readers, and refuses by name everything this port does not yet implement. The pack `TensorFold/Hy4-flash-MLX-4bit-MTP` moves from `unsupported` to "claimed, loads far enough to fail on weights".

## Tasks

### Task 1: config.zig (MLX-free claim + parse + bill)
- Copy `src/plugins/_template/` to `src/plugins/hy_v4/`.
- Parse `text_config` sub-object as the trunk source; root carries vision + quant map.
- Hard-required invariants (refuse with a named error if violated): `gated_mla=true`, `use_mla=true`, `qk_norm=true`, `learnable_sink=true`, `enable_ihc=true`, `num_nextn_predict_layers=0`, `swiglu_limit` present, `swa_pattern`/`layer_types` consistent with the indexer weights (accept the HF-offset: the pack carries indexer weights exactly on the 12 `deepseek_sparse_attention` layers), `engram_layer_ids` (0-based HF) ↔ pack layers (`-1` offset) reconciled at load, per Plan 02.
- Refuse by name: `compressed_indexer_reuse_fine_indexer=true`, `full_nope=true`, `gate_lora_op != "none"`, `o_lora_op != "none"`, `num_nextn_predict_layers > 0`, `engram_max_ngram_size != 3`, `dfly_num_hidden_layers != 5`, `dfly_mtp_fusion != "slot0"`, `hc_mult != 4`, `mtp_fusion` gate mismatch.
- Memory bill: per-tensor bytes from the quant map (`quantization.quantizations` per-tensor overrides: engram tables g32, default g64 4-bit affine); engram tables (4 × ~33 GB packed) counted as a stream bank (see Plan 02), not resident.
- Unit tests bottom of file: fixture parses, each refusal fires, bill matches a hand-computed tensor census.
- **Verify:** `zig build check`; `zig build test -Dtest-filter="hy_v4"`; `mlx-serve list` shows the pack as claimed (not `unsupported`).
- **Done:** claim + refusal tests green; `list` no longer says unsupported.
- **HY4-09 (sampling defaults):** read the pack's `generation_config.json` (temp 0.9, top_p 1.0, eos set) into the `generation_defaults` path; decide `return_bos_score`/`return_max_logit` — implement cheaply if they ride the sampler, else refuse-with-record (one gotcha line); unit test the precedence chain (body > model defaults > …) on this config.
- **Evidence hook:** the reader found the pack's per-tensor quant map may be ABSENT (no `quantization_config` key) — while `embed_tokens` is also quantized (contradicts HF `convert→dequantize`). Task 2 must test both the map-absent path (fall back to a name-regex default g64/g32-engram) and a map-present fixture, and flag the finding for Plan 02 + a gotcha.

### Task 2: fixture config + stub table + docs row
- `src/fixtures/model-configs/hy-v4-flash.json`: minimal real-shaped config (from the pack, all 42 layer_types, swa_pattern, quant map).
- `plugin.zig`: table with Caps honest at this stage (no batching, no spec, vision off until Plan 07); forward returns an explicit "not yet" error after the loader lands in Plan 02.
- Stub `src/plugins/hy_v4/AGENTS.md`; architecture-table row in root `AGENTS.md` marked WIP (flip when Plan 08 verifies).
- **Verify:** `zig build unit-closure` passes for the plugin group (imports stay in sdk/abi/mlx/kernels); fixture loads through the registry test path.
- **Done:** plugin builds as its own unit; registry scans and claims it.

## Verification (plan-level)
`zig build check` + `zig build test` green; `mlx-serve list` shows claimed; all Task-1 refusals have failing-then-green tests (TDD order).
