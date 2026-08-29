# Needs additional testing

What we could not measure ourselves. A paste of the numbers in an issue or on
PR #252 is all we need. Build: `zig build -Doptimize=ReleaseFast`, serve with
`--port 11250`, run on AC power.

## Neural Engine prefill (`--ane-prefill`)
- **M3 Ultra, M2 Ultra, M2/M3 Max, M1 Max/Ultra**: one 16k-token prompt with and without `--ane-prefill`, paste the two `[prefill: X tok/s]` lines from the log. Measured so far: M4 Max +19%, M4 +32%, M1 Pro +35%.
- **M3 Ultra**: dual ANE is now the default at share 0.50 (measured +7%); one run with `MLX_SERVE_ANE_DUAL=0` confirms the default is the right way round on your box.
- **M5 Pro/Max/Ultra**: `MLX_SERVE_ANE_FORCE=1` to get past the M5 refusal; we expect a loss, say if not.

## Self-tuning speculation width
- Any Mac we have not measured (M2 family, M3 non-Ultra, M4 Pro, M5 family): an echo prompt ("repeat this text three times") three times in a row, default flags, then the same with `MLX_SERVE_MTP_COST_TABLE=0`. Paste tok/s and the `[spec-stats]` lines. It should be equal or faster; the first request may be slower while it learns.

## MTP on M3 Ultra: bimodal speed
- One tester saw identical echo requests land at either ~65 or ~150 tok/s (2.1 vs 6.7 tokens per step) on Qwen3.8 27B, and decode drop 99 -> 88 tok/s between runs. Re-run on this release: three echo requests in a row, paste each `[spec-stats]` line (`depth=`, `avg_per_round=`, `table=`) and the `[mtp] adaptive depth cap` line. Also one run with `--ane-prefill` off to see whether decode moves with the ANE's wired memory.

## DFlash / DSpark block chooser (opt-in, `MLX_SERVE_DFLASH_CHOOSER=1`)
- Any LFM2.5 or Muse-Glimmer with its `drafter/` folder: with and without the env, echo prompt and a normal prose prompt. Known rough edge: LFM2.5-8B-A1B (MoE) — report it either way.

## Models
- Muse-Glimmer 30B + DFlash 2 on M3 Ultra (block 8 row): tok/s with and without `--draft-block-size 5`.
- Ling 3.0 flash quants still do not load (layout change pending).
