# qwen3_5 fixtures

The oracle is the legacy layered forward on tiny random packs written by `sdk/bases/proof/qwen35_pack.zig`'s
`writePack` (seeds in `proof_test.zig`). The real config `src/fixtures/model-configs/minicpmv4_7.json` is claimed
here. Gate models: `mlx-community/Qwen3.5-4B-4bit`, `mlx-community/Qwen3.6-35B-A3B-4bit`.

`*.tape`: what each proof compares, recorded from the legacy forward (`MLX_SERVE_PROOF_RECORD=<repo root>`,
`engines/mlx/plugin/tape.zig`, `qwen3_5/served_test.zig`'s `Oracle`) on applegpu_g17d. Proofs play them back beside
the live legacy shell; once that forward is deleted they are the oracle. Logits, layer captures, greedy streams and
accepted counts (recurrent states stay live-only: the logits that follow carry them).
