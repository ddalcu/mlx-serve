# qwen3_next fixtures

The oracle is the legacy layered forward on tiny random packs written by `proof_test.zig`'s `writePack`
(seeds there). Gate model: `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit`.

`*.tape`: what each proof compares, recorded from the legacy forward (`MLX_SERVE_PROOF_RECORD=<repo root>`,
`engines/mlx/plugin/tape.zig`, `qwen3_5/served_test.zig`'s `Oracle`) on applegpu_g17d. Proofs play them back beside
the live legacy shell; once that forward is deleted they are the oracle. Logits, layer captures, greedy streams and
accepted counts (recurrent states stay live-only: the logits that follow carry them).
