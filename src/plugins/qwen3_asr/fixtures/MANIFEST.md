# qwen3_asr fixtures

The oracle is the legacy shell on tiny random packs written by `plugins/qwen3/proof_test.zig`'s `writePack`
under a `thinker_config` (seeds in `proof_test.zig`). Gate models: `mlx-community/Qwen3-ASR-0.6B-8bit`,
`mlx-community/Qwen3-ForcedAligner-0.6B-8bit`.

`*.tape`: what each proof compares, recorded from the legacy forward (`MLX_SERVE_PROOF_RECORD=<repo root>`,
`engines/mlx/plugin/tape.zig`, `qwen3_5/served_test.zig`'s `Oracle`) on applegpu_g17d. Proofs play them back beside
the live legacy shell; once that forward is deleted they are the oracle. Logits, layer captures, greedy streams and
accepted counts (recurrent states stay live-only: the logits that follow carry them).
