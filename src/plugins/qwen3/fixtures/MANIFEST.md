# qwen3 fixtures

The oracle is the legacy forward on tiny random packs the tests write (`sdk/bases/proof/fixture.zig`'s
`Pack`, seeds in `proof_test.zig`). Gate models: `mlx-community/Qwen3-4B-4bit`, `mlx-community/Qwen3-30B-A3B-4bit`.

`*.tape`: what each proof compares, recorded from the legacy forward (`MLX_SERVE_PROOF_RECORD=<repo root>`,
`engines/mlx/plugin/tape.zig`, `qwen3_5/served_test.zig`'s `Oracle`) on applegpu_g17d. Proofs play them back beside
the live legacy shell; once that forward is deleted they are the oracle. Logits, layer captures, greedy streams and
accepted counts (recurrent states stay live-only: the logits that follow carry them); `both-3e01`/`both-3e02` are
Qwen3-Embedding's pooled, normalized rows (f32) for one sequence and two padded batches.
