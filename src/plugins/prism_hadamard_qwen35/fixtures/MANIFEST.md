# prism_hadamard_qwen35 fixtures

The oracle is the legacy layered forward on a tiny Hadamard pack written by `proof_test.zig`'s `writePack` (2-bit
group 128, biases at -scales, block-1024 sign vectors, f32 unpacked tensors; seeds there). Gate model:
`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`.

`*.tape`: what each proof compares, recorded from the legacy forward (`MLX_SERVE_PROOF_RECORD=<repo root>`,
`engines/mlx/plugin/tape.zig`, `qwen3_5/served_test.zig`'s `Oracle`) on applegpu_g17d. Proofs play them back beside
the live legacy shell; once that forward is deleted they are the oracle.
