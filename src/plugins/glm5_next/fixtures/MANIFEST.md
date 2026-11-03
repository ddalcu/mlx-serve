# glm5_next fixtures

`*.tape`: the legacy forward's values (`forwardGlm5With`, `forwardGlmBatchedDecode`, the `.glm` MTP head) recorded
on a tiny random pack written in memory by `fixture.zig` (`glmFiles` / `glmSushiFiles`, seed 7: 4 layers, KDA at 2
heads of 128, one DSA layer at the release's 64 heads of 256 over a 512 latent, a 4-token indexer pool with an 8-token
budget, 8 experts top 2, 4-bit group 64; the Sushi pack's experts EXL3). Arrays past 1 KiB are kept as a digest and a
64-element sample (`Tape.compact_over`). Replayed by `proof_test.zig` (`steps`, `batched`, `sushi`, `head`, `media`)
and `served_test.zig` (`served`, `served_sushi`, `served_head`); recorded with `MLX_SERVE_PROOF_RECORD=<repo root>`
while the legacy forward existed, so they cannot be regenerated.
Configs claimed here: `src/fixtures/model-configs/glm5_next-*.json` (mlx-community 4-bit, NVIDIA FP4, Sushi 2.4bpw).
Gate model: `ddalcu/GLM-5.3-Flash-MLX-Serve-iQ-MLX-5.1bpw`.
