# embedding_gemma fixtures

- Conformance: `plugin.zig conformanceFixture`, a two-layer f32 pack with a two-layer head, built in memory.
- Proof oracle: tapes recorded from the legacy engine (`*.tape`, `engines/mlx/plugin/tape.zig`); `proof_test.zig`
  replays them on tiny random packs written by `sdk/bases/proof/fixture.zig`. Re-record with `frozen = false` and
  `MLX_SERVE_PROOF_RECORD=<repo root>` while the legacy forward exists.
- Integration: `tests/test_embeddings.sh`.
