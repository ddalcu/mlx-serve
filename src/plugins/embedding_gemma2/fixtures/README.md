# embedding_gemma2 fixtures

- Proof oracle: tapes recorded from the legacy engine (`*.tape`, `engines/mlx/plugin/tape.zig`); `proof_test.zig`
  replays them on tiny random packs written by `sdk/bases/proof/fixture.zig`. Re-record with `frozen = false` and
  `MLX_SERVE_PROOF_RECORD=<repo root>` while the legacy forward exists.
- Reference oracle: the transformers fixtures `generate.zig` reads (`tests/dump_embeddinggemma2_fixtures.py`).
