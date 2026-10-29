# gemma4 fixtures

- Conformance: `plugin.zig conformanceFixture`, a four-layer f32 pack (sliding, a wider global with K=V, a KV-shared
  global; per-layer embeddings, layer scalars, softcap), built in memory.
- Proof oracle: tapes recorded from the legacy engine (`*.tape`, `engines/mlx/plugin/tape.zig`); `proof_test.zig`
  replays them on tiny random packs written by `sdk/bases/proof/fixture.zig`. Re-record with `frozen = false` and
  `MLX_SERVE_PROOF_RECORD=<repo root>` while the legacy forward exists.
- Parse fixtures: `src/fixtures/model-configs/gemma4-text.json`, `gemma4-unified-text.json` (draft-model configs:
  claimed by type, declined `unsupported`).
- Kernel contract: `contract.zig` (random, non-finite and stress inputs generated per gate).
