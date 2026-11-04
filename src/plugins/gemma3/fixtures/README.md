# gemma3 fixtures

- Conformance: `plugin.zig conformanceFixture`, a four-layer f32 pack (sliding/global pattern, softcap), built in
  memory.
- Proof oracle: tapes recorded from the legacy engine (`*.tape`, `engines/mlx/plugin/tape.zig`); `proof_test.zig`
  replays them on tiny random packs written by `sdk/bases/proof/fixture.zig`. Re-record with `frozen = false` and
  `MLX_SERVE_PROOF_RECORD=<repo root>` while the legacy forward exists.
- Parse fixtures: `src/fixtures/model-configs/gemma3-text.json` (harrier 270m, transformers 5 nested rope).
- Real-model oracle (EmbeddingGemma, legacy path): `tests/dump_embeddinggemma_fixtures.py`.
