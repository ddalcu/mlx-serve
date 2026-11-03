# qwen4_exp fixtures

- Plugin proofs: `fixture.zig` writes tiny random packs (bf16 and 4-bit, QSA budget 8, PLE on a GDN layer,
  raw bf16 `ngram_table.bin`); the oracle is `*.tape` here, recorded from the legacy forward on an M5 Ultra before
  it was deleted (`proof_test.zig`, `served_test.zig`, `oracle_test.zig`); the tapes cannot be re-recorded.
- Parse: `src/fixtures/model-configs/qwen4_exp-iq-mlx-4.7bpw.json` (ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw)
  and `qwen4_exp-nvidia-fp4.json` (NVIDIA's ModelOpt FP4), against the engine's `parseConfig` (`model_config.zig`).
- HF reference (env-gated, `forward.zig`): `tests/dump_qwen4_exp_fixtures.py build|dump` writes a tiny HF model,
  its converted pack and `fixture.safetensors` under `~/claude-tmp/qwen4-tiny/`; run with `QWEN4_TEST_MODEL=<pack>`
  plus `QWEN4_FIXTURE`, `QWEN4_FIXTURE_BF16` (streams at cos 0.999) or `QWEN4_FIXTURE_VISION`;
  `-Dtest-filter="qwen4 fixture"`. YaRN: `tests/dump_qwen4_yarn_fixtures.py`.
- Real pack: `QWEN4_TEST_MODEL=<Flash Next pack>` arms the long-context, MTP and PLE tests; `tests/qwen4_engagement.sh
  <pack>` (boots a server) prints which tuned paths engaged.
