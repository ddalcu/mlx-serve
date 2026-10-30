# Add a model

1. Read mlx-lm's model file for the arch (merged or open PR) and the MLX packs already on Hugging Face; match their
   weight names, config keys and quant layout.
2. Copy `src/plugins/_template/` to `src/plugins/<name>/` (a family that is one block away from an existing plugin
   extends it instead: `extend-a-model.md`). The build registers it by scan.
3. `config.zig`: claim the `model_type`s, parse the config with the checked readers, refuse what you do not implement
   by name. It stays MLX-free (the build refuses one that reaches `mlx/`, `engines/`, `host/`, `kernels/`).
4. `plugin.zig`: build the forward from `src/sdk/bases/` and `src/sdk/blocks/`; declare `Caps` honestly (batching,
   MoE, exactness). Import only `sdk/`, `abi/`, `mlx/`, `kernels/`, `env.zig` and your folder (`zig build unit-closure`).
5. Proof: a fixture config in `src/fixtures/model-configs/`, a parity test against the reference (an env-gated dump
   script `tests/dump_<name>_fixtures.py`), a contract for each Metal kernel (`add-a-kernel.md`), a bf16 pack and a
   quantized one.
6. `AGENTS.md` beside it (what it owns, its rules, its packs and oracle) and a row in the root's architecture table.
