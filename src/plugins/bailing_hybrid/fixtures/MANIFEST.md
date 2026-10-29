# bailing_hybrid fixtures

`*.tape`: the legacy forward's values (`engines/mlx/forward/layered.zig`'s `forwardMoeWith`) recorded on a tiny
random pack written in memory by `glm5_next/fixture.zig` (`lingFiles`, seed 11: 4 layers, KDA at 4 heads of 128, one
MLA layer of 4 heads (nope 64, rope 64, v 64, q/kv LoRA 64), 8 experts in 4 groups, 4-bit group 64). Arrays past 1 KiB
are kept as a digest and a 64-element sample. Replayed by `glm5_next/proof_test.zig` (`steps`) and
`glm5_next/served_test.zig` (`served`); recorded while the legacy forward existed, so they cannot be regenerated.
Configs claimed here: `src/fixtures/model-configs/bailing_hybrid-ling-3.0-*.json` (tiny 4-bit, flash oQ4e-MTP).
Gate model: `rapid-mlx/Ling-3.0-tiny-MLX-4bit`.
