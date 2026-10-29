# glm_moe_dsa plugin

GLM-5.3 (`glm_moe_dsa`, GlmMoeDsaForCausalLM), served only from mlx-stream packs (`expert_streaming`): the routed
experts stream from the affine or EXL3 bank (`bank.zig`, `exl3_bank.zig`), the residents load from the shards the
index names (`glm_moe_dsa.zig` checks every tensor against their headers).

- Not a `plugin.zig` table: `arch.zig` wraps the arch for the engine's `modules/stream_glue.zig`, reached once per
  prompt, step, handover or draft round, never per layer. `glm5_next` claims `glm_moe_dsa` only to refuse it there.
- A pack this build does not implement is refused by name before any MLX array exists.
- Draft lane: the release's MTP layer (`glm_moe_dsa_mtp`), off unless the model's `mtp_depth` sets it.
- Oracle: the checkpoint's bundled `glm_moe_dsa.py` (mlx-lm's `deepseek_v32` with GLM's indexer schedule, fp32 indexer
  scores and router logits). Tests: `zig build mlx-stream-test`.
