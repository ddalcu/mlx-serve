# TODO

* Upscale Images & Video using SeedVR2 One-step diffusion DiT + 3D-causal VAE
* Built in Transcriber
* Prompt to Lyrics helper
* Expand Chat Tools to be able to generate media

## Speed (post MTP-round-v2 follow-ups)

* Flip the last 3 contested prefill cells vs MTPLX (0.5k/8k/16k, −0.3 to −2%): port their vendored Metal kernels — `MTPLX/native_extensions/verify_mlp` (fused gate_up + gdn_tail) and `MTPLX/vllm_metal` (paged attention ops) — into our Zig/mlx-c stack via `mlx_fast_metal_kernel`, and/or their mlx-fork qmm patches (fragment `mlx-mtplx-0.31.2-qmm`)
* Custom head_dim-256 steel prefill SDPA kernel: MLX 0.32 `sdpa_full` caps at hd 128 while every Gemma-4/Qwen-3.5/3.6 ships hd 256, so prefill materializes the scores tensor (decode `sdpa_vector` already covers 256); a fused 256 prefill kernel lifts the whole fleet
* MoE MTP sidecar arm in mtp.zig (35B-A3B Forge artifacts pack `mtp.layers.0.mlp.experts.gate_up_proj` + shared expert + gate) — today the 35B serves with the head disabled via graceful degrade
* MTP history last-window policy (others runs window 8192 above 16k ctx): bounds the head's prefill history build + its KV growth at long context; needs an acceptance A/B before shipping
* Re-evaluate `DEFAULT_DEPTH` 1 → 3 for MTP now that partial accepts are ~free (capture rollback + 3-bit draft head): needs creative-content validation beyond the coding-agent bench before flipping the shipped default
* Stretch: DSpark/EAGLE3-class trained drafter (ARahim3/mlx-dspark) — needs a per-target trained 5-layer backbone (none exists for Qwen3.6-27B); 