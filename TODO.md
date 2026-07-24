# TODO

* Upscale Images & Video using SeedVR2 One-step diffusion DiT + 3D-causal VAE
* Built in Transcriber
* Prompt to Lyrics helper
* Expand Chat Tools to be able to generate media
* Stretch: DSpark/EAGLE3-class trained drafter (ARahim3/mlx-dspark) — needs a per-target trained 5-layer backbone (none exists for Qwen3.6-27B); 
* M5 Nax
* P2P
* Launch code via cli
* MageFlow 4-bit. 8-bit SHIPPED (`MfLinear` + `tests/convert_mageflow_weights.py` + the two `ddalcu/Mage-Flow-*-MLX-Serve-8bit` mirrors, 17.5GB → 9.2/9.7GB, judged visually equivalent on txt2img/edit/multi-ref). `MfLinear` already solves (bits, group_size) per tensor from geometry, so 4-bit is a converter flag + validation, no engine work. Watch the AdaLN modulation projections (`img_mod.1`/`txt_mod.1`, 33% of the DiT, on the timestep path): at 8-bit they cost ~half the txt2img fixture error and `--keep-bf16 img_mod.1,txt_mod.1` buys e2e 0.990 → 0.995 for +1.27GB (not worth it at 8-bit; may be at 4-bit). Also unquantized and worth ~0.4GB: the TE embedding table, which needs a gather-then-dequantize path (it is read with `mlx_take_axis`, not a matmul).
* 