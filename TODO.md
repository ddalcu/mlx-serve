# TODO

* Upscale Images & Video using SeedVR2 One-step diffusion DiT + 3D-causal VAE
* Built in Transcriber
* Prompt to Lyrics helper
* Expand Chat Tools to be able to generate media
* Stretch: DSpark/EAGLE3-class trained drafter (ARahim3/mlx-dspark) — needs a per-target trained 5-layer backbone (none exists for Qwen3.6-27B); 
* M5 Nax
* P2P
* Launch code via cli
* Parallel chunked downloads in the app. Measured against the HF CDN on 2026-07-25: 1 conn 22.6 MB/s, 8 conns 41.5 MB/s, 16 conns 46.3 MB/s (same file, same minute), so we get about a third of the line. `DownloadManager.swift` pulls files serially (`:543`), one `dataTask` per file with no ranged chunking (`:1181`), and builds a fresh `URLSession` per file per retry (`:1163`) so every file re-pays TCP + TLS + the `resolve/main` 302. Fix = split each file into N ranged chunks written at their own offsets in the `.partial` (the 206 path already exists at `:1723`) + one shared session. Ollama does 16 parts by default, huggingface_hub does it via hf_transfer. Separately: we never send `HF_TOKEN` from the app (the Zig CLI does, `cli.zig:238`), which costs us gated repos and API rate limits, not speed.
* MageFlow 4-bit. 8-bit SHIPPED (`MfLinear` + `tests/convert_mageflow_weights.py` + the two `ddalcu/Mage-Flow-*-MLX-Serve-8bit` mirrors, 17.5GB → 9.2/9.7GB, judged visually equivalent on txt2img/edit/multi-ref). `MfLinear` already solves (bits, group_size) per tensor from geometry, so 4-bit is a converter flag + validation, no engine work. Watch the AdaLN modulation projections (`img_mod.1`/`txt_mod.1`, 33% of the DiT, on the timestep path): at 8-bit they cost ~half the txt2img fixture error and `--keep-bf16 img_mod.1,txt_mod.1` buys e2e 0.990 → 0.995 for +1.27GB (not worth it at 8-bit; may be at 4-bit). Also unquantized and worth ~0.4GB: the TE embedding table, which needs a gather-then-dequantize path (it is read with `mlx_take_axis`, not a matmul).
* 