# TODO

* Upscale Images & Video using SeedVR2 One-step diffusion DiT + 3D-causal VAE
* Built in Transcriber
* Prompt to Lyrics helper
* Expand Chat Tools to be able to generate media
* Stretch: DSpark/EAGLE3-class trained drafter (ARahim3/mlx-dspark) — needs a per-target trained 5-layer backbone (none exists for Qwen3.6-27B)
* M5 Nax
* P2P
* Launch code via cli
* DSV4 prefill attention: hd-512 flash kernel with in-kernel sinks + block-skip pre-pass (reference in `lib/ds4/metal/flash_attn.metal`); post-sorted-gather prefill is ~all attention-side. Related A/Bs from the mxfp4 branch: direct-RHS dequant-in-kernel GEMM for the 32-2048 M band, tiny_pair_mv at verify width (C≈5, untested — the −2.5% MOE_GATEUP number was M=1).
* DSV4 back-pocket: w2 3→4-bit requant (~+11 GB at g128); weak case after imatrix calibration, only if decode becomes kernel-bound.
* DSV4 loose ends: prefill side-by-side vs the ds4 GGUF engine (sequential boots only); `docs/reference.md` has no deepseek_v4 section; retiring `lib/ds4` is a call to make once parity is proven.
* DSV4 paper cuts: `--dspark` setenv clobbers `MLX_SERVE_DSV4_DSPARK=force` (main.zig:553); converter `SOURCE_REPO` stamp missing "-0731"; `iogpu.wired_limit_mb=124000` does not survive reboot (LaunchDaemon or a louder fit-gate boot warning).
* mlxfast round-5 port candidates: nvfp4 nibble bit-trick + steel tile probe first.
* CLAUDE.md root is over the ~100 KB growth-policy target. Needs a curation pass.
* `zig build test` reports a "failed command" on FIRST runs and is 100% green on a direct re-run. Suspected Metal contention between parallel test binaries. Benign, but worth pinning down.
* Streaming with `tools`: replace the buffer-the-whole-turn design with an INCREMENTAL parser that emits diffs and holds back only the minimal ambiguous suffix — what vLLM (`extract_tool_calls_streaming`) and llama.cpp (`common/chat.cpp` partial parse + diff) do. Today `streamShouldBufferForTools` is already fine-grained (false ⇒ no markup, no partial marker at the tail), so the tool side is close; what's missing is that every non-tool decision on that branch is all-or-nothing. The 2026-08-04 stopgap streams reasoning during `.hold_thinking` (see `reasoning_streamed` in server.zig) — it fixes the visible symptom only. The real version also unlocks streaming tool-call ARGUMENTS progressively, which we deliberately do not do today (args ship in ONE delta so clients never render half-written file contents) — decide that separately, it is a client-visible contract change, not a free win.
* MageFlow 4-bit (8-bit shipped; `MfLinear` already solves bits/gs per tensor, so it's a converter flag + validation). Watch the AdaLN modulation projections (`img_mod.1`/`txt_mod.1` — `--keep-bf16` may be worth it at 4-bit) and the unquantized TE embedding table (needs a gather-then-dequantize path).
