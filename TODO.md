# TODO

* Upscale Images & Video using SeedVR2 One-step diffusion DiT + 3D-causal VAE
* Built in Transcriber
* Prompt to Lyrics helper
* M5 Nax
* P2P
* Distributed inference between mlx-serve peers: wire mlx_distributed_* (already in mlx-c pin, ring backend over TCP/TB), lan.zig discovery builds the hostfile, pipeline-parallel layer shards first (rank 0 keeps HTTP/scheduler; spec/prefix-cache/batching off for sharded models); win case = models too big for one box
* Settings UI for `--prefill-decode-share` (PR #568): done, draft PR #982 (blocked only by the #978 no-merge freeze) in `app/Sources/MLXServe/Views/SettingsView.swift` + `ServerOptions` field/`toCLIArgs` emit; drop this line once #982 merges
* DeepSeek-V4.1 MLX packs (REAP50, Jundot oQ3e) run on plain MLX ops: decode speed vs oMLX on the same pack, then the hot stages (CSA attention, hc mixing, routed experts) one kernel at a time, each with its own A/B. Cold Engram rows cost most on an external SSD (REAP50 ~15 tok/s cold vs ~31 warm): a decode lookup preads its 24 rows a layer on one thread (`Table.gather` parallelizes from 512 rows).
* DeepSeek-V4.1 EXL3 repack: move the `lib/mlx-stream` pin from our fork to upstream mlx-stream once it carries the host `sdk/` and the `mlx_host` import.
* Prefix reuse for DeepSeek-V4 / V4.1: today V4.1 resumes one conversation (`resumePrompt`, the plugin's `restorePrefix`), V4 none.
* sushi: `trellisAdmitted` checks widths in multiples of 16, the kernels rotate whole 128-wide Hadamard blocks (`format.HAD_DIM`): a bank narrower than a block decodes to garbage. Require multiples of `HAD_DIM` in our fork and upstream.
