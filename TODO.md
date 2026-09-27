# TODO

* Upscale Images & Video using SeedVR2 One-step diffusion DiT + 3D-causal VAE
* Built in Transcriber
* Prompt to Lyrics helper
* M5 Nax
* P2P
* Distributed inference between mlx-serve peers: wire mlx_distributed_* (already in mlx-c pin, ring backend over TCP/TB), lan.zig discovery builds the hostfile, pipeline-parallel layer shards first (rank 0 keeps HTTP/scheduler; spec/prefix-cache/batching off for sharded models); win case = models too big for one box
* `/v1/messages` + tools does not stream thinking (every Claude Code turn): the gated stream's `.hold_thinking` arm (server.zig ~16163) only buffers, so the whole thought ships as ONE `thinking_delta` at `</think>` and Claude Code looks frozen on long thoughts (measured on Flash-Next: 1 delta at 2.4 s with tools vs 116 deltas from 0.47 s without). Port the chat-completions stopgap (server.zig ~10642, `unstreamedReasoning` + `reasoning_streamed`): open the thinking block on the first reasoning token, emit only the fresh remainder per tick, and make `.split_think` and the end-of-stream flush send the rest into the SAME block and close it, never a resend. Red test: tighten `tests/test_messages_stream_thinking_tools.sh` to require several thinking deltas before the text block starts (today it only checks >= 1).
* /v1/responses a per-model enable_thinking/reasoning_effort default applies on chat/completions and messages but not Responses, so Codex on the same model gets the arch default. Inconsistent, but Responses deliberately never thinks unasked today, so acceptable if documented in the CLAUDE.md line. Check it says so.
* Settings UI for `--prefill-decode-share` (PR #568) in `app/Sources/MLXServe/Views/SettingsView.swift` + `ServerOptions` field/`toCLIArgs` emit, once merged
