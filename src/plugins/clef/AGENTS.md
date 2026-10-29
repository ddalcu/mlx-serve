# clef

Cloudflare Clef joint schema decisions on `POST /v1/decisions`, matching the mlx-community MLX packs: a dense qwen3_5
trunk (+ its Qwen3-VL tower when the pack ships one) and the joint head (`joint_head.safetensors`), every field of a
schema decided jointly.

## Files
- `plugin.zig`: the media table (through `laya/decision_job.Single`, one request at a time) and the tiny pack.
- `config.zig`: claims by marker (`joint_head_config.json`), bills.
- `clef.zig`: the rendering (sorted-key JSON, the system/suffix prompt), images, the trunk readout, the joint head.
- `http_test.zig`: the live HTTP suite (`CLEF_HTTP_URL`).

## Request contract
Laya's shape (`laya/AGENTS.md`) plus `truncate` (bool, default true; checked before the rest) and `images` (up to 16
base64 images or data URLs; remote URLs, `videos` and `media_kwargs` are refused by name). The context is
`min(16384, max_input_tokens)`; past it the refusal names the limit.

## Rules
- Only a dense qwen3_5 trunk (no MoE, no Hadamard pack) is served; anything else is refused by name.
- The head runs bf16 (cast at load); its geometry is checked against `joint_head_config.json` before any allocation.

## Fixtures
- `conformanceFixture`: `sdk/testing/tiny_qwen35` (no tower) and a joint head at 32, one routing + one decoder layer.
- Reference: `clef: live joint head agrees with the published MLX loader` (env-gated in `clef.zig`).
