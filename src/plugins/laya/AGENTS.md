# laya

Laya typed decisions on `POST /v1/decisions`: an mmBERT/ModernBERT encoder + two head layers + marker scorer + act
head (fp16), prompt layout and output JSON mirroring `laya_mlx`. Packs: `aac6fef/laya-multilingual-mlx`.

## Files
- `plugin.zig`: the media table (single job and the merged `step_batch` pass) and the tiny conformance pack.
- `config.zig`: claims (no root config.json: told by `rl_agent_config.json` + `encoder/config.json`), bills.
- `decision_job.zig`: what every decision plugin shares (Laya, Kev, D1, Clef): the host's limits from
  `LoadCtx.settings_json`, the body parse, the named refusals.
- `laya.zig`: config, tokenizer layout, encoder + head forward, `predictMany`, the Python-exact JSON writer, the
  request JSON helpers Kev/D1/Clef reuse.

## Request contract
`{"state": <string|object|array>, "questions": {id: {"type": choice|score|noul, "instructions", "criteria"}}}`; the
answer echoes `Request.model_id` (never the body's `model`). Limits: `max_questions` (64), `max_input_tokens` (32768),
set on the host by `MLX_SERVE_LAYA_MAX_QUESTIONS` / `_MAX_INPUT_TOKENS`; a refusal names the value in force.

## Rules
- Cross-request merging is a speed feature: requests queued together run through ONE `predictMany` (`step_batch`),
  each failing on its own (its prompt, its token limit, a non-finite row of its own).
- Refusal texts are byte-identical to the legacy endpoint (composed ones through `Request.refuseFmt`).
- JSON nesting is bounded before the parse (`checkJsonDepth`): recursive walkers overflowed a thread stack.

## Fixtures
- `conformanceFixture`: ModernBERT at 32, one global + one sliding layer, byte tokenizer, one head layer.
- Reference oracles: `LAYA_TEST_MODEL` / `LAYA_FIXTURES` env-gated tests in `laya.zig`.

## The row from the root architecture table

Laya typed-decision checkpoints (no root config.json: classified from `encoder/config.json` + `rl_agent_config.json`
by this plugin's `claims`, app twin `DownloadManager.markerModelType`): ModernBERT encoder + 2 head layers +
marker scorer + act head, fp16
