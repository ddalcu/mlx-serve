# kev

Kev typed decisions on `POST /v1/decisions`: a qwen3_5 trunk + Kev's pointer head (`kev_head.safetensors`,
`z_i = k(h_close_i) · q(h_decide) / sqrt(P) / T`), converted by `tests/convert_kev_weights.py`.

## Files
- `plugin.zig`: the media table (through `laya/decision_job.Single`, one request at a time) and the tiny pack.
- `config.zig`: claims by marker (`kev_config.json` wins over the trunk's qwen3_5 `config.json`; app twin
  `DownloadManager.markerModelType`), bills.
- `kev.zig`: the request rendering (`kev.api.render`, not JSON), the five delimiters, the branch scoring, the head.

## Request contract
As Laya's (`laya/AGENTS.md`): `state` + `questions` (noul / choice / score), the shared limits, the answer echoing
`Request.model_id`. `kev_config.json` is refused unless it is exactly the converter's format.

## Rules
- The state runs once; each question runs from a restored snapshot of the state's KV cache, GatedDeltaNet state and
  position (`scoreBranches`), so questions never see each other. The snapshot covers exactly qwen3_5's caches: any
  other base is refused by name (`KevUnsupportedBase`).
- Caller text goes through the marker escape before the tokenizer: a `<|name|>` in a state never becomes a delimiter.

## Fixtures
- `conformanceFixture`: `sdk/testing/tiny_qwen35` (two layers, GDN + gated attention) over a byte tokenizer with the
  delimiters as added tokens, a pointer head at 32.
- Reference oracle: `KEV_TEST_MODEL` + `KEV_FIXTURES` env-gated test in `kev.zig`.

## The row from the root architecture table

Kev typed-decision packs: qwen3_5 trunk + pointer head; `kev_config.json` wins over the trunk's config.json
(this plugin's `claims`, app twin `DownloadManager.markerModelType`); the state runs once, each question from a
KV/SSM snapshot (`kev.zig`)
