# d1

LiquidAI D1-3B typed decisions on `POST /v1/decisions` (bf16, the card's own layout): an LFM2-VL trunk whose answer
is a softmax over each option's token group at the LM head; no new weights. Oracle: `tests/dump_d1_fixtures.py`.

## Files
- `plugin.zig`: the media table (through `laya/decision_job.Single`, one request at a time) and the tiny pack.
- `config.zig`: claims by marker (root `config.json` `auto_map` names `modeling_d1.D1Model`; app twin
  `DownloadManager.markerModelType`), bills.
- `d1.zig`: the prompt (state block, question text, option codes), the readout groups, the answers.

## Request contract
As Laya's (`laya/AGENTS.md`); `images` is refused by name (text only in this build).

## Rules
- The state runs once (via `kev.scoreBranches`); each question from its snapshot. The trunk is the lfm2 plugin's,
  bound here (`bindTrunk`: the D1 marker keeps the registry's claim off the pack) and loaded by the served driver;
  `plugin.zig` proves its branch logits bit-equal to the engine's own trunk (`fixtures/trunk.tape` once it is gone).
- Option codes need single tokens (a letter and its spaced form, digits, yes/no): a tokenizer without them is refused
  by name (`D1NoSingleToken`), never answered from a multi-token readout.
- A torchao int8 pack (`quantization_config`) is refused by name: only the bf16 pack is served.

## Fixtures
- `conformanceFixture`: two LFM2 layers (gated conv, attention) at 128 under an LFM2-VL config with the card's
  `auto_map`, byte tokenizer; choice and score questions (bytes hold no single-token "yes").

## The row from the root architecture table

LiquidAI D1-3B decision packs (bf16, the card's own layout): LFM2-VL trunk whose root config.json `auto_map` names
`modeling_d1.D1Model` (this plugin's `claims`, app twin `DownloadManager.markerModelType`); the answer is a softmax over
each option's token group at the LM head, no new weights (`d1.zig`, state runs once via `kev.scoreBranches`); text
only, torchao int8 refused by name. Oracle: `tests/dump_d1_fixtures.py`
