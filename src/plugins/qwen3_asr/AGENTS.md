# qwen3_asr plugin

Owns `qwen3_asr` (Qwen3-ASR and Qwen3-ForcedAligner packs, mlx-audio's layout). The table is the qwen3 plugin's
dense trunk (`plugins/qwen3`) parsed from `thinker_config.text_config`, the quantization at the outer root, and
`Parsed.placeholder_ids.audio` = the thinker's `audio_token_id`: the driver hands the tower's rows in
`Forward.media` and the decoder splices them there.

- The audio tower (`audio_tower.zig`), the transcription flow and the aligner (`forced_align.zig`) stay the engine's.
  They read the trunk through the shell. The aligner reads hidden rows (`skip_lm_head`) and the head that
  `trunk_weight` lends the shell (`lmHeadForDraft`).
- Proof (`proof_test.zig`): served through the driver vs the legacy shell, 4-bit: audio rows spliced across a chunk
  boundary, the aligner's hidden-row + head read.
- Parse declines a thinker trunk other than qwen3 and everything the qwen3 parse declines.

## Rules (from root AGENTS.md)
- A placeholder id the splice mask does not name never reaches the prompt; an id of 0 is UNSET, never token 0
  (audio-only Qwen3-ASR heard silence).
- The aligner shares the root `model_type`; the thinker's `qwen3_forced_aligner` is the tell (capability
  `alignment`, loaded when a request asks for times, never a route of its own).

## The row from the root architecture table

Qwen3-ASR (mlx-audio layout, `thinker_config` nesting): a plain `qwen3` trunk + audio tower whose rows splice at
`<|audio_pad|>`; serves `/v1/audio/transcriptions` only (no chat template, not chat-pickable). Its forced aligner
shares the root `model_type`; the thinker's `qwen3_forced_aligner` is the tell (arch hint of that name, capability
`alignment`, loaded when a request asks for times, never a route of its own). Oracles:
`tests/dump_qwen3_asr_fixtures.py`, `tests/dump_qwen3_aligner_fixtures.py`
