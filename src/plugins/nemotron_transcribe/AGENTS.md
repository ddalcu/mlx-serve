# nemotron_transcribe

Owns `nemo_speechlm`: NVIDIA Nemotron-3.5-Transcribe-30B-A4B, a speech-augmented Nemotron-H trunk (transcription, translation,
speaker-attributed and rich transcription, audio questions). Pack layout = mlx-audio's `nemotron_transcribe` (Blaizzy/mlx-audio
#1087); converter `scripts/convert_nemotron_transcribe.py` (`--bits 8 --group-size 128`, or `--bf16`). Served on
`/v1/audio/transcriptions` and as chat `input_audio`, both through the checkpoint's own template.

## Files
- `config.zig`: MLX-free claim; the trunk parses through `nemotron_h` (the NeMo `llm_config`); `parseAudio` reads the
  perception module (`perception`, `pe_encoder_config`).
- `plugin.zig`: `nemotron_h`'s table as it is, with `load`, `forward` and `spec` wrapped (the trunk loads under
  `llm.backbone`, the tower beside it) and `MediaHooks` (`prepare`: the clip's samples, `tower`: the rows, `word_times`:
  the CTC word timestamps, `diarize`: the speaker expert's 10 ms activity).
- `perception.zig`: NeMo log-mels, the ASR expert (32-layer RoPE Transformer, q/k LayerNorm), the speaker expert (31
  layers) streaming through a FIFO and an arrival-order speaker cache (`nemotron_diarization`'s `aoscKeep`), their fusion.
- `perception.zig` also holds the CTC timestamp head (`ctc_timestamp.*`: ASR rows x2 by a transposed convolution, a
  2-layer bridge scaled 0.1 and added back, a pointwise projection onto 23,862 tokens + blank) run over the same windows.
- `ctc_align.zig`: NVIDIA's single-pass banded aligner (speaker streams mapped to Sortformer columns by token
  distribution and a linear assignment, an anchor-guided band, the log activity as a prior, boundary refinement inside
  the diarization's segments). MLX-free; `fixtures/ctc_align.json` is NVIDIA's own functions on seeded scenarios
  (`tests/dump_ctc_align_fixtures.py`) and the Zig tests demand the same times to 1 µs.
- `proof_test.zig`: the tower, the hooks and a spliced prefill over the tiny pack.

## Facts
- The perception module stays bf16 in a quantized pack; only `llm.*` quantizes. Mamba `A_log`/`D`/`dt_bias`, the router and
  its correction bias keep f32 in the pack.
- The prompt's rows are `<|audio|>` (the engine requires its `placeholder_ids.audio` in the run), id `vocab_size`: one
  past the embedding table. `forward` swaps those ids for `<SPECIAL_999>` (`config.spliceId`; a pack too small for it
  takes its last id, `parse` refuses a clash) before the trunk gathers them, and the splice (`base.placeholders.audio`)
  overwrites the rows. The engine hands its ids as a device array (`Forward.ids`, `tokens` empty), so the swap covers
  both forms; a test that passes host `tokens` only never runs the engine's path (`docs/gotchas/engine-mlx.md`).
- A user message that tokenizes to `<SPECIAL_999>` (the stand-in) is spliced like an audio row: accepted, unlikely.
- The speaker expert reads the RAW mels and the ASR expert the normalized ones (`perception.py`).
- The MTP head (`llm.mtp.*`: one attention block and one MoE block reused for two depths) is `nemotron_h`'s own head
  (`mtp_head.zig`, TensorFold's layout under `llm.mtp.`), loaded from the main pack's shards; `/v1/audio/transcriptions`
  decodes with it (greedy output equals serial decode), chat uses it by the usual MTP rules.
- Word times are the alignment of the transcript the model wrote: each word tokenizes alone with the pack's
  `alignment_tokenizer/`, a word is the CTC head's frames on the path (end capped `maximum_token_len`), then refined to the
  next onset or the speaker's segment end. The constants are the checkpoint's `ctc_timestamps.alignment_config` where it
  names a key NeMo's aligner reads; its `boundary_interpolation` and band settings belong to code that is not in NeMo
  Speech and are not used.
- `diarized_json` asks the model for the card's speaker-tagged transcript; the aligner reads the `<spk:N>` turns, maps each
  tag to a Sortformer column of the WHOLE clip's activity (so a voice keeps its column across the 180 s pieces) and
  names every word's speaker from it. A transcript without tags is timed without speaker information and its speakers
  come from the activity overlap. The model's own tagging is what it is: macOS `say` voices collapse to two tags.
- A clip is split into 40 s windows with 50 rows of context either side; features are normalized over the WHOLE clip first.

## Rules
- A routed expert's `fc2` reads 1856 columns, which no 128-wide group divides: it quantizes at group size 64 as a per-path
  entry of the config's `quantization` (24 of them, the MTP block's included); the reader resolves each weight from its own geometry.
- The pack's `tokenizer_config.json` drops its inline `chat_template` (the converter): it is the older template, without
  audio parts, and the host prefers an inline template to `chat_template.jinja`.
- Transcription is the card's recipe (`transcription.nemoPrompt`) as the user turn, thinking off; `language` names the
  language in it and `prompt` takes the contextual recipe, never both.

## Not served
- `language` is the Qwen3-ASR list (the host's `/v1/audio/transcriptions` validation), not the card's 40 languages: a
  language outside it is a named 400; leave `language` out and the model detects it.
- The DFlash2 drafter (`NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2`) and NVIDIA's tuned band settings.
- Words without spaces (Chinese, Japanese, Thai) are timed as one unit per whitespace run: NeMo splits them into
  characters or words, which is not ported.
