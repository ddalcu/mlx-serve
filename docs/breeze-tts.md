# Breeze-TTS-2 speech

Breeze-TTS-2 runs as the built-in `breeze_tts` media plugin on
`POST /v1/audio/speech`. It supports plain text-to-speech, speaker slots,
instruction-based voice design, and zero-shot voice cloning from a reference
recording **and its transcript**. The published pack produces 24 kHz mono,
16-bit PCM WAV.

This guide covers the server API, not a Breeze-specific app download entry,
Audio pane, or hands-free Voice Mode integration. You need a server build that
includes the Breeze plugin; downloading a model does not add support to an
older binary. See [building from source](building.md) if needed.

## Model and setup

Use the dense bf16
[`mlx-community/Breeze-TTS-2-mlx`](https://huggingface.co/mlx-community/Breeze-TTS-2-mlx)
pack, whose `config.json` declares `"model_type": "breeze_tts"`. This is the
plugin's reference pack. Other quantizations are published, but do not assume
they work with this loader just because their model names are similar.

**Model license:** Breeze-TTS-2 is licensed for research and non-commercial use
only. The server's MIT license does not replace the weights' license. Read the
[model card and responsible-use terms](https://huggingface.co/BreezeBlue/Breeze-TTS-2#license-and-responsible-use)
before using it. Clone only voices you have permission to use.

Download the **whole repository**, not just the root weight shards. For
example, with the separately installed [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
hf download mlx-community/Breeze-TTS-2-mlx \
  --local-dir "$HOME/models/Breeze-TTS-2-mlx"

mlx-serve --model "$HOME/models/Breeze-TTS-2-mlx" \
  --host 127.0.0.1 --port 11234
```

The download includes the root config, tokenizer, weight shards and index, plus
`audio_tokenizer/config.json` and `audio_tokenizer/model.safetensors` (the shared
Mimi audio codec). Keep their directory layout intact. A pack missing
`audio_tokenizer/model.safetensors` is incomplete and is not discovered as a
usable media model.

The bf16 pack is several GB on disk; runtime memory also includes host-side
tables, model state and synthesis buffers. Leave headroom beyond the weight
size, especially when serving chat alongside speech. This guide does not claim
a measured minimum-RAM configuration or a real-time generation speed.

In another terminal, wait for an actual successful health response before
generating speech; a listening port alone does not mean model loading finished:

```bash
curl --fail-with-body http://localhost:11234/health
curl --fail-with-body http://localhost:11234/v1/models
```

The examples below omit `model` because this server was started with one pinned
model. With a multi-model server, copy the Breeze model's exact `id` from
`GET /v1/models` into each request's `model` field. If a pack was downloaded
under a configured `--model-dir` while the server was running, call
`POST /v1/models/rescan` to discover it.

## Plain speech and speaker slots

```bash
curl --fail-with-body http://localhost:11234/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Hello world. This is Breeze speaking.",
    "voice": "S0",
    "seed": 42
  }' \
  --output breeze.wav
```

Without `stream`, success returns raw WAV bytes with `Content-Type: audio/wav`,
not JSON or a hosted download URL. Play the file with any WAV player (`afplay
breeze.wav` on macOS). On a failed request, inspect the error body instead of
trying to play the output file.

`voice` is optional. When supplied it selects a speaker slot `S0` through `S9`
(lowercase `s0` through `s9` also works). These are model speaker tokens, not
Kokoro voice names or Qwen3-TTS CustomVoice names; `"default"`, `"alloy"` and
`"af_heart"` are not Breeze slots. Omitting `voice` leaves out the speaker token;
it is not an alias for `S0`.

## Instruction-based voice design

Use `instruction` to describe a voice or its delivery, separately from the text
to speak:

```bash
curl --fail-with-body http://localhost:11234/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "The forecast is clear skies with a light breeze.",
    "voice": "S2",
    "instruction": "A calm, warm voice with clear articulation.",
    "seed": 7
  }' \
  --output breeze-designed.wav
```

The OpenAI-style spelling `instructions` is an alias for `instruction`. If both
string fields are present, singular `instruction` takes precedence. The same
instruction control can accompany a clone request. Voice descriptions are
model conditioning, not guaranteed numeric speed, pitch, or emotion controls.

## Zero-shot voice cloning

Unlike this server's Qwen3-TTS Base path, Breeze conditions on **both** the
reference audio and the words spoken in that audio. Send:

- `input`: the **new** text to synthesize.
- `ref_audio`: standard base64 of the entire reference WAV file, including its
  header. This is not a filename, URL, data URI, MP3, or raw PCM.
- `ref_text`: a non-empty transcript matching the **reference** recording,
  not the new `input`.

Prepare a clean, short, single-speaker reference as **24 kHz mono PCM WAV**.
Normalize it before sending; do not rely on the Breeze route to resample an
arbitrary recording. A few seconds is a useful starting point, not an enforced
duration limit. Longer references consume prompt space and leave less room for
the target speech.

For example, convert a recording with an existing FFmpeg installation:

```bash
ffmpeg -i recording.m4a -ar 24000 -ac 1 -c:a pcm_s16le reference.wav
```

Build the JSON with a serializer so quotes, newlines and base64 are escaped
correctly. This example uses Python only as a client-side file/JSON utility;
the server itself runs natively:

```bash
python3 - <<'PY' > breeze-clone.json
import base64
import json
from pathlib import Path

print(json.dumps({
    "input": "These are new words, spoken in the reference voice.",
    "ref_audio": base64.b64encode(Path("reference.wav").read_bytes()).decode("ascii"),
    "ref_text": "Replace this sentence with the exact words in reference.wav.",
    "seed": 42,
}))
PY

curl --fail-with-body http://localhost:11234/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-binary @breeze-clone.json \
  --output breeze-cloned.wav
```

Replace the example `ref_text` before running it. Missing or empty `ref_text`,
bad base64, or an undecodable reference WAV is a named request error, not a
fallback to a plain voice. `ref_text` alone does not activate cloning.
The Mimi encoder loads lazily on the first clone request, so that request can
take longer than subsequent ones. Breeze has no `warm_only` request field.

## Request fields

| Field | Type / default | Breeze behavior |
|---|---|---|
| `model` | string; server default | Select the Breeze model by its registry id on a multi-model server. |
| `input` | non-empty string; required | Text to speak. `text` is an alias; `input` takes precedence when both are strings. |
| `voice` | string; omitted | Speaker slot `S0` to `S9`, case-insensitive. |
| `instruction` | string; omitted | Voice/delivery description. `instructions` is an alias. |
| `ref_audio` | base64 string; omitted | Reference WAV for cloning; requires `ref_text`. Prepare 24 kHz mono PCM. |
| `ref_text` | string; omitted | Non-empty transcript of `ref_audio`. |
| `seed` | non-negative JSON integer; `42` | Request-local sampling seed. Repeating the same request with the same model/build is deterministic; it is not a cross-version audio guarantee. |
| `max_frames` | non-negative JSON integer; automatic | Upper bound on generated codec frames **per text segment**, not seconds, text tokens, or a requested duration. Use a positive budget for speech. |
| `stream` | boolean; `false` | `true` selects SSE progress and a final base64 WAV, not incremental PCM playback. |

Unknown plugin fields are refused by name. In particular, do not send
`language`, `temperature`, `top_p`, `top_k`, `max_tokens`, `speaker`,
`warm_only`, or Qwen/Ming-specific task controls. Breeze infers language from the
text; this route has no language override. The plugin allows the common speech
keys `response_format` and `speed` for compatibility, but currently always
returns WAV at the model's generated rate: they do not enable MP3/PCM output or
adjust speaking speed. Omit them rather than assuming OpenAI's full speech
parameter set is implemented.

### Length, stopping and long text

The published codec runs at 12.5 frames per second (one frame is 80 ms).
Generation normally stops on the model's EOS token. The engine estimates a
budget from the text and clamps it by `max_frames`, the pack's configured frame
ceiling (default 1500), and available decoder context. Raising `max_frames`
does not force a longer clip or override those other limits. A low cap can
truncate speech.

Long input is split into segments at sentence boundaries, with oversized runs
split further, to fit the pack's context (2048 positions by default). Speech
from those segments is joined into one output WAV; prosody can restart at
segment boundaries. `max_frames` applies to each segment, so it is **not** a
whole-request duration limit. Long references or instructions still consume
context; shorten them if the prompt cannot fit.

## Progress over SSE

```bash
curl --fail-with-body --no-buffer http://localhost:11234/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "This request reports progress before returning its audio.",
    "voice": "S0",
    "stream": true
  }'
```

Read `data:` events as JSON. `progress` events use stage `"breeze"` with `step`
and `total` counters; these are progress estimates, not playable audio chunks.
Success ends with:

```json
{"type":"complete","format":"wav","data":"<base64 WAV bytes>"}
```

Base64-decode `data` from the `complete` event to obtain the same kind of WAV
as the non-streamed response. A failure after SSE starts arrives as an `error`
event; HTTP 200 alone is not proof of completed synthesis. The server media API
does not expose Breeze PCM chunks or a Breeze WebSocket speech endpoint.

## Troubleshooting and validation limits

| Symptom | What to check |
|---|---|
| Model absent or incomplete | Confirm `model_type: breeze_tts`, all root shards/tokenizer files, and the complete `audio_tokenizer/` directory; rescan the configured model directory after the download finishes. |
| Unsupported geometry / missing weight at load | Use the reference bf16 pack without renaming config fields or mixing files from different conversions. The loader reads geometry from the pack and refuses unsupported layouts. |
| Named HTTP 400 | Read the error message: common causes are an unknown control, an invalid voice slot, missing input, or invalid reference audio/transcript. If SSE already started, inspect its `error` event instead. |
| Wrong words or poor clone | Check that `ref_text` matches the recording, use a short clean reference, and normalize it to 24 kHz mono PCM before encoding it. |
| Cut-off or segmented speech | Remove an unnecessarily low `max_frames`; shorten long references/instructions, or submit shorter sentences. Segmentation can reset prosody. |
| HTTP 503 | The server may still be loading or lack generation headroom. Inspect its logs and free competing model/workload memory. A successful health/status request does not replace an actual speech request. |

The plugin has synthetic conformance tests for generation/stopping and
repeated-forward/decode memory retention, and a real-pack Mimi encoder parity
test. Those are **not** an end-to-end numerical or perceptual oracle for the
published Breeze model. No bit-exact reference-synthesis, clone-quality, or
real-time performance claim is made here.

For contributors, the existing HTTP integration script starts its own server
and checks speech, voice design, seeded repeatability, cloning, SSE and invalid
requests. It skips if the pack is missing and requires a built server:

```bash
BREEZE_MODEL="$HOME/models/Breeze-TTS-2-mlx" \
  bash tests/test_breeze_tts.sh 11441
```

See also the [HTTP API](api.md#breeze-tts-2-speech),
[supported models](models.md#breeze-tts-2), and
[plugin implementation notes](../src/plugins/breeze_tts/AGENTS.md).
