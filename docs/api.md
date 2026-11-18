[English](api.md) · [简体中文](zh-CN/api.md)

# HTTP API

Everything lives on one port (`http://localhost:11234` by default): OpenAI, Anthropic and Ollama wire protocols, plus native media generation endpoints.

## POST /v1/chat/completions

```bash
curl http://localhost:11234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a haiku about programming."}],
    "max_tokens": 256,
    "stream": true
  }'
```

Supports `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, `stream_options`, `tools`, `response_format`, `repetition_penalty`, `presence_penalty`, `logprobs` / `top_logprobs`, `reasoning_effort` / `enable_thinking` / `reasoning_budget_tokens`, plus per-request `kv_quant`, `kv_attn_mode` and `suppress_cjk` overrides. Messages can include `image_url` content blocks (base64 or URL) for vision-capable models. Usage always carries `prompt_tokens_details.cached_tokens`, and `completion_tokens_details.reasoning_tokens` when the reply reasoned (streamed on the `stream_options.include_usage` chunk). A reply cut short because the model went in circles reports `finish_details: {"type": "repetition_loop"}` beside `finish_reason`.

`"suppress_cjk": true` keeps every token that writes CJK (Han, kana, Hangul, CJK punctuation, fullwidth forms) out of sampling, drafts included, on `/v1/messages`, `/v1/responses` and `/v1/completions` too. A non-boolean is a 400, `null` leaves it off, and a model whose engine samples elsewhere (GGUF, Neural Engine) refuses it by name.

### Directional ablation

With `--ablation-profiles <dir>` and `--ablation-key-file <path>`, `/v1/chat/completions` and `/v1/completions` take
`"ablation": {"profile": "<name>", "strength": 0.0-1.0}` (strength defaults to 1) with the key in the
`X-Ablation-Key` header. For that request only, the model removes the profile's direction from everything it writes to
its residual stream (Arditi et al. 2024, "Refusal in Language Models Is Mediated by a Single Direction"), and the
response carries `X-Ablation: profile=<name>; strength=<s>; hash=<hex>`, the only confirmation that it ran ablated.
A missing or wrong key is a 401. Every other problem is a named 400: an unknown, refused or mismatched profile, a bad
strength or field, a model whose plugin lacks `Caps.ablation`, or the field sent to `/v1/messages`, `/v1/responses`,
the Ollama routes or a proxied `<id>@<provider>` / `<id>@<peer>` model. Other endpoints ignore it. The key is
compared with surrounding whitespace trimmed; browsers may send `X-Ablation-Key` and read `X-Ablation` cross-origin.

A profile is a directory: `manifest.json` (`"format": "mlx-serve-ablation"`, `"version": 1`, `hidden_size`,
`base_config_sha256`: sha256 of the base pack's exact `config.json` bytes) and `direction.safetensors` (one tensor
`direction`, `[hidden]`, F32/F16/BF16, normalized at load). Make the key with
`openssl rand -hex 32 > key && chmod 600 key` and serve only over TLS or loopback: the header is a bearer secret.

## POST /v1/messages (Anthropic)

```bash
curl http://localhost:11234/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "mlx-serve",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Write a haiku about programming."}]
  }'
```

Compatible with Claude Code (`ANTHROPIC_BASE_URL=http://localhost:11234 claude`) and Anthropic SDKs. Supports streaming, tool calling, and extended thinking. Usage always carries `output_tokens_details.thinking_tokens` (0 without thinking; streamed on `message_delta`).

## POST /v1/responses (OpenAI Responses API)

```bash
curl http://localhost:11234/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "input": "Write a haiku about programming.",
    "stream": true
  }'
```

Stateful chains via `previous_response_id`, full streaming SSE with per-event `sequence_number`, schema-conformant envelope with `tools` / `tool_choice` / `text` / `reasoning` / `usage` echo. `POST /v1/responses/compact` returns an opaque base64 history blob that round-trips back as a `compaction` input item without any LLM call. Same endpoint also accepts an `Upgrade: websocket` handshake — each text frame is a `response.create` JSON message, and each SSE event becomes one outbound text frame.

## Ollama API

`/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/ps`, `/api/embed`, `/api/pull` speak the Ollama wire (NDJSON streaming, tool calls with object arguments, `thinking`, `format` JSON schemas, `name:latest` model names), so the whole Ollama client ecosystem works against mlx-serve unchanged — point Raycast, Obsidian, Enchanted, Open WebUI, `ollama-python`/`js` at `http://localhost:11234` where they had `http://localhost:11434`.

## Other endpoints

- `GET /` — built-in web console (MLX Serve Studio): chat, image/audio/video generation, Library, Monitoring, API reference
- `GET /health` — health check
- `GET /v1/models` — list loaded models with capabilities + engine info
- `POST /v1/completions` — text completions; `prompt` is a string or token ids (`[1, 2, 3]`, as lm-eval sends them), one prompt per request. `echo: true` returns the prompt ahead of the completion and, with `logprobs`, every prompt token's logprob (the first is `null`); it is refused with `stream`
- `POST /v1/embeddings` — text embeddings (BERT, EmbeddingGemma, and last-token pooling models like Qwen3-Embedding; pooling follows the checkpoint's sentence-transformers metadata, `dimensions` truncates and renormalizes; EmbeddingGemma 2 wants its task prefix in the text, `task: search result | query: …` for a query and `title: none | text: …` for a document). EmbeddingGemma 2 also embeds images and video: send `messages` (one prompt, one embedding, like vLLM's chat embeddings) instead of `input`, with `{"type": "text"}`, `{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}` and `{"type": "video_url", "video_url": {"frames": ["data:image/jpeg;base64,…", …]}}` parts. The parts are concatenated in order (system turns first, as the model's own template does), images keep their aspect ratio and are sized to 280 soft tokens and video frames to 140 (`processor_config.json`), at most 32 frames of a longer clip are kept, and the whole prompt must fit the 8192-token window. Remote URLs are not fetched and audio is refused by name.
- `POST /v1/decisions` — Laya typed decisions: `{"model", "state": <string|object>, "questions": {id: {"type": "choice"|"score"|"noul", "instructions", "criteria"}}}` returns laya's `predict` schema (`answers` with `type`, `confidence`, `action.act_probability`, plus `choice`+`probabilities`, `score`+`legend`+`probabilities`, or `noul`). An object state is serialized like Python's `json.dumps`. Limits: 64 questions (`MLX_SERVE_LAYA_MAX_QUESTIONS`), 32768 input tokens (`MLX_SERVE_LAYA_MAX_INPUT_TOKENS`), 4 MB body. A Kev pack (`kev_config.json`) takes the same request, `choice` criteria as an object or a list of labels, and answers without `action` (and without `confidence` on `noul`); the same question and input-token limits (the `MLX_SERVE_LAYA_*` pair above), 255 options, 8192 tokens per question including the state. A D1 pack (`LiquidAI/d1-3B`, bf16) takes the same request as text only: `images` is a 400, and its `choice` and `score` answers carry a `confidence` (the chosen probability) while `noul` answers carry none
- `POST /v1/images/generations`, `POST /v1/images/edits` — image generation and instruction edits; the edits endpoint speaks the OpenAI SDK's multipart shape (`client.images.edit`), including repeated `image[]` for multi-reference
- `POST /v1/audio/speech` — Qwen3-TTS (Base: `ref_audio` clones a voice; CustomVoice: `voice` names a built-in speaker, `instructions` style it; VoiceDesign: `instructions` describe the voice; `seed`), Kokoro (`voice` picks or blends one of 54) or Ming-3.0-tts (below), WAV out
- `POST /v1/audio/transcriptions` — speech to text with Qwen3-ASR (mlx-community's `Qwen3-ASR-0.6B`/`1.7B` packs), the OpenAI SDK's multipart form (`client.audio.transcriptions.create`): a WAV `file` (any rate and channel count; one past 20 minutes is transcribed in pieces cut where it is quietest), optional `language` (ISO code or name; omitted, the model detects it), `prompt` (context such as names to spell), `temperature`, `response_format` `json`, `text`, `verbose_json`, `srt`, `vtt` or `diarized_json`, and `timestamp_granularities[]` (`word` adds `words`). Times come from Qwen3-ForcedAligner (`mlx-community/Qwen3-ForcedAligner-0.6B-8bit`, listed with the `alignment` capability and loaded when a request asks for times): `verbose_json` segments end at sentences and pauses, every word carries `start`/`end`, `srt`/`vtt` are those segments, and a clip past 180 s is cut where it is quietest and each piece transcribed and timed, as the official toolkit does. It times Chinese, Cantonese, English, German, Spanish, French, Italian, Portuguese, Russian, Korean and Japanese; another language, or no aligner on disk, gives one segment over the clip (`srt`/`vtt` without an aligner are a named 400). `diarized_json` is OpenAI's speaker-labelled shape (`segments` with `speaker` "A", "B", … in the order they first speak) from Nemotron 3 Diarization (`mlx-community/Nemotron-3-Diarization-8bit`, up to eight speakers, listed with the `diarization` capability), run once over the whole clip; a segment also closes where the speaker changes. Without the diarizer on disk it is a named 400; `known_speaker_names[]` is too. With Nemotron 3.5 Transcribe loaded none of this needs a companion: it times every word with its own CTC head (NVIDIA's banded aligner; a pack converted without the head, like mlx-audio's, answers `verbose_json` untimed and `srt`/`vtt` with a named 400), `diarized_json` asks it for the card's speaker-tagged transcript and names each word's speaker from its own speaker model, and it drafts with its MTP head. No `stream`; other audio formats are named 400s
- `POST /v1/audio/music-generations` — text-to-music, WAV out: ACE-Step (48 kHz stereo, fast) MiniMax Music 3 (`lyrics` required, 44.1 kHz, songs up to six minutes) or YuE2 (`lyrics` required, 48 kHz, plans an editable ABC score: `cot` and `abc`)
- `POST /v1/video/generations` — LTX-Video 2.3 / 2.5 or MiniMax-H3; base64 `rgb8` frames plus `pcm_s16le` audio, mux on your side. LTX 2.5 takes `"decoder": "diffusion"` for its sharper diffusion decoder; long H3 clips chain via `chain_windows`. Opt-in `"preview": true` on `"stream": true` attaches a Latent2RGB JPEG to each denoise `progress` event (`preview_frames`, `preview_max_side`)
- `POST /v1/3d/generations` — Hunyuan3D-2.1, TRELLIS.2, Pixal3D and Idol (an image to a 3D Gaussian splat: a GLB with `KHR_gaussian_splatting`), base64 GLB
- `POST /v1/load-model`, `POST /v1/unload-model` — load a discovered model (or one by absolute path), free one now; `"default": true` makes the loaded model the serving default without a restart
- `POST /v1/models/rescan` — pick up models downloaded while the server runs (the app calls it after every download)
- `POST /tokenize`, `POST /detokenize`, `GET /props` — tokenizer round-trip and llama.cpp-style server props
- `GET /metrics`, `GET /metrics.json` — Prometheus + JSON (needs `--metrics`)
- `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}` — fetch / delete stored responses

Every media endpoint takes `"stream": true` for SSE progress ending in a base64 `complete` payload. Video streams also accept `"preview": true` for a cheap JPEG on each denoise step (off by default; cached-velocity H3 steps stay preview-less). Media LoRAs use one grammar everywhere: `lora_paths` + `lora_scales`, up to 8, stacked. The paths are files on the server's disk, so only a client on the server's own machine may send them; from anyone else (LAN sharing, `--api-key`) a request carrying them is a 403.

### Qwen3-TTS voice cloning limits

On `POST /v1/audio/speech`, Qwen3-TTS uses `ref_audio` (a base64-encoded WAV) to produce a speaker embedding. This is not transcript-conditioned cloning:

- `language` pins the spoken language: a name from the checkpoint's `talker_config.codec_language_id` (any case: `english`, `portuguese`, `spanish`, …). Omit it or send `auto` to let the model infer the language from the text, as before. Any other value, a dialect key included, returns HTTP 400. Pin it when the text could pass for a close language (Brazilian Portuguese read with Spanish phonetics).
- The backend does not read `ref_text`; a reference transcript does not condition the clone.
- A non-empty `ref_audio` must be a decodable WAV and the model must have a speaker encoder. Invalid base64/WAV or a missing encoder returns HTTP 400 before synthesis or SSE headers. Omit `ref_audio` (or send an empty string) to use the plain voice.
- `voice` (or `speaker`) and `instructions` are honoured or refused, never dropped. A CustomVoice pack needs `voice`, one of its `talker_config.spk_id` names (the 400 lists them); its 1.7B size also takes `instructions` as a speaking style. A VoiceDesign pack needs `instructions`, a description of the voice. A Base pack takes neither (`voice` may be absent or `default`).
- To check a reference without generating speech, send `"warm_only": true` with `ref_audio` to the same endpoint. Missing or undecodable reference audio and a missing speaker encoder return HTTP 400 instead of silently falling back. This checks reference handling, not pronunciation or accent quality.

### Ming-3.0-tts speech

`POST /v1/audio/speech` on a Ming-3.0-tts-0.5B pack answers a 44.1 kHz mono WAV. The server loads the bf16 pack (`inclusionAI/Ming-3.0-tts-0.5b`, `mlx-community/Ming-3.0-tts-0.5B-bf16`) and the affine quantized packs (`mlx-community/Ming-3.0-tts-0.5B-8bit`, `jorgepb96/Ming-3.0-tts-0.5B-4bit`). `input` takes up to 50,000 characters: long text is split by paragraph, sentence and clause, and joined with pauses.

- `voice`: a built-in voice (`US_Emma`, `UK_George`, `UK_Charlotte`, or a Chinese name). `instruction`: a description of a designed voice.
- `ref_audio` (base64 WAV, at most 20 s) clones a voice; `ref_text` is its transcript and improves the clone. French, German, Spanish, Japanese and Korean `input` needs `ref_text` to clone in that language's path.
- `language`: `fr`, `de`, `es`, `ja`, `ko`, `zh`, `en` or `auto`. Absent, these languages are detected from `input`. French numbers, dates and typography are spelled out first.
- `task` with `controls` selects the reference's other tasks: `BASIC` (`speed`, `pitch`, `volume`), `EMOTION` (`emotion`), `DIALECT` (`dialect`), `IP`, `FREEFORM`, `VOICE_DESIGN` (`style`), `MULTILINGUAL`. A field a task does not take is a named 400.
- `seed` (default 0) repeats a take while no chat runs on the server: with chat alongside, the same seed gives a different take. `pause_scale` (0 to 4) scales the pauses between fragments. Anything else is a named 400.

### Qwen-Image transparent PNG output

`POST /v1/images/generations` accepts `"transparent": true` for Qwen-Image-2.1.
It preserves the VAE's native fourth (alpha) channel in the returned PNG, including
SSE `complete` responses. Omitted or `false` keeps the existing RGB output; other
image backends return HTTP 400 when `transparent` is true.

Use the [official RGBA prompt convention](https://github.com/QwenLM/Qwen-Image-2.1#transparent-image-generation-rgba), for example:
`This is an RGBA image with transparency. A red apple. The image has alpha channel and the background is transparent.`
The option preserves generated alpha; it does not rewrite the prompt. It also applies
to `"mode":"edit"`, so an instruction such as "remove the background" returns the
edited image with its alpha channel.
