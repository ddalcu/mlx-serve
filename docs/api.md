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

Supports `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, `stream_options`, `tools`, `response_format`, `repetition_penalty`, `presence_penalty`, `logprobs` / `top_logprobs`, `reasoning_effort` / `enable_thinking` / `reasoning_budget_tokens`, plus per-request `kv_quant` and `kv_attn_mode` overrides. Messages can include `image_url` content blocks (base64 or URL) for vision-capable models. Usage always carries `prompt_tokens_details.cached_tokens`, and a reply cut short because the model went in circles reports `finish_details: {"type": "repetition_loop"}` beside `finish_reason`.

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

Compatible with Claude Code (`ANTHROPIC_BASE_URL=http://localhost:11234 claude`) and Anthropic SDKs. Supports streaming, tool calling, and extended thinking.

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

- `GET /` — built-in web console: chat playground, Monitor, image and audio tools, API reference
- `GET /health` — health check
- `GET /v1/models` — list loaded models with capabilities + engine info
- `POST /v1/completions` — text completions
- `POST /v1/embeddings` — text embeddings (BERT, EmbeddingGemma, and last-token pooling models like Qwen3-Embedding; pooling follows the checkpoint's sentence-transformers metadata, `dimensions` truncates and renormalizes)
- `POST /v1/images/generations`, `POST /v1/images/edits` — image generation and instruction edits; the edits endpoint speaks the OpenAI SDK's multipart shape (`client.images.edit`), including repeated `image[]` for multi-reference
- `POST /v1/audio/speech` — Qwen3-TTS (`ref_audio` clones a voice) or Kokoro (`voice` picks or blends one of 54), WAV out
- `POST /v1/audio/music-generations` — text-to-music, WAV out: ACE-Step (48 kHz stereo, fast) or MiniMax Music 3 (`lyrics` required, 44.1 kHz, songs up to six minutes)
- `POST /v1/video/generations` — LTX-Video 2.3 / 2.5 or MiniMax-H3; base64 `rgb8` frames plus `pcm_s16le` audio, mux on your side. LTX 2.5 takes `"decoder": "diffusion"` for its sharper diffusion decoder; long H3 clips chain via `chain_windows`. Opt-in `"preview": true` on `"stream": true` attaches a Latent2RGB JPEG to each denoise `progress` event (`preview_frames`, `preview_max_side`)
- `POST /v1/3d/generations` — Hunyuan3D-2.1, base64 GLB
- `POST /v1/load-model`, `POST /v1/unload-model` — load a discovered model (or one by absolute path), free one now; `"default": true` makes the loaded model the serving default without a restart
- `POST /v1/models/rescan` — pick up models downloaded while the server runs (the app calls it after every download)
- `POST /tokenize`, `POST /detokenize`, `GET /props` — tokenizer round-trip and llama.cpp-style server props
- `GET /metrics`, `GET /metrics.json` — Prometheus + JSON (needs `--metrics`)
- `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}` — fetch / delete stored responses

Every media endpoint takes `"stream": true` for SSE progress ending in a base64 `complete` payload. Video streams also accept `"preview": true` for a cheap JPEG on each denoise step (off by default; cached-velocity H3 steps stay preview-less). Media LoRAs use one grammar everywhere: `lora_paths` + `lora_scales`, up to 8, stacked.
