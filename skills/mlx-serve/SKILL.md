---
name: mlx-serve
description: Hook an app, game or script up to the local mlx-serve server for LLM chat, embeddings, image, speech, music, video and 3D generation, and Laya/Kev typed decisions. Use when code should call mlx-serve.
---

# mlx-serve

mlx-serve runs MLX models on this Mac behind one HTTP port: OpenAI, Anthropic and
Ollama compatible chat, plus native endpoints for images, speech, music, video, 3D
and decisions. Everything below is plain HTTP + JSON, so any language works.

## Connect

- Base URL: `$MLX_SERVE_URL` when set, else `http://127.0.0.1:11234`. Make it a
  config value or env var in the code you write, never a hardcoded LAN IP.
- Auth: none from this Mac. A server started with `--api-key` wants
  `Authorization: Bearer <key>` from other machines; a 401 means that.
- CORS is open (`Access-Control-Allow-Origin: *`): a browser game can call it
  directly.
- Up? `curl -s "$MLX_SERVE_URL/health"`

## Pick models from `/v1/models`, never guess ids

```sh
curl -s "${MLX_SERVE_URL:-http://127.0.0.1:11234}/v1/models" \
  | jq -r '.data[] | "\(.id)\t\(.capabilities | join(","))\t\(.state)"'
```

Each row has `id` (like `org/name`), `capabilities`, `state` (`ready`,
`unloaded`, `remote`) and `context_length`. Choose by capability:

| capability | endpoint | details |
|---|---|---|
| `chat` | `POST /v1/chat/completions` (also `/v1/messages`, `/v1/responses`) | chat.md |
| `embeddings` | `POST /v1/embeddings` | chat.md |
| `image` | `POST /v1/images/generations`, `POST /v1/images/edits` | media.md |
| `audio` without `music` | `POST /v1/audio/speech` (TTS) | media.md |
| `music` | `POST /v1/audio/music-generations` | media.md |
| `video` | `POST /v1/video/generations` | media.md |
| `3d` | `POST /v1/3d/generations` | media.md |
| `decisions` | `POST /v1/decisions` | decisions.md |

Read the linked file (next to this one) before writing client code for that
endpoint. If no model has the capability the user needs, say so and tell them to
download one in the MLX Core app (Models). Do not invent an id.

## Loading and memory

- Any request names its model with `"model": "<id>"`. An unloaded model loads on
  the first request, which can take seconds to minutes.
  `POST /v1/load-model {"model": "<id>"}` pre-warms one;
  `POST /v1/unload-model {"model": "<id>"}` frees it.
- All models share one GPU memory budget. A 503 `out_of_memory` means unload
  something first (typically the media model when you are done with it).
- Generation runs on one inference thread: a 3-minute video render stalls chat
  replies for 3 minutes. Decisions and embeddings are milliseconds.

## Rules for apps and games

- Generate media AHEAD of time or in the background, never on the frame or
  input path. Cache results on disk keyed by model + prompt + seed + size, and
  ship the cache (or regenerate lazily) instead of calling on every run.
- Rough cost on Apple Silicon: image 5-60 s, speech about real time or faster,
  music 30 s to minutes, 3D 1-5 min, video minutes. Set client timeouts to
  match (10+ minutes for video), and keep the game playable while waiting.
- Send `"seed"` for reproducible assets.
- Long jobs: send `"stream": true` and read the SSE `progress` events to drive a
  progress bar. See media.md.
- Errors are JSON: `{"error": {"message": "..."}}` with a 4xx/5xx status. The
  message names the bad field. Show it; never retry a 400 unchanged.
- Keep the server's own URL and model ids in one config module so the user can
  swap models without touching game code.

## Smoke test

```sh
BASE="${MLX_SERVE_URL:-http://127.0.0.1:11234}"
IMG=$(curl -s "$BASE/v1/models" | jq -r '[.data[] | select(.capabilities | index("image"))][0].id')
curl -s "$BASE/v1/images/generations" -H 'content-type: application/json' \
  -d "{\"model\":\"$IMG\",\"prompt\":\"pixel art treasure chest on a plain white background\",\"size\":\"512x512\",\"seed\":7}" \
  | jq -r '.data[0].b64_json' | base64 -d > chest.png
```
