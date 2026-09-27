# mlx-serve chat and embeddings

## Chat: `POST /v1/chat/completions` (OpenAI wire)

Any OpenAI SDK works: point it at `<base>/v1` with any non-empty API key.

```js
import OpenAI from "openai";
const client = new OpenAI({ baseURL: `${process.env.MLX_SERVE_URL ?? "http://127.0.0.1:11234"}/v1`, apiKey: "mlx-serve" });
const r = await client.chat.completions.create({
  model: "<id with the chat capability>",
  messages: [{ role: "system", content: "You are Grimble, a grumpy blacksmith. One sentence." },
             { role: "user", content: "Can you fix my sword?" }],
  max_tokens: 200,
});
console.log(r.choices[0].message.content);
```

```python
from openai import OpenAI
import os
client = OpenAI(base_url=os.environ.get("MLX_SERVE_URL", "http://127.0.0.1:11234") + "/v1", api_key="mlx-serve")
```

Supported: `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`,
`stop`, `stream` (+ `stream_options.include_usage`), `tools` / `tool_choice`,
`response_format` (`json_object` or `json_schema`), `logprobs`,
`presence_penalty`, `repetition_penalty`. Images go in as `image_url` content
parts (base64 data URLs) on models with the `vision` capability.

- Reasoning models (capability `reasoning`) think before answering. The thinking
  comes back separately in `message.reasoning_content` (streaming:
  `delta.reasoning_content`), never inside `content`. For snappy NPC lines send
  `"reasoning_effort": "none"` or `"enable_thinking": false`; for hard
  planning leave it on or pass `"reasoning_effort": "high"`.
- Structured output for game logic: `"response_format": {"type": "json_schema",
  "json_schema": {"name": "npc_reply", "schema": {...}}}`. The server constrains
  decoding to the schema, so the JSON always parses.
- Context: `context_length` from `/v1/models`. A prompt that does not fit is a
  400 naming both token counts; trim history yourself.
- Multiple concurrent requests are fine; the server batches them.
- Latency: first token after the prompt is processed; repeated prefixes (same
  system prompt + history) are cached, so keep the stable part of the prompt
  first and append new turns at the end.

`POST /v1/messages` (Anthropic SDK) and `POST /v1/responses` (OpenAI Responses)
serve the same models. Ollama clients work against `/api/*`.

## Embeddings: `POST /v1/embeddings`

For a model with the `embeddings` capability. OpenAI shape:

```json
{"model": "<id>", "input": ["iron sword", "rusty dagger"]}
```

→ `{"data": [{"embedding": [...], "index": 0}, ...]}`. Compare vectors with cosine
similarity. `"dimensions": N` truncates (and renormalizes).
Use for semantic search over lore, item lookup, or matching player text to known
intents; for typed yes/no/choice questions prefer `/v1/decisions`.
