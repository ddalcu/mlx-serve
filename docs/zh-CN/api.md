[English](../api.md) · [简体中文](api.md)

# HTTP API

一切都跑在一个端口上（默认 `http://localhost:11234`）：OpenAI、Anthropic 和 Ollama 协议，外加原生的媒体生成端点。

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

支持 `messages`、`max_tokens`、`temperature`、`top_p`、`top_k`、`stream`、`stream_options`、`tools`、`response_format`、`repetition_penalty`、`presence_penalty`、`logprobs` / `top_logprobs`、`reasoning_effort` / `enable_thinking` / `reasoning_budget_tokens`，以及按请求覆盖的 `kv_quant` 与 `kv_attn_mode`。对于支持视觉的模型，messages 里可以带 `image_url` 内容块（base64 或 URL）。响应里的 usage 始终带 `prompt_tokens_details.cached_tokens`；如果模型陷入循环导致回复被截断，会在 `finish_reason` 旁边报告 `finish_details: {"type": "repetition_loop"}`。

## POST /v1/messages（Anthropic）

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

兼容 Claude Code（`ANTHROPIC_BASE_URL=http://localhost:11234 claude`）和 Anthropic SDK。支持流式、工具调用和扩展思考。

## POST /v1/responses（OpenAI Responses API）

```bash
curl http://localhost:11234/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "input": "Write a haiku about programming.",
    "stream": true
  }'
```

通过 `previous_response_id` 做有状态串联，完整的流式 SSE 带每个事件的 `sequence_number`，符合 schema 的信封会回显 `tools` / `tool_choice` / `text` / `reasoning` / `usage`。`POST /v1/responses/compact` 返回一段不透明的 base64 历史 blob，它可以在不做任何 LLM 调用的情况下以 `compaction` 输入项的形态回传。同一端点也接受 `Upgrade: websocket` 握手 —— 每个文本帧就是一条 `response.create` JSON 消息，每个 SSE 事件则变成一条出站文本帧。

## Ollama API

`/api/chat`、`/api/generate`、`/api/tags`、`/api/show`、`/api/ps`、`/api/embed`、`/api/pull` 遵循 Ollama 协议（NDJSON 流式、参数为对象的工具调用、`thinking`、`format` JSON schema、`name:latest` 形式的模型名），所以整个 Ollama 客户端生态都能原样对接 mlx-serve —— 把 Raycast、Obsidian、Enchanted、Open WebUI、`ollama-python`/`js` 原本指向 `http://localhost:11434` 的地方改成 `http://localhost:11234` 即可。

## 其它端点

- `GET /` —— 内置 Web 控制台：聊天演练场、Monitor、图像与音频工具、API 参考
- `GET /health` —— 健康检查
- `GET /v1/models` —— 列出已加载的模型，含能力与引擎信息
- `POST /v1/completions` —— 文本补全
- `POST /v1/embeddings` —— 文本嵌入（BERT、EmbeddingGemma，以及 Qwen3-Embedding 这类末位 Token 池化模型；池化方式跟随检查点的 sentence-transformers 元数据，`dimensions` 会截断并重新归一化）
- `POST /v1/images/generations`、`POST /v1/images/edits` —— 图像生成与按指令编辑；edits 端点采用 OpenAI SDK 的 multipart 形态（`client.images.edit`），包括用重复的 `image[]` 传多张参考图
- `POST /v1/audio/speech` —— Qwen3-TTS（`ref_audio` 克隆音色）或 Kokoro（`voice` 从 54 种音色中挑选或混合），输出 WAV
- `POST /v1/audio/music-generations` —— 文生音乐，输出 WAV：ACE-Step（48 kHz 立体声，快）或 MiniMax Music 3（必须提供 `lyrics`，44.1 kHz，歌曲最长六分钟）
- `POST /v1/video/generations` —— LTX-Video 2.3 / 2.5 或 MiniMax-H3；base64 的 `rgb8` 帧加 `pcm_s16le` 音频，封装由你自己完成。LTX 2.5 传 `"decoder": "diffusion"` 可使用更锐利的 diffusion 解码器；较长的 H3 片段通过 `chain_windows` 串联。在 `"stream": true` 时选传 `"preview": true`，会给每个去噪 `progress` 事件附上一张 Latent2RGB JPEG（`preview_frames`、`preview_max_side`）
- `POST /v1/3d/generations` —— Hunyuan3D-2.1，base64 的 GLB
- `POST /v1/load-model`、`POST /v1/unload-model` —— 加载已发现的模型（也可按绝对路径加载），立即释放一个模型；`"default": true` 会让刚加载的模型成为对外服务的默认模型，无需重启
- `POST /v1/models/rescan` —— 收取服务器运行期间下载的模型（应用每次下载完成后都会调用它）
- `POST /tokenize`、`POST /detokenize`、`GET /props` —— 分词器往返与 llama.cpp 风格的服务器属性
- `GET /metrics`、`GET /metrics.json` —— Prometheus + JSON（需要 `--metrics`）
- `GET /v1/responses/{id}`、`DELETE /v1/responses/{id}` —— 获取 / 删除已存储的响应

每个媒体端点都接受 `"stream": true`，以获得以 base64 `complete` 载荷收尾的 SSE 进度。视频流还接受 `"preview": true`，在每个去噪步骤给出一张廉价的 JPEG（默认关闭；cached-velocity 的 H3 步骤不带预览）。媒体 LoRA 在各处都只用一套文法：`lora_paths` + `lora_scales`，最多 8 个，依次叠加。
