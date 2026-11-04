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
- `POST /v1/completions` —— 文本补全；`prompt` 可以是字符串或 token id（`[1, 2, 3]`，lm-eval 就这样发送），每个请求一个 prompt。`echo: true` 会在补全前返回 prompt，带 `logprobs` 时还返回每个 prompt token 的 logprob（第一个为 `null`）；不能与 `stream` 同用
- `POST /v1/embeddings` —— 文本嵌入（BERT、EmbeddingGemma，以及 Qwen3-Embedding 这类末位 Token 池化模型；池化方式跟随检查点的 sentence-transformers 元数据，`dimensions` 会截断并重新归一化；EmbeddingGemma 2 需要把任务前缀写在文本里，查询用 `task: search result | query: …`，文档用 `title: none | text: …`）。EmbeddingGemma 2 还能嵌入图片和视频：用 `messages`（一个提示得到一个向量，与 vLLM 的对话式嵌入相同）代替 `input`，其中可含 `{"type": "text"}`、`{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}` 和 `{"type": "video_url", "video_url": {"frames": ["data:image/jpeg;base64,…", …]}}`。各部分按顺序拼接（system 消息在前，与模型自带模板一致），图片保持宽高比并缩放到 280 个软 token，视频帧为 140 个（取自 `processor_config.json`），更长的视频最多保留 32 帧，整个提示必须落在 8192 个 token 的窗口内。不会抓取远程 URL，音频会被明确拒绝。
- `POST /v1/images/generations`、`POST /v1/images/edits` —— 图像生成与按指令编辑；edits 端点采用 OpenAI SDK 的 multipart 形态（`client.images.edit`），包括用重复的 `image[]` 传多张参考图
- `POST /v1/audio/speech` —— Qwen3-TTS（`ref_audio` 克隆音色）或 Kokoro（`voice` 从 54 种音色中挑选或混合），输出 WAV
- `POST /v1/audio/transcriptions` —— 用 Qwen3-ASR（mlx-community 的 `Qwen3-ASR-0.6B`/`1.7B`）把语音转成文字，采用 OpenAI SDK 的 multipart 表单：WAV 格式的 `file`（任意采样率和声道数，最长 20 分钟），可选 `language`（ISO 代码或语言名；省略时由模型识别）、`prompt`（人名等上下文）、`temperature`，`response_format` 为 `json`、`text`、`verbose_json`、`srt` 或 `vtt`，以及 `timestamp_granularities[]`（`word` 会加上 `words`）。时间由 Qwen3-ForcedAligner（`mlx-community/Qwen3-ForcedAligner-0.6B-8bit`，以 `alignment` 能力列出，请求需要时间时才加载）给出：`verbose_json` 的分段止于句末和停顿，每个词带 `start`/`end`，`srt`/`vtt` 就是这些分段；超过 180 秒的音频在最安静处切开，逐段识别并对齐，与官方工具一致。支持中文、粤语、英语、德语、西班牙语、法语、意大利语、葡萄牙语、俄语、韩语和日语；其他语言或磁盘上没有对齐模型时，整段音频为一个分段（没有对齐模型时 `srt`/`vtt` 返回指明原因的 400）。不支持 `stream`；其他音频格式会返回指明原因的 400
- `POST /v1/audio/music-generations` —— 文生音乐，输出 WAV：ACE-Step（48 kHz 立体声，快）MiniMax Music 3（必须提供 `lyrics`，44.1 kHz，歌曲最长六分钟）或 YuE2（必须提供 `lyrics`，48 kHz，先写可编辑的 ABC 乐谱：`cot` 与 `abc`）
- `POST /v1/video/generations` —— LTX-Video 2.3 / 2.5 或 MiniMax-H3；base64 的 `rgb8` 帧加 `pcm_s16le` 音频，封装由你自己完成。LTX 2.5 传 `"decoder": "diffusion"` 可使用更锐利的 diffusion 解码器；较长的 H3 片段通过 `chain_windows` 串联。在 `"stream": true` 时选传 `"preview": true`，会给每个去噪 `progress` 事件附上一张 Latent2RGB JPEG（`preview_frames`、`preview_max_side`）
- `POST /v1/3d/generations` —— Hunyuan3D-2.1，base64 的 GLB
- `POST /v1/load-model`、`POST /v1/unload-model` —— 加载已发现的模型（也可按绝对路径加载），立即释放一个模型；`"default": true` 会让刚加载的模型成为对外服务的默认模型，无需重启
- `POST /v1/models/rescan` —— 收取服务器运行期间下载的模型（应用每次下载完成后都会调用它）
- `POST /tokenize`、`POST /detokenize`、`GET /props` —— 分词器往返与 llama.cpp 风格的服务器属性
- `GET /metrics`、`GET /metrics.json` —— Prometheus + JSON（需要 `--metrics`）
- `GET /v1/responses/{id}`、`DELETE /v1/responses/{id}` —— 获取 / 删除已存储的响应

每个媒体端点都接受 `"stream": true`，以获得以 base64 `complete` 载荷收尾的 SSE 进度。视频流还接受 `"preview": true`，在每个去噪步骤给出一张廉价的 JPEG（默认关闭；cached-velocity 的 H3 步骤不带预览）。媒体 LoRA 在各处都只用一套文法：`lora_paths` + `lora_scales`，最多 8 个，依次叠加。这些路径是服务器磁盘上的文件，所以只有服务器本机上的客户端可以发送；来自其他客户端（局域网共享、`--api-key`）的请求一旦带上它们就返回 403。

### Qwen3-TTS 音色克隆的限制

在 `POST /v1/audio/speech` 上，Qwen3-TTS 使用 `ref_audio`（base64 编码的 WAV）提取说话人嵌入，并不使用参考音频的转录文本进行条件生成：

- `language` 指定朗读语言：取值为检查点 `talker_config.codec_language_id` 中的名称（不区分大小写，如 `english`、`portuguese`、`spanish`）。省略或传 `auto` 时，与以前一样由模型根据文本推断语言。其他取值（包括方言键）返回 HTTP 400。当文本可能被误认为相近语言时（例如巴西葡萄牙语被读成西班牙语发音），请指定该字段。
- 后端不读取 `ref_text`；参考音频的转录文本不参与音色克隆的条件生成。
- 非空的 `ref_audio` 必须是可解码的 WAV，且模型必须具有说话人编码器。无效的 base64/WAV 或缺少编码器时，会在合成开始或发送 SSE 响应头之前返回 HTTP 400。省略 `ref_audio`（或传空字符串）可使用默认音色。
- 若要检查参考音频而不生成语音，可向同一端点发送 `"warm_only": true` 和 `ref_audio`。参考音频缺失或无法解码、或模型没有说话人编码器时，会返回 HTTP 400，而不是静默回退。这只检查参考音频的处理，不验证发音或口音质量。
