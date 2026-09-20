[English](../cli.md) · [简体中文](cli.md)

# CLI 与服务器参数

## Ollama 风格的命令

```bash
mlx-serve run gemma4        # 下载 Gemma 4 E4B（4-bit）、对外提供服务，并直接在终端里聊天
mlx-serve pull qwen3.6:27b  # 只下载（断点续传，直接来自 Hugging Face）
mlx-serve list              # 磁盘上有什么
mlx-serve serve             # 对外提供所有已拉取模型的服务 —— 模型按名称按需加载
mlx-serve launch claude     # 配置并启动一个针对本地服务器的编码 Agent CLI
```

`launch` 支持 claude、pi、omp、opencode、opencode2、codex、hermes 与 aider。它会读取运行中服务器的模型列表与真实上下文窗口，把 Agent 配置写进专用的 `~/.mlx-serve/<agent>/` 文件夹（绝不碰你真实的 Agent 配置），然后启动该 Agent。如果服务器没在运行，它会先启动 MLX Core 应用。`--model <id>` 选择模型，`--print` 只显示启动脚本而不真正运行，`--` 之后的任何内容都会传给 Agent（`mlx-serve launch codex -- resume`）。`opencode2` 还会安装 mlx-serve 监控插件，这要求服务器以 `--metrics` 启动。每种 Agent 的完整细节见 [integrations.md](integrations.md)。

短名称、`org/repo` 形式的 HuggingFace id 以及 `name:tag` 都可用。模型会落进一个共享的 `~/.mlx-serve/models` 存储目录，MLX Core 应用用的也是它。

## 直接驱动服务器

脚本、launchd 或无图形界面的 Mac 需要的方式：

```bash
# 单个模型，在进程的整个生命周期内固定
mlx-serve --model ~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit --serve --port 11234

# 整个文件夹，每个模型按名称按需加载
# （`mlx-serve serve` 和应用所做的正是这件事）
mlx-serve --serve --model-dir ~/.mlx-serve/models

# GGUF 使用完全相同的参数；服务器会识别它并路由到内置的 llama.cpp
mlx-serve --model ~/models/Qwen3.5-4B-Q4_K_M.gguf --serve
```

`mlx-serve --help` 列出全部参数。默认值是 `--host 0.0.0.0 --port 11234`。

## 一次性运行，无需服务器

```bash
mlx-serve --model /path/to/model --prompt "What is 2+2?"
```

## CLI 选项

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model PATH` | 必填 | 模型目录或 `.gguf` 文件的路径 |
| `--serve` | 关闭 | 启动 HTTP 服务器 |
| `--host ADDR` | `0.0.0.0` | 绑定地址（所有网络接口 —— 只想本机访问就设为 `127.0.0.1`） |
| `--port N` | `11234` | HTTP 服务器端口 |
| `--prompt TEXT` | `"Hello"` | 交互模式的提示词 |
| `--max-tokens N` | `100` | 最多生成多少 Token |
| `--temp F` | `0.0` | 采样温度（0 = 贪心） |
| `--ctx-size N` | 自动 | 上下文窗口大小（自动 = 依据 GPU 内存计算） |
| `--embedding-max-length N` | 自动 | `/v1/embeddings` 每个输入的 Token 上限（自动 = 模型声明的窗口；超限输入返回 400，绝不静默截断） |
| `--timeout N` | `300` | 停滞超时 —— *没有新 Token* 的秒数（持续有产出的请求永不超时） |
| `--reasoning-budget N` | `-1` | 思考 Token 预算（`-1` = 不限，`0` = 不思考） |
| `--no-vision` | 关闭 | 即使模型支持也禁用视觉编码器 |
| `--pld` / `--no-pld` | 开启 | Prompt Lookup Decoding（与模型无关的投机解码） |
| `--pld-draft-len N` | `5` | PLD 每一步的最大草稿 Token 数 |
| `--pld-key-len N` | `3` | PLD 的 N-gram 匹配键长度 |
| `--drafter DIR` | 无 | 投机解码的草稿模型检查点：Gemma 4 assistant 或 DFlash 配套草稿模型。自带 `drafter/` 子目录的模型（Muse-Glimmer 构建）会自动加载自己的 |
| `--no-drafter` | 关闭 | 永不加载草稿模型，包括检查点内自带的那个 |
| `--draft-block-size N` | 自动 | 草稿模型每轮的草稿数（自动取这台 Mac 的验证路径可用的值） |
| `--no-mtp` / `--mtp` | 有 sidecar 时开启 | 禁用 / 强制原生 MTP 头（MoE 主干默认关闭） |
| `--mtp-depth N` | `3` | 每轮 MTP 最多草拟的 Token 数（自适应控制器在 `[1, N]` 内调节） |
| `--mtp-history-window N` | `0`（完整） | 超过 16K Token 的提示词只为最后 N 个 Token 构建 MTP 头历史（开窗会在原版 Qwen 头上损失接受率） |
| `--dspark` | 关闭 | DeepSeek V4 自有的块并行草稿阶段（在模型之上额外约 11 GB） |
| `--ssd-streaming` | 关闭 | 仅 ds4 / DeepSeek-V4-Flash GGUF：从 SSD 流式读取专家权重，而不是把整个模型放在内存里 |
| `--prefill-chunk N` | `8192` | 每个预填充分块最多前向的 Token 数（还会按模型进一步自动设上限）；调低可削减预填充峰值内存 |
| `--no-decode-attn-quant` | 开启 | 禁用仅解码阶段对稠密 bf16 注意力权重的重新量化（即 “Fast decode for bf16-attention models” 开关） |
| `--kv-quant {off,4,8}` | 关闭 | KV cache 量化方案（MLX 路径） |
| `--kv-attn-mode {auto,dense,fused}` | 自动 | 量化 KV 的解码读取路径：`fused` 原地读取打包后的 cache，`auto` 从 8K 提示词 Token 起启用（仅在 `--kv-quant 4/8` 时有效；单请求的 `kv_attn_mode` 可覆盖） |
| `--llama-kv-quant {off,q8,q4}` | 关闭 | GGUF 的 KV cache 量化（llama.cpp 路径） |
| `--llama-cache-entries N` | `4` | llama.cpp 的多会话 LRU（预热多文档 Agent） |
| `--tokenize-cache-entries N` | `4` | chat template + 分词缓存大小 |
| `--max-concurrent N` | `1` | 连续批处理解码的并行度 |
| `--prefix-cache-entries N` | 自动 | 共享前缀 KV cache 的条目上限 |
| `--prefix-cache-mem N{KB,MB,GB}` | `2 GB` | 共享前缀 KV cache 的内存上限 |
| `--prefix-cache-disk N{MB,GB}` | 关闭 | SSD 层级：前缀能跨重启存活（11K Token 的重启 TTFT 从 5.9 s 降到 0.7 s） |
| `--metrics` | 关闭 | Prometheus `/metrics` + `/` 上的实时仪表盘面板 |
| `--api-key KEY` | 无 | 要求非 localhost 请求携带令牌（localhost 保持开放） |
| `--lan-share <all\|id,...>` | 关闭 | 通过 Bonjour 把列出的模型（或全部）共享到局域网 —— 只暴露推理，模型管理仍留在主机本地 |
| `--lan-discover` | 关闭 | 发现其它 Mac 共享的模型：它们在 `/v1/models` 中以 `model@peer` 出现，请求会代理到那台 Mac |
| `--lan-name NAME` | 主机名 | 其它 Mac 看到的 Bonjour 名称 |
| `--model-dir PATH` | 无 | 发现并对外提供某个文件夹里的所有模型（LRU 常驻集）。可重复使用 —— 多个文件夹合并，先出现者优先 |
| `--max-resident-mem N{MB,GB}` | 自动 | 已加载模型的内存总上限；它决定一个模型能否加载（auto = MLX wired limit 的 80%，`0` 表示禁用） |
| `--max-resident-models N` | `3` | 同时保持加载的模型数量（按 LRU 淘汰） |
| `--idle-evict-secs N` | 关闭 | 闲置超过这么多秒后卸载无人使用的模型 |
| `--no-warmup-eager` | 关闭 | 跳过启动时的立即预热（用于基准测试 / 最小占用部署） |
| `--skip-mem-preflight` | 关闭 | 加载时跳过空闲内存预检（上面的上限先检查，并且仍然适用） |
| `--no-tool-autocorrect` | 关闭 | 关闭对模型输出的工具参数所做的 schema 驱动修复 |
| `--log-level` | `info` | 日志级别（error、warn、info、debug） |
| `--log-file PATH` | `~/.mlx-serve/logs/` | 服务器日志的落盘位置 |
