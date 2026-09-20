[English](../faq.md) · [简体中文](faq.md)

# 常见问题

## mlx-serve 比 LM Studio 更快吗？

是，但取决于你跑什么。在 v26.8.3 的发布矩阵上（M4 Max、LM Studio 0.4.19+2、完全相同的 MLX 权重文件、**两个引擎都用默认设置**），在 LM Studio 同样具备的四个模型上，mlx-serve 的**解码 +26%（几何均值）**、**预填充 +36%（几何均值）**。

分布形态比均值更重要。在稠密 Gemma 上，原始的单流解码如今打平（E4B 上 −0.5%，31B 上 −0.8%） —— LM Studio 在这块已经追上。差距在预填充，E4B 上 +117%，26B-A4B MoE 上 +35%；还有投机解码：Qwen 3.6 27B 上 mlx-serve 会加载检查点自带的 MTP 头，LM Studio 不会，于是同一个文件上解码快 **+145%**。更早的版本通过为每个模型挑选最佳的投机配置，报出了更大的几何均值；这一次是默认设置对默认设置，也就是你实际拿到的数字。

## mlx-serve 能替代 LM Studio 吗？

对大多数场景而言，能。mlx-serve 运行同样的 MLX 与 GGUF 模型，在同类端口上暴露 OpenAI 兼容 API，并且提供原生菜单栏应用而不是 Electron 应用。它在 API 接口面上比 LM Studio 较新的兼容端点走得更深 —— Anthropic Messages 与 OpenAI Responses 覆盖更完整，另有 WebSocket 传输与响应压缩 —— 并补上了 LM Studio 没有的东西：MCP 工具调用、带 10 个内置工具的 Agent 模式、KV cache 量化、连续批处理，以及面向 DeepSeek V4 Flash 的 [antirez/ds4](https://github.com/antirez/ds4) 引擎。

## mlx-serve 能替代 Ollama 吗？

在 Apple Silicon 上，能 —— mlx-serve **原生支持 Ollama API**（`/api/chat`、`/api/generate`、`/api/tags`、`/api/embed`、`/api/pull`……），所以 Raycast、Obsidian、Enchanted、Open WebUI 以及 `ollama-python`/`js` 都能照旧工作：把原先 `http://localhost:11434` 的位置换成 `http://localhost:11234` 即可直接替换。CLI 工作流也对应（`mlx-serve run gemma4`、`pull`、`list`、`serve`）。底层你同时得到 llama.cpp **和** 原生 MLX，并带有 Ollama 没有发布的 Mac 专属优化（通过 mlx-c 的 Metal kernel、投机解码、共享前缀 KV cache、Gemma 4 交叉注意力草稿模型）。

## 我能在 Mac 上不用 Python 运行 GGUF 模型吗？

能。mlx-serve 把 llama.cpp 的推理库（`libllama`）嵌进同一个已签名、已公证的二进制文件里。把 `--model` 指向任意 `.gguf`，服务器会自动检测格式并路由到正确的引擎 —— 不用 `pip`，不用 venv，不用单独安装 `llama-server`。DeepSeek V4 Flash 的 GGUF 则改走专用的 [antirez/ds4](https://github.com/antirez/ds4) 引擎，同样是嵌入的。

## mlx-serve 能和 Claude Code 一起用吗？

能 —— 原生支持。mlx-serve 实现了 Anthropic 的 `/v1/messages` 端点，包含流式、工具调用与扩展思考。把 Claude Code 指向它，设置 `ANTHROPIC_BASE_URL=http://localhost:11234`。MLX Core 应用提供一键“启动 Claude Code”按钮，替你配好环境变量，`mlx-serve launch claude` 在终端里做同样的事。其他 Agent 也一样：pi、oh-my-pi、OpenCode、Codex、hermes、aider，以及 Zed 这类编辑器。各自的配置见 [integrations.md](integrations.md)。

## 我的多台 Mac 能通过网络共享模型吗？

能 —— 局域网共享，默认关闭。在模型所在的 Mac 上打开共享（设置 ▸ 局域网共享，或 `mlx-serve --serve --lan-share all`），在想用模型的 Mac 上打开发现（`--lan-discover`）。它们通过 Bonjour 互相发现 —— 不用 IP，不用配置 —— 共享模型会以“model · peer”出现在每个模型选择器中，并以 `model@peer` 出现在 `/v1/models` 里，所以即便 Claude Code 指向 `localhost` 也能跑在另一台 Mac 的模型上。聊天以及图像/语音/音乐/视频/3D 生成都适用；模型在主机上按需冷加载；只暴露推理（模型管理、指标与状态页对每台 Mac 保持私有）。

## OpenAI SDK、Continue、Cursor、Open WebUI 呢？

都能用 —— 任何使用 OpenAI 聊天补全或 Anthropic Messages 协议的客户端都能用。mlx-serve 还实现了较新的 OpenAI Responses API（`/v1/responses`），供想要通过 `previous_response_id` 做有状态链式调用的客户端使用，并在同一个端点上提供 WebSocket 传输。

## mlx-serve 能在本地运行 DeepSeek V4 Flash 吗？

能，在 128 GB+ 的 Apple Silicon Mac 上。打开 MLX Core 模型浏览器，选 DeepSeek-V4-Flash，点下载。自 v26.7.12 起，safetensors 版本跑在我们自己的 MLX 引擎上，而不是经由 GGUF：284B 参数、13B 激活、1M 上下文，支持聊天、思考、工具调用与流式，M4 Max 上单流解码约 30 tok/s，用 DSpark（`--dspark`，检查点自带的草稿阶段）大约翻倍。`.gguf` 版本仍路由到内置的 [ds4](https://github.com/antirez/ds4) 引擎。Agent 模式与 MCP 工具在 DSV4 上同样可用。它需要检查点的 0731 发行版；更早的预览版会在加载时被拒绝。

## 支持哪些模型？

原生 MLX 调度支持 Gemma 3/4、DiffusionGemma、Qwen 3 / 3.5 / 3.6 / 3.8 / 3-Next、Meta 的 Muse-Glimmer-30B、inclusionAI Ling 3.0、腾讯混元 3（295B）、Thinking Machines Inkling Small（276B）、poolside Laguna S 2.1、Llama 3.x、Mistral、Nemotron-H、LFM2.5（包括 VL 视觉版本）以及 DeepSeek V4 Flash。其他一切都可以作为 GGUF 通过内置的 llama.cpp 运行 —— Qwen、Llama、Mistral、Gemma、DeepSeek、Phi、Yi 以及 HuggingFace 上可用的数千个模型。完整表格：[models.md](models.md)。

## mlx-serve 能在本地运行腾讯混元 3（295B）吗？

能 —— mlx-serve 运行的最大开源模型。2-bit 混合精度版本（`mlx-serve run hy3`，磁盘上约 105 GB）在 M4 Max 上解码约 26 tok/s、预填充约 235 tok/s，思考、工具调用与全部四个 API 接口面都能工作。推荐用在统一内存 **超过 128 GB** 的 Mac 上；在 128 GB 的 Mac 上它能加载并正确作答，但权重旁边只放得下极小的上下文窗口（约 3K Token） —— 做短对话没问题，跑 Agent 任务就吃紧。检查点自带的多 Token 预测头也受支持（每次请求 `enable_mtp: true`，配合 `--mtp-depth 1` 最佳）。

## 与 MTPLX 在 Qwen MTP 模型上相比如何？

[MTPLX](https://github.com/youssofal/MTPLX) 是一个围绕 Qwen 原生多 Token 预测头构建的专注型 Python 运行时，它在这里立下了标杆。mlx-serve 零配置加载同样的 MTP sidecar 工件（包括 MTPLX 发布的那些），并且在同机、同检查点、同提示词、同采样的正面对比中（v26.8.3 对 MTPLX 2.5.3，两边都用默认设置），解码快 **+10%**，预填充快 **+17%**，首 Token 时延只有三分之一（494 ms 对 1528 ms）。你还一并得到技术栈的其余部分 —— OpenAI/Anthropic/Ollama API、GGUF、Agent 应用 —— 都在一个二进制文件里，且无需 Python。

## 支持工具 / 函数调用吗？

支持，两种 API 接口面都支持。服务器跨架构检测工具调用模式（Hermes XML、Gemma 4 `<|tool_call>`、MiniCPM5 V3 的属性引号式 `<function name="…">` XML、原始 JSON、ChatML），修复常见的 Qwen 3.5/3.6 转义怪癖，并在 SSE 流中输出 OpenAI 风格的 `tool_calls` 增量。MLX Core 应用内置 10 个工具（shell、文件 I/O、搜索、浏览、网页搜索、记忆），并能从精选市场连接 MCP 服务器。

## 它怎么做到这么小 / 这么快？

用 Zig 加直接的 `mlx-c` FFI —— 没有 Python 运行时，没有 Electron，没有 IPC 桥接。发行版二进制文件约 7 MB。启动时的主动预热会触发权重缺页并预编译解码 kernel（首个请求快 3.5 倍）。多轮 Agent 循环跨轮复用 KV，并通过共享前缀缓存跳过系统提示词的重复预填充。分词缓存把长对话上的第二次命中变成一次 memcpy。

## 推理是精确的，还是有量化输出漂移？

对贪心解码（temp=0），mlx-serve 在前约 30-80 个生成的 Token 上与参考实现逐字节相同，之后是 INT4 浮点归约顺序固有的长尾分歧（记录在 `CLAUDE.md` 中）。对 temp > 0，Leviathan 概率比采样器让投机解码在分布上数学精确。等价性由 `tests/test_pld_equivalence.sh`、`test_drafter_equivalence.sh` 与 `test_kv_quant_equivalence.sh` 钉住。

## 我能在不同运行与设置之间得到逐字节相同的贪心输出吗？

能，只要你关掉那些会合法地重排浮点数学的东西。投机解码在一次 forward 中校验多个 Token，而更宽的 forward 会挑到与单 Token forward 不同的 Metal kernel，所以接近平局的 argmax 可能翻转。这不是损坏，也不是种子问题（`seed` 在 temp 0 下不起作用）；这是每个推理服务引擎都具备的批宽特性。逐字节稳定的配方：

- 关闭投机：请求里写 `enable_mtp: false`（或启动时加 `--no-mtp`），再加上 `--no-drafter`；若你开过 `--pld`，也要关掉
- `--kv-quant off` 或 `8`
- 在混合架构上（Qwen 3.5/3.6/3.8、LFM2、Nemotron-H）：`--prefix-cache-entries 0`，因为缓存命中会用不同的块大小重跑递推

如果你的测试门禁要求精确字符串匹配，就用这套配方跑。要让投机路径本身也逐字节相同，就意味着强制每个 kernel 的归约顺序与批宽无关，而这正是快速校验 kernel 换掉的东西。

## 我的数据去哪儿了？

不离开你的机器。一切都在本地运行 —— 没有分析，没有遥测，没有云端调用。HTTP 服务器默认监听你的局域网网络接口（`--host 0.0.0.0`），好让你自己的设备能访问；设成 `--host 127.0.0.1` 可以让它严格本地，或用 `--api-key` 给每个非 localhost 的请求设门槛。打开局域网共享后，发给共享模型的提示词只在你本地网络内传输到托管该模型的那台 Mac。MIT 许可协议下的开源项目。

## 怎么更新？

MLX Core 应用通过检查 GitHub releases 源自动更新。CLI：`brew upgrade --cask mlx-core` 或 `brew upgrade mlx-serve`。
