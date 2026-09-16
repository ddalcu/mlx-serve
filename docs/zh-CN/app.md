[English](../app.md) · [简体中文](app.md)

# MLX Core（macOS 应用）

菜单栏应用，用完整的 UI 包住服务器。[下载最新版本](https://github.com/ddalcu/mlx-serve/releases/latest)或执行 `brew install --cask mlx-core`。自 v26.8.4 起，它是一个窗口，含侧边栏、内容列与详情栏：模型、任务、设置以及媒体生成器都是聊天窗口的模式，而不再是独立窗口。

- **模型浏览器** —— 从 HuggingFace 下载，支持断点续传的多连接传输（每个文件最多 16 个连接），自动发现 LM Studio 已有的模型文件夹（`~/.lmstudio/settings.json`）和你的 Hugging Face 缓存（`HF_HOME` / `HF_HUB_CACHE` 指向哪里就找哪里），这样磁盘上已有的内容不会再下载一遍，GGUF 行会显示一个 RAM 估算的最小–最大区间。多版本仓库让你自己挑量化版本，**设置 ▸ 模型文件夹**决定下载落到哪里。
- **无需重启即可切换模型** —— 选中一个聊天模型会把它加载进正在运行的服务器并设为默认（通过 API 调用 `POST /v1/load-model`，带上 `"default": true`）。
- **聊天界面** —— 多会话聊天，带 markdown 渲染。可以随文本一起放入 PDF（由 PDFKit 提取）或图像。
- **Agent 模式** —— 10 个内置工具（shell、cwd、readFile、writeFile、editFile、searchFiles、listFiles、browse、webSearch、saveMemory），带自动工具调用循环和逐工具审批对话框（**允许** / **拒绝** / **本次会话始终允许**）。
- **Agents** —— 具名助手，各自拥有性格、音色、模型、工具、工作区和唤醒词，以及各自固定的采样参数（temperature、top-p、top-k、惩罚项、推理预算）。应用会替你写好人设提示词，而且每一种发起对话的方式都能以某个 Agent 运行：聊天标签页、免手操作语音、定时任务、Telegram、快速启动器。
- **MCP 客户端** —— 精选的 stdio + HTTP MCP 服务器市场（GitHub、Azure DevOps、DBHub、Docker、Kubernetes、Playwright、Slack、Notion、Filesystem、Shell），另外你还可以通过 `~/.mlx-serve/mcp.json` 添加自己的服务器。
- **Agent 沙盒** —— 打开一个开关，此后每条 Agent shell 命令都在基于 Apple Virtualization 框架构建的隔离 Linux VM 中运行：不到一秒即可启动，客户机服务器实时映射到 `localhost`（客户机 8080 端口上的 Express 应用就是你 Mac 上的 `http://localhost:8080`），命令隔离运行时工具栏会显示一枚绿色盾牌。这是一个正常的 Linux，`apt-get install` 可用，最多惰性提交 4 GB 内存。放手让 Agent 折腾 —— 你的 Mac 毫发无损。
- **⌃Space 快速启动器** —— 一个 Spotlight 风格的提示词面板，浮在任何应用之上：按 ⌃Space，提问，答案就从你的本地模型流式返回。追问会保留上下文；⌘↩ 把对话交接给完整的聊天窗口。
- **免手操作语音模式** —— 说一句“Hey Loki”然后直接开口就行：端上语音识别（音频从不离开这台 Mac）、可打断的语音回复，以及语音驱动的 Agent 工具 —— 全部来自菜单栏，无需打开窗口。回复会用你挑选的 54 种内置 Kokoro 音色说话（完全本地，约 17× 实时，一次 345 MB 的下载，可混合），或者用 Qwen3-TTS 克隆你自己的音色。
- **局域网共享** —— 把你选定的模型分享给家里的其他 Mac，也使用它们的模型：对端会自动出现，共享模型在菜单栏面板和每个生成面板中都显示为“模型 · 对端”，请求会流式转发到托管权重的 Mac —— 聊天、图像、语音、音乐、视频、3D 都一样，模型在主机上按需加载。设置里有逐模型的共享复选框；发往共享模型的提示词会在托管 Mac 上运行（也对该 Mac 可见）。
- **Telegram 桥接** —— 从手机上给你的本地模型发消息：没有公网 URL、没有端口转发、没有云端中继。Agent 工具和定时任务都能远程使用；机器人会锁定到第一个给它发消息的聊天。
- **定时任务** —— 用大白话给 Agent 一个目标和一份时间表（“工作日早上 8 点，检查我关注的网站并写一份简报”），它就会无人值守地运行，并保存记录。
- **文档文件夹 RAG** —— 挂载一个装着各种文件的文件夹，然后就它们提问；GPU 批处理嵌入在约 7 s 内索引约 500 个文件，一切都在内存中，什么都不离开这台 Mac。
- **可编辑的系统提示词 + 持久记忆** —— `~/.mlx-serve/system-prompt.md` 和 `~/.mlx-serve/memory.md`。
- **基于提示词的技能** —— 把带 YAML frontmatter 的 `.md` 文件放进 `~/.mlx-serve/skills/`，即可教给 Agent 由关键词触发的自定义能力；或者在聊天框里输入 `/` 挑一个技能，在任何聊天中按名字运行它，无论是否处于 Agent 模式。
- **引擎感知的设置窗口**（Cmd+,）—— 每一项服务器启动参数与每个请求的默认值，分区只显示与你已加载引擎相关的旋钮（MLX vs GGUF vs ds4）。
- **服务器管理** —— 启动 / 停止、实时日志缓冲、参数变更后提示重启的横幅。
- **图像 / 视频 / 音乐 / 语音 / 3D 生成** —— FLUX.2、Krea-2、Mage-Flow、LTX-Video 2.3 / 2.5、MiniMax-H3、ACE-Step、MiniMax Music 3、Qwen3-TTS、Kokoro 和 Hunyuan3D，全部通过 mlx-serve zig 服务器原生运行。

## 图像 / 视频 / 音乐 / 语音 / 3D 生成

一个服务器，五种模态 —— **Image**、**Video**、**Audio**（语音 + 音乐）和 **3D** 创建面板。它们在 MLX 上原生运行 [FLUX.2](https://huggingface.co/black-forest-labs) / Krea-2 / Microsoft Mage-Flow、[LTX-Video 2.3 与 2.5](https://github.com/dgrauet/ltx-2-mlx) / [MiniMax-H3](https://huggingface.co/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit)、[ACE-Step 1.5](https://huggingface.co/ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-8bit) / MiniMax Music 3、[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) / [Kokoro-82M](https://huggingface.co/ddalcu/Kokoro-82M-MLX-Serve)，以及 [Hunyuan3D-2.1](https://huggingface.co/ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit)。点开一个面板，按 **Download**，生成。把文件拖到任何面板上，它会落到正确的槽位，参考图列表也包括在内。每个面板都会在会话之间记住你上次用的模型、质量、分辨率、步数和 seed。

面板还会列出你自己添加的检查点：你的模型文件夹里，凡是系列能被服务器运行的，都会带着该系列对应的控件出现在 **On This Mac** 下，而模型浏览器会提供这些系列的社区包，在 Download 按钮出现之前会先做布局检查。

你也可以**直接从聊天中生成媒体**：要一张图像、一句台词、一曲音乐或一段短片，它就会在对话里内联渲染并显示进度条，使用你为该模态保存的设置。附上一张照片并说“make it winter”，编辑结果就在同一个会话里返回。双击聊天中的任何图像，可在 Preview 中以完整尺寸打开。

而它远不止文生 X 这么简单：

- **按指令编辑照片** —— 附上一张图片，输入 *“make the hair blue”* 或 *“remove the monitor in the background”*，FLUX.2-klein 或 Mage-Flow Edit 就会改动它，同时保持主体、姿态和场景不变（上下文内参考条件化 —— 实测结构相关性 0.97）。Mage-Flow Edit 还能组合多个参考（“把图像 2 中的物体放进图像 1”），并能仅凭普通指令完成控制图、图像修复与重新打光。源图保持自己的宽高比，绝不压缩变形。从 OpenAI SDK 同样可用：`client.images.edit(image=…, prompt=…)` 打到 `POST /v1/images/edits`。
- **图像到图像变体** —— 每个图像模型（包括 Krea-2）都接受一张源图加一个强度滑块，从轻微改写直到完全重构。
- **让照片动起来** —— 把一张图片放进 Video 面板的 First-frame 槽位，LTX 就从它出发向前动画，正好从你的图像开始。
- **会说话的角色** —— 在视频提示词里用引号写出台词、附上一段真实的语音或音乐片段，或者为 Qwen3-TTS 输入一句台词让它配音 —— 视频会*围绕*这条音轨生成，表演与之同步，原始音频（不是重新合成）会进入 mp4。
- **用几秒音频克隆音色** —— 在设置 ▸ 语音中录制或挑选一段片段，Qwen3-TTS 就用那个音色说话 —— 在 AudioGen 面板里、在免手操作语音模式里，处处如此。
- **谱写完整的音乐曲目** —— ACE-Step 1.5 把一段风格提示词（以及可选歌词）变成 48 kHz 立体声曲目：一首 30 秒的歌约 4 秒渲染完成。
- **把照片变成 3D 模型** —— Hunyuan3D-2.1 把一张图像转换成水密的 GLB 网格，可选带完整 PBR 纹理 —— 直接放进游戏引擎或切片软件。
- **自带音轨的视频** —— MiniMax-H3（Hailuo 3.0）在一趟里同时对片段和立体声音轨去噪，因此声音是与视频一起生成的，而不是事后配音。先描述场景，再在 `overall_soundscape:` 后面写你想听到的声音。REF2VA 版本围绕你附上的图片、片段或音频构建片段（提示词中的 `<Picture 1>`、`<Video 1>`、`<Audio 1>`）；FL2VA 版本接受首 / 尾关键帧，并把多个窗口串成更长的片段。**Turbo** 用 4 步而不是 30 步渲染。两种方式都慢：1344×768、124 帧在 M4 Max 上约 50 分钟。
- **风格 LoRA** —— 在运行时挂载 diffusers、kohya 或 PEFT 的 `.safetensors` 适配器，为 FLUX、Krea、Mage-Flow、LTX 或 MiniMax-H3 的生成换风格。一次最多 8 个，相加而不是合并，因此不会有任何东西被重新量化，基础权重零质量损失。每个适配器按它自己文件里声明的强度运行。

### 模型

| 功能 | 默认 | 其它选项 | 大致 RAM |
|---|---|---|---|
| 图像 | FLUX.2-klein 4B 4-bit（mflux，预量化约 5 GB） | FLUX.2-klein 9B（10 GB）、Krea-2-Turbo、Mage-Flow Turbo / Edit 8-bit（8.5 / 9.1 GB） | 8 / 12 / 16 GB |
| 视频 | LTX-Video 2.5 4-bit（36 GB，自带文本编码器） | LTX-Video 2.5 8-bit（59 GB，更锐利 + 扩散解码器）、LTX-Video 2.3 Q4（约 50 GB）、MiniMax-H3（Hailuo 3.0）4-bit / 8-bit，一趟同时生成视频**和**匹配的音轨 | LTX 24 GB RAM；H3 26 GB（40 GB）或 44 GB（69 GB） |
| 语音 | Qwen3-TTS 1.7b（语音克隆） | Qwen3-TTS 0.6b、Kokoro-82M（54 种音色，约 345 MB） | 8 GB RAM，首次运行约 3.5 GB 下载 |
| 音乐 | ACE-Step 1.5 XL Turbo 8-bit（快，8 步） | MiniMax Music 3 8-bit（演唱你的歌词，歌曲最长 6 min，人声最强） | ACE 8 GB RAM，约 6.2 GB 下载；Music 3 约 20 GB RAM，13.6 GB 下载 |
| 3D | Hunyuan3D-2.1 8-bit（形状 + PBR 纹理） | — | 16 GB RAM |

> 41 GB 的 LTX 2.3 快照**同时**带有两种 transformer 变体（1 阶段蒸馏版 + 2 阶段 dev 版，每个约 11 GB）外加一个 7.6 GB 的蒸馏 LoRA，因此你可以在 Fast/Good/Quality/Super 之间离线切换，无需重新下载。

> LTX-Video 2.5 自带文本编码器，所以首次使用时没有额外的 8 GB 下载。8-bit 包保住了 4-bit 包丢掉的细节，并新增一个 **Diffusion decoder** 开关（Lightricks 自家发布的片段所用、通过 API 指定 `"decoder": "diffusion"` 的那个解码器），让纹理和边缘更锐利。默认画布和帧阶梯按 Mac 定制；两阶段档位按所选尺寸的一半去噪再放大。

> MiniMax Music 3 需要歌词；`[verse]` 和 `[chorus]` 这类结构标签各占一行。ACE-Step 的速度、调性、拍号和语言控件在它上面不存在，所以把这些信息写进描述文本里。当你向聊天要一首歌时，内置的 **music3** 技能会按该模型训练时所用的描述格式来写。

输出进入 `~/.mlx-serve/generations/`，按模态、按日期分文件夹。

> 如果没有足够的空闲 RAM，应用不会让你启动生成。如果 mlx-serve 服务器正在运行并争抢内存，会先提示你停止它。
