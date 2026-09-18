[English](../integrations.md) · [简体中文](integrations.md)

# 集成

mlx-serve 支持这些标准 API（OpenAI、Anthropic、Ollama、OpenAI Responses），所以大多数编码 Agent 和编辑器只要一个 base URL 加一个模型 id 就能接上。本页先讲零配置的启动器，再按工具逐个讲手动配置。

## 连接信息

| 项目 | 值 |
|---|---|
| OpenAI 兼容 | `http://127.0.0.1:11234/v1`（chat/completions、embeddings、images、audio） |
| OpenAI Responses | `http://127.0.0.1:11234/v1/responses`（Codex 用的就是这个） |
| Anthropic 兼容 | `http://127.0.0.1:11234`（设置 `ANTHROPIC_BASE_URL`；对外提供 `/v1/messages`） |
| Ollama 兼容 | `http://127.0.0.1:11234/api/*` |
| API key | 任意占位值（例如 `mlx-serve`）。回环连接即使设置了 `--api-key` 也豁免；非本机客户端需要真实 key |
| 模型 id | `GET /v1/models`。每一行都会在 `meta.context_length` 里公布自己的上下文窗口，并作为顶层 `context_length` / `max_model_len` 公布，供基于发现的客户端使用 |

如果你改过主机或端口，请相应替换（默认端口是 11234）。

任何手动配置里都有一件事要做对：上下文窗口。服务器在加载模型时确定真实的窗口大小（受内存约束，通常远低于模型上限），并在 `/v1/models` 里公布。在 Agent 配置里硬编码一个更大的数字会导致溢出；硬编码一个更小的数字则浪费窗口。下面的启动器会从服务器读取该值，所以你永远不用手填。

## 零配置：启动器

有两种方式可以跳过下面所有内容：

- **MLX Core 应用**：菜单栏面板里的 **Code** 按钮会检测你 Mac 上安装的 CLI，并以对接运行中服务器的预设配置启动它们（App Store 版本改为显示可复制的命令，因为它不允许启动其它应用）。
- **`mlx-serve launch <agent>`**：在终端里做同一件事，Ollama 风格：

```bash
mlx-serve launch claude              # 可选：claude、pi、omp、opencode、opencode2、codex、hermes、aider
mlx-serve launch codex --model Qwen3.5-27B-MLX-4bit
mlx-serve launch codex -- resume     # -- 之后的参数全部传给 agent
```

如果没有服务器在运行，`launch` 会启动 MLX Core 应用并等待；没装该应用时它会提示你先运行 `mlx-serve serve`。可用参数：`--model`、`--url`、`--port`、`--print`（只写出配置并打印启动脚本，而不真正运行）、`--no-start`。

两个启动器都会把配置写进专用的 `~/.mlx-serve/<agent>/` 目录，绝不碰你真正的 Agent 配置（`~/.claude`、`~/.pi`、`~/.omp`、`~/.codex`、`~/.hermes` 还是你的）。

## 编码 Agent（手动配置）

把 `MODEL_ID` 换成 `GET /v1/models` 里的某个 id，把 `CTX` 换成该行的 `meta.context_length`。

### Claude Code

只用环境变量，不需要配置文件。mlx-serve 原生提供 Anthropic Messages API。

```bash
export ANTHROPIC_BASE_URL='http://127.0.0.1:11234'
export ANTHROPIC_API_KEY=
export ANTHROPIC_AUTH_TOKEN=mlx-serve
export ANTHROPIC_DEFAULT_OPUS_MODEL=MODEL_ID
export ANTHROPIC_DEFAULT_SONNET_MODEL=MODEL_ID
export ANTHROPIC_DEFAULT_HAIKU_MODEL=MODEL_ID
claude --model MODEL_ID
```

### pi

pi 需要一个 `models.json` 来指定提供商；`PI_CODING_AGENT_DIR` 可以改变配置目录的位置，这样你真正的 `~/.pi/agent` 就不会被动到。

```bash
mkdir -p ~/.mlx-serve/pi
cat > ~/.mlx-serve/pi/models.json <<'EOF'
{
  "providers": {
    "mlx": {
      "baseUrl": "http://127.0.0.1:11234/v1",
      "api": "openai-completions",
      "apiKey": "mlx-serve",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "maxTokensField": "max_tokens",
        "thinkingFormat": "qwen"
      },
      "models": [
        {"id": "MODEL_ID", "name": "MODEL_ID (mlx-serve)", "input": ["text"],
         "contextWindow": CTX, "maxTokens": 8192, "reasoning": true}
      ]
    }
  }
}
EOF
export PI_CODING_AGENT_DIR="$HOME/.mlx-serve/pi"
pi --provider mlx --model MODEL_ID
```

### oh-my-pi（omp）

思路与 pi 相同，只是文件换成 `models.yml`（YAML）。注意：omp 读取配置目录时用的仍然是 `PI_CODING_AGENT_DIR` 这个拼写。

```bash
mkdir -p ~/.mlx-serve/omp
cat > ~/.mlx-serve/omp/models.yml <<'EOF'
providers:
  mlx:
    baseUrl: http://127.0.0.1:11234/v1
    api: openai-completions
    apiKey: mlx-serve
    compat:
      supportsDeveloperRole: false
      supportsReasoningEffort: true
      maxTokensField: max_tokens
      thinkingFormat: qwen
    models:
      - id: "MODEL_ID"
        contextWindow: CTX
        maxTokens: 8192
        reasoning: true
        input: [text]
EOF
export PI_CODING_AGENT_DIR="$HOME/.mlx-serve/omp"
omp --model mlx/MODEL_ID
```

omp 也支持实时发现，而不必用静态列表：把 `models:` 块换成 `discovery: { type: openai-models-list }`，它就会在启动时读取 `/v1/models`，并从每一行取该模型的真实上下文（mlx-serve 正是为此才在顶层公布它）。静态列表能把媒体模型挡在选择器之外，启动器因此采用它。

### OpenCode

不需要文件。`OPENCODE_CONFIG_CONTENT` 会合并到你自己的配置之上，所以插件和设置照样生效。

```bash
export OPENCODE_CONFIG_CONTENT='{"$schema": "https://opencode.ai/config.json", "provider": {"mlx": {"npm": "@ai-sdk/openai-compatible", "name": "MLX Serve (local)", "options": {"baseURL": "http://127.0.0.1:11234/v1"}, "models": {"MODEL_ID": {"name": "MODEL_ID (mlx-serve)", "limit": {"context": CTX, "output": 8192}}}}}}'
opencode --model mlx/MODEL_ID
```

### OpenCode 2

`opencode2`（npm 包 `@opencode/cli`）复用同一份 `OPENCODE_CONFIG_CONTENT` 提供商 JSON。启动器还会写出一个专用配置目录，并注册 mlx-serve 监控插件（侧边栏统计 + 底栏轮次计量，读取 `GET /metrics.json`）。

**要用 `--metrics` 启动服务器**（`mlx-serve serve --metrics`）。CLI 版服务器默认关闭 metrics，对 `/metrics.json` 返回 503；此时插件会显示 `feed --metrics off` 和一块空白面板，而底栏的轮次计量仍能靠流式响应工作。MLX Core 应用默认启用 metrics。`mlx-serve launch opencode2` 会探测该端点，发现它关闭时打印一条警告。

```bash
export XDG_CONFIG_HOME="$HOME/.mlx-serve/opencode2"
export OPENCODE_CONFIG_CONTENT='{"$schema": "https://opencode.ai/config.json", "provider": {"mlx": {"npm": "@ai-sdk/openai-compatible", "name": "MLX Serve (local)", "options": {"baseURL": "http://127.0.0.1:11234/v1"}, "models": {"MODEL_ID": {"name": "MODEL_ID (mlx-serve)", "limit": {"context": CTX, "output": 8192}}}}}}'
```

`$XDG_CONFIG_HOME/opencode/cli.json`（插件的 `package` 相对该配置目录）：

```json
{ "plugins": [ { "package": "./plugins/mlx-serve", "options": { "metricsUrl": "http://127.0.0.1:11234/metrics.json" } } ] }
```

回环会省略 `metricsToken`；非回环 URL 则加上 `"metricsToken": "mlx-serve"`（`Authorization: Bearer`）。`mlx-serve launch opencode2` 会把插件复制到 `~/.mlx-serve/opencode2/opencode/plugins/mlx-serve/`，绝不写入 `~/.config/opencode/`。

```bash
opencode2 --model mlx/MODEL_ID
```

### Codex

当前的 Codex 只支持 OpenAI Responses 协议，mlx-serve 在 `/v1/responses` 提供它。`CODEX_HOME` 会改变它整棵配置树的位置（该文件夹必须在 codex 运行前就存在）。无需配置 key：没有设置 `env_key` 时，codex 会跳过登录界面。

PATH 里没有 `codex`，但你有 ChatGPT 桌面应用？它把 CLI 打包在 `/Applications/ChatGPT.app/Contents/Resources/codex`（启动器会自动在那里找到它；`mlx-serve launch chatgpt` 同样可用）。

Codex 每一轮都会打印 `Model metadata for <id> not found. Defaulting to fallback metadata`。那是它内部维护的 OpenAI 模型 id 目录，任何自定义提供商的模型都会触发；这只是表面现象。真正要紧的部分 —— 上下文窗口 —— 来自配置里的 `model_context_window`，它会覆盖那个回退值。

```bash
mkdir -p ~/.mlx-serve/codex
cat > ~/.mlx-serve/codex/config.toml <<'EOF'
model = "MODEL_ID"
model_provider = "mlx"
model_context_window = CTX

[model_providers.mlx]
name = "MLX Serve (local)"
base_url = "http://127.0.0.1:11234/v1"
wire_api = "responses"
EOF
export CODEX_HOME="$HOME/.mlx-serve/codex"
codex
```

### Hermes

Hermes 从 `HERMES_HOME` 读取整棵配置树。`.env` 很关键：设置 `OPENAI_BASE_URL` 才是告诉 hermes 它已经配置好的信号，否则每个会话都会打开设置向导。

```bash
mkdir -p ~/.mlx-serve/hermes
cat > ~/.mlx-serve/hermes/config.yaml <<'EOF'
model:
  default: "MODEL_ID"
  provider: custom
  base_url: "http://127.0.0.1:11234/v1"
  api_key: "mlx-serve"
  api_mode: chat_completions
custom_providers:
  - name: mlx-serve
    base_url: "http://127.0.0.1:11234/v1"
    api_key: "mlx-serve"
    model: "MODEL_ID"
    api_mode: chat_completions
    models:
      "MODEL_ID":
        context_length: CTX
EOF
cat > ~/.mlx-serve/hermes/.env <<'EOF'
OPENAI_BASE_URL=http://127.0.0.1:11234/v1
OPENAI_API_KEY=mlx-serve
EOF
export HERMES_HOME="$HOME/.mlx-serve/hermes"
hermes
```

### Aider

环境变量加一个 litellm 元数据文件，好让 aider 知道真实的上下文窗口（没有它，未知的 `openai/` 模型只能拿到 litellm 的默认值）。

```bash
mkdir -p ~/.mlx-serve/aider
cat > ~/.mlx-serve/aider/model-metadata.json <<'EOF'
{
  "openai/MODEL_ID": {
    "max_input_tokens": CTX,
    "max_output_tokens": 8192,
    "max_tokens": 8192,
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
    "litellm_provider": "openai",
    "mode": "chat"
  }
}
EOF
export OPENAI_API_BASE='http://127.0.0.1:11234/v1'
export OPENAI_API_KEY=mlx-serve
aider --model openai/MODEL_ID --model-metadata-file ~/.mlx-serve/aider/model-metadata.json
```

## 编辑器与应用

### Zed

在 Settings > 打开 `settings.json`，加入该提供商。`max_tokens` 就是上下文窗口；Zed 索要 key 时，把占位 key 填在提供商设置界面里（Zed 拒绝在 settings.json 里写 key）。

```json
{
  "language_models": {
    "openai_compatible": {
      "mlx-serve": {
        "api_url": "http://127.0.0.1:11234/v1",
        "available_models": [
          {
            "name": "MODEL_ID",
            "display_name": "MODEL_ID (mlx-serve)",
            "max_tokens": CTX,
            "max_output_tokens": 8192
          }
        ]
      }
    }
  }
}
```

### OpenClaw

把该提供商加到 `~/.openclaw/openclaw.json`，然后选择模型：

```json5
{
  models: {
    providers: {
      mlx: {
        baseUrl: "http://127.0.0.1:11234/v1",
        apiKey: "mlx-serve",
        api: "openai-completions",
        models: [
          {
            id: "MODEL_ID",
            name: "MODEL_ID (mlx-serve)",
            reasoning: true,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: CTX,
            maxTokens: 8192,
          },
        ],
      },
    },
  },
}
```

然后运行 `openclaw agent --model mlx/MODEL_ID`，或在 Agent 默认设置里把它设为 `model.primary`。

### 其它任何 OpenAI 兼容的工具

Continue、Cline、Goose 以及大多数其它工具都有 “OpenAI-compatible” 提供商选项。指向 `http://127.0.0.1:11234/v1`，API key 随便填，再从 `/v1/models` 里挑一个模型 id。如果该工具需要填上下文窗口，用该模型的 `meta.context_length`。

## 沙盒中的 Agent

MLX Core 应用也能在 Linux 虚拟机（Agent 沙盒）里运行 pi 和 hermes，并替你注入配置。见 [docs/app.md](app.md)。
