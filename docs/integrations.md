# Integrations

mlx-serve speaks the standard APIs (OpenAI, Anthropic, Ollama, OpenAI Responses), so most coding agents and editors connect with a base URL and a model id. This page covers the zero-config launchers first, then manual setup per tool.

## Connection details

| What | Value |
|---|---|
| OpenAI-compatible | `http://127.0.0.1:11234/v1` (chat/completions, embeddings, images, audio) |
| OpenAI Responses | `http://127.0.0.1:11234/v1/responses` (what Codex uses) |
| Anthropic-compatible | `http://127.0.0.1:11234` (set `ANTHROPIC_BASE_URL`; serves `/v1/messages`) |
| Ollama-compatible | `http://127.0.0.1:11234/api/*` |
| API key | Any placeholder (e.g. `mlx-serve`). Loopback connections are exempt even with `--api-key` set; non-local clients need the real key |
| Model ids | `GET /v1/models`. Each row advertises its context window in `meta.context_length` and as top-level `context_length` / `max_model_len` for discovery-based clients |

Substitute your host/port if you changed them (default port is 11234).

One thing to get right in any manual config: the context window. The server pins the real window at model load (memory-bounded, often well under the model max) and advertises it in `/v1/models`. Hardcoding a bigger number in an agent's config means overflows; hardcoding a smaller one wastes the window. The launchers below read it from the server so you never type it.

## Zero-config: the launchers

Two ways to skip everything below:

- **MLX Core app**: the tray's **Code** button detects the CLIs installed on your Mac and launches them preconfigured against the running server (the App Store build shows copy-paste commands instead, since it isn't allowed to launch other apps).
- **`mlx-serve launch <agent>`**: same thing from the terminal, ollama-style:

```bash
mlx-serve launch claude              # any of: claude, pi, omp, opencode, codex, hermes, aider
mlx-serve launch codex --model Qwen3.5-27B-MLX-4bit
mlx-serve launch codex -- resume     # everything after -- goes to the agent
```

If no server is running, `launch` starts the MLX Core app and waits; without the app installed it tells you to run `mlx-serve serve` first. Flags: `--model`, `--url`, `--port`, `--print` (write the configs and print the launch script instead of running), `--no-start`.

Both launchers write configs into dedicated `~/.mlx-serve/<agent>/` folders and never touch your real agent configs (`~/.claude`, `~/.pi`, `~/.omp`, `~/.codex`, `~/.hermes` stay yours).

## Coding agents (manual setup)

Replace `MODEL_ID` with an id from `GET /v1/models` and `CTX` with that row's `meta.context_length`.

### Claude Code

Env vars only, no config file. mlx-serve serves the Anthropic Messages API natively.

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

pi needs a `models.json` naming the provider; `PI_CODING_AGENT_DIR` relocates the config dir so your real `~/.pi/agent` is untouched.

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

### oh-my-pi (omp)

Same idea as pi, but the file is `models.yml` (YAML). Note: omp still reads the `PI_CODING_AGENT_DIR` spelling for the config dir.

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

omp also supports live discovery instead of a static list: replace the `models:` block with `discovery: { type: openai-models-list }` and it reads `/v1/models` at startup, taking each model's real context from the rows (mlx-serve advertises it top-level exactly for this). The static list keeps media models out of the picker, which is why the launchers use it.

### OpenCode

No file needed. `OPENCODE_CONFIG_CONTENT` merges over your own config, so plugins and settings keep working.

```bash
export OPENCODE_CONFIG_CONTENT='{"$schema": "https://opencode.ai/config.json", "provider": {"mlx": {"npm": "@ai-sdk/openai-compatible", "name": "MLX Serve (local)", "options": {"baseURL": "http://127.0.0.1:11234/v1"}, "models": {"MODEL_ID": {"name": "MODEL_ID (mlx-serve)", "limit": {"context": CTX, "output": 8192}}}}}}'
opencode --model mlx/MODEL_ID
```

### Codex

Current Codex speaks only the OpenAI Responses wire API, which mlx-serve serves at `/v1/responses`. `CODEX_HOME` relocates its whole config tree (the folder must exist before codex runs). No key setup: with no `env_key` configured, codex skips the login screen.

No `codex` on PATH but you have the ChatGPT desktop app? It bundles the CLI at `/Applications/ChatGPT.app/Contents/Resources/codex` (the launchers find it there automatically; `mlx-serve launch chatgpt` works too).

Codex will print `Model metadata for <id> not found. Defaulting to fallback metadata` on every turn. That's its internal catalog of OpenAI model ids and it fires for any custom provider's model; it's cosmetic. The part that matters, the context window, comes from `model_context_window` in the config above, which overrides the fallback.

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

Hermes reads its whole tree from `HERMES_HOME`. The `.env` matters: `OPENAI_BASE_URL` set is what tells hermes it's configured, otherwise every session opens the setup wizard.

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

Env vars plus a litellm metadata file so aider knows the real context window (without it, unknown `openai/` models get litellm defaults).

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

## Editors and apps

### Zed

Settings > open `settings.json`, add the provider. `max_tokens` is the context window; put the placeholder key in the provider settings UI when Zed asks (Zed refuses keys in settings.json).

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

Add the provider to `~/.openclaw/openclaw.json` and pick the model:

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

Then `openclaw agent --model mlx/MODEL_ID`, or set it as `model.primary` in the agent defaults.

### Anything else OpenAI-compatible

Continue, Cline, Goose and most other tools have an "OpenAI-compatible" provider option. Point it at `http://127.0.0.1:11234/v1`, use any API key, and pick a model id from `/v1/models`. If the tool asks for a context window, use the model's `meta.context_length`.

## Sandboxed agents

The MLX Core app can also run pi and hermes inside a Linux VM (Agent Sandbox) with the config injected for you. See [docs/app.md](app.md).
