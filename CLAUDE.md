# mlx-serve – project context for AI

Native Zig server that runs MLX-format LMs on Apple Silicon and exposes OpenAI-compatible and Anthropic-compatible HTTP APIs. No Python.

## Stack

- **Zig** 0.15+
- **mlx-c** (Apple) via Homebrew; FFI in `src/mlx.zig`.
- **Jinja engine** (lib/jinja_cpp): llama.cpp's C++17 Jinja2 implementation with nlohmann/json. Pre-compiled as `libjinja.a` (rebuild: see comment in `build.zig`).
- **safetensors** for weights; BPE tokenizers (SentencePiece / byte-level)

## Layout

| Path | Role |
|------|------|
| `src/main.zig` | Entry, CLI (`--model`, `--serve`, `--host`, `--port`, `--prompt`, `--max-tokens`, `--temp`, `--ctx-size`, `--timeout`, `--reasoning-budget`, `--log-level`, `--version`, `--help`) |
| `src/mlx.zig` | mlx-c FFI |
| `src/model.zig` | Config + safetensors loading; supports Gemma-3, Gemma-4, Qwen3, Qwen3.5 MoE, Qwen3-next, Llama, Mistral |
| `src/tokenizer.zig` | BPE tokenizer |
| `src/transformer.zig` | Forward pass (embedding, attention, MLP, MoE, GatedDeltaNet); architecture dispatch |
| `src/generate.zig` | Autoregressive generation, sampling (temperature, top-k, top-p, repeat penalty, presence penalty, logprobs) |
| `src/chat.zig` | Chat template formatting (ChatML, Gemma turns, Llama-3, Jinja2 via llama.cpp engine); thinking/reasoning tags; tool call parsing |
| `src/server.zig` | HTTP server: `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/messages` (OpenAI + Anthropic compat, stream + non-stream, tool calling, KV cache) |
| `src/status.zig` | TUI status bar (CPU, memory, GPU metrics) |
| `src/log.zig` | Leveled logging (error, warn, info, debug) |
| `build.zig` | Zig build; links mlx-c and pre-compiled libjinja.a |

### MLX Claw (Swift macOS app)

| Path | Role |
|------|------|
| `app/Package.swift` | Swift package; `MLXClaw` executable + `MLXClawTests` test target |
| `app/Sources/MLXServe/MLXServeApp.swift` | App entry, menu bar + Chat/Browser windows |
| `app/Sources/MLXServe/AppState.swift` | Global state, chat session management, persistence |
| `app/Sources/MLXServe/Models/ChatModels.swift` | `ChatMessage`, `SerializedToolCall`, `ChatSession` |
| `app/Sources/MLXServe/Models/AgentModels.swift` | `AgentToolKind`, `AgentPlan`, `StepResult` |
| `app/Sources/MLXServe/Services/APIClient.swift` | HTTP + SSE streaming client for mlx-serve |
| `app/Sources/MLXServe/Services/AgentPrompt.swift` | System prompt, tool definitions (7 tools), `SkillManager` (prompt-based skills from `~/.mlx-serve/skills/`) |
| `app/Sources/MLXServe/Services/ToolExecutor.swift` | Tool handlers: shell, readFile, writeFile, editFile, searchFiles, browse, webSearch |
| `app/Sources/MLXServe/Services/BrowserManager.swift` | WKWebView (headless, created eagerly for background browsing) |
| `app/Sources/MLXServe/Services/ServerManager.swift` | mlx-serve process lifecycle, stderr capture (`serverLog`), auto-start |
| `app/Sources/MLXServe/Services/TestServer.swift` | Embedded HTTP server (port 8090) for test automation — same code path as UI |
| `app/Sources/MLXServe/Services/AgentMemory.swift` | Agent context memory (recent dirs, commands) |
| `app/Sources/MLXServe/Views/ChatView.swift` | Chat UI + `runAgentLoop()` + `buildAgentHistory()` |
| `app/Sources/MLXServe/Views/StatusMenuView.swift` | Menu bar UI, server log viewer, Claude Code launcher |
| `app/Sources/MLXServe/Views/BrowserView.swift` | Browser window (uses shared WKWebView) |

## Testing

- `zig build test` — unit tests (chat, server, generate, model, log, tokenizer)
- `cd app && swift test` — Swift unit tests (agent harness, SSE parsing, serialization, history)
- `./tests/integration_test.sh [model_dir] [port]` — 36 end-to-end API tests (needs a model)
- `./tests/test_tool_response.sh [port]` — tool calling round-trip tests (needs running server)
- `./tests/test_kv_cache_poison.sh [port]` — KV cache poisoning regression test (needs running server)
- `./tests/test_anthropic_api.sh [port]` — Anthropic Messages API integration tests (needs running server)
- Always run `zig build test` and `swift test` before submitting changes
- Add tests for new pure logic functions in the same source file (Zig convention)
- Shell integration tests go in `tests/` and need a running server with a loaded model

## Building

- **Full app bundle**: `cd app && SKIP_NOTARIZE=1 bash build.sh` — builds Zig + Swift, assembles `.app`, signs (requires `APPLE_DEVELOPER_ID` and `APPLE_TEAM_ID` env vars)
- Zig server only: `zig build -Doptimize=ReleaseFast`
- Swift app only: `cd app && swift build -c release`
- For tests: `zig build test` (Zig) and `cd app && swift test` (Swift)
- **Rebuild Jinja library** (after changing `lib/jinja_cpp/*.cpp`): `cd lib/jinja_cpp && for f in jinja_wrapper caps lexer parser runtime jinja_string value; do clang++ -std=c++17 -O2 -DNDEBUG -I . -c $f.cpp -o obj/$f.o; done && ar rcs libjinja.a obj/*.o`

## Conventions

- Prefer minimal, DRY Zig; avoid unnecessary abstraction.
- Chat templates live in model dirs; llama.cpp's Jinja engine renders them (with fallback formatting).
- Server supports concurrent health checks via threaded connections, single-slot generation.
- KV cache reuse across requests via prompt prefix matching; invalidated after tool-calling requests and pad-only generations.
- Tests go at the bottom of each source file (Zig convention).
- Jinja static library must be rebuilt with system clang++ after changing `lib/jinja_cpp/*.cpp` (see build command in `build.zig`).

## Anthropic Messages API

The server exposes `POST /v1/messages` for Anthropic API compatibility, enabling Claude Code and other Anthropic SDK clients to use local models.

### Request/Response mapping
- **System prompt**: Anthropic puts `system` at top level → converted to internal system message
- **Content blocks**: Anthropic messages use typed content blocks (`text`, `tool_use`, `tool_result`, `thinking`) → converted to internal `Message` structs
- **Tools**: Anthropic `input_schema` → converted to OpenAI `parameters` format for chat template compatibility
- **Tool results**: Anthropic `tool_result` in user messages → internal `role: "tool"` messages
- **Thinking**: `thinking` config parsed → maps to `enable_thinking` + `reasoning_budget`; thinking blocks emitted with fake `signature` field
- **Stop reasons**: `stop` → `end_turn`, `length` → `max_tokens`, `tool_calls` → `tool_use`

### Streaming format
Anthropic SSE uses named events: `message_start`, `content_block_start`, `content_block_delta` (with `text_delta`, `thinking_delta`, `signature_delta`, `input_json_delta`), `content_block_stop`, `message_delta`, `message_stop`. Each content block has an explicit start/stop lifecycle with an index.

### Claude Code integration
The MLX Claw app has a "Launch Claude Code" button (visible when server is running) that opens Terminal with the `claude` CLI configured to use the local server:
- `ANTHROPIC_BASE_URL` → local server URL
- `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` → dummy values (local server, no auth)
- `ANTHROPIC_DEFAULT_*_MODEL` → `mlx-serve` (routes all model tiers through local)
- `CLAUDE_CODE_SUBAGENT_MODEL` → `mlx-serve`

## Tool Calling Architecture

### Server side (Zig)
- **Tool call detection**: When `tools` param is present, server buffers tokens and checks for tool call patterns. If thinking is enabled, thinking tokens are buffered separately and not flushed as content. After generation, `chat.parseToolCalls()` checks for patterns (`<tool_call>`, Hermes XML, Gemma 4 `<|tool_call>`, raw JSON). Gemma 4 double-brace args (`{{"key":"value"}}`) are unwrapped before JSON parsing.
- **Message serialization** (`chat.serializeMessagesJson`): Converts `Message` structs to JSON for Jinja templates. `role: "tool"` messages are passed natively (no transformation) — Gemma 4 templates handle them directly as `<|turn>tool`. Tool call `arguments` are serialized as JSON strings (not objects) so templates render them correctly.
- **Streaming SSE**: Tool call arguments are sent in a single delta (name + id + full args) to prevent client-side accumulation bugs. Thinking content (`<|channel>thought`) is detected during streaming and buffered until the closing tag, then emitted as `reasoning_content`.
- **Fallback formatter** (`chat.fallbackFormatChat`): Used when Jinja fails. Handles ChatML (`<tool_call>/<tool_response>`), Llama (`ipython` role), Gemma (`Tool result:` in user turn).
- **KV cache**: `reuseKVCache()` compares token-by-token prefix. Cache is automatically invalidated after tool-calling requests (generated tool-call tokens corrupt the cache for the next request) and after pad-only generations. Sliding window layers keep full buffers (no trimming) — views return the last `sw` entries during decode, all entries during prefill.

### Client side (Swift)
- **Agent loop** (`ChatView.runAgentLoop`): Up to 30 iterations. Calls model with tools → parses tool calls → executes locally → feeds results back → repeats until model responds without tool calls. Adds synthetic user nudge after tool results for models that need it.
- **History builder** (`ChatView.buildAgentHistory`): Converts `ChatMessage` array to OpenAI API format. Filters out error messages, pad-only content, and agent summaries. Truncates assistant messages at 500 chars. Last 30 messages max.
- **SSE parsing** (`APIClient.performStream`): Accumulates streamed tool call deltas. Server sends full arguments in one delta. Emits `.toolCalls` event on `finish_reason: "tool_calls"`. Fallback emission if stream drops without finish_reason.
- **Tool call storage**: `SerializedToolCall` (id, name, arguments as JSON string) stored on `ChatMessage.toolCalls`. Persisted via Codable for history replay. Backwards-compatible with old history files (field is optional).
- **Error recovery**: Tool execution errors include what args were sent and ask the model to retry, enabling self-correction in the agent loop.

## Prompt-based Skills

Users can teach the agent new capabilities by dropping `.md` files in `~/.mlx-serve/skills/`. Each file has YAML frontmatter:

```markdown
---
name: deploy
description: Deploy the project to production
trigger: deploy, release, push to prod, ship it
---
When asked to deploy:
1. Run `git push origin main`
2. Check CI with `gh run list --limit 1`
```

- `trigger` — comma-separated keywords; if the user's message contains ANY keyword (case-insensitive substring), the skill body is injected into the system prompt
- A short skill index (name + description) is always included so the model knows what's available
- `SkillManager` in `AgentPrompt.swift` scans the directory on each agent loop iteration (re-scans when dir modification date changes)
- Skills are injected in `ChatView.runAgentLoop()` between the base system prompt and agent memory context
- UI: folder icon button in menu bar and chat toolbar opens `~/.mlx-serve/` in Finder

## Resumable Downloads

Large model downloads (e.g., 26B at ~15 GB) use streaming writes to `.partial` files:

- `DownloadManager` uses `URLSessionDataTask` with `StreamingDelegate` — bytes are written to `<file>.partial` as they arrive
- If a download is interrupted (network drop, app crash), the `.partial` file survives on disk
- On retry/resume, sends `Range: bytes=<existingSize>-` header; if server returns 206, only the remainder is downloaded; if 200, truncates and restarts
- Automatic retry: 3 attempts per file with 2s/4s backoff; status text shows "Connection lost, retrying..."
- Cancellation preserves `.partial` files for future resume
- UI shows "Resume" instead of "Download"/"Retry" when `.partial` files exist (`hasPartialDownload()`)
- Already-completed files are skipped (size check against HuggingFace metadata)

## Debugging

### Server logs
- Start server with `--log-level debug` for verbose output (Jinja errors, cache hits, token counts)
- The MLX Claw app starts the server as a subprocess; stderr is captured in `ServerManager.serverLog` (64KB rolling buffer). View it via the log button (text-align icon) next to Start/Stop in the menu bar.
- To see logs from a manually-started server: `./zig-out/bin/mlx-serve --model <path> --serve --port 8080 --log-level debug 2>&1`
- Key log patterns:
  - `jinja error: ..., using fallback` — Jinja template failed, check template compatibility
  - `[cache] reusing N/M tokens` — KV cache hit; if N is close to M, most of prompt is cached
  - `[cache] invalidated` — cache was reset (tools config changed, etc.)
  - `<- N+M tokens (Xms) [reason]` — N prompt tokens, M completion tokens, finish reason
  - `tool_msgs=N` — count of `role: "tool"` messages in the request

### Swift app logs
- `print()` in the Swift app goes to stdout, not visible when launched via `open`. To see it: run the binary directly from terminal, or write to a file.
- The app dumps every agent loop request to `~/.mlx-serve/last-agent-request.json` (debug aid). Replay with: `curl -sf http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d @~/.mlx-serve/last-agent-request.json`
- Chat history is persisted at `~/.mlx-serve/chat-history.json`

### Reproducing issues
- To test tool calling without the app: use `curl` with `stream: false` first (simpler to inspect), then `stream: true` (matches app behavior).
- To test the Jinja template offline: `pip3 install jinja2`, then render with Python using the model's `chat_template.jinja` file and the dumped request JSON.
- To test KV cache effects: restart the server fresh between tests (`pkill -f mlx-serve`). A single bad request can poison the cache for all subsequent requests.

## Gotchas

### KV cache after tool calls
After a tool-calling request, the KV cache is automatically invalidated. The generated tool-call tokens are in the cache but not in `cached_prompt_ids`, so reusing the cache for the next request (which includes tool results) would corrupt attention. Similarly, pad-only generations trigger cache invalidation.

### Sliding window KV cache
Models with sliding window attention (e.g., Gemma 4 E4B with 512-token window) keep the full KV buffer — no trimming. During prefill, all entries are returned so Q and K dimensions match. During decode, views return only the last `sw` entries. The sliding window mask handles attention scope. This matches mlx-lm's `RotatingKVCache` behavior.

### Gemma 4 tool calling format
Gemma 4 templates handle `role: "tool"` natively (producing `<|turn>tool`). No transformation is needed — the server passes tool messages through as-is. The `tool_responses` field is NOT added (it causes duplicate content in rendered prompts). Tool call arguments are serialized as JSON strings so the template renders them verbatim.

### Streaming with tools and thinking
When `tools` are present, the server buffers tokens to detect tool call patterns. If thinking is also enabled, `<|channel>thought` tokens are detected and kept buffered (not flushed as content) until the closing `<channel|>` tag. After generation, thinking content is split from visible content and emitted as `reasoning_content`. Channel tags (`<|channel>`, `<channel|>`) are stripped from visible content.

### mlx-c API changes
mlx-c 0.6.0 added a `global_scale` parameter (may be null) to `mlx_dequantize` between `mode` and `dtype`. The FFI declaration in `mlx.zig` must match the installed header. When upgrading mlx-c, diff the headers in `/opt/homebrew/include/mlx/c/ops.h` against the `extern "c"` declarations in `src/mlx.zig`.

### Two binaries in the app bundle
The MLX Claw `.app` bundle contains TWO binaries: `MLXClaw` (Swift UI) and `mlx-serve` (Zig server). Both must be updated when making changes. The Swift app starts the Zig server as a child process. Forgetting to copy one binary after a rebuild is a common source of "it still doesn't work."

### WebSearch and Browse
The `webSearch` tool navigates to DuckDuckGo HTML search and extracts structured results (titles, URLs, snippets) via JavaScript. The `browse` tool's `readText` action navigates to the URL first, then extracts text — this ensures each browse returns the correct page content (not the previous page's).

### WKWebView requires main thread
`BrowserManager` is `@MainActor`. All WKWebView operations (navigate, readText, evaluateJS) must happen on the main thread. The WKWebView is created eagerly at app launch so tools work without the Browser window being open.

### Swift JSONSerialization quirks
- `[String: Any]` dictionaries serialize with non-deterministic key order
- Empty string `""` stays as `""` in JSON (not `null`); the server treats both as empty
- `Double` values like `0.7` serialize as `0.69999999999999996` (floating point); this is fine
- `arguments` in tool_calls must be a JSON String (e.g., `"{\"command\":\"ls\"}"`) not a nested dict; the server checks `if (v == .string)` to extract it
