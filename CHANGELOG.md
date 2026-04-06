# Changelog

## 2026.4.6 — Gemma 4 MoE, Jinja Upgrade, Tool Calling Overhaul

### Gemma 4 Full Support
- **Gemma 4 MoE (26B-A4B)**: Sigma-MoE routing, separate shared/routed expert branches, 5 feedforward norms, layer scalar, GeGLU activation
- **Gemma 4 E2B/E4B**: Per-Layer Embeddings (PLE) with gated projection, per-layer input scaling
- **ProportionalRoPE**: Correct frequency computation for global attention layers with positive exponents and full head_dim denominator
- **K=V attention**: Global layers share K projection as V (no separate v_proj) with automatic fallback
- **Per-weight quantization detection**: Auto-detect quant bits per weight instead of global default (fixes 8-bit shared expert in 4-bit model)
- **Sliding window attention**: Correct KV cache view handling — full buffer during prefill, windowed during decode (matches mlx-lm's RotatingKVCache)
- **Logit softcapping**: Applied after final norm + lm_head projection

### Jinja Template Engine Upgrade
- **Replaced jinja.hpp with llama.cpp's Jinja engine**: Full-featured C++17 Jinja2 implementation with nlohmann/json
- **Fixed tool call argument rendering**: Old engine produced empty args (`{command:{}}` instead of `{"command":"echo hi"}`)
- **Fixed tool parameter types**: Old engine lost type info (`type:<|"|><|"|>` instead of `type:<|"|>STRING<|"|>`)
- **Removed broken tool message transformation**: `role:"tool"` messages now passed natively to templates (both E4B and 26B templates handle it correctly)
- **Removed redundant tool_responses field**: Was causing duplicate content in rendered prompts (~66 extra chars per tool response)
- **Tool call arguments serialized as JSON string**: Prevents templates from re-serializing parsed objects incorrectly

### Tool Calling Reliability
- **Gemma 4 double-brace parsing**: Model generates `{{"key":"value"}}` — outer braces now unwrapped before JSON parsing
- **Streaming SSE argument fix**: Full arguments sent in single delta instead of empty + chunks (prevents `""query` double-quote accumulation on client)
- **KV cache invalidated after tool-calling requests**: Prevents stale attention state from generated tool-call tokens poisoning the next request
- **User nudge after tool results**: Synthetic user message added when last history entry is `role:"tool"` (some models can't generate without it)
- **Improved tool descriptions**: Examples in each tool definition guide parameter usage
- **Better error messages**: Tool errors include what args were sent and ask model to retry with correct parameters

### Thinking/Reasoning Mode
- **Fixed thinking leak with tools**: `<|channel>thought` content no longer streamed as visible content when tools are present
- **Gemma 4 channel tag stripping**: `<|channel>` and `<channel|>` tags stripped from content output
- **Partial thinking detection**: Buffers tokens when `<|channel>` prefix detected (before `thought` suffix arrives) to prevent premature flushing
- **`splitThinkBlock` fixes**: Correct handling of Gemma 4's `<|channel>thought...<channel|>\n<|channel>\ncontent` format

### MLX Claw App
- **Auto-start server on launch**: Toggle in menu bar, persists selected model in UserDefaults
- **Selected model persistence**: Last used model path saved across launches
- **Test API server** (port 8090): REST endpoints for automated testing (`/test/start`, `/test/stop`, `/test/reset`, `/test/chat`, `/test/agent`, `/test/history`, `/test/status`)
- **Concurrent request handling**: TestServer accept loop runs on background GCD thread, each request handled concurrently
- **Health polling fix**: DispatchSource timer for reliable server status detection (replaces Timer/Task approaches that failed with MenuBarExtra apps)
- **Agent loop iterations**: Increased from 10 to 30 for complex multi-step tasks
- **Browse tool fix**: `readText` action now navigates to URL first (was returning previous page's content)
- **WebSearch results**: Structured extraction of titles/URLs/snippets instead of raw DuckDuckGo HTML
- **UI**: "Tool Call" label with wrench icon (was "Summary"), folder button opens `~/.mlx-serve/`
- **`buildAgentHistory` fixes**: Filters "couldn't generate" error messages, strips `<pad>`, truncates assistant content at 500 chars, matches ChatView exactly in TestServer

### Server Fixes
- **Pad-only cache invalidation**: Detects all-zero token IDs and invalidates cache + prompt IDs
- **Sliding window cache check**: Reset cache when either previous or new prompt exceeds window (was only checking `cached > sw AND shared < sw`)
- **`moe_seq_offset` sync**: Properly updated on both truncation and full reset paths

---

## 2026.4.5 — Prompt-Based Skills, Resumable Downloads

- **Prompt-based skills system**: User-defined agent capabilities via `~/.mlx-serve/skills/*.md` with YAML frontmatter (name, description, trigger keywords)
- **Resumable downloads**: Streaming writes to `.partial` files, Range header support for resume, 3 automatic retries with backoff
- **Disk space safety**: Pre-check available space before large downloads
- **SkillManager**: Scans skills directory on each agent loop, re-reads when directory modification date changes

## 2026.4.4 — KV Cache & Tool Calling Fixes

- **KV cache corruption fix**: Invalid suffix cache invalidation, SSM state reset
- **Tool calling reliability**: Improved tool call parsing, agent harness stability
- **App bundle packaging**: Removed Bundle.module dependency, fixed codesigning

## 2026.4.3 — MLX Claw Major Update

- **Native tool calling UI**: 7 built-in tools (shell, readFile, writeFile, editFile, searchFiles, browse, webSearch)
- **Agent mode**: Automatic ReAct loop with tool execution and result feeding
- **Browser integration**: WKWebView-based browsing, headless operation for background tool use
- **Streaming chat**: SSE parsing with delta reconstruction for real-time responses
- **Multi-session chat**: Persistent chat history with session management

## 2026.4.2 — MLX Claw Initial Release

- **Swift macOS menu bar application**: Server management, model selection, chat interface
- **Server lifecycle**: Subprocess launch/termination with stderr capture
- **Model discovery**: Local model scanning from `~/.mlx-serve/models/`

## 2026.3 — Embeddings, Reasoning, Jinja

- **Embedding support**: BERT and encoder-only models via `/v1/embeddings`
- **Reasoning budget**: `--reasoning-budget` CLI flag to limit thinking tokens
- **Jinja_cpp integration**: Replaced vibe-based Jinja (macros caused infinite loops)
- **Qwen3.5 MoE support**: GatedDeltaNet linear attention, shared expert routing
- **TUI status bar**: Live CPU, memory, GPU metrics

## 2026.2 — Initial Release

- **Zig native server**: OpenAI-compatible HTTP API on Apple Silicon
- **MLX-c FFI**: GPU-accelerated tensor operations via Apple's MLX C API
- **Model support**: Llama 3, Mistral, Qwen 3
- **BPE tokenizer**: SentencePiece and byte-level BPE
- **Streaming generation**: SSE-based real-time token delivery
- **KV cache reuse**: Prompt prefix matching across requests
- **Sampling**: Temperature, top-p, top-k, repeat penalty
