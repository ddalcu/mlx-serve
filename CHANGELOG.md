# Changelog

## v26.4.21 — Vision Pipeline, Prefill/Decode Metrics, 3x Prefill Speedup

### Vision Encoder (Gemma 4 SigLIP)
- **Full vision pipeline**: Gemma 4 models can now process images end-to-end — SigLIP vision encoder with patch embedding, 2D RoPE, clipped linears, position pooling, and embedding projection
- **Image decoding**: JPEG/PNG via stb_image, WebP via libwebp — decoded, resized to model's expected resolution, converted to float32 CHW format
- **OpenAI `image_url` content blocks**: Supports `data:image/jpeg;base64,...` and preprocessed `data:image/x-mlx-pixels;base64,...` in chat completion requests
- **`--no-vision` flag**: Disables vision encoder at startup to save ~340MB GPU memory when not needed
- **KV cache invalidation on image requests**: Image tokens have identical IDs but different vision embeddings — cache is reset to prevent stale feature reuse
- **New FFI bindings**: `mlx_equal`, `mlx_remainder`, `mlx_ones`, `mlx_minimum`, `mlx_cos`, `mlx_sin`, `mlx_array_data_bool` added to `mlx.zig`
- **Build changes**: Links stb_image and libwebp; `build.zig` updated for both exe and test targets

### Prefill/Decode Metrics
- **Separate timing in server logs**: All 6 handler paths (non-streaming/streaming × completions/chat/Anthropic) now report prefill and decode tok/s independently instead of a single combined figure
- **Old**: `<- 1133+256 tokens (5906ms, ~43 tok/s) [length]`
- **New**: `<- 1133+256 tokens (5906ms) [prefill: 606 tok/s, decode: 63 tok/s] [length]`

### 3x Prefill Speedup — Split Prefill (matches mlx-lm)
- **Root cause**: The full forward pass applied `lm_head` projection (`[seq_len, 1536] @ [1536, 262144]`) over the entire prompt — for 870 tokens that's a ~912MB output tensor computed and immediately discarded (only the last position's logits are needed for sampling)
- **Fix**: `Generator.init()` now splits prefill into two phases, mirroring mlx-lm's `generate_step`:
  1. **Prefix pass** (N-1 tokens): Forward pass builds the lazy graph including `lm_head`, but only KV cache entries are evaluated — MLX's lazy evaluation skips the `lm_head` matmul entirely since nothing depends on it
  2. **Last token pass** (1 token): Forward + `lm_head` on a single token produces the logits needed for sampling, then chains into the decode pipeline via `async_eval`
- **`mlx_clear_cache()`** between phases frees intermediate Metal buffers from the prefix pass
- **Conditional KV array allocation**: Moved 9 temp array declarations inside the non-shared KV block in `forwardStandard()` — eliminates 180 unnecessary `mlx_array_new()`/`free()` calls per forward pass for Gemma 4 E2B (20 of 35 layers share KV)

#### Benchmark: Gemma 4 E2B-it 4-bit, Apple Silicon (mlx 0.31.1, mlx-c 0.6.0, llama.cpp b8680)

**Prefill (tok/s)**

| Prompt length | mlx-serve | mlx-lm | llama.cpp |
|---------------|-----------|--------|-----------|
| Short (~20 tok) | 22 | — | 189 |
| Medium (~100 tok) | 252 | — | 397 |
| Long (~900 tok) | **1,266** | ~equal wall | 554 |

**Decode (tok/s)**

| Test | mlx-serve | mlx-lm | llama.cpp |
|------|-----------|--------|-----------|
| 300-token generation | **63.7** | 61.7 | 48.0 |
| Reasoning Q&A | **64.3** | 60.8 | 47.9 |
| Code generation | **63.4** | 59.9 | 46.7 |
| Tool calling | **62.4** | N/A | 47.1 |

**Wall time (ms) — agentic workloads**

| Test | mlx-serve | mlx-lm | llama.cpp |
|------|-----------|--------|-----------|
| Tool call selection | **798** | N/A | 6,767 |
| Multi-turn tool fix | **8,278** | 8,567* | 11,293 |
| Code gen (600 tok) | **9,657** | 9,960 | 13,003 |

*mlx-lm has no tool calling support — tool tests adapted to plain prompts

- **2.3x faster prefill** than llama.cpp on long prompts
- **35% faster decode** than llama.cpp, 5% faster than mlx-lm
- **8.5x faster tool calling** than llama.cpp (only engine producing valid structured tool calls)
- **25–35% faster wall time** than llama.cpp on generation-heavy tasks

### MLX Core App — Image Support
- **Image attachment UI**: Drag-and-drop or paste images into the chat input; thumbnails with remove buttons shown before sending
- **`ImagePreprocessor`**: Resizes and converts images to float32 CHW pixel data for the vision encoder
- **`ChatImage` model**: JPEG image data attached to `ChatMessage`, persisted via Codable
- **Multimodal content blocks**: `buildMultimodalContent()` constructs OpenAI-format `image_url` content blocks with preprocessed pixel data
- **Screenshot capture from browse tool**: Browse results can include screenshots sent as vision input

### MLX Core App — Agent Improvements
- **`cwd` tool**: New "Change Directory" tool lets the agent switch working directory for subsequent shell commands
- **Shell output includes cwd**: Shell tool results now show `[cwd: /path]` prefix so the model knows where commands ran
- **Context monitor**: Real-time prompt token / context length usage bar shown above the input area
- **Context-aware image handling**: Images stripped from history replay to avoid stale vision features and context waste

### MLX Core App — UI Refinements
- **StatusMenuView cleanup**: Streamlined server controls layout
- **TestServer vision support**: Test API endpoints handle image content blocks for automated vision testing

### Model Support
- **Vision config parsing**: `ModelConfig` now parses `vision_config` from `config.json` (hidden_size, num_layers, num_heads, head_dim, intermediate_size, patch_size, pooling, position_embedding_size, etc.)
- **`loadWeightsWithVision()`**: Loads both language model and vision tower weights from safetensors

### Testing
- **`tests/test_vision.sh`**: Vision pipeline integration tests
- **`tests/fixtures/`**: Test image fixtures for vision tests

---

## v26.4.20 — Tool Reliability, Thinking+Tools, Truncation Recovery

### Tool Parameter Key Order (3-layer fix)
- **Pre-serialized tool JSON**: `toolDefinitionsJSON` is a hand-crafted JSON string with guaranteed `path` before `content` in all file tools — bypasses `JSONSerialization`'s non-deterministic key ordering
- **Request body splicing**: `streamChat()` splices the pre-serialized tools JSON directly into the HTTP body instead of letting Swift re-order keys
- **Truncated JSON recovery**: `extractPathFromTruncatedJSON()` uses string search to find `"path":"..."` in malformed/truncated args even when JSON parsing fails entirely
- **Improved JSON repair**: Repair block now tracks unmatched `{`/`[` openers respecting quoted regions and appends correct closing characters (was blindly appending single `}`)
- **Path cleaning**: `cleanPath()` strips spurious surrounding quotes from paths — models sometimes generate `"\"app.py\""` which becomes `"app.py"` with literal quotes after JSON unescaping

### Thinking + Tools Fix
- **Streaming**: When `has_tools=true`, thinking blocks were detected and **stripped** via `stripThinkBlock()` but never emitted as `reasoning_content`. Now uses `splitThinkBlock()` to separate reasoning from content and emits both
- **Non-streaming**: Tool call responses had `"content":null` with no `reasoning_content` field. Now includes `reasoning_content` when thinking is enabled
- **Root cause**: The `if/else if` control flow meant the thinking branch could never execute when the tools branch was active

### Gemma 4 Tool Call Parsing
- **Nested brace/array values**: `convertGemma4ArgsToJson()` now handles bare nested objects (`{config:{"port":3000}}`) and arrays (`{stops:["Rome","Venice"]}`) via depth-tracked brace matching — was previously falling through to bare-value parsing and producing invalid JSON

### Agent Prompt & Token Limits
- **Default max_tokens**: 8192 → 32768 — prevents tool call argument truncation for large file writes
- **writeFile size guidance**: System prompt and tool description now direct models to use `shell` with `cat` heredoc for files over 100 lines — writeFile's JSON-escaped content inflates token cost ~30% vs raw heredoc
- **Max tokens warning**: New `SSEEvent.maxTokensReached` emitted when server returns `finish_reason: "length"` — chat shows "Output truncated — max tokens (N) reached"
- **Tool output overflow**: Truncation message no longer tells model to `readFile` the overflow file (which is outside the workspace and gets blocked by confinement) — just shows `[... truncated at N of M chars]`

### Claude Code Launcher
- **Removed from chat toolbar**: Claude Code button removed from ChatView — only in tray menu now
- **Directory picker**: Tray menu button shows `NSOpenPanel` folder picker before launching (defaults to `~/.mlx-serve/workspace`)
- **White icon**: Uses `ClaudeIcon(size: 12)` with `.foregroundStyle(.white)` matching the chat icon style

### Dead Code Removal
- **Removed `chatWithTools()`**: Non-streaming tool call method was never called — app uses streaming-only. Removed along with `ToolCallResult` struct

### Testing
- **`tests/test_thinking_tools.sh`**: 27 integration tests covering all 8 permutations of thinking × tools × streaming, plus mixed-mode scenarios
- **`tests/test_swift_agent.sh`**: Comprehensive Swift agent harness test — exercises all 9 tools (writeFile, readFile, editFile, shell, searchFiles, listFiles, webSearch, browse, saveMemory) through the TestServer API
- **`ToolKeyOrderTests.swift`**: 34 unit tests for JSON key order, request body splicing, `extractPathFromTruncatedJSON` edge cases, JSON repair, and end-to-end `parseToolCallArgs` integration

---

## 2026.4.12 — MLX Core Rename, Agent Overhaul

### Rename: MLX Claw → MLX Core
- **Full rebrand**: App renamed from "MLX Claw" to "MLX Core" across all source, build scripts, CI/CD, docs, tests, and bundle identifiers (`com.dalcu.mlx-core`)

### New Tool: listFiles
- **Dedicated file listing**: `listFiles` tool with glob pattern matching and recursive traversal — replaces shell `ls`/`find` in agent workflows
- **Tool routing guidance**: System prompt now directs the model to use dedicated tools (`readFile`, `writeFile`, `editFile`, `searchFiles`, `listFiles`) instead of shell equivalents

### Agent Loop Improvements
- **150 max iterations** (up from 30): Supports complex multi-step agent tasks
- **Token-aware context management**: `buildAgentHistory()` now estimates token costs per message and fits history to the context budget instead of using a fixed 28-message window
- **Tool result overflow**: Oversized tool results saved to `~/.mlx-serve/tool-output/` with a truncated preview in context plus a pointer to the full file — prevents context blowout from large shell output or file reads
- **Per-tool context caps**: shell 6K, readFile 8K, searchFiles/listFiles 4K, browse 3K, webSearch 2K, editFile/writeFile 2K
- **Working directory injection**: Agent system prompt now includes the working directory path so relative paths resolve correctly
- **Pad retry with backoff**: Failed empty generations use exponential backoff (`RetryPolicy.aggressive`) instead of immediate retry

### System Prompt Redesign
- **Hardcoded base + additive user customization**: Base system prompt is no longer editable — `~/.mlx-serve/system-prompt.md` now contains only user additions appended via `# User Instructions`
- **File editing rules**: Explicit instructions for readFile → editFile workflow (line numbers, exact match, startLine/endLine for large files)
- **Error recovery section**: Structured guidance for tool failure diagnosis
- **Memory cap**: Memory entries capped at last 30 lines / 2000 chars to prevent context bloat

### Tool Enhancements
- **readFile with line numbers**: Output now shows `N| text` format so the model can reference specific lines for editFile
- **readFile large file headers**: Files >200 lines or >6KB show metadata header with total line count and byte size
- **searchFiles upgrade**: Uses ripgrep when available, supports `include` glob filter, `context` lines, and `maxResults` parameter
- **writeFile unescape**: Handles `\\n`, `\\t`, `\\\"` double-escaping from smaller models
- **Shell output uncapped**: Removed 8KB truncation on shell results (overflow system handles large output instead)

### API Client
- **Retry with exponential backoff**: Network errors (connection lost, timeout, cannot connect) retry up to 5 times with jitter — replaces single-retry on connection lost
- **Tool call argument cleanup**: Strips surrounding quotes from tool call parameter values that smaller models add

### Claude Code Launcher
- **Folder picker**: Claude Code button in chat toolbar opens NSOpenPanel to select working directory before launch
- **Working directory support**: `launchClaudeCode()` now accepts and `cd`s into the chosen directory
- **Claude icon**: Custom SwiftUI `ClaudeShape` renders the official Claude AI logo SVG as a native Shape

### UI
- **Scroll tracking fix**: Replaced window-frame-based scroll detection with proper GeometryReader preference keys; upward mouse scroll disengages auto-scroll, scrolling back to bottom re-engages
- **Compact bottom bar**: Browser and Claude Code buttons use icon-only style with tooltips
- **Overflow file cleanup**: Old tool-output files (>24h) cleaned up on session start

### TestServer
- **Non-blocking agent jobs**: `POST /test/agent` now returns immediately with a `job_id`; poll `GET /test/agent/status` for progress and results

---

## 2026.4.11 — Anthropic API, Claude Code, KV Cache Fix

### Anthropic Messages API
- **`POST /v1/messages`**: Full Anthropic API compatibility layer — Claude Code and other Anthropic SDK clients can use local models
- **Request conversion**: Anthropic content blocks (`text`, `tool_use`, `tool_result`, `thinking`) converted to internal format; `system` as top-level field; tools `input_schema` converted to OpenAI `parameters` for chat template compatibility
- **Streaming SSE**: Named events (`message_start`, `content_block_start/delta/stop`, `message_delta`, `message_stop`) with `text_delta`, `thinking_delta`, `signature_delta`, `input_json_delta` delta types
- **Non-streaming**: Anthropic response format with content block arrays, `stop_reason` mapping (`stop`→`end_turn`, `length`→`max_tokens`, `tool_calls`→`tool_use`)
- **`HEAD /` handler**: Claude Code sends a connectivity check before API calls — now returns 200
- **Query string stripping**: `POST /v1/messages?beta=true` now correctly routes (Claude Code appends `?beta=true`)

### Claude Code Integration
- **Launch button**: "Launch Claude Code" button in MLX Core menu bar (visible when server is running) — opens Terminal with `claude` CLI configured to use the local server via `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, model tier overrides
- **Binary detection**: Finds `claude` via login shell PATH (`/bin/zsh -l -c "which claude"`) with fallback to `~/.local/bin/claude`
- **`.command` file approach**: Uses `NSWorkspace.open()` on a temp `.command` file — no AppleScript permissions needed

### GPU Memory Safety
- **Attention memory preflight check**: Estimates peak GPU memory (attention matrix + KV cache + 20% margin) before prefill and rejects with a 400 error instead of crashing the server with an uncatchable Metal C++ exception
- **Dynamic Metal limit**: Queries actual system memory via `sysctl hw.memsize` × 75% (not hardcoded) — works correctly on 16GB and 128GB machines
- **Context size auto-detection**: Default context size computed from GPU memory at startup (replaces hardcoded 16K cap). Logged as `Context size: N tokens (auto, from GPU memory)`

### Context Size UI
- **Context size selector**: New UI in MLX Core server block with Auto/16K/32K/64K/128K presets
- **Auto mode**: When set to Auto (default), server computes safe max from model architecture + available GPU memory
- **GPU safe max indicator**: Shows "GPU safe max: XXK" under buttons; turns orange when selected size exceeds safe limit
- **`--ctx-size` passed to server**: Only when manually selected (Auto = no flag, server computes)
- **`/props` endpoint**: New `max_safe_context` field in memory info

### KV Cache Sliding Window Fix
- **Removed incorrect cache invalidation**: The old code reset the entire KV cache when prompts exceeded the sliding window size (512 for Gemma 4), causing full re-encoding of the entire prompt on every request
- **Root cause**: The comment claimed sliding window layers "trimmed their KV buffers" — but `KVCache.update()` stores ALL tokens in the buffer and only creates windowed views for decode. `truncate()` is safe since it only updates offsets
- **Impact**: Claude Code agent loop requests with shared 24K-token prefix go from full prefill (~2s) to prefix reuse (~0.5s) — **3-4x faster per iteration**

### Testing
- **`tests/test_anthropic_api.sh`**: 40 integration tests — non-streaming, streaming, tool calling, tool result round-trip, error handling, streaming event order, model name echo, stop_sequences, Anthropic headers
- **`tests/test_kv_cache_sliding_window.sh`**: KV cache reuse validation for sliding window models — measures cache-hit vs cache-miss timing, verifies prefix reuse across multi-turn tool-calling conversations

---

## 2026.4.10 — Deep Agent Loop Reliability

### Tool Call Argument Fixes
- **KV cache reuse after tool calls**: Removed unnecessary full cache invalidation after tool-calling requests — `reuseKVCache()` already truncates to the shared prefix via `cache.truncate()`, correctly discarding stale generated-token entries. Significant performance win in deep agent loops (avoids re-encoding system prompt + tool definitions every round).
- **History windowing fix**: `buildAgentHistory()` now pins the first user message (the original task) even when `.suffix(28)` drops it. Prevents the model from losing task context after ~10 iterations.
- **Progressive tool result truncation**: Older tool results truncated to 500 chars, last 2 kept at 2000 chars — slows context growth by ~75% per old tool result without losing recent context.
- **Generation budget warning**: `clampMaxTokens` warns when remaining tokens fall below 25% of requested `max_tokens`, flagging potential tool call argument truncation.
- **Diagnostic logging**: Logs prompt token count, effective max generation tokens, and context size after every request. Logs byte count of generated text before tool call parsing.

### Tool Calling UX
- **Pre-validation of required params**: Before executing a tool, checks required parameters against the tool definition. Missing params get a detailed error with the expected JSON format (from the tool definition's example), reducing retry rounds.
- **Raw args in UI**: Tool call summary now shows the raw arguments the model generated, so users can see exactly what the model tried to send (instead of just "empty args").
- **Browse URL auto-fix**: `BrowserManager.navigate()` prepends `https://` when the model omits the URL scheme — models frequently send `httpbin.org/html` instead of `https://httpbin.org/html`.
- **Browse tool example fix**: Tool definition example changed from `readText` to `navigate` (the typical first action), preventing misleading error messages.
- **Error message clarity**: Changed "Expected format:" to "Example:" in tool error feedback to avoid implying a single valid format.
- **TestServer nudge fix**: Changed nudge from "Process the tool results above and respond" to match ChatView's "Continue. If the task is done, summarize the result. If not, take the next step."

### Code Quality
- **`sampleTokenLazy` refactor** (generate.zig): Replaced three boolean ownership flags (`scaled_owned`, `topk_owned`, `topp_owned`) with a single `current` variable pattern — each step frees the old intermediate before reassignment. Fixes a memory leak when `temperature=1.0` with top-k/top-p applied.
- **`APIClient.ToolCall.rawArguments`**: New field preserves the raw JSON string from the server, so downstream code can display what the model actually generated.

### Testing
- **Deep agent loop test**: New `tests/test_deep_agent_loop.sh` — 12-iteration tool call regression test via curl, checking finish_reason, argument validity, required keys, and token budget squeeze per iteration.
- **New Swift tests**: `testHistory_ProgressiveTruncation`, `testHistory_FirstUserMessagePinned`, `testHistory_FirstUserNotDuplicatedWhenInWindow`, `testHistory_TruncationAt28PlusPinnedFirst`.

---

## 2026.4.9 — Inference Performance Optimization

### Async Pipeline Overhaul
- **Submit-first pipeline**: Reordered generation loop to build and `async_eval` the next step BEFORE eval'ing the current token — the GPU computes the pending token as a dependency of the next forward pass, making `eval()` return instantly. Matches mlx-lm's `_step(y) → async_eval(next_y) → y.item()` pattern.
- **Fully-lazy token pipeline**: Sampled tokens stay as lazy MLX arrays fed directly into the next forward pass — no GPU→CPU→GPU roundtrip for token IDs between decode steps.
- **Deferred token eval**: Token materialization deferred to the start of the next iteration, giving the GPU maximum overlap time during HTTP streaming and caller processing.

### JIT-Compiled Activations
- **Compiled GELU**: `mlx_compile(shapeless=true)` fuses 8 separate ops (power, multiply, add, tanh, etc.) into a single GPU kernel, matching mlx-lm's `@mx.compile` on `nn.gelu_approx`.
- **Compiled GeGLU**: Fuses `gelu(gate) * up` into one kernel for the MLP block (called 42x per token).
- **Compiled softcap**: Fuses `tanh(x/cap) * cap` logit softcapping into one kernel.

### Memory & Scheduling
- **GPU memory wiring**: `mlx_set_wired_limit` set to `max_recommended_working_set_size` to prevent model weight paging.
- **Periodic cache clearing**: `mlx_clear_cache()` every 256 tokens to reduce memory fragmentation.
- **Dedicated generation stream**: Uses `mlx_stream_new_device` for generation compute.

### Minor Optimizations
- **Eliminated per-call `detectQuantBits`**: Quantization bits read once from config instead of re-detected via shape inspection on every matmul (~378 calls/token saved).

### Results
- **Decode: ~33 tok/s** on Gemma-4 E4B 4-bit (Mac Mini M4 16 GB), matching mlx-lm (Python)
- **Prefill: ~300 tok/s** on 840-token prompt
- **Memory: 4.0 GB** (7% less than mlx-lm's 4.3 GB)
- **Startup: ~2s** (3x faster than mlx-lm, no Python runtime)

---

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

### MLX Core App
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

## 2026.4.3 — MLX Core Major Update

- **Native tool calling UI**: 7 built-in tools (shell, readFile, writeFile, editFile, searchFiles, browse, webSearch)
- **Agent mode**: Automatic ReAct loop with tool execution and result feeding
- **Browser integration**: WKWebView-based browsing, headless operation for background tool use
- **Streaming chat**: SSE parsing with delta reconstruction for real-time responses
- **Multi-session chat**: Persistent chat history with session management

## 2026.4.2 — MLX Core Initial Release

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
