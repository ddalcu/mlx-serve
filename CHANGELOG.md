# Changelog

## v26.5.4 — Speculative decoding (MTP / PLD / Gemma 4 drafter), Settings window, tokenizer fix

- **MTP (Multi-Token Prediction)**: native self-speculative decoding for Qwen3.5/3.6/Qwen3-Next checkpoints that ship MTP weights. `--mtp` flag and per-request `enable_mtp`. Snapshot/restore handles hybrid GatedDeltaNet rollback; tools/logprobs/grammar auto-disable.
- **PLD (Prompt Lookup Decoding) on by default**: model-agnostic n-gram speculative decoding works on every supported architecture (Gemma, Qwen, Llama, Mistral, Nemotron-H, LFM2.5). Up to 1.82× on heavy-echo Gemma-4-E4B, 1.16× on RAG-style retrieval. `--no-pld` to disable.
- **Gemma 4 assistant drafter**: cross-attention drafter using Google's `gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant-bf16` checkpoints. `--drafter <dir>` activates it; 1.98× decode on echo-heavy E4B-4bit (3.0/3 max acceptance). Streaming supported across chat / Anthropic / Responses paths.
- **Adaptive prompt-time gate**: per-request 3-gram repetition score on the prompt disables PLD/drafter on novel content (`spec_gate_threshold = 0.01`). Validated 9/9 on a tuning corpus. Bypass with explicit `enable_pld:true` / `enable_drafter:true` in the request body.
- **Runtime acceptance gate**: mid-decode fallback when actual draft acceptance is below break-even — < 0.30 after 5 attempts for PLD/drafter, < 0.70 after 8 attempts for MTP (binary outcome → separate threshold). Sticky per-request; protects against workloads the prompt-time gate misjudged.
- **Settings window** (MLX Core, Cmd+,): single source-of-truth for server-launch flags (port, ctx-size, log-level, vision, MTP/PLD/drafter, draft lengths) and per-request defaults (max-tokens, temperature, top-p/top-k, repeat/presence penalty, reasoning budget, thinking, per-request spec-decode overrides). Restart banner appears when launch flags change; per-request fields apply on the next chat.
- **Tokenizer correctness fix**: GPT-2 pre-tokenizer rewritten as a priority-ordered state machine matching the reference regex. Four classes of splits now correct — leading-space + letters as one pre-token (` total`), leading-space + punct (` +=`), multi-space runs preceding identifiers (`    total`), and digits as single codepoints (`100` → 1, 0, 0). Old impl perturbed BPE merges on every subsequent word.
- **Markdown rendering**: assistant messages render in a single NSTextView so drag-select spans paragraphs / lists / code blocks / tables. Adds GFM table parsing with column alignment; small in-prompt nudge steers smaller models toward GFM table syntax for plain-chat tabular output.
- **`/v1/models` meta additions**: `model_max_tokens` (architectural cap, independent of `--ctx-size`) and `supports_mtp` (config declares MTP layers).
- **Build**: Swift 5 language mode globally (`-Xswiftc -swift-version -Xswiftc 5`) — required under Swift 6.3 / Xcode 26+ because the pinned `swift-sdk` 0.10.x trips new `SendingRisksDataRace` diagnostics. No-op on the Swift 6.1 CI runner.
- **Tests**: PLD / MTP / drafter byte-equivalence suites (greedy temp=0); streaming-vs-non-streaming byte-equivalence; long-greedy memorized-prompt test that asserts byte-identical first 30 tokens (INT4 float-noise tail documented in CLAUDE.md). New `bench_spec.sh` with `--corpus` and `--gated` modes.

---

## v26.5.3 — Real Sonoma compatibility, CI test gate, dependency pinning

- **Bundled dylibs are now actually Sonoma-compatible.** Switched the release runner from `macos-26` to `macos-14`; Homebrew bottles for `mlx`, `mlx-c`, `webp`, and `libsharpyuv` come out stamped `minos 14.0` instead of `minos 26.0`. v26.5.2 fixed the Zig binary's minOS but the bundled libs still required Tahoe — dyld would refuse them on Sonoma at first launch, surfacing as "Server failed to start" in MLX Core.
- **CI test gate**: `zig build test` and `swift test` now run between build and packaging. A regression that breaks the suite no longer ships.
- **Post-build smoke tests**: `mlx-serve --version` runs against both the freshly built binary and the install_name_tool-rewired CLI artifact, so missing-dylib failures surface before the notarize step burns a submission slot.
- **Homebrew dependencies pinned in `build.zig`**: builds now hard-fail with a clear message if `mlx`, `mlx-c`, or `webp` are below the minimum versions the codebase expects (mlx >= 0.31.2 — the version the v26.4.33 thread-local-stream hotfix targeted).
- **Zig 0.16+ enforced**: `comptime` check at the top of `build.zig` produces "needs Zig 0.16, run brew upgrade zig" instead of a cryptic `StdIo.inherit` enum error on older Zig. Belt-and-suspenders to `build.zig.zon`'s `minimum_zig_version`, which Zig 0.15 doesn't enforce for root projects.
- **`Brewfile`**: declarative dep manifest. `brew bundle install` from a fresh checkout (or in CI) covers `zig`, `mlx-c`, `webp`, `create-dmg`.
- **`workflow_dispatch` version scheme fixed**: now reads the latest `vYY.M.N` release tag and increments N (matching the documented CalVer scheme). Was using `github.run_number`, a global counter, which would have produced versions like v26.5.1234.

---

## v26.5.2 — Sonoma compatibility for CLI binary

- **Fix `mlx-serve` failing to launch on macOS 14 (Sonoma)**: pin `LC_BUILD_VERSION minos` to 14.0 in `build.zig` so binaries built on the `macos-26` (Tahoe) CI runner still load on Sonoma. dyld refuses any image whose minOS is newer than the running OS. MLX Core (Swift) was already fine via `Package.swift`'s `.macOS(.v14)`; only the Zig binary was affected.
- **SDK auto-detection workaround**: setting any non-default target field in Zig disables native macOS framework discovery, so `build.zig` now resolves the SDK with `xcrun --sdk macosx --show-sdk-path` and adds its `Frameworks` dir as a search path. No workflow change needed.

---

## v26.5.1 — OpenAI Responses API + WebSockets, tokenizer arena fix, LM Studio discovery

- **Tokenizer ~30× faster load**: `loadTokenizer` keeps the parsed `tokenizer.json` arena alive and borrows vocab/merge string pointers from it instead of duping per entry; hashmaps pre-sized to skip rehashing. Headline downstream effect: **Qwen3.5-4B prefill 144 → 383 tok/s** (+165%, now ~93% of mlx-lm 0.31.2 reference) on 844-token prompts. Gemma-4-E4B and LFM2.5-350M within run-variance of prior numbers.

- **OpenAI Responses API (`POST /v1/responses`, `GET`/`DELETE /v1/responses/{id}`)**: stateful chains via `previous_response_id`, in-memory `ResponseStore`, streaming SSE with per-event `sequence_number`, schema-conformant envelope (`tools` / `tool_choice` / `text` / `reasoning` / `usage` echo). `experiments/openresponses` compliance suite passes 17/17. Plus `POST /v1/responses/compact` — opaque base64 history blob (`{v:1, msgs:[…]}`) that round-trips back as a `compaction` input item without an LLM call.

- **WebSocket transport on `/v1/responses`**: standard `Upgrade: websocket` handshake, each text frame is a `response.create` JSON message and each SSE event becomes one outbound text frame. New `src/ws.zig` (RFC 6455 framing, server-side). Per-connection `WsLocalCache` for `store: false` responses; no `[DONE]` on success — `response.completed` is the per-response terminator.

- **PDF chat attachments** (MLX Core): drag-drop or paperclip-pick a PDF; PDFKit extracts the text into the message preamble. Encrypted or scan-only PDFs surface a clear error alert instead of silently dropping.

- **LM Studio model auto-discovery** (MLX Core): reads LM Studio's `downloadsFolder` from `~/.lmstudio/settings.json` (falls back to `~/.lmstudio/models`), scans two levels deep for valid MLX models, groups them in the picker under "Other Discovered Models" alongside "MLX-Serve Models". GGUF folders skipped automatically via the existing `.safetensors` check. The Model Browser's "Downloaded" tab still shows only mlx-serve-managed models.

- **Server auto-restarts on model-dropdown change** (MLX Core): switching model while the server is running stops and relaunches with the new model. Fixed `ServerManager.stop()` to detach the dying process's `terminationHandler` + stderr handler so its trailing "Shutting down gracefully…" can't bleed into the new server's log or hijack `status = .starting` into `.error("Failed to start")`.

- **Native NSAlert on download failure** (MLX Core): "Not enough disk space. Need 8.4 GB but only 4.6 GB available." now pops as a modal alert in addition to the inline red text — doesn't get missed when the menu bar popover closes.

---

## v26.4.33 — Hotfix: thread-local streams in mlx 0.31.2

- **Inference now runs on the listener thread.** mlx 0.31.2 made GPU streams thread-local — model weights loaded on the main thread couldn't be evaluated from connection threads, so any chat completion crashed with `MLX error: There is no Stream(gpu, 1) in current thread.`. Removed the thread-per-connection spawn in `server.zig` and handle connections inline. The `inference_mutex` was already serializing the slow path, so this doesn't reduce real concurrency — only quick endpoints (`/health`, `/v1/models`, `/props`) get briefly delayed during generation, which is fine.
- **Transformer uses the current thread's default GPU stream** (`mlx.gpuStream()`) instead of a dedicated stream created at init time. Adds `useCurrentThreadStream()` for any future call sites that need to rebind.
- v26.4.32 fixed the `libjaccl.dylib` bundling issue but still hit this stream issue at the first inference. v26.4.33 is the actual working build.

---

## v26.4.32 — Hotfix: `libjaccl.dylib` not found at startup

- **Bundle all sibling dylibs from `/opt/homebrew/opt/mlx/lib/`**, not just `libmlx.dylib`. mlx 0.31.2 (the version on the macOS-26 GitHub runner) added a new `@rpath/libjaccl.dylib` dependency that we weren't copying — caused the v26.4.31 binary to fail at startup with `Library not loaded: @rpath/libjaccl.dylib`.
- **Add `@loader_path` to `libmlx.dylib`'s rpath** so future `@rpath` sibling deps from mlx resolve cleanly to the bundled Frameworks dir without further workflow changes.
- v26.4.31 had the same MCP + Zig 0.16 changes — this is purely a packaging fix. If you already grabbed v26.4.31 and got the dyld error, just download v26.4.32.

---

## v26.4.31 — MCP Client + Marketplace, Zig 0.16

- **MCP toggle pill**: Purple **MCP** capsule next to Think and Agent in the chat toolbar with an embedded gear icon that opens a marketplace sheet. Works with or without Agent mode.
- **swift-sdk integration**: `MCPManager` spawns each enabled stdio server via `/bin/zsh -lc 'exec npx …'`, wires stdio into `StdioTransport`, and namespaces tools as `<server>__<tool>` so cross-server collisions are impossible.
- **HTTP transport too**: URL-based MCP entries (just `"url": "https://…"`) connect via `HTTPClientTransport` with SSE streaming — no subprocess. Marketplace shows them with a blue HTTP pill.
- **10-server curated catalog**: GitHub, Azure DevOps, DBHub (universal SQL via dbhub.ai), Docker, Kubernetes, Playwright, Slack (Zencoder fork), Notion, Filesystem, Shell — each with inline `SecureField`s for required env vars / args.
- **Claude Desktop config format**: `~/.mlx-serve/mcp.json` follows the `{"mcpServers": {...}}` shape so configs paste straight across. **Source order preserved** through save/load via `OrderedDictionary` + manual outer-object emit + raw-text key-order recovery on load (Foundation's JSON encoder/decoder both shuffle keys via a hash store).
- **Auto-encoded secrets**: New `envEncoded` input kind base64-encodes ADO PATs as `base64("x:<pat>")`. Conditional `argsWhenPresent` lets ADO default to interactive browser auth and switch to PAT mode when the optional field is filled.
- **Live status per row**: Toggle a server in the marketplace and you get instant feedback — yellow "starting" → green dot + "N tools" on success, red dot + tooltip with stderr on failure. Auto-spawns on toggle so the indicator is meaningful without leaving the sheet.
- **Auto-reload on app activate**: Edit `mcp.json` in your editor, switch back to the app, and the marketplace re-hydrates from disk. No close/reopen needed.
- **Pre-flight runtime check**: `command -v <command>` runs in a login zsh before spawn — if `npx` / `docker` / etc. is missing, throws `MCPSpawnError.commandNotFound` with an install hint instead of a 30s dead-wait.
- **Fast-fail on subprocess crash**: `Process.terminationHandler` resumes a one-shot continuation the moment the child exits — docker-mcp dies in 0.6s when the daemon is down, k8s-mcp similar with broken kubeconfig, etc. We surface the captured stderr in the chat warning instead of timing out.
- **Stale errors purge on disable**: Toggling a server off clears its old `startErrors` entry instead of letting it linger in the inline chat warning.
- **Inline chat warnings**: Failed MCP startups show as a warning bubble in chat, not just hidden behind the marketplace gear.
- **Default cwd `~/.mlx-serve/workspace`**: Spawned MCP servers (filesystem, shell, etc.) anchor at the same workspace dir the agent uses by default, with per-entry `cwd` override via mcp.json. New chat sessions inherit it; old sessions saved before this default existed get backfilled on load.
- **Session cwd → MCP cwd**: When MCP servers spawn, they pick up the active chat session's `workingDirectory`. Per-entry `cwd` in mcp.json still wins.
- **Empty-arg fix**: `convertArguments` always returns a (possibly empty) dict so `"arguments": {}` lands on the wire — fixes ADO and other strict-Zod servers rejecting empty calls before auth could fire.
- **Friendly context-overflow error**: Typed `APIError.badStatus` replaces the cryptic `NSURLErrorDomain -1011`; suggests context bump / smaller toolset when the model context is exceeded.
- **Spinner cleared on agent error**: Orphaned streaming bubble no longer keeps `GeneratingIndicator` running forever.
- **Tool-call watchdog**: GCD timer (immune to Swift cooperative-pool saturation from the SDK's hot-spinning message loop) caps tool calls at 90s, terminates the child, and detaches a `client.disconnect()` to resume the pending continuation.
- **mcp.json no longer escapes slashes**: `JSONEncoder.outputFormatting.withoutEscapingSlashes` drops the `\/` legacy HTML-safety escapes, so the file matches what Claude Desktop emits.
- **Zig 0.16 migration**: `minimum_zig_version` 0.15.2 → 0.16.0, new `main(init: std.process.Init)`, `Conn` wrapper bundling `std.Io.net.Stream` + Reader/Writer state, `std.Thread.Mutex/Condition` → `std.Io.Mutex/Condition` with explicit `io` parameter, `mod.linkFramework` for IOKit/CoreFoundation, new `src/io_util.zig` for shared timing helpers.
- **Tests**: 162 Swift unit tests (incl. real `npx -y docker-mcp` integration covering missing-command / missing-package / daemon-down / fast-fail timing, plus key-order round-trip), 210 Zig server tests.

---

## v26.4.30 — Gemma 4 Vision Fix, /v1/models Capabilities, Responses Streaming

- **Gemma 4 vision fix**: `populateUserTurnMarker` encodes the user-turn prefix from each model's `chat_template` at boot, replacing hardcoded Gemma 3 token IDs. Image tokens now insert at the right position; Gemma 4 actually sees attached images.
- **`/v1/models` capabilities**: New `capabilities` array (`chat`, `tool_use`, `streaming`, `vision`, `reasoning`, `json_schema`, `embeddings`), `input_modalities` array, and `meta.architecture`. Model id is now the directory basename so quantization variants are distinguishable.
- **Anthropic `/v1/messages` vision**: Base64 and URL image blocks accepted and routed through the SigLIP pipeline; same-message text + image bundling.
- **`/v1/responses` live streaming**: Reasoning, message, and function-call output items now stream incrementally with proper `delta` / `done` lifecycle events instead of buffering server-side.
- **Browse `extractText`**: New action runs `querySelectorAll(selector)` and returns up to 50 elements joined by `\n---\n`. `readText` now picks `<main>` / `<article>` and strips combobox menus.
- **Schema enforcement repair**: `parseTextFormat` and `parseResponseFormatAlias` accept both flat and nested-`json_schema` shapes on both `text.format` and `response_format` fields — no more silently-dropped schemas.
- **Default port 8080 → 11234**: Avoids conflict with common dev tools.
- **Orphan-process reaper**: `ServerManager` SIGTERMs leftover `mlx-serve` processes holding the target port before launching its own child.

---

## v26.4.28 — Grammar-Constrained JSON Schema Decoding

- **Token-level mask**: `response_format: json_schema` now filters every sampled token against a streaming JSON grammar derived from the schema. Non-conforming output is structurally unreachable, replacing the prior soft prompt-side instruction.
- **Supported subset**: type, properties, required, additionalProperties (defaults false), items, enum, const, min/maxLength, min/maximum, exclusive variants, regex patterns. `anyOf` / `oneOf` relaxed to "any JSON" at branch points.
- **EOS gating**: End-of-sequence masked off until the grammar reports the root value as fully parsed — eliminates premature truncation.
- **Graceful fallback**: Dead grammar states flip the mask to "everything allowed" and log a warning — request still completes.
- **Token-byte cache**: Per-id byte sequences computed once at first use (~50ms for 100k vocab), reused across requests; per-token mask building runs in 1–5ms.
- New modules: `json_schema.zig`, `regex.zig` (Thompson NFA), `json_grammar.zig`, `token_mask.zig`. New integration script `tests/test_json_schema.sh`.

---

## v26.4.27 — Multi-CLI Launcher (Claude Code / pi / OpenCode)

- **Menu-bar dropdown**: Replaces the single Launch button. Detects installed CLIs via login `zsh -l` and shows one entry per installed agent (Claude Code, pi, OpenCode).
- **Smart visibility**: Single button when one CLI is installed, dropdown for 2+, hidden when none.
- **Per-CLI config staging**: pi gets `~/.pi/agent/models.json`, OpenCode gets a dedicated `OPENCODE_CONFIG` in `$TMPDIR` so the user's main config is left untouched.
- **Real model id**: All three launches use the served model id from `/v1/models` instead of a hardcoded alias.

---

## v26.4.26 — Qwen 3.5/3.6 Tool-Call Reliability, Thinking Streaming, Swift Agent Robustness

- **Qwen 3.5/3.6 tool-call repairs**: Walks down nested-name wrappers (`{"name":{"name":{…}}}`), fixes missing `"arguments":` quote/colon, fixes unquoted-key variants. KV cache reset on identical-prompt replay.
- **Thinking-tag streaming**: Handles template-pre-injected `<think>\n` openers via 9-byte look-behind buffer; dual close-tag scan (`</think>` and `<channel|>`).
- **Swift agent watchdog**: 90s SSE inactivity watchdog around the agent-loop consumer, surfaces a clear stall error instead of hanging forever.
- **`failedRetry` flag**: Pad-retry and truncation recovery flag the streamed message instead of removing it — reasoning stays visible in the UI but excluded from API history.
- **Per-tool 30s timeout**: Browse and webSearch capped via task group; BrowserManager `evaluateJavaScript` capped at 25s.
- **Anthropic streaming parity**: Same think-tag handling applied to `/v1/messages` for Claude Code clients.

---

## v26.4.25 — Nemotron-H, LFM2, Qwen3.5 GatedDeltaNet Fixes

- **Nemotron-H Mamba2 SSM**: `A_neg` cast to float32 (BF16 broke decay precision across 42 layers); `time_step_limit` defaults to `(0.0, inf)` matching Python — no more dt clipping with stale config values.
- **Qwen 3.5 GatedDeltaNet**: Pass `ones([dk], bf16)` for parameter-free RMS norm (mlx-c rejects null); SSM state init checks `ssm_state.ctx == null` instead of the prematurely-set `initialized` flag.
- **Qwen 3.6 compatibility**: `qwen3_5_moe` model_type with both GatedDeltaNet and MoE works after the fixes.
- **Bench suite**: `bench.sh` rewrite with deterministic prompts, warmup exclusion, mlx-lm side-by-side reference, `BenchmarkLog.md` for tracking across releases.
- **CalVer auto-increment**: `build.sh` uses `YY.M.N` versioning where N is auto-incremented from the last GitHub release for the current month.

---

## v26.4.22 — Model Browser, Menu Bar Status Icon

- **HuggingFace search**: New Model Browser window with sortable columns (downloads, likes, RAM estimate, last updated), capability badges, RAM-fit indicator, architecture detection.
- **Resume support**: Downloads track `.partial` files and active downloads appear in the Downloaded tab with progress bars.
- **Vision crash fix**: Models with `vision_config` but no vision weights (e.g. text-only quantized Qwen 3.5) return `MissingVisionWeights` instead of crashing.
- **Status-tinted tray icon**: Menu-bar icon turns red when stopped, orange when starting, normal tint when running. `AppState` forwards `ServerManager.objectWillChange` so MenuBarExtra reacts.

---

## v26.4.21 — Vision Pipeline, Prefill Speedup, AgentEngine

- **Gemma 4 SigLIP vision**: Full pipeline — patch embedding, 2D RoPE, clipped linears, position pooling, embedding projection. JPEG/PNG/WebP decode via stb_image + libwebp. KV cache invalidation on image requests so vision features don't get reused.
- **3× prefill speedup (split prefill)**: Prefix pass builds the lazy graph but only KV cache entries are evaluated — MLX skips the `lm_head` matmul over the whole prompt. Last-token pass produces the logits for sampling. Matches mlx-lm; ~1,266 tok/s prefill on long prompts.
- **AgentEngine refactor**: Extracted ~350 lines of duplicated agent logic from ChatView and TestServer into a shared module — history building, tool execution, repetition tracking, overflow management.
- **Tool blocking overhaul**: Arg-aware repetition keys (`listFiles:src` and `listFiles:lib` are different), three-phase warn → soft-block → escalate, write tools exempt.
- **Image attachment UI**: Drag-drop / paste, thumbnails, `ImagePreprocessor` for vision encoder input.
- **Generating indicator**: Animated dual-arc GPU/Memory visualization with live stats and rotating whimsy text.
- **JPEG orientation fix**: `CGImageSource` with `kCGImageSourceCreateThumbnailWithTransform` so camera JPEGs aren't sideways.
- **Welcome window**: First-launch onboarding via direct NSWindow (MenuBarExtra apps don't auto-open SwiftUI scenes).

---

## v26.4.20 — Tool Reliability, Thinking+Tools, Truncation Recovery

- **Tool parameter key order**: Pre-serialized `toolDefinitionsJSON` with guaranteed `path` before `content`; request body splicing bypasses Swift's non-deterministic key ordering.
- **Truncated JSON recovery**: `extractPathFromTruncatedJSON` finds `"path":"..."` even when JSON parsing fails. Improved repair tracks unmatched `{` / `[` openers respecting quoted regions.
- **Thinking + tools fix**: Streaming and non-streaming paths both emit `reasoning_content` for tool-using turns instead of stripping silently.
- **Gemma 4 tool args**: Depth-tracked brace matching for nested objects (`{config:{...}}`) and arrays — was previously falling through to bare-value parsing.
- **Default max_tokens 8192 → 32768**: Prevents tool-call argument truncation for large file writes.
- **Max tokens warning**: `SSEEvent.maxTokensReached` surfaces a clear "Output truncated" message in chat.

---

## 2026.4.12 — MLX Core Rename, Agent Overhaul

- **Rename**: MLX Claw → MLX Core across all source, scripts, CI, docs, and bundle id (`com.dalcu.mlx-core`).
- **`listFiles` tool**: Dedicated file listing with glob and recursive traversal — system prompt steers the model toward dedicated tools instead of shell equivalents.
- **150 max iterations** (up from 30); token-aware history fitting; per-tool context caps; tool result overflow saved to `~/.mlx-serve/tool-output/` with truncated preview.
- **System prompt redesign**: Hardcoded base + additive user customization; explicit readFile → editFile workflow; structured error-recovery section.
- **Tool enhancements**: `readFile` shows `N| text` line numbers; `searchFiles` uses ripgrep with `include` / `context` / `maxResults`; `writeFile` unescapes double-escaping from smaller models.
- **API client**: Retry with exponential backoff on network errors (was single-retry).
- **Workspace context injection**: Working directory listing auto-injected each iteration so the model knows what files exist without calling `listFiles`.

---

## 2026.4.11 — Anthropic API, Claude Code, KV Cache Fix

- **`/v1/messages` Anthropic compat**: Full conversion of Anthropic content blocks (text, tool_use, tool_result, thinking), `input_schema` → `parameters`, named SSE events, stop_reason mapping.
- **Claude Code launcher**: "Launch Claude Code" button opens Terminal with `ANTHROPIC_BASE_URL` configured; binary detection via login shell PATH.
- **GPU memory preflight**: Estimates peak attention + KV memory with 20% margin, rejects with HTTP 400 instead of crashing on Metal C++ exceptions. Dynamic Metal limit from `sysctl hw.memsize`.
- **Context size auto-detection**: Default context computed from GPU memory at startup; new Auto / 16K / 32K / 64K / 128K UI presets.
- **KV cache sliding window fix**: Removed incorrect cache reset for prompts > sliding window. 3–4× faster Claude Code agent loops with shared 24K-token prefix.

---

## 2026.4.10 — Deep Agent Loop Reliability

- **KV cache reuse after tool calls**: Removed unnecessary full invalidation — `cache.truncate()` already discards stale generated-token entries. Major perf win in deep loops.
- **History windowing**: First user message pinned even when `.suffix(28)` would drop it. Progressive truncation: older tool results to 500 chars, last 2 to 2000.
- **Generation budget warning**: Logs when remaining tokens fall below 25% of `max_tokens` — flags potential argument truncation.
- **Pre-validation of required params**: Detailed error with example JSON instead of forcing the model to retry blind.
- **Browse URL auto-fix**: `BrowserManager.navigate()` prepends `https://` when scheme is missing.
- **`sampleTokenLazy` refactor**: Replaced 3 boolean ownership flags with a `current` variable pattern — fixes a memory leak when `temperature=1.0` with top-k/top-p applied.

---

## 2026.4.9 — Inference Performance Optimization

- **Submit-first pipeline**: Build and `async_eval` next step BEFORE eval'ing current token — `eval()` returns instantly. Matches mlx-lm's `_step → async_eval → y.item()` pattern.
- **Fully-lazy token pipeline**: Sampled tokens stay as lazy MLX arrays into the next forward pass — no GPU↔CPU roundtrip between decode steps.
- **JIT-compiled activations**: `mlx_compile(shapeless=true)` fuses GELU (8 ops → 1 kernel), GeGLU, and softcap.
- **GPU memory wiring**: `mlx_set_wired_limit` set to `max_recommended_working_set_size` to prevent weight paging.
- **Periodic cache clearing**: `mlx_clear_cache()` every 256 tokens reduces fragmentation.
- **Results**: Decode ~33 tok/s on Gemma-4 E4B 4-bit (M4 16GB), matching mlx-lm. Memory 4.0 GB (7% less than mlx-lm). Startup 3× faster — no Python runtime.

---

## 2026.4.6 — Gemma 4 MoE, Jinja Upgrade, Tool Calling Overhaul

- **Gemma 4 MoE (26B-A4B)**: Sigma-MoE routing, separate shared/routed expert branches, 5 feedforward norms, GeGLU activation.
- **Gemma 4 E2B/E4B**: Per-Layer Embeddings (PLE) with gated projection and per-layer input scaling. ProportionalRoPE for global attention. K=V attention. Sliding window with full prefill / windowed decode views.
- **Per-weight quantization detection**: Auto-detects quant bits per weight instead of using a global default — fixes 8-bit shared expert in a 4-bit model.
- **Jinja upgrade**: Replaced jinja.hpp with llama.cpp's Jinja engine. Fixes empty tool-call args (`{command:{}}`), missing parameter types, and broken tool-message transformation.
- **Tool calling reliability**: Gemma 4 double-brace unwrapping; full SSE arg deltas in single chunk; KV cache invalidated after tool-calling requests; user nudge after tool results for models that need it.
- **Thinking with tools**: `<|channel>thought` no longer streamed as visible content; `<|channel>` and `<channel|>` tags stripped; partial-tag detection prevents premature flushing.
- **MLX Core test API**: Port 8090 with REST endpoints (`/test/start`, `/test/chat`, `/test/agent`, etc.) for automated testing.

---

## 2026.4.5 — Prompt-Based Skills, Resumable Downloads

- **Prompt-based skills**: User-defined agent capabilities via `~/.mlx-serve/skills/*.md` with YAML frontmatter (name, description, trigger keywords).
- **Resumable downloads**: Streaming writes to `.partial` files, Range header support for resume, 3 automatic retries with backoff.
- **Disk space safety**: Pre-check available space before large downloads.
- **SkillManager**: Scans skills directory on each agent loop, re-reads when directory modification date changes.

## 2026.4.4 — KV Cache & Tool Calling Fixes

- **KV cache corruption fix**: Invalid suffix cache invalidation, SSM state reset.
- **Tool calling reliability**: Improved parsing, agent harness stability.
- **App bundle packaging**: Removed Bundle.module dependency, fixed codesigning.

## 2026.4.3 — MLX Core Major Update

- **Native tool calling UI**: 7 built-in tools (shell, readFile, writeFile, editFile, searchFiles, browse, webSearch).
- **Agent mode**: Automatic ReAct loop with tool execution and result feeding.
- **Browser integration**: WKWebView-based browsing, headless operation for background tool use.
- **Streaming chat**: SSE parsing with delta reconstruction.
- **Multi-session chat**: Persistent history with session management.

## 2026.4.2 — MLX Core Initial Release

- **Swift macOS menu bar app**: Server management, model selection, chat interface.
- **Server lifecycle**: Subprocess launch/termination with stderr capture.
- **Model discovery**: Local model scanning from `~/.mlx-serve/models/`.

## 2026.3 — Embeddings, Reasoning, Jinja

- **Embedding support**: BERT and encoder-only models via `/v1/embeddings`.
- **Reasoning budget**: `--reasoning-budget` CLI flag to limit thinking tokens.
- **Jinja_cpp integration**: Replaced vibe-based Jinja (macros caused infinite loops).
- **Qwen3.5 MoE support**: GatedDeltaNet linear attention, shared expert routing.
- **TUI status bar**: Live CPU, memory, GPU metrics.

## 2026.2 — Initial Release

- **Zig native server**: OpenAI-compatible HTTP API on Apple Silicon.
- **MLX-c FFI**: GPU-accelerated tensor operations via Apple's MLX C API.
- **Model support**: Llama 3, Mistral, Qwen 3.
- **BPE tokenizer**: SentencePiece and byte-level BPE.
- **Streaming generation**: SSE-based real-time token delivery.
- **KV cache reuse**: Prompt prefix matching across requests.
- **Sampling**: Temperature, top-p, top-k, repeat penalty.
