# Server, HTTP & streaming lifecycle — war stories (moved out of CLAUDE.md)

Full histories: live failures, measurements, diagnosis ladders, dead ends. The distilled RULES live in the root CLAUDE.md "Rules" section — when a rule changes, update the story here too. New gotchas in this domain: add the 1-3 line rule to root, the full story here.

### A `seed` that only the synchronous sampler read (seeded replies flipped between identical requests)

`integration_test.sh`'s "same seed produces same first token" went red on the v26.8.11 release run: `Ephemeral` vs `**Ephemeral**` with identical top-2 logprobs (gap 0.75 nats, not a tie). The lazy decode sampler (`sampleTokenLazy`, every serial/batched/spec site) passed a null key to `mlx_random_categorical`, i.e. MLX's global RNG; only the synchronous `sampleToken` (the `logprobs` path) built a key from `seed`, and it built the SAME key every step, so a seeded reply was a single coin flip replayed. The test had passed for months on an 82/18 draw. Fix: `seedKey(sampling)` mixes `seed` with a per-draw index (`SamplingParams.draw`); `Generator.sampleLazy` is the one lazy sampler a slot calls and advances the index; init paths hand the Generator `draw = 1` after drawing t1. Bar: seeded replay identical at temp 1.0 across cold and prefix-cache-hit requests; no seed still varies.

### A client-supplied path handed straight to mlx is a one-request server kill (lora_path)
Found by a test that expected a 400 and got `000` — curl couldn't complete, because the server was gone. `POST /v1/images/generations` with `{"lora_path":"/tmp/nope.safetensors"}` flows into `lora.loadFile` → `mlx_load_safetensors`, which for a missing file raises an MLX error; mlx-c errors are FATAL, so the process dies. Log's last line is `MLX error: [load_safetensors] Failed to open file …` and nothing after it. This isn't a MageFlow issue — it's every image backend, and every client on the box loses its connection because one request named a moved or mistyped adapter. The path check that existed (`isAbsolute`, added for the `openFileAbsolute` UB class) proves the shape of the string, not that a file is there. Fix: open + stat before mlx sees it, and require a REGULAR FILE — a directory opens fine and would die one layer deeper — returning `error.BadLoraPath` → the existing 400. General rule: any request-supplied path that flows into an mlx loader must be validated on OUR side of that boundary, the same way `textGenRejectReason` 400s before prefill rather than letting a null transformer deref take the server down. Guards: two `loadFile` unit tests (missing file, directory) and the LoRA case in `tests/test_mageflow_edit.sh`. Multi-LoRA (`lora_paths`, an array) goes through the same `loadFile` per entry in `ImageEngine.setLoras`/`VideoEngine.setLoras` — a bad path anywhere in the array 400s before any adapter in that request attaches (partial stacks never install; `lora.Stack.deinit` unwinds whatever loaded before the failing entry).

### An error message that quotes a field value is not a JSON string (media-gen 400 bodies)
Found live while checking that a MageFlow txt2img checkpoint correctly refuses an edit request: the 400 came back as `{"error":{"message":"instruction editing (mode:"edit") requires a FLUX.2 or Mage-Flow-Edit model"}}` — raw double quotes inside a JSON string, so every client sees a parse error instead of the (perfectly good) explanation. The Zig source reads `"… (mode:\"edit\") …"`, which is a Zig escape producing a real `"` byte; `gen.sendError` then interpolated it with `{s}` straight into a JSON body. Six messages in `gen.zig` had it (`'mode' must be "edit" or "variation"`, the edit/variation gates, `'ref_images' requires mode:"edit"`, the content-filter refusal), and the SSE variant shared the flaw. A second failure hid behind the same line: both senders build into a fixed 256-byte buffer with `bufPrint(...) catch return`, so a message longer than the buffer sent NO body at all — a bare status code with an empty payload. Fix is at the SINK, not the literals (a future message must not be able to reintroduce it): `gen_sse.jsonEscapeMessage(out, msg)` escapes `"` `\` and the control bytes, maps other sub-0x20 bytes to a space, and TRUNCATES to fit while backing off to a UTF-8 boundary so the tail can never be a torn sequence; both `gen.sendError` and `gen_sse.sendError` route through it into a 640-byte body buffer. Pinned by a hermetic test that feeds the exact live message through a real `std.json` parse. Same class as the tool-calling `appendJsonString` rule — the mistake there is trusting model output, here it's trusting your own literal; both are just bytes going into a JSON string.

### A buffered streaming surface must beat on SOCKET SILENCE, not on token arrival (client idle-timeout class)
Every streaming surface buffers generated tokens while it might be looking at a tool call (`chat.streamShouldBufferForTools`) or an unclosed thinking block (`chat.streamThinkGate` → `.hold_thinking`); `/v1/responses` buffers a tool-active request outright (`if (active_has_tools) continue;`). During that span the handler emits NOTHING. The keepalive used to fire only on the `.idle` arm of `ts.nextOrIdle` — i.e. only while WAITING for the first token (long prefill) — so once tokens started flowing into a buffer the socket went dead silent for the whole tool call. **Tokens flowing ≠ bytes flowing**, and only bytes hold off a client's idle-body timeout. Live failure 2026-07-08: a pi agent session (Node `fetch` → undici, default `bodyTimeout: 300_000`) building a JS game lost two ~5-minute `write` calls to `TypeError: terminated` / `BodyTimeoutError` — ~10 minutes of 27B GPU work discarded, twice, and the agent never learned why. Reproduced exactly: old binary dies at 301.6 s having received 1 chunk / 267 bytes; fixed binary streams a 612 s generation to completion with 122 keepalives and a 5.0 s max gap. Symptom signature: a client-side `terminated` / read-timeout at almost exactly the client's idle timeout, `chunks=1` before it, the server log showing the request later completing normally (the server never noticed) or a `[cancel] client disconnected` line one keepalive later.
- Fix: `Conn.heartbeat` (`server.StreamHeartbeat`) is stamped by `Conn.writeAll`/`writeAllNoFlush`/`flush` — the only places bytes reach the socket — and every token loop calls `beatStreamKeepalive(stream, .sse_comment | .anthropic_ping)` once per iteration, at the BOTTOM of the loop (so all branches, including the ones that wrote nothing, are covered) or before an early `continue` (`/v1/responses`). It emits only when `Conn.keepaliveDue()` (no bytes for `STREAM_KEEPALIVE_MS` = 5 s), so a normally-streaming request pays one timestamp per token and sends nothing extra. WS transports no-op both senders (a raw comment would corrupt framing) and are stamped anyway.
- **Rule: liveness is a property of the SOCKET, never of the generator.** Any new streaming surface, or any new branch that swallows a token into a buffer, must beat once per loop iteration. Never gate the keepalive on "no token available".
- `StreamHeartbeat` is the mirror of `generate.StallClock`: StallClock protects the SERVER from a wedged model (silence = no new *tokens*), StreamHeartbeat protects the CLIENT from a wedged-looking socket (silence = no new *bytes*). Confusing the two is what produced the bug.
- Guards: `tests/test_stream_keepalive.sh` (class guard — asserts for chat + messages + responses that the max inter-chunk gap stays under 15 s across a long buffered tool call, that a keepalive/ping actually arrived, and that the tool call still parses with valid JSON args so the injected bytes never corrupt the stream; SKIPs when the generation was too short to exercise the buffer) plus the `StreamHeartbeat` unit tests in server.zig (verified red-on-revert: all three surfaces FAIL with `max_gap ≈ 17.8s, keepalives=0`).
- KNOWN GAP: the Ollama surface is still exposed. `Conn.writeAll` feeds `ollama_sink`, whose SSE re-framer DROPS comment lines (`ollama.zig` `if (line[0] == ':') continue;`), and NDJSON has no comment/ping form — so a buffered tool call over `/api/chat` still writes nothing. Fix (if an Ollama client ever reports it): translate the keepalive comment into an empty-content `{"message":{"role":"assistant","content":""},"done":false}` line in the sink.
- Related but NOT the same bug: pi's `~/.pi/agent/models.json` declared `contextWindow: 32768` for a model whose server advertises `meta.context_length` ≈ 96k, so pi's own `max_tokens` budget collapsed late in the session and its `write` calls truncated mid-argument — surfacing as our (deliberate) truncation salvage: tool name recovered, `arguments: {}`, `finish_reason: "length"`. A client that validates args against the schema instead of honoring `finish_reason: "length"` reads that as a malformed call. Clients should read `context_length` off `/v1/models`.

### Shutdown race: drain connection threads before `Scheduler.deinit` (SIGSEGV in `complete`)
Per-connection threads are spawned in `server.serve`'s accept loop. On shutdown the accept loop breaks and `serve` returns, firing `defer scheduler.deinit()` — which frees the slot queues (`pending`/`decoding`/`cleanup_queue`) on the assumption that "all conn threads called `complete` properly". They hadn't: a conn thread still inside `Scheduler.complete` (touching those very lists) raced the free → use-after-free SIGSEGV (crash report `mlx-serve-2026-06-20-141700.ips`: thread 0 in `Scheduler.deinit`/`Thread.join`, thread 13 in `Scheduler.complete`; null-deref at +0x18). Triggered by a shutdown/model-switch while a stream was in flight. Fix (three parts):
- `server.serve` tracks live conn threads in an atomic `active_conn_threads` (inc before spawn, dec in `handleConnectionThread`'s first-declared `defer`). After the accept loop it calls `scheduler.cancelAllInFlight()` then **waits for the counter to reach 0** (bounded ~30s) before returning — so `deinit` always runs after every `complete()` has finished.
- `Scheduler.cancelAllInFlight()` sets `cancel()` on every pending+decoding slot so blocked readers wake.
- `Slot.waitNext`/`waitNextTimeout` now return `.done` when `cancelled` is set — previously `cancel()` only broadcast, so a reader blocked in `waitNext` never woke on cancel and the drain could never complete. The inference thread is still alive during the drain (deinit joins it only after `serve` returns), so cancelled slots settle promptly.
- Smoke test `tests/test_shutdown_midstream.sh` (SIGTERM during concurrent streams → clean exit, never rc=139). NOTE: the race is timing-sensitive and the plain-SIGTERM test does not deterministically reproduce it on the old binary (the live crash was a messier model-switch/relaunch) — the fix is correct by construction (deinit cannot run until conn threads drain), the test is a regression smoke guard, not a red-on-revert proof. **Rule: every spawned thread handle must live until `join()` or `detach()`; detached workers touching shared state still need an explicit lifetime drain before teardown.**

### Dropped connection-thread handles retain one stack mapping per request
The shutdown counter fixed the lifetime race above but did not reap pthread resources. The accept loop discarded each successful `std.Thread.spawn` return value without calling `join()` or `detach()`. Returning from `handleConnectionThread` ends execution, but a joinable pthread retains its stack mapping and kernel bookkeeping until it is reaped. In a long-running Claude Code workload this looked like a model leak: E2B QAT-4bit stayed near 4–5 GB of MLX-active memory while process footprint climbed toward 10 GB and never fell. The decisive capture showed `/props` `cache_bytes` at only 2.9 MB, 155,786 stack regions using about 2.43 GB, and about 1.01 GB of page tables; 300 requests added 301 stack regions.

The fix keeps the existing per-connection concurrency model and calls `conn_thread.detach()` immediately after a successful spawn. `active_conn_threads` still provides the shutdown lifetime barrier; detach only tells pthreads to reclaim resources automatically after return. Lowering `MLX_SERVE_CACHE_LIMIT`, changing the 8192-token KV growth cap, or calling `mlx_clear_cache()` more often cannot release pthread stacks. Guard: `each connection thread handle is detached after spawn` source-scans the accept-loop span so the handle cannot silently become discarded again.

### A text-gen request routed at a non-text model must 400 BEFORE prefill (one-request server kill)
Live SIGSEGV 2026-07-06 (`mlx-serve run <flux dir>`): a chat request whose resolved model is a MEDIA entry has only the gen stub CPU state — the empty stub tokenizer yields `0 tokens`, prefill derefs `transformer == null`, and the WHOLE process dies. Any client (remote included) naming a media/encoder model on a chat surface could kill the server. The guard is `server.textGenRejectReason` (pure, hermetically tested) applied at ALL text-gen surfaces — `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/responses` (POST + WS upgrade), `/api/chat`, `/api/generate` — TWICE: a pre-`ensureLoaded` peek (so naming an unloaded 15 GB media stub doesn't cold-load just to earn its 400; detects via discovery `arch_hint`) and the post-load authoritative check (engine slots / `is_encoder_only` / ready-with-no-LM catch-all — `--model` primaries carry no hint, so the peek alone has gaps). Rules: (1) a new text-gen SURFACE must call the same guard (extend `isTextGenRoute` for the peek); (2) a new MODALITY is covered automatically via its engine slot + `modalityFromType(arch_hint)` — keep both in sync when adding one; (3) the same classification feeds `mlx-serve run`'s preflight (`model_discovery.classifyModelPath`/`ModelKind`) and the `list` TYPE column — one taxonomy, three surfaces. Guards: `textGenRejectReason`/`isTextGenRoute` unit tests (server.zig), `modelKindFromType`/`classifyModelPath` tests (model_discovery.zig), and the 4b case in `tests/test_unified_gen.sh` (chat + /v1/messages at a RESIDENT image model → 400s, server alive).

### Auto-context is PINNED at load, with headroom (clients budget against it)
`getEffectiveContextLength(config)` resolves in three steps: explicit `--ctx-size` (`server_config.max_context_size`) wins; else the model's `pinned_context`, frozen once by `pinAutoContext` (at `serve()` startup for the `--model` primary, and right after each on-demand `ensureLoaded`); else — for a discovery stub that was never loaded — a fresh `autoContextFor`. Pre-2026-07-08 there was no pin: the value was recomputed from LIVE memory on **every request**, so the number `/v1/models` advertised drifted with system load (measured 92,387–94,883 across one session; 71,610 with a second 27B resident). That is fatal for agent CLIs, which read `meta.context_length` **once**, bake it into a config file, and budget their own `max_tokens` against it forever.
- `autoContextFor(config)` = `min(safeAutoContext(computeMemoryContext(config)), max_position_embeddings)`. The `auto_ctx_safety_pct` (85%) margin applies to the **memory** ceiling ONLY, before the model-max clamp — a 131,072-token checkpoint that fits comfortably in RAM keeps all 131,072 rather than being shaved to 111k. Pinned by the `autoContextFor: the safety margin applies to MEMORY, never to the model's own max` test (red on revert). `computeMemoryContext` passes `max_pos = 0` into `safeContextForBudget` so the clamp happens exactly once, outside the margin.
- The margin exists so the prefix cache can fill, a second model can load beside this one, and another app can take RAM without pushing us into the uncatchable Metal OOM below. `checkAttentionMemory` (the per-request prefill guard) deliberately stays DYNAMIC — it is the OOM guard and must see current pressure.
- Consequence: a server that starts while something big holds RAM pins low for its whole life. A restart re-pins. That is the accepted trade for a stable advertised context.
- `clampMaxTokens(max_tokens, prompt_len, effective_ctx)` and `omittedMaxTokensDefault(effective_ctx)` take the context EXPLICITLY. Both used to branch on `server_config.max_context_size`, which is set only by `--ctx-size` — so under auto-context the server never clamped a client's `max_tokens`, never emitted the "generation budget squeezed" warning, and an omitted `max_tokens` silently capped at **4096** (same class as the 256 default it replaced). Rule: never gate context behavior on `server_config.max_context_size`; ask `getEffectiveContextLength(config)`.
- Client side: `app/Sources/MLXServe/Services/AgentBudget.swift` derives `(context, output)` from `ModelInfo.contextLength` and `AgentConfigs` writes them into `~/.pi/agent/models.json` (`contextWindow`/`maxTokens`), the opencode provider config (`models.<id>.limit.{context,output}`), and Claude Code's `CLAUDE_CODE_MAX_OUTPUT_TOKENS` (Claude Code has no context-window env var). These were hardcoded to `32768`/`8192`, which is what actually killed long pi sessions on a 94k-context model. The advertised context is declared **verbatim** — the server already reserved 15%, so a second client-side margin double-counts it AND makes the CLI report a different number than Settings shows (opencode said 75K where the server said 77K). Guard: `AgentBudgetTests`.
- UI: Settings → Context size shows three counts that are easy to confuse — **Model max** (`max_position_embeddings`, architectural), **GPU-safe max** (`/props` `maxSafeContext`, what memory could hold *now*), and **In use** (`meta.context_length`, the pinned value actually enforced and handed to agent CLIs). `ContextSizeDisplay` owns the formatting + the one help string, shared with `ServerOptions.serverFlagFields["ctxSize"].explainer` so the two descriptions of "Auto" cannot drift. The shipped copy claimed Auto "uses the model's declared maximum" — it never has. Guard: `ContextSizeDisplayTests`.

### Auto-context budget + the misleading libllama OOM backtrace
A runtime Metal OOM during MLX generation (`[METAL] Command buffer execution failed: Insufficient Memory`) prints a backtrace whose top frames are `libllama.dylib` (`ggml_print_backtrace` / `ggml_uncaught_exception`) — even for a pure-MLX model. That's a RED HERRING: libllama installs a global `std::set_terminate` handler at load (for GGUF support), so it prints the trace, but the throw is from `libmlx` (`mlx::core::gpu::check_error`). Don't chase a GGUF/llama bug — it's MLX exceeding the GPU working set. The auto-context budget (`computeMaxSafeContext` → `safeContextForBudget`, server.zig) must therefore: (1) ceiling = `max_recommended_working_set_size` (`getGpuWorkingSetLimit`), NOT `hw.memsize × 0.75` (`getMetalBufferLimit`) — the latter over-estimates the real limit on small-RAM Macs (16 GB: 12 GB vs ~11.9 GB recommended); (2) reserve the FULL hot prefix cache budget (`prefix_cache_mem_bytes`, default 2 GB) up front — it fills over an agentic session, so an auto-ctx computed against an empty cache (24k on a 16 GB Mac) later collides with the filled cache + a large cold MoE prefill and crashes. `checkAttentionMemory` (the per-request prefill guard) shares the same `getGpuWorkingSetLimit` ceiling. When the budget tightens, an oversized prompt hits the graceful `400 "Prompt exceeds maximum context length"` gate (all four HTTP paths) instead of the process-killing Metal allocator. **(3) — the ceiling must also see EXTERNAL memory pressure (#64, 2026-07):** `getGpuWorkingSetLimit()` is a STATIC device max (128 GB Mac → 115 GB) that assumes the whole GPU working set is MLX's to claim; it is blind to memory held by OTHER processes (the field crash: a Claude Code session running a docker-compose stack — firecrawl/rabbitmq/postgres/playwright — held tens of GB, so the guard budgeted 115 GB, admitted a 90 K-token MoE prefill, and Metal OOM'd). Both guards now budget against `currentGpuMemoryCeiling(active) = min(getGpuWorkingSetLimit(), mlx_active + mlx_cache_memory + getAvailableMemBytes())` — capping the static max by what's PHYSICALLY reachable now (MLX's own footprint + free system RAM via `status.getAvailableMemBytes`, which counts wired+compressed+internal-anon, so docker's pages tighten it). Idle machines see `mlx_active + cache + free ≈ static max` → no auto-ctx regression (verified: 128 GB Mac ctx 169516 idle → 55787 under a 55 GB hog, oversized prompt then 400s, server stays alive). The OOM is NOT catchable — the throw is async on a Metal completion-handler GCD thread (`addCompletedHandler`) via `std::terminate`, so PREVENTION (a tighter guard) is the only lever, not a try/catch. Pure-helper unit tests in server.zig (`physicalMemoryCeiling …`, `safeContextForBudget …`). NOTE: the per-token working-set term still under-models a batched MoE prefill spike — if OOMs persist on ≤16 GB, fall back to `--ctx-size <N>` + `--prefix-cache-mem 512MB`.

### Request timeout is a STALL timeout, and a truncated generation must keep finish_reason "length" through tool-call parse
Two coupled rules from one live failure (2026-07-03, Qwen3.6-27B agent writing a website): the model one-shot ~33KB `writeFile` calls (~8-10K tokens ≈ 5 min at 30 tok/s); the old wall-clock `--timeout` (default 300s, measured from request start) guillotined every round that ran a few seconds long, mid-tool-call; `parseToolCalls`' truncation salvage then recovered a name + path-only/`{}` call; and the tool-parse sites OVERRODE the generator's `finish_reason="length"` with `"tool_calls"` — so the app saw a "complete" call with no content, told the model IT forgot `content`, and the model re-emitted the same doomed mega-call. Three of six writeFile rounds (≈15 min of GPU) were silently discarded. Diagnosis signature: failed agent rounds whose `completion_tokens / tok_s ≈ the timeout` exactly, while raw speed looks nominal; app-side `tool-calls.log` shows `EMIT rawArgs={}`/path-only after a full-length generation. Rules: (1) `--timeout` counts seconds WITHOUT a new token (`generate.StallClock` — progress detected from `generated_ids.len` at the check site, so every decode path resets it without instrumentation); a request that keeps producing never times out. (2) Every site that sets a tool-call finish reason goes through `server.toolCallFinishReason(pre_parse)` — "length" survives the parse (all four surfaces), so clients' truncation recovery (the app's chunk-and-retry nudge; APIClient emits accumulated calls on "length" too) actually fires. (3) `AgentPrompt.outputBudgetGuidance` is a SCARCITY warning and is emitted ONLY when the effective budget is tight (< `outputBudgetGuidanceThreshold`, 12288 — a one-shot file write measured live runs 8–10.7K tokens); roomy machines get NO section at all, because an honest "~419430 tokens per response" on a 1M-ctx machine reads as an invitation to one-shot mega-calls and OVERRIDES the writeFile description's ~200-line chunking convention (two prompt layers in conflict → the specific number wins). Guards: `toolCallFinishReason` + `StallClock` unit tests, `testOutputBudgetGuidanceOnlyAppearsWhenBudgetIsTight`. Harvest aid: `MLX_SERVE_RAW_DUMP_FILE=<abs path>` (with `--log-level debug`) writes the FULL pre-parse text of streamed tools requests — the inline debug dump caps at 4KB.

### A transient mDNS hiccup must never evict a live LAN peer, and dead dns_sd refs must revive (peer-table flap class)
Live 2026-07-19 (two-Mac session, app proxying chat to `gemma-4-e4b-it-4bit@Davids-MacBook-Pro`): chats through the LAN proxy ALTERNATED between success and `404 "LAN peer for this model is offline (waited 15 s)"` within one session, while the peer Mac's server stayed up and continuously advertising the whole time. The user experienced it as "depending on the MCP/Agent toggle state it works or 404s" — the toggles were innocent (both body shapes carry the same `server.chatModelId` and route identically); what correlated was TIMING: each toggle-then-send landed moments after an mDNS hiccup. Two structural holes:
- **`resolveAndInstall` evicted the peer on ONE transient failure.** Resolve timeout (3 s), no-IPv4, or an unreachable fetch each called `removePeer` instantly ("so stale entries never linger") — but a busy mDNSResponder, a 3 s resolve timeout while the peer's GPU is pinned by a model load, or interface churn (the Agent Sandbox's VZ NAT bridge or a docker bridge appearing/vanishing triggers per-interface mDNS remove/add storms) all produce exactly one such failure against a LIVE peer whose cached `ip4:port` still tunnels fine. The next chat then hits `peer_unknown` → the 15 s wait races a browse thread that itself serializes 3+3 s resolve dances per known service → often 404. Fix: `resolveAndInstall` failure paths never touch the peer table; `attemptKnown` owns ALL removal via the pure `knownFailureAction(fails)` policy — an installed peer survives `PEER_DROP_FAILS − 1` (= 2) consecutive failures (~20-30 s grace at the 10 s refresh cadence), `KNOWN_MAX_FAILS` (24) still forgets the service. A genuinely-dead peer now delists in ~2-3 refresh cycles instead of instantly; a tunnel that picks it during the grace window answers 502 honestly. Guard: `lan: transient resolve failures retain a live peer; only persistent failure drops it` (lan.zig) + the `B drops the peer's models once it goes offline` / `chat fired during peer restart` cases in `tests/test_lan_share.sh` (timing loops already cover the grace).
- **A dead dns_sd ref was permanent.** `DNSServiceProcessResult` failing on the browse ref deallocated it and left it null FOREVER (discovery silently off for the process lifetime — the field signature: a server sharing 6 models that lists ZERO `lan_peer` entries for 20+ minutes while the peer is up and advertising); a failure on the ADVERTISE ref wasn't handled at all (dead fd → poll hot-spin, registration gone, peers see this host vanish while it keeps serving); an initial `startAdvertise` failure with discover off never even spawned the browser thread, so nothing could ever retry it. An mDNSResponder restart (macOS update, daemon crash, sleep/wake) invalidates every ref at once and used to hit all three. Fix: the browser loop REVIVES — browse and advertise refs are re-created every `REVIVE_INTERVAL_MS` (5 s) while their role (`l.discover` / `l.share != null`) wants them; POLLHUP/ERR/NVAL and ProcessResult errors tear the ref down with a warn (`dns_sd browse/advertise connection lost`) instead of spinning; `Lan.start` spawns the thread whenever sharing was REQUESTED, not only when the first registration succeeded.
Symptom signatures for the class: alternating success/`peer offline` 404s on consecutive requests to a peer that is provably up; a discovering server whose `/v1/models` shows no `lan_peer` rows while `dns-sd -B _mlxserve._tcp` sees the advertisement; `[lan] resolve timed out` / `no IPv4` debug lines immediately preceding a user-visible 404. Related but NOT bugs: a peer Mac asleep IS offline (honest 404 until wake re-announces); a peer booted via bare `mlx-serve --model X --serve` (no `--lan-share`) advertises nothing, and one booted without `--model-dir` shares only its primary — "The LAN peer no longer shares this model" is then the truthful answer.

### Ownership decided by CONTENT equality leaks the honest empty case (sentinel-by-content class)
Found 2026-07-19 by the integration run's SafeAllocator right after adding the `reasoning_effort` opt-in: both non-streaming text formatters used `const escaped_text = jsonEscape(...) catch "\"\"";` with `defer if (!std.mem.eql(u8, escaped_text, "\"\"")) allocator.free(escaped_text);`. The defer decides ownership by comparing CONTENT against the OOM-fallback literal — but escaping a legitimately EMPTY string also yields `""`, and that one IS allocated. Every request whose visible content is empty (an all-reasoning generation with thinking enabled, an empty completion) leaked its 2-byte escape. Invisible in normal runs; the debug allocator flags it instantly. Fix: `jsonEscapeOrEmpty` returns `{slice, owned}` — ownership by PROVENANCE (did the fallback fire), never inferred from what the bytes look like. Same shape as the LAN `\/`-canonicalization class: any time a sentinel VALUE doubles as a legal payload, the check must key on where the value came from, not what it equals. Guard: `jsonEscapeOrEmpty: escaping an empty string is OWNED` (server.zig, std.testing.allocator fails on leak) + zero `leaked` lines in the integration-test server log.

### A READY model must never advertise LESS capability than its unloaded stub (empty-caps class, second bite)
Live 2026-07-21 (two-Mac LAN session): the app tray showed "No models yet" while the user was actively chatting on the peer's DeepSeek-V4-Flash GGUF — the loaded model itself rendered `capabilities:[]` in `/v1/models`. The ready path gated `has_chat` on `chat_config.chat_template.len > 0`, but embedded-engine GGUFs (ds4/llama) can ship NO chat_template in the header and still serve chat via fallback formatting. Ironically the UNLOADED gguf stub path already advertised `["chat","tool_use","streaming","json_schema"]` unconditionally — only loading the model made it vanish from every capability-driven client (the tray's LAN chat count, the "On Your Network" pickers). Same class as the ready-path `.mesh`/"3d" hole the `ReadyCaps` comment documents. Fix: `readyHasChat(is_encoder_only, chat_template_len, has_embedded_lm)` — template presence is NOT the gate for ds4/llama entries; used by BOTH renderModelEntry and the index page. App side: `ModelInfo.lanAdvertises(capability)` treats an empty capabilities array on a `lan_peer` entry as chat (old-peer tolerance — media entries always advertise their modality, so empty == this bug). Guards: `readyHasChat` test (server.zig), `LanModelCapabilityTests` (app).

### @peer proxying is bounded by the TUNNEL MARKER, not by loopback-ness (sandbox 403 class)
Live 2026-07-21: pi/hermes running in the Agent Sandbox VM got `403 "Remote (@peer) model ids are host-local"` for the model the host app was happily chatting on. The guest reaches the host over the VM NAT interface (`192.168.64.1`), so it is non-loopback BY CONSTRUCTION — and both the keyless LAN gate (`lanShareDenial`) and the proxy dispatch required loopback to initiate an @peer hop. Worse, with `--api-key` set the gate is skipped but dispatch still required loopback, so a keyed guest request naming @peer fell through to the unknown-id strip and would have been answered by the LOCAL default model silently. The loop/amplification bound never actually needed loopback: `lan.tunnel` has always stamped `X-MLX-LAN: 1` on every request it forwards, and the forwarded body carries the BARE id. New rule: any DIRECT client (loopback app, sandbox guest, phone on the LAN) may initiate exactly ONE hop; a request carrying the tunnel marker is never proxied again (`isTunneledRequest` at the gate AND at dispatch). Access-wise this exposes nothing new — the peer's own share gate still governs its models, and a LAN client could always ask the peer directly. Guards: `lanShareDenial` + `isTunneledRequest` tests (server.zig); `tests/test_lan_share.sh` "tunneled request never hops again" / "direct @peer id proxies" / "non-loopback client of B chats on @peer model".

### Same-machine peers bypass the non-loopback list filter → remote stubs re-export as @a@b chains; stale self-records self-mirror
Found 2026-07-21 while the user's live server shared on the same Mac as the test servers: a fresh discovering server listed `DeepSeek…@M4Max@MiniMac` — a mirror OF a mirror. Two servers on one Mac resolve each other LOOPBACK-FIRST (macOS Local Network privacy makes loopback the reliable path), and `/v1/models` only served the lan-filtered list (shared-only, no remote stubs) to `lanGateApplies` clients — which is false on loopback. So each same-Mac peer saw the other's full list INCLUDING its remote stubs and re-mirrored them (`test_lan_share`'s "A mirrored itself" check red for the same reason: A mirrored `@lantest-a@MiniMac` from the third, unfixed server). Related hole: a stale Bonjour record of a FORMER self (same name + port after a restart, different TXT token) passes the resolve-time TXT self-check, and the loopback-first fetch happily installs our own models as a "peer". Three-part fix, each independent: (1) discovery fetches self-identify — they already send `X-MLX-LAN: 1` — and now get the FILTERED list even over loopback; (2) `parsePeerModels` NEVER mirrors an entry that itself carries `lan_peer` (defends against old/unfixed peers); (3) `/v1/models` responses carry `X-MLX-LAN-Token: <process token>` and `fetchPeerModels` returns `error.SelfFetch` on a match → `.self_ad` → the service is forgotten. Guards: parsePeerModels lan_peer-skip + headerValueCI tests (lan.zig), the "peer-marked fetch never sees remote stubs" script check, and the (previously red) "A does not list its own shared models as remote" check.

### A serve path that hand-rolls its ServerConfig silently eats CLI flags (headless PLD class)
Found by review of PR #95 (2026-07-23), which fixed a third of it. `runHeadlessServe` builds its own `ServerConfig` literal rather than sharing the `--model` startup path's, and shipped with all three PLD fields written out by hand: `.default_enable_pld = false`, `.default_pld_draft_len = 5`, `.default_pld_key_len = 3`. Nothing the user typed reached a headless request. `--pld` parsed fine, `--help` documented it, the server started clean, and the one line that would have exposed it — server.zig's `PLD speculative decoding: ENABLED (draft_len=…, key_len=…)` banner — is gated on the same dead `default_enable_pld`, so it simply never printed. The only way to get PLD in headless mode was an explicit per-request `"enable_pld": true`.

That matters more than "one serve path is wrong": **headless is the mode the Swift app ALWAYS launches.** `ServerOptions.toCLIArgs` passes `--model-dir` and never `--model` (`app/Sources/MLXServe/Models/ServerOptions.swift:455`), and unconditionally emits all three spec-decode flags (`:497-499`) precisely so "the server's CLI defaults can't drift out from under the UI" — while the Settings UI describes Auto as "follow the server's `--pld` setting". Every app-launched server since headless mode landed ran with PLD forced off regardless of what Settings said. Published benchmarks are NOT affected: `tests/bench.sh:808` boots with `--model`, the non-headless path, which honored the flags all along.

PR #95 threaded `enable_pld` into the function and left the two literals beside it untouched — so `--pld-draft-len` / `--pld-key-len` stayed silent no-ops, in the same struct literal, two lines down. That is the actual lesson: the bug is not "someone forgot a field", it is that **related settings written as sibling literals drift one at a time**, and each fix looks complete because the field it touched now works. Fix: the three travel as ONE value, `server.PldDefaults` (`fromCli` for text-gen paths, `.off` for media-gen / ds4 / llama.cpp whose decode never routes through the PLD-capable generator), built once after arg parsing in `main()` and passed whole. A future edit cannot honor one field and drop its neighbours because there is only one field. Same role `effectiveSsmCheckpointStride` plays for `LoadParams` builders.

Diagnosis signature for the class: a flag that parses, documents, and boots without complaint but produces no behavioral difference, on ONE serve path only, where that path constructs a config aggregate by hand. Grep for the config literal, not the flag — the flag's parse site is always innocent. Guards: `server.PldDefaults` unit tests + `tests/test_headless_spec_flags.sh`, which boots headless over an EMPTY `--model-dir` (discovers zero models, needs no checkpoint, runs in seconds) and asserts the boot banner echoes non-default lengths — red on the pre-fix binary at `draft_len=5, key_len=3`.

### `name=` matched inside `filename=` → a well-formed image upload 400'd "missing image"
Found by pre-merge review of the MageFlow branch (2026-07-25), in `multipart.zig`, before it could reach a user. `paramValue(line, key)` pulled a `Content-Disposition` parameter with a plain `indexOfIgnoreCase` substring search. `name=` is a substring of `filename=`, so the value it returned depended entirely on which parameter the client wrote FIRST:

```
Content-Disposition: form-data; name="image"; filename="dog.png"   -> name="image"    (correct)
Content-Disposition: form-data; filename="dog.png"; name="image"   -> name="dog.png"  (the filename)
```

RFC 7578 fixes no order for the two. Only convention puts `name` first, and curl, the OpenAI SDK and browsers all follow it — which is exactly why this would have sat there. A client that didn't (a hand-rolled form, a proxy that reorders, a language binding with a dict-ordered serializer) would upload a perfectly valid image and get back `400 missing image`, with a server log showing a part named `dog.png` that matches no field we look for. Nothing in the message would point at parameter ordering.

- **Fix**: the match must sit at a parameter boundary — position 0, or preceded by `;` and optional whitespace. Non-matching hits advance the cursor and the search continues, so `filename=` is skipped rather than mistaken for `name=`.
- **Rule**: a header-parameter lookup keys on the PARAMETER, never on a substring. Any future reader (`/v1/audio/transcriptions` is the same shape when it lands) owes the same boundary check.
- **Guard**: `paramValue keys on the PARAMETER, not a substring of a longer one` runs the SAME part through the parser in both orders and requires identical `name`/`filename`. Verified red on revert: the original returns `dog.png` for the field name.

### An unbounded debug body log meets its first BINARY body
Same review pass. `logHttpBody` had always dumped a request/response body verbatim at `--log-level debug`, which was fine while every endpoint we served was JSON. `/v1/images/edits` is `multipart/form-data`, so the body is now raw PNG/JPEG bytes — up to the 64 MB request cap. One image upload at debug level wrote megabytes of binary into `~/.mlx-serve/logs/mlx-serve-<port>.log`, including NUL bytes, and a large enough upload rotated the 32 MB log away entirely. The file whose whole purpose is post-mortem (the app's buffer dies with the app) is destroyed by the request you were trying to debug.

The fix had to not break the thing the log is for: reading a complete request body out of it is the documented way to reproduce a tool-calling bug, so truncating everything to N KB would have traded one debugging failure for another.

- **Fix**: `bodyIsText` splits the two cases (printable + ordinary whitespace, with multibyte UTF-8 counting as text so an emoji in a chat body doesn't demote it). Text logs WHOLE, unchanged. Non-text logs `bodyPreview` — a bounded, strictly-printable-ASCII copy capped at `min(caller's buffer, BODY_LOG_LIMIT)` — labelled with the true byte count.
- **Rule**: adding an endpoint whose body is not text means auditing every place that treats a body as printable. The size cap alone is not enough; NUL bytes in a log break the tools that read it.
- **Guard**: `debug body log bounds BINARY but never truncates text (multipart PNG class)` pins both halves, including the real shape (text multipart framing wrapped around a binary payload) and that a generous caller buffer can't reintroduce the megabyte dump.

### The multipart `model` FIELD was invisible to model resolution — every image edit ran against the DEFAULT model (2026-07-25)
`handleConnection` resolves the target model BEFORE dispatching to a route, via `parseModelFromBody` — a linear scan for the JSON object key `"model"`. That is correct for every endpoint we serve except one: `/v1/images/edits` is `multipart/form-data`, where the same value arrives as `Content-Disposition: form-data; name="model"` and there is no `"model":` key anywhere in the body. So the scan returned null, the id was treated as omitted, and the request silently got default-model semantics. The route's own `openaiEditFormToJson` *does* read the field and puts it in the translated JSON, but by then resolution has already happened, so that value only reached the handler, never the scheduler.
- **Two different symptoms, one cause.** With a chat model as the default, the edit reached `handleGen(.image)` on a text model and returned `400 "Target model does not support this media modality"` — the reported bug (Open WebUI's `image_edits()`, which posts `model`, `prompt`, `n`, `size`, `response_format` as scalar fields and the file last). With a HEADLESS boot and no default at all (`--serve --model-dir`, no `--model` — the mode the app always launches), the same request returned `503 "No default model configured"` even though the client had named a model that was sitting right there on disk. Neither message points at the form field, which is what made it read like a Mage-Flow or a multipart-parsing bug; the multipart parser was fine and had just been hardened for a different client (`name=` inside `filename=`).
- **Why every existing test was green.** `test_mageflow_edit.sh` and `test_image_gen.sh` both boot `"$BIN" --model "$MODEL" --serve`, so the default model IS the model under test. The ignored field then selects exactly the model that would have been used anyway, and all eight edit-surface assertions pass. **A test cannot see "the id was ignored" while the default is already the right answer** — the guard has to run against a server with no default (new final section in `test_mageflow_edit.sh`: headless over the checkpoint's discovery root, edit names the model by id, expects 200; red-on-revert 503).
- **Fix**: `parseModelFromRequest(body, content_type)` dispatches on the content type (multipart → walk the form for a non-empty `model` part; otherwise the existing JSON scan) and is now the ONE way anything learns which model a request names. It feeds both `handleConnection`'s resolution and `lanShareDenial`, which matters because `/v1/images/edits` is `model_gated` in `lan.routeClass` and that function's whole contract is that the share gate can never disagree with what dispatch would run — fixing only the dispatch side would have left the gate approving an edit on the strength of the wrong model's share status.
- **Rule**: any value read out of a request body BEFORE the body is normalized (model id, and anything added later) must be readable from every body shape the server accepts, through a single shared reader. A new non-JSON endpoint is not just a new parser in a handler; it is a new shape for every pre-dispatch scan. Diagnosis pattern: `--log-level debug` logs the request body, and for a binary/multipart body a bounded sanitized preview — aiohttp and most clients write scalar fields BEFORE the file part, so the first 4 KB shows every field name and value, which is how this was localized in one request.

### A headless server answered 503 for paths that don't exist — every endpoint-probing client read it as a catch-all (2026-07-25)
Sibling of the multipart-`model` bug above, same root cause: **model resolution runs before dispatch.** On a server with no default model (`--serve --model-dir`, no `--model` — the mode the Swift app always launches), `ensureLoaded("")` fails with `NoDefaultModel` and returns 503 *before the route chain ever runs*, so a path that does not exist reports "No default model configured" instead of 404. Measured side by side, same binary, same paths:

| POST | headless | with `--model` |
|---|---|---|
| `/v1/__no_such_endpoint__` | **503** | 404 |
| `/v1/chat/completions` `{}` | **503** | 400 |
| `/v1/images/edits` `{}` | **503** | 400 |

- **Why it matters beyond tidiness.** Endpoint discovery works by probing: send an empty body and read the status — 404/405 means absent, anything else means present (llmprobe's `classifyStatus`), with a sentinel request to a nonsense path to detect servers that answer everything (LM Studio's HTTP-200-with-an-error-body). Our headless 503 on the sentinel tripped exactly that defence: llmprobe concluded "server answers unknown paths with HTTP 503" and scored **every** surface absent — chat, responses, messages, embeddings, images, audio. The server looked like it implemented nothing while serving fine. Note the asymmetry that makes this hard to spot: 503 on a REAL endpoint is harmless (it classifies as present); it is the 503 on the FAKE one that poisons everything.
- **Fix**: the `NoDefaultModel` arm answers 404 when `!routeExists(path)` before falling through to the 503. `ROUTE_PATHS` lists the 31 dispatched paths (plus the `/v1/responses/{id}` prefix) — a second list that must agree with the `if/else` chain, which is a drift class this file warns about repeatedly. The guard is a unit test that reads the chain out of `@embedFile("server.zig")`, scans for the path-equality call form, and fails on any literal missing from the table: a new route arm that forgets the table breaks CI instead of silently becoming a headless 404. (It caught a literal inside its own doc comment on the first run, which is the cheapest possible demonstration that it works.)
- **Rule**: anything answerable without a model — does this path exist, is this body well-formed — must be answered before the model is resolved, not after. And when adding a route, remember the table; the test will remind you. Related: `/props` was already special-cased in this arm for the same reason (the app's tray polls it and read a 503 as "0 MB"), which was the hint that the arm was doing too much.

### The index page rendered from a `*LoadedModel`, so the server's own front page 503'd on a headless boot (2026-07-25)
Third in the same family as the two stories above: **model resolution runs before dispatch**, and `GET /` sat on the far side of it. `handleStatusPage(allocator, stream, lm)` took a `*LoadedModel` and rendered 21 `std.fmt` slots off it — id, arch, quant bits/group, layers/hidden/heads/kv, head dim, vocab, context, model max, active + peak MB, capability pills. With no default model the arm was never reached and the root answered `503 {"error":"No default model configured"}`. That is the boot mode the app always uses, and since `mlx-serve serve` / bare `--serve` started discovering the shared models root and loading on demand, it is the default way the server starts at all — so the first page a person opens was an error object.
- **The page also documented 22 of 31 endpoints.** The API reference is hand-written prose; the entire Ollama `/api/*` surface (chat, generate, tags, show, ps, pull, version, embed, embeddings) had never been added to it. Nothing could notice, because "is the reference complete?" was an inspection, not a test.
- **Fix**: `GET /` moved up beside `/health` and `/v1/models`, above resolution, and `handleStatusPage` lost its `lm` parameter entirely. Everything model-shaped is now fetched client-side from `/v1/models` (which already returns id, capabilities, state, bytes, meta per entry) and `/props` (live memory). That is not just a workaround for the 503 — the page is now a model PICKER, so it has to render before anything is loaded by construction, and the picker follows loads and unloads without a refresh.
- **The `std.fmt` trap that shapes the whole file layout.** `index.html` is `@embedFile`d as a FORMAT STRING, so every literal `{`/`}` inside it must be doubled — which is why a page with real CSS and JS cannot be one file. `metrics.js` already had the answer: inject it as a RUNTIME `{s}` argument, because std.fmt does not re-parse runtime args. `app.css` and `app.js` follow the same pattern, and the slot count dropped from 21 to 6. Do not inline CSS or JS back into `index.html`.
- **Guards** (three layers, because the page has three failure modes):
  - `the index page documents every endpoint the server serves` (server.zig) — every `ROUTE_PATHS` entry must appear in `@embedFile("html/index.html")`. Red today with the nine `/api/*` paths. Same shape as the `ROUTE_PATHS`↔dispatch-chain guard, and it makes "are we missing endpoints?" un-repeatable rather than re-inspectable.
  - `tests/test_index_page.sh` — headless over an EMPTY `--model-dir` (no checkpoint, seconds): `GET /` → 200 `text/html`, the tab/control markup is present, every endpoint path is in the served bytes, and the `#mlx-metrics` mount appears with `--metrics` and not without. Red-on-revert: 1/20 with the arm moved back below resolution.
  - `tests/html_console_test.mjs` — the pure decision layer (capability filtering per picker, SSE frame cutting across split chunks, request/form construction, auth passthrough), plus a static cross-check that every id `app.js` reaches for exists in `index.html` and every rendered control is read by `app.js`. That last one covers the class no HTTP assertion can see: a typo'd id makes `$('chat-sned')` return null, the listener is never attached, and the button is silently dead while the page still renders, still serves, and still passes every byte-level check.
- **Note on media**: edit capability is not API-visible — both Mage-Flow-Turbo and Mage-Flow-Edit-Turbo report `capabilities: ["image"]`, ship byte-identical configs, and the server itself gates on the directory NAME (`mage_flow.dirIsEdit`). The console mirrors that rule client-side rather than inventing one. An explicit `image_edit` capability on `/v1/models` would replace both halves; it is a server API change, not console work.

### The console is a chat with tools, not a page of forms — and the live runs wrote the rules (2026-07-25)
Second pass on the console. Images and Audio stopped being tabs: the tabs are **Monitor** (default, first — the live metrics panel plus the full model inventory), **Chat**, and **API**. Media is something you ASK for, so the chat is handed one tool per modality this server can actually serve (`mediaTools`), executes what the model calls, and renders the picture or the player inline in the assistant's bubble. The user-editable system-prompt box is gone because the console now needs that slot itself: the prompt carries the tool instructions, the model inventory, and the API reference, which is what lets the same chat answer "which endpoint edits an image?".

Everything below was found by driving the real page in a real browser over CDP against a real server — none of it is visible from unit tests, and every fix landed in the pure, tested layer rather than in the DOM.
- **A round cap does not bound cost; a budget does.** Asked for "an image of a fox", a 2B model generated the fox and then invented three more edits nobody requested — four GPU generations, tens of seconds each, off one sentence. `MAX_TOOL_ROUNDS` cannot fix that: every round is another picture. `toolInvocation` now takes `ctx.mediaUsed` and refuses beyond one media generation per user turn. The refusal has to be a SENTENCE the model can act on ("already produced one result for this request; tell the user what you made and let them ask for the next change") — a model that gets silence, or a bare error, just calls again. The same instinct applies to the tool RESULT text: "Generated the image" reads as an invitation to continue, so it ends with "Reply with one short sentence now. Do not call another tool."
- **A tool's `model` enum and its resolution must be the same list.** The edit tool enumerated every image model while resolution merely *preferred* an edit-capable one — and an explicit choice beats a preference, so the model picked `Mage-Flow-Turbo` straight out of the enum and the edit 400'd. Offering a choice that is guaranteed to fail is the same class as advertising a capability you don't have. `editableIds` is now one list feeding both, pinned by a test that resolves every id the enum offers and asserts it comes back unchanged.
- **Rank candidates by how likely they are to WORK.** Two Qwen3-TTS checkpoints on disk, the bf16 one an incomplete download (config + tokenizer, no safetensors). It sorted first, so every "say this out loud" spent a load attempt on it — `NoWeightFiles`, "Model load failed" — before a retry found the sibling. The pre-load tell is in `/v1/models` already: discovery sums the checkpoint's `*.safetensors`, so `bytes_on_disk: null` means the shards are missing. `rankedIds` orders resident (free, and provably loadable) → sized → unsized → `error`, and a failed tool call refreshes the model list so a retry inside the same turn ranks past the entry the registry just marked. The picker deliberately does NOT reorder: it refreshes every 15 s and would shuffle under the cursor.
- **Whatever the system prompt leaves out, the model invents.** With only paths and one-line descriptions in the prompt, "how do I edit an image?" produced `curl -X POST https://your-ollama-ip-address/api/v1/images/edits -F "ref1=<base64>"` — wrong host, wrong path prefix, invented field names. The prompt now carries `location.origin` and a short true list of real request fields. Listing accepted and rejected fields in one sentence was not enough either: the model presented `mask`, `n`, `response_format:"url"` as available options, so rejections are now a separate, explicitly-labelled clause. And "give me a curl for the edit endpoint" was answered by GENERATING A PICTURE until the prompt said in as many words that questions are answered in text with no tool call at all.
- **The API reference has one source.** The prompt's endpoint list is scraped from the API tab's own rendered markup (`#tab-api .ep`), so the page and the assistant cannot disagree, and the Zig drift guard (every `ROUTE_PATHS` entry appears in `index.html`) covers both at once.
- **Guards**: `tests/html_console_test.mjs` grew to 44 tests over `mediaTools` / `toolInvocation` / `accumulateToolCalls` / `systemPrompt` — each of the bullets above is a named regression test. `tests/test_index_page.sh` pins the tab set, that Monitor ships `class="panel active"` (what a visitor sees before any JS runs), that Images/Audio tabs are GONE, and that no user system-prompt box came back.

### Third pass: a sidebar, persisted chats, and the metric a client cannot measure (2026-07-25)
Layout moved to a sidebar — **New chat / Monitor / API**, plus **Recents** — and chat became the landing view: a greeting and a centred composer that turns into a transcript on the first send. It is ONE composer element in two layouts (`.panel.empty` flips it), because two composers is two sets of listeners and one of them always rots. Temperature and max-tokens went away; model choice and Extended thinking live in the composer's pill menu, both remembered in localStorage.
- **Recents is localStorage, and what you DON'T store is the design.** A single 1024² PNG is ~1.5 MB of base64 and the whole origin gets ~5 MB, so persisting one image-generating conversation would evict every other one. `storableTurns` replaces every `image_url` part with an `image_omitted` marker and keeps everything else — crucially including `tool_calls` and the `tool` results, or a reloaded chat could not be continued. `historyUpsert` caps by count AND by serialized size, dropping oldest-first, so a few very long chats can't wedge the store.
- **Markdown is rendered from ESCAPED input, always.** Model output is untrusted — it routinely quotes the user, and the user may have pasted anything. The renderer escapes first and then builds a whitelisted subset (headings, lists, fences, inline code, emphasis, links), so `<script>` survives as text even inside a fence, and `[x](javascript:alert(1))` degrades to plain text because only `http(s)` produces an `href`. Text streams as plain text and is re-rendered as markdown once the turn closes: parsing per token is wasted work and fights half-written syntax.
- **A client cannot measure decode rate against our own server.** The console showed **937 tok/s on a 2B**, and it was not an arithmetic slip: with `tools` present the server buffers tokens for tool-call detection and flushes at the end (documented above, under the keepalive class), so every SSE delta arrives in one burst — first-byte and last-byte are milliseconds apart and wall-clock decode time is ~0. The fix is not a cleverer clock: the final chunk already carries `timings` (`prompt_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`, `predicted_per_second`) measured on the server around the actual forward passes, which buffering cannot distort. The console sums that block across the turn's rounds, which also keeps a minutes-long image generation out of the denominator for free. Verified against the server's own log line for the same request: console 104.6 tok/s vs `decode: 102.1 tok/s`. **`stream_options.include_usage` is load-bearing** — the server gates the entire final chunk on it, so dropping it silently removes the only trustworthy timing a client can get. Related trap of the same shape: TTFT. A buffered stream has no observable first token either, so the console reports the server's `prompt_ms` as "prefill" rather than claiming a time-to-first-token it cannot see.
- **A menu that opens upward is bounded by what's above it.** The model picker lists every chat model — 16 on this box — and a `max-height: 60vh` box anchored above the composer ran off the top of the window with its first entries unreachable. Clamp to `pill.top - container.top`, measured at open time.

## `--model=<path>` was silently dropped (arg loop with no else) — 2026-07-25

```
./zig-out/bin/mlx-serve --serve --model=~/.mlx-serve/models/…/Nanbeige… --metrics
…
[args] model:
mlx-serve 0.1.0-dev (headless — models load on demand)
```

`main.zig`'s flag loop matches every flag by EXACT name and reads its value from
the next argv slot:

```zig
} else if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
    i += 1;
    model_dir = args[i];
```

There is no `--model=X` arm, and the loop ended at a bare `}` with **no else
branch at all**. zsh passes `--model=…` through as one token, nothing matched
it, and it fell out of the loop in silence.

Everything downstream then looked healthy. `[args] model:` printed empty, the
server took the headless path (the same path the app always launches), and on
the first chat request the registry auto-picked an unrelated default —
`[registry] default model -> prism-ml/Ternary-Bonsai-27B-mlx-2bit` — which
promptly crashed on a separate kernel bug (docs/gotchas/engine-mlx.md). The
user spent the whole session believing they were debugging the model named on
the command line. They were not; that model had never loaded, and its
`model_type` was unsupported anyway.

Same family as the `runHeadlessServe` entry above: the flag parses as far as the
user can tell, `--help` documents it, boot is clean, and nothing anywhere
reports that the request was ignored. A launcher that quietly ignores what it
was asked for is worse than one that refuses to start.

The loop now rejects anything it did not consume, via a pure classifier in
`cli.zig` (main.zig is not in the test aggregator, so the testable helper lives
there):

- `.equals_form` — starts with `-` and contains `=`. The actual trap; the
  message names the shape that works: *"flags take their value as a separate
  argument (--model <path>, not --model=<path>)"*.
- `.missing_value` — a flag in the LAST argv slot. This is precisely the case
  the `i + 1 < args.len` guards let fall through, and it needs no list of flag
  names to detect: position alone identifies it.
- `.unknown` — everything else, pointed at `--help`.

Positional subcommand arguments are consumed before the loop via `arg_start`
(3 for `run <model>`, 2 for `serve`), so tightening the loop cannot break
`mlx-serve run qwen3`.

**Rollout check for a change like this** — hard-failing on unknown args breaks
every caller that passes a stale flag, so before shipping, diff what callers
send against what the loop matches:

```sh
grep -oE 'args\[i\], "(--?[a-z0-9-]+)"' src/main.zig | grep -oE '"--?[a-z0-9-]+"' | tr -d '"' | sort -u > known
grep -rhoE '(zig-out/bin/mlx-serve|\$\{?BIN\}?)[^|;&]*' tests/*.sh | grep -oE '\-\-[a-z0-9-]+' | sort -u > used
comm -13 known used      # must be empty
```

Also confirm `ServerOptions.toCLIArgs` emits no `=`-joined flag
(`grep -rnE '"--[a-z-]+=' app/Sources`) — the app launches the server on every
boot, so an `=` form there would have turned this fix into a launch failure.

### Serial ≠ exclusive: dsv4's module-owned decode state needed admission-level single-flight (cross-request stream corruption, 2026-08-02)

Live incident: pi was mid-generation on the DSV4 mirror when the Swift app asked the same server a question. pi's SSE stream started emitting the APP's conversation — "The code looks like it's the DeepSeek… I'm'm here to to assist assist you you … any any any…" — every word doubled, then degenerate.

Mechanism, verified in code: dsv4 is the only text arch whose per-request decode state does NOT live in the per-slot KVCache — it is ONE `Dsv4Model.dec_state` per loaded model. `forwardDsv4WithImpl` documents "SERIAL-ONLY: history lives on the model; a fresh request is detected by cache.step == 0" — but nothing enforced it. The scheduler admitted any number of slots: the inference loop drained every pending slot and prefilled it, and the decode tick gave every active non-batchable slot its own `runSingleDecodeTick` per tick, interleaved. So the app's request hit `cache.step == 0` → deinit + rebuild of pi's `dec_state` with the app's prompt; pi's next tick decoded the APP's state (the leak); then both slots alternately appended tokens to the ONE state (the doubled words) until position bookkeeping diverged (the degeneration).

Why the existing "Serial (isMoe)" protection didn't cover it: `modelBatchable=false` only excludes a model from the BATCHED decode kernel. Non-batchable slots still interleave serial ticks — which is perfectly safe for laguna/hy3, whose state is per-slot. dsv4 broke the per-slot-state invariant that interleaving silently relies on, and no admission gate existed. Same class as the 2026-07-31 PLD-on-dsv4 corruption: a documented property ("serial-only") enforced at only one dispatch layer.

Where to block, and where NOT to: `submit()` already single-flights llama (`llama_session_busy`, claimed in submit, released in `complete`) — but submit blocks the conn thread BEFORE any response bytes, and a dsv4 wait can be minutes, which lands squarely in the streaming-keepalive class (undici kills silent sockets at ~300 s). Holding the slot in `pending` instead keeps the conn thread in its existing `waitNextTimeout(STREAM_KEEPALIVE_MS)` loop, where SSE keepalives already flow. StallClock can't fire during the wait — it lives on the Generator, which is only created at prefill. So the second dsv4 request QUEUES (FIFO) until the active one finishes, same UX as queueing behind `--max-concurrent`.

Fix shape (`src/scheduler.zig`): pure `admitPendingTick(cands, active, out)` (the `specTickMode` pattern — contract-tested decision fn + a source-scan pinning the call site) gates the pending drain in `inferenceLoop` step 1, under `queue_mu`. An exclusive candidate admits only if its model is neither among live decoding slots nor claimed by an earlier admitted exclusive candidate this tick (same-tick siblings aren't in `decoding` yet — the claim covers the window). `modelExclusiveDecode` keys on the transformer's dsv4 pointer, mirroring `runPrefill`'s `is_dsv4` — NOT on `!modelBatchable` (would wrongly serialize laguna/hy3) and NOT on a model_type string. Non-exclusive candidates always admit, and a candidate for another model behind a held exclusive one still admits — no head-of-line blocking. No release bookkeeping: the busy signal IS presence in `decoding`; step-5 culls finished slots under the same mutex, and the loop never blocks while `pending` is non-empty, so a held slot admits on the first tick after the active one is culled.

Repro/guard: `tests/test_dsv4.sh` [7] — greedy solo baseline, then the same request with a short marker chat ("Reply with exactly: Kangaroo") fired mid-generation. Pre-fix RED reproduced the exact incident signature: the concurrent long output truncated and LITERALLY contained "Kangaroo" (`… 2, 3, 5, 7, "Kangaroo`). Post-fix: byte-equal to solo, marker answered after queueing, no leak — 17/17.

---

## Logprobs were misleading, not missing — three defects, one broken instrument (2026-08-05)

Found while chasing a model-quality artifact. The hunt ate a day partly because
an early read of "logprob -0.004, therefore the model is 99.6% confident" came
straight out of this field. It was wrong in three independent ways, all on
main, all model-agnostic.

**1. The values were post-temperature.** `sampleToken` threads its WORKING
logits into `computeLogprobs`, and by that point they carry the client's
temperature (and any repeat/presence penalty). So the same prompt, the same
chosen token, reported `-0.2129 / -0.0607 / -2.1566` at temp 0 / 0.6 / 2.0 — a
number that belongs to the model moving with a knob the client set. Worse, at
temp 0 there is no division at all and `log_softmax` over raw logits SATURATES:
many entries report exactly `0.0` (p = 1.0), which reads as certainty and is
actually just the absence of scaling. OpenAI's logprobs are the model's, so
they now read the position's raw logits — before temperature, penalties and
top-k/top-p.

**2. Token ids were recovered by scanning the vocab for float equality.** For
each `mlx_topk` value the producer walked all 157k entries looking for a slot
whose logprob compared `==`, skipping ids it had already used. Under the
saturation above, ties are everywhere and the winner is whichever id the scan
reached first. The same comment claimed `mlx_topk` returns values "in
descending order" — it does not (argpartition class) — and the caller then kept
the first `top_n` of `top_n + 1` unordered candidates, so the true argmax could
be the one dropped. Ids now travel WITH their values: `mlx_argpartition_axis`
on the negated logprobs, slice the leading k, `mlx_take_axis` the values, sort
host-side (ties break on the lower id so the ranking is deterministic).

**3. The result was paired with the NEXT token.** This is the one that made the
output look like a broken RANKING rather than a broken PAIRING. The decode loop
returns `next_token_id` and, in the same call, forwards it to sample its
successor — then published THAT `sampleToken` result as `last_logprob`, which
the scheduler appends and `formatLogprobsObject` zips with token ids by index.
So every entry carried the distribution of the token that FOLLOWED it. A
one-token "OK" reply came back as:

```
token "OK"  logprob 0.0   top_logprobs[0] = { "<|role_end|>", 0.0 }
```

which is exactly what the model wants AFTER "OK" — both values 0.0 because the
chosen id being reported was `<|role_end|>`'s, not "OK"'s.

Fix is a one-token delay: `Generator.pending_logprob` holds the freshly sampled
result and `last_logprob` is only ever assigned from it. The first returned
token needs a seed, because t1 is sampled from the PREFILL's final forward,
which the decode loop never sees — `firstTokenLogprobs` reads that distribution
in `initWithOptions` (both branches), against the id the lazy sampler actually
drew rather than re-sampling (which would disagree at any temperature > 0).

Observable bar, and the one to re-run: at temp 0 the emitted token IS the
argmax, so `top_logprobs` rank 1 must equal it. Measured 0 of 5 positions on a
trivial prompt before; after, every position agrees and the values are real
(-0.547 on a first token instead of a saturated 0.0). Guards: two hermetic
tests in generate.zig (a tie-saturated distribution whose rank 1 must be the
argmax; three temperatures against one independently computed log_softmax
value), a source-scan class guard pinning that `last_logprob` may only be
assigned from `pending_logprob`, and `tests/test_ling.sh` [11] live on both the
chat and completions surfaces.

## `/v1/completions` ignored its `logprobs` field (2026-08-05)

Same session. The handler hardcoded `logprobs_n = 0` and still emitted a
`logprobs` key in the response — the silently-ignored-field class, which reads
to a client as "this model has no opinion" rather than "this server never
asked". Two things differ from chat and both matter:

- the request field is an INTEGER (how many alternatives per token), not a bool
  plus a separate `top_logprobs` count;
- the response is four PARALLEL ARRAYS — `tokens`, `token_logprobs`,
  `top_logprobs`, `text_offset` — where `text_offset` is each token's byte
  offset within the completion text (what a FIM client uses to align
  alternatives with its buffer).

`top_logprobs` there is a MAP keyed by token TEXT. That is OpenAI's own shape
and we reproduce it, but it means two byte-fragment token ids that both render
as U+FFFD COLLIDE and the larger value is lost. The first pass of the
collapse metric was built on this surface and manufactured a run of fake
`p = 0.000%` positions out of exactly that. **Any measurement over logprobs
must use the chat surface**, whose `top_logprobs` is a list.

## `/detokenize` emitted raw control bytes (2026-08-04)

The handler hand-rolled a five-character escape table (`"`, `\`, `\n`, `\r`,
`\t`) and passed every other byte through verbatim. Any token whose bytes are
below 0x20 — and a byte-level BPE vocab has plenty — therefore produced a body
that NO JSON parser accepts, from an endpoint whose entire job is to hand text
back to a client.

Same class as the tool-calling `appendJsonString` rule, one surface over: a
literal is arbitrary bytes too, and there is exactly one correct escaper in the
tree. `detokenizeResponseJson` now routes through `chat.appendJsonString`, with
a test that feeds it a control byte and parses the result.

The general form: any handler that builds JSON with `allocPrint` and a string
it did not escape at a SINK is one unusual input away from an unparseable
response. Grep for hand-rolled escape tables when a client reports "invalid
JSON" from an endpoint that is otherwise working.


## The streaming think gate was O(buffer) per token (2026-08-05)

`chat.streamThinkGate2` runs `indexOf(buf, "<|channel>thought")`,
`indexOf(buf, "<channel|>")`, `indexOfThinkOpenTag(buf, 0)` and
`indexOfThinkCloseTag(buf, 0)` over the WHOLE accumulated buffer on every token,
for as long as thinking markup is present and no close has arrived. That is
O(n²) over a reasoning block. Hermetic bench, 4000 tokens growing to 113 KB:

```
no close in buffer:  47.99 us/token   (every scan runs to the end)
close present early: 16.98 us/token   (the close scan stops at ~byte 14)
```

0.3% at ~15 ms/token, ~1% on a 5 ms/token model with a long unclosed
thought, and it grows with the buffer — a 32K-token thought is 8x this bench.

**What makes a cursor exact.** `tagSuffixChar` excludes `<`, so every recognized
marker contains exactly ONE `<`, at its start. A marker straddling the
scanned/unscanned boundary must therefore begin at the LAST `<` in the buffer —
and if that `<` can no longer grow into a marker (`isPartialSuffixedTag`, or a
strict prefix of the two channel spellings), nothing straddles the boundary at
all. No overlap constant to get wrong, and no length bound needed even for an
arbitrarily long `</think:suffix>`. The plain substring needles get a fixed
16-byte overlap (`<|channel>thought` is the longest at 17).

**The close tag is the one value that cannot be latched.**
`thinkCloseIsToolCallPayload` looks FORWARD past the close for a `</tool_call`,
so a close that is a real block close at token N is reclassified as argument
payload at token N+k. Latching it diverged from the fresh gate at prefix 104 of

```
<think>plan<tool_call>f<arg_key>k</arg_key><arg_value>closes with </think> inside</arg_value></tool_call>done</think>visible
```

— the incremental arm said `.split_think` where the fresh gate said
`.hold_thinking`. So the latch is gated on "no `tool_call` substring seen yet",
and once tool markup appears the scan falls back to the exact full
`indexOfThinkCloseTag(buf, 0)` every call. That costs a handful of tokens, not a
block: the caller latches `think_closed` immediately after a split.

Measured after: 203 MB scanned → 164 KB over the same 4000 tokens (1238x), and
the memoized total is ~1.6 passes over the final buffer, i.e. linear.

Pinned by three tests — prefix-by-prefix equivalence against the fresh gate over
every marker family (which IS the split-across-arrivals case, at every possible
split point), a flat-cost invariant on `last_scan_span`, and a reset test for
the buffer the stream loop clears at every emit. Plus a wiring scan: both
streaming handlers must hold a persistent `ThinkScan` and reset it where they
call `text_buf.clearRetainingCapacity()`, because a memoization nobody threads
through is output-identical to no memoization at all.

## A gate that runs before the estimator that knows better IS the estimator (#126, 2026-08-05)

`ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit` was unloadable on a 48 GB M5 Pro. Every
`POST /v1/load-model` came back:

```
HTTP 503 {"error":{"message":"Not enough memory to load model; retry after
current requests complete","type":"out_of_memory"}}
```

on an idle server, zero models loaded, RSS 21 MB — so the advice was
unactionable, and nothing was logged at the point of refusal.

The numbers, from the reporter:

| file | bytes |
|---|---|
| `text_encoder.safetensors` | 15,804,791,921 |
| `transformer.safetensors` | 18,698,813,290 |
| `video_vae.safetensors` | 5,207,808,496 |
| `audio_vae.safetensors` | 605,254,808 |
| sum | 40,316,668,515 (37.55 GiB) |

`ensureLoaded`'s eviction gate estimated post-load bytes from `entry.bytes_on_disk`
regardless of backend and added 10%: **41.30 GiB**, against a
`--max-resident-mem auto` of 80% of the 38338 MB wired limit = **30.0 GiB**.
`planEvictionsLocked` found no victim (nothing was loaded), returned null, and
the load became `error.NotEnoughMemory`.

The correct number already existed twenty lines further down the call chain.
`gen.h3PeakBytes` bills `max(TE, DiT) + video_vae + audio_vae` = **22.83 GiB**,
because `minimax_h3.generate` runs the text encoder and FREES it before the DiT
loads. The staged-residency fix had landed in the media PREFLIGHT, which runs on
the inference thread — after the registry gate. For H3 the gate always won, so
on any machine where `sum x 1.1 > max_resident_mem` the model was permanently
unloadable: every 48 GB Mac at stock settings. `--skip-mem-preflight` did not
help (it bypasses the free-RAM preflights, not the registry cap) and the app
passes no `--max-resident-mem` at all, so there was no way out from the UI.

The class is not the formula, it is that **two sites computed the same bill
differently and the stricter one ran first**. Both now read one estimator:
`scheduler.mediaPeakFor` (peeks the dir's real backend type — `arch_hint` when
discovery supplied one, `gen.peekModelType` otherwise, because a media stub's
`config.model_type` is the MODALITY static "AudioVideo") feeding
`gateEstimateBytes`. The peek happens OUTSIDE the registry mutex: it stats the
model dir and no other load should block on our filesystem.

Two secondaries from the same report:

- **The commit disagreed with the reservation.** `doLoadGenOnInferenceThread`
  called `markReadyLocked` with `bytes_on_disk`, so after a successful load H3
  sat in the residency budget at 37.55 GB while holding almost nothing (the
  engine holds only paths until a generation arrives — the hermetic guard's
  `200 OK` on four SPARSE files is that fact, reproduced). A media model parked
  at 14.7 GB more than it can ever hold evicts genuinely-resident LLMs for bytes
  nobody is using. `genLoadResidentBytes` reads the same estimator, so reserve
  and commit now differ only by the gate's 10% headroom.
- **The refusal named the wrong subsystem and logged nothing.** "retry after
  current requests complete" points at concurrency; the cause is a static cap.
  The message is one constant now (`server.not_enough_memory_message`, both 503
  sites) naming `--max-resident-mem`, and the gate logs estimate / cap /
  currently-resident / model count before returning.

Guard: `tests/test_media_eviction_gate.sh`, hermetic — the "model" is four sparse
files (`dd seek=`) carrying the real pack's byte sizes plus a `config.json`
naming the backend. Nothing is ever read, so the load fails at engine build, and
WHICH failure it is is the assertion: past the gate the answer is no longer the
gate's 503. Verified red-on-revert, where it reproduces the reporter's 41.30 GiB
exactly.

## A loop cut that says only "length" reads as a limit nobody set (2026-08-05)

A pi session against a collapse-prone 4-bit MoE repeated itself and then died with
"Model stopped because it reached the maximum output token limit", while its own
status bar read `32.4%/66k` — two thirds of the context free. The two readings
look contradictory. They are not: **context and output are different budgets,
and neither one is what stopped it.**

End to end:

1. A garbage token landed in a file the agent was writing:
   `v(cx - 5, 6, cz + z, C.roofRedDark);!placeholder`. That is the checkpoint's
   own logit collapse, not a server bug — correct sampling (the card's top_k)
   and the reserved-token suppression mask are what move its rate.
2. The model spotted its own corruption and could not repair it, restating the
   same intent with different wording — the near-repeat shape
   `generate.isNearRepeatTailLoop` exists for.
3. The server cut it, five times, at 1254 / 1071 / 1079 / 102 / 58 generated
   tokens. **The shrinking lengths are the diagnosis**: each retry re-entered
   the loop sooner, because the client re-sent the cut turn as history and the
   model read its own loop back. That is the error-echo class (Inkling
   name-salvage) with the server's own output as the error.
4. `finish_reason: "length"` is deliberate and cannot move — `"stop"` became
   `"tool_calls"` and presented a server-cut fragment as a completed write
   (the 2026-07-14 php.html post-mortem). pi renders `length` the only way the
   OpenAI schema allows. pi never set a `max_tokens` at all (the log shows the
   unbounded sentinel `1073741823`), so the message names a limit neither side
   imposed.

Two fixes, and the split between them is forced by the transport:

- **The cause rides beside the reason.** `finish_details:{"type":
  "repetition_loop"}` on chat + completions, stream and non-stream. Unknown
  causes are dropped rather than interpolated (`finishDetailsField`) — this
  string is spliced into a JSON literal, and a literal is arbitrary bytes too.
  `/v1/messages` is deliberately excluded: `anthropicStopReason` maps a loop cut
  to `max_tokens` (the same misattribution), but inventing a key inside
  Anthropic's schema is worse than the gap.
- **The trim is what breaks the spiral, and it only reaches non-streaming.**
  `generate.degenerateTail` returns where the degenerate span STARTS, not just
  that one exists: the exact tiers walk their cycle back past the repetitions
  that convicted it and keep ONE copy (a truncated answer should still show what
  the model got stuck on, and one copy cannot sustain a loop), while the
  near-repeat tier slides its 1024-token window back in 128-token steps while it
  keeps convicting — a restatement loop that ran 3000 tokens is degenerate for
  all 3000, and trimming only the window hands the rest back.

Why streaming keeps the tail: a delta cannot be retracted. It is worth being
precise about why the tokens are already gone, because "with tools present the
server buffers" is true only of tool MARKUP — `streamShouldBufferForTools`
returns false for prose, so a restatement loop streams incrementally. Measured
on the reproduction: 113 separate content deltas over ~1 s before the cut, with
and without `tools`. For a streaming client the SIGNAL is the whole deliverable.

Reproduction, no checkpoint-specific behaviour needed: ask any model to "Output
the exact line 'ping pong ping pong' over and over, hundreds of times, with no
other text and no ending" — the period-1..8 tier convicts within ~130 tokens.
`tests/test_loop_stop_signal.sh` is that prompt across four surfaces, and its
last section boots with `MLX_SERVE_LOOP_TRIM=0` so the trim's own red-on-revert
is part of the run (32 repetitions in the body with the trim off, 2 with it on).

## Streaming chat accepted `logprobs`, paid for them, and dropped them (2026-08-07)

Found during pre-release validation of 26.8.3, by a hand-written probe rather
than by the conformance suite — which is the point of the story.

**Symptom.** `POST /v1/chat/completions` with `"logprobs": true, "stream": true`
returned a well-formed SSE stream with no `logprobs` anywhere in it. The
non-streaming form of the same request was perfect.

**What made it expensive rather than merely absent.** Requesting logprobs
disables every speculative path (`pickStreamMode` — PLD/drafter/MTP all gate on
`logprobs_n == 0`, because a spec round has no per-step distribution to report).
So the request paid the full serial-decode cost to honour a field that was then
thrown away. Worst of both ends.

**Why nothing caught it.**

- Output-equality tests are structurally blind: the content deltas are
  byte-identical whether or not the logprobs ride along.
- `llmprobe` probes logprobs on the NON-streaming surface only. In the same
  session it scored this server `Logprob consistency 100% — 36/36 items:
  emitted token = argmax, valid distribution` while streaming returned nothing
  at all. **A conformance suite's silence is not coverage** — a green run says
  what it checked passed, never that the surface works.
- The three logprobs defects fixed on 2026-08-05 (temperature-scaled logits,
  float-equality id recovery, the one-token offset) were all found and fixed
  non-streaming, and their guards live there too.

**Root cause.** There is exactly ONE chat streaming chunk template, and its
choice object was `{"index":0,"delta":…,"finish_reason":…}` — no `logprobs`
field had ever existed on it. `formatLogprobsObject` was called from the
non-streaming handler only.

**Fix, and the three things that were not obvious.**

1. `logprobs` is a SIBLING of `delta` on the choice, not a field inside it.
   Added as `ChunkExtras` (defaulted, so all 22 existing emitters keep their
   exact bytes — a stream that did not ask for logprobs is byte-unchanged,
   which is what makes the change additive).
2. Entries cannot be paired 1:1 with chunks. The think gate and tool detection
   buffer many tokens into one delta, and some tokens produce no chunk at all.
   So `StreamLogprobs` drains against a HIGH-WATER MARK (`emitted`), and each
   entry ships EXACTLY once: a delta cannot be retracted, so a re-send is as
   wrong as a drop.
3. The publish is a THREAD-SAFETY problem, not a formatting one.
   `slot.logprobs_buf` is written by the inference thread and was documented
   as conn-thread-readable *at completion*. Reading it mid-stream races two
   ways: the entry for token i may not be visible when token i is handed over
   (the append sat AFTER `pushToken`, whose mutex release is what publishes),
   and a concurrent grow reallocates the backing array under a reader
   mid-copy. Both fixed by moving token and entry into ONE critical section
   (`Slot.pushTokenWithLogprob`) and copying out under the same lock
   (`Slot.copyLogprobsFrom`). The copy is shallow on purpose: each entry's
   `top_logprobs` is its own allocation, stable for the life of the slot and
   owned by the slot; only the ArrayList's backing array is at risk, and that
   is exactly what the lock covers.

Safe by construction on the concurrency side: `Scheduler.batchable` returns
false when `logprobs_n > 0`, so such a slot always runs `runSingleDecodeTick`
and appends exactly one entry per token, in order.

**Guards.** `tests/test_logprobs.sh` section [4] asserts streaming carries
logprobs AND agrees with non-streaming token-for-token and VALUE-for-value on
the same greedy request — which is what catches a partial drain, a duplicated
one, or an off-by-one high-water mark, none of which "is the field present?"
can see. Class guard: a source scan (`every streaming chat emitter carries
logprobs`) rejects a bare `.{}` on any `sendSSEChunk` inside the streaming
handler, so a NEW emitter cannot silently forget, plus an assertion that both
halves of the path to the wire still exist (the extra is read, and the rendered
bytes are interpolated into the chunk).

**Rule of thumb this leaves behind:** when a request field costs something to
honour, check that the cost buys delivery. A field that disables an
optimisation and then is not emitted is strictly worse than one that 400s.

---

## `logprobs.content` described the thought, not the answer (2026-08-07)

Found while checking an outside report that llmprobe's fidelity score could not
rank eight local checkpoints. The report blamed the metric. Half of it was ours.

### The symptom

`logprobs.content` is defined by OpenAI as the tokens of the message CONTENT.
We built it from the raw `token_ids` of the generation, which also carries the
reasoning block, leaked tool markup, and anything the loop-trim cuts. So on any
model that thinks, the array described text the client never received:

| model | reasoning | content | logprobs entries |
|---|---|---|---|
| Qwen3.6-27B (3 builds) | 693 ch | 8 ch | 186 |
| Qwen3.6-35B-A3B distill | 404 ch | 8 ch | 105 |
| gemma-4-31b | 138 ch | 8 ch | 37 |
| gemma-4-e4b | 303 ch | 8 ch | 79 |
| Qwen3-4B | 443 ch | 8 ch | 104 |
| LFM2.5-2.6B | 186 ch | 8 ch | 44 |

`logprobs.content[0]` came back as `'Here'` / `'Thinking'` / `'<|channel>'` /
`'<think>'` — the opening token of the model's reasoning. LFM2.5 does it with
thinking OFF as well, because its template opens `<think>` unconditionally, so
the block exists and is stripped regardless of the request flag.

This is why an outside fidelity probe read `'The'` (from "The user is asking
for…") as the answer token on every item of its battery, and concluded the
measurement was saturated. It was pointed at the wrong position.

### Why nothing caught it

Same shape as the streaming drop above. Output-equality tests see identical
text either way. The conformance suite that would have noticed reads
`logprobs.content[0]` and trusts it — it has no independent idea of where the
answer starts. And on a model that does NOT think, the array is correct, so any
spot-check against Gemma answering directly looks fine.

### The fix, and the trap inside it

Non-streaming is arithmetic: the split helpers return raw slices into the
generated text, so `contentTokenRange` recovers the content's byte offset by
pointer comparison and walks per-token decoded lengths to a token index. A
content slice it cannot locate (a future transform that rewrites rather than
cuts) keeps the FULL range — an array we cannot align still beats no array.

Streaming cannot use that directly, because the emit sites fire on the gate's
cadence and the pending window does not correspond to any one buffer. So it is
structural instead:

- reasoning emitters never drain (they pass `.{}`);
- a content chunk emitted after a block calls `StreamLogprobs.skipToContent`
  first, which indexes `ids`/`lens` — complete by construction — rather than the
  pending window, so a token whose logprob has not published yet still has a
  length and cannot skew the boundary.

**The trap: empty content.** A think block that closes with nothing after it
emits no content chunk at all. `skipToContent` returning early on empty content
therefore left the entire thought pending, and it rode the NEXT chunk — measured
42 entries on a one-character delta. Four sites needed the `dropPending` arm,
and the first three attempts at this fix were verified against the wrong code
path entirely: the model in hand routes through the prompt-opened-think arm, not
`.split_think`, and dumping the raw SSE frames was the only thing that showed
which chunk actually carried the entries. **Read the wire before deciding which
branch to patch.**

The emitter scan is now two-sided: a content emitter must drain, a reasoning
emitter must not. A one-sided guard would have accepted the version that shipped
the thought's entries alongside the answer.

## A logprobs token string is a BPE fragment (2026-08-07)

A single token can carry HALF a multi-byte character; the rest arrives in the
next token. `jsonEscape` passes every byte >= 0x20 through verbatim, so a
`top_logprobs` candidate of `b"\xf0\x9f"` — the leading half of a 4-byte emoji,
seen on Jundot/Qwen3.6-27B-oQ4e-mtp — went into the JSON string as raw bytes:

```
"bytes":[10,10]},{"token":"\xf0\x9f","logprob":-15.000000,"bytes":[240,15…
                           ^ byte 75009 of a 77211-byte body
UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 75009-75010
```

The whole response fails to parse. Not a degraded field — an unusable response,
from one candidate in one top-5 list. It surfaced as a bare decode error while
sweeping models for the alignment bug, on one model out of seven.

`jsonEscapeLossy` emits U+FFFD per invalid sequence, using the maximal-subpart
rule so a character split across two tokens costs one replacement rather than
one per byte. `bytes` is untouched and still carries the exact bytes, which is
OpenAI's own shape and lets a client reassemble across tokens.

Two things not to do:

- **Do not widen it to `jsonEscape` or `appendJsonString`.** Every other string
  we emit is complete decoded text and valid UTF-8 by construction; the fragment
  problem is unique to per-token decodes.
- **Do not "fix" the legacy collision.** `/v1/completions` keys `top_logprobs`
  by token TEXT, so two invalid candidates both render U+FFFD and collide into
  one key. That is OpenAI's shape reproduced faithfully; a test that pins it away
  would be pinning our own invention.

The integration guard checks EVERY response body the script produces rather than
one crafted request, because which request happens to draw a split candidate
into its top-5 is luck.

## The H3 residency bill: a staged plan billed as if it were not one (2026-08-07)

Reported as two issues against the app: "MLX Core.app can't reach the workaround"
and "the staged-peak formula still overcounts, by a lot". Both were right.

The `h3PeakBytes` shipped with #126 was `max(TE, DiT) + video_vae + audio_vae`,
and its own doc comment admitted the VAE term was not an accounting claim but a
"direction-safe margin for decode-phase activations". Two overcounts rode on
that:

1. **The VAEs are billed against the DiT.** `minimax_h3.generate` scopes the DiT
   in a block that closes before the VAE decode — its own log narrates the whole
   chain (`encoder released` → `dit resident` → `DiT released` → `video decoded
   (load+decode)`). The two VAEs DO coexist (the video decoder's `defer` runs at
   function scope, so it is still resident when the audio VAE loads), which is
   why they stay one stage.

2. **`transformer.safetensors` is a bad proxy for DiT residency.**
   `precomputeAdaln` tables the whole schedule's modulation and frees the 13B
   AdaLN weights — roughly 39% of the DiT's parameters, so the surviving share
   barely moves with quant width. Measured 32.83 → 20.19 GiB on the 8-bit pack
   and 17.41 → 10.84 on the 4-bit. The comment right above the call already said
   "~22 GB instead of ~35"; the estimator two files away had never heard.

On the real 8-bit pack that is 38.97 GiB, ×1.1 = 42.87 against a 48 GB Mac's
29.95 GiB auto cap, for a process whose measured peak (`footprint --sample 2`,
768×448/124f) is 26 GB. Refused, permanently, on every Mac under ~96 GB.

### What the fix could NOT be

The obvious replacement — `max(te, dit_resident + lora, vaes)` — drops the only
slack covering generation transients, which are real: the reporter's own 4-bit
row shows a 17 GB process peak against 10.84 GiB of DiT weights. Those transients
scale with pixels × frames (1344×768 is 3× the area of the measured cell), and a
per-MODEL load gate cannot see a request's shape, so the allowance is a
judgement call rather than a derivation. What it is NOT is uniform across stages:
the TE stage is one forward over a few hundred prompt rows. A shared
`max(stages) + activations` bills the biggest stage for transients it never
allocates — which, with the TE at 26.28 GiB, is exactly what kept the 8-bit pack
refused even after the first two fixes.

So: `max(te, max(dit_resident, vaes) + H3_ACTIVATION_BYTES)`, with the allowance
at 6 GiB against a measured 4.0–5.0.

`h3DitResidentBytes` takes `precompute` as a PARAMETER and the caller reads
`minimax_h3.adalnPrecomputeOn()` — the same predicate `generate` branches on. A
bill that assumes the weights are shed while `MINIMAX_H3_ADALN_PRECOMPUTE=0` runs
the full DiT under-bills by ~12 GiB, and an under-bill here is an uncatchable
Metal OOM, not a 400.

### The 10% that was double-counted

`gateEstimateBytes` added 10% headroom "for KV / vision / drafter overhead" to
every bill including a media peak. Those are text-model concepts — a media engine
has no KV cache and no drafter — and the media estimator now carries an explicit
transient term of its own. The commit side (`genLoadResidentBytes`) had never
taken the 10%, so removing it from the gate also makes reserve and commit agree
exactly, which is what that function's comment always claimed.

Real packs, gate estimate before → after: FL2VA-8bit 42.87 → 28.06 GiB,
REF2VA-8bit 42.87 → 27.34, FL2VA-4bit 25.11 → 17.32. All three now clear a 48 GB
Mac's auto cap; the 8-bit bill still sits ~4 GiB above the measured peak.

### Why it was uncheckable

The DiT term was wrong for months because the only number anyone could compare it
against was the DiT's own `dit resident` log line — added when a weights map
outliving `Model.load` pinned 13 GB. The TE stage had no such line, so its bill
(its full file size, still) is now logged the same way. A staged bill whose stages
do not each report their residency is not auditable, and the audit is the only
thing that catches this class.

## A cancellation signal that only exists on one response shape (2026-08-07)

Same report: "a disconnected client leaves an orphan job that holds the server
queue until finished."

`StreamCtx` latched `cancelled` when an SSE progress write FAILED. That is the
only cancellation media generation ever had, so a non-streaming request — no
progress writes, nothing to fail — had none. The connection thread is parked in
`Scheduler.runGeneration` while the job runs on the inference thread, so a client
that hangs up (or trips its own timeout — a 1344×768 H3 clip outlasts most
defaults) leaves the GPU producing a video nobody will receive, with every other
request queued behind it.

The fix is the sink on both paths: `stream = false` no-ops `cb` (an SSE event
spliced into a single JSON body is unparseable) and keeps only the probe. The
probe is `Conn.peerClosed`, which the text-generation paths have used for years
and which media never adopted — and it fixes the streaming path too, where a
failed write is a LATE signal because TCP send buffers absorb hundreds of events
after the peer's FIN.

Scope, honestly: only backends that POLL `Progress.cancelled` can be stopped —
H3, LTX, hunyuan3d, acestep. The image backends never poll it, so a FLUX/Krea
generation still runs to completion. Those are seconds to a minute; the class is
the same and the fix would be per-backend loop wiring.

The audit turned up a second thing: `src/gen_sse.zig` was never listed in
`src/tests.zig`, so its tests had never run. A filter that matches no test still
reports "1/1 tests passed" (the other test step's), which is why nobody noticed —
when checking that a new test is red, check it against a deliberately bogus
filter too.

### Does the H3 shape generalize? Not by itself — and LTX proved it

Asked directly after the fix: does this work for LTX, and for future backends?
No. `estimatePeakResidentBytesIn` had one `if (model_type == "minimax_h3")` arm
and everything else fell through to `sumSafetensorsIn`, so the answer for every
other backend was still "assume nothing is ever freed and everything lives in
this directory".

For most of them that assumption holds. krea, mage_flow, hunyuan3d, acestep and
tts each keep text-encoder + DiT + VAE as fields on one Engine struct for the
engine's lifetime, all inside the model dir — the sum IS their peak.

LTX breaks it twice, in opposite directions (measured on
`dgrauet/ltx-2.3-mlx-q4`, 29.61 GiB of safetensors):

- `transformer-dev.safetensors` and `transformer-distilled.safetensors` are
  10.54 GiB each and BOTH ship. `LtxVideoEngine.ensureTransformer` frees the
  resident one before loading the other, with a comment saying exactly why
  ("so dev + distilled (11 GB each) never coexist"). The sum bills a phantom
  10.54 GiB.
- The Gemma text encoder is not in the model dir at all. `resolveGemmaDir`
  points at the shared `mlx-community/gemma-3-12b-it-4bit` repo (7.5 GiB), and
  `ltx_video.gemmaCapture` loads it per generation, uses it, and frees it on
  return — on top of the entire resident engine. The sum bills it at zero.

The two errors partially cancel on a two-variant pack, which is why nothing had
been reported. They do NOT cancel on a pack shipping one variant: there the sum
under-bills by the whole encoder, and under-billing is the uncatchable-OOM side.

So the fix was to name the shape rather than add a second special case:
`stagedPeakBytes(resident, stages)` — `resident` for what the engine holds
forever, `stages` for groups that are loaded and freed and therefore never
coexist, peak = resident + the biggest stage. H3 is `stagedPeakBytes(0, {TE, DiT
+ act, VAEs + act})`; LTX is `stagedPeakBytes(dir_sum − spare_variant, {gemma})`.
Out-of-dir stages resolve in the OUTER `estimatePeakResidentBytes` so the
per-directory function stays hermetically testable.

LTX gets NO activation term. H3's 6 GiB is derived from H3 measurements; LTX has
none, and a fabricated allowance would newly refuse loads that work today. An
unmeasured number is not a safe default just because it is conservative.

## `--model-dir` is REPEATABLE, and a scan path the SERVER never hears about is a browse-only folder (2026-08-06)

The flag took one directory, so the app's "Custom folder" fed only its OWN picker — a model there was absent from `/v1/models`, and selecting it made `discoveryModelDir` point the server at that model's parent INSTEAD of the library, so the choice was always either/or. `model_discovery.discoverModelsMany` merges N roots FIRST-WINS on a repeated id (not tidiness: `registerStubWithArch` answers `error.DuplicateId` and `registerDiscovered` does `try`, so an un-deduped merge fails registry init and the server does not start), skips an unopenable root with a warning (the second folder can be on an unplugged drive), and the arg loop REFUSES a 9th folder by name rather than dropping it.

App side: `ModelRoots` is the one answer to both "where do downloads go" and "what does the server scan" — the destination is a real setting now (`ServerManager.modelsRoot` was a second hardcoded copy of `DownloadManager`'s path), it leads the scan list so its copy wins a duplicate, and the built-in root stays in that list forever so a moved destination never hides the library already on disk. Gated to Developer ID (`BuildFeatures.customModelFolders`): under MAS the helper is signed `com.apple.security.inherit`, which inherits the app's CONTAINER but NOT its security-scoped grants, so the app could pick a folder the process that reads the weights cannot open. Guards: `tests/test_multi_model_dir.sh`, `ModelRootsTests` (incl. a source scan that only `ServerOptions` may spell `--model-dir` and only `ModelRoots` may build the models root).

**Second bite (2026-08-08): the SERVER kept both roots, the APP's own reads did not.** `scanRoots` (what `--model-dir` gets) kept the built-in root after a destination move, but every app-side read resolved against `modelsDir` alone: `discoverLocalModels` (the whole pre-move library vanished from the picker while `/v1/models` was still serving it), `existingModelDir(for:)`/`isReady` (browser rows offered a re-download of models already on disk), `ServerManager.resolveModelDir` + `componentReady` (a media pack downloaded pre-move read `.modelMissing` — a 69 GB re-download offer), `discoverDrafters`, and `deleteModel`'s root scoping (the trash rendered for a built-in-root model and silently did nothing — dead-control class). Fix: `ModelRoots.ownedRoots` / `DownloadManager.ownedRoots` — destination first, built-in second, FIRST root winning a repeated id (the server's own first-wins rule) — behind every read named above. Three deliberate exclusions: WRITES stay on `modelsDir` (`newLayoutDir` — downloads go where the setting says), CANCEL cleanup stays destination-scoped (a cancel cleans what THIS transfer wrote, never a same-named quant in the built-in root; transfers only ever write into `modelsDir`), and a test-pinned `DownloadManager` keeps `ownedRoots == [modelsDir]` so a temp-dir test can never resolve into — or delete from — the developer's real library. Guards: `OwnedRootDiscoveryTests` (first-wins dedup, built-in fallback, media-gen resolution, pinned-root hermeticity) + the `ownedRoots` case in `ModelRootsTests`.

## Console voice mode = browser STT + Kokoro TTS

`#chat-voice` lives in the COMPOSER row, hidden when `sttSupported` is false — never a button that cannot work. The mic runs ONLY in the `listening` state: leave it live during playback and the page transcribes the assistant's own voice and answers its own sentence. Replies go through `speakableChunks` (markdown STRIPPED not escaped — a fence is announced as "(code block)", a bare URL as "a link") and are synthesized one chunk ahead of playback so the first sentence starts while the rest generates.

## Context-overflow 400s name BOTH counts (`contextOverflowMessage`)

All four text-gen surfaces: "Prompt exceeds maximum context length: N tokens requested, M available". The legacy sentence stays the PREFIX (clients key on it); the counts are only knowable server-side, since the request is rejected before any usage is reported, and without them a client can only say "too long" instead of offering the one action that fixes it. bufPrint failure falls back to the bare sentence rather than sending no body (the media-gen fixed-buffer class). The app renders it as a card.

## A load failure crosses the inference-thread boundary by NAME (#144)

Issue #144: Krea-2-Turbo mixed 4/8 answered `HTTP 500 {"message":"Model load
failed"}` on a reporter's machine while loading fine elsewhere. The real reason
(the media memory preflight refusing — peak ~14.7 GB against their free RAM)
existed only as a server-log line; the reporter deleted and redownloaded the
model, which could never have helped.

On-demand loads run on the inference thread and failures come back as
`req.error_name` — and `ensureLoaded` FREED the name unread, returning bare
`error.LoadFailed`. So every cold-load failure (memory refusal, missing file,
malformed config) collapsed into one unactionable 500, while the eviction-gate
refusal (`error.NotEnoughMemory`, raised on the CONN thread) had a named 503
the whole time. The message quality depended on which THREAD noticed the
problem, not on what the problem was.

Fix: map the name back to a typed error at the boundary
(`ModelRegistry.loadErrorFromName`, also applied on both `.error_state` fast
paths so retries answer the same). `InsufficientMemory` gets its OWN 503
(`insufficient_free_memory_message`) rather than folding into the gate's: a
preflight refusal is about free RAM and `--skip-mem-preflight`, the gate's is
about `--max-resident-mem` — different knobs, and #126 says name the knob.
Everything else stays `LoadFailed` and the HTTP arm echoes the registry's
stored name: `Model load failed: FileNotFound`.

One subtlety: a memory refusal now resets the entry to `.unloaded` instead of
`.error_state`. The 503 tells the user to close apps and retry — with a sticky
error_state that retry failed fast until a server restart, so the error message
itself promised a remedy the state machine forbade. Transient refusals must not
poison the entry.

Guard: `model_registry.zig` test "memory-refused loads keep their identity,
other failures expose their name".

**Third bite (2026-08-09): `ownedRoots` fixed the destination move and became the next too-narrow list.** A Mage-Flow pack sitting in the CUSTOM scan folder (`/Volumes/G Drive SSD/models`, one of the server's `--model-dir` roots) was served by `/v1/models` while the Image pane showed a BundleDownloadBar over it — `bundleReady`/`componentReady`, `existingModelDir(for:)` and `ServerManager.resolveModelDir` all read `ownedRoots` (destination + built-in only), which deliberately excluded LM Studio + custom folders. The exclusion conflated two questions: "may the app DELETE here?" (no — other tools'/the user's trees) and "is this repo on disk?" (must check everywhere the server serves). Fix: `ModelRoots.readRoots` ≡ `scanRoots` (destination, built-in, LM Studio, custom — same first-wins order the server uses) behind every read: `existingModelDir(for:)` (which also targets the Turbo-adapter fetch — the adapter belongs beside the pack wherever it lives), `componentReady`, `discoverDrafters`, `resolveModelDir`, the voice-clone disk check. Writes, cancel cleanup and delete scoping stay on `ownedRoots`/`modelsDir`; a test-pinned root still stands alone. Guard: `testReadRootsCoverEveryServedFolderButOwnedRootsStayNarrow` (ModelRootsTests).

## A per-surface spec re-derivation is a list of ONE (DFlash serial-decode miss, live 2026-08-10)

First live boot of the DFlash block-drafter: the boot log said `DFlash
speculative decoding: ENABLED`, the request parse said `drafter=enabled
(block_size=16)` — and every request decoded serial at 16 tok/s with no
`[spec-stats]` line at all. The parse-time `enable_drafter` was correct;
what dropped the sidecar was the NEXT layer down: four per-surface
re-derivations (`use_drafter = ... lm.drafter != null ...` in the
completions, chat non-streaming, Anthropic messages and Responses handlers)
plus two parse-default/fallthrough guards, all written against the Gemma
drafter's handle only. `enable_drafter` arrived true, the guard saw
`lm.drafter == null` (the sidecar loaded as `lm.dflash`), and the submit
passed neither handle. Nothing errored — the regular-decode fallback is
output-identical, which is exactly why engagement must be asserted by
COUNTS (`[spec-stats] attempts>0`), never by output shape.

Fix: every drafter-loaded gate reads `lm.drafter != null or lm.dflash !=
null`, and a source scan in dflash.zig fails any non-comment server.zig
line that mentions `lm.drafter != null` without a `dflash` sibling on the
same line ("every server-side drafter-loaded gate also consults lm.dflash").
Same class as the dsv4 PLD-dispatch hole: the wiring that matters is not
where the flag is PARSED but every site that re-derives it.

## The usage chunk restated the ending, and every per-event client rendered it twice (PR #147, 2026-08-11)

OpenAI's `stream_options.include_usage` contract: the usage-carrying chunk
ships `"choices": []`. Ours re-sent `finish_reason` + `finish_details` beside
the usage object — a second "the reply ended, here's why" event. Any client
that acts per event acted twice: the app appended its truncation banner once
per event carrying a truncation cause, so one loop cut rendered TWO "⚠️
Stopped — the model started repeating itself" banners (PR #147's report; its
`TruncationGate` stays as defense-in-depth against other backends that
restate).

Fix: chat streaming's include_usage chunk goes through a dedicated
`sendSSEUsageChunk` — `"choices":[]`, `usage` + `timings` only, no delta, no
finish, no logprobs (all per-choice fields; the final chunk already carried
them, and the pending-logprobs drain lives there alone now). The completions
streaming path had usage riding the finish chunk itself — same deviation, one
event — and now emits the same empty-choices usage chunk after its final
chunk. Blast radius checked: the ollama sink returns early on an empty
choices array and reads usage off the root before that check; the app and the
console both read `usage`/`timings` from the chunk root; `test_timings.sh`
keys on the usage object, not choices.

The finish event is now stated on exactly ONE chunk of every stream. Guards:
`tests/test_loop_stop_signal.sh` [2] (finish_details AND finish_reason
exactly once, usage chunk `"choices":[]`) and [4] (completions usage chunk
shape) — all four red on the pre-fix build.

## JSON mode answered "## Attributes": the grammar mask was built from another model's vocabulary (2026-08-11)

llmprobe against two different models on the same box:

```
✗ chat/completions: JSON mode
    → not valid JSON: #(tr)
✗ responses: JSON mode
    → not valid JSON: 郑重(郑重)
✗ chat/completions: structured outputs (json_schema strict)
    → not JSON: <<<<<<< Vcc
```

Streaming the failing request showed the model in a two-token cycle — `##`,
` Attributes`, `##`, ` Attributes` — under a `json_object` constraint that
should have allowed exactly `{` and whitespace at position 0. The server had
logged `[grammar] enforcing JSON schema`, so the constraint was installed and
the spec-decode gates (which all key on `sampling.constraint == null`) had
correctly stayed off. `/tokenize` + `/detokenize` round-tripped every id
involved, so the tokenizer was fine too.

The mask size in the log was the tell. Muse-Glimmer's vocabulary is 202048;
its request logged `mask=125017b` — LFM2.5's. `getOrBuildTokenBytes` was:

```zig
var global_token_bytes: ?token_mask_mod.TokenBytes = null;
fn getOrBuildTokenBytes(gpa, tok) !*const TokenBytes {
    if (global_token_bytes) |*tb| return tb;   // keyed on NOTHING
    ...
}
```

— "built lazily on the first JSON-schema request and reused for the lifetime
of the server", written when a process served one model. The multi-model
registry and hot model switching made that comment false without touching the
line: whichever model served the first constrained request owned the table,
and every other model masked its logits against a foreign vocabulary. Ids are
only bytes in the vocabulary they were decoded from, so the mask let through
tokens whose real bytes are off-schema, `acceptByte` then rejected the bytes
it got back, the grammar went dead, and the mask fell open to the whole vocab
— free-running output under a constraint the client was told was enforced.

Which model works and which breaks is decided by request order, so the same
build passes for one caller and fails for the next.

Fix: the table lives on `LoadedModel` (`grammarTokenBytes`, guarded by a
per-entry mutex, freed immediately before the `tokenizer` it was decoded
from), beside `prefix_cache` and `tokenize_cache` — the same per-model
ownership the tokenizer itself already had. Both call sites pass `lm`; the
singleton and its shutdown hook are gone. The build line now names the model,
which is the assertion the integration guard reads: a shared table logs
exactly one build.

Guards: `tests/test_json_mode_multi_model.sh` (two resident models, A then B
then A, both surfaces, plus one build line per id) and a `model_registry`
unit test that hands two entries different vocabularies. Both red on revert —
the shell guard reproduces the reported symptom verbatim (`[`, `[\n {common`).

## An empty grammar mask is not a constraint (same session)

With the vocabularies straightened out, `tests/test_json_schema_enforcement.sh`
still went 0/6, on a different failure: a conforming object with one extra
key spliced in — `{"name":"Mira Chen","age":34,"email":"...","!__employee_id__":null}`
— against a schema with `additionalProperties:false`.

The log named it: `sampled token 0 produced byte 0x21 that was rejected`.
Token 0 is what argmax returns when every logit is `-inf`, i.e. when the mask
was all false. `stepObject`'s `.after_value` accepted `,` unconditionally,
which lands in `.expect_key`, where — every declared property seen and
`additionalProperties:false` — no byte is legal. The sampler cannot express
"nothing", so it drew id 0, whose bytes failed `acceptByte`, which switched
enforcement off for the rest of the generation. One unreachable state, and
the schema stopped applying entirely.

Two fixes, both load-bearing: `stepObject` rejects the comma when
`allPropertiesSeen` and additional properties are off, so `}` is the only way
out; and `nextConstrained` treats a zero-count mask as a bug it names in the
log before degrading, instead of sampling through it. The prompt-side
instruction had been carrying these cases — that is why an
`additionalProperties` violation read as a model-quality problem.

## `--no-drafter` did not survive a model switch, and two flags before it didn't either (2026-08-11)

Audit prompted by the grammar-mask singleton above: same shape, different
state. `ensureLoaded`'s cold-load path builds its own `LoadRequest` — a second
construction site next to main.zig's boot `LoadParams` — and anything it
omits takes the struct default. The comments in that function already record
two prior rounds: prefix-cache settings ("silently crippled warm reuse after
every model switch") and the MTP + llama group. A third had accumulated:

- `--no-drafter` — the consequential one. `dflash.resolveInDirDrafter` probes
  `<model_dir>/drafter` at load, and both muse mirrors ship that subdir, so a
  server launched with speculation off re-enabled it on every model switched
  to. This flag was inert on the cold path until the in-dir probe landed; the
  dflash change made it load-bearing and nothing wired it.
- `--draft-block-size N` — fell back to `DEFAULT_BLOCK_SIZE` with
  `explicit=false`, so `resolveBlockSize` re-derived from config/hardware
  instead of honoring the clamp.
- `--ssd-streaming` — set only in the boot params, so a ds4/GGUF model loaded
  later ran without it, and got the MTP sidecar that flag suppresses.
- `--drafter <path>` — the standing `"Phase E will wire the load-model API to
  set this"` TODO.

Fix: five fields retained on `Scheduler`, re-applied in the cold-load request
— except the drafter PATH, which is deliberately not propagated. `--drafter`
names a sidecar for the checkpoint it was passed beside; handing it to
whatever model is swapped in next loads a mismatched assistant. So
`coldLoadDrafterDir(no_drafter, primary_model_dir, drafter_dir, entry_path)`
applies it only when the entry IS the launch model — which is what makes a
reload-after-eviction get its drafter back — and leaves every other model to
the in-dir probe. `--no-drafter` is a policy and silences all of them.

The class guard is the source scan, not the behaviour test: every field
retained on the Scheduler for this purpose must appear as
`.<field> = self.<field>,` in the cold-load request, so the NEXT flag someone
adds is caught rather than the three that already were. The behaviour test
(`tests/test_cold_load_launch_flags.sh`) proves the wiring reaches a live
server without needing a real 2.5 GB sidecar: the scratch drafter is a
config.json declaring the contract and nothing else, so the probing arm fails
its load loudly and the `--no-drafter` arm says nothing at all.

## A typo'd URL cost 121 GB and two minutes: the 404 ran after the load (2026-08-11)

Driving a hot-switch test by hand, `POST /v1/load` — the route is
`/v1/load-model`. The reply was the correct 404. It arrived 2 minutes 42
seconds later, and the server log showed the full DeepSeek-V4 checkpoint
resident at 120.67 GB.

Dispatch resolves the request's model before it dispatches, and the
existence check lived inside `ensureLoaded`'s `error.NoDefaultModel` arm:

```zig
const lm = scheduler.ensureLoaded(requested_model_id) catch |err| switch (err) {
    error.NoDefaultModel => {
        if (!routeExists(path)) { ...404...; return; }   // only on THIS arm
```

That covers exactly one case — an unknown path on a server with nothing to
load. When the body names a model the registry can resolve, `ensureLoaded`
does not fail: it succeeds, cold-loading the checkpoint, and the unknown path
404s afterwards on the dispatch chain's own fallthrough. The comment above
`ROUTE_PATHS` had the principle right ("one question has to be answerable
BEFORE a model is resolved") — the implementation only answered it when there
was no model to resolve. The rule is an ordering claim, and it was enforced
as an error arm.

`curl -d '{"model":"<anything big>"}' http://host/v1/anything` is therefore a
one-line way to pin the box, no auth surface required (`--api-key` exempts
loopback, and the gate that would refuse this is downstream of the load).

Fix: one unconditional `if (!routeExists(path))` above `ensureLoaded`, below
the LAN proxy block so an `<id>@<peer>` hop is unchanged, and the copy in the
error arm deleted — a question with one answer gets one gate.

This had also been propping up `tests/test_cold_load_launch_flags.sh`, written
the same session: it posted to `/v1/load` with `curl -sf`, so the 404 was
swallowed and the cold load happened as a SIDE EFFECT of the bug. Red-on-revert
still passed, so the wiring it guards was genuinely proven — but the test would
have gone silently vacuous the moment this was fixed. It now posts to
`/v1/load-model`, checks the HTTP status, and asserts the clone reached `ready`
in the arm whose evidence is a MISSING log line. A silent arm has to prove it
did the work.

## A stream and a non-stream answer must be the same bytes (2026-08-13)

Found by `tests/test_logprobs.sh`, which reported three streaming failures:

```
FAIL  [stream] same entry count as non-streaming   17 vs 16
FAIL  [stream] tokens match non-streaming          [0, 1, 2, 3, 4]
FAIL  [stream] logprob VALUES match non-streaming  [0, 1, 2, 3, 14]
```

Logprobs was innocent. Each surface described its OWN content faithfully; the
content itself differed. Same model, same request, same seed:

```
non-stream : 'One, two, three, four, five, six, seven, eight.'
stream     : '\n\nOne, two, three, four, five, six, seven, eight.'
messages   : '\n\nOne, two, three, four, five, six, seven, eight.'
```

Every non-streaming delivery goes through `splitThinkBlock`, whose no-tag branch
ends in `trimStart(content, "\n ")`. The streaming paths flush token text
verbatim, so they kept the whitespace the non-streaming split had always dropped.

The model that exposes it is LFM2.5, whose template opens `<think>`
**unconditionally**: with thinking off the prompt gets `</think>` appended
(`chat.noThinkTailSuffix`), so the model's first generated token is the `'\n\n'`
that follows the closer. But the class is wider than one checkpoint — any model
whose first visible token is whitespace is in it.

### Why it hid

`LOGPROBS_TEST_MODEL` defaults to `~/.mlx-serve/models/mlx-community/LFM2.5-2.6B-8bit`.
That path stopped existing when the library moved to the external drive, and the
script exits 0 on a missing model, so it had been SKIPPING. Three suites were
skipping for the same reason. A skipped arm reads as a pass.

### The fix, and its two deliberate limits

`chat.streamContentLead(chunk, content_started)`, wired into all four content
emitters on both streaming surfaces (chat completions: the tools flush arm, the
plain token arm, the post-close remainder, the end-of-stream tail; `/v1/messages`:
the same four).

Leading whitespace is the ONE thing a stream can still withhold — nothing visible
has been sent, so this is a suppression, never a retraction.

1. **It never cuts inside a token.** A partial trim would ship a logprobs entry
   whose `token` no longer appears in `content`; the collector describes whole
   tokens and cannot split one. So a chunk is suppressed only when it is
   ENTIRELY whitespace. A token mixing whitespace and text rides whole — the
   narrow residual, and the honest one.
2. **A suppressed chunk retires its pending logprob** (`lps.dropPending()`, the
   arm the empty-content case already used). Without it the entry rode the next
   chunk and the stream reported 17 entries for 16 content tokens — the original
   symptom, now caused by the fix instead of the bug.

Interior whitespace is untouched: `'line one\n\nline two'` streams intact.

### A thinking-enabled STREAM that answers into an empty content field

LFM2-VL, live 2026-08-13. The app shows a Thinking block containing a complete,
correct answer — and nothing under it. Same request non-streaming: the answer is
in `content`, `reasoning_content` is null. Two surfaces, one prompt, different
bytes.

All three streaming handlers seeded their think state as
`enable_thinking OR prompt_opened_think`. The OR is the bug. It encodes an
assumption — "a thinking-enabled model opens `<think>` as its first token" —
that is true for Qwen and Gemma and irrelevant for them: their templates RENDER
the opener when thinking is on, so `promptOpensThink` already sees it in the
rendered bytes and the flag adds nothing. It is false for LFM2-VL, whose
generation prompt is a bare `<|im_start|>assistant\n` and whose model answers
directly. So the stream opened a block nobody opened, every token routed to
`reasoning_content`, no `</think>` ever arrived to close it, and `content`
finished empty.

The existing rule (`in_think_block` starts from the PROMPT, not the request
flag) was written for the end-of-stream FLUSH, and `streamTailIsReasoning`
enforces it there — positive evidence required, `prompt_opened_think` or
`saw_think_open`. The SEED was never covered, so the per-token routing did the
damage before the flush's guard could matter. A rule that names one site is a
rule about one site.

Signature to recognize it: the whole answer arrives as reasoning, `content` is
empty, the non-streaming request is fine, and nothing in the log looks wrong —
`thinking=true` is exactly what was asked for. It also is not vision-specific,
even though a VL checkpoint is what surfaced it: any model whose template does
not render the opener is in the class, text requests included.

The guard is a source scan (`a stream never starts inside a think block because
the REQUEST asked for thinking`) with the needle `++`-split so the test's own
comment cannot satisfy it — the first version of this scan passed green against
the broken build for exactly that reason. The integration bar is the
stream-vs-non-stream byte invariant rather than a phrasing check, because the
broken build produced perfectly good prose; it just filed it under the wrong key.
## A KV bill that assumes every layer caches, at one width for K and V (2026-08-11)

`computeMemoryContext` and `checkAttentionMemory` both billed `layers × 2 × kv_heads × head_dim × 2` bytes per token. On every arch that shipped before, that was exact. `bailing_hybrid` breaks both factors at once: 18 of its 24 layers are Kimi-Delta-Attention, which holds a FIXED-SIZE recurrent state (~9 MB for the whole model, per request, independent of context) rather than a per-token cache; and its MLA stores a 192-wide key against a 128-wide value. Real bill: `6 × 16 × (192+128) × 2 = 61,440` B/token. Billed: `24 × 2 × 16 × 128 × 2 = 196,608`. A 3.2x over-bill, which showed up as auto-context pinning to 14336 tokens on a machine that fits 29696 — on a model whose entire architectural argument is cheap long context.

The fix is `ModelConfig.kvBytesPerToken()`, fed by an honest `attnCacheLayerCount()` (which the struct's own doc comment had anticipated: "a hybrid MLA arch can carry attention on a fraction of its layers"). `prefillMemoryNeeded` lost its `layers` parameter in favour of that per-token figure — `layers` only ever fed the KV term, and neither the layer count nor the per-head width is uniform once an arch interleaves recurrent layers or stores keys wider than values.

Two things this class insists on. The sizer and the ADMISSION GUARD must move together: raising auto-context while the prefill guard still bills the uniform figure produces a server that advertises 29696 tokens and then 400s a 25000-token prompt. And the wiring is source-scan-pinned at the call site for the same reason the `attn_keys` argument is — an estimator that TAKES a per-token KV bill proves nothing if its one caller recomputes a uniform product on the way in.

Note this also corrects the bill for the other hybrids (qwen3.5/3.6 GDN: 10 of 40 layers cache), which raises their auto-context too. The per-request recurrent state those layers do hold is a constant, not a per-token term, and is small next to the KV it replaces.

## The stored width is not the scored width (2026-08-12)

Follow-up on the KV-bill class above. `prefillMemoryNeeded` took one `hdim` and used it twice: for the quantized-KV dequant transient (correct — that reads the STORED width) and for `prefillHeadDimFused(hdim)`, which decides whether the composed SDPA path materializes a `[heads, chunk, seq]` score tensor. Those are different numbers the moment an arch scores wider than it stores. `bailing_hybrid` declares `head_dim` 128, scores over 192, and its own MLA comment says mlx has a fused vector kernel for that pair at DECODE and "falls back to the composed path at prefill widths" — so the score tensor is real, while `prefillHeadDimFused(128)` is true and billed it at zero. At 32K with a 4096 chunk that is ~4 GB of scratch the admission guard could not see, and under-billing does not produce a 400: it produces an uncatchable Metal OOM, or all-zero logits at the working-set edge.

The patch that introduced `ModelConfig.prefillScoreHeadDim()` wired it into the prefill CHUNK cap — the same rule, one call site short. The estimator now takes `score_hdim` beside `hdim`, and both are inside the string the call-site source scan pins, so neither can be quietly recomputed on the way in. Guard: `prefillMemoryNeeded: the SCORE width decides the score term, not the stored width`.

## Auto-context billed a chunk-bounded transient per token, and the KV at fp16 (2026-08-14)

Third bite of the KV-bill class, and this one is the sizer's own half. `computeMemoryContext` built a `per_tok` out of two terms and both were wrong for a machine where the ceiling actually binds:

```
kv_per_tok  = config.kvBytesPerToken()          // always fp16, --kv-quant invisible
work_per_tok = 8 * max(hidden, ffn) * 2         // the PREFILL-CHUNK envelope, per TOKEN
```

On Qwen3.8-27B (hidden 5120, ffn 17408, 16 caching layers x 4 kv heads x head_dim 256) that is 64 KB/token of KV and **272 KB/token of activations** — 81% of the budget spent on a transient that does not scale with the context length at all. With a 10.6 GB working set and an 8.6 GB pack resident, the reported context was under 4k tokens, and no amount of shrinking the weights moved it, because the term that dominated never depended on them. The KV half compounded it: a server launched `--kv-quant 4` stores 18432 B/token, not 65536, and the sizer had no idea.

The activation term being chunk-bounded is not an argument from the code, it is measured. On the shipped 4-bit pack, peak-above-steady-state for a single request, `--prefix-cache-entries 0`, prompts from 3k to 51k tokens:

| chunk | 3.2k | 12.8k | 25.7k | 51.4k |
|---|---|---|---|---|
| 2048 | 3.35 GB | 3.35 | 3.33 | 3.34 |
| 8192 | 4.78 | 10.09 | 10.97 | 10.89 |

Flat in prompt length, and tracking the chunk once the prompt exceeds it (the 3.2k cells are narrower forwards — `fwd = min(chunk, seq)`). So it is a one-off RESERVE keyed on `--prefill-chunk`, subtracted alongside the hot-cache budget, never a multiplier on the context being solved for. `prefillTransientReserve` is the same `prefillMemoryNeeded` the admission guard calls, at the widest chunk any prompt can run (`effectivePrefillChunk(..., total_ctx = 0)` — every branch that narrows the chunk narrows it for LONGER contexts), with the KV term zeroed because the KV is the unknown. The KV half goes through `kvBytesPerTokenAtBits`, which both the sizer and the guard now call. Both reads are pinned by the existing call-site source scan, for the reason that scan already existed: an estimator that takes the right parameters proves nothing if a caller recomputes them on the way in.

Corrected, the same 16 GB profile at `--kv-quant 4` and a 512-token chunk reports ~51k tokens instead of 3.7k.

**What the same measurement said about the guard, which is fixed in the section below.** Fit the two rows above and the transient is `~0.8 GB + 1.24 MB per chunk-token`, while `prefillMemoryNeeded` billed `3 * 8 * chunk * max(hidden, ffn) * 2 * 5/4` = 1.04 MB per chunk-token and no constant: 2.14 GB against a measured 3.34 at chunk 2048, 8.55 against 10.95 at 8192. The sizer's own reserve is the same expression, so the two stayed consistent either way; the guard is a cross-arch number tuned over many releases, so retuning it got its own change with its own per-arch measurements.

## The prefill admission guard billed one arch's envelope for every arch (2026-08-14)

`prefillMemoryNeeded` models peak prefill memory and 400s a request that would not fit. Exceeding the Metal working set for real throws an uncatchable C++ exception on a completion-handler thread and kills the process, so this estimate is the only lever — and it estimated LOW on the archs that matter most.

Measured across five checkpoints on an M4 Max (peak GPU bytes above steady state for ONE request on a clean boot, `--prefix-cache-entries 0 --no-mtp --no-drafter --no-pld`, `/props` `peak_bytes` minus `active_bytes`, one prompt per boot because `peak_bytes` is a high-water mark that never resets). Repeat boots return byte-identical peaks, so these are exact figures, not samples:

| checkpoint | chunk 256 | 512 | 1024 | 2048 | 4096 | 8192 | old bill @2048 |
|---|---|---|---|---|---|---|---|
| lfm2 2.6B (conv hybrid) | 0.78 | 0.91 | 1.01 | 1.49 | 1.67 | 2.53 | 2.11 (1.42x) |
| qwen3_5 4B (GatedDeltaNet) | | 1.04 | 1.65 | 2.03 | | 5.11 | 1.51 (0.75x) |
| qwen3_5 27B (GatedDeltaNet) | | | | 3.98 | | 10.84 | 2.90 (0.73x) |
| gemma4 26B-A4B (MoE) | | | | 2.95 | 3.96 | | 3.20 (1.08x) |
| muse_glimmer 30B (dense) | | | | 2.18 | | 3.29 | 3.11 (1.43x) |

Seven of those sixteen cells were billed SHORT, the worst at 0.58x — every GatedDeltaNet cell, and the MoE at its own chunk cap. (lfm2 escapes only because `attnCacheLayerCount` has no notion of `layer_block_types`, so its KV is billed at 30 caching layers when 8 cache: a 3.75x KV over-bill covering a slope under-bill. Same class as the bailing_hybrid KV fix, not fixed here.) The per-chunk-token slope is the whole story, and it is not proportional to `max(hidden, ffn)` at all: 15.9 bytes per unit of ffn on lfm2, 9.5 on muse, 55.7 on the 4B, 72.6 on the 27B, 86.4 on gemma4. One envelope cannot be stretched to cover a 9x spread — over-billing muse 5x to cover the 27B is what the old constant was already doing, and it still came up 33% short.

**The hypothesis in the plan was wrong, and the kill switch is what said so.** The missing term was supposed to be the dequantized weight working set of a quantized checkpoint. Two experiments killed it: the same lfm2 in **dense bf16** peaks within 0.19 GB of the 8-bit build at both chunks (a dense checkpoint pays the constant too), and `MLX_SERVE_PREFILL_DQ_GEMM=0` accounts for exactly that 0.19 GB — +0.51 GB on the 27B at chunk 2048, +0.17-0.21 on lfm2, and ~0 at chunk 8192 where the envelope dominates. The dequant route is real, but it is a few hundred MB, chunk-independent, and only fires at forwards at least `PREFILL_DQ_GEMM_MIN_M` wide. It was never the 22%.

**What the excess actually is: streams the envelope does not model.** Subtract ONE MLP envelope (`8 x max(hidden, ffn) x 2` per token) from each measured slope and the remainder lands on the arch's own geometry:

- **Linear-attention hybrids hold one chunk-wide q/k/v stream per LINEAR layer** — all of them, not the ~3 the eval cadence bounds. The 27B: 48 GatedDeltaNet layers x (2x16x128 key + 48x128 value) elems x 2 B = 983,040 B/chunk-token against a measured excess of 985,172 (1.00x). The 4B: 24 layers x 8192 elems x 2 B = 393,216 against 366,044 (0.93x). Two independent checkpoints, both within 7%.
- **A MoE prefill sorts and gathers**, which replicates the hidden stream `top_k` times per layer beside the expert rows, for the 4 layers `MOE_EVAL_EVERY_N_LAYERS` lets coexist: `4 x top_k x 2 x (hidden + moe_intermediate) x 2 B` = 450,560 B/chunk-token on gemma4 against a measured excess of 396,688 (1.14x).
- **A plain attention arch has neither**, and its measured slope is at or below one envelope (lfm2 0.99x, muse 0.60x) — which is why the old three-envelope bill looked fine there and nowhere else.

So the envelope became `max(3 x mlp, mlp + fwd x prefillStreamBytesPerToken(config))`. The `max` is load-bearing: it is a FLOOR at the historical bill, so no arch's admission can loosen, and the stream arm only ever raises it.

**The chunk-independent part is a runtime floor, not a weight set.** Fitted intercepts across the five: 0.39 GB (27B), 0.67 (4B), 0.81 (lfm2), ~0.1 (gemma4), 1.27 (muse, most of it the dequant route). It does not scale with the weights in either direction, a dense checkpoint pays it, and it is what MLX's own scratch plus the KV cache's proportional capacity growth (old and new buffer coexist across a grow) costs. `PREFILL_RUNTIME_FLOOR_BYTES` bills it as what it measures as: a constant, 512 MiB, on both the guard and the sizer, and on the `deepseek_v4` sibling too (that arch was NOT re-measured — its own estimator was calibrated from a live false refusal, and the 2026-08-01 case still admits at 2984 MB against 3610 free).

Corrected, every one of the sixteen measured cells is covered, 1.06x to 3.7x, with the widest slack exactly where it was already widest (muse). The cost is real and it is the sizer's: on the 16 GB profile the reported context for the 27B at chunk 512 goes from ~51k tokens to ~18k, because the machine genuinely peaks 1.06 GB above steady state there. A `--prefill-chunk` the machine can afford buys it straight back, which is the honest lever.

Guards: `prefillMemoryNeeded: every MEASURED prefill peak on the box is billed for` (the table above, as assertions), `the new terms fire only where the measurement put them` (per-chunk-token vs per-prompt vs chunk-independent, and that a dense/non-affine checkpoint is billed nothing for dequant), `prefillStreamBytesPerToken`/`prefillDequantWeightBytes` unit tests, and the existing call-site scan extended to pin both new arguments at BOTH consumers.

## The prefill chunk was never sized to the machine, and a hybrid's KV was billed for every layer (2026-08-15)

Two defects, one symptom: 16 and 32 GB users running coding agents got
`Prompt (N tokens) requires ~XMB GPU memory but only ~YMB available` — or an
auto-context of 1024 — on prompts the box could actually serve.

### The chunk

`prefillMemoryNeeded` is dominated by the MLP envelope, `3 x 8 x chunk x
max(hidden, ffn) x 2`. `chunk` came from `--prefill-chunk`, which defaults to
8192 and which the app never passes, so a 16 GB Mac reserved the same 5-7 GB
envelope a 128 GB one does.

Measured (Mistral-7B-4bit, 10,348-token prompt, one request per boot,
`--prefix-cache-entries 0 --no-pld --no-drafter --no-mtp`, `peak_bytes` minus
boot `active_bytes`, M4 Max):

| chunk | measured peak | billed | ratio |
|---:|---:|---:|---:|
| 512 | 1.875 GiB | 2.61 GiB | 1.39x |
| 2048 | 2.262 GiB | 4.26 GiB | 1.88x |
| 8192 | 2.391 GiB | 9.18 GiB | 3.84x |

Slope 72,150 B/chunk-token against one envelope's 229,376 — the real envelope is
0.31 of ONE and we bill three. Coefficients across the six archs measured so far
(`measured / (8 x max(hidden, ffn) x 2)`): mistral 2.52, qwen3_5-4B 3.48,
qwen3_5-27B 4.54, muse-30B 4.77, gemma4-26B-A4B 5.40, lfm2 7.96. `mlp + fwd x
stream` covers every one of them; the `3 x mlp` floor is 1.7x-10x of it on plain
attention. That floor was NOT loosened here — it stays as the conservative
direction — but it is why sizing the chunk matters so much.

On a 16 GB Mac (Metal recommended working set ~11.9 GiB, Mistral-7B-4bit at
3.80 GiB resident, so 8.10 GiB free on a completely IDLE machine) the released
v26.8.6 billed 8.14 GiB for that prompt and refused it; the tree with the
runtime floor billed 9.18. Both against a measured 2.39.

`resolvePrefillChunk` now walks `PREFILL_CHUNK_LADDER` (8192 -> 512) and takes
the widest rung whose `prefillTransientReserve` is at most a QUARTER of what is
left after the weights and the hot-cache budget — past that the machine is
trading the whole session's context for one forward's speed. Frozen at load by
`pinPrefillChunk` (which must run BEFORE the sizer, above the `--ctx-size`
early-out, because the guard reads it on every request), and read by all three
consumers so bill and forward cannot drift. Projected advertised context:
Mistral-7B on 16 GB 1,024 -> 22,528; gemma3-12B on 32 GB 5,120 -> 20,480;
Muse-30B on 32 GB 1,024 -> 7,168; Qwen3.8-27B on 32 GB 1,024 -> ~42,000.

**The pin does not reach the forward through `xfm.config`.** `Transformer` holds
a COPY of the ModelConfig taken when it was built, and the pin is written to the
registry's config afterwards — reading it off the transformer is a silent no-op
(live: pinned 4096, prefilled at 8192, caught only because `--prefill-trace`
prints the width). It rides `InitOptions.pinned_prefill_chunk`, which the
scheduler sources from `slot.model.config` — the same object
`checkAttentionMemory` bills against. Source-scan-pinned both ways.

**A vision prefill is UNCHUNKED** (`generate`: `if (has_vision) loop_end`), so
the chunk-bounded envelope is not what it allocates. The guard billed the chunk
unconditionally, which was already an under-bill for a >8192-token vision prompt
and would have become one at every size once the chunk narrowed;
`checkAttentionMemory` now takes `unchunked_prefill` (from `local_ve != null` at
the three vision-capable surfaces) and bills the real width.

### The hybrid KV

`isLinearLayer` keys only on `full_attention_interval`. Two families never set
it — lfm2/lfm2_vl (`layer_types`) and nemotron_h (`hybrid_override_pattern`) —
so `attnCacheLayerCount` counted EVERY layer. LFM2.5-2.6B caches 8 of 30 and was
billed 3.75x (61,440 vs 16,384 B/token); Nemotron-H, whose `*` layers are a
handful of 52-98, worse. Only the `.attention` blocks reach `ctx.cache` in the
hybrid forward — gated_conv and mamba2 hold a fixed-size recurrent state in
`ssm_entries`. Independently confirmed by the peak table: lfm2 at chunk 512
peaked 0.782 GB total for a 10,089-token prompt, and the billed KV alone would
have been 0.62 GB of that. Layers past the 128-entry table keep the array's
`.attention` default (the over-billing direction).

### Not defects, checked

- **Multi-turn re-bills the resident KV.** `active_bytes` retains the prompt's
  KV in the hot prefix cache and the guard bills the full KV again on top;
  measured overshoot is only 8-11% (Qwen3.5-4B, three growing turns to 56,729
  tokens), and the re-bill doubles as cover for `nextCapacity`'s +25% growth
  double-buffer.
- **The advertised context is pinned at load; the guard reads LIVE memory.**
  Pressure arriving after load (another app, a second model, the hot cache
  filling) drops `available` below the bill while `/v1/models` still advertises
  the load-time number. That is why users saw the MEMORY 400 rather than the
  context-overflow one. Still open.
- **Under `--kv-quant` the guard bills the dense-rebuild term per PROMPT token
  while `prefillTransientReserve` bills it for one chunk** (1.44x disagreement
  on Qwen3.5-4B at 4 bits). Absorbed by the sizer's 0.544 compound margin today;
  it belongs in `per_tok`. Still open.

## opencode plan mode answered "What would you like to accomplish?" to every prompt: a content array's text parts were last-wins (2026-08-16, issue #195)

A user in `mlx-serve launch opencode` put the session in plan mode and every
first message was ignored — the model replied "I'll help you plan. What would
you like to accomplish?" as if the prompt were empty, while the same model on
LM Studio worked (issue #195; also reproduced locally — the SECOND plan-mode
message worked, which was the tell).

Opencode's `SessionReminders.apply` appends its "Plan Mode - System Reminder"
block as a SECOND synthetic text part on the LAST user message when entering
plan mode (and only then — later plan-mode turns get no reminder, which is why
"asking again" worked). The AI SDK openai-compatible provider ships a
multi-part user message as `content: [{type:"text",...},{type:"text",...}]`.
Our `/v1/chat/completions` parser read that array with
`if (text == .string) text_content = text.string;` — every text part
OVERWROTE the previous one, so the model saw only the reminder and never the
prompt. It then did exactly what the reminder asks with no task: offered to
plan. Build mode sends ONE text part, so it never fired.

Two more arms of the same class sat on `/v1/messages`: the top-level `system`
array took only the FIRST text block (`break :blk text.string` on first hit
— Claude Code sends identity + instructions as two blocks), and a
`tool_result` content array took only its first text block. The Responses API
parser already joined with `'\n'`, and the Anthropic user/assistant text
blocks already accumulated — the bug was per-arm, which is what made it a
class.

Fix: ONE collector, `server.joinedTextParts` — every `{type:"text"}` part
joined in order with `'\n'` (matching the Responses parser and the Anthropic
user arm), single-part borrows the JSON's bytes / 2+ parts allocate
(`{text, owned}`, the provenance rule), wired at all three sites. Guard:
unit tests on the exact opencode two-part shape.


## An adopted spec cache has ONE owner at a time (issue #266)

SIGSEGV in `KVCache.deinit -> freeKVEntry -> mlx_array_free` on the inference
thread, during long agent sessions with hot-cache hits + MTP/DFlash, typically
in a client-disconnect storm (repeated retries of a 100k+ prompt, each
cancelled mid-prefill).

`scheduler.runPrefill` restores the MTP history / DFlash context from the hot
prefix cache into locals guarded by `errdefer ... .deinit()`, then hands them
to `Generator.initWithOptions` — which adopts them and ALSO guards them with
its own `errdefer`. Any init failure past the adoption point (the common one:
`error.Cancelled` from the chunk loop when the conn thread flags the slot on
disconnect) frees them inside the generator; the `try` then unwinds
runPrefill's errdefers and frees them AGAIN. `MtpCacheRef` and `DflashCtx`
hold their `KVCache` by value, so both copies share one entries slice and the
same mlx handles: the second `freeKVEntry` walks freed array ctxs — an
EXC_BAD_ACCESS several frames from the disconnect that caused it (the
`KVCache.reinit` class again, in cross-function form). It only bites on the
restored path — a hot-cache MISS builds the caches inside the generator, where
one errdefer owns them.

Fix: ownership transfers AT THE CALL. runPrefill moves the restored caches
into `dflash_pass`/`mtp_pass` and nulls the errdefer-guarded locals BEFORE the
`try`, so on failure exactly one owner (the generator) frees them. Guard:
`scheduler.zig` test "runPrefill clears restored spec-cache ownership BEFORE
Generator.initWithOptions (issue #266)" pins the clear-before-call ordering.

## `messages.deinit(allocator)` frees the Message array and NOTHING it points at

Live 2026-08-25. `handleChatCompletions` and the Anthropic `/v1/messages`
handler each decoded a request's `images[].pixels` (and, on the OpenAI surface,
`videos[].pixels` and `audio[].samples`) into a local `std.ArrayList`, handed
the buffer to `chat_mod.Message` with `toOwnedSlice`, and then only ever ran
`defer messages.deinit(allocator)`. That defer frees the Message array. The
decoded CHW buffers each Message points at had no owner at all, so EVERY
request carrying an image leaked its full decoded pixel buffer on the SUCCESS
path — and on every early return after the parse loop too (context-overflow
400, the memory preflight's 503, a client disconnect mid-stream). Measured on
the shipped binary: 40 chat requests each carrying a 3 MB `x-mlx-pixels`
payload grew RSS by 121 MB, i.e. exactly the payload, retained forever.

Only the Anthropic path had a partial guard: an `errdefer` covering the
`try`s between the decode and the `messages.append`. It covered the ERROR path
and not the success one, which is the inverse of the usual bug and is why the
leak survived review — the file looked like it was already thinking about
ownership.

The Responses API was already correct: `responses.ParsedInput` carries an
`owned_images` list and frees `pixels` in its `deinit`. That is the shape the
fix generalizes.

Fix: `server.RequestMedia`, one owner per request, installed beside the
`messages` list in both handlers. Ownership is by PROVENANCE — a media list is
only obtainable from `openImages`/`openVideos`/`openAudio` and is owned from
its first append, so a `try` between the decode and the message append cannot
leak either, and a new early return cannot leak by omission. `Message` borrows
the slice (`imagesSlice` returns null for an empty slot, so a message with no
media keeps a null field rather than an empty slice). Slots are INDICES, not
pointers: opening another slot may move every `ArrayList` header in the bag,
while the buffers those headers point at stay put. A fourth modality that
spells its buffer something other than `pixels`/`samples` fails to COMPILE in
`mediaBuffer` rather than leaking quietly.

Guards: the source scan "a request handler never owns its own media list —
RequestMedia does" (no `std.ArrayList(chat_mod.{Image,Video,Audio}Data).empty`
anywhere in server.zig, and at least two `RequestMedia.init(allocator)` sites),
plus the `std.testing.allocator` unit test "RequestMedia frees every decoded
buffer it was handed" — gut the free in `MediaBag.deinit` and it reports 4
leaked allocations.


## Vision prefixes in the prefix cache are keyed on the pixels (2026-08-26)

`commitSlotIfApplicable` and the lookup both returned early on `vision_embeddings != null`, so an image conversation re-prefilled everything on every turn. The KV under image placeholder tokens is a function of the pixels, and two different images produce IDENTICAL token sequences, so a plain token-prefix match would restore the wrong image's rows. The fix keys every RAM entry on `vision_key` beside `has_tools`: `server.mediaKey` hashes the request's image/video/audio bytes in `processVisionImages` (0 = text only), the key rides `SubmitParams` → `Slot` → `commitWithState` / `lookupAndRestore` / `findBestMatch`. A text entry never serves an image request and vice versa (no partial-prefix sharing across the two — deliberate, KISS). Vision entries stay in RAM: the SSD tier's manifest has no key column and is never consulted for `vision_key != 0`.

The second half is the splice: a restored prefix already holds its image rows, so the Generator's placeholder-row counter starts at `countSpliceRows(full_prompt[0..hot_matched])` (`InitOptions.vision_rows_before`) and `vision_splice_offset` is set on BOTH the chunk loop and the final-span forward whenever the request has vision, not only under chunked-vision. M-RoPE positions/delta are re-derived per request from the same images, so nothing else needs to ride the entry.

Guard: `tests/test_vision_prefix_cache.sh` (not yet run live as of 2026-08-26) + the `vision_key` arms of the `findBestMatch` unit test.

First live run on qwen4_exp: `[hot-cache] hybrid miss (no checkpoint <= 514 of 514)` — the tokens matched, but `shouldCheckpointSsmPrefill` still returned false for any vision prompt (from before #197, when vision prefilled in one un-chunked forward and had no boundary to snapshot at). It now follows `visionChunkedPrefillEnabled()`. After that: 483/514 reused, the warm answer moved by one token (the hybrid restore class, so the script asserts content, not bytes).

## A missing tensor ended the process (issue #217, 2026-08-28)

`transformer.zig`'s weight getters logged `MISSING WEIGHT: <name>` and hit `unreachable`. In ReleaseFast that is a process exit: one checkpoint the loader cannot read (a converter's tensor naming we don't probe, one shard short from a bad download) took the server down with every request queued behind it — three times in one reporter's benchmark run. The scheduler already crosses load failures by name (`req.error_name` → `loadErrorFromName` → 503), so the fix is only that the getters return `error.MissingWeight` and the ~200 call sites `try`. Guard: `test "a missing weight is a load ERROR, not a process exit (issue #217)"`. Not covered: a weight that is PRESENT but the wrong shape still dies inside MLX (uncatchable, see the Metal OOM story).

## `--max-resident-mem` billed shards nothing reads (issue #274, 2026-08-28)

`scheduler.modelDiskBytes` summed every `*.safetensors` in the directory. A third-party gemma-4 E4B pack shipped two shards no `weight_map` entry references; the bill was 2x the loaded size, so loading a small image model evicted the chat model. The index is the truth when present: `indexShardSet` reads `model.safetensors.index.json` and only named shards count. Guard: `test "modelDiskBytes bills only the shards the index names (issue #274)"`.

## The edit form dropped LoRA fields (issue #268, 2026-08-28)

`gen.openaiEditFormToJson` rebuilds the multipart body into the JSON `mode:"edit"` request field by field; `lora_paths`/`lora_scales` were not in the list, so a client attaching adapters through the OpenAI surface got an un-adapted edit with a 200. Now forwarded verbatim (array forms as raw JSON text, scalar `lora_path` JSON-escaped) so `parseLoraFields` sees the same body the native endpoint would. Guard: the lora case in `test "openaiEditFormToJson: OpenAI multipart becomes our edit request"`.

## A llama session trim that was never checked served the previous request (#286, 2026-08-28)

Reported as "PLD leaks tool calls across requests": a request offering ONE tool (`read`) came back calling `bash` with the previous session's `cd /private/tmp/mlxcode && pytest` command, on a qwen35moe GGUF and on a Flash-Next pack; `--no-pld` "fixed" it. Reproduced on a Qwen3.5-0.8B GGUF with `--no-pld` already set: request A (bash+glob tools), then request B (read only) three times, B answered `read` once and then A's `bash` call twice, with the log showing `285 cached / 286 total`.

The mechanism is the persistent llama session's prefix reuse. `LlamaSession.sync` computes the common prefix, calls `mlx_llama_session_trim(common)`, shrinks its resident-token mirror and decodes the suffix. The shim called `llama_memory_seq_rm(mem, 0, n_keep, -1)` and ignored the result, on the comment "removing a whole tail never fails". That is true for attention KV and false for recurrent memory: `llama_memory_recurrent::seq_rm` can roll a tail back only within its per-token snapshot window (`n_rs_seq`, one or a few tokens) and otherwise returns false having mutated nothing; `llama_memory_hybrid::seq_rm` tries the recurrent half first and bails before touching the attention KV. So after any generated tail longer than the window, NOTHING was trimmed, `pos` and the mirror said it was, and the new suffix was decoded at positions the old tail still occupied, under a recurrent state that had read the old conversation. The model continued the old conversation. Every hybrid GGUF arch is exposed (qwen35, qwen35moe, qwen3next, nemotron_h, lfm2, any Mamba); dense GGUFs were fine, which is why the existing warm-reuse test never saw it.

Fix: the shim checks the return and, on refusal, clears the memory and returns 1; `sync` reads 1 as "nothing resident" and cold-prefills the whole prompt. On a hybrid that means prefix reuse only survives when the tail is inside the snapshot window (rarely), which is the correct trade: a cold prefill costs milliseconds, a poisoned state costs the user's trust in the model. The MLX-pack half of the report (Flash-Next) is a different path and was not reproduced here; the GGUF half is closed by this fix. The PLD attribution was coincidence: the llama tick has no PLD at all. Issue #287 ("streaming drops tool calls", `stream=true` answered `stop` with empty content) is the same bug seen from the other side: on the reporter's Ornith-1.5-35B (qwen35moe) the SECOND request on the release tarball logged `280+0 tokens (279 cached)`, the model hitting EOS at once on the poisoned state, and the fix makes four streamed requests in a row answer the call.

Test: `llama: re-sync after a long generated tail is a cold decode` (needs `LLAMA_TEST_MODEL` pointing at a HYBRID GGUF, e.g. Qwen3.5-0.8B) greedy-matches a fresh session after a 30-token junk tail; the older `prefix reuse is byte-identical` test no longer demands `cached > 0`, since 0 is the right answer on a hybrid.

## A chained video render finished and died at the socket (#283, 2026-08-28)

MiniMax-H3, Turbo, 1056x864, 141 frames x 5 chained windows: every window sampled and decoded, `chain joined: 5 windows -> 701 frames`, then `[video] -> 701f 864x1056 (1918743552 rgb bytes)` and `video job failed: WriteFailed`. `WriteFailed` is the socket: the app hung up on a 2.5 GB base64 body. The app already knows a response can carry at most `maxFramePayloadBytes` (768 MB raw) and trims the frame picker to it, but the bill was per WINDOW: `chain_windows` multiplied the delivery and nothing looked. The server had no cap at all, so 29 s of GPU time went into a body nobody could receive.

Fix: `gen.videoRgbTransportReason(delivered, w, h)` with `MAX_VIDEO_RGB_BYTES` (the app's number) refuses at admission on both video paths, naming frames, canvas, MB and the cap; the H3 site bills `chainDeliveredFrames(windows, frames)`. The app's `frameOptions` takes `chainWindows` and the stepper stops at 6 (the server refused 7-8 anyway). Lifting the cap is a transport change (stream frames or mux server-side), not a number to raise.

Also filed the same day, #285: a Mage-Flow-Edit pack failing `MissingMageFlowWeight model.visual.patch_embed.proj.weight`. The reporter's `text_encoder/model.safetensors` loaded 902 tensors; the published Edit pack's has 1425 (523 `model.visual.*`), 902 is the TURBO text encoder. A pack-content problem on the user's disk, not a loader bug; the log line was relabeled from `MISSING VAE WEIGHT` to name the file and both counts.

### A tool-response turn can render under the USER marker (2026-08-30)

PR #318's `activeTurnMediaMessage` counts the user-role messages after the media
message so `userTurnInsertPos` can pick the marker that opens the media turn
instead of the last one. That count assumed one rendered user marker per
user-ROLE message — but ChatML-family templates (Qwen 3/3.5/3.8) wrap each
tool-response RUN in its own `<|im_start|>user`, merging consecutive tool
messages into one wrapper, and the match is token-exact: `<tool_response>` is a
special token, so the marker's trailing `\n` stays its own token (`[im_start,
"user", "\n"]` appears verbatim). With `[user(image), assistant(tool_call),
tool(result), user(context)]` the role count says one marker after the media
message while the render has two, and the image pads land after the tool
response. Llama renders tool results under an `ipython` header and adds no
user marker, so a fixed per-family rule in the counter would just move the bug.

`server.resolvedUserMarkersAfter` lets the rendered prompt arbitrate: it counts
the marker's occurrences in `prompt_ids` and compares against the conversation's
role totals — `users + tool_runs` (ChatML, runs merged), `users + tool_msgs`
(per-message wrapping), or `users` (no tool marker). The matching convention's
tool count is added to `user_markers_after`; an unrecognized total (template
merged or dropped something) falls back to the role-only count, which is the
pre-fix behavior. Guard: `insertMultimodalTokens counts a ChatML tool-response
user marker` (server.zig) pins all three conventions by insert position.

## The sleep-inhibition gate sat in the tick that never runs (issue #251, 2026-08-30)

A long render needs sleep protection only while work is active. Use `PreventUserIdleSystemSleep`; display sleep remains allowed.

Release immediately before `queue_cond.wait` and acquire after wake. The startup load runs before that loop, so it acquires separately. The deferred release covers shutdown and load failure.

`tests/test_sleep_inhibit.sh` checks idle, active generation, return to idle, startup load, and `--no-prevent-sleep`. It matches the assertion name because powerd uses the same assertion type.
