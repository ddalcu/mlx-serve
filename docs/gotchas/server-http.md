# Server, HTTP & streaming lifecycle — war stories (moved out of CLAUDE.md)

Full histories: live failures, measurements, diagnosis ladders, dead ends. The distilled RULES live in the root CLAUDE.md "Rules" section — when a rule changes, update the story here too. New gotchas in this domain: add the 1-3 line rule to root, the full story here.

### Historical images decoded on every text-only continuation (2026-08-30)

The active-turn media fix stopped the vision tower from re-encoding images behind the latest assistant boundary, but both chat parsers still eagerly base64-decoded, JPEG-decoded, resized, normalized and patchified every attachment before that selector ran. A Harness session retaining 24 images therefore logged 24 image decodes on every later text-only request. Qwen's 44x44 patch grid retains about 8.7 MiB of preprocessed float data per image until the request ends, so the request also carried roughly 200 MiB of avoidable transient buffers. Warm prefix reuse hid most of the latency at 24 images, but the CPU and allocation work grew linearly with conversation history and multiplied under concurrency.

The fix performs a metadata-only pass over the parsed JSON tree first. `activeWireMediaIndex` mirrors `activeTurnMediaMessage` across ordinary assistant boundaries, assistant-prefix continuations and tool-call/result chains, with separate OpenAI and Anthropic wire shapes. The handler's existing parse loop then materializes attachments only for that selected raw message; historical data URLs remain borrowed JSON strings and allocate no media buffers. Skipped image-only history must still append its empty user message, because the role boundary is part of the rendered prompt even when its pixels are not active.

The HTTP regression sends one historical image, twenty historical images, an image-only historical turn and an Anthropic historical image. All must complete with zero new `Decoded … image` log lines; the image-only case also compares prompt-token counts against a dropped empty turn so preserving the boundary is observable. Fresh images, trailing Harness context, assistant-prefix continuation, growing image conversations and changed-image prefix reuse remain covered in the same script.

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

`tests/test_sleep_inhibit.sh` checks idle, active generation, return to idle, startup load, and `--no-prevent-sleep`. It matches the assertion by owner pid AND name: powerd holds the same assertion type, and the bare name matches any other mlx-serve instance on the box (a live server generating concurrently false-fails the idle/opt-out arms).

## The hot-cache budget was never validated against what the weights left (2026-08-30)

A long agentic run on the ~70 GB Flash-Next pack (143k-token prompt) killed the
server with an uncatchable Metal OOM (`kIOGPUCommandBufferCallbackErrorOutOfMemory`).
The memory never fit: weights + a 40 GB `--prefix-cache-mem` budget (32.5 GB
resident at death) + the live 143k KV and prefill transients against the ~96 GB
Metal working-set limit on a 128 GB Mac. The registry had refused the load at
the 64 GB cap; the app's `--skip-mem-preflight` let it through, and nothing
anywhere compared the cache budget to the headroom the weights left.

Fix: `server.clampedPrefixCacheMem` (pure) caps the budget at
`gpu_ceiling − (weights + ctx KV + prefill transient reserve)`. The impure
wrapper `prefixCacheMemForLoad` runs at the load site in
`scheduler.doLoadOnInferenceThread` — weights are resident there, so
`mlx_get_active_memory` is honest — reached through a
`prefix_cache_mem_resolver` fn pointer because the scheduler deliberately has
no server.zig import (the `ane_chunk_resolver` pattern). `requested == 0`
(byte cap disabled) is bounded too — an uncapped cache beside a large model is
exactly this crash — and the clamp never returns 0, because `initWithMem`
reads 0 as "no byte cap". When it bites: `[hot-cache] budget clamped … MB`.
Crash-case numbers: 96 − 70 − ~6.4 (262k ctx KV) − ~4 ≈ 15 GB instead of 40.

Follow-up correction: RAM restore does **not** materialize a copy.
`KVCache.restore` rebinds the destination handles with `mlx_array_set`, so the
slot refcount-shares the entry's buffers until a later grow allocates its
destination buffer. The admission calculation already covers both sides:
`active_mem` includes the resident entry, and `prefillMemoryNeeded` bills the
full destination KV capacity. Adding the largest entry once more invented a
third copy and falsely rejected a 138k warm prompt — as well as cache misses,
because the guard ran before lookup. Do not add resident cache bytes to
`needed`; they belong only in the active-memory side of the equation.

The byte budget is also a hard cap now. Previously the eviction loop emptied
the cache but appended an oversized candidate once no entries remained, so a
single long conversation could exceed the load-time clamp. An entry larger
than `max_kv_bytes` is trimmed to the longest restorable prefix that fits
(see the #330 story below); only a candidate with nothing above the floor is
declined, preserving any existing smaller prefix.

Guards: `clampedPrefixCacheMem` unit test (the crash's numbers), the
`initWithMem`-site source scan in scheduler.zig (a second construction site
can't skip the clamp — the cold-load flags class), and the oversized sole-entry
regression in prefix_cache.zig.

App-side twin: the crash banner showed `> "9:23:51 PM [vite] Pre-transform
error: …"` — the agent conversation's own request-preview log line — because
`summarizeCrash`'s `error:` needle matched it (the Metal marker only reaches
os_log, not stderr). Preview lines (`> "` prefix) are now skipped by both the
needle scan and the last-line fallback (`isRequestPreviewLine`).

## An oversized hot-cache decline is a cliff, not a cap (#330)

#326 made `--prefix-cache-mem` a hard cap by declining any commit whose
snapshot exceeds the budget. In a long agent session that is the steady state,
not an edge case: the conversation's KV crosses the budget once mid-session
and from that turn on NOTHING commits — the reporter's log showed 9/9 requests
skipped, zero `reused`/`resident=` lines, ~23 minutes of re-prefill in one
window, a cap "enforced" by holding zero bytes. Reproduced locally on
Qwen3.5-2B at `--prefix-cache-mem 400MB`: pre-fix the cache froze at turn 1's
21k-token entry while prompts grew to 107k.

The fix costs the prefix, not the whole entry:

- `trimLenForBudget` (prefix_cache.zig) picks the longest retainable length.
  Hybrids must cut at a CHECKPOINT's `pos` (a KV-only hybrid prefix restores
  as a cold miss while occupying an LRU slot) and the cost includes every
  checkpoint kept. Plain attention prices tokens directly. Floor:
  `MIN_CANCELLED_COMMIT_TOKENS`. Media entries cap the trim at `media_start`
  — trimming INTO placeholder rows is not a shape we reason about.
- `KVCacheSnapshot.trimmedCopy` (transformer.zig) is a REAL slice + copy per
  array against its own shape, batch-evaled. `KVCache.truncate` is
  offset-only and a refcount-shared snapshot bills the parent's CAPACITY —
  an offset trim would be an accounting fiction that retains the full 6 GB.
- Spec payloads (dflash/mtp snaps) describe the full-length state and are
  dropped on trim; the first reused turn rebuilds them.
- ONE-SHOT: when the resident covered entry already retains ≥ the trim
  target, the candidate is dropped and the entry kept. The target is
  budget-derived and stable, so without this every turn re-copies an
  identical multi-GB prefix. (On hybrids the steady state usually converges
  via the decline arm instead — the candidate's own checkpoints all sit
  above the restored prefix — which preserves the resident entry too.)

Two adjacent defects from the same report:

- The replace path could evict its own sole entry: inherited SSM checkpoints
  push the merged entry over a cap the pre-check passed (it only prices
  `new_bytes`), and the old loop evicted down to empty — commit → evict
  everything → cold prefill, every near-budget turn. Now it evicts OTHER
  entries first, then `shedCheckpointsToFit` drops this entry's checkpoints
  (interior thinning, same rule as the merge cap); sole-entry eviction stays
  as the last resort that keeps the load-time clamp real.
- `ssm_cps` double free: the commit's dupe/append error paths freed the
  checkpoints AND the scheduler's catch arm freed them again — with a
  different allocator. Contract now: ownership transfers to the cache on
  EVERY outcome (after a trim the slice may be a cache-allocated
  replacement, so only the cache can free correctly); the scheduler frees
  nothing, and `commitCancelledPrefillSlot` detaches the salvage BEFORE the
  call so `Slot.deinit` can't free what the cache owns.

Guards: the `#330` unit tests in prefix_cache.zig (attention + hybrid trim
with byte-exact restored rows, one-shot handle identity, no-fit decline,
shed-not-evict, FailingAllocator ownership) and the catch-arm/detach source
scan in scheduler.zig. NOT built: the issue's proposed `--prefix-cache-mem
auto` context-sized floor — the default-sizing question is still open.

## The trim #330 promised never fired: retention had end-anchored the checkpoints (#330 follow-up, 2026-09-05)

**Symptom.** qwen4_exp, SSM checkpoint stride 4096, a long agent session. The
commit of a 383,069-token entry logged

    [hot-cache] skipped oversized entry (383069 tokens, 8757.79 MB > 3873.54 MB budget)

— the FLAT DECLINE #330 exists to prevent. Every later turn cold-prefilled
while the cap "held" zero bytes, exactly the cliff #330 was written for.

**Three defects, one line.**

**1. Retention was end-anchored.** Both capture sites in `generate.zig` (the
stride capture in the chunk loop and the end-of-prompt snap after it) honoured
`ssm_checkpoint_max` with `orderedRemove(0)` — drop the OLDEST. With max 16 and
stride 4096 the survivors of a 383k prefill are the highest 16 positions, i.e.
they cover only the last `max * stride` = 65k tokens; the LOWEST survivor sits
around 320k, and 320k rows at 13,056 B/token plus its checkpoint is ~4,004 MB
against a 3,873.54 MB budget. `trimLenForBudget` walks the list downward
looking for an affordable position, finds none at all, and returns null — the
decline. Thinned span-preservingly instead, the same 16 survivors span the
whole prompt (4,096 … 383,039) and a trim point like 126,976 = 31 x 4096 costs
about 1.73 GB. The disk tier (`ssmTargetPositions`) thinned from the front for
the same stated reason and had the same consequence for a restore that
diverges early.

The hot cache had ALREADY solved this twice — `mergeCheckpointLists` and
`shedCheckpointsToFit` both thin the interior, and the merge comment even
names the 415 s cold prefill that taught it. Two policies for one decision is
how the third site kept the old one, so the selection is now ONE pure helper,
`transformer.spanPreservingDropIndex` (typed as `ssmCheckpointDropIndex` /
`positionDropIndex`), and all four sites call it: keep index 0 (a prompt that
diverges early restores only there) and the last (where warm turns match),
drop whichever interior checkpoint sits between the closest pair of
neighbours. Under three there is no interior, so the oldest goes.

**2. The trim billed checkpoints the commit sheds anyway.** At candidate `k`
the cost was `p * row_bytes + Σ bytes(list[0..k+1])` — every LOWER checkpoint
at full price, although the commit path's own `shedCheckpointsToFit` thins
them the moment the entry lands over the cap. The bill is now what SURVIVES a
span-preserving shed to the remaining allowance (`shedSurvivorBytes`, the same
selection helper, simulated on stack arrays; a list past `SHED_SIM_MAX` falls
back to the old all-lower bill, which can only pick a shorter prefix). The
promise has to be kept on the other side too: the replace path already ended
with `shedCheckpointsToFit`, and the NEW-entry path now does as well.

**3. One line for three outcomes.** `trimLenForBudget` returning null, the
`trimmedCopy` failing, and the trimmed checkpoint list's `dupe` failing all
printed the identical `skipped oversized entry` line, and the two failures
swallowed their error entirely — so the live log above could not be read as
"the arithmetic declined" versus "a copy failed". `TrimDecline` now names
which, with `@errorName` appended, and a failed `trimmedCopy` retries at the
next-lower checkpoint (`limit = tl - 1`) before declining rather than treating
one width's failure as a verdict on the entry. The `dupe` arm leaks nothing —
`new_snap` is by then the trimmed copy and the decline path deinits exactly
that — but it throws the copy away, which is worth its own line.

Not changed, worth knowing: `new_bytes` bills the slot's RESERVED capacity
rows, not the token count, so the MB in that line reads high for a slot that
grew its KV buffer past the prompt.

**The count is a spacing decision, priced against the tier (follow-up).** Span-preserving
survivors sit ~`L/K` apart, so a warm turn that diverges BETWEEN two of them re-prefills that gap:
the policy trades an unbounded loss (no checkpoint at or below the match — a full cold prefill)
for a bounded one. The disk tier's `SSM_DISK_MAX_PER_ENTRY` was the weakest cell at K=8, and a
checkpoint is not a constant — on qwen4_exp it measures 83 MB + ~3 KB per token of its own
position (191.3 MB at position 36,864), so the bill grows with K and with where the survivors sit.
At a 383k entry, 4 entries in the tier, 900 tok/s:

| K | cps/entry | 4 entries (+KV) | max spacing | worst-case re-prefill |
|---|---|---|---|---|
| 8 (was) | 5.6 GB | 41 GB | ~54,700 tok | ~61 s |
| **16 (is)** | **10.6 GB** | **61 GB** | **~25,500 tok** | **~28 s** |
| 24 | 15.6 GB | 81 GB | ~16,700 tok | ~19 s |
| 32 | 20.7 GB | 101 GB | ~12,400 tok | ~14 s |

K=32 does not fit a 100 GB tier at all, and K=24 — the smallest holding a ~16k spacing — leaves
19 GB for a tier that must also hold every other entry and not thrash. 16 halves the worst case at
61% of the tier. The RAM tier's own cap (32) is unchanged: its checkpoints are already resident,
so the count costs residency, not disk.

**An even spread is the wrong shape at the end (audit S15a).** Thinning the whole interior
uniformly makes a warm turn that EDITS NEAR THE END restore up to a full spacing back, where
drop-oldest restored one stride back — the opposite trade from the one the 383k measurement
covered, and it binds on every hybrid arch (lfm2, nemotron_h, qwen3_5, bailing_hybrid), not just
qwen4_exp. `spanPreservingDropIndex` therefore never selects from the newest QUARTER: that part
stays at capture density and everything below it is thinned span-preservingly. Measured on the
383k shape at K=32, the last gap goes 10,303 -> 2,111 tokens while the widest gap grows
16,384 -> 20,480 — about 9 s bought at the end for 4.5 s given up in the middle at 900 tok/s. Both
halves are pinned: the retention test asserts stride-spaced final gaps AND a front gap many
strides wide, so neither an even spread nor a return to drop-oldest passes.

Guards (hermetic, no engine): `transformer.zig`'s span-preserving retention
test (94 positions at 4096 thinned to 16 — both ends kept, no gap past twice
the ideal), and in `prefix_cache.zig` the 383k arithmetic both ways
(end-anchored survivors → null, thinned survivors → a trim ≥ 126,976 that
fits once shed), `shedSurvivorBytes`, the retry-lower selection, the three
distinct `TrimDecline` reasons, and a class scan that no retention site has
gone back to drop-oldest. `kv_disk_cache.zig`'s retention test changed sides
with the policy: it asserted the lowest position was dropped and now asserts
both ends survive.

## The schema thinking-off gate lived on one surface of three (#331, 2026-08-31)

A JSON-schema grammar mask constrains from token 0 and cannot express "think
first, then JSON". On templates that end the rendered prompt inside a bare
`<think>` block (qwen3.5/3.8), `</think>` is not valid JSON, so the model
emits the schema-valid object inside the reasoning block: `reasoning_content`
carries the JSON, `content` ships empty, streaming routes it through
`delta.reasoning_content`.

`/v1/messages` got the rule in the output_config fix (live: qwen3.5, effort
high + schema): schema + no tools ⇒ thinking forced OFF in the prompt (the
noThinkTailSuffix machinery). `/v1/chat/completions` and `/v1/responses`
built the same mask with no gate — issue #331 re-found the identical symptom
via `reasoning_effort` + `response_format`.

Fix: one predicate (`server.schemaMasksThinking`) consulted at all three
mask-building sites. Tools present = no mask on every surface (tool calls
must stay reachable), so thinking stays whatever the request resolved.
"Real reasoning then schema-valid JSON" would need a mask that arms only
after the think block closes — not built; schema stays a content-only
contract.

Guards: `tests/test_json_schema_thinking.sh` (all three surfaces + stream arm
+ mask-engagement count) and the server.zig source scan pairing every
`[grammar] enforcing` site with a gate call.

## A 458k prefill killed the server, and the fatal part was mlx-c's default error handler (#353, 2026-09-03)

**Symptom.** Apple M5 Max 128 GB, Qwen3.8-Flash-Next `mixed-4-8bit`, main at
`fa960f1`, `--ctx-size 1048576 --kv-quant 8 --mtp --mtp-depth 5
--prefix-cache-entries 4 --prefix-cache-mem 24GB`. A cold prefill of 458,832
tokens passed the memory preflight (billed ~18.1 GB against 26.6 GB
available) and then the PROCESS died mid-prefill with

    Command buffer execution failed: Insufficient Memory
    (kIOGPUCommandBufferCallbackErrorOutOfMemory) at transforms.cpp:15

No 503, no connection close, every other in-flight request gone. The rung
before it — 393k, cold — had completed at `peak_bytes` 103.0 GB under a
103.4 GB ceiling, and its own 6478 MB hot-cache entry was resident when the
458k request was admitted.

**It was filed as an uncatchable Metal abort. It was not.** The tell is the
suffix: `" at %s:%d"` is appended by mlx-c's own `_mlx_error`
(`lib/mlxc-src/mlx/c/error.cpp`), and the file it names is
`mlx/c/transforms.cpp` — `mlx_eval`. So the exception had already been caught
at the C boundary. Reading the pinned MLX (0.32.2) end to end:

* a failed command buffer's status is stored into `CommandEncoder::error_` by
  the completion handler and re-thrown from `synchronize()` /
  `get_command_encoder()` — `mlx/backend/metal/device.cpp:518-579`;
* an exception raised on a stream thread is captured into
  `StreamThread::error` and re-thrown on the main thread at the next enqueue
  — `mlx/scheduler.cpp:77-79, 99-112`;
* every mlx-c entry point is
  `try { … } catch (std::exception& e) { mlx_error(e.what()); return 1; }`.

What killed the server is the handler nobody had replaced:

    static void mlx_error_handler_default_(const char* msg, void* data) {
      printf("MLX error: %s\n", msg);
      exit(-1);
    }

`mlx.installErrorHandler` (once, from `main()`, beside `applyMlxCacheLimit`)
latches the message and returns; `mlx.checkError()` turns the latch into
`error.OutOfMemory` (memory class, by message) or `error.MlxFailure`, checked
once per prefill chunk. MLX clears its own error state as it throws, so the
next request starts clean. `MLX_SERVE_MLX_ERROR_LATCH=0` restores `exit(-1)`.

**What the guard could not see.** Two terms, both proportional, both billed as
constants or not at all:

1. *Growth coexistence.* `KVCache.growQuantBuf` allocates the whole new
   capacity with `mlx_zeros` and slice_updates the old buffer into it; the QSA
   indexer history is a `mlx_concatenate_axis(old, new)` per chunk per
   full-attention layer. Both leave the old buffer live in the chunk's lazy
   graph. `PREFILL_RUNTIME_FLOOR_BYTES` claims this in its own doc comment and
   prices it at a flat 512 MB — measured as an intercept on prompts of a few
   thousand tokens. At 458,832 it is ~7.75 GB.
2. *Retained SSM checkpoints.* 36 GatedDeltaNet layers x (state + conv) =
   58.8 MB each, up to `--ssm-checkpoint-max` (32) held while the prefill
   runs: 1.88 GB, billed at zero, and saturated at both 393k and 458k.

**Why the obvious fix was parked.** `parked/admission-guard-rewrite` bills the
growth copy at the worst chunk. It is correct and it refuses the request that
died — and it roughly HALVES the maximum admissible context on every arch,
because the peak is then modelled as two copies of everything the prefill has
accumulated. Pricing a transient you can delete is the wrong trade.

**What shipped instead.** Remove the transient, then let the cache pay:

* `KVCache.reservedTokens` (prompt + max_tokens + one chunk, only past
  `RESERVE_MIN_TOKENS` = 32k) is reserved before the first chunk writes, so a
  long prefill grows its buffers exactly ONCE and nothing coexists. The guard
  bills that reservation's headroom — tens of megabytes — instead of a second
  copy of the cache. Short prompts keep the proportional policy untouched.

  **`MLX_SERVE_KV_RESERVE=0` restores the ALLOCATION policy, not the bill**
  (audit N3, correcting an earlier draft of this line that said it "restores it
  everywhere"). The switch is read in `KVCache.nextCapacity`, so with it off the
  cache returns to proportional growth; but `server.reservedCacheTokens`
  delegates to `KVCache.reservedTokens`, which does NOT read it, so the
  admission guard keeps billing the reservation either way. The direction is
  conservative — the bill exceeds what the engine then allocates, so a request
  is refused slightly early rather than admitted and OOMed — but it is not the
  byte-for-byte return the line promised, and a wrong restore claim sends the
  next person hunting the wrong file when the numbers do not match.
* `retainedSsmCheckpointBytes` and `statePerTokenBilled` join the estimator,
  the latter read by the auto-context sizer AND the guard so advertised and
  admitted contexts cannot diverge. The QSA history was billed at TWO copies
  while it re-concatenated per chunk; with the capacity-buffer append in the
  same tree (`qsaAppendKeys` -> `capBufAppend`) the bill is ONE copy. That
  halving is not cosmetic: at
  `--ctx-size 1048576` on qwen4_exp the doubled bill clamped a requested
  24,576 MB hot cache to 5,703 MB, and the single copy gives most of it back.
  (Later: the prefill-end checkpoint attach turned out to MATERIALIZE a second
  copy anyway, so 63cf6bd re-doubled the bill; the commit-time handoff of
  2026-09-05 removed that copy and the bill is one copy + the f32 score bank —
  story in `engine-mlx.md`, "HELD twice from prefill end to commit".)
  The bill reads the two LEVERS that decide the copy count (the commit handoff
  and the reservation), not the shape of the append: a `QSA_HISTORY_GROWS_IN_PLACE`
  flag once stood here and was removed because a fact about the append cannot
  say whether a second copy exists. If the append ever returns to
  `mlx_concatenate_axis(old, new)`, `statePerTokenBilled` is what has to move.
* `HotPrefixCache.evictLruToAdmit` gives memory back rather than refusing: a
  cached prefix is an optimization, the request is the work. It runs on the
  inference thread (sole mlx caller, even for frees) after the prefix restore
  — which makes the entry in use the most-recently-used one, so LRU reaches it
  last and `protect_mru` stops before it. The fits predicate is re-asked after
  every eviction, never compared against a precomputed shortfall (#126).
  Refusal happens only with an empty cache, and quotes the hot-cache bytes it
  counted.

**Bar.** The unit tests pin the terms and the single-grow invariant; the live
bar is a 64k-step ladder 384k → 512k on a 128 GB box: every rung either
completes or answers a 503, and the process is still serving at the end.

### Three defects the #353 branch shipped with (audit, 2026-09-05)

**1. The guard billed the UNCLAMPED `max_tokens`.** All four text surfaces ran

    if (!try checkAttentionMemory(… max_tokens …)) return;
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len, effective_ctx);

in that order — harmless while the guard ignored `max_tokens`, fatal the
moment it started billing a reservation from it. An omitted `max_tokens` is
`omittedMaxTokensDefault` = `maxInt(u32)/4` = 1,073,741,823, which on
qwen4_exp at 8-bit is ~26 TB of KV: **every prompt past
`RESERVE_MIN_TOKENS` with no `max_tokens` field would have been refused 400**
— the overwhelmingly common shape for an agent client. The 512k ladder rung
missed it because the runner sets `max_tokens: 16`.

Fixed at BOTH ends, because a reservation is a number two subsystems must
agree on and neither may trust its caller for it: the four surfaces clamp
first and hand the guard `effective_max_tokens` (scan-pinned, including that
a clamp precedes each guard within 900 bytes), and `KVCache.reservedTokens`
takes the context and bounds the headroom by `ctx - seq` itself. `ctx == 0`
means "unknown" and imposes no clamp.

**2. The evict-to-admit probe billed a different request than the guard
admitted.** `prefillFitsNow` hardcoded `kv_override = null` and
`unchunked_prefill = false` while `checkAttentionMemory` passed the request's
own `kv_quant_override` and `visionPrefillUnchunked(...)`. A `kv_quant: 4`
request therefore priced its cache at fp16 on the inference thread —
over-billing ~2.4x, evicting a hot cache that did not need to go, and able to
refuse by name a request the connection thread had already admitted. One
estimator means one set of INPUTS too: the hook now carries the scheme and the
chunking, the scheduler reads them off the slot (`slot.cache.config`,
`slot.vision_embeddings`), and both are scan-pinned.

**3. A decode-time MLX failure was never consumed.** `mlx.checkError` lived
only in the prefill chunk loop. A decode forward that failed left the slot
sampling from buffers Metal never wrote — the request finished **200** with
whatever those bytes decoded to — and the latch waited for the NEXT request's
first prefill chunk, which answered 503 for a failure that was not its own:
two wrong answers from one error. `runSingleDecodeTick` and
`runBatchedDecodeTick` are now thin wrappers that run the tick and then
consume the latch (`mlx.checkErrorDecode`), attributing the failure to that
slot — or, for a batched group, to every slot in it, since they share the one
forward and there is no honest way to blame one. A latched MLX message is the
root cause and any Zig error above it is the symptom, so the latch wins;
either way it is cleared before the next tick.

**Bar.** `MLX_SERVE_MLX_FAULT_STEP=<n>` is the decode sibling of
`MLX_SERVE_MLX_FAULT_CHUNK`, with its own counter (a decode-step test must not
have its count eaten by the prefill's chunks); `tests/test_mlx_error_recovery.sh`
arms it, asserts the faulting request fails and the next succeeds, and sends
one long prompt with **no** `max_tokens` field — the class defect 1 belongs to.

### The reservation bought a generation nobody asked for (#353 follow-up, 2026-09-05)

Functional check on the mega candidate (`longctx-mega-cand-nk` @ de41ffc,
`--ctx-size 1048576 --kv-quant 8 --mtp --prefix-cache-mem 24GB
--prefill-chunk 4096 --prefix-cache-disk 100GB`), three rows:

- 50k prompt, no `max_tokens` → **200**.
- 512k prompt, `max_tokens: 16` → **200** (524,253 tokens, 862 tok/s prefill,
  peak 100.7 GB, liveness after it 200).
- **374k prompt, no `max_tokens` → 503**, with the abandoned-generation
  message. The log:

      generation budget squeezed: 665509/1073741823 tokens remaining (prompt=383067, ctx=1048576)
      prompt 383067 tokens needs ~37394MB, ~37130MB available + ~1564MB evictable hot cache — admitting, the prefill will evict
      [hot-cache] reused 51662/383067 tokens (matched 51662; entry 1/1)
      [scheduler] prefill refused: 383067 tokens do not fit even with an empty hot cache

No `[mlx]` line: nothing ran out of anything, and nothing was abandoned. Two
defects, and it took both to produce that 503.

**1. The reservation was sized by the CONTEXT, not by a generation.**
`clampMaxTokens` turns an omitted `max_tokens` into `ctx - prompt` — here
665,509 tokens — and `reservedTokens` reserved every one of them: ~8.3 GB of
8-bit KV headroom plus ~4.8 GB of QSA indexer history (billed at both copies),
**13 GB** of a 37.4 GB bill, for a generation the request would never run. The
context bound added by the previous audit is the right bound for a caller who
NAMES a huge `max_tokens`; it is the wrong bound for one who names none.

What the reservation is for is the prefill's old+new coexistence: a grow
allocates the new capacity beside the old one and both stay live in the
chunk's lazy graph. That is a PREFILL property. Past the prompt, a decode grow
takes the ordinary +25% policy — one grow per quarter of the cache (~94k
tokens at 374k), between decode steps, with nothing else live. So the
reservation now buys `seq + min(max_tokens, RESERVE_GEN_HEADROOM) + chunk`,
with the headroom at 8192 tokens (~100 MB of KV here): more than any real
answer, and 130x less than the window. The same `KVCache.reservedTokens` is
still the ONE definition the guard bills and the engine allocates.

**2. "Fits after eviction" credited memory eviction would never return.**
The bill was 264 MB over the headroom and was admitted only because the hot
cache held 1,564 MB. That was ONE entry — the one this very prompt then
restored from. A restore refcount-shares the entry's buffers with the slot's
cache, so `evictLruToAdmit` protects it by construction (`protect_mru`) and
evicting it would return nothing anyway; `lruIndexExcluding` had nothing else
to offer and the request was refused after being promised.

The connection thread cannot know which entry a prompt will match, but it
knows a restore pins at most ONE, so the provable credit is the residency
minus the largest entry: `HotPrefixCache.reclaimableBytes`, published beside
`residentBytes` and read by the guard as `AdmissionBill.reclaimable`.
`fitsAfterEviction` credits that; `evictable` stays in the message, because
the operator's question is still "why didn't it drop a cache entry?".

**3. And the refusal was dishonest.** The scheduler returned
`error.OutOfMemory`, which is the MLX latch's name, so the client was told the
engine "ran out of GPU memory during this request and it was abandoned" for a
request that never ran a forward. It is now `error.PrefillDoesNotFit` — a
named **400** on all four surfaces, with the compared byte counts logged
(`refused before prefill`) by the estimator that owns them.

**Also seen in the same boot, and expected:** `[hot-cache] budget clamped
24576 -> 5703 MB (weights + ctx KV + prefill reserve vs GPU ceiling)` and
`[hot-cache] skipped oversized entry (524269 tokens, 11349.42 MB > 5703.95 MB
budget)`. At 1M context the clamp reserves the whole context's KV, and an
entry is billed at the QSA history's two copies — 11.3 GB for 524k tokens. The
entry lands on the SSD tier instead; that is the cap working, not a defect.

---

## The chunk sizer read the cache ASK, so a smaller ask bought a smaller cache

Two boots of the same 69 GB `qwen4_exp` pack on the same 137 GB Mac, same
`--ctx-size 1048576 --kv-quant 8`, differing only in one flag:

```
--prefix-cache-mem 10GB  ->  [hot-cache] budget clamped 10240 -> 3873 MB
--prefix-cache-mem 24GB  ->  [hot-cache] budget clamped 24576 -> 5703 MB
```

Asking for **less** got **less than half** the cache. Not noise — arithmetic,
and reproducible.

### The loop

Both boots imply the same headroom: `ceiling - weights` = 28,909 MiB, and the
1M context's KV is `(13,056 + 7,680) B/tok x 1,048,576` = 20,736 MiB. So both
have 8,173 MiB to split between the prefill transient and the hot cache. What
differed was the split, and the ask decided it:

1. `pinPrefillChunk` passed the RAW requested budget into `resolvePrefillChunk`
   as `hot_cache_reserve`. The rung bar is
   `(ceiling - weights - hot_cache_reserve) / 4`.
2. At a 10 GB ask that bar admitted rung **1024**, whose transient reserve is
   ~4,300 MiB. At a 24 GB ask the bar dropped below every rung's reserve, so
   the ladder bottomed out at **512** — reserve ~2,470 MiB.
3. `prefixCacheMemForLoad` then billed *that* chunk's reserve back against the
   *same* request: `min(ask, ceiling - weights - ctx_kv - reserve)`.
   - 28,909 - 20,736 - 4,300 = **3,873**
   - 28,909 - 20,736 - 2,470 = **5,703**

A bigger ask stepped the chunk DOWN, which shrank the reserve, which left the
cache MORE bytes. The ask was on both sides of the equation with opposite
signs, so the composed function was not monotone in it — and the operator's
lever pointed the wrong way over part of its range.

A later probe swept the ask further and found the inversion is not confined to
the low end. Same pack, quiet box, `--prefill-chunk 4096` pinned, wired 120000:

| ask | resolved | `max_safe_context` |
|---|---|---|
| 10 GB | 9,765 MB | 1.03M |
| 24 GB | 13,244 MB | 565k |
| 40 GB | **15,917 MB** | 34k |
| 60 GB | **15,830 MB** | 1,024 |
| 10 GB (repeat) | 9,562 MB | 1.02M |

Two more things fall out of that table. **40 GB buys more cache than 60 GB** —
a second inversion, at the top of the range, where the extra ask pushes the
sizer across a rung boundary and the wider reserve costs more than the ask
gains. And the 10 GB repeat differs from the first 10 GB boot by 2.1% (9,765 vs
9,562) with nothing changed but the minute — that is the free-RAM term in the
ceiling, which the static-ceiling fix further down removes.

### The fix is an ORDER, not a formula

The clamp already subtracts the chunk's reserve from the cache's headroom.
That makes the cache the **residual claimant**: it gets what the chunk leaves.
Pre-charging the cache in the sizer and then charging the chunk to the cache in
the clamp is the same bytes billed twice, in a circle. So the dependency runs
one way now:

- `resolvePrefillChunk` no longer takes the ask at all — the parameter is gone,
  which is what makes the property structural rather than a comment. Its share
  bar prices the whole post-weights serving budget.
- `clampedPrefixCacheMem` then hands the cache the remainder.

Monotonicity is now by construction: nothing before the clamp reads
`requested`, and the clamp is `@min(requested, headroom)` over a headroom the
ask cannot move. A fixpoint iteration (chunk -> clamp -> chunk until stable)
would also have worked and was rejected as more machinery for the same answer.

### Second bite: `requested == 0` used to feed the sizer a zero reserve

The old code was accidentally *right* about one thing — with the ask in the
bar, a big ask forced a narrow chunk, which left the cache something. Remove
the ask and a wide rung can eat the whole remainder, and
`clampedPrefixCacheMem`'s `@max(..., 1)` "never return 0" floor then hands the
cache **one byte**: an enabled cache that cannot hold a single turn. The clamp
cannot defend a budget the chunk has already spent, so the defence moved one
step earlier, into the sizer, as a second ask-independent bar:

```
cap = min( serving / PREFILL_RESERVE_BUDGET_SHARE,
           (serving - ctx_kv) - HOT_CACHE_FLOOR_BYTES )
```

`HOT_CACHE_FLOOR_BYTES` is 1 GiB. That is a PICK, not a measurement: one
agentic prefix on a large model is order-GB of KV, so below it the cache cannot
retain a single turn and the feature is dead weight. It is the only tunable
this fix introduced.

`ctx_kv` here is `sizerCtxKvBytes` — the PINNED context's bill under an
explicit `--ctx-size`, and **0** while the context is auto. Auto-context is
derived FROM the chunk (`pinAutoContext` pins the chunk first, deliberately),
so reading it here would be circular; and it does not need to be read, because
`computeMemoryContext` subtracts the chunk's reserve itself and shrinks to fit
whatever rung comes out. Both terms are ask-independent, so neither reopens the
loop above.

### Third bite: `--prefill-chunk` was honored by the forward and ignored by the bill

`generate.effectivePrefillChunk` lets an explicit `--prefill-chunk` outrank the
machine-sized pin. `pinPrefillChunk` did not — only its *log line* consulted
`prefill_chunk_explicit`. So on the deployed flags (`--prefill-chunk 4096`) the
clamp reserved a 512- or 1024-token transient for a forward that ran 4096: an
under-reserve of several GB, in the one estimator whose entire job is to stop
the cache from filling into an uncatchable Metal OOM. `explicitPrefillChunk()`
mirrors the precedence and `billedPrefillChunk` is where both readers meet; the
test calls *both* with an explicit chunk set and demands the same number.

### Fourth bite: the clamped value went nowhere

`prefixCacheMemForLoad`'s answer reached `initWithMem` and stopped there. The
process-global `prefix_cache_mem_bytes` still held the raw ask, and
`computeMemoryContext` and `aneGateHeadroom` both read it — so the auto-context
sizer and the ANE admission gate went on reserving 24 GB for a cache that had
been given 5.7. The clamp now publishes into `hot_cache_mem_resolved` and every
post-load reserve reads `resolvedPrefixCacheMem()`. It is a SEPARATE global
rather than an overwrite of the ask: the ask is the launch flag, and
re-clamping an already-clamped value on the next model load would ratchet the
budget toward 1 byte across a model swap. A line scan over `server.zig` pins
that no code path reads the ask directly except the accessor's own fallback.

**Known limit — the ANE gate reads the budget before the load publishes it.**
`aneGateHeadroom` is the one remaining consumer (after the sizer fix above,
`computeMemoryContext` no longer reads the published budget at all), and it is
NOT ordered behind the resolver: `doLoadOnInferenceThread` calls
`buildAnePrefill(..., params.ane_headroom_resolver)` at `scheduler.zig` ~3929,
while `prefixCacheMemForLoad` publishes ~74 lines later at ~4004. So on a first
boot the gate reads the pre-load fallback — the raw ask, which is the LARGER
reserve, so the direction is conservative — and after a model switch that leaves
the previous entry resident, or a load of a model whose prefix cache is off
(`scheduler.zig` ~4110 is `if (entry.prefix_cache) |*hc|` with no else, so
neither the resolver nor the retire hook runs), model B's gate can reserve model
A's budget. What DID land is the staleness half: `hot_cache_mem_resolved` is
atomic, and `scheduler.hot_cache_budget_invalidate` retires the published number
at every site that drops the resident cache (five sites; the test scans for the
PAIRING, so a sixth drop site fails the scan). The ordering half is open. It is
reachable only through `--ane-prefill`, which is opt-in and M4-and-below, and
this box cannot exercise that path — so the reorder is recorded as a follow-up
rather than done blind. Do not read this section as "S7 fixed".

**The damage was worse than a mis-sized reserve: it floored the ADVERTISED
context.** Right-hand column of the probe table above — `max_safe_context` goes
1.03M -> 565k -> 34k -> **1,024** as the ask grows 10 -> 24 -> 40 -> 60 GB.
`computeMemoryContext` subtracted the whole RAW ask from the serving budget, so
a 60 GB ask reserved 60 GB of cache the clamp had already refused to grant, and
the context sizer had nothing left to report.

Which surface that reaches depends on one flag, and the distinction is worth
keeping straight:

- **`/props` `max_safe_context`** comes from `computeMaxSafeContext`, called on
  every boot with no `--ctx-size` early-out. It misreports either way — but it
  is a diagnostic, so the cost is a misleading number.
- **The ADVERTISED context** — `/v1/models` `context_length`, which agent CLIs
  read ONCE and budget their own `max_tokens` against for the rest of the
  session — comes from `getEffectiveContextLength`, and both it and
  `pinAutoContext` return `server_config.max_context_size` immediately when
  `--ctx-size` is set. So **a manual `--ctx-size` boot is protected**: the
  advertised value is the operator's own number and never routes through
  `computeMemoryContext`.

The bite is therefore on **AUTO-context boots** — no `--ctx-size`, which is the
default. There `pinAutoContext` -> `autoContextFor` -> `computeMemoryContext`,
and a generous `--prefix-cache-mem` silently pins a 1,024-token advertised
context for the life of the process. The operator's lever for "hold more cache"
was quietly also the lever for "advertise almost no context", on exactly the
configuration where nothing pins it back.

The log line names what it billed, so the next report is one grep:

```
[hot-cache] budget clamped 24576 -> 5703 MB (chunk 512 reserve 2470 MB, ctx KV 20736 MB)
```

### Not fixed here

The `ctx_kv` term itself double-counts: the clamp bills the FULL configured
context against a budget the hot cache shares with the live KV, so at 1M
context 20.7 GB is subtracted from the cache's headroom for KV that only exists
once a request is that long. That is being removed arch-gated on the SSD-first
branch. This fix makes the default sane with that bill still in place.

**And there was a second inversion, in the CEILING.** Same 10 GB ask, same
pack, same session; only `iogpu.wired_limit_mb` differs — from
`judge_mega3_fix.budget.txt`, boot 1 at 2026-09-05 13:42:36:

```
sysctl iogpu.wired_limit_mb = 120000
[preflight] weights ~70.13 GB, available 108.72 GB
[hot-cache] budget clamped 10240 -> 1076 MB
```

against the default ceiling's `10240 -> 3873 MB`. **Raising the ceiling ~10 GB
shrank the cache 3.6x.** Mechanism: the extra headroom bought the sizer a wider
rung, the rung's reserve grew faster than the headroom did, and the cache — the
residual claimant — paid for the upgrade. Making the ask monotone did not touch
this: the rung ladder is discrete, so any ceiling increase that crosses a rung
boundary can cost more than it gains.

The fix is to notice what the load-time reserve is FOR. **It is a promise to
the FIRST request.** On an arch that picks its width per request (below), every
request re-prices its own real width at admission against live memory, and
`fitsAfterEviction` can hand hot-cache bytes back to admit it — so the promise
only has to cover the narrowest forward the box can ever be asked for. That is
the ladder floor, and it does not move with the ceiling:

```
budget = ceiling - weights - ctx_kv - reserve(floor)
```

monotone in the ceiling AND in the ask. `clampReserveWidth` is the one place
that decides it; `HotCachePlan` carries both widths (`chunk` for the boot log
and the default pin, `reserve_chunk` for what the clamp billed) and the log
line names both, because they answer different questions:

```
[hot-cache] budget clamped 10240 -> 7000 MB (chunk 1024, reserve at width 512 = 2470 MB, ctx KV 20736 MB)
```

Two carve-outs. An explicit `--prefill-chunk` is billed as-is — the operator
pinned a width, every request runs it, nothing narrower is ever chosen. And an
arch without the per-request gate keeps the sizer's rung, because it has no
per-request re-bill to fall back on; for those the load-time promise really is
the promise.

**The residual risk, stated:** the floor promise leans on evict-to-admit, and
`reclaimableBytes` is residency minus the largest entry — so a cache holding
one giant entry cannot give it back. That entry is the one a restore would
share, which is why it is excluded; the exposure is a wide forward arriving
against a cache that is one huge unreclaimable entry. The admission bill still
refuses by name rather than OOMing in that case.

**And the whole thing was non-reproducible anyway.** Two boots of the identical
binary, flags and wired limit, 11 minutes apart on 2026-09-05:

```
13:42 (straight after two suites)  [preflight] available 108.7 GB -> 10240 -> 1076 MB
13:53 (quiet box)                  [preflight] available 116.6 GB -> 10240 -> 9757 MB
```

A 9x spread on the same machine and the same ask, because
`currentGpuMemoryCeiling` is `min(recommendedMaxWorkingSet, footprint + free
RAM)` and the load-time clamp read it. The free-RAM half is a property of the
minute the server happened to start in; the budget it produced was therefore
whatever the box was doing at the time, and the user's bar is one boot that
works.

So the LOAD-TIME clamp bills the STATIC term only — `staticGpuMemoryCeiling()`,
Metal's `max_recommended_working_set_size` or the wired limit:

```
budget = staticGpuMemoryCeiling() - weights - ctx_kv - reserve(floor|explicit)
```

Every term is a property of the machine and the model, so two boots agree.

**The free-RAM term is moved, not deleted.** It stays exactly where it can be
acted upon: request-time admission (`prefillAdmissionBill`, and `prefillFitsNow`
re-asking it after every eviction) and the `active_mem` guards. A cache sized
above what free RAM currently allows is not a crash — it is entries the
admission pass evicts on the first request that needs the room, which is the
#353 machinery doing its job. What it must never be is a budget that silently
shrinks 9x because a test suite was running.

Deliberately NOT changed: `computeMemoryContext` (the auto-context sizer) and
`pinPrefillChunk` (the memory-sized rung) both still read the live ceiling. The
advertised context is pinned once and read by agent CLIs that never re-read it,
and the rung feeds the boot log and non-gated archs; neither is the hot-cache
budget, and neither was the thing that was irreproducible in a way anyone felt.
The registry's eviction gate (`--max-resident-mem`, `gateEstimateBytes`) never
touched this helper at all.

Under external pressure the load line says so without acting on it:

```
[hot-cache] budget 9757 MB (static ceiling); free at load 38.6 GB — live admission will evict as needed
```

**CORRECTED BY MEASUREMENT.** I first wrote here that the wide rung would
arrive when the SSD-first branch removed this bill. It does not need to, and
the arithmetic I used to say so was wrong in two ways: I was comparing a full
admission bill against a transient reserve, and I was assuming the load-time
`PREFILL_RESERVE_BUDGET_SHARE` still gated the outcome. The per-request chooser
below does not go through the load-time sizer at all — it prices the REQUEST
against live free memory — so on this box a 300k or 384k prompt already gets
chunk 4096 today, with this bill still in the clamp. What the SSD-first change
buys is a bigger hot cache, not a wider chunk.

`PREFILL_RESERVE_BUDGET_SHARE` and the reserve estimator both stay untouched:
the judge at chunk 4096 peaked 90.3 GB of a ~93 GiB ceiling at 384k, so the
estimator is not loose, and the share is what stops one forward from trading
the whole session for its own speed at LOAD time.

**The ordering wrinkle was NOT moot — it was the whole of live check #6.**
An earlier draft of this section said the clamp reading a stale context was
"bounded, and moot once the SSD-first branch removes the full-context bill."
Both halves were wrong, and the live check found it:

    # auto-context boot (no --ctx-size), --prefix-cache-mem 60GB
    [hot-cache] budget clamped 61440 -> 48673 MB (chunk 4096, reserve at width 512 = 1474 MB, ctx KV 26 MB)
    Context size: 870 tokens (auto: 85% of the 1024-token memory ceiling)
    # the same boot at the default ask
    Context size: 1048576 tokens (auto: the model's maximum; memory would allow 1113088)

`ctx KV 26 MB` is the tell. The clamp runs during the load and `pinAutoContext`
runs after it, so `getEffectiveContextLength` answers with the 1024-token floor;
the clamp subtracts ~26 MB of "context" instead of ~20.7 GB, grants almost the
whole ask, and the sizer that runs afterwards has nothing left. An
over-generous `--prefix-cache-mem` therefore pinned an **870-token advertised
context** for the life of the process — and agent CLIs read that number once
per session. Publishing the RESOLVED budget rather than the raw ask made it
48,673 instead of 61,440, which is smaller and still fatal: **an ORDER bug is
not fixed by better arithmetic on the wrong input.**

It is also not SSD-first's to fix. Both arms of the load-time bill had it, and
both now read ONE pure resolver, `resolvedContextForLoad` — explicit
`--ctx-size` wins, else a pinned context, else a sized one that mirrors
`autoContextFor`'s margin-then-cap WITHOUT calling the accessor it works
around. The arms differ in exactly one argument, the cache reserve:

- **SSD-first passes 0.** Its resident entry IS the live KV, so the honest
  load-time reserve genuinely is zero and no fixpoint exists.
- **RAM-first passes `CTX_SIZING_CACHE_RESERVE`**, a fixed 2 GiB. It holds cache
  in ADDITION to the live KV, so it has no such out — which is why the
  parameter exists at all.

That constant must never be derived from the ask, and passing the ask itself is
the same bug one level in: `ceiling - active` is 39,568 MiB against a 61,440 MiB
ask, so sizing the context against it saturates to zero usable and returns the
1024-token floor — the 870 boot again. **Context is the primary claimant and the
cache is the residual**, the same order the prefill chunk follows. An ask must
never be able to shrink the context that is then used to bill that ask.

The manual path is untouched by all of it: `--ctx-size` wins in the resolver's
first branch, before any memory arithmetic runs, so a pinned boot bills exactly
what it billed before.

### The width is re-asked after the eviction pass (N5)

`prefillAdmissionBill` computes `available` WITHOUT `reclaimable` and hands
that to `chooseRequestPrefillChunk`, so a resident hot cache narrows the
prefill width — the ladder floor instead of 4096 — rather than being evicted.
That is the right call at that moment: crediting `reclaimable` there would
choose a width justified by memory eviction has not actually returned yet, the
same "credit what you cannot prove" shape as the `evictable` vs `reclaimable`
split this work introduced after a 383k prompt was admitted on 1,564 MB of
cache that turned out to be the single entry the prompt itself had just
restored. The two failure costs are not symmetric: a narrow width costs prefill
throughput, a width justified by absent memory costs an uncatchable Metal OOM.

The fix is therefore not a wider bill but **a second look**. The scheduler's
prefill path now asks the width TWICE around the eviction pass:

1. `admitted_prefill_chunk` — the pre-eviction reading, taken through the same
   `prefill_request_chunk` hook, with the same inputs and the same live memory
   as the probe that just failed. It is captured ONLY on the path that actually
   evicts: when the probe already fits, no memory moves and there is nothing to
   compare a second reading against.
2. `evictLruToAdmit` runs and returns an `EvictionReport`.
3. `evicted_live_bytes = report.bytes` — the ALLOCATOR's delta, never
   `accounted_bytes` (a shared snapshot bills megabytes and returns none of
   them) and never `reclaimable` (a projection).
4. The re-ask, then `scheduler.postEvictionPrefillChunk(admitted, reasked)`.

**The re-ask is the truth, and the comparison is only a log.** This shipped as
`@max(admitted, reasked)`, argued as an assertion: an eviction only ever
RETURNS memory, so post-eviction availability is `>=` pre-eviction, the max
changes nothing on the expected path, and it "catches" the impossible case. The
external review of PR #363 turned that argument around. A max is not an
assertion — it is a clamp, and the ONLY case in which it does anything is the
case where the invariant broke. There it discards the reading taken after the
pass against LIVE memory and keeps the pre-eviction one because it is WIDER,
which is widening onto memory that has already gone to a co-tenant slot's
decode: the uncatchable Metal OOM, arrived at by way of a safety check. The
costs are not symmetric and the safe arm is the narrow one, so the second
reading — taken by the same estimator that admits the request, after the pass,
against what is actually free — is the width that runs, whichever direction it
moved. The floor still holds because the CHOOSER holds it: its ladder falls
through to the floor rung and `postEvictionPrefillChunk` computes no width of
its own. Both directions log under `[prefill] re-ask:` (a distinct prefix from
the per-chunk `[prefill] width N -> M at pos P` contract line), the widening at
info naming the bytes the pass returned, the narrowing at warn naming both
widths — a narrow re-ask is rare and worth seeing, it is just not worth
overruling.

The re-asked width has to reach two consumers with different rules, and a
change that wires one and not the other ships a width nobody runs:

* The request's own chunk: `req_prefill_chunk` →
  `InitOptions.pinned_prefill_chunk` → `effectivePrefillChunk` →
  `PREFILL_CHUNK` → `default_chunk` → `cur_chunk`.
* The per-chunk adapter's ceiling, `cap_adapt`. It is built from the arch
  resolver with the pin left OUT (a literal `0` rung), so it is `>=` any ladder
  width the chooser can return and cannot clip a re-ask. If it ever starts
  reading the pin, the widened width becomes its own ceiling and the adapter
  can only narrow.

Both are scan-pinned (`the re-asked width reaches the request's chunk AND is
never capped by the adapter`), as is the ORDER of the four steps above.

### The admission bill is logged on ADMITTED requests too

The tight-admission path used to log nothing when it went through: the numbers
only ever appeared once a request had already been refused, so there was no way
to watch the machine walk up to the edge. `checkAttentionMemory` now emits ONE
line per REQUEST — never per chunk — at the admission decision:

```
[admission] needed=N MB available=A MB reclaimable=R MB width=W verdict=admit|evict|refuse
```

Every field comes off the SAME `AdmissionBill` the three arms act on and the
refusal message quotes, so the two cannot name different numbers for the same
request; a second estimator call here would be #126 ("a gate that runs BEFORE
the estimator that knows better IS the estimator") in log form, quoting a bill
nobody acted on. `admissionVerdict` derives the verdict from that bill with the
same two predicates the arms branch on (`fits`, `fitsAfterEviction`).

The level is chosen by `admissionLogLevel`: info on the first request after a
model becomes resident (`model_registry.load_generation`, consumed by one
compare-and-swap so N racing connection threads produce ONE line), info
whenever `needed > 0.9 * available` — the band where the next request is the
one that evicts or refuses — and debug otherwise, because a roomy machine
would otherwise write one info line per request for the life of the process.
The level check runs FIRST and the byte divisions and verdict string happen
only after it; under `--log-level warn` the call returns before it can even
consume the post-load token, which must not be spent on a line that was never
going to be written.

### The billed session and the advertised session were two numbers (2026-09-05)

Found by the SSD-first owner while folding these branches together, and it is
the same family as audit S6 one level out.

On an SSD-first boot the load-time bill and the pinned context are supposed to
be one session:

- `ssdFirstSessionTokensNow` bills the session the budget FLOOR is built for
  (`ssdFirstPrefixCacheMem` floors the budget at one entry at the working
  context), and it passed `cache_reserve = 0` — the reasoning being that the
  resident entry IS the live KV, so the cache reserves nothing beyond the
  session.
- `pinAutoContext` -> `computeMemoryContext` then sized the ADVERTISED context
  against `CTX_SIZING_CACHE_RESERVE`.

Same box, same instant, two reserves, two sessions. An agent CLI reads the
smaller number out of `/v1/models` ONCE and budgets against it for the whole
session, while the cache floor holds RAM for the larger one. It survived review
because neither number looks wrong on its own: both are real contexts, a few
percent apart, and the wrongness is only visible when you ask which one "one
session" means.

The `= 0` argument is true of the RESIDENT entry and false of the mode's IDLE
allowance, which is RAM in addition to it. But the honest idle allowance is
`--prefix-cache-mem`, i.e. the ASK — and reading the ask is live check #6 (a
60GB ask collapsed the advertised context to 870 tokens), while reading the
budget resolved from it is audit S6, the same bug as a one-step loop. Context is
the primary claimant and the cache is the residual. So the reserve is the
CONSTANT on both arms and on both sides, and the constant is the server's own
`--prefix-cache-mem` default, so a default boot is unchanged.

The relation, stated exactly, is now one expression in one helper
(`autoContextFrom`) that both sides run:

```
advertised = autoContextFrom(
    safeContextForBudget(ceiling, active,
                         CTX_SIZING_CACHE_RESERVE + transient,
                         per_tok, 0),
    ctx_cap)
```

The 85% memory margin lives INSIDE it and the checkpoint cap is applied AFTER
the margin, un-margined — get that order wrong and the two sides differ by
exactly the margin, which is how the S6 test read at the fold (it compared the
raw memory context against the clamp's margined answer).

Explicit `--ctx-size` boots are untouched: the operator's number is the
resolver's FIRST early return and `pinAutoContext` returns it before any sizing
runs. Pinned as INVARIANCE in the reserve rather than against a recorded
constant, because that is the property the change has to preserve.

### Rules this produced

- The load-time session bill and the advertised context are ONE number. They are
  computed in different functions at different times, so they must read one
  reserve and one margin helper, or they are two answers to "how big is one
  session" and the cache floor is sized for a session nobody can request.
- A per-arm reserve is a per-arm SESSION. If an arm has a reason to reserve
  differently, the sizer needs the same reason — otherwise the arm is billing a
  machine the server does not advertise.
- A margin and a cap that are applied in a fixed ORDER belong in one helper. Two
  sites spelling "85% then cap" agree until one of them is edited.
- A width chosen from free memory is asked TWICE around an eviction pass, and the
  second reading is credited only the bytes the ALLOCATOR returned — never
  `reclaimable`, never `accounted_bytes`. The clamp to the max is the assertion
  that an eviction cannot take memory away.
- An admission path logs its bill on the requests it ADMITS, not only on the
  ones it refuses; both arms format the same fields from the same bill.
- A resolver whose output is billed back against its own input is a loop; check
  monotonicity in the input before believing the output.
- A knob that is read on both sides of a bill with opposite signs is not a
  knob. Give the residual claimant the remainder instead of pre-charging it.
- A precedence rule (`--prefill-chunk` outranks the pin) is a property of the
  PAIR of readers. Pin it with a test that calls both.
- A clamp's "never return 0" floor is not a guarantee that anything useful
  survives; the floor has to be defended by whoever spends first.

---

## The prefill width is a property of the REQUEST, not of the boot

`--ctx-size 1048576`, no other flags. The load-time sizer does its job
correctly and the result is still wrong for almost every request the server
then serves.

### Why load-time sizing cannot win here

`resolvePrefillChunk` runs once, with the weights resident and nothing else
known. The only context it can reserve for is the one the operator configured,
and at 1M that context's own KV is 20,736 MiB of a 28,909 MiB serving budget.
Whatever is left has to hold both the prefill transient and the hot cache, so
the widest affordable rung is 1024 — and every prompt for the rest of the boot
prefills at 1024, including the 4k ones.

But a 4k prompt does not hold a 1M-token cache. Neither does a 384k one: the
judge ran 384k prompts at chunk 4096 well inside the ceiling (peak 90.3 GB of
~93 GiB). The width the machine can afford is a function of THIS request's KV,
and the estimator that already knows THIS request's KV is the admission bill.

So the decision moved to where the information is:

```
scheduler prefill: admit -> evict -> CHOOSE WIDTH -> InitOptions.pinned_prefill_chunk
```

`chooseRequestPrefillChunk` walks `PREFILL_CHUNK_LADDER` and takes the widest
rung whose bill fits the memory that is free right now. Post-eviction is
load-bearing: it is the first moment the free memory is the memory the prefill
will actually run in.

### It is priced by the estimator that admits it, or it is a second bill

A chooser with its own arithmetic is #126 ("a gate that runs BEFORE the
estimator that knows better IS the estimator") with the biggest term — the
chunk — as the difference. So `prefillNeededAtChunk` was factored out of
`prefillAdmissionBill`, and both go through it; a source scan pins that there
is exactly one definition and that both callers reach it. Whatever the chooser
picks, `needed(picked) <= available` by construction, so the forward is billed
and fits.

### Two details that are easy to get wrong

**Price the WIDTH, not the rung, and return the width.**
`generate.effectivePrefillChunk` still caps by arch — a hd-256 MoE never
forwards wider than 4096 — so rungs 8192 and 4096 are the same forward.
Pricing the raw rung would over-bill the top of the ladder and then pick it for
the wrong reason, and pinning 8192 would describe a width that never runs.
Candidates are priced at `effectivePrefillChunk(rung)` and the chooser returns
that width, which makes the pin a fixpoint of the resolver.

**The connection thread admits on the load-time pin, and that is fine.** Its
bill uses the narrower width, so it admits requests the wide path might not
fit — but the scheduler's choice is bounded by live memory, so a request that
only fits at 512 gets 512. The dangerous direction (admit narrow, run wide) is
closed by the chooser never returning a width whose bill exceeds `available`.

### The other direction: the guard refused at a width nobody was going to run

Choosing the width at the scheduler left the ADMISSION probe still pricing the
load-time pin, and that is a refusal, not a slowdown. A prompt that fits at 512
and nothing wider was refused by name — `error.PrefillDoesNotFit`, a 400 — for
a width the forward would never have used.

**Measured, and not what I assumed.** I wrote this expecting the 1M-token
prompt to be the case it serves. It is not: on this box a 1M prompt bills
30,261 MB even at the ladder floor against 28,909 MB free, so it does not fit
at ANY width and is refused whatever the ladder does — the advertised context
is larger than the machine can prefill, which is a ctx-KV problem and not a
width one. The case this actually serves is the long-but-not-maximal prompt:
at 384k the floor bills 12,258 MB and the load-time pin's width bills 12,632,
so a budget between them is admitted at 512 instead of refused at 1024.

So `prefillAdmissionBill` picks its width the same way, through the same
`chooseRequestPrefillChunk`, which walks the ladder from the widest affordable
rung DOWN. Since `prefillFitsNow` (the inference-thread probe, re-asked after
every eviction) and `checkAttentionMemory` (the connection thread) are both
that bill, all three now ask one question. The named 400 fires only when
NOTHING on the ladder fits, and the message says which width it gave up at:

```
prompt 1048576 tokens needs ~31402MB at prefill chunk 512 (the narrowest width
tried), ~28909MB available after evicting ~0MB of hot cache — refused before
prefill
```

`AdmissionBill.chunk` carries that width so the message and the number it
quotes cannot describe different forwards.

An explicit `--prefill-chunk` keeps TODAY's behaviour on purpose: the chooser
returns it unchanged and the refusal fires at that width. The operator picked
it; silently downgrading the width they asked for would be a worse answer than
saying no.

### The measured ladder

From the estimator itself, bills in MB for the deployed shape (8-bit KV,
`max_tokens` 2048), against the live box's 28,909 MiB of free memory:

| seq | w4096 | w2048 | w1024 | w512 | chooser picks |
|---|---|---|---|---|---|
| 4,096 | 3,981 | 2,441 | 1,521 | 1,136 | **4096** |
| 60,000 | 6,227 | 4,732 | 3,770 | 3,045 | **4096** |
| 300,000 | 12,743 | 11,249 | 10,351 | 9,978 | **4096** |
| 384,000 | 15,024 | 13,529 | 12,632 | 12,258 | **4096** |
| 1,048,576 | 33,027 | 31,532 | 30,635 | 30,261 | floor, refused |

Load-time sizing hands all five of those rows chunk 1024. Note the ladder is
seq-INDEPENDENT here (rung 8192 and 4096 both forward at 4096, the rest map to
themselves): that is the hd-256 non-sliding branch of `boundedPrefillChunk`. On
a sliding-band arch the composed-causal score budget collapses every rung to
the floor past ~300k, which is a different policy family and one reason the
gate is an opt-in of one.

### Scope

Gated to `qwen4_exp` (`ModelConfig.perRequestPrefillChunk`) and killable with
`MLX_SERVE_PREFILL_CHUNK_PER_REQUEST=0`. It is the arch with a 1M advertised
context and the QSA terms that make the load-time bill lopsided; every other
arch keeps the load-time pin, unmeasured. An explicit `--prefill-chunk` still
wins outright and is billed as-is. The `<- N+M tokens` accounting is untouched
— this changes a width, not a token count.

One line per request at debug, because a narrowed prefill reads as an
unexplained slowdown and a widened one as an unexplained peak:

```
[prefill] chunk 4096 for this request (reserve 15667 MB beside KV 6075 MB)
```

### Rules this produced

- A resource decision made at load can only reserve for the configured worst
  case. If the per-request estimator knows better, that is where the decision
  belongs.
- Two estimators for the same quantity is one estimator and one bug. Factor the
  shared function and scan-pin that both callers reach it.
- Price a candidate at the value that will actually be used, not at the knob
  you turned — a capped resolver makes those different numbers.
## SSD-first prefix cache (qwen4_exp) — the disk tier only ever got what RAM kept (2026-09-05)

*Symbols moved here from CLAUDE.md (2026-09-06, to keep `## Rules` under its byte cap):*
`capturePendingDisk`, `encodeSafetensors`, `writeThroughSpanReached`, `reserve()`,
`kv_cap_buf_grows`, and the disk-budget formula `min(cap, free − min(64 GiB, 10% vol))`,
*which CLAUDE.md now calls "the free-space reserve". Each is described below.*

At 1M context on the M5 Max the arithmetic does not close for a RAM-first
cache. The weights are ~70 GB resident, one session's entry is ~24 GB
(~24 KB/token: 12.3 KB of 8-bit KV, 3.8 KB of QSA indexer history, the SSM
checkpoints, the pooled block banks) and the prefill transients are 3–7 GB,
against a ~107.5 GB ceiling. There is room for the model plus *one* session
and nothing else — so the SSD has to be the capacity tier, not a nice-to-have.

Five things stood between the code and that, and every one of them was a place
where the disk tier was treated as a junior copy of RAM rather than as the tier
that holds everything RAM cannot.

**1. The flush read the RAM entry.** `flushPendingDisk` took the snapshot the
commit had just stored, and the commit trims an oversized candidate to the byte
budget (#330). So the disk copy was capped by the *RAM* budget: past the budget
a long prefix trimmed in RAM and persisted trimmed, and the tier that was
supposed to be the deep one never held more than the shallow one. The fix is a
`PendingDiskFlush` captured in `commitWithMediaState` *before* the trim, holding
the live snapshot, the FULL token record, this turn's checkpoints and the spec
snaps. Everything in it is refcount-SHARED with the live KV, so the record costs
bookkeeping and not GPU bytes; it is consumed once by the flush and dropped on
invalidation.

**2. `max_flush_bytes` bounded a stall by truncating the entry.** The 512 MB
per-commit cap existed because the write ran synchronously on the inference
thread — the sole mlx caller — and a 4 GB write measurably stalled the next
request. At 8-bit KV that cap is 41,984 tokens per finished request, so a 374k
session needed ~9 turns to land whole and a restart before that re-prefilled
~340k (~8 minutes). A byte cap is the wrong instrument: it bounds the stall by
making the *data* wrong.

The split that works: the inference thread keeps the device→host readback (mlx
arrays are inference-thread-owned, frees included) and hands ONE writer thread a
plain host byte buffer per file. Only bytes cross. `serializeSafetensors`
reproduces mlx's own image byte for byte — 8-byte little-endian header length,
a JSON header carrying `__metadata__` and each tensor's dtype/shape/data_offsets,
then the tensors' bytes in header order — so `mlx_load_safetensors` reads back
what we wrote without knowing a Zig writer produced it. Two bounds keep the
writer honest: a ~1 GiB host-byte permit, so `submit` blocks rather than trading
GPU memory for an unbounded host queue, and an epoch fence at the tier's ONE
directory-removal site.

The fence is prefix-scoped, and that is load-bearing rather than tidy: a global
fence would have thrown away the bytes an `appendCommit` had just staged for the
entry it was writing, because the same call can evict an LRU entry on its way
out. Only the doomed directory's blobs go.

Durability comes from the queue being FIFO. An entry's `meta.json` is submitted
after its chunks, so it is the last file to land; a kill -9 mid-flush leaves
chunks with no index, which `scan` already reads as a miss. Every file is
written `.tmp` then renamed, so a crash can leave a `.tmp` but never a truncated
file under its real name.

**3. Nothing persisted until the request finished.** A 1M prefill that was
cancelled — or killed — threw away every chunk it had forwarded, and the
end-of-request flush had the whole entry in front of it. But a chunk's bytes are
final the moment it is evaluated. `Generator.WriteThroughHook` fires at each
completed prefill chunk with the absolute KV position and the checkpoints so
far, and the scheduler hands the tier `full_prompt[0..abs_pos]` — a genuine
prefix of this turn's prompt, which is exactly the shape `appendCommit` already
recognises as an EXTEND. Chunks `[0, kv_len/chunk)` are kept as written, so a
1M session writes each chunk once and not once per turn, and the end-of-request
flush is only the tail.

**4. A checkpoint was budgeted behind the chunks it makes usable.** The SSM
checkpoints shared the per-flush byte budget with the KV and were written after
it, so the first flush of a long hybrid entry landed KV with zero recurrent
state — and KV without recurrent state is not restorable on a GDN trunk. The
entry was dead weight until a later turn topped it up. Checkpoints now ride
outside the byte budget, beside the chunk that closes their position, and an
entry restores from its first flush.

**5. The budget counted the active session's KV twice.** `clampedPrefixCacheMem`
treats the hot cache as what is left over after the live session's
full-context KV reserve. But a restore refcount-shares the entry's buffers with
the slot's cache: the resident entry for the session being served costs nothing
beyond the reserve already billed.

The subtraction is not a rounding error. `prefixCacheMemForLoad` bills the FULL
configured context — at 1M and 8-bit KV that is 13,056 B/token of KV plus
7,680 B/token of recurrent and indexer state, 20,736 MiB — against the very
budget the hot cache SHARES with the live KV. Measured on the 137 GB box: a
10 GB ask resolved to 3,873 MB and a 24 GB ask to 5,703 MB. Neither holds one
entry, so the entry could not stay resident and every warm turn cold-prefilled
while the cap "held" — the #330 cliff again, one level down, and this time the
cap was right about its own arithmetic and wrong about what it was counting.

`ssdFirstPrefixCacheMem` floors the budget at one entry at the working context
and gives `--prefix-cache-mem` to IDLE entries on top of it; 0 means none idle,
which is the mode's whole point. It is a SEPARATE function selected by the arch
gate through a one-line call site rather than a branch woven into
`prefixCacheMemForLoad`, because that function's other inputs — the prefill
chunk it pins, the transient reserve it derives — have their own defects under
repair, and both arms must inherit those fixes without this one being
re-litigated in the merge.

The disk side gets the reciprocal treatment: the budget is
`min(operator cap, free - min(64 GiB, 10% of the volume))` with a 1 GiB store
floor, re-read from the volume before every store, because free space moves
under us and a cache must never fill a user's disk. Below the floor no NEW
entry persists and what is already there stays restorable — evicting a
restorable 1M entry to free a gigabyte is a bad trade. `volumeSpace` hand-
declares darwin's `struct statfs` (std has no binding in this Zig) with a
generous tail and a plausibility check — `f_bsize` a power of two in
[512 B, 1 MiB], available never past total — so a wrong layout fails SAFE back
to the operator cap instead of inventing a budget.

**A deliberate semantics change, qwen4_exp only.** `--prefix-cache-mem` means
something different on this arch: it is the allowance for IDLE entries on top
of the one-session floor, and `0` means "no idle entries" rather than today's
"no explicit cap, use all the headroom". At long context that is strictly
better — the headroom at 1M is less than one entry, so the old reading cached
nothing usable — but at SHORT context it is a real reduction: a 32k-context
qwen4_exp server that passes no flag now keeps one session resident where it
used to keep several. That is the design ("model + transients + one current KV
copy in RAM, the rest on SSD"), and evict-on-idle makes the two readings
converge at rest anyway, since RAM holds one session between requests either
way. The resolved budget is logged once at load, naming the flag, so nobody
has to infer which reading applied.

**A knock-on that turned out to belong somewhere else: the prefill chunk.** The
generic sizer spends at most a QUARTER of the post-weights serving budget on the
one-off prefill transient (`PREFILL_RESERVE_BUDGET_SHARE`), because it cannot
know what the hot cache will actually need and a quarter is a safe guess for a
16 GB Mac. On this 137 GB box that quarter is ~7.2 GiB against rung 4096's
~15.3 GiB reserve, so a no-flag boot prefilled at 1024 — measured 25–34% slower
at ≤256k — while the real headroom was several times the reserve.

Under SSD-first the cache's need is not a guess: it is one session at the
working context, exactly the quantity mechanism 5 already computes. So this work
briefly carried its own arch-gated chunk chooser, sized against
`serving − one_session − slack` instead of the quarter-share.

It was removed before shipping, and the reason is the more useful lesson: the
width is a per-REQUEST property, not a boot property. A LOAD-time arm must bill
a session at the CONFIGURED context, and almost no request is that long, so it
is structurally pessimistic no matter how honest its arithmetic — at an
explicit `--ctx-size 1048576` it lands on 1024 while a real 300k prompt bills
only 12.7 GB at width 4096 and a 384k prompt 15.0 GB, against 28.9 GB free.
The per-request chooser prices the prompt in front of it and gets 4096, and it
is gated on `ModelConfig.perRequestPrefillChunk()` — the SAME arch as
`ssdFirstCapable()`. Two arch-gated choosers for one value on one arch is a
rule that can only disagree with itself, so there is exactly ONE: the
per-request one. SSD-first buys a whole-session hot cache, and nothing about
the forward's width.

**How one-session-resident meets evict-to-admit.** The load-time clamp reserves
only the ladder FLOOR once the per-request width gate is on, so a wide request
depends on request-time admission and, when it does not fit outright, on
evict-to-admit: `fitsAfterEviction` credits `reclaimableBytes`, which is
residency minus the LARGEST entry. Under SSD-first the steady state is exactly
ONE resident entry, so that credit is 0. Two cases, and they are not the same
case.

*The prompt EXTENDS the resident session.* Then the resident entry is the
request's own KV: the restore refcount-SHARES those buffers into the slot's
cache, so evicting the entry would return nothing live, and `active_mem` already
counts the bytes so the guard must not add them again. A credit of 0 is not a
conservatism here, it is the correct number — and it is the same number the
#353 follow-up already established for a matching prompt. Nothing to reclaim,
nothing double-billed.

*A DIFFERENT session's request arrives.* Now the resident entry belongs to
someone else. Its disk copy is COMPLETE — its own turn flushed it, and
mechanisms 1–4 are what make that true — so evicting it costs restore time and
nothing else. The inference thread can in fact evict it: `last_restored_used` is
cleared at the start of every lookup and only set by a restore, so a
non-matching prompt protects nothing and `evictLruToAdmit` will take it.

**The connection thread could not see that — so it was given a way to.**
`reclaimableBytes` subtracts the largest entry because the guard cannot know
WHICH entry a prompt will match, only that a restore pins at most one. With one
resident entry that always subtracts the whole cache, so a wide request for a
different session was judged as if a fully-flushed 24 GB entry were immovable,
and refused by name while the eviction it needed was available all along. The
refusal even said "the entry a restore would share is not evictable" when no
restore would have shared it.

The obvious repair is wrong: crediting residency outright fails the MATCHING
case, where the shared buffers genuinely return nothing however complete the
disk copy is — durability is not liveness. The correct rule needs the PROMPT,
and the guard runs on a connection thread that may never touch
`hot_prefix_cache` (inference-thread state, freed on every model switch).

So the prompt meets the cache through a published snapshot instead. At every
site that already republishes the residency scalars, the inference thread also
builds an immutable `[]EntryDigest` — `{fingerprint, len, kv_bytes}`, where the
fingerprint hashes the entry's first `MIN_CANCELLED_COMMIT_TOKENS` ids — and
swaps it under a small mutex, freeing the superseded slice afterwards. The
connection thread hashes the same prefix of the incoming prompt, and under that
lock reduces to a scalar: residency minus the largest entry whose fingerprint
matches. No pointer into cache-owned memory ever leaves.

Two details are load-bearing. An entry shorter than the restore floor gets no
digest at all — nothing can pin it, so omitting it correctly credits its bytes.
And the pin test is the fingerprint match ALONE, deliberately not also
`len <= prompt_len`: an entry whose record is LONGER than the prompt still
shares the floor-width prefix and `restore` clamps to the shorter of the two, so
it really can be restored from. Excluding it would credit bytes about to be
pinned, and over-crediting is the unsafe direction — the guard promises, the
inference thread then evicts nothing, and the request is refused after being
admitted. The refusal text is conditional now too: it claims a restore would
share the difference only when something actually withheld it.

An allocation failure leaves the PREVIOUS snapshot standing rather than clearing
it. A stale digest is a hint; an empty one is a claim that the cache holds
nothing, and that is a lie that credits bytes which exist.

**What SSD-first cannot do: serve a 1M prompt on this box.** The live KV must
be resident — that is the one thing no tier can move — so the longest prompt
this machine can serve is set by the prefill peak, not by the cache. At kv8 a
1M prompt bills 30.3 GB even at the ladder FLOOR against 28.9 GB free, and
guard 5c measures a 512k prompt peaking at 99.05 GB with `max_safe_context`
475,737. So the max servable prompt here is ~476k, and SSD-first does not raise
it by a token. What it changes is what happens to the sessions you are NOT
currently serving: they survive on disk instead of being re-prefilled. Anyone
reading "1M context" off the config and expecting a 1M prompt to be admitted is
reading the wrong number, and it is the same number before and after this work.

**Blast radius.** Every one of these is gated on `ModelConfig.ssdFirstCapable()`
(`model_type == "qwen4_exp"`) AND `MLX_SERVE_PREFIX_SSD_FIRST`, read at exactly
one place — the scheduler's disk-tier attach — into `HotPrefixCache.ssd_first`
and mirrored onto `DiskTier.ssd_first` and the writer arm. Every mechanism reads
that field; none reads a model_type. A source scan pins the single arming site,
and each mechanism's test carries an arm B asserting the legacy path is what
runs when the predicate is false.

**No manifest bump.** SSD-first changes WHEN chunks are written and WHICH
checkpoints are present — never the on-disk format — so the manifest stays at
v5 and no existing entry becomes a miss. What buys that decision is a test
rather than a version number: an entry written by the legacy path must restore
under SSD-first, and an entry written by SSD-first (hand-serialized safetensors
out of the background writer, never `mlx_save_safetensors`) must restore under
the legacy path. Both directions, same values, in one test.

**Two companions.** The `#353` reservation sizes the KV to prompt + max_tokens
up front, and a grow is not in place, so a restore that provokes one holds two
copies of the whole cache at the tightest possible moment. It does not, because
`snapshot`/`restore` refcount-SHARE the capacity buffer: the entry carries the
PREVIOUS turn's reservation with it, and the grow guard (`offset + new_len >
bufferCapacity`) then simply does not fire. That was true by construction and
untested, which is the same as untrue — `KVCache.kv_cap_buf_grows` now counts
the moments a second copy exists, and the guard asserts zero across a restore
whose donor capacity suffices.

Writing that guard turned up something stronger than the claim it was meant to
pin: a reservation is NOT retroactive. `reserve()` raises the capacity of a grow
that HAPPENS; it does not provoke one. So a restored slot that merely reserves
more than the donor holds still allocates nothing — the copy is deferred until
the data actually needs the room, and on a warm turn it usually never does. The
negative arm therefore cannot be "reserve more than the donor" (that also
counts zero, which is why the first version of the test failed); it has to be a
write that genuinely runs past the donor's capacity.

### The external review of PR #363: six defects, all of them in the EVICTION half (2026-09-05)

Every mechanism above is about writing to the SSD. An external reviewer on an
M4 Max with 14 GiB free on `/` found that the writing was fine and the
*discarding* was not — six defects, five of which only bite on a box the
authors did not have.

**A free-space probe that reads the real volume makes every test a property of
the tester's disk.** `refreshDiskBudget` runs on every store in this mode and
calls `statfs`, so with 14 GiB free `diskBudgetFromFreeSpace` returned null
against the 64 GiB reserve, the tier latched `store_declined`, and the
write-through test went red with `kv_len` stuck at the first commit's value.
The engine was correct; the suite was measuring the machine. `DiskTier.space_probe`
is now injectable (`armTestSpace`), every SSD-first test arms it, and a scan
pins the pairing so the class cannot come back. Writing the inverse case — 10
GiB free, the store must decline — immediately found a second defect underneath:
the `store_declined` early-out sat ABOVE the refresh, so the commit that first
observed a short volume latched the flag and then wrote anyway. A filling disk
always got one entry more than the budget allowed.

**A bool that means "nothing more to write" also means "I wrote nothing".**
This is the one that mattered. `appendCommitWithSpec` returned `true` for a
completed copy AND for every silent skip: under `MIN_PERSIST_TOKENS`,
TurboQuant, a layer offset short of the range, a non-B1 shape, a declined
volume. `spillIdleEntries` read that bool as "the SSD holds this session" and
called `evictAt`. So on qwen4_exp with `--prefix-cache-disk` on and a disk under
~65 GiB free, EVERY idle hot-cache entry was discarded from RAM at the end of
EVERY request with nothing whatsoever written in its place — the mode's promise
("idle sessions live on the SSD") inverted into "idle sessions are deleted".
The same shape, quietly, for a TurboQuant boot or a sub-512-token session on any
disk. `PersistOutcome { persisted, partial, skipped }` names the three, and only
`.persisted` licenses eviction. And `.persisted` is only the write path's
*claim*, so the INDEX must agree too (`holdsFullPrefix`: an entry at the same
key whose `kv_len` reaches the persist target and whose `chunk_bytes` has one
non-zero size per chunk that length implies — the same array `scan` clamps
against real file sizes after a `kill -9`).

**Writing and evicting are two decisions.** The spill evicted every non-newest
entry on every `finishSlot`, ignoring the resolved `--prefix-cache-mem`
entirely, so two alternating sessions bounced off the SSD on every single turn
even though RAM had been budgeted to hold both. Writing is cheap and stays
unconditional; EVICTING is what the allowance bounds. The allowance had to be
plumbed to reach the cache at all — `prefixCacheMemForLoad` now writes it
through an out-parameter beside its return value, because on this arm the flag
means the idle allowance and the return value is the whole budget including the
live session.

The allowance is a HARD cap, shed in two tiers, and the ordering is what lets
it coexist with the rule above: shed idle entries that have a proven durable
copy first, oldest first; only if still over, shed the rest, oldest first, with
a log line naming why. An unpersistable entry therefore survives while the cache
is under the cap and is dropped only past it — losing WORK (a cold prefill),
never data. A soft cap was the tempting alternative and it is wrong: the
allowance exists to bound RAM for the NEXT admission, and "0 = nothing idle
stays resident" has to mean what it says even when the disk refuses.

**A mode is not an arch.** `ssdFirstEnabled()` never checked that a disk tier
exists, and `--prefix-cache-disk` is OFF by default — so out of the box
qwen4_exp armed the mode with no tier underneath it, took the "one full-context
session + idle" budget floor (~20 GB, plus the whole ask at 1M) and could run
none of the spill machinery, because every mechanism needs somewhere to write.
A budget sized for a tier that does not exist is RAM the server cannot use.
`ssdFirstActive(config, has_disk)` is now THE predicate, asked at both sites
(the load-time budget and the arming), scan-pinned; the arming moved BELOW the
attach block, because the tier is part of the answer.

**An arm that short-circuits above a fix does not get the fix.** Audit S1
caught this for the GPU ceiling; the same argument list carried a second
instance. `ssdFirstBudgetForLoad` returns before `planHotCache`, so
`clampReserveWidth` — the fix that made the clamp monotone in the ceiling by
reserving only the ladder FLOOR on an arch that re-bills its width per request —
never ran on the one arch these gates are about. The arm now derives its clamp
reserve through the same helper. Two reserves, exactly as on the RAM arm: the
SESSION is billed at the pinned width (that is what `computeMemoryContext`
advertises against), the BUDGET at the floor (the load-time reserve is only a
promise to the FIRST request; every request re-bills its real width at
admission).

**A durability check that WAITS is a decode stall.** The audit-S3 fix above
("drain the writer, then re-read `writeErrors()`") was correct about durability
and wrong about where it ran: `drainWriter` waits on the whole queue, and
`spillIdleEntries` runs on the INFERENCE thread inside `finishSlot`. Every
finished request with a flush outstanding parked decode until the background
writer caught up — precisely the stall the writer was added to remove. The
question is now asked without blocking (`Writer.pendingPrefix` ->
`DiskTier.entryWritesPending`): an entry whose files are still in flight is not
evictable on THIS pass, the pass returns, and the next one asks again. Nothing
is lost by waiting — the entry is safe in RAM meanwhile — and no drain runs on
the inference thread at all. Scan-pinned over both end-of-request functions.

### The JSON grammar's token->bytes table was a process singleton (relocated from CLAUDE.md, 2026-09-05)

The table is derived from a MODEL's vocabulary, but it was built once per process. A second model
loaded into the same server then masked with a foreign vocab — the grammar was right and the bytes
it matched belonged to someone else. It lives on `LoadedModel` now (`grammarTokenBytes`), which is
the general rule: a cache derived from a model is per-model, never per-process.

### Three audit findings that stayed stories (2026-09-05)

These three came out of the bundle audit. They are rules in spirit, but CLAUDE.md was at its
byte floor when they landed, and the growth policy's own answer to that is a story here rather
than a symbol dropped there. Named so the next reader can find them.

**A published snapshot is freed by the PUBLISHER, below the join** (`hot_cache_digests`, audit
B0). The inference thread publishes an immutable digest slice and frees the one it supersedes,
once per request. A `deinit` that frees it *above* `t.join()` therefore races the publisher: either
a double free, or a connection thread reading a slice that has already gone. The ordering is the
invariant — the free belongs below the join, in the same thread that publishes — and it is
scan-pinned, because the wrong order compiles and serves correctly until the one interleaving.

**A "complete" disk commit is STAGED, not durable** (audit S3). The background writer logs a failed
blob, counts it and drops it; the entry is still "committed" from the caller's point of view. So
anything that discards the RAM copy on the strength of a disk commit must first confirm the files
actually landed. Treating the commit itself as durability is how a cache entry that exists in
neither tier gets created. The first fix confirmed it by DRAINING the writer, which was a decode
stall on the inference thread — see item 6 of the external review below: the confirmation is now
non-blocking (`entryWritesPending`), and an entry still in flight simply waits for the next pass.

**An index-less entry directory is another process's flush in progress until proven old** (audit
S4). Meta lands last by design, so a directory without it is indistinguishable from a partial
write — and our epoch fence is per-process, so it cannot tell a *second* server's in-flight flush
from our own debris. The sweep therefore only reclaims past `STRAY_MIN_AGE_NS`, and treats
unreadable as young rather than as garbage.

## The prefill width, re-chosen per chunk (2026-09-05)

The per-request width above answers "how wide may this prompt prefill?" once,
at admission, and then holds that answer for the whole prefill. A 1M-token
prefill on Flash Next runs for minutes and 256 chunks. Everything the answer
was based on can move in that time.

### What actually moves, and what does not

The obvious story — "the KV fills as the prefill proceeds, so the width has to
narrow" — is FALSE for every prompt this feature targets, and getting that
wrong would have shipped a mechanism that fires for a reason that does not
exist. `KVCache.reservedTokens` reserves `seq + headroom + chunk` up front for
any prompt past `RESERVE_MIN_TOKENS` (32,768) and `nextCapacityReserved` grows
the buffer exactly once, so the whole request's KV — 21.7 GB at 1M tokens on
the deployed pack — is resident before the first chunk runs. Re-billing it per
boundary would have subtracted the same bytes twice and walked every prompt to
the ladder floor.

What does move between boundaries:

- Another slot's decode, hosted by `interleaveDecodeTick` at the very
  boundaries this decision is taken at. Nothing bills it against this prefill.
- A process outside the server. `currentGpuMemoryCeiling` is
  `min(static, footprint + free RAM)`, so a docker stack starting mid-prefill
  lowers the ceiling under a forward that was admitted against the old one.
- The state that genuinely accretes per chunk: the QSA indexer key history
  (`statePerTokenBilled`), the retained SSM checkpoints (~1.9 GB at the cap),
  the MTP and DFlash contexts.
- `MLX_SERVE_KV_RESERVE=0`, and any prompt under 32,768 tokens, where the KV
  really does grow chunk by chunk.

So this is a safety net and a multi-slot feature, not a single-session speed
win. On a quiet box serving one request it changes nothing at all, which is
the correct amount.

### The rule is asymmetric because the failure is

A Metal working-set abort is uncatchable — the process dies. Being one rung
too narrow costs throughput. The two directions are therefore not the same
bet:

- **HOLD at margin 1.0.** `prefillMemoryNeeded` already multiplies its whole
  bill by 5/4; a second 1.25 on the hold would have taken a 768k prompt at the
  default ceiling from the width admission ADMITTED (2048) down to 1024 at its
  very first boundary — the feature making the request slower than not having
  it. The margin belongs on the direction that is a bet, not on the one that
  is a continuation.
- **STEP DOWN immediately**, by as many rungs as it takes. Waiting a boundary
  per rung is waiting inside the abort.
- **WIDEN one rung at 1.25, after TWO consecutive supporting probes.** The
  probe is taken after the boundary's `mlx_clear_cache`, so it never contains
  a chunk's own peak; the second reading is what pays for that blind spot.
- **One-way ratchet.** A prefill that has stepped down never widens again. The
  pressure that took the width away is a property of this minute, and
  re-widening into it turns a safety net into a metronome — the step-down
  frees exactly the bytes that make the wider rung look affordable again.

### The cliff is 2048, not the ladder

`PREFILL_DQ_GEMM_MIN_M` is 2048: widths at or above it take the dequant+GEMM
prefill route, 1024 and 512 do not, and most of the measured 25-34% spread
between the widest and narrowest rung lives at that step. 2048 is the rung
worth defending; 512 versus 1024 is per-chunk fixed cost only. A policy that
treats the ladder as uniform will spend its caution in the wrong place.

### Two placements that no unit test can see

The probe sits AFTER the chunk's `mlx_clear_cache` and BEFORE the interleave
tick. Before the clear it reads the chunk's own peak and narrows on memory
that is already gone; after the tick it reads a co-tenant's decode
allocations as this prefill's pressure. Both are scan-pinned, by index, in
generate.zig.

### Checkpoints do not move

`ssm_cp_stride` is coarsened against the LAUNCH width
(`max(PREFILL_CHUNK, prefill_chunk_override)`), never the per-request pin and
never the per-chunk width, and every ladder rung divides it. `nextChunkEnd`
truncates any chunk that would cross a stride boundary, in ABSOLUTE position.
So a prefill that walks 4096 -> 512 -> 2048 ends on exactly the stride
boundaries a fixed-width one would have, in the same order — the prefix cache
sees the same restore points either way, and the hermetic test walks a
deliberately nasty width sequence at two strides, two offsets and two prompt
lengths to say so.

Output is NOT byte-stable across widths, and no test asserts that it is: a
different width is a different GEMM shape, and 4-bit near-ties flip. That is
the same bar `#197` set for chunked vision prefill (perceived content), and
the same reason `tests/test_vision_chunked_prefill.sh` compares colours rather
than bytes.

### Scope and switches

`ModelConfig.perRequestPrefillChunk` (qwen4_exp) gates both features, so the
per-request kill switch takes this one with it;
`MLX_SERVE_PREFILL_CHUNK_ADAPTIVE=0` disables only this one. An explicit
`--prefill-chunk` or `MLX_SERVE_PREFILL_CHUNK` outranks it — an operator who
pinned a width pinned every forward — and so does the unchunked vision arm,
which has no next chunk to size. The ADMISSION bill never learns about any of
it: the width a request was admitted at is the width it is billed at.

One line per transition, and one summary per request that moved:

```
[prefill] width 4096 -> 2048 at pos 606208 (headroom 2216 MB, reserve 2287 MB)
[prefill] adaptive: 261 chunks, width 1024..4096, 2 change(s)
```

### What the audit caught (2026-09-05, fix round)

Four findings, all of them the same shape: a decision taken at a NEW moment
kept reading numbers built for an OLD one.

**The bill was a load-time bill (blocker).** `prefillChunkCost` reached
`prefillTransientReserve`, which prices the QSA score sheet at `kv = chunk`
because at load there is no KV. Past the indexer budget
`qsaScoreRowsPerChunk` saturates, so the real sheet is ~its whole budget
rather than the tiny `fwd x nb` product — ~0.5 GB under-billed per decision at
the default 256 MB, ~9 GB at `MLX_SERVE_QSA_SCORE_SHEET_MB=4096`. The ratchet
did not cover it: it blocks widening only AFTER a step-down, so the first
widen at high `pos` was unguarded. `pos` was reaching
`adaptivePrefillWidthNow` and being spent on the LOG LINE. Fixed by
`prefillTransientReserveAtKv`, with `prefillTransientReserve` as its
`kv = chunk` case so every load-time caller keeps its number. The
project's own rule was already written down one function away: *one estimator
means one set of INPUTS too*.

**The log quoted a smaller sibling of the number it compared.** The decision
was `prefillChunkCost`; the line printed `prefillTransientReserve`. An
under-bill therefore rendered as a comfortable margin.

**Attribution and safety wanted opposite orderings (S17).** Probing before the
interleave tick keeps a co-tenant's decode out of this prefill's pressure —
right for attribution. But a widen decided on pre-tick headroom then forwards
into memory the tick allocated and the probe never saw. The two directions
split: a step-down commits from the pre-tick branch, a widen is re-priced
after the tick (`adaptivePrefillWidenStillFits`) and withdrawn if it no longer
fits. A step-down never waits — waiting is waiting inside the abort.

**Host bytes are memory too (S11).** The SSD write-through stages a full host
copy of the chunk behind a ~1 GiB permit, at exactly this boundary, and
`mlx_get_active_memory` counts device bytes only. On unified memory both come
out of one pool, so the writer publishes `staged_disk_host_bytes` and
`prefillHeadroomNow` subtracts it.

**And an old constant became reachable (S18).** `TAIL_MERGE_MAX` is a flat 512
justified as "~6% at 8192". At `PREFILL_CHUNK_FLOOR` it is +100% of the
transient the step-down just bought. `tailMergeMax(width)` keeps the original
~6% bound at every rung and is a no-op at 4096 and 8192.

**...and it is GATED, because "only this feature can reach it" was false.**
The first version replaced the constant inside `nextChunkEnd`, which has no
arch parameter — so the scaling applied to every arch whose resolved chunk is
under 4096, which is a lot of them: `resolvePrefillChunk`'s machine ladder
puts a 27B on a 16 GB Mac at 512; `boundedPrefillChunk`'s score-budget floor
and its composed-causal 2048 cap hit gemma4, qwen3_5/3_6, muse_glimmer and
deepseek_v4; any `--prefill-chunk` or `MLX_SERVE_PREFILL_CHUNK` under 4096
hits everything. Chunk boundaries are not byte-stable, so that was a
behaviour change on archs where the bound was never measured — and the
sentence claiming otherwise is exactly what a future reader would use to skip
the A/B. The bound now goes through `tailMergeMaxFor(width, adaptive_width)`
and every other arch keeps the flat constant byte for byte.

**And the first gate was wrong in the same shape, one level down.** It read
`adapt_chunked and options.chunk_width_hook != null`, on the belief that the
hook exists only where the adaptive width does. `serve` installs
`prefill_chunk_adapt` **unconditionally and process-wide**, and the
scheduler's one `InitOptions` site installs the hook whenever that global is
non-null — so under a real server the hook is non-null on *every* arch and the
scaled bound was live everywhere again. The width still held elsewhere, but
for a different reason than the gate assumed: the POLICY behind the hook
declines (`adaptivePrefillChunkEnabled`), the hook is not absent. Two tests
sat green over the defect because both exercised the flag and neither
exercised the wiring. The flag now comes from the arch predicate itself,
per model: `scheduler.prefill_chunk_adaptive_enabled` (fifth of the admission
hook family, `= &adaptivePrefillChunkEnabled`) →
`scheduler.adaptiveChunkWidthFor(cfg)` → `InitOptions.adaptive_chunk_width`.
Guards: `server.zig` "the tail-merge gate reads the ARCH, not the installed
hook" (installs the whole family, then asserts the hook's presence and the
arch's answer DIFFER for qwen3_5), `server.zig` "the tail-merge bound scales
ONLY where the per-chunk adaptive width is live" (the predicate), and
`generate.zig` "the scaled tail-merge bound is gated on the per-chunk adaptive
width" (both arms, plus a scan that the loop reads the flag and that the
hook-presence spelling is absent from the loop body).

### Rules this produced

- A helper's arguments encode the QUESTION it was written for. Reusing it from
  a new moment silently answers the old question — thread the new input
  through and keep the old call as a named special case.
- A constant a new feature makes *interesting* is not a constant only that
  feature can *reach*. Before changing one in a shared helper, enumerate every
  route into it; if any predates the feature, gate the change on the feature
  rather than on the value that made you look.
- **A hook's PRESENCE is not a capability gate.** A hook installed once at
  startup is a property of the HOST, not of the model, the arch or the
  request; if the policy behind it answers "no" per model, ask the policy.
  Reading `!= null` as "this arch opted in" is how a correctly-gated feature
  becomes ungated again — and a test that exercises the resulting FLAG cannot
  see it, only one that builds the wiring the way the server does.
- When attribution and safety want opposite orderings, do not pick one. Split
  the decision by direction: the safe direction commits early, the risky one
  is confirmed late.
- Host allocations on unified memory are working-set pressure. A probe that
  reads only the device allocator is optimistic by whatever the host side is
  holding.
- Establish which quantity actually moves before you write the controller. A
  plausible mechanism that does not happen is worse than no mechanism: it
  fires for the wrong reason and the reason is invisible.
- When one direction of a control loop is fatal and the other is merely slow,
  the margins are not equal, and the safe-looking symmetric choice is a
  regression.
- A probe taken where the transient has just been freed cannot see the
  transient. Pay for that with a second reading, not with a bigger margin.

## N5: the prefill width is re-asked after the eviction pass, and admission logs on the admit path too (2026-09-05)

The per-request prefill width is a function of what memory is FREE at the
moment it is chosen. A request whose bill only fit at the ladder's narrowest
rung was admitted at that floor — correctly, given what was free at admission
time — but if admission then ran an eviction pass that returned gigabytes back
to the pool, the prefill ran at the floor width for its ENTIRE length, never
re-pricing against the memory eviction had just freed.

The fix is a second look, not a wider bill up front: crediting `reclaimable` or
`accounted_bytes` to the chooser ahead of eviction would justify a width against
memory the allocator has not actually returned yet, and the failure costs are
asymmetric — a needlessly narrow width only costs throughput, while a width
justified by memory that turns out not to be free is an uncatchable Metal OOM.
So `scheduler.postEvictionPrefillChunk` asks the width chooser TWICE around the
eviction pass: once before (the same hook, the same inputs, the same live
memory reading as the probe that just failed to admit at a wider rung — and
only on the path that actually runs eviction), then again after, crediting only
the LIVE delta the allocator actually reports (`EvictionReport.bytes` — the
allocator's own before/after reading), never the pre-eviction ESTIMATE
(`accounted_bytes`/`reclaimable`), which prices what eviction is expected to
free rather than what it did.

`postEvictionPrefillChunk` RUNS THE RE-ASK, in both directions. It shipped as
`@max(admitted, reasked)`, sold as an assertion that post-eviction available
memory can only rise — a canary for an impossible case, free on the expected
path. The external review of PR #363 pointed out that this is backwards: a max
is not an assertion, and the only case in which it does anything at all is the
case where the invariant BROKE. Where the invariant holds the max is a no-op;
where memory moved under us between the two readings (a co-tenant slot's decode
allocations) it discards the reading taken against live memory and keeps the
stale, WIDER one — widening onto memory that is gone, which is exactly the
uncatchable Metal OOM the asymmetry argument above exists to avoid. The narrow
arm only ever costs throughput. So the second reading is the one that runs, and
the direction is only LOGGED: `[prefill] re-ask:` at info when the pass bought a
wider rung, at warn when it came back narrower. The width still never falls
below the ladder floor, because the CHOOSER cannot produce anything narrower
(`chooseRequestPrefillChunk` falls through to the floor rung) and this function
computes no width of its own. `cap_adapt`,
the per-chunk adaptive ceiling, is built PIN-LESS — via `effectivePrefillChunk(…,
0)`, a literal zero rung rather than the request's pinned width — specifically
so it can never be the thing that clips a re-ask back down to the pre-eviction
number.

The width reaches its consumers through one chain: `req_prefill_chunk` ->
`InitOptions.pinned_prefill_chunk` -> the runtime `PREFILL_CHUNK` constant ->
`cur_chunk`. Both that chain and the `cap_adapt` seam are scan-pinned so a
future refactor cannot quietly reintroduce a second, narrower source of truth
for the chunk width.

**The second half of N5** closes a related visibility gap: the admission path
only ever logged when a request was REFUSED, so the numbers that explained a
refusal only ever appeared after the fact — there was no way to see, on a
request that succeeded, how close it had come to the ceiling. `server.checkAttentionMemory`
(or the equivalent admission chokepoint) now emits exactly ONE line per
request, on every request, regardless of outcome:

```
[admission] needed=<bytes> available=<bytes> reclaimable=<bytes> width=<chunk> verdict=admit|evict|refuse
```

The line is built from the SAME `AdmissionBill` value the three admission arms
(admit outright, evict-then-admit, refuse) already act on and that a refusal
already quotes — this is not a second estimator call, which would reproduce the
class of bug #126 documents (a gate computing its own answer instead of reading
the one that runs the actual decision). `admissionVerdict` derives the verdict
string from the bill using the arms' own two predicates, so the logged verdict
and the actual decision cannot diverge.

Logging on every admitted request unconditionally would be too noisy at
sustained load, so `admissionLogLevel` picks the level BEFORE any string is
formatted (so a `debug`-level line that the configured level would drop is
never built at all): `info` on the first request after a model load
(`model_registry.load_generation`, consumed by one compare-and-swap so
concurrent racing request threads produce exactly ONE `info` line, not one per
thread) or whenever `needed` exceeds 0.9x `available` (the request is close
enough to the ceiling that an operator watching info-level logs should see it
land); `debug` otherwise. The load-generation token is checked for
consume-ability before the level decision short-circuits, so an operator
running below `debug` doesn't pay for a CAS on every request only to discard
the log line.

Tests: `postEvictionPrefillChunk` (widens after an eviction that freed memory,
holds the single ask when eviction freed nothing, RUNS a narrower re-ask and
flags it as `moved`, is a no-op when the two readings are unchanged, and hands
back the ladder floor unmodified), a scan that its body contains no `@max` over
the pair, the
composition of the two-ask logic against `chooseRequestPrefillChunk` including
both the gate-disabled and non-qwen4 arms, a scan of the four-step admit/evict/reask/apply
order, the `cap_adapt` pin-less-seam scan, `admissionLogLevel`'s
load-generation and 0.9x-threshold branches, `admissionVerdict` against the
arms' own predicates, a same-bill scan proving the admit and refuse arms format
from one shared value, the post-load single-CAS arming scan, and
`log.enabled`/`log.atLevel` gating.

## The SSD-first write-through was 100% of a warm turn's TTFT regression (2026-09-05)

On the warm control ladder (a restored prefix plus a 31-token instruction tail)
the SSD-first bundle prefilled new tokens at 60-98 tok/s where the branch
without the write-through ran 196-235. Decode was equal, the adaptive width
probe never fired (a warm turn is one chunk, so `pos == loop_end` and the hook
is not called), and every other candidate was bounded by a 2-3 ms residual. The
whole gap is the prefill write-through, which runs inside the chunk loop on the
inference thread and therefore lands inside `prompt_ms` and inside TTFT:

| rung | `prompt_ms` with | without | Δ | write-through persist | unexplained |
|---|---|---|---|---|---|
| 16k (16,357 tok) | 316.1 | 133.3 | **182.8** | 181 ms / 203.7 MB / +16 chunks | 1.8 ms |
| 32k (32,427 tok) | 506.3 | 144.9 | **361.4** | 358 ms / 403.9 MB / +32 chunks | 3.4 ms |
| 64k (65,665 tok) | 899.3 | 156.5 | **742.8** | 741 ms / 818.0 MB / +65 chunks | 1.8 ms |

Matched 1:1 by the persist at every rung, reproduced on a second boot
(186/369/753 ms of persist against Δ 204/371/755 ms), four warm arms per rung
all within 6 ms. Not variance. Source: `~/claude-tmp/bench-qwen4-ladder/WARM_TURN_TRACE.md`.

**Half of it was a per-TENSOR GPU sync wearing bandwidth's clothes.**
`DiskTier.serializeSafetensors` called `mlx_array_eval` on one array at a time.
12 KV layers x 6 affine buffers x 32 chunks is ~2,300 full syncs on a 32k warm
turn — a fixed per-tensor cost that reads as a flat ~1.13 GB/s, against the
7-8 GB/s the same data reaches through `mlx_save_safetensors` on the same box.
The tell was in the same log: the end-of-request disk commit goes through mlx's
own saver (which evals ONCE) and ran 7x faster per byte than the write-through.

The split is `materializeContiguous` (the contiguous pass, then ONE batched
`mlx_eval` over a vector — exactly what `mlx::core::save_safetensors` does) plus
`encodeSafetensors` (pure header + payload; no stream, no eval). The output is
byte-identical, and that is the point: **the eval strategy is a latency
decision, never a format one.** A regression here is byte-invisible, so it is
pinned three ways — a timing-free `serialize_eval_count` demanding exactly one
eval per chunk file, the pre-fix per-tensor materializer kept as a test-only
golden with the two images compared byte-for-byte, and a scan fixing the SHAPE
(one eval, batched form, at function-body indent after the loop closes).

**The other half was persisting a prefix that had nothing new in it.**
`writeThroughArmed` now also takes this turn's un-cached span
(`prefill_tokens.len`) and declines below one `chunk_tokens`
(`writeThroughSpanReached`, a pure helper with its own unit test; a scan pins
that the arming site passes the un-cached tail and that the gate compares
against the tier's OWN `chunk_tokens`). A restored 32k prefix plus a 31-token
tail has no chunk-aligned progress a cancel could lose, so the turn was paying
in-request for a prefix the end-of-request commit persists anyway — and that
commit runs after the response text, where it costs the user nothing. Declining
is announced once, not per request.

The rule underneath both halves: **the write-through's job is to make a KILLED
prefill restartable, not to be the persistence mechanism.** It buys nothing on a
turn that cannot lose a chunk, and anything it does buy is charged to TTFT — so
its cost per chunk is a first-class number, not an implementation detail.

## The QSA indexer history was copied per snapshot, and billed as if it were not (2026-09-05)

Flash Next's sparse-attention indexer keeps a raw-key history beside the KV. A
prefix-cache commit used to hand the new entry its OWN copy of that history, so
a slot and the entry it just committed held two copies of the same bytes — and
the admission bill priced only one. Both halves are now one decision:
`handoffQsaHistoryToLatest` lets the newest snapshot take a VIEW of the live
buffer at COMMIT (ONE copy per slot ∪ entry), and `statePerTokenBilled` prices
what is actually held — ONE history copy plus the f32 score bank, 5,376 B/tok.

The bill moves from 20,736 to 18,432 MiB at 1M context, and measured decode
residency falls 3-4 GB in the 786k-1M band. `MLX_SERVE_QSA_HISTORY_SHARE=0`
restores the prefill-end copy and the 2x bill, which is also what
`MLX_SERVE_KV_RESERVE=0` leaves in place — both arms are billed, so the
estimator is never optimistic about the arm that is running.

The rule: **a buffer two owners can share is billed once only after the sharing
is real.** The pre-fix code had the sharing in neither place and the bill in
one; the fix is the pair, not either half.

## SSD-first: every prefix-diverging turn re-persisted the whole entry — chunk sharing by hard link (Defects A+B, 2026-09-05)

Control run on the bundle: `[disk-cache] persisted 32426/32426 tokens (+32 chunks, 403.9 MB, 363ms)` on a 31-token warm turn; the judge's 64k rung: `reused 24576/65665 … persisted 32768/32768 tokens (+32 chunks, 464.5 MB, 375ms)` — TTFT +369 ms at 32k, +737 ms at 64k, ~10 s per turn at 786k. Two defects.

**Defect B** (fixed in 9f43b7a): the write-through serialized with one eval per tensor and ran on sub-chunk warm turns. **Defect A** (this story): an SSD-first entry is committed at the END of a request, so its last tokens are the reply the model just generated — and the next turn's prompt contains that reply and then diverges INSIDE it, so a strict-prefix scan finds no usable common tail and rewrites every chunk of a multi-hundred-MB entry, every turn, for a conversation whose bytes are already on disk. Mechanically: `DiskTier.appendCommitWithSpec`'s extend scan reuses chunk files only under a STRICT prefix relation — `e.tokens` a prefix of the new tokens, or vice versa. A persisted entry's tokens are `prompt ++ generated`, so the next prompt (re-tokenized assistant text + `<|im_end|>` + the user turn) diverges INSIDE the generated span, neither relation holds, `next_id` mints a fresh `e<id>` and `writeChunkFile` writes every chunk from 0 again. Nothing bounded it on the SSD-first arm (`max_flush_bytes` = 2 GiB), and the caller was `prefillWriteThroughCb` at the first chunk boundary — inside the prefill, before the first token.

Fix, two halves. (1) `chunkShareDonor` picks the resident entry (same `has_tools`, same kv-quant config by `std.meta.eql`, and the same model fingerprint by construction — `self.root` IS the fingerprint dir) with the most WHOLE chunks under the common prefix, clamped to the KV this commit holds and the donor's persisted length; `linkInheritedChunks` HARD-LINKS the donor's LANDED leading chunks into the new `e<id>/` (`std.Io.Dir.hardLink`, same filesystem) — a chunk has landed when its final-name file exists at the manifest's size and `Writer.isPending` says no write to it is queued or in flight — and the heir writes everything from the first un-landed chunk on. Audit B-A1: the first cut called `Writer.fence(donor_dir)` to wait for the donor's files; `fence` DISCARDS what it matches (it exists for a directory about to be deleted), so it destroyed the donor's queued chunks and meta while the donor still claimed them — the common state of a donor once the write-through lands one chunk per boundary. The donor's queue is never touched now; the "never rewrite an inherited chunk" invariant is a runtime check that cuts the links from `keep` on and writes those chunks, not a `std.debug.assert`. `[disk-cache] chunk share: e5 inherits 4 chunks (…) from e3 by hard link` is the engagement line. (2) `prefillWriteThroughCb` passes `WRITE_THROUGH_FLUSH_BOUND_BYTES` (1) to `appendCommitBounded` — one chunk per boundary, so a killed prefill still leaves a chunk-aligned restorable prefix; `flushPendingDisk` in `finishSlot` completes the rest after the response.

Accounting — `total_bytes` is bytes on disk BY CONSTRUCTION, and the filesystem is the refcount: an heir bills 0 for inherited chunks (`inherited_chunks`, meta.json v6; `bytes` = files the entry created); `removeAt` frees `nonChunkBytes` plus every chunk file whose `stat.nlink == 1` (`bytesFreedByRemoving`), so the donor's removal keeps the shared files billed and the LAST holder frees them; `scan` counts every inode once (`billChunksOnce`), so a heir whose donor died before a reboot becomes the payer — the budget, the sweep and the eviction all read the same number, and a shared chunk cannot be freed while another entry still names it; extend and ssm-only commits move `total_bytes` by file DELTAS, never by recomputing an entry's bill. A manifest counter would have gone stale in the crash window between the link and `meta.json`; `nlink` cannot — and a crash there leaves an `e<id>` without meta.json, which `scan` deletes, dropping the extra link. A rewrite never lands on a link: links are whole chunks below `inherited`, and the rewritten partial chunk sits at `keep >= inherited`. Bars: the donor-then-heir / heir-then-donor test (`total_bytes` returns to the pre-commit value both ways), the link+restore test (one inode, two links, heir restores whole through a fresh scan with the same bill), the legacy-arm + kill-switch test (never links), v6 round-trip + v5 load, and the hook-bound scan pin. `MLX_SERVE_SSD_CHUNK_SHARE=0` restores the write-everything commit.

Follow-up: the LEGACY (non-SSD-first) tier has the same per-turn rewrite and is deliberately gated OFF here (qwen4_exp blast radius); it shares `appendCommitWithSpecBounded`, so enabling it is `chunkShareDonor`'s `ssd_first` check plus its own A/B.

## Restore by MOVE: a shared restore cannot donate, and the first append pays for it

**Symptom.** Warm TTFT on qwen4_exp scaled linearly with context even though the turn
forwarded 31 tokens: 198 ms at 128k, 277 ms at 256k, 360 ms at 384k — `prompt_ms ≈ 117 ms +
0.617 µs × prompt_tokens`, i.e. two thirds of the wait was proportional to a prefix that was
already in the cache. At 1M that projects to ~0.6 s of pure bookkeeping per warm turn.

**Cause.** `KVCache.restore` binds the entry's buffers into the slot with `mlx_array_set`,
which is a C++ `array` copy-assign in mlx-c — a refcount bump, deliberately, because the hot
cache must keep the entry restorable for the next request. But mlx donates a `slice_update`'s
input only when `array::is_donatable()` holds, and that is
`array_desc_.use_count() == 1 && data.use_count() == 1` (`mlx/array.h`). The entry's second
reference fails it, so the very first `writeAtOffset` of the turn fell through
`SliceUpdate::eval_gpu` → `copy_gpu` and privatised the ENTIRE prefix: 13,056 B/tok × 392,966
tokens = 5.13 GB, ~110 ms, 45% of the warm TTFT. Every subsequent append donated (its input
was the uniquely-owned output of the first), and a cold prefill was never affected — which is
why nothing in the KV growth counters could see it: **a copy-on-write is not a grow.**

The same root cause fired a second time on the QSA side for a different reason. A restore
republishes `aux_state` (the authority) but leaves the append ACCELERATOR empty, so
`qsaAppendKeys` re-seeded with `materializedOwnedCopy` — producing a TIGHT `held`-row buffer
that `capBufAppend` then grew one line later, `mlx_zeros` of the whole reservation plus a
`slice_update` of the entire history. Two full passes over the prefix where one does, on the
raw-key bank (3,072 B/tok) and again on the pooled bank (768 B/tok): ~81 ms at 384k.

**Fix, and the ownership it costs.** On an SSD-first FULL-prefix hit — the entry's whole token
record is a prefix of this prompt, which is what makes the commit's replace path land on this
same entry — the restore is followed by a CHECKOUT: `KVCacheSnapshot.releaseHandles()` drops
the entry's own handles, the slot becomes the sole owner, and the append donates in place. The
copy disappears rather than moving.

The price is that the record now outlives its bytes, so the entry is marked
`checked_out_by = <slot>` and:

* it is **invisible** to every other reader — `findBestRestorableMatch` (a second slot gets a
  MISS, never an empty snapshot), `lruIndexExcluding` (eviction would free nothing and discard
  the record the owner is about to replace), `spillIdleEntries`, `digestsAlloc`,
  `reclaimableBytes`/`reclaimableBytesFor` (its bytes come OFF the base — crediting them is the
  unsafe direction), and the commit path's "kept resident prefix" one-shot;
* it is **reclaimed** by the commit that replaces it, which installs the grown buffers and
  clears the mark;
* it is **dropped** at slot end otherwise, with `[hot-cache] checked-out entry dropped: <reason>`.

That last rule is where the class bug lives. There are TWO slot-end paths and they do not
chain: `finishSlot`, and the inference loop's cleanup drain — which is where a decode-phase
cancel lands, having been pulled straight out by `complete()` without ever reaching
`finishSlot` (the same seam the commit-in-the-drain guard exists for). Both releases are
unconditional (a pad-only, errored, oversized-and-declined or refused slot ends without a
commit, and "the commit ran" is exactly the fact this cannot depend on) and the drain's sits
BEFORE `s.deinit()`, which frees the very buffers the entry would otherwise still claim. Scan-pinned.

**Rules.**

* Restore by move is gated to the SSD-first arm (`ssd_first`, qwen4_exp today) and killed by
  `MLX_SERVE_RESTORE_MOVE=0`, which is the refcount-share byte-for-byte. Partial-prefix hits
  keep the share: they make no promise about the commit, and the entry is still worth keeping.
* A checkout is declined while a `pending_disk` record shares the same buffers. mlx's own
  use_count test would decline the donation anyway; the guard is there so nobody has to reason
  about a flush reading buffers a slot is appending into.
* `seedCapBuf` seeds a history accelerator AT its reservation. Byte-identical to the pair it
  replaces — `capBufAppend`'s grow builds exactly that array — and the bar is that the grow it
  would have done becomes a no-op: **1 seed, 0 `qsa_cap_buf_allocs`.**

**The observable.** Buffer IDENTITY is the only in-process evidence of donation:
`testKeyDataPtr` reads the layer's data pointer after an eval, and "the appended buffer sits at
the address the prefix already occupied" IS "no copy happened". The copy arm cannot fake it —
its donor is still alive in the entry, so the allocator cannot hand the copy that same address.
Both arms then assert the SAME BYTES: the arms differ in ownership, never in output.

## A WARM turn was billed for the prefix it was about to SHARE

Live 2026-09-05, qwen4-ladder run on `4d180d1`. The 768k rung's SECOND request
— 786,707 tokens, 31 more than the entry the turn before it had just committed
— was refused before its first forward:

```
[admission] needed=25866 MB available=19032 MB reclaimable=10294 MB width=512 verdict=evict
  prompt 786707 tokens needs ~25866MB, ~19032MB available + ~10294MB reclaimable
    hot cache (of ~24826MB resident) — admitting, the prefill will evict
  [hot-cache] reused 786676/786707 tokens (matched 786695; entry 2/2)
  [hot-cache] evicted 1 entries (9397 MB live, 10294 MB billed) to admit a 786707-token prefill
[scheduler] prefill refused: 786707 tokens do not fit even with an empty hot cache
  prompt 786707 tokens needs ~25866MB at prefill chunk 512 …, ~19320MB available
    after evicting ~14531MB of hot cache — refused before prefill
```

It restored its own 786,676-token entry from RAM, destroyed the OTHER session's
523,887-token entry on the way, and was then refused anyway. The same request
served on `5c4b4bc`, so it read as a fix-round regression — and it is not. No
commit in `5c4b4bc..4d180d1` touched the bill: `prefillMemoryNeeded`,
`prefillRequestTerms`, `reservedCacheTokens`, `statePerTokenBilled` and
`qsaMaskBytes` are byte-identical across the range. What changed was the
machine's slack. The blindness is older than the fix round; the fix round only
took away the room it had been hiding in.

**The term.** `prefillMemoryNeeded` bills `seq * kv_per_tok` and
`prefillRequestTerms` bills `reserved * statePerTokenBilled` — the whole
prompt's KV and the whole prompt's QSA indexer history, unconditionally. A RAM
hot-cache restore rebinds the entry's MLX handles by REFCOUNT: those rows are
already inside `mlx_get_active_memory`, so the `available` the bill is compared
against is already net of them. Billing them again is a second copy nobody
allocates. At 786,707 tokens on qwen4_exp at 8-bit that is 786,676 x (13,056 +
3,840) = 13.3 GB of the 25.9 GB bill — against 19.0 GB free.

The existing rule ("`active_mem` already holds the resident entry, so the
prefill guard must never ADD it again") had only ever been read as a
prohibition on adding a term. The other half — that the terms already there
describe rows the request will not allocate — was never written down.

**Why it is not a subtraction.** The credit is gated on CAPACITY, not on the
match. Past the restored buffer's capacity `KVCache.nextCapacityReserved`
allocates the whole new capacity beside the old one (`growQuantBuf`
slice_updates the old into the new and both live until the eval) — and the old
one belongs to the hot entry, which eviction PROTECTS, so it does not go away.
A chain extension that outgrows its entry therefore really does have to find a
full fresh buffer, and must be credited nothing. `WarmPrefix.creditedRows` is
all-or-nothing on `reserved <= capacity_tokens` for exactly that reason:

* 786,707 over a 786,676-row entry: `reserved` 787,419 <= capacity 787,456 —
  no grow, full credit, bill 26.6 GB -> 10.8 GB, ADMITTED with no eviction.
* 1,047,556 over the same entry: `reserved` 1,048,268 > 787,456 — a grow, no
  credit, bill unchanged. That request is refused by MARGIN (33.2 GB billed
  against 31.7 GB free; the terms sum to ~27.9 GB before the estimator's 5/4),
  which is a separate question from this one.

**Two threads, one subtraction.** The credit leaves the bill through exactly
ONE door — `PrefillRequestTerms.shared_resident_bytes`, assigned in one place
and subtracted in one place, scan-pinned — because the other failure mode of
this rule is crediting the same prefix twice: once out of `needed` and once as
reclaimable hot cache. `HotPrefixCache.reclaimableBytesFor` already withholds
the matched entry (eviction refuses to take it), and that withholding is what
the connection thread's `pinnedResidentBytes` reads.

The connection thread does not price the credit at all. It has no slot, no
cache and no capacity reading, and a published `EntryDigest` fingerprints only
the first `DIGEST_PREFIX_TOKENS` ids — so it cannot know how much of a matching
entry a prompt actually shares, and guessing is the OVER-credit direction,
which ends in an uncatchable Metal abort rather than a 400. Instead it defers:
when `pinnedResidentBytes(bill) > 0` the guard admits and says so, and the
inference thread — which holds the real cache and `KVCache.residentCapacityTokens()`
— bills the warm request and still refuses by NAME if it does not fit. A gate
that runs BEFORE the estimator that knows better IS the estimator (#126); the
honest form of that rule here is not to decide.

The 1M rung had been naming the defect in its own refusal text: "of which ~0MB
can be reclaimed (the entry a restore would share is not evictable)" — the
guard reporting that the bytes it was billing were bytes it had also proved
nobody was going to hand back.

Tests: the two live-numbered bills (786,707 @ 19,032 MB admits at ~10.8 GB;
1,047,556 over the same entry is credited nothing until its buffer covers the
reservation), the exactly-once scan across both files, and `pinnedResidentBytes`
at the live 24,826/10,294 MB reading.

### Follow-ups the 1M rung is still waiting on

The 1,047,556-token chain extension above is refused by MARGIN, not by this
defect. The terms sum to ~27.9 GB against 31,717 MB free; the estimator's 5/4
takes the bill to 33.2 GB (the log's own number, 32,953 MB). Two levers were
costed and DECLINED for the change that fixed the warm bill, because shaving a
margin on a path where a real OOM is uncatchable is not a trade a warm-bill fix
gets to make:

* **(B) one copy of `state_bytes` while the QSA capacity buffer is being
  reallocated.** `statePerTokenBilled` bills two — the live history plus the
  copy `attachQsaHistoryToLatest` materializes at the end of the prefill. On a
  GROW the old history belongs to the protected hot entry and is already in
  `active_mem`, so the pair that actually coexists may be one new buffer plus
  the attach, not two new ones. Worth ~4 GB at 1M. Needs the attach's lifetimes
  checked against the grow's, not assumed.
* **(C) exempt exactly-known buffer sizes from the 5/4.** The margin exists for
  transients whose peak is estimated; a reserved KV buffer's size is arithmetic,
  not an estimate. Worth ~1.5 GB at 1M — probably not enough on its own.

Note also that the QSA-history bill fix landing beside this one takes 2,304 MiB
off the 1M bill directly (33.2 GB -> ~30.2 GB against 31,717 MB free), so the
rung may admit without either lever. The re-judge is what decides that, not this
arithmetic.

One seam between the two changes, since they touch the same terms — and the
first version of this paragraph got it wrong, which is why the rule is now
spelled at the site. The credit is `credited *| kv_per_tok`: **KV only**. The
QSA history travels with the restore but is PRIVATISED on arrival
(`restoreQsaHistory` -> `seedCapBuf`, a `materializedOwnedCopy` when the
reservation already fits, a slice_update into fresh `mlx_zeros` when it does
not), so the slot allocates its own and not one byte of it is shared. Crediting
one copy of it read as a harmless conservatism; folded with the one-copy
`statePerTokenBilled` this tree now ships, the history term CANCELLED — ~3.0 GB
under-billed at 786k, on a path whose failure mode is an uncatchable Metal
abort rather than a 400 (audit W-2).

The rule that survives is narrower than "the credit is what a restore shares":
**the credit is what the restore hands over WITHOUT allocating**. A buffer
handed to the slot qualifies; a view copied on arrival does not. A change to
how many copies the bill charges must therefore not move the credit at all, and
a scan pins the credit expression against ever naming `qsaHistoryBytesPerToken`
or `statePerTokenBilled` again.

The capacity gate has the same shape and was got wrong the same way (audit
W-1). `KVCache.entries` is allocated at `num_hidden_layers`, but on a GDN trunk
only the ATTENTION layers ever call `update` — 12 of 48 on qwen4_exp — so 36
entries are `initialized == false` forever. The first `residentCapacityTokens`
vetoed on the first of those (`if (!e.initialized) return 0;`), which made the
capacity 0 on every qwen4_exp request and the whole warm credit unreachable on
the one arch it was written for: the `kvLenForBatching` class, a guard reading a
value that is zero forever on a linear-layer trunk. The fold now SKIPS an
uninitialized entry and takes the minimum over those that hold a buffer, 0 when
none does — which is also the safe direction, since a caching layer with no rows
yet can only happen while the whole cache is cold.

### ...and "hands over without allocating" means the CHECKOUT, not the restore (audit B-A3)

The third bite of the same rule, and the one that shows how narrow it really is.
The credit fired on `matched > 0`: any restore, any arch. But `KVCache.restore`
is `mlx_array_set` — a refcount bump — and the story two sections up
("Restore by MOVE") is the proof that a refcount-shared prefix is COPIED by the
turn's first `writeAtOffset`: `is_donatable()` is
`array_desc_.use_count() == 1 && data.use_count() == 1`, the entry's second
reference fails it, and `SliceUpdate::eval_gpu` falls through to `copy_gpu` on
the whole prefix. A shared restore therefore allocates exactly what the COLD
bill charges. Only the CHECKOUT — `HotPrefixCache.checkoutEligible`, which drops
the entry's own handles so the slot is the sole owner — makes the credited rows
rows nobody allocates.

The checkout is narrow: SSD-first (qwen4_exp AND a disk tier) ∧ full-prefix hit
∧ no pending disk record ∧ `MLX_SERVE_RESTORE_MOVE`. So the credit was an
under-bill on every non-qwen4 arch, on every PARTIAL hit (the ordinary
chain-divergence case), while a flush is pending, and on both documented
kill-switch arms — 12,244 MB removed at the pinned 786,707-token scenario, on a
path whose failure mode is an uncatchable Metal abort rather than a 400.

The fix carries the decision rather than re-deriving it: `checkoutIfEligible`
RETURNS whether it took the checkout, `LookupResult.checked_out` carries it out
of the cache, the scheduler's `hot_checked_out` rides beside `hot_matched` into
both inference-thread guards (the fits probe and the refusal that must quote the
number it compared), and `WarmPrefix.will_donate` gates `creditedRows` ahead of
the capacity gate. ONE predicate: a scan pins that the estimator never calls or
spells `ssdFirstEnabled`/`restoreMoveEnabled`/`pending_disk`, because the credit
and the checkout drifting apart IS the defect. The connection thread is
unchanged — it still has no slot, no cache and no capacity reading, so it defers
(`pinnedResidentBytes`) instead of pricing any of this.

Numbers at the pinned scenario are unchanged where the checkout is taken: cold
24,475 MB, credit 786,676 × 13,056 × 5/4 = 12,244 MB, warm 12,231 MB against
19,032 MB free. Everywhere else the bill is the cold one again.
## The 400/503 mapping existed only where nobody was looking (external review of PR #363, item 2, 2026-09-06)

Two named, actionable errors reach a client from the generation path: a memory
503 when the MLX working-set latch (#353) abandons a request mid-forward, and a
400 when the admission estimator refuses a prompt that does not fit even with an
empty hot cache. Both messages name the levers. Both were written four times —
once per surface — on the NON-STREAMING arm only.

The streaming arms did something else entirely. `/v1/chat/completions` wrote
`data: {"error":{"message":"Internal server error: GenerationOutOfMemory",
"type":"server_error"}}`. `/v1/messages` wrote the same string into an
Anthropic `error` event. `/v1/completions` logged the error and wrote
**nothing** — the client got a dropped socket mid-stream. `/v1/responses` had no
`catch` anywhere on its path, streaming or not, so the error propagated to the
route dispatcher and the socket died there too.

Agents stream. So the surface that carried the actionable message was the one
almost nobody reached, and the surface everyone reached carried a Zig error name.

The fix is ONE mapping, `server.mapGenerationError`, asked by both paths:

* `error.GenerationOutOfMemory` / `error.OutOfMemory` → 503, `GEN_OOM_MSG`;
* `error.PrefillDoesNotFit` → 400, `PREFILL_NOFIT_MSG` (a request refused before
  its first forward is the client's request being too large, not an engine
  fault, so `invalid_request_error` on BOTH dialects);
* `error.GenerationFailed` → 500, "generation failed";
* anything else → 500 keeping the error NAME in the message — the streaming
  arms' one virtue over the non-streaming ones, which sent it bare.

`sendGenerationError` is the single arm every surface's `catch` calls, and it
branches on ONE fact: has the SSE response head gone out?

* **Not yet** — the status line is still ours, so the streaming request gets
  byte-for-byte the response its non-streaming twin would have got. A client
  should not have to parse a different error shape because it asked for a
  stream.
* **Already** — the status is spent, so the same `type` and `message` ride the
  surface's terminal `error` event: an OpenAI `data:` frame that also carries
  `choices[0].finish_reason: "error"` (so a client parsing only chunks
  terminates instead of waiting out a stream that will never deliver another
  delta) followed by `[DONE]`; an Anthropic `event: error` (which IS the
  terminator there — no `[DONE]`); a Responses `event: error` with its
  `sequence_number`, then `[DONE]` on HTTP and nothing extra over the WS bridge.

That fact needs one owner, so `Conn.sse_headers_sent` is set in exactly one
place — `sendSseHeaders`, which is now the only site in server.zig that spells
`Content-Type: text/event-stream`. Three of four surfaces setting a flag is the
same defect one layer down.

Two structural consequences fell out. `/v1/responses` gained a wrapper
(`handleResponses` around `handleResponsesInner`) purely to have an error arm at
all, and it owns the SSE sequence counter so the terminal `error` event can
carry the number the spec requires on every Responses event; the WS transport
keeps the error instead, because it has its own terminal frame AND
borrowed-store cleanup to run at the call site (it now maps through the same
function). And the four inner `catch |err| switch (err)` blocks around
`nonStreamingViaScheduler` are GONE: they mapped the same three classes a second
time, which is exactly how the two paths came to disagree. Every class now
propagates to the surface's one arm.

Tests: `mapGenerationError` per class including the unknown arm; the three SSE
body builders carrying the mapped type, code and message plus their surface's
terminator; absent-form scans for every pre-review spelling of a raw error name
in a client-visible frame; a count of the surfaces routing through the sender;
and arm [6] of `tests/test_mlx_error_recovery.sh`, which fires
`MLX_SERVE_MLX_FAULT_STEP` at a STREAMING request and asserts the 503 message
and `finish_reason: "error"` on the wire, with the raw name absent.

## A finish is not a finish when the last forward failed (external review of PR #363, item 3, 2026-09-06)

S20 already knew that `runSingleDecodeTickInner` can reach `finishSlot` — and
so the hot-cache commit — on an EOS the FAILING forward itself produced. Metal
returns ZEROS before it aborts, so a plausible EOS sampled out of buffers it
never wrote is exactly what that failure looks like, and the tick wrapper's
`checkErrorDecode` has not run yet. S20's guard (`if (mlx.errorPending())
return;` inside `commitSlotIfApplicable`) keeps that garbage out of the prefix
cache.

It does nothing for the client. The same EOS ran on through `finishSlot` to
`slot.markFinished("stop")`, the connection thread's `waitNext` returned
`.done`, and the request answered **200 with fabricated text**. The wrapper's
`markError` a moment later hit `if (self.error_code != null or self.finished)
return` and was a no-op. The commit was guarded; the response was not.

`finishSlot` now reads the latch BEFORE anything finalizes the request and
publishes its terminator through `publishSlotTerminator`, the only caller of
`Slot.markFinished` in the file: latched → `markError(name)`, clean →
`markFinished(reason)`. The name is `@errorName`'s, so `errorNameIsMemory`
classifies it exactly as a consumed failure and the client gets the memory 503
through item 2's mapping — on the streaming surfaces too, which is why the two
items ship together. The metrics sink is billed `"error"` rather than the reason
the request never earned.

The read is a PEEK (`mlx.peekErrorName`), never a consume, and that is the
subtle half. The decode-tick wrapper still owns the latch: on a BATCHED group
one forward serves every slot, and the wrapper's `checkErrorDecode` is what
fails all of them. A slot that consumed the latch on its way out would answer
itself correctly and leave its siblings finishing 200 on the same dead buffers —
the original bug, minus one victim.

Tests: `publishSlotTerminator` against a stub slot (clean finishes, latched
never publishes a reason at all, and the published name carries the memory
class); a scan that the latch read precedes the publish, that the publish is the
only `markFinished` caller, that `finishSlot` does not CONSUME, and that S20's
own `errorPending` guard is still on the commit path; and `peekErrorName`'s own
test in mlx.zig — same name as the consuming path, latch still pending
afterwards.
## PR #363 blast-radius ledger — what runs on archs the PR never measured

PR #363 is described as a qwen4_exp (Qwen3.8-Flash-Next) long-context
optimization. It is not one: a systematic sweep of
`git diff a93e2c0..aad0315 -- src/` found behaviour changes reaching llama,
mistral, gemma3/4, muse_glimmer, qwen3_5 (the 27B sidecar-MTP pack),
qwen3_5_moe, qwen3_next, lfm2, nemotron_h, bailing_hybrid and deepseek_v4 —
none of which was benchmarked, and several of which can only lose.

The policy this ledger encodes: **every long-context behaviour change is
qwen4_exp-gated by ONE predicate (`ModelConfig.longCtxGated()`); other archs
are byte-identical to a93e2c0, characterization-pinned. Genuine bug fixes are
NOT gated — they ship for everyone.**

Classes: **A** = bug fix, keep for all archs. **B** = output-preserving, proof
cited. **C** = behaviour change on a non-qwen4 arch, must be gated. **D** =
already unreachable off qwen4 by a verified guard (a comment claiming a guard
is not a guard — every D below was checked by reading the predicate body).

### Class C — gated by this round

| # | site | what changed for other archs | gate | pin |
|---|---|---|---|---|
| 1 | `KVCache.reserve` from `generate.runPrefill` | every arch past a 32k prompt allocated prompt + 8192 + chunk of KV up front instead of the +25% ladder | `generate.reservedPrefillTokens` returns 0 ungated ⇒ `reserve_tokens` stays 0 ⇒ `nextCapacityReserved` IS `nextCapacity` | `reservedPrefillTokens:` tests + the one-call-site scan |
| 2 | `scheduler.batchKvLenOf` / `Slot.batchKvLen` | a93e2c0 fed the pad-waste cap `cache.step`, which is 0 forever on a linear-layer-0 trunk, so the cap was DEAD there; the PR wakes it and skewed groups now split on qwen3_5 dense and qwen3_next | `batchKvLenOf(cache, cfg)` — `cfg.longCtxGated()` picks the kv-length rule, else `cache.step` | `batchKvLen:` characterization over the four hybrid families |
| 3 | `transformer.spanPreservingDropIndex` at five retention sites | a93e2c0 had TWO baselines: drop-oldest at the two prefill-capture sites and at the disk tier, min-span-with-NO-recency-quarter at `mergeCheckpointLists`/`shedCheckpointsToFit`. The PR replaced both with min-span + a dense newest quarter, so which checkpoint a warm turn restores from moved on every hybrid | a per-site `ThinPolicy` chosen by the ONE predicate; the ungated arm names which a93e2c0 policy it reproduces | retention characterization per site |
| 4 | `server.prefillRequestTerms` (all four terms) | reserved KV headroom billed on every arch while its allocator twin was gated; retained SSM checkpoints (~1.9 GB) billed on qwen3_5/qwen3_5_moe/qwen3_next/bailing_hybrid but NOT lfm2/nemotron_h; the warm credit under-bills on every arch | `if (!config.longCtxGated()) return .{};` — `.{}` is the identity for `prefillMemoryNeeded`, so the bill is a93e2c0's 13-argument expression | bill characterization + the sizer/bill consistency test on qwen4 |
| 5 | `server.prefillChunkCap` (ex-`resolvePrefillChunk`) | TWO changes at once: the hot-cache ask was dropped from the serving budget (widens the rung) and the ctx bar was added (can pin chunk 512 for a process) | one gated helper; the ungated arm is a93e2c0's `(ceiling − (active + ask)) / SHARE` | `prefillChunkCap:` characterization over seven archs |

### Class C — found by the sweep (all closed in the second round, below)

| site | reach | note |
|---|---|---|
| `prefix_cache.evictLruToAdmit` + `reclaimableBytes`/`digestsAlloc`/`reclaimableFromDigests` + `scheduler`'s `prefill_admission_fits` pass + `error.PrefillDoesNotFit` | every arch, every request | wholly new (a93e2c0 has none of it). A long prefill now EVICTS hot-cache entries and can be refused by a name that did not exist. The `publishHotCacheResidency` half is class A (it fixes a real connection-thread use-after-free on `hot_prefix_cache`); the CREDIT and the EVICTION are the policy change |
| `prefix_cache.trimLenForBudget` shed simulation | every hybrid, no flag needed | a93e2c0 billed EVERY lower checkpoint; the PR bills only shed survivors, so it systematically retains a LONGER prefix at the same budget |
| `server.prefixCacheMemForLoad` | every arch | three simultaneous input changes to the resolved hot-cache budget: static instead of live ceiling, `ramFirstContextForLoad` instead of `getEffectiveContextLength`, `planHotCache` instead of the direct clamp |
| `server.computeMemoryContext` → `CTX_SIZING_CACHE_RESERVE` | every arch | changes the ADVERTISED `context_length` whenever `--prefix-cache-mem` differs from the 2 GiB default; agent CLIs read that number once per session |
| `server.checkAttentionMemory` new evict / warm-deferral admit arms | every arch, four surfaces | a93e2c0 refused pre-flight with a clean 400; the PR admits and can die later after evicting the whole hot cache |
| `kv_disk_cache.SSM_DISK_MAX_PER_ENTRY` 8 → 16 | every hybrid with `--prefix-cache-disk` | doubles the persisted checkpoint footprint per entry and changes `gcToBudget` pressure |
| `kv_disk_cache` meta.json `"v":4` → `"v":6`, written unconditionally | every arch with disk on | forward-compatible, NOT backward: an a93e2c0 binary rejects v6 and discards the whole persisted tier on downgrade |
| `prefix_cache` trim retry loop | every arch, error path | a failed `trimmedCopy` retries at the next-lower checkpoint instead of declining |
| `ModelConfig.perRequestPrefillChunk` | — | hand-rolls `model_type == "qwen4_exp"` instead of delegating; `model.zig`'s own doc forbids exactly that (a second predicate drifts) |

### Class A — kept for every arch, deliberately ungated

The MLX error latch (`mlx.installErrorHandler`, `checkError` in the prefill
chunk loop and both decode ticks, `commitSlotIfApplicable`'s `errorPending`
guard — a Metal working-set abort used to `exit(-1)`, and a poisoned prefix
used to be committed from an EOS the failing forward produced); the
`Transformer` teardown double free of `aux_state`/`qsa_pooled`; the
`publishHotCacheResidency` scalar (the guard used to dereference
inference-thread state); `hot_cache_digests` freed below the thread join;
`warmQsaEnvCaches`/`warmEnvCaches` on the main thread (first touch of a lazy
`?bool` cache from two threads is a race); `snapshotRowBytes`' ndim guard (an
out-of-bounds shape read on a dense snapshot); the slot error-NAME mapping
(`slotFailure`, `errorIsMemory`) that turns a memory failure into a 503 and a
pre-prefill refusal into a 400 instead of a generic 500; and
`kv_disk_cache.appendSsmOnly`'s byte accounting, which drops the spec
sidecar's size delta and drifts `total_bytes` on every sidecar-only commit
(reachable on any arch with a dflash/MTP snap plus `--prefix-cache-disk`).

### Class C — gated by the second round (open list closed)

| # | site | reach off qwen4 | gate | test |
|---|---|---|---|---|
| A-1 | `ModelConfig.perRequestPrefillChunk` (`src/model.zig:790`) | hand-rolled `"qwen4_exp"`; the ONLY gate on the admission re-ask, the tail-merge bound and the per-chunk adapter | delegates to `longCtxGated()` | `no policy predicate hand-rolls the qwen4_exp literal` — a PRODUCTION-WINDOW scan of the literal, since counting predicate CALLS cannot see an inline `std.mem.eql` |
| 6 | `kv_disk_cache.SSM_DISK_MAX_PER_ENTRY` 8 → 16 | every hybrid with `--prefix-cache-disk`: double the persisted checkpoint footprint per entry, and different `gcToBudget` pressure for the whole tier | `DiskTier.ssm_max_per_entry`, mirrored beside `cp_thin` at the ONE wiring site; `SSM_DISK_MAX_PER_ENTRY_LEGACY` (8) is the default | `the per-entry checkpoint cap is gated; a legacy tier keeps a93e2c0's 8` |
| 7 | `meta.json` `"v":6` written unconditionally | every arch with disk on: a93e2c0's reader accepts 2/3/4 only, so a downgrade discards the WHOLE persisted tier | `metaVersionFor` stamps the LOWEST version that describes the entry (v6 = inherited chunks, v5 = the MTP head's QSA half, else v4). Not an arch gate — a property of the entry | `the manifest stamps the LOWEST version that describes the entry` |
| 8 | `server.prefixCacheMemForLoad` | every arch, THREE inputs at once: static ceiling, `ramFirstContextForLoad`, `planHotCache` + `clampReserveWidth` | ungated arm is a93e2c0's verbatim, placed ABOVE both the SSD arm and the plan | `the ungated budget arm is a93e2c0's arithmetic` |
| 9 | `server.computeMemoryContext` → `CTX_SIZING_CACHE_RESERVE` | every arch: a different ADVERTISED `context_length` at any `--prefix-cache-mem` but the 2 GiB default, read once per session by every agent CLI | `ctxSizingCacheReserve(config)`; both load-time wrappers read it too, so the billed session and the advertised session stay one number | `the advertised context is a93e2c0's on every other arch` |
| 10 | evict-to-admit: the two credits, the two new `checkAttentionMemory` admit arms, `evictLruToAdmit`, `error.PrefillDoesNotFit` | every arch, every request | the two CREDITS zeroed in `prefillAdmissionBill` (one lever: `fitsAfterEviction()` becomes `fits()`, `pinnedResidentBytes` becomes 0, so both arms go dead together) + the same predicate on the scheduler's pass | `the evict-to-admit credits are qwen4_exp-only` |
| 11 | `prefix_cache` trim retry loop | every arch, error path: retries an allocation immediately on a path whose failure is memory pressure | `self.cp_thin == .min_span` → decline, a93e2c0's behaviour; the error is still LATCHED on both arms | the retry test's gate/log/latch ordering assertions |

### Class A — the second round's additions

`kv_disk_cache.appendSsmOnly` billed its `total_bytes` delta from the
checkpoint files alone while also writing `spec.safetensors`, and assigned
`e.spec_bytes` before taking the delta — so the sidecar was written and never
billed, `gcToBudget` priced the tier low, and the footprint drifted past
`--prefix-cache-disk`. Reachable on any arch with a dflash/MTP snap plus a disk
tier; fixed for everyone (`an ssm/spec-only append bills the SPEC sidecar's byte
delta`, red at 80,739 vs 118,668).

**This fix IS live on qwen4_exp** — correcting a premise this ledger and the
round-2 brief both carried. The claim was that qwen4 never writes a spec
sidecar because `MtpCacheRef.kv()` returns null for the in-checkpoint head.
That was true at a93e2c0 and is **false on this tree**: `generate.kv()`'s
`.qwen4` arm returns `&t.qwen4_mtp.?.cache` whenever `mtpHeadPersistEnabled()`,
and `mtpHeadPersistFromEnv(null)` returns **true** — head persistence is
default-ON (`MLX_SERVE_MTP_HEAD_PERSIST=0` turns it off). `metaVersionFor`
returns 5 for exactly that case. So on qwen4_exp with `--mtp` +
`--prefix-cache-disk` a v5 sidecar IS written, `specWorkPending` CAN be true —
notably through its own v5-upgrade arm, which fires on every entry of a tier
carried over from an older build — and the delta really moves `e.bytes` and
`tier.total_bytes`.

What the delta cannot reach is a DECISION: its only consumer is `gcToBudget`,
which is a no-op unless `total_bytes > max_bytes`. The judge boots at
`--prefix-cache-disk 100GB` against tiers of ~20-28 GB, so the budget never
binds and no eviction moves — which is why the root_all5 judge table still
stands for root_all6. **That is a run-configuration argument, not a property:**
under a tight `--prefix-cache-disk` this fix does change qwen4 eviction. Nobody
writing the PR body should repeat the `kv()` → null claim.

`publishResolvedPrefixCacheMem` stays ungated **on purpose**: a93e2c0's ANE
gate reserved the RAW `--prefix-cache-mem` while the cache could only ever hold
the clamp, so three places disagreed about how many bytes the cache holds.
Publishing a number the cache is actually capped at cannot over-admit.

Its reach is **wider than "`--ane-prefill` boots"**, which this ledger claimed
and the code does not support. `resolvedPrefixCacheMem()` has three ungated
production readers: the `serve()` boot banner (`Hot prefix cache: ENABLED
(mem-cap=…)` now prints the RESOLVED cap where a93e2c0 printed the raw ask —
cosmetic, operator-visible, every arch); `aneGateHeadroom`; and — until this
round — `pinPrefillChunk`. The third was the one that mattered and is now
gated: off `longCtxGated` the pin reads `legacyPrefixCacheAsk()`, because in a
multi-model registry the SECOND model to load was sizing its prefill width
against the FIRST model's clamped budget, an input a93e2c0 never fed it.
Single-model boot was identical either way, which is why it went unnoticed.

### Class C — the completeness sweep's seven, gated in round 3

The auditor's line-by-line sweep of `a93e2c0..00b7a8e -- src/` found seven
changed hunks the ledger had no row for. Five change behaviour off qwen4_exp
and are gated; two are log-only and are recorded as such.

| # | site | what changed for other archs | gate |
|---|---|---|---|
| 12 | `pinPrefillChunk` → `billedPrefillChunk` override precedence, and `chooseRequestPrefillChunk`'s short-circuit | a93e2c0 pinned the machine RUNG and let `generate.effectivePrefillChunk` apply an explicit `--prefill-chunk` / `MLX_SERVE_PREFILL_CHUNK` at forward time. The PR moved that precedence into the PIN — the audit-S8 fix — and the pin is read by the admission bill, by the ungated `clampedPrefixCacheMem` reserve and by `computeMemoryContext`'s **advertised** context, so three numbers moved at once on every arch that sets the flag | `pin_override` is `explicitPrefillChunk()` only when gated, else 0; the chooser asks the arch BEFORE the override |
| 13 | `pinPrefillChunk` reading `resolvedPrefixCacheMem()` | in a multi-model registry the second model to load pinned its width against the first model's CLAMPED budget — an input a93e2c0 never fed it. Single-model boot is identical, which is why it went unnoticed | ungated reads `legacyPrefixCacheAsk()`, the raw `--prefix-cache-mem` |
| 14 | the memory-refusal 400 body | gate 4 zeroes both credits off the gate, and the message formats them: every non-qwen4 refusal read "the hot prefix cache holds ~0MB more, all of which can be reclaimed (~0MB — the shortfall is elsewhere)" on a box with a multi-GB resident cache. Client-visible, and false | `memoryRefusalMessage` — one formatter, two arms, discriminated on the ARCH GATE (`config.longCtxGated()`, passed in); the ungated arm is a93e2c0's exact sentence, pinned byte for byte, and the gated arm keeps qwen4's bytes verbatim |
| 15 | `scheduler`'s batched-group sort | the stable insertion sort replaced `std.sort.pdq`, which is UNSTABLE. On equal keys the two hand `batchedKvKeepCount` different slots in the tail, and the tail falls to SERIAL decode — a different kernel. Off qwen4 every key is `cache.step`, 0 forever on a hybrid, so EVERY key ties and the sort decides the whole ordering | `gate_batch_kv_len`, the same predicate as the kv-length rule, read once per group so a group cannot be ordered by one key and capped by the other |
| 16 | `round_cost.BUCKET_NAMES[5]` | the legacy grid's top bucket is unbounded `32k+`; the array spells the long grid's `32-64k`. Every `[spec-stats]` / `[mtp-trace]` / adaptive-switch line on a sidecar-MTP pack labelled a 374k request "32-64k" — the number right, the label a lie | `bucketName(layout, b)`; labels only, no cell/edge/plan moves |

**Row 14 was nearly keyed on the wrong thing.** The first shape discriminated
on `bill.evictable == 0` — "name the cache only when there is a cache", which
reads better on the merits and is wrong here twice over:

* it CHANGES qwen4's bytes. That arch reaches the refusal arm with an empty hot
  cache (a large first request, before anything is resident), where it used to
  render the long sentence with two zeroes; a byte-count discriminator would
  have silently switched it to a93e2c0's short one. This round's mandate is
  zero qwen4-path changes, so the gate decides and qwen4 keeps the zeroes.
* it splits on ONE credit while the guard's arms read BOTH. `evictable == 0`
  with `reclaimable > 0` is not a state the cache produces (`reclaimable` is
  residency minus the largest entry, so it is bounded by `evictable`), but
  nothing in the formatter said so, and `pinnedResidentBytes` couples the two.
  A message keyed on half of a coupled pair is a latent disagreement with the
  arm that refused. The gate keys on neither and cannot drift from either.

The merits question — whether a GATED bill on an empty cache should name the
cache at all — is real and deferred, to a change that can be measured on its
own rather than folded into an arch gate.

**Known limit of the ungated arm of row 12,** inherited from a93e2c0 and stated
rather than silently fixed: with `--prefill-chunk` set, a non-qwen4 boot bills
and advertises against the machine rung while the forward runs the flag's
width. That inconsistency *is* a93e2c0's behaviour; closing it is a change that
owes the archs it touches a measurement.

### Log-only, recorded so the sweep is closed

`[spec-stats]`'s new fields (`width_trials=`, `table=`, the serial-row and
EV-plan terms) and the `[admission]` verdict line are **emitted, never read**.
`logAdmissionDecision` derives its verdict from the same `AdmissionBill` the
guard acts on, so off the gate it prints `reclaimable=0 MB` and the arm it
names is a93e2c0's reject — a true statement about a bill whose credits the
gate dropped. No decision keys on either line.

### Two ungated costs gate 4 leaves behind

Recorded because they are real and unmeasured, not because they are wrong:

1. **The 400's text** — closed by row 14 above.
2. **Publishing machinery every ungated arch pays for and never reads.**
   `publishHotCacheResidency` calls `publishHotCacheDigests`, which allocates
   and takes `digest_mu` on every commit, eviction, invalidation and model
   switch; and `prefillAdmissionBill` calls `reclaimableHotCacheBytesFor` **per
   request on the connection thread**, hashing the prompt under the same mutex
   — then discards the result at the ungated return. a93e2c0 had none of it. No
   correctness impact: the two scalars are the class-A use-after-free fix and
   must stay. The DIGEST half arguably belongs inside the gate; left as-is this
   round because moving it is a threading change, not a text change, and it
   owes a measurement of its own.

**A coupling worth stating:** gate 4's ungated identity rests on gate 1 holding.
`needed` reduces to a93e2c0's expression only because the reservation is zeroed
(`generate.reservedPrefillTokens`) and `checkAttentionMemory` passes a cold
`.{}` warm prefix. Break gate 1 and gate 4's arm stops being a93e2c0's.

### Sweep extension: aad0315 → f3f0a25

| commit | site | class | note |
|---|---|---|---|
| `c7f0657` | ONE error mapping for streaming and non-streaming (`mapGenerationError` / `sendGenerationError`) | **A** | a streaming fault answered differently from the same non-streaming fault. `mapGenerationError` is the one classifier and `sendGenerationError` the one emitter, so a surface cannot answer twice. Pinned by `a streaming fault answers with the SAME mapped error a non-streaming one does` |
| `035da33` | a latched MLX failure ends the request, never a 200 (`publishSlotTerminator`) | **A** | `publishSlotTerminator` is the ONE publisher and it reads the latch BEFORE it publishes, so no path can end a latched request with a success terminator. `peekErrorName names the class WITHOUT consuming the latch`, `a finish over a latched MLX failure ends the request as an ERROR, never a 200`, `the slot terminator reads the MLX latch BEFORE it publishes, and is the only publisher` |
| `db9fa58` | `DiskTier.deinit` lifts the writer pause before `w.drain()` | **A** | a paused background writer deadlocked `zig build test` (the B-A1 hang this branch reproduced on aad0315). `DiskTier: deinit RETURNS with the writer paused, and lands what was queued`, `every test that PAUSES the background writer owes a deferred unpause`, `kv_disk_writer: a PAUSED writer deinits without blocking` |
| `8959c78` | the post-eviction re-ask is the width that runs, in both directions | **C, already gated** | `postEvictionPrefillChunk` runs only where there IS a per-request width, i.e. behind `perRequestPrefillChunk` — which A-1 now routes through `longCtxGated()`. `the post-eviction re-ask never exceeds what live memory affords, and never runs on an arch that has no per-request width` |
| `124f19d` | the warm KV credit fires only where the restore CHECKS OUT its entry (B-A3) | **C, subsumed** | `WarmPrefix.will_donate` gates `creditedRows`; off qwen4_exp the credit is already zero because `prefillRequestTerms` returns `.{}` (gate 4). The two compose — the fold's fix tightens the gated arm, this branch removes the ungated one |
| `e59e259` | one round-cost `Layout` resolver (`layoutFor`, `migrateLegacy`) | **D** | `layoutFor` is the single resolver and `Layout.legacy` keeps a sidecar pack booting warm off the `rc1` file 26.9.1 wrote. `migrateLegacy` lifts a six-bucket table onto the long grid behind `if (layout != .long) return null;`, so a sidecar pack's `rc1` file is read in place and never rewritten. Verified gated, not assumed: `layoutFor is THE round-cost layout resolver`, `every round-cost layout assignment routes through the ONE resolver` |

### The round-cost / planner cluster — class D, verified

The adaptive MTP width/depth planner, the EV cost profiles and the persisted
round-cost table do not reach a non-qwen4 arch's behaviour: the adaptive-serial
block is behind `mtpAdaptiveArchEligible`, `MtpCostProfile` falls back to
`generic`/cap-6 for any unmeasured fingerprint, and the table's on-disk layout
is resolved by the ONE `layoutFor` with `.legacy` preserving the `rc1` file a
sidecar pack (the qwen3_5 27B) already has. The 27B's SIDECAR MTP head
therefore boots warm off its existing table rather than re-measuring — which is
the property that would otherwise have been a silent first-request regression
on that pack.

### The rule this leaves

A "qwen4_exp long-context" change that touches a shared function is a
cross-arch change until a predicate says otherwise. The predicate is
`ModelConfig.longCtxGated()`, it has ONE body, and a site that cannot see a
ModelConfig mirrors it ONCE into a field at wiring time
(`HotPrefixCache.cp_thin`, the `qsa_history_required` pattern) and
is scan-pinned to it.
