# Server, HTTP and streaming lifecycle: gotchas

One story per gotcha: the defect, the cause, the fix, the guard. The rules live in the root `AGENTS.md` (`## Rules`);
what is still open lives in `docs/known-issues.md`. An env lever a story names that `src/env.zig` no longer lists was
retired once its arm was decided: the measured default stays, the losing opt-in was deleted.

## Requests and media

### Literal HTML became a RADIO video attachment
Defect: a `<video>` in message text became a media placeholder (a 400 on text-only requests, an extra atom beside a
real clip). Cause: the rendered prompt was rewritten `<video>` → `<img>`. Fix: the sentinel is emitted only when
serializing an attached video part (`ChatConfig.video_placeholder`). Guard: `RADIO video placeholders` tests, the
media-atom corpus, `tests/test_nemotron_bolt.py`.

### Five request edges answered 200 with the wrong thing
Defect: `stop: ""` matched at 0 and returned an empty reply; a `json_schema` without an object schema fell open to
prompt-only JSON; an undecodable `image_url` (a remote URL, never fetched; bad base64) was dropped and the model said
it saw no image; an empty `/v1/embeddings` input was a 500; Ollama's load handshake (`/api/generate` with no prompt)
was a 400. Fix: empty stops skipped at every parse site; a schema-less `json_schema` (and `output_config.format`,
`text.format`) is a named 400; an active-turn image that did not append is a named 400 (history images are never
decoded, so never refused); empty embedding entries 400; the handshake answers `done_reason: load` without a forward.
Guard: `tests/test_api_edges.sh`.

### A bare-string `image_url` vanished from the prompt
Defect: `{"type":"image_url","image_url":"data:..."}` (the Responses shape) answered 200 about a picture the model
never saw. Cause: the parser skipped any part whose `image_url` was not an object, and every other malformed shape
too. Fix: `imageUrlOfPart` reads the object and string forms; any other shape is the named decode 400, on chat and
embeddings. Guard: its unit test + `tests/test_api_edges.sh`.

### Only the latest turn's images reached the model
Defect: pi reading pages 1-3 saw page 3 only (Flash invented a SKU, the 27B OCR'd the screenshots with tesseract);
images in an Anthropic `tool_result` never reached the model. Cause: only the active turn's media was encoded, spliced
after a searched user marker; the Anthropic parser kept only a `tool_result`'s text. Fix: the vLLM/llama.cpp design:
media serializes as content parts, the model's template renders one placeholder per item where the message sits,
`expandMediaPlaceholders` turns placeholder k into item k's run (a count or kind mismatch is a named 400); every item
is encoded in prompt order and cached by pixel hash (`EmbeddingCache`, never evicting what the current request used);
the prefix cache keys each item (`MediaSpan`). History images are still decoded each turn to key the cache. Guard:
`media renders one template placeholder per item`, `expandMediaPlaceholders` tests,
`parseInput keeps a function_call_output input_image` (Codex's `view_image`, #974), `tests/test_multi_image_history.sh`.

### A content array's text parts were last-wins (opencode plan mode, #195)
Defect: in opencode plan mode the first message of every session was answered "What would you like to accomplish?".
Cause: opencode appends its plan-mode reminder as a second text part on the last user message, and the chat parser
overwrote each text part with the next; `/v1/messages` took only the first `system` block and the first `tool_result`
text block. Fix: one collector, `joinedTextParts`: every text part joined in order with `\n`, at all three sites.
Guard: unit tests on the exact opencode two-part shape.

### The multipart `model` field was invisible to model resolution
Defect: every `/v1/images/edits` request ran against the default model: a 400 "does not support this media modality"
with a chat default, a 503 "No default model" headless. Cause: resolution scanned the body for a JSON `"model"` key
before dispatch; tests passed because their default WAS the model under test. Fix: `parseModelFromRequest(body,
content_type)` is the one reader for every body shape, feeding dispatch and the LAN share gate. Guard: the headless
section of `tests/test_mageflow_edit.sh`.

### `name=` matched inside `filename=`
Defect: an upload whose `Content-Disposition` put `filename` before `name` 400'd "missing image". Cause: `paramValue`
used a substring search. Fix: a parameter match sits at a boundary (start, or after `;` and whitespace). Guard:
`paramValue keys on the PARAMETER, not a substring of a longer one` (both orders).

### An unbounded debug body log met its first binary body
Defect: one image upload at debug level wrote megabytes of PNG (NULs included) into the log and could rotate it away.
Fix: `bodyIsText` logs text whole and binary as a bounded printable preview with the true byte count. Guard: `debug
body log bounds BINARY but never truncates text`.

### `messages.deinit` freed the Message array and nothing it pointed at
Defect: every chat or `/v1/messages` request carrying an image leaked its decoded pixel buffer (40 requests x 3 MB =
121 MB retained). Cause: buffers handed to `Message` had no owner; the Anthropic path's `errdefer` covered only the
error path. Fix: `RequestMedia`, one owner per request, by provenance; `Message` borrows; slots are indices (an
`ArrayList` header may move). Guard: lint pin `RequestMedia.init(allocator)` + `RequestMedia frees every decoded
buffer it was handed`.

### A text-gen request at a non-text model killed the server
Defect: `mlx-serve run <flux dir>` SIGSEGV'd: the media stub tokenizer gave 0 tokens and prefill dereffed a null
transformer; any client naming a media model on a chat surface could do it. Fix: `textGenRejectReason` at every
text-gen surface, twice: a pre-load peek (`isTextGenRoute`, discovery `arch_hint`) and the post-load check. A new
surface calls the same guard; a new modality is covered through its engine slot + `modalityFromType`. Guard: unit
tests + `tests/test_unified_gen.sh` 4b.

### A client-supplied path handed to mlx killed the server (LoRA)
Defect: `{"lora_path":"/tmp/nope.safetensors"}` ended the process (mlx-c errors were fatal then). Fix: `lora.loadFile`
opens and stats a REGULAR file on our side first (`error.BadLoraPath` → 400); a bad entry anywhere in `lora_paths`
400s before any adapter attaches. Guard: the two `loadFile` tests, `tests/test_mageflow_edit.sh`.

### A host path a remote client may name is a file-existence oracle (#540)
Defect: a missing LoRA file 400'd differently from an existing non-LoRA, so a LAN or keyed client could probe host
paths. Fix: `parseLoraFields(…, local)` refuses LoRA fields from non-loopback peers with a named 403 before touching a
path, on every handler including the multipart edit. Guard: `parseLoraFields: a LoRA path is accepted only from this
machine`.

### The edit form dropped LoRA fields (#268)
Defect: adapters attached through `/v1/images/edits` were ignored with a 200. Fix: `openaiEditFormToJson` forwards
`lora_paths`/`lora_scales`/`lora_path`. Guard: the lora case in its test.

### An error message that quotes a field value is not a JSON string
Defect: media 400s came back as `{"message":"… (mode:"edit") …"}`, unparseable; a message longer than 256 bytes sent
no body at all. Fix at the sink: `gen_sse.jsonEscapeMessage` escapes and truncates on a UTF-8 boundary into a 640-byte
buffer, for both senders. Guard: a test that parses the live message with `std.json`.

### `/detokenize` emitted raw control bytes
Defect: tokens below 0x20 made the response unparseable. Cause: a hand-rolled five-character escape table. Fix:
`detokenizeResponseJson` uses `appendJsonString`. Grep for hand-rolled escape tables when a client reports invalid
JSON. Guard: a control-byte test that parses the result.

### Context-overflow 400s name both counts
Defect: clients could only say "too long". Fix: `contextOverflowMessage`: "Prompt exceeds maximum context length: N
tokens requested, M available" on all four text surfaces, the old sentence kept as the prefix; a format failure falls
back to the bare sentence. The app renders it as a card.

### Bytes past `Content-Length` corrupted the heap
Defect: a body longer than its `Content-Length` (or a length near 2^64) answered the wrong body and crashed later on
an unrelated request. Cause: head gulps were copied whole into a `head + Content-Length` buffer, and the sum wrapped.
Fix: `readRequest` keeps only the declared bytes and adds saturating (a wrapping length meets the 413 cap);
`Scheduler.submit` refuses an empty prompt by name. Guard: `readRequest: bytes past Content-Length are dropped, a
wrapping Content-Length is a 413`.

### A deeply nested tool schema overflowed a connection thread's stack
Defect: tool `parameters` nested 50,000 deep killed the server. Cause: `Stringify` and the schema walkers recurse per
level. Fix: every non-multipart body and Responses WS frame is scanned for its depth (`jsonNestingDepth`) and refused
past `MAX_JSON_DEPTH` (4096). Guard: `jsonNestingDepth: brackets inside strings never count`.

## Streams and sockets

### Liveness is a property of the socket, not the generator
Defect: a pi session lost two 5-minute `write` calls to undici's 300 s `bodyTimeout` (`TypeError: terminated`). Cause:
the keepalive fired only while waiting for the first token; a buffered tool call wrote nothing for minutes. Fix:
`Conn.heartbeat` is stamped by every socket write; every token loop calls `beatStreamKeepalive` once per iteration,
sending a comment/ping only after 5 s of byte silence; WS transports no-op the senders. `StreamHeartbeat` guards the
client the way `StallClock` guards the server. Guard: `tests/test_stream_keepalive.sh` + `StreamHeartbeat` tests.

### A reader that lags the end of a stream got ECONNRESET instead of EOF
Defect: a slow SSE reader read through `[DONE]`, then failed with `Connection reset by peer`. Cause: the body is
close-delimited; closing the fd orphaned the socket in FIN_WAIT_2 and macOS RSTs it after `fin_timeout` (60 s). Fix:
for a close-delimited body `Conn.close` sends FIN and waits (≤ 5 min) for the peer; a `Content-Length` response sets
`Conn.length_framed` and closes at once. Guard: the two `Conn.close` tests, `tests/test_responses_streaming.sh` [G].

### Shutdown raced `Scheduler.deinit`
Defect: SIGSEGV in `Scheduler.complete` on shutdown or a model switch with a stream in flight. Cause: `deinit` freed
the slot queues while a connection thread was inside `complete`, and `cancel()` did not wake blocked readers. Fix:
`active_conn_threads` counts live threads; shutdown runs `cancelAllInFlight` and waits for zero before `deinit`;
`waitNext` returns `.done` on cancel. Guard: `tests/test_shutdown_midstream.sh` (a smoke guard; the race is
timing-sensitive).

### Dropped thread handles retained one stack mapping per request
Defect: in long Claude Code sessions the footprint climbed toward 10 GB while MLX memory stayed flat (155,786 stack
regions, 2.4 GB). Cause: the accept loop never joined or detached threads. Fix: `conn_thread.detach()` after spawn.
Guard: lint pin `conn_thread.detach();`.

### A cancellation signal that existed on one response shape
Defect: a client that hung up on a non-streamed media job left the GPU rendering a video nobody would receive. Cause:
`StreamCtx` cancelled only when an SSE write failed. Fix: both shapes probe `Conn.peerClosed` (also an earlier signal
than a failed write). Only backends that poll `Progress.cancelled` can stop. Guard: `tests/test_image_cancel.sh`. Side
lesson: `host/media_jobs/sse.zig` was never in `src/tests.zig`; check a new test against a bogus filter too.

### The streaming think gate was O(buffer) per token
Defect: four full-buffer scans per token over an unclosed thought (48 µs/token at 113 KB, growing). Fix: `ThinkScan`
keeps a cursor: every marker holds exactly one `<`, so only the last `<` can straddle the boundary; the close tag is
never latched once a `tool_call` substring is seen (`thinkCloseIsToolCallPayload` looks forward). Guard:
prefix-by-prefix equivalence against the fresh gate, the flat-cost invariant, lint pins on the two stream files.

### Stream and non-stream must be the same bytes
Defect: LFM2.5 streamed `'\n\nOne, two…'` where non-stream gave `'One, two…'`; the logprobs suite caught it after
skipping for months on a moved model path. Cause: the non-stream split trims leading whitespace; streams flushed
verbatim. Fix: `streamContentLead` withholds a chunk only while it is entirely whitespace and nothing visible has been
sent (never a cut inside a token), retiring its pending logprob. Guard: `tests/test_logprobs.sh` stream-vs-non-stream
sections.

### A thinking-enabled stream answered into an empty content field (LFM2-VL)
Defect: the whole answer arrived as reasoning and `content` was empty; non-stream was fine. Cause: streams seeded
`in_think_block = enable_thinking or prompt_opened_think`; LFM2-VL's prompt opens nothing and the model answers
directly. Fix: seed from `prompt_opened_think` alone. Guard: lint pins on both stream files; the stream-vs-non-stream
byte invariant.

### The usage chunk restated the ending
Defect: per-event clients rendered the ending twice (two "Stopped" banners in the app). Cause: the `include_usage`
chunk re-sent `finish_reason` + `finish_details`; completions put usage on the finish chunk. Fix: `sendSSEUsageChunk`:
`"choices":[]`, `usage` + `timings` only. Guard: `tests/test_loop_stop_signal.sh` [2], [4].

## Logprobs

### Logprobs were misleading, not missing
Defect: "-0.004, so 99.6% confident" was wrong three ways: values were post-temperature (saturating to 0.0 at temp 0),
ids were recovered by float equality from an unordered top-k, and each entry described the NEXT token. Fix: raw logits
before temperature/penalties/top-k; ids travel with values (`argpartition`, lower id wins ties); a one-token delay
through `Generator.pending_logprob`, with `firstTokenLogprobs` seeding t1 from the prefill. Bar: at temp 0, rank 1 ==
the emitted token. Guard: tie-saturation and three-temperature tests in `generate`, `tests/test_logprobs.sh`.

### `/v1/completions` ignored `logprobs`
Defect: the handler hardcoded 0 and still emitted the key. Fix: the field is an integer; the response is four parallel
arrays (`tokens`, `token_logprobs`, `top_logprobs`, `text_offset`). Its `top_logprobs` is a map keyed by text, so two
U+FFFD fragments collide (OpenAI's shape): measure on the chat surface.

### Streaming chat accepted `logprobs`, paid for them, and dropped them
Defect: streamed chat with logprobs disabled every spec path and emitted none; llmprobe tested non-streaming only and
scored 100%. Fix: `logprobs` rides as a sibling of `delta` (`ChunkExtras`, default empty so other streams are
byte-unchanged); `StreamLogprobs` drains against a high-water mark, each entry exactly once; token and entry publish
in one critical section (`pushTokenWithLogprob`, `copyLogprobsFrom`). Guard: `tests/test_logprobs.sh` [4] (token- and
value-equal to non-stream), lint pin on the content emitter.

### `logprobs.content` described the thought, not the answer
Defect: on every thinking model `logprobs.content[0]` was the reasoning's first token. Fix: non-stream maps the
content slice to a token range (`contentTokenRange`); streams drain only on content emitters, `skipToContent` before
the first content chunk, and `dropPending` when a block closes with no content. Read the wire before deciding which
branch to patch. Guard: two-sided emitter checks + `tests/test_logprobs.sh`.

### A logprobs token string is a BPE fragment
Defect: a top-5 candidate holding half an emoji made the whole response invalid UTF-8. Fix: `utf8Sanitize` emits one
U+FFFD per invalid sequence (maximal subpart); `bytes` keeps the exact bytes. Do not widen it to the other escapers.
Guard: the logprobs script checks every body it produces.

## Loading, registry and lifecycle

### Idle eviction: a reload that leaked, readers it could free underneath, a poll that undid it
Defect: `--idle-evict-secs` swept nothing; once it did, each cycle leaked ~196 MB, and the app's 3 s `/props` poll
cold-loaded the model straight back. Cause: install sites overwrote retained CPU state without freeing it;
`/api/tags`, `/api/show` and the pre-load gate read that state with no refcount; `/props` reached `ensureLoaded`. Fix:
`releaseRetainedCpuState` at each install, run while `.loading`; readers take the mutex and skip `.loading`; status
routes answer from counters. Guard: `tests/test_idle_evict.sh`.

### Two models under the registry cap still killed the server
Defect: loading a 128 GB pack beside an idle 70 GB one, under the 230 GB GPU cap, ran macOS out of memory and jetsam
killed the server (no crash report). Fix: before reading weights the inference thread unloads idle models, oldest
first, until the load fits what the OS grants less `osReserveBytes` (`makeRoomInRam`), past `--skip-mem-preflight`
too. Tell: `OS_REASON_JETSAM` in `log show`. Guard: the `RAM:` tests in `host/registry/registry.zig`.

### Auto-context is pinned at load, with headroom
Defect: the advertised context drifted per request with system load (agent CLIs read it once); an omitted `max_tokens`
silently capped at 4096 under auto-context. Fix: `pinAutoContext` at load; the 85% margin applies to the memory
ceiling only, never the model's own max; `clampMaxTokens` and `omittedMaxTokensDefault` take the effective context.
Never gate on the raw `--ctx-size`; ask `getEffectiveContextLength`. Client side the app writes the advertised window
verbatim into agent configs, including Claude Code's `CLAUDE_CODE_MAX_CONTEXT_TOKENS`. Guard: `autoContextFor` tests,
`AgentBudgetTests`.

### The GPU ceiling must see external memory pressure (#64)
Defect: Metal OOM with docker holding tens of GB; the backtrace's libllama frames were a red herring (its terminate
handler prints the trace; the throw is MLX's, on a completion thread, uncatchable). Fix: budget against
`currentGpuMemoryCeiling` = min(static working-set max, MLX active + cache + what the OS grants); reserve the full
hot-cache budget up front. Prevention is the only lever. Guard: `physicalMemoryCeiling` / `safeContextForBudget`
tests.

### Request timeout is a stall timeout; truncation keeps "length" through the tool parse
Defect: an agent's 5-minute `writeFile` rounds were cut by a wall-clock `--timeout`, salvaged as `{}`, and reported
`tool_calls`, so the model re-emitted the same call. Fix: `--timeout` counts seconds without a new token
(`StallClock`); `toolCallFinishReason` keeps `length`; the app's budget guidance appears only when the budget is
tight. Guard: `toolCallFinishReason` + `StallClock` tests.

### A split GGUF on ds4 killed the process with nothing in the log (#586)
Defect: shard 1 of a split carries no tensors; ds4 printed to stderr and called `exit(1)`. Fix: `split.count` is read
and a split goes to llama.cpp (`preferredEngine`); `Ds4Engine.open` refuses one by name. Guard: `preferredEngine: a
split GGUF goes to llama.cpp`.

### A DeepSeek-V4 GGUF refused to open with default flags
Defect: every antirez DSV4 GGUF failed without `--no-mtp`. Cause: in-checkpoint MTP armed for any arch declaring
`nextn_predict_layers`; DSV4 ships its head apart. Fix: `ds4_embedded_mtp_archs`. Guard: the embedded-MTP test,
`tests/test_ds4_serve.sh`.

### A ready model advertised less than its unloaded stub
Defect: a loaded ds4/llama GGUF rendered `capabilities:[]`, vanishing from capability-driven clients. Cause: chat
capability was gated on a template; engine GGUFs may ship none. Fix: `readyHasChat` (template presence is not the gate
for engine models), used by both emitters; the app treats an empty `lan_peer` list as chat. Guard: `readyHasChat`
test, `LanModelCapabilityTests`.

### Ownership decided by content leaked the empty case
Defect: every empty-content reply leaked its 2-byte escape. Cause: the defer compared the result to the OOM fallback
literal `""`. Fix: `jsonEscapeOrEmpty` returns `{slice, owned}`: ownership by provenance. Guard: the owned-empty test
under `testing.allocator`.

### A serve path that hand-rolls its config eats CLI flags (headless PLD)
Defect: `--pld`, `--pld-draft-len`, `--pld-key-len` did nothing headless, the mode the app always launches; a fix to
one left the other two. Cause: `runHeadlessServe` wrote them as sibling literals. Fix: one value, `PldDefaults`, built
once in `main()`. Grep for the config literal, not the flag. Guard: `tests/test_headless_spec_flags.sh` (boots over an
empty `--model-dir`).

### `--model=<path>` was silently dropped
Defect: `--model=X` did nothing; the server went headless and auto-picked an unrelated default. Cause: the arg loop
matched exact names and had no else branch. Fix: `classifyUnparsedArg` refuses the `=` form, a flag in the last slot
and anything unknown, by name. Before hard-failing on unknown args, diff what callers send against what the loop
matches. Guard: `tests/test_headless_spec_flags.sh`.

### Endpoint existence never depends on model state
Defect: headless, unknown paths answered 503 "No default model", and llmprobe scored every surface absent; with a
model named in the body, `POST /v1/load` (typo) cold-loaded 121 GB before 404ing; `GET /` 503'd headless. Fix: one
unconditional `routeExists` check before resolution (`ROUTE_PATHS`); status routes never reach `ensureLoaded`; the
console renders with no model and fetches model data client-side. Guard: `tests/test_route_404_no_load.sh`,
`tests/test_index_page.sh`.

### The console's live runs wrote its rules
Defect: driving the page in a browser found a 2B model generating four images off one request, a tool `model` enum
offering a model resolution would never pick, an incomplete download ranked first, a model inventing API fields the
prompt left out, and a 937 tok/s reading. Cause: per-event clients cannot time a buffered stream. Fix: a per-turn
media budget with a refusal sentence the model can act on; one list feeds the enum and resolution; candidates ranked
by how likely they are to work; one `apiReference` source for page and prompt; the console sums the server's `timings`
(`include_usage` is load-bearing). Guard: `app-web/test/`.

### `--model-dir` is repeatable
Defect: the app's custom folder fed only its own picker; choosing it pointed the server away from the library. Fix:
repeatable `--model-dir` merged first-wins (a duplicate id would fail registry init), an unopenable root skipped; app
reads use `readRoots` (everything served), writes and deletes use `ownedRoots`. Guard:
`tests/test_multi_model_dir.sh`, `ModelRootsTests`, `OwnedRootDiscoveryTests`.

### A load failure crosses the inference thread by name (#144)
Defect: a memory refusal on Krea came back as a bare 500 "Model load failed"; the reporter re-downloaded the model.
Cause: `ensureLoaded` freed `req.error_name` unread. Fix: `loadErrorFromName`; `InsufficientMemory` gets its own 503;
other failures echo the name; a memory refusal resets to `.unloaded` (not a sticky error state) so the advised retry
can work. Guard: "memory-refused loads keep their identity, other failures expose their name".

### `--no-drafter` and its siblings did not survive a model switch
Defect: a cold load after a switch re-enabled in-dir drafters and forgot `--draft-block-size` and `--ssd-streaming`.
Cause: `ensureLoaded` builds a second `LoadRequest`. Fix: load-shaping flags are retained on the Scheduler and
re-applied; `--drafter <path>` applies only to the launch model (`coldLoadDrafterDir`). Guard: lint pin +
`tests/test_cold_load_launch_flags.sh` (a silent arm must prove it did the work).

### Serial is not exclusive (DeepSeek-V4)
Defect: pi's DSV4 stream started emitting the app's conversation with doubled words. Cause: DSV4 keeps per-request
decode state on the model (`dec_state`), and nothing stopped a second slot from being admitted and interleaved. Fix:
`admitPendingTick` holds a second request for a `modelExclusiveDecode` model in `pending` (where keepalives flow)
until the first finishes. Guard: `tests/test_dsv4.sh` [7] (a marker chat fired mid-generation; red reproduced
"Kangaroo" in the output).

### A per-surface spec re-derivation is a list of one
Defect: DFlash decoded serially with no `[spec-stats]`. Cause: four handlers re-derived `use_drafter` from the Gemma
drafter handle only. Fix: one derivation (`requestEnableDrafter`, `requestSpecModes`). Engagement is asserted by
counts, never output shape.

### An adopted spec cache has one owner (#266)
Defect: SIGSEGV in `KVCache.deinit` in disconnect storms with hot-cache hits + MTP/DFlash. Cause: restored spec caches
were guarded by errdefers in both `runPrefill` and the generator. Fix: ownership moves at the call: the locals are
nulled before the `try`. Guard: the ordering test in `engines/mlx/scheduler.zig`.

### A missing tensor ended the process (#217)
Defect: `MISSING WEIGHT` hit `unreachable` and took the server down. Fix: `error.MissingWeight`, a named 503. A
present tensor of the wrong shape still dies inside MLX. Guard: "a missing weight is a load ERROR, not a process
exit".

### The disk bill and the shard index (#274)
Defect: unreferenced shards doubled a pack's bill; later a re-sharded Hub repo with a stale index failed
`NoWeightFiles`. Fix: `indexShardSet` bills only named shards, and returns null (all shards count) when none it names
exists. Guard: `modelDiskBytes bills only the shards the index names`, `loadWeights ignores an index that names no
shard on disk`.

## Memory bills and admission

### A gate that runs before the estimator that knows better is the estimator (#126)
Defect: MiniMax-H3 was permanently unloadable on 48 GB Macs: an idle server answered 503 "retry after current requests
complete", logging nothing. Cause: the registry gate billed `bytes_on_disk` x 1.1 before the media preflight's staged
estimate could run, and the commit billed the disk size too. Fix: one estimator (`mediaPeakFor` → `gateEstimateBytes`)
for gate and commit; the refusal names `--max-resident-mem` and logs estimate, cap and residents. Guard:
`tests/test_media_eviction_gate.sh` (sparse files with the real sizes).

### Staged bills: H3 and LTX
Defect: H3 was still refused (VAEs billed against the DiT; the DiT's file size, though AdaLN precompute frees 39% of
it); LTX billed a spare transformer variant and missed its out-of-dir Gemma encoder. Fix: `stagedPeakBytes(resident,
stages)`: resident + the biggest stage, each stage logging its residency; the DiT bill follows the same predicate
`generate` branches on; no uniform 10% on media. An unmeasured allowance is not a safe default. Guard: the staged-bill
tests.

### KV bills: caching layers, widths, chunks
Defect: the KV bill assumed every layer caches at one K/V width (Ling over-billed 3.2x; LFM2 and Nemotron-H counted
every layer); the score width was billed at the stored width (Ling scores 192, stores 128: ~4 GB unseen at 32K);
auto-context billed KV at fp16 and a chunk-bounded transient per token (a 16 GB Qwen3.8-27B reported 3.7k). Fix:
`kvBytesPerToken` over `attnCacheLayerCount`, `prefillScoreHeadDim` beside the stored width, `kvBytesPerTokenAtBits`,
a one-off `prefillTransientReserve`; sizer and guard move together. Guard: `engines/mlx/admission/bills_test.zig`.

### The prefill guard billed one arch's envelope for every arch
Defect: measured peaks on five checkpoints: every GatedDeltaNet cell and the MoE at its cap were billed short (worst
0.58x), an uncatchable OOM risk. Cause: GDN holds a chunk-wide q/k/v stream per linear layer; a MoE prefill replicates
the hidden stream `top_k` times; plain attention has neither. Fix: envelope = max(3 x MLP, MLP + chunk x
`prefillStreamBytesPerToken`) (a floor, never loosening) + a 512 MiB `PREFILL_RUNTIME_FLOOR_BYTES`. Guard: `every
MEASURED prefill peak on the box is billed for`.

### The prefill chunk was never sized to the machine
Defect: 16-32 GB users got memory 400s or a 1024 auto-context on prompts the box could serve. Cause: the default 8192
chunk reserved the same envelope everywhere. Fix: `resolvePrefillChunk` takes the widest rung whose reserve fits a
quarter of the room, pinned at load and passed through `InitOptions.pinned_prefill_chunk` (the Transformer holds a
config copy); vision prefills bill their real width. Guard: the chunk-choice tests.

## Sampling, grammar, loop stops

### A seed that only the synchronous sampler read
Defect: identical seeded requests flipped between `Ephemeral` and `**Ephemeral**`. Cause: the lazy sampler used MLX's
global RNG; the synchronous one built the same key every step. Fix: `seedKey` mixes the seed with a per-draw index
(`SamplingParams.draw`); `Generator.sampleLazy` is the one lazy sampler. Bar: seeded replay identical at temp 1.0,
cold and warm.

### JSON mode answered "## Attributes": the mask was another model's vocabulary
Defect: JSON mode free-ran on the second model a server served. Cause: the token→bytes table was a process singleton
built for whichever model served the first constrained request. Fix: the table lives on `LoadedModel`
(`grammarTokenBytes`), freed before its tokenizer. Guard: `tests/test_json_mode_multi_model.sh` (one build line per
model).

### An empty grammar mask is not a constraint
Defect: an extra key appeared under `additionalProperties:false`. Cause: `.after_value` accepted `,` with every
property seen, leaving no legal byte; the sampler drew id 0 and enforcement switched off. Fix: the comma is refused
when nothing can follow; a zero-count mask is a named bug before degrading. Guard:
`tests/test_json_schema_enforcement.sh`.

### JSON-schema decode was 60x slower on Flash Next (#380)
Defect: 0.9 tok/s against 54 without `response_format`. Cause: `buildMask` walked all 248k tokens through the grammar
per step. Fix: `TokenBytes` indexes the vocabulary by first and second byte with a plain-string split, so only tokens
whose first two bytes are legal are walked (0.1-0.4 ms per mask); the constrained step samples lazily so the graph
build overlaps the GPU. Guard: the brute-force oracle test in `sdk/grammar/token_mask.zig` + a probe-count
bar. Trap: a per-model `"mtp": true` overrides `--no-mtp`; read `[spec-stats]` before calling an arm serial.

### A loop cut must not say "length"
Defect: pi at 32% of its context said "reached the maximum output token limit" and compacted; each retry re-entered
the loop sooner because the cut turn came back as history. Fix: `stop` + `finish_details.type "repetition_loop"`
(Anthropic maps to `end_turn`); non-streaming replies are trimmed to the span start, keeping one copy
(`degenerateTail`); Responses history omits loop-cut turns. A delta cannot be retracted, so streams keep the tail.
Guard: `tests/test_loop_stop_signal.sh` (with a trim-off arm).

## LAN

### A transient mDNS hiccup must never evict a live peer, and dead dns_sd refs must revive
Defect: chats through a LAN peer alternated between success and 404 "peer offline"; a discovering server listed no
peers for 20 minutes. Cause: one resolve timeout or unreachable fetch removed the peer; a failed browse ref stayed
null forever and a dead advertise ref hot-spun. Fix: `attemptKnown` owns removal (`knownFailureAction`: 2 failures of
grace); browse and advertise refs are re-created every 5 s while wanted. Guard: `lan: transient resolve failures
retain a live peer`, `tests/test_lan_share.sh`.

### The LAN gate must resolve a model name exactly like dispatch
Defect: an alias reached an unshared model through the keyless gate. Cause: the gate checked the default model for any
non-id name. Fix: one resolver, `resolveRequestModelId`, for dispatch, the gate, load/unload and `/api/show`. Guard:
`tests/test_model_alias.sh` [2].

### An escaped id killed the server (SIGBUS in `rewriteModelValue`)
Defect: `org\/repo@peer` from a Swift client, or a Windows path id, made the proxy splice compute an offset outside
the body. Cause: the proxies got a canonicalized copy of the id where the splice needs a slice of the body; three row
emitters spliced ids raw. Fix: the proxies take the raw body slice and splice the re-escaped id; `rewriteModelValue`
refuses a value outside the body; emitters `jsonEscape` the id. Guard: `lan: rewriteModelValue refuses a model value
that is not a slice of the body`, `renderModelEntry: a Windows path id stays valid JSON`.

### @peer proxying is bounded by the tunnel marker, not loopback
Defect: the Agent Sandbox guest got 403 on `@peer` models (it is non-loopback by construction); a keyed request fell
through to the local default. Fix: any direct client may start one hop; a request carrying `X-MLX-LAN: 1` is never
proxied again (`isTunneledRequest` at gate and dispatch). Guard: `tests/test_lan_share.sh` hop checks.

### Same-machine peers re-exported each other's mirrors
Defect: a server listed `model@a@b`; a stale self-record mirrored its own models. Cause: same-Mac peers resolve
loopback-first and got the unfiltered list. Fix: discovery fetches get the filtered list even on loopback; entries
carrying `lan_peer` are never mirrored; `X-MLX-LAN-Token` detects a self-fetch (`error.SelfFetch`). Guard:
`parsePeerModels` tests + script checks.

## Engines (ds4, llama.cpp)

### A llama session trim that was never checked served the previous request (#286, #287)
Defect: a request offering one tool came back calling the previous session's `bash` command; a streamed request
answered `stop` with nothing. Cause: the shim ignored `llama_memory_seq_rm`'s result; recurrent and hybrid memory can
only roll back within a small snapshot window, so nothing was trimmed while `pos` said it was. Fix: on refusal the
shim clears memory and `sync` cold-prefills. Every hybrid GGUF arch was exposed; the PLD attribution was a
coincidence. Guard: `llama: re-sync after a long generated tail is a cold decode` (needs a hybrid `LLAMA_TEST_MODEL`).

## Admission, budgets and long context

### A chained video render finished and died at the socket (#283)
Defect: an H3 render of 5 chained windows decoded 701 frames, then failed `WriteFailed`: a 2.5 GB body the app hung up
on. Cause: the transport bill was per window. Fix: `videoRgbTransportReason` refuses at admission against
`MAX_VIDEO_RGB_BYTES` (the app's 768 MB) on delivered frames, chain windows included. Lifting the cap is a transport
change, not a number to raise.

### The hot-cache budget was never validated against what the weights left
Defect: a 143k-token Flash Next prompt OOM'd Metal with a 40 GB `--prefix-cache-mem` beside 70 GB of weights. Fix:
`clampedPrefixCacheMem` caps the budget at ceiling − (weights + context KV + prefill reserve), at the load site; a
disabled byte cap is bounded too and the clamp never returns 0; the budget is a hard cap. A RAM restore
refcount-shares the entry's buffers, so never add resident cache bytes to `needed`. Guard: the clamp test with the
crash's numbers + the lint pin on the load site.

### An oversized hot-cache decline is a cliff, not a cap (#330)
Defect: once a conversation's KV crossed the budget, nothing committed again (9/9 skipped, 23 minutes of re-prefill).
Fix: `trimLenForBudget` keeps the longest restorable prefix (a hybrid cuts at a checkpoint; media entries stop at
`media_start`), `trimmedCopy` is a real copy (an offset trim would bill the parent's capacity), spec payloads drop on
trim, one-shot so the same prefix is not re-copied each turn; the replace path evicts others and sheds its own
checkpoints before itself; `ssm_cps` ownership moves to the cache on every outcome. Guard: the #330 tests in the
prefix cache.

### The schema thinking-off gate lived on one surface of three (#331)
Defect: JSON schema + reasoning on qwen3.5/3.8 put the object in `reasoning_content` with empty content, on chat and
Responses. Cause: only `/v1/messages` forced thinking off for schema requests. Fix: one policy
(`schemaMasksThinking`): defer the mask only across a bare template-opened `<think>` with an atomic closer and no
finite budget; otherwise thinking off with the mask from token 0. Guard: `tests/test_json_schema_thinking.sh`.

### The sleep-inhibition gate sat in a tick that never runs (#251)
Defect: the Mac slept during long renders. Fix: `PreventUserIdleSystemSleep` is released right before the inference
thread's `queue_cond.wait` and taken after wake; the startup load takes it separately. Guard:
`tests/test_sleep_inhibit.sh` (matches the assertion by owner pid AND name), lint pin `holdsSleepWhileParked`.

### A Metal working-set OOM killed the server through mlx-c's default handler (#353)
Defect: a 458k-token prefill died with `Insufficient Memory` and took the process. Cause: mlx-c's default error
handler `exit`s. Fix: `installErrorHandler` latches; `checkError` per chunk (before the chunk is persisted: Metal
returns zeros before it aborts) and `checkErrorDecode` per tick; a finishing slot peeks the latch; one
`mapGenerationError` for every surface and stream phase. The latch exposed dormant double-frees (errdefer + defer on
one handle) and a per-tick leak that hit the 65,535 MTLSharedEvent cap. Guard: `tests/test_mlx_error_recovery.sh`.

### The admission guard: evict to admit, credit only what is provable, defer what it cannot price (#353)
Defect: a 450k prefill was refused while a 6.5 GB entry from the previous turn held the memory. Fix: admit when the
bill fits after eviction and evict on the inference thread (`evictLruToAdmit`, re-asking after each, never the entry
being restored); credit only reclaimable residency from published digests; `restoreWouldPinEntry` declines a tiny
match on a huge entry; a warm prompt is deferred and billed with `WarmPrefix`; retained SSM checkpoints and a KV
reservation past 32k are billed; `max_tokens` is clamped first and growth bounded by `RESERVE_GEN_HEADROOM`. The
refusal is a named 400 (`PrefillDoesNotFit`) with one `[admission]` line. Guard: the admission tests.

### The prefill width is a property of the request, then of the chunk (qwen4_exp)
Defect: at a 1M context the load-time sizing left every prompt prefilling at 1024. Fix: `chooseRequestPrefillChunk`
picks the widest rung the estimator affords for this request after eviction; `adaptivePrefillWidth` re-prices at each
chunk boundary (down at once, up one rung with hysteresis). Guard: the width tests.

### The load-time bills ran before what they asked existed
Defect: bills inside `Scheduler.init` read a dense KV width, a placeholder context and the raw cache ask: wrong
budgets that swung between boots and a bigger ask that bought less cache. Fix: publish `configuredKvQuant()` before
the load, resolve the context the advertiser's way, reserve the constant `CTX_SIZING_CACHE_RESERVE`, bill the static
ceiling at load, the cache is the residual claimant (`longCtxGated` archs). Guard: the init bill tests.

## Prefix cache (RAM + SSD)

### SSD-first prefix cache (qwen4_exp + `--prefix-cache-disk`)
Defect: review found the spill dropping RAM entries it had never written (every silent skip read as persisted),
ignoring the cache cap, draining the writer on the inference thread, stamping a manifest version that a downgrade
discarded, persisting a warm turn's whole prefix inside TTFT, and evaluating once per tensor. Fix: `PersistOutcome`
(only a landed, indexed, stat-licensed write discards RAM), a two-tier hard cap, in-flight entries not evictable,
`metaVersionFor`, write-through of new spans only, one eval per chunk file; restore by move: a full-prefix hit CHECKS
the entry out so the first append can donate the buffer. Guard: the SSD-first tests (`armTestSpace`).

### The SSD tier outgrew its cap (#573)
Defect: under an agent workload the store passed 300 GB against a 40 GB cap. Cause: `appendSsmOnly`'s per-term delta
dropped whole checkpoint lists in ReleaseFast; `persistQsaHistory` reported 0 for a file still on disk. Fix: bill by
measure (`nonChunkBytes` after − before). Guard: red only under `zig build test -Doptimize=ReleaseFast`.

### The SSD tier refused a volume with 117 GB usable
Defect: `statfs` showed 36 GB on a volume Finder showed 117 GB free, so every persist was declined. Fix:
`msv_volume_free_for_use` (`volumeAvailableCapacityForImportantUsage`) is `free`; statfs is the fallback. Guard:
`volumeSpace: free is what the OS grants`.

### A checkout is a promise until the append donates
Defect: a request refused after a restore-by-move lost a 364k-token, 7.8 GB session. Cause: the entry's handles were
released at restore time. Fix: `donateCheckout` runs right before `Generator.initWithOptions`, below every arm that
can refuse; `releaseCheckout` hands an undonated entry back intact; the eviction credit and both refusal sentences
read one `AdmissionBill.evictionCredit()`. Guard: the checkout tests.

### Checkpoint retention thinned the wrong end
Defect: drop-oldest left a 383k entry's lowest checkpoint above the budget, so the #330 trim declined. Fix: one
policy, `spanPreservingDropIndex`: thin the interior by smallest span, keep both ends and the newest quarter dense,
protect the checkpoint at a media boundary (`boundaryCheckpointIndex`). Guard: the thinning tests.

### Every long-context mechanism is gated by one predicate
Defect: mechanisms measured on one checkpoint at 100k-1M tokens (reservations, pad-waste caps, retention, admission
terms, chunk bars) are trades. Fix: they are opt-in by arch through `ModelConfig.longCtxGated()`; the fixes that rode
with them are not gated.

### The cancelled-prefill commits that never were, and the disk tier that never served
Defect: a 122k agent session cold-prefilled ~95k tokens on every retry while the log claimed `committed 80896/121956`.
Cause: four stacked bugs: a budget decline was logged as a commit; a cancel commit kept the pixel key on rows before
the media, locking out the shared text prefix; the SSD tier skipped any media request; checkpoints came from the
per-flush budget's leftovers (always zero). Fix: `CommitStatus`; the pixel key follows the rows covered; disk restores
cap at the media boundary; checkpoints come off the top of the flush budget; a RAM decline spills to the SSD tier
(`spillDeclinedToDisk`, flush budget floored at 4 GB); the disk hybrid arm ranks by restorable checkpoint
(`bestHybridMatch`). Guard: the prefix-cache tests for each.

### A failed restore left a half-bound cache; a fully checked-out commit spun forever
Defect: a `QsaHistoryGap` after binding left the prompt in KV twice with wrong RoPE positions; with every entry
checked out, a commit's eviction loop never progressed. Fix: an `errdefer` returns an empty cache;
`evictOneLruProgress` gives every eviction loop a no-progress exit; the disk arm adopts the sidecar of the entry it
restored and honours `poisoned`; `meta.json` scalars are checked. Guard: unit tests in `engines/mlx/cache/` (the
eviction one is red as a timeout).

### The hot-cache budget was clamped once and never revisited (#364)
Defect: a second model loaded beside a 70 GB one got a ~0 cache budget for the life of the process. Fix:
`reviseHotCacheBudgets` after every load and unload (and before each prefill for 10 s after an unload, while freed
memory reaches the OS), excluding each model's own cache bytes; `setBudget` evicts down to a smaller cap. Guard:
`tests/test_prefix_cache_budget_revisit.sh`.

### An image turn could not inherit the checkpoints it restored from
Defect: after 43 screenshot turns, a text-only turn cold-prefilled 348k tokens. Cause: `bestCheckpointDonor` demanded
exact pixel-key equality, so image commits held only their own tail checkpoints. Fix: the donor scan applies the
cross-key rule below the media boundary; thinning protects the boundary checkpoint. Guard: `a text turn after image
turns restores the pre-media prefix`, `tests/test_vision_prefix_cache.sh`.

### A placeholder id in ordinary text capped every SSD restore
Defect: a text-only 73k chat restored 16,384 tokens from SSD. Cause: `firstMediaPlaceholder` scanned for media token
ids in a text-only prompt (a pasted source held one). Fix: a boundary exists only where media rows do (`has_media`).
Guard: `firstMediaPlaceholder: a placeholder id in ORDINARY TEXT is not a media boundary`.

### Stream and non-stream disagreed on three edges
Defect: a spent reasoning budget streamed the rest of the thought as the answer; a non-stream tool reply dropped its
visible preamble; a non-stream disconnect answered 200 `length`; streams withheld one token on a stop match where
non-stream cut at the index; `seed: -1` was ReleaseFast UB. Fix: the budget is a delivery latch;
`visibleToolPreamble`; `client_disconnect`; `stopSequenceCut`; request ints are clamped (`parseRequestSeed`,
`clampJsonI32`). Guard: unit tests on the helpers.

### A reply that is not valid UTF-8 was lost
Defect: a lone UTF-8 lead byte in `content` made the whole reply unparseable. Fix: sanitizing lives inside every
escaper (`chat.utf8Next`, one U+FFFD per invalid sequence). Guard: `EVERY JSON string escaper survives bytes that are
not valid UTF-8`.

### Constrained JSON routing and buffer lifetimes
Defect: a reasoning marker inside schema-valid JSON was re-parsed as markup; Anthropic's thinking block freed the
router whose buffer the later text block read. Fix: the payload offset is authoritative
(`reasoning_protocol.Delivery`); both buffers live until serialization. Guard:
`tests/test_json_schema_protocol_routing.py`, the protocol tests.

### Qwen3-Embedding sub-batches shared one KV cache
Defect: a split `/v1/embeddings` request returned wrong vectors or a 500, and the next request crashed. Cause: the
decoder cache was reset per request, not per sub-batch. Fix: reset before every sub-batch (`computeEmbeddingsBatch`).
Guard: `tests/test_embeddings.sh` [4c].

### A grammar-masked model idled on whitespace
Defect: after a warm restore the model's argmax was off-schema; it emitted free `\n` until the loop guard cut an empty
`length`. Fix: no whitespace outside the root value; the model's own layout inside, capped (`MAX_FREE_WS`). A fully
compact grammar cost extraction quality on a 0.8B. Guard: `grammar admits free whitespace inside the root value only`.

### ds4 sessions were per request; embeddings segfaulted on an engine model
Defect: llmprobe on ds4 was `Killed: 9` (four 13 GB sessions); `/v1/embeddings` dereffed a null transformer. Fix: one
persistent ds4 session per model under the `session_busy` claim; engine models refuse embeddings by name. Guard:
`tests/test_ds4_serve.sh`.

### Engine models counted reasoning with an empty tokenizer
Defect: ds4 replies reported `reasoning_tokens: 0`. Cause: engine models have a stub tokenizer. Fix:
`server.encodeText` owns the engine branch for every count. Guard: the format matrix's GGUF arm.

### Logprobs in the logits dtype; Responses dropped the budget
Defect: bf16 logprobs were off by up to 0.06, f16 underflowed to NaN (invalid JSON); `/v1/responses` discarded
`reasoning.effort`'s budget. Fix: `logits - logsumexp` in f32; Responses takes chat's precedence and arms the bound.
Guard: `computeLogprobs: f16 logits keep finite, exact log-probabilities`, `tests/test_reasoning_budget_stream.sh`.

### Cancel mid-tick read a freed ThinkBound
Defect: stopping a pi request while a second MTP stream decoded SIGSEGV'd in `thinkBoundTick`. Cause: `think_bound`
and `constraint` point into the handler's frame, which was freed while a tick's snapshot still held the slot. Fix:
`Slot.in_pass` counts inference-thread passes holding the slot; `complete` waits for zero. Guard:
`tests/test_cancel_mid_tick.sh`.

### Concurrent long prompts were each admitted against the same free memory
Defect: four 64K requests on a 27B were all admitted against the same 19.9 GB; on macOS 26.5 that was an IOGPU kernel
panic. Then restored 32K prompts still overran: a restored prefix shares cached buffers until its first append, so
active memory does not move at admit. Fix: the inference thread re-bills before each prefill (`slotHoldsForMemory`;
with company it waits in `pending`, alone it proceeds); `PromiseLedger` counts what admitted requests were promised
and have not allocated; `osReserveBytes` stays out of the plan; DFlash context K/V is billed per row. Guard:
`tests/test_memory_pressure_4way.sh`.

### A chunk boundary is the only yield point, so the chunk is the stall
Defect: a cold 8K prefill stalled decoding streams for 33 s; interleaving ticks at boundaries did nothing because an
8K prompt at chunk 8192 has none. Fix: `companyPrefillChunk` narrows the prefill to 2048 while anyone decodes. This
bounds the inter-token gap, not 4-way aggregate throughput. Guard: `tests/test_prefill_interleave.sh`.

### The load preflight never saw the GPU limit
Defect: with `iogpu.wired_limit_mb` at 36 GB, a 70 GB pack passed the preflight, failed warmup and refused every
request. Fix: available = min(free, `maxRecommendedWorkingSet()`) at both preflight sites. Embedded engines keep their
own open-time failure. Guard: `effectiveAvailableBytes is capped by the GPU working-set limit`.

### Tokenizer-derived config is set in one place
Defect: app-loaded models placed every image at the prompt's end. Cause: the user-turn marker and LFM2 image tokens
were set only on the `--model` path. Fix: `ModelConfig.applyTokenizer` for both load paths (placement now comes from
template placeholders anyway).

### An SSD restore held one file open per chunk, and its failure failed the fallback (#495)
Defect: a 319k-token restore failed at chunk 242 and the cold fallback failed too. Cause: lazy `mlx_load_safetensors`
keeps its file open until evaluated (256 fds); the swallowed failure left the MLX latch set. Fix: evaluate each chunk
before the next; restore paths drop their own latch. Guard: the fd-limit and latch tests,
`tests/test_prefix_cache_disk.sh` [7].

### A stop string inside the thought ended the whole reply (#549)
Defect: `stop: ["```"]` on a thinking model ended inside `<think>` with empty content. Fix: `chat.answerStopIndex`
counts a match only where the split delivers it as content, judged on the text up to the match (`/v1/completions`
keeps raw matching). Guard: corpus `a stop string ends the answer, never the reasoning`,
`tests/test_stop_in_reasoning.sh`.

### Logprobs were null whenever `response_format` was set (#515)
Defect: constrained requests returned `"logprobs": null`. Cause: no arm of `nextConstrained` computed one. Fix: it
computes the chosen token's entry from the raw logits, before the mask. Guard: `tests/test_logprobs.sh` [7].

### Two servers listened on one port
Defect: two instances both reached "Server listening"; clients hit an untracked one. Cause: the port probe ran before
the load and std's `reuse_address` sets SO_REUSEPORT. Fix: `listenExclusive` clears SO_REUSEPORT on the bound socket
(Linux never sets it). Guard: `listenExclusive: a second server cannot bind a port that is already listening`.

### A model that failed to load was answered by the default model (#585)
Defect: a chat naming a failed pack by path got 200 from another model. Fix: `resolveRequestModelId` resolves paths
through `peekByPath`; only an unknown NAME falls back; an unregistered path is a 404. Guard:
`tests/test_load_failure_no_fallback.sh`.

### A stop string spanning tokens leaked its first bytes on every stream
Defect: `stop: [", 12"]` streamed `…11, 1`. Fix: `StopStream` holds a tail that may still begin a stop until the next
token decides it. Guard: `StopStream` test + stream == non-stream in `tests/test_api_edges.sh`.

### A media job held the inference thread for its whole life
Defect: pi on GLM-5.3 showed "Operation aborted" for 10 minutes while its own textured 3D job ran. Fix: every
backend's step poll is `Progress.boundary()`, which serves chat for as long as the step took; pure-CPU stages run on a
worker (`gen_sse.offload`); the job's peak is billed to chat admission (`gen_reserve_bytes`); decode clocks reset at
each boundary so a media step never folds into the round-cost table. Guard: `tests/test_gen_chat_interleave.sh`.

### An image turn never reached the SSD tier (#494, #782)
Defect: once a session carried an image nothing persisted; a restart restored 123k of 364k tokens. Fix: v9 disk
identity is (tokens, spans): the full record persists, image rows keyed per item, and the clamp happens at match
(`mediaSharedBound`): agreeing spans read the media rows back, a divergent list stops below the first differing item.
Guard: `an image turn persists its full media-keyed prefix to the SSD tier`, `tests/test_prefix_cache_disk.sh` section
9.

### A streamed Responses turn reported no prompt-cache hit
Defect: streamed `response.completed` read `cached_tokens` 0 and 0 ms timings. Cause: the streaming arm rebuilt its
result by hand. Fix: `StreamingTokenStream.generationResult` from everything `finalize` snapshotted. Guard:
`tests/test_responses_streaming.sh` [F].

### The console showed 0 models behind a proxy that mounts it under a path (#698)
Defect: under `tailscale serve --set-path`, the console fetched root-absolute URLs and got the proxy's 404 HTML. Fix:
`apiPrefix(pathname)` resolves every fetch and link against the mount. Guard: `app-web/test/connection.test.ts`.

### A GGUF picked by its file 404'd on the first chat
Defect: the app loads a GGUF quant by its file path and `probeModelDir` took folders only. Fix: `probeModelDir`
accepts a `.gguf` file (not sidecars or embedding GGUFs), named by its stem. Guard: `registerByPath takes a .gguf
file`, `tests/test_llama_gguf.sh` [8].

### MiniMax-H3 admitted a canvas it could not hold (#764)
Defect: a 1536x672 x 124f run died in an uncatchable Metal OOM on its first step; every request printed the same peak.
Fix: `h3RequestRows` x per-row bytes (+ the fast recipe's per-block attention cache) bills each request before any
stage loads, prices the resident set too, and `handleVideoH3` refuses the staged plan by name; on CUDA the DiT bills
to RAM. Guard: `h3 request bill`, `h3CanvasBill`.

### A loop-stop inside the thought ended the agent's turn
Defect: Flash Next looping `?!?!` in its thought was cut with `stop` and no tool call; omp stopped. Fix:
`loopRecoveryAction` commits the early-stop line and closer once and decoding continues; every request arms the bound
(`thinkBoundBudget`); a second or answer-side loop still cuts. Guard: `loop recovery:` tests.

### A refused load failed the next request
Defect: after a corrupt pack or a non-safetensors LoRA path, the next unrelated request answered 500. Cause: MLX's
latch was not dropped where the failure was swallowed. Fix: `dropLatchedErrorUnless(had_error)` at the load catch,
`lora.loadFile` and `runGenJobs`; memory failures answer 503 (`jobFailureStatus`). Guard: the latch tests.

### Model churn: six models under a cap of three, and nameless "Model load failed"
Defect: 16 clients over seven small models left six resident and some requests 500'd nameless. Cause: the planner
counted only `.ready` models; a just-loaded model had refcount 0 until its requester woke. Fix: other `.loading`
entries count (`countLoadingLocked`); `ensureLoaded` pins the requester's reference when it claims the load. Guard:
`planEvictions: another model still loading counts against the model cap`.

### A Stop click mid-flush crashed the server
Defect: eight agent clients with random disconnects crashed Flash Next in ten minutes
(`POINTER_BEING_FREED_WAS_NOT_ALLOCATED`). Cause: held token texts were freed per item while sending and again by the
handler after a failed send. Fix: both arms free and clear the list in one place. Guard: live only (20 minutes clean).

### A load right after an unload was refused for memory it was about to get
Defect: unload then reload GLM-5.3: 503 "only 66 GB"; under churn, a thousand refusals in 15 minutes. Cause: the OS
unwires freed pages over one to two seconds. Fix: `settledAvail` re-reads every 100 ms while the figure climbs, up to
5 s. Guard: `settledAvail: a refused reading waits while freed memory still reaches the OS`.

### An idle media model got an agent turn refused
Defect: a 36.7k-token GLM turn 400'd with Krea idle beside it; pi ended the session. Fix: before refusing,
`fitAfterUnloadingIdle` unloads idle resident models, least recently used first, never the request's own. Guard: `a
prompt that does not fit unloads idle models, oldest first, until it fits`.

### An embedding model's load hid the chat model's warm cache (#854)
Defect: with bge-small loaded, the chat model's warm entry billed as 0 and was then evicted. Cause: every load pointed
the scheduler's hot-cache view at the loaded model's cache. Fix: only a chat-capable MLX load takes the view;
`moveHotCacheViewOff` hands it back before a cache is freed; a revise after an unload only raises budgets. Guard: the
two `hot-cache view:` tests.

### A 16 GB Mac advertised 870 tokens beside a big model
Defect: a 9B pack auto-sized to 870 tokens (gemma-3-12B to 45). Cause: the sizer billed a fixed 2 GB cache reserve
first; the floor applied only at exactly 0. Fix: the reserve is at most half the room beside the weights and the
prefill (`ctxCacheShare`); floor below 1024. Guard: `context sizer: the prompt-cache reserve never takes more than
half the room beside the weights`.

### An agent started from a piped script got `/dev/tty`
Defect: from the console's `curl … | sh` launch, Bun agents died on `EINVAL … kqueue`, Grok exited and omp's terminal
queries landed in the shell. Cause: macOS kqueue refuses `/dev/tty`. Fix: the agent's stdin is a dup of whichever of
stderr/stdout is a terminal, riding the agent's `exec`; `website/agent.sh` writes the script to a file and execs it.
Guard: `tests/test_launch_script.sh` [6], `tests/test_website_agent_install.sh` [3].

### A faulted CUDA context took the server down through an empty array
Defect: after `illegal memory access` on an RTX 5060 Ti, Kokoro indexed an empty array and SIGSEGV'd. Cause: CUDA
errors are sticky and host-to-device copies assume they cannot fail. Fix: the latch classifies a sticky CUDA message
(`mlxErrorIsContextLost`); `ensureLoaded` refuses with 503 `gpu_context_lost`; `/health` answers 503. Guard:
`test_mlx_error_recovery.sh` [7].

### Chat interleaved inside an image job faulted the CUDA context
Defect: the stress shape faulted 5 times in 8: a decode inside an image job's yields left state the next job's first
step touched after an eviction. Fix: the chat-interleave hook is off on CUDA (`genYieldEnabled`), Metal unchanged.
Guard: `genYieldEnabled: off on CUDA`.

### A short GLM prompt without `max_tokens` was refused for 43 GB
Defect: a 431-token prompt needed ~43 GB. Cause: decode growth was priced for the rest of a 1M window. Fix: growth
billed to `RESERVE_GEN_HEADROOM` (8192). Guard: `GLM bill: decode growth is billed to the generation headroom`.

### A strength-0 LoRA moved int8 renders
Defect: an attached adapter at scale 0 changed Krea/Qwen-Image bytes on the NAX int8 route. Cause: fused paths keyed
on adapter count. Fix: `lora.anyActive` (a nonzero scale) gates them. Guard: `tests/test_multi_lora.sh` [2].

### An SSD-restored 38k prompt was refused on a 16 GB Mac
Defect: the restored prompt was refused (`PrefillDoesNotFit`). Cause: every buffer was built at exactly the restored
length, so the one re-forwarded token grew (copied) every layer. Fix: the restore leaves the first grow's room
(`KVCache.nextCapacity`). Guard: `HotPrefixCache: a disk restore leaves room for the token it re-forwards`.

### `/v1/images/generations` closed the socket on a Metal OOM
Defect: the client got no reply. Cause: the lazy denoise ran inside `toPng`, whose bare `try` skipped the failure arm.
Fix: one `imageJobFailed` arm (503 for memory) for the generate call and the encode. Guard: `a failed image job
answers by name, never a closed socket`.

### A textured 3D job answered 200 with a broken mesh
Defect: a Metal `Invalid Resource` in the paint stage still returned an 8 MB GLB. Cause: the stages read results on
the host, leaving no error to catch. Fix: `latchedJobFailure` after the shape stage and after the job. Guard: `a
latched Metal failure fails the 3D job by class`.

### A 200,000-character word hung `/v1/embeddings`
Defect: one request pinned a connection thread for good. Cause: greedy WordPiece is cubic in the word. Fix: a word
past `max_input_chars_per_word` is one `[UNK]`, as HF does. Guard: the WordPiece test.

### A seeded request answered differently every run under the Gemma drafter
Defect: seeded runs with a drafter produced five texts in five runs. Cause: the verify round's correction draw used
MLX's global RNG. Fix: `correctionDraw(accepted)` keys it at the output position. Guard: "a seeded correction draw
replays", `tests/test_drafter_equivalence.sh`.

### A vision agent's session died on the 64 MB chat body cap
Defect: after 20 screenshots in history, every request was a 413 and the session ended at 117k of 256k tokens. Fix:
chat routes share the media body cap (`max_media_request_bytes`); only `/v1/decisions` stays small. Guard: `the chat
body cap admits an agent session that keeps its screenshots`.

### A media stream fell silent in a CPU stage
Defect: xatlas unwrapped a 2M-face mesh for 300 s and undici cut the stream. Fix: `gen_sse.offload` runs a beat thread
that writes an SSE comment after the keepalive interval (streaming jobs only; a comment before a JSON head would
corrupt it). Guard: `offload keeps a streaming job's socket alive through a long CPU stage`.

### A 26k-token agent turn was refused beside a resident model (#932)
Defect: Qwen3.8-27B on a 48 GB laptop 400'd every turn after a while. Cause: purgeable pages (browser caches the OS
drops first) read as taken. Fix: purgeable joins the grantable sum in `computeAvailableBytes` and the app's twin.
Guard: `computeAvailableBytes sums the reclaimable classes…`, `testPurgeablePagesCountAsGrantable`.

### A client that hung up during a cold prefill left the GPU prefilling for a ghost
Defect: Claude Code retrying a 40K-token prompt queued abandoned prefills while no client was connected. Cause: the
handler checked the peer only after a token arrived. Fix: keepalives flow while waiting for the first token, each
probe notices a gone peer, and the chunk loop aborts with `error.Cancelled`. Guard: `tests/test_disconnect_cancel.sh`.

### Claude Code aborted a tool turn with "Content block not found"
Defect: `/v1/messages` with thinking + tools on a template-opened-think model leaked `</think>` into text and opened a
block at an index still held. Fix: think state seeds from the prompt; the stream tracks which block is open and closes
it before the next starts. Guard: `tests/test_messages_stream_thinking_tools.sh`.

### A llama.cpp session reused across turns failed its second prefill on Qwen3 GGUFs
Defect: the second turn answered 500 `failed to find a memory slot`. Cause: on sliding-window layers the non-full SWA
cache never exposed trimmed slots. Fix: sessions use `swa_full = true`; `syncWithFallback` resets and retries once
cold. Guard: `tests/test_llama_persistent_session.sh`.

### `/props` said `decode_attn_quant: false` for every plugin-served MoE pack
Defect: plugin-served packs with dense attention projections reported the requant off while it ran. Cause: only the
legacy load set the shell's `dense_attn_proj`. Fix: one predicate (`load/init.denseAttnProj`) feeds both loads.
Guard: `laguna plugin: a served pack reports its dense attention projections to /props`.
