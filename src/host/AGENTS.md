# host/ — the model-agnostic server

HTTP surfaces, request parsing, streams, sampling surfaces, LAN. Stories: `docs/gotchas/server-http.md`. Chat templates and tool calls: `chat/AGENTS.md`.

## API surfaces

### HTTP APIs

- **OpenAI chat/completions + Responses**: usage ALWAYS carries `prompt_tokens_details.cached_tokens` (ONE `formatChatUsage`); thinking opt-ins = `reasoning_effort` OR `enable_thinking`, top level or in `chat_template_kwargs` (`reasoning_budget_tokens` outranks); `n>1` 400s. `/v1/responses`: every SSE event has `sequence_number`, `background:true` → 400, stateful via `ResponseStore`, WS via Upgrade, NO `[DONE]` on WS.
- **Continuing a partial reply** (`chat.continuationRequested`/`continuationPrefill`): trailing assistant message becomes a PREFILL (whitespace trimmed). `/v1/chat/completions` takes `continue_final_message` EXPLICITLY; `/v1/messages` INFERS; `/v1/responses` neither. Tool-call replies never continue; in the tokenize-cache key; ds4 can't (`continuationRejectReason`).
- Context-overflow 400s name BOTH counts (`contextOverflowMessage`); the app renders it as a card.
- **Anthropic `/v1/messages`** (Claude Code): typed blocks, `input_schema`→`parameters`, stop-reason map incl. `stop_sequence` echo, full SSE block lifecycle. Launcher env: `ANTHROPIC_BASE_URL` + dummy keys + `ANTHROPIC_DEFAULT_*_MODEL=mlx-serve`.
- **Ollama `/api/*`**: pure translation, no-model endpoints answered pre-scheduler; `resolveName` handles `name:tag`. One path must never register under TWO ids (`registry.peekByPath` — double-residency OOM).
- `/v1/models` rows carry `context_length`+`max_model_len` at TOP level twinning `meta.context_length` (#188): discovery clients never read `meta.*`; an outside plugin's pack adds `plugin`. Both emitters. Guard: `tests/test_models_capabilities.sh` [4b].
- **Providers**: `<model>@<name>` from `~/.mlx-serve/providers.json` proxies `/v1/chat/completions` ONLY (other surfaces = named 400) through curl with the provider's key; rows ride `/v1/models` with a `provider` badge while the probe answers, a non-empty `models` list FILTERS the provider's list (and IS the list when it has none), none when unreachable. Never shared to the LAN. Design: `docs/reference.md`.
- **LAN sharing**: proxy is a TRANSPORT; keyless gate = `routeClass` × `SharedSet`; `<id>@<peer>` mirroring; loops impossible by construction (self-token + tunnel marker, one hop). Design: `docs/reference.md`.
- **Observability** (`--metrics`): zero cost off; TTFT at prefill completion; live tok/s via ONE atomic per tick. `--api-key`: loopback exempt; `/health`+OPTIONS open; `constTimeEql`. No admin surface.

### Tool calling (server pipeline)

With `tools`, tokens buffer for detection (all tag families + raw JSON); thinking buffers separately. Parse chain: strict → tolerant repairs → truncation salvage, then the ONE chokepoint `server.parseToolCallsForRequest` = parse → inferred-name filter → parallel clamp → buried-param hoist → schema coercion (the last two gated by `--no-tool-autocorrect`; emitted `arguments` ALWAYS valid JSON). Serialization `chat.serializeMessagesJson`: role "tool" native, args as JSON STRINGS, every string via `appendJsonString`. Streaming: full args in ONE SSE delta, thinking → `reasoning_content`. Fallback `fallbackFormatChat`. Client: `app/CLAUDE.md`.

## Rules

The MLX engine boundary:
- **The host reaches the MLX engine only through `host/engines/mlx.zig`** (a table the engine exports, `engines/mlx/api.zig`): no `@import` of `src/engines/`, and the engine's objects held on the host (`abi/mlx_handles.zig`) are stored and null-checked, never opened; a fact the host needs is a table call (`mlx.facts`). Under `-Dunits` the host unit sees those handles opaque, so a reach-in fails the build.
- **An error crosses the table by NAME** (`sdk/util/err_name.zig`): every compile unit numbers its own error set. A new error a handler switches on joins `err_name.Known`.
- **A table or function is told apart by a value it carries, never its address** (`CacheOps.sdk_slots`, `Source.resident`, a table's `name`): each unit holds its own copy, so qwen4 wrote its MTP rollback captures into its own slot and every request failed.
- **An `Io` handed into another unit is rebuilt there** (`io_util.local`, at the engine table, the module calls and the stream table): its vtable returns the numbers of the unit that built it, so the host's `FileNotFound` read as `HostUnreachable` and qwen3_asr failed to load.
- **State two units touch is `Shared`** (`sdk/util/shared.zig`), the host owns it and lists it in `src/shared_all.zig`; a plain `pub var` the engine reads splits into two copies under `-Dunits` (`zig build units-check` names them). A plugin is told apart by its table's NAME, never its address: each unit holds its own copy of a table.

Reasoning, budgets, agents:
- **A reasoning budget is enforced at DECODE** (`armThinkBound` → `think_bound`, `thinkBoundTick`): early-stop line + atomic closer committed as ONE forward (`commitForcedTokens`); bare-`<think>` family only; all three surfaces (`reasoning_budget_tokens`). Guard: `tests/test_reasoning_budget_stream.sh`.
- **Effort budgets are pi's ladder** (`responses.effortBudget`): minimal 1024, low 2048, medium 8192, high/xhigh uncapped; the word maps to a budget only where the bound can arm (`effortWordOnly`).
- **Agent output share is ctx/2; compaction reserve ctx/4 capped 20000** (`host/launch/budget.zig`, served as each chat row's `agent_budget`; the app and the pi extension read it, never derive it); pi `settings.json` is MERGED, opencode2 carries `compaction` + `limit.output`.
- **A launch below the agent's context floor WARNS, never blocks** (`budget.contextFloor`: claude 64k, opencode 32k, others 16k; the row's `agent_budget.too_small_for` drives the app's NSAlert BEFORE launch).
- **A constrained JSON payload offset is AUTHORITATIVE** (`reasoning_protocol.Delivery`): nothing after it is re-parsed or cleaned. Guard: `tests/test_json_schema_protocol_routing.py`.
- **Schema-mask surfaces share one policy** (`schemaMasksThinking`, #331): defer only across bare `<think>` + atomic closer with no finite budget; tools present = no mask. Guard: `tests/test_json_schema_thinking.sh`.

Request parsing + media:
- **OpenAI audio OUT is the card's recipe, never a free prompt** (`api/speech.zig`): `modalities` + `audio{voice, format}` injects the voice-chat system line and prefills `mixed_start`; a spoken reply is stored (`Store`, 1 h) so a later turn's `audio.id` replays its transcript; with `tools` only where the card speaks them (`SpeechFacts.tools`, the `speech_tools` capability: no prefill, a reply that never speaks is parsed for `tool_calls`), else a named 400; unknown id, mp3, a wav stream are named 400s. Guard: `tests/test_lfm3.sh`.
- **`messages.deinit` frees the Message array and NOTHING it points at**: media is owned by ONE `server.RequestMedia`; `Message` BORROWS; slots are INDICES.
- **Undecodable media anywhere in the conversation is a NAMED 400** (`IMAGE_DECODE_REJECT`; remote URLs never fetched); `stop: ""` skipped; schema-less `json_schema` 400; empty embedding input 400; Ollama promptless generate = load handshake. Guard: `tests/test_api_edges.sh`.
- **Latest-turn user media on a tower-less model is refused by NAME** (`mediaRejectReason`), older/tool media replaced by a note the agent can read (`dropMedia`); non-text model on a text surface 400s BEFORE prefill (`textGenRejectReason`; new surface → `isTextGenRoute`, new modality → `modalityFromType`).
- **A Responses `reasoning` item replays as the assistant turn's `reasoning_content`** (plaintext only, joined with the answer or calls that follow, `joinReasoningTurn`); an item this server cannot read (another provider's `encrypted_content`, a malformed part) is DROPPED, never a 400. Guard: `tests/test_responses_reasoning.sh`.
- **Media atoms come from media parts, never rewriting user/tool text** (`chat.appendMediaContentParts`); item k expands to run k (`expandMediaPlaceholders`), count/kind mismatch = named 400. Encoder cache keys pixels; text parts JOIN in order. Guards: format corpus, `tests/test_nemotron_bolt.py`.
- **A dispatch field must be readable from every body SHAPE** (`parseModelFromRequest(body, content_type)`); header parameter lookups key at a boundary (`name=` vs `filename=`); binary bodies are not logged (`bodyIsText`); request ints clamp (`parseRequestSeed`, `clampJsonI32`).
- **A client-supplied PATH is proven on OUR side of the mlx boundary** (`lora.loadFile` stat → 400) and accepted only from LOOPBACK (`parseLoraFields(…, local)` → 403, #540: missing vs not-a-LoRA is a host-file oracle). `/v1/images/edits` forwards the LoRA fields (#268).
- **Hand-written error text is not JSON**: escape at the SINK (`jsonEscapeMessage`), truncate on a UTF-8 boundary. NO model-byte string is guaranteed UTF-8 — sanitizing lives INSIDE every escaper (`chat.utf8Next`); logprobs `bytes` keeps exact bytes.
- **A list of owned buffers is freed in ONE place**: a per-item `defer free` in a send loop plus the handler's defer over the same list double-freed when a send failed (client gone mid-flush; `flush_text` on chat + messages).
- **A body is exactly its `Content-Length`, read within `REQUEST_READ_LIMIT_MS` of silence** (`readRequest`, `awaitReadable`: a socket timeout panics std's reader; extra head bytes overran the heap, a 2^64 length wrapped) and nests at most `MAX_JSON_DEPTH` (recursive schema walkers overflowed a thread stack); zero prompt tokens are refused at `Scheduler.submit`; a client string spliced raw into a template is vetted first (`echoModelName`).

Sampling + logprobs + streams:
- **A `seed` binds EVERY sampler with a fresh key PER DRAW** (`generate.seedKey` + `SamplingParams.draw`), the verify round's bonus/residual draw included (`correctionDraw`: keyed at the output position; an unkeyed `sampleFromProbs` made every seeded drafter run differ).
- **Logprobs are the MODEL's distribution**: pre-temperature, `logits - logsumexp` in f32 (`computeLogprobs`), ids travel with values, entry belongs to the RETURNED token (`pending_logprob`), pre-GRAMMAR-mask under `response_format` (`nextConstrained`). Bar: temp-0 rank 1 == chosen.
- **Streaming logprobs**: a SIBLING of `delta`/`text` (`ChunkExtras`), ONE collector `StreamLogprobs` with a high-water mark, shipped once; `logprobs.content` describes `message.content` (`contentTokenRange`, `skipToContent`, `dropPending`). `/v1/completions` logprobs is an INTEGER + four arrays. Guard: `tests/test_logprobs.sh`.
- **A client stop string ends the ANSWER, never the reasoning** (#549, `chat.answerStopIndex`): a match counts only where the split delivers it as content, judged on the text UP TO the match, so stream and non-stream cut the same byte. Guard: corpus `a stop string ends the answer`.
- **Stream and non-stream are the SAME BYTES**; only an all-whitespace lead chunk may be withheld (`streamContentLead`). Also agree: spent budget WITHHOLDS the rest; tool replies carry `visibleToolPreamble`; disconnect = `client_disconnect`; stop sequences cut at INDEX, a tail that may still begin one is HELD (`StopStream`).
- **A client cannot time our stream**: final-chunk server `timings`; the `include_usage` chunk ships `"choices": []`. Guard: `tests/test_loop_stop_signal.sh`.
- **Liveness is a property of the SOCKET** (`beatStreamKeepalive`, 5 s byte-silence; `StreamHeartbeat` mirrors `StallClock`; a media job's CPU stage beats via `gen_sse.offload`). GAP: Ollama sink drops SSE comments. `--timeout` is a STALL timeout; `toolCallFinishReason` preserves "length".
- **A media handler's LAZY result evaluates at the encode** (`gen.imageJobFailed`): the denoise ran inside `toPng`, whose bare `try` escaped the failure arm and closed the socket on a Metal OOM; every failure after the guarded call goes through the same arm (503 for memory). The 3D job READS its stages on the host, so a Metal failure returned a zeroed GLB as a 200: `latchedJobFailure` checks the latch after the shape stage and after the job.
- **A long job's abort must not depend on the RESPONSE SHAPE** (`gen_sse.StreamCtx.stream`, `Conn.peerClosed`).
- **`Conn.close` sends FIN, then waits for the PEER to hang up (≤ 5 min) when the body is close-delimited** (SSE, NDJSON: no `Content-Length`): closed under a lagging reader, the kernel's `fin_timeout` RSTs it and the client gets ECONNRESET after every byte. A response written with a `Content-Length` sets `Conn.length_framed` and closes at once; an unmarked head keeps the wait. Guard: `tests/test_responses_streaming.sh` [G].
- **Grammar**: every state has a legal byte; the mask never walks the vocabulary (#380, `token_mask.buildMask`, `nextConstrained` lazy); NO whitespace OUTSIDE the root, the model's own layout inside (`MAX_FREE_WS` 16); token→bytes tables are per-MODEL (`grammarTokenBytes`). Guard: `tests/test_json_mode_multi_model.sh`.

LAN + console:
- **LAN**: per-INTERFACE dns_sd callbacks, loopback-first fetches, eviction via `attemptKnown`; proxying bounded by the TUNNEL MARKER (`isTunneledRequest` at gate AND dispatch, ONE hop, `error.SelfFetch`).
- **Console** (`app-web/`, Svelte): `src/html/index.html` is BUILT, never hand-edited: change the source, `npm run build`, commit both. Rules + guards: `app-web/CLAUDE.md`; `tests/test_index_page.sh` serves it headless. Metrics are on when `GET /metrics.json` is not 503.
