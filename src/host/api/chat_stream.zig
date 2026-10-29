//! StreamMode, StreamingTokenStream; handleStreamingGeneration

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const kv_quant = @import("../../sdk/blocks/attention/kv_quant.zig");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const generate_mod = @import("../../sdk/request/request.zig");
const chat_mod = @import("../chat/chat.zig");
const rp_mod = @import("../../sdk/grammar/protocol.zig");
const log = @import("../../sdk/util/log.zig");
const MediaSpan = @import("../../sdk/kv/media_span.zig").MediaSpan;
const mlx_engine = @import("../engines/mlx.zig");
const Stream = @import("../engines/stream.zig").Stream;
const model_registry_mod = @import("../registry/registry.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const LoadedModel = model_registry_mod.LoadedModel;
const io_util = @import("../../sdk/util/io_util.zig");
const nowMs = io_util.nowMs;
const testing = std.testing;
const http_server = @import("../http/server.zig");

const Conn = @import("../http/conn.zig").Conn;
const resolveKvAttnFused = @import("request_options.zig").resolveKvAttnFused;
const mtpCapable = @import("request_options.zig").mtpCapable;
const getTimeoutNs = @import("../http/server.zig").getTimeoutNs;
const decodeTokens = @import("tokenize.zig").decodeTokens;
const promptOpensThink = @import("../chat/reasoning/request.zig").promptOpensThink;
const promptOpensMuseHeader = @import("../chat/reasoning/request.zig").promptOpensMuseHeader;
const reasoningDialect = @import("../chat/reasoning/request.zig").reasoningDialect;
const toolCallFinishReason = @import("../chat/tools/pipeline.zig").toolCallFinishReason;
const streamToolCallDelta = @import("../chat/tools/pipeline.zig").streamToolCallDelta;
const shouldParseToolCalls = @import("../chat/tools/pipeline.zig").shouldParseToolCalls;
const appendRawToolDump = @import("../chat/tools/pipeline.zig").appendRawToolDump;
const parseToolCallsForRequest = @import("../chat/tools/pipeline.zig").parseToolCallsForRequest;
const slotFailure = @import("../http/errors.zig").slotFailure;
const StopStream = @import("../chat/stops.zig").StopStream;
const emitConstrainedChat = @import("chat_completions.zig").emitConstrainedChat;
const requestHasCompany = @import("request_options.zig").requestHasCompany;
const pickStreamMode = @import("request_options.zig").pickStreamMode;
const StreamLogprobs = @import("logprobs.zig").StreamLogprobs;
const sendSSEChunk = @import("../http/sse.zig").sendSSEChunk;
const sendSSEUsageChunk = @import("../http/sse.zig").sendSSEUsageChunk;
const SSE_ALLOW_HEADERS_DEFAULT = @import("../http/respond.zig").SSE_ALLOW_HEADERS_DEFAULT;
const sendSseHeaders = @import("../http/respond.zig").sendSseHeaders;
const logHttpSseData = @import("../http/respond.zig").logHttpSseData;
const utf8TrailingIncomplete = @import("format.zig").utf8TrailingIncomplete;
const formatChatUsage = @import("format.zig").formatChatUsage;
const formatTimingsObject = @import("format.zig").formatTimingsObject;
const formatPerfBracket = @import("format.zig").formatPerfBracket;
const MropeData = @import("../media/encode.zig").MropeData;
const sendStreamKeepalive = @import("../http/sse.zig").sendStreamKeepalive;
const beatStreamKeepalive = @import("../http/sse.zig").beatStreamKeepalive;

/// Token-stream adapter that drives the streaming SSE state machine the same
/// way regardless of whether speculative decoding (PLD / drafter) is on. Each
/// `next()` call yields exactly one token (or null on EOS / max_tokens), so
/// the per-token state machine in `handleStreamingGeneration` /
/// `handleAnthropicStreaming` / `handleResponses` does not need to know
/// anything about multi-token batches that PLD and drafter emit per step.
///
/// EOS-in-batch behavior matches the non-streaming `generatePld` /
/// `generateDrafter`: the stop token is NOT yielded — the loop just
/// terminates. This keeps tokens leaking past EOS impossible regardless of
/// where in an N-token batch the EOS lands.
///
/// `pld_drop_buf` is a heap-owned pending buffer (PLD can yield up to
/// `1+max_draft_len`=16 tokens per step; drafter yields up to `block_size`).
/// Caller must `deinit`.
pub const StreamMode = enum { regular, pld, drafter, mtp };

/// The streamed tokens of one request, read from its `Stream` one at a time. Post-generation stats are copied in by
/// `finalize()`.
pub const StreamingTokenStream = struct {
    stream: *Stream,
    mode: StreamMode,
    eos_token_ids: []const u32,
    finished: bool = false,

    prompt_tokens: u32 = 0,
    cached_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    finish_reason: []const u8 = "stop",
    /// Set only by the degenerate-tail guard. Rides the FINAL chunk beside `finish_reason`; the tokens themselves
    /// are already gone: a delta cannot be retracted, so streaming gets the signal, not the trim.
    finish_details: ?[]const u8 = null,
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,

    /// `mode` is recorded for telemetry: the engine already routed PLD/drafter/MTP.
    pub fn initFromStream(stream: *Stream, mode: StreamMode, eos: []const u32) StreamingTokenStream {
        return .{ .stream = stream, .mode = mode, .eos_token_ids = eos };
    }

    pub fn deinit(_: *StreamingTokenStream, _: std.mem.Allocator) void {}

    /// Copies the request's numbers out of the stream; idempotent.
    pub fn finalize(self: *StreamingTokenStream) void {
        const s = self.stream;
        self.prompt_tokens = s.prompt_tokens;
        self.cached_tokens = s.cached_tokens;
        self.completion_tokens = s.completion_tokens;
        self.finish_reason = s.finish_reason;
        self.finish_details = s.finish_details;
        self.prefill_ns = s.prefill_ns;
        self.decode_ns = s.decode_ns;
    }

    /// The finalized stream as a `GenerationResult`, for surfaces that share
    /// their post-generation tail with the non-streaming path.
    pub fn generationResult(self: *const StreamingTokenStream, text: []u8, token_ids: []u32, finish_reason: []const u8, constraint_payload_byte: ?usize) generate_mod.GenerationResult {
        return .{
            .text = text,
            .token_ids = token_ids,
            .prompt_tokens = self.prompt_tokens,
            .completion_tokens = self.completion_tokens,
            .finish_reason = finish_reason,
            .prefill_tps = generate_mod.prefillTokensPerSec(self.prompt_tokens, self.cached_tokens, self.prefill_ns),
            .decode_tps = generate_mod.tokensPerSec(self.completion_tokens, self.decode_ns),
            .prefill_ns = self.prefill_ns,
            .decode_ns = self.decode_ns,
            .cached_tokens = self.cached_tokens,
            .finish_details = self.finish_details,
            .constraint_payload_byte = constraint_payload_byte,
        };
    }

    const NextOrIdle = union(enum) { token: u32, done, idle };

    /// The next token, the end, or `.idle` after `timeout_ms` with no progress so the caller can poll the peer
    /// socket and send SSE keepalives during a long prefill.
    pub fn nextOrIdle(self: *StreamingTokenStream, _: std.mem.Allocator, timeout_ms: i64) !NextOrIdle {
        if (self.finished) return .done;
        const nr = self.stream.waitNextTimeout(timeout_ms) orelse return .idle;
        switch (nr) {
            .token => |t| return .{ .token = t },
            .done => {
                self.finished = true;
                return .done;
            },
            .err => return slotFailure(self.stream),
        }
    }
};

pub fn handleStreamingGeneration(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    include_usage: bool,
    has_tools: bool,
    /// OpenAI-shape tools JSON (for bare-args tool-call inference); null when
    /// the request defined no tools.
    tools_json: ?[]const u8,
    /// false = client set parallel_tool_calls:false (Anthropic:
    /// tool_choice.disable_parallel_tool_use) — at most one call per response.
    allow_parallel_tools: bool,
    logprobs_n: u32,
    enable_thinking: bool,
    reasoning_budget: i32,
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    vision_embeddings: ?mlx.mlx_array,
    media: []const MediaSpan,
    cache_key: u64,
    mrope: MropeData,
    /// Wave 1.A: per-request KV-quant override; null = inherit scheduler default.
    kv_quant_override: ?kv_quant.KVQuantConfig,
    kv_attn_explicit: ?bool,
    /// Iteration 1: tokenize_ns measured by the request handler before
    /// dispatching here. Surfaced via `timings.tokenize_ms` on the final
    /// usage SSE chunk so streaming clients see the same metric as
    /// non-streaming.
    tokenize_ns: u64,
) !void {
    // Vision array ownership: held by this handler on entry, transfers to
    // the slot on submit (slot.deinit frees). Nulled before transfer so
    // the early-return defer is a no-op.
    var ve_local = vision_embeddings;
    defer {
        if (ve_local) |arr| _ = mlx.mlx_array_free(arr);
    }

    const config = lm.config.?;
    const chat_id = nowMs(stream.io);

    // Template-opened think block (Qwen 3.5/3.6): unclosed buffered output is
    // reasoning, never content (mirrors the non-streaming split policy).
    // `prompt_opened_think` drops the `enable_thinking` conjunct on purpose:
    // whether the prompt ends inside a think block is a property of the
    // RENDERED BYTES. LFM2.5 pre-opens `<think>` unconditionally, so with
    // thinking off its reasoning is tag-free prose the gate flushed as the
    // visible answer. Families that gate the opener on
    // enable_thinking render the closed signature and stay false here.
    const prompt_opened_think = promptOpensThink(allocator, lm, tok, prompt_ids);
    const opens_think = prompt_opened_think;

    // Pick the speculative-decoding mode (regular / PLD / drafter). The
    // per-token state machine below is driven by `StreamingTokenStream`,
    // which feeds `next` (regular), `nextPld` (1..1+draft_len tokens/step),
    // or `nextDrafter` (1..block_size tokens/step) through the same
    // one-token-at-a-time interface.
    const stream_mode = pickStreamMode(enable_pld, enable_drafter, enable_mtp, lm.drafter != null, lm.dflash != null, mtpCapable(lm), sampling.shapesLogits(), logprobs_n, requestHasCompany());
    if (stream_mode == .pld) log.info("  pld=enabled (streaming, draft_len={d}, key_len={d})\n", .{ http_server.serverState().server_config.default_pld_draft_len, http_server.serverState().server_config.default_pld_key_len });
    if (stream_mode == .drafter) log.info("  drafter=enabled (streaming, block_size={d})\n", .{lm.drafter_block_size});
    if (stream_mode == .mtp) log.info("  mtp=enabled (streaming, depth={d})\n", .{lm.mtp_depth});

    // Scheduler's inference thread runs prefill + per-tick decode (regular
    // / PLD / drafter) and pushes generated tokens into the slot's output
    // ring; this thread reads via `slot.waitNext` through the
    // `StreamingTokenStream.initFromSlot` adapter.
    var running: ?mlx_engine.Running = null;
    defer if (running) |r| r.done();

    // Transfer vision ownership into the slot.
    const slot_ve_s = ve_local;
    ve_local = null;
    const sch = http_server.launch().global_scheduler.?;
    running = try mlx_engine.Running.submit(sch, .{
        .model = lm,
        .prompt_ids = prompt_ids,
        .full_prompt = prompt_ids,
        .cached_tokens = 0,
        .has_tools = has_tools,
        .enable_thinking = enable_thinking,
        .sampling = sampling,
        .eos_token_ids = eos_token_ids,
        .max_tokens = max_tokens,
        .timeout_ns = getTimeoutNs(),
        .enable_pld = stream_mode == .pld,
        .enable_drafter = stream_mode == .drafter,
        .drafter = if (stream_mode == .drafter) lm.drafter else null,
        .dflash = if (stream_mode == .drafter) lm.dflash else null,
        .drafter_block_size = lm.drafter_block_size,
        .allow_batch_mtp = allow_batch_mtp,
        .enable_mtp = stream_mode == .mtp,
        .mtp = if (stream_mode == .mtp) lm.mtp else null,
        .mtp_depth = lm.mtp_depth,
        .pld_draft_len = http_server.serverState().server_config.default_pld_draft_len,
        .pld_key_len = http_server.serverState().server_config.default_pld_key_len,
        .kv_attn_fused = resolveKvAttnFused(config, kv_attn_explicit, prompt_ids.len, kv_quant_override),
        .logprobs_n = logprobs_n,
        .vision_embeddings = slot_ve_s,
        .media = media,
        .cache_key = cache_key,
        .client = stream.client,
        .mrope_pos = mrope.pos,
        .mrope_total = mrope.total,
        .mrope_delta = mrope.delta,
        .kv_quant_config = kv_quant_override,
    });
    var ts = StreamingTokenStream.initFromStream(running.?.stream, stream_mode, eos_token_ids);
    defer ts.deinit(allocator);

    // Send SSE headers (no Content-Length — we stream until done)
    try sendSseHeaders(stream, "chat.completions", SSE_ALLOW_HEADERS_DEFAULT);

    // Logprobs ride the chunks beside their deltas. Requesting them already
    // forced this stream off every speculative path (see `pickStreamMode`), so
    // the cost is paid whether or not they are delivered — dropping them was
    // the worst of both.
    var lps = StreamLogprobs{
        .allocator = allocator,
        .tok = tok,
        .stream = running.?.stream,
        .enabled = logprobs_n > 0,
    };
    defer lps.deinit();

    // First chunk: role announcement
    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = "assistant", .content = "" }, null, null, null, .{ .logprobs_json = try lps.take() });

    // Buffer for stop sequence and tool call detection
    var text_buf = std.ArrayList(u8).empty;
    var stop_stream: StopStream = .{};
    defer stop_stream.deinit(allocator);
    defer text_buf.deinit(allocator);
    // Memoized marker scan for the think gate. Owned BESIDE text_buf and reset
    // with it — the gate is otherwise O(buffer) per token for as long as a
    // think block is open and unclosed (47.99 us/token at 113 KB).
    var think_scan: chat_mod.ThinkScan = .{};
    // When tools are present, buffer individual token texts for deferred streaming
    var token_texts = std.ArrayList([]const u8).empty;
    defer {
        for (token_texts.items) |t| allocator.free(t);
        token_texts.deinit(allocator);
    }
    var stopped = false;
    var client_gone = false;

    // Buffer for incomplete UTF-8 sequences split across BPE tokens
    var utf8_carry: [3]u8 = undefined;
    var utf8_carry_len: u8 = 0;

    // Thinking state for real-time streaming of reasoning_content vs content
    // Supports both <think>...</think> and Gemma 4's <|channel>thought\n...<channel|>
    // Starts true when thinking is enabled (model outputs <think> first) OR
    // the prompt itself opened a think block: generated reasoning is always
    // DELIVERED as reasoning_content, never paid-for-and-dropped. Thinking-off
    // is enforced prompt-side (chat.noThinkTailSuffix commits the channel).
    // A stream starts inside a think block only when the RENDERED PROMPT ends
    // inside one — never because the request asked for thinking. Qwen/Gemma
    // templates render the opener when thinking is on, so `promptOpensThink`
    // sees it; LFM2-VL's generation prompt is a bare `<|im_start|>assistant`
    // and its model answers directly, so seeding from the flag routed the whole
    // answer into reasoning_content and left `content` empty.
    // A model that opens the block itself is picked up by `saw_think_open`.
    var channel_armed = false; // the `<|channel>` marker that may open a Gemma 4 thought
    const gated_stream = has_tools or (sampling.constraint == null and reasoningDialect(config) == .harmony);
    // Constrained requests route through the protocol, not the response-side
    // close scan. The stream seeds the think arm while the payload has not
    // begun (a generated opener must strip, not leak), and flips straight to
    // the content arm when the generator reports a DIRECT answer — the whole
    // point is that no reasoning block exists to scan for.
    const constrained_proto = sampling.constraint != null and sampling.constraint.?.proto != null;
    var in_think_block = prompt_opened_think and !constrained_proto;
    var delivery: ?rp_mod.Delivery = if (constrained_proto) rp_mod.Delivery.init(sampling.constraint.?.proto.?) else null;
    defer if (delivery) |*d| d.deinit(allocator);
    var think_closed = false; // a complete think block was already split+emitted this stream
    // Leading whitespace is suppressed until the first visible byte, so the
    // stream reaches the same content bytes as splitThinkBlock's own
    // trimStart (chat.streamContentLead).
    var content_started = false;
    var think_buf = std.ArrayList(u8).empty; // buffer to detect close tag across token boundaries
    defer think_buf.deinit(allocator);
    var think_close_tag: []const u8 = "</think>"; // will be updated if Gemma 4 format detected
    var skipped_think_open = false; // track if we've skipped the initial think tag
    // Positive evidence the model itself emitted a think OPENER. Distinct
    // from `skipped_think_open`, whose else-branch also fires for "no known
    // opener — the template must have injected one", which is the very case
    // that has to be told apart at the end-of-stream flush below.
    var saw_think_open = false;
    // Muse-Glimmer: dropping a segment header in the plain arm — <|start|>
    // arms it, <|message|> disarms; the role+recipient text between them is
    // ordinary tokens that must never reach the client as content.
    var muse_skip_header = promptOpensMuseHeader(allocator, lm, tok, prompt_ids);
    var muse_head = std.ArrayList(u8).empty; // header bytes held while skipping
    defer muse_head.deinit(allocator);
    // Bytes of THIS turn's reasoning already streamed. The tools path emits the
    // thought incrementally now, so every later emit site sends the remainder.
    var reasoning_streamed: usize = 0;
    var reasoning_tokens_sent: usize = 0; // reasoning deltas actually emitted (the budget is counted in these)
    var think_tokens: i32 = 0; // count of tokens generated in think block
    var budget_exhausted = false; // true when reasoning budget hit

    // Generate tokens via the adapter — yields one decoded token id per call
    // regardless of whether the underlying decode is regular, PLD, or drafter.
    while (true) {
        // A stop cut resolved on the previous token ends the turn here.
        if (stopped or stop_stream.flushed) break;
        const next_id: ?u32 = switch (try ts.nextOrIdle(allocator, Conn.STREAM_KEEPALIVE_MS)) {
            .token => |t| t,
            .done => if (stop_stream.holding()) null else break,
            .idle => {
                // No tokens yet (long prefill). Probe the peer: an abandoned
                // request must cancel instead of grinding a ghost prefill
                // (Claude Code retries pile up serially otherwise), and the
                // keepalive stops clients timing out on stream silence.
                if (stream.peerClosed()) {
                    log.info("  [cancel] client disconnected while waiting for tokens — cancelling slot\n", .{});
                    running.?.stream.cancel();
                    client_gone = true;
                    break;
                }
                sendStreamKeepalive(stream) catch {
                    log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
                    running.?.stream.cancel();
                    client_gone = true;
                    break;
                };
                continue;
            },
        };
        if (stream.peerClosed()) {
            running.?.stream.cancel();
            client_gone = true;
            break;
        }
        // Prepend any carried-over bytes from a previous incomplete UTF-8 sequence,
        // then strip any new trailing incomplete bytes into the carry buffer.
        var token_text = if (next_id) |token_id| blk: {
            try lps.note(token_id);
            const strip = tok.tok_type == .sentencepiece_bpe;
            const raw_decoded = try decodeTokens(allocator, lm, tok, &[_]u32{token_id}, strip and false);
            if (delivery) |*d| d.noteToken(raw_decoded.len, running.?.stream.constraintPayloadStart());
            // Step 1: prepend carry-over from previous token
            const with_carry = if (utf8_carry_len > 0) cc: {
                const combined = try allocator.alloc(u8, utf8_carry_len + raw_decoded.len);
                @memcpy(combined[0..utf8_carry_len], utf8_carry[0..utf8_carry_len]);
                @memcpy(combined[utf8_carry_len..], raw_decoded);
                allocator.free(raw_decoded);
                utf8_carry_len = 0;
                break :cc combined;
            } else raw_decoded;

            // Step 2: check for trailing incomplete UTF-8 sequence
            const tail = utf8TrailingIncomplete(with_carry);
            if (tail > 0) {
                @memcpy(utf8_carry[0..tail], with_carry[with_carry.len - tail ..]);
                utf8_carry_len = @intCast(tail);
            }

            // Step 3: if everything was incomplete, skip this iteration
            if (with_carry.len == tail) {
                allocator.free(with_carry);
                continue;
            }

            // Step 4: if we trimmed trailing bytes, reallocate to the complete prefix
            if (tail > 0) {
                const trimmed = try allocator.dupe(u8, with_carry[0 .. with_carry.len - tail]);
                allocator.free(with_carry);
                break :blk trimmed;
            }

            break :blk with_carry;
        } else try stop_stream.takeHeld(allocator);

        // Accumulate for stop sequence and tool call detection

        // The stop cut is an INDEX, not a token boundary: bytes before the match still go out.
        if (stop_sequences.len > 0) {
            const fed = try stop_stream.feed(allocator, &text_buf, token_text, stop_sequences, prompt_opened_think and !constrained_proto and !think_closed);
            allocator.free(token_text);
            token_text = fed.send;
            if (fed.stop != null) stopped = true;
            if (token_text.len == 0) {
                allocator.free(token_text);
                continue;
            }
        } else if (gated_stream) try text_buf.appendSlice(allocator, token_text);
        lps.sent(token_text.len, utf8_carry_len);

        if (delivery) |*d| {
            defer allocator.free(token_text);
            try d.feed(allocator, token_text);
            try emitConstrainedChat(allocator, stream, chat_id, model_name, &lps, d);
            try beatStreamKeepalive(stream, .sse_comment);
            continue;
        }

        if (gated_stream) {
            // Stream tokens until we detect a tool call pattern starting, then buffer.
            // Detection rules live in `chat.streamShouldBufferForTools` — it
            // covers the full `<tool…>` family (including bare DSV4 `<tool>`),
            // Gemma 4 `<|tool_call`, raw JSON, plus partial-prefix growth from
            // a single `<` all the way to `<|tool_cal`. Without the partial
            // coverage, a `<tool` single-BPE-token leaks as visible content
            // (the bug surfaced on DSV4 where the model emits `<tool` then
            // gets stuck looping the partial).
            try token_texts.append(allocator, token_text);
            // text_buf already updated above

            const buf = text_buf.items;
            const maybe_tool = chat_mod.streamShouldBufferForTools(buf);

            if (!maybe_tool) {
                // No tool call pattern — ask the shared gate (chat.streamThinkGate,
                // also used by /v1/messages) whether the buffer is thinking that
                // must be held, a completed think block to split, or visible
                // prose to flush. Hermetically pinned per recorded model family
                // by the format corpus streaming-gate test.
                switch (chat_mod.streamThinkGateScan(buf, enable_thinking, think_closed, prompt_opened_think, &think_scan)) {
                    .hold_thinking => {
                        // Stream the thought AS IT ARRIVES instead of holding the
                        // whole block. Safe here by construction:
                        // `streamShouldBufferForTools` returned false just above,
                        // so the buffer carries no tool markup and no partial
                        // marker at its tail — these bytes are reasoning and
                        // nothing else. Without it a tools request shows NOTHING
                        // until `</think>` (measured 4.4 s on a 2.6B, and it grows
                        // with the thought) while the no-tools path is already
                        // streaming at 0.05 s.
                        //
                        // A reasoning BUDGET is a cap, not a reason to hide the
                        // thought: you never exceed a cap you STOP emitting at.
                        // This was gated on `reasoning_budget < 0`, so a capped
                        // tools request showed nothing for the whole generation
                        // and got one dump at the end — a capped agent session
                        // looked frozen (live, pi on Qwen3.8-27B).
                        // Each loop iteration is one model token, so an
                        // iteration that produced fresh reasoning IS one
                        // reasoning token; at the cap the latch stops every
                        // later emitter from shipping the remainder.
                        if (!budget_exhausted) {
                            const so_far = chat_mod.splitThinkBlock(buf, true, opens_think);
                            if (so_far.reasoning_content) |split_rc| {
                                const rc = chat_mod.streamableReasoning(split_rc);
                                if (chat_mod.unstreamedReasoning(rc, reasoning_streamed)) |fresh| {
                                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = fresh }, null, null, null, .{});
                                    reasoning_streamed = rc.len;
                                    reasoning_tokens_sent += 1;
                                    if (reasoning_budget >= 0 and reasoning_tokens_sent >= @as(usize, @intCast(reasoning_budget))) {
                                        budget_exhausted = true;
                                        log.info("  reasoning budget exhausted ({d}/{d} tokens, streamed)\n", .{ reasoning_tokens_sent, reasoning_budget });
                                    }
                                }
                            }
                        }
                    },
                    .split_think => {
                        // Complete thinking block — split into reasoning + content
                        const split = chat_mod.splitThinkBlock(buf, true, opens_think);
                        // The thought's tokens never reach the client, so their
                        // entries must not ride the content chunk below — the
                        // reasoning emitters above deliberately do not drain.
                        // Empty content means the block closed with nothing
                        // after it: everything buffered so far was reasoning,
                        // and leaving it pending hands it to the NEXT chunk
                        // (measured: 42 entries on a 1-char delta).
                        if (split.content.len > 0) lps.skipToContent(buf, split.content) else lps.dropPending();
                        for (token_texts.items) |tt| allocator.free(tt);
                        token_texts.clearRetainingCapacity();
                        text_buf.clearRetainingCapacity();
                        think_scan.reset();
                        think_closed = true;
                        // Past the cap the tail is exactly what the budget
                        // exists to withhold. This arm applied no budget at all,
                        // so a capped request received the WHOLE thought the
                        // moment the block closed.
                        if (!budget_exhausted) {
                            if (split.reasoning_content) |rc| {
                                if (chat_mod.unstreamedReasoning(rc, reasoning_streamed)) |fresh| {
                                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = fresh }, null, null, null, .{});
                                }
                            }
                        }
                        // Block done and `text_buf` was just cleared — a thought
                        // re-opened later in the turn starts counting from zero.
                        reasoning_streamed = 0;
                        if (split.content.len > 0) {
                            content_started = true;
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = split.content }, null, null, null, .{ .logprobs_json = try lps.take() });
                        }
                    },
                    .flush_text => {
                        // Freed here, not per item: a failed send returns mid-loop, and the
                        // handler's own defer would free the already-sent items a second time.
                        defer {
                            for (token_texts.items) |tt| allocator.free(tt);
                            token_texts.clearRetainingCapacity();
                        }
                        for (token_texts.items) |tt| {
                            // Skip bare channel/think tags that leak without a full block
                            if (chat_mod.isChannelMarkerToken(tt)) {
                                continue;
                            }
                            const vis = chat_mod.streamContentLead(tt, content_started);
                            // Suppressed lead: nothing pending describes text the
                            // client received, so retire it rather than letting it
                            // ride the next chunk (logprobs.content ↔ content).
                            if (vis.len == 0) {
                                lps.dropPending();
                                continue;
                            }
                            content_started = true;
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = vis }, null, null, null, .{ .logprobs_json = try lps.take() });
                        }
                    },
                }
            }
            // Otherwise keep buffering — tool call may be in progress
        } else if (in_think_block) {
            // Inside <think> block — stream as reasoning_content with </think>
            // detection. Also covers thinking-off + template-opened (LFM2.5
            // renders `…assistant\n<think>` unconditionally, live):
            // the block used to be buffered and DROPPED there; now it streams
            // as reasoning_content like any other thought — generated tokens
            // are always delivered.
            defer allocator.free(token_text);
            try think_buf.appendSlice(allocator, token_text);
            think_tokens += 1;

            // Skip the initial think tag prefix (<think> or <|channel>thought\n).
            // Many templates (e.g. Qwen 3.5/3.6, some Gemma 4 variants) pre-inject
            // the opener into the prompt so the model's first tokens are already
            // INSIDE the thinking block — no opener appears in the streamed text.
            if (!skipped_think_open and (think_buf.items.len >= 7 or chat_mod.cannotOpenThink(think_buf.items))) {
                if (chat_mod.thinkOpenTagLenAt(think_buf.items)) |olen| {
                    // Remove the opener (<think> or the Hy3-suffixed form) and
                    // any leading newline.
                    saw_think_open = true;
                    var skip: usize = olen;
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                } else if (std.mem.startsWith(u8, think_buf.items, "<|content_thinking|>")) {
                    // Inkling thinking message (opener is ONE special token) —
                    // the message closes with <|end_message|>.
                    saw_think_open = true;
                    think_close_tag = "<|end_message|>";
                    var skip: usize = "<|content_thinking|>".len;
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                } else if (chat_mod.harmonyThinkOpenerAt(think_buf.items) == .analysis) {
                    // gpt_oss: `[<|start|>assistant]<|channel|>analysis<|message|>`
                    // opens a reasoning segment that closes at <|end|>. Consume
                    // the header — every byte of it is ordinary text, and the
                    // close matcher below assumes reasoning starts at byte 0.
                    const skip = switch (chat_mod.harmonyThinkOpenerAt(think_buf.items)) {
                        .analysis => |hl| hl,
                        else => unreachable,
                    };
                    saw_think_open = true;
                    think_close_tag = "<|end|>";
                    var skip2 = skip;
                    while (skip2 < think_buf.items.len and think_buf.items[skip2] == '\n') skip2 += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip2..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                } else if (chat_mod.harmonyThinkOpenerAt(think_buf.items) == .growing) {
                    // Harmony header still arriving — wait, or `<|channel|>anal`
                    // leaks as reasoning.
                } else if (chat_mod.museThinkOpenerAt(think_buf.items) == .self_opened) {
                    // Muse-Glimmer reasoning segment (` to=self<|message|>`) —
                    // closes at <|eom|>. The direct-answer header
                    // (` to=user<|message|>`) is handled by the close matcher
                    // below (immediate close, empty reasoning), and .growing
                    // deliberately falls through to wait for more tokens.
                    const skip = switch (chat_mod.museThinkOpenerAt(think_buf.items)) {
                        .self_opened => |hl| hl,
                        else => unreachable,
                    };
                    saw_think_open = true;
                    think_close_tag = "<|eom|>";
                    var skip2 = skip;
                    while (skip2 < think_buf.items.len and think_buf.items[skip2] == '\n') skip2 += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip2..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                } else if (chat_mod.museThinkOpenerAt(think_buf.items) == .growing) {
                    // Muse header still arriving token by token — wait before
                    // deciding, or the ` to=self` text leaks as reasoning.
                } else if (think_buf.items.len >= 17 and std.mem.startsWith(u8, think_buf.items, "<|channel>thought")) {
                    // Gemma 4 think format — switch close tag
                    saw_think_open = true;
                    think_close_tag = "<channel|>";
                    var skip: usize = 17; // len of "<|channel>thought"
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                } else if (think_buf.items.len < 17 and std.mem.startsWith(u8, "<|channel>thought", think_buf.items)) {
                    // Buffer is still a partial prefix of `<|channel>thought` —
                    // wait for more tokens before deciding.
                } else if (think_buf.items.len < 32 and chat_mod.endsWithPartialThinkOpen(think_buf.items)) {
                    // Still a growing suffixed opener (`<think:opensou`) —
                    // wait before deciding, or the raw tag leaks as reasoning.
                } else {
                    // Not a known opener — template already injected one.
                    // Stay inside the think block; close tag is detected dynamically below.
                    skipped_think_open = true;
                }
            }

            // A spent budget caps DELIVERY only: the tail stays withheld reasoning until the close tag.
            if (!budget_exhausted and reasoning_budget >= 0 and think_tokens >= reasoning_budget and skipped_think_open) {
                budget_exhausted = true;
                log.info("  reasoning budget exhausted ({d}/{d} tokens)\n", .{ think_tokens, reasoning_budget });
            }

            // Check for the close tag — accept whichever appears first.
            // Models with templates that pre-inject the opener (Qwen 3.5/3.6,
            // some Gemma 4 variants, Hy3's suffixed form) don't reveal which
            // format they use until the close tag arrives, so look for both
            // families (indexOfThinkCloseTag covers bare AND suffixed </think…>).
            const think_match = chat_mod.indexOfThinkCloseTag(think_buf.items, 0);
            const channel_pos = std.mem.indexOf(u8, think_buf.items, "<channel|>");
            // Inkling: an Inkling thinking message closes at <|end_message|>
            // (gated on the opener having switched think_close_tag so prose
            // that MENTIONS the literal in another family never matches), and
            // a leading <|content_text|> means the model answered DIRECTLY —
            // close immediately with empty reasoning.
            const inkling_pos: ?usize = blk_i: {
                if (std.mem.startsWith(u8, think_buf.items, "<|content_text|>")) break :blk_i 0;
                if (std.mem.eql(u8, think_close_tag, "<|end_message|>")) {
                    if (std.mem.indexOf(u8, think_buf.items, "<|end_message|>")) |p| break :blk_i p;
                }
                break :blk_i null;
            };
            const inkling_len: usize = if (inkling_pos != null and inkling_pos.? == 0 and std.mem.startsWith(u8, think_buf.items, "<|content_text|>")) "<|content_text|>".len else "<|end_message|>".len;
            // Muse: a resolved non-self header is a DIRECT answer — close at 0
            // with empty reasoning (the header bytes are the "tag"); a
            // reasoning segment closes at its <|eom|> (gated on the muse
            // opener having latched think_close_tag).
            const muse_close: ?struct { pos: usize, len: usize } = blk_m: {
                switch (chat_mod.museThinkOpenerAt(think_buf.items)) {
                    .direct => |hl| break :blk_m .{ .pos = 0, .len = hl },
                    else => {},
                }
                if (std.mem.eql(u8, think_close_tag, "<|eom|>")) {
                    if (std.mem.indexOf(u8, think_buf.items, "<|eom|>")) |p| break :blk_m .{ .pos = p, .len = "<|eom|>".len };
                }
                break :blk_m null;
            };
            // Harmony (gpt_oss): the analysis channel closes at <|end|> (or
            // <|return|>, if the model skipped straight to a return). Its
            // reasoning also has a HEADER prefix to drop, which the pos/len
            // shape can't express — `harmony_reason_from` below carries that.
            const harmony_close: ?struct { pos: usize, len: usize } = blk_h: {
                if (!std.mem.eql(u8, think_close_tag, "<|end|>")) break :blk_h null;
                if (std.mem.indexOf(u8, think_buf.items, "<|end|>")) |q|
                    break :blk_h .{ .pos = q, .len = "<|end|>".len };
                // The model can skip straight to the return without an <|end|>.
                if (std.mem.indexOf(u8, think_buf.items, "<|return|>")) |q|
                    break :blk_h .{ .pos = q, .len = "<|return|>".len };
                break :blk_h null;
            };
            const close_match: ?struct { pos: usize, len: usize, is_channel: bool } = blk: {
                if (harmony_close) |m| break :blk .{ .pos = m.pos, .len = m.len, .is_channel = false };
                if (muse_close) |m| break :blk .{ .pos = m.pos, .len = m.len, .is_channel = false };
                if (inkling_pos) |p| break :blk .{ .pos = p, .len = inkling_len, .is_channel = false };
                if (think_match == null and channel_pos == null) break :blk null;
                if (think_match == null) break :blk .{ .pos = channel_pos.?, .len = "<channel|>".len, .is_channel = true };
                if (channel_pos == null) break :blk .{ .pos = think_match.?.pos, .len = think_match.?.len, .is_channel = false };
                if (think_match.?.pos <= channel_pos.?) break :blk .{ .pos = think_match.?.pos, .len = think_match.?.len, .is_channel = false };
                break :blk .{ .pos = channel_pos.?, .len = "<channel|>".len, .is_channel = true };
            };

            if (close_match) |m| {
                if (m.pos > 0 and !budget_exhausted) {
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items[0..m.pos] }, null, null, null, .{});
                }
                const after = m.pos + m.len;
                var content_after = std.mem.trimStart(u8, think_buf.items[after..], "\n ");
                // Strip Gemma 4 content channel tag: <|channel>\n or <|channel>
                if (std.mem.startsWith(u8, content_after, "<|channel>\n")) {
                    content_after = content_after[11..];
                } else if (std.mem.startsWith(u8, content_after, "<|channel>")) {
                    content_after = content_after[10..];
                }
                // Strip Inkling content-message markers after the close; the
                // content message ends at its own <|end_message|>.
                if (std.mem.startsWith(u8, content_after, "<|message_model|>")) content_after = content_after["<|message_model|>".len..];
                if (std.mem.startsWith(u8, content_after, "<|content_text|>")) content_after = content_after["<|content_text|>".len..];
                // Harmony: the answer arrives as a whole new segment —
                // `<|start|>assistant<|channel|>final<|message|>` — and every
                // byte of that header is ordinary text that would otherwise
                // ride out as content.
                content_after = chat_mod.stripHarmonySegmentHeader(content_after);
                if (std.mem.indexOf(u8, content_after, "<|end_message|>")) |em| content_after = content_after[0..em];
                // Strip a muse next-segment header (<|start|>assistant
                // to=user<|message|>). A header still ARRIVING (start marker,
                // no message marker yet) arms the plain-arm skip instead —
                // its remaining tokens drop until <|message|> passes.
                if (chat_mod.museContentHeaderSkip(content_after)) |hl| {
                    content_after = content_after[hl..];
                } else if (std.mem.startsWith(u8, content_after, "<|start|>")) {
                    muse_skip_header = true;
                    content_after = "";
                }
                if (std.mem.indexOf(u8, content_after, "<|eot|>")) |et| content_after = content_after[0..et];
                content_after = std.mem.trimStart(u8, content_after, "\n ");
                if (content_after.len > 0) {
                    // The reasoning above this point never reaches the client,
                    // so its entries stop here rather than riding the answer.
                    lps.skipToContent(think_buf.items, content_after);
                    const vis_content_after = chat_mod.streamContentLead(content_after, content_started);
                    if (vis_content_after.len > 0) {
                        content_started = true;
                        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = vis_content_after }, null, null, null, .{ .logprobs_json = try lps.take() });
                    }
                } else {
                    lps.dropPending();
                }
                think_buf.clearRetainingCapacity();
                in_think_block = false;
                think_close_tag = if (m.is_channel) "<channel|>" else "</think>";
            } else if (skipped_think_open) {
                // Flush reasoning tokens that can't be part of a close tag.
                // Hold back a still-growing partial close tag (suffixed forms
                // like `</think:opensource>` included), then back the cut off
                // to a UTF-8 boundary — a flush cut mid-codepoint ships a lone
                // continuation byte in the delta JSON (live hy_v3
                // `"reasoning_content":"2\xc2"` — '²' split across deltas).
                var safe_len = think_buf.items.len - chat_mod.partialThinkCloseSuffixLen(think_buf.items);
                while (safe_len > 0 and safe_len < think_buf.items.len and (think_buf.items[safe_len] & 0xC0) == 0x80) safe_len -= 1;
                if (safe_len > 0) {
                    if (!budget_exhausted) {
                        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items[0..safe_len] }, null, null, null, .{});
                    }
                    const remaining = try allocator.dupe(u8, think_buf.items[safe_len..]);
                    think_buf.clearRetainingCapacity();
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                }
            }
        } else {
            defer allocator.free(token_text);
            // Muse segment-header skip: everything between <|start|> and
            // <|message|> is the role+recipient header, never content.
            const was_skipping = muse_skip_header;
            muse_skip_header = chat_mod.museHeaderSkipNext(muse_skip_header, token_text);
            if (was_skipping or muse_skip_header) {
                if (muse_skip_header) try muse_head.appendSlice(allocator, token_text);
                // `<|message|>` resolved a header the model wrote: `to=self` is reasoning.
                if (was_skipping and !muse_skip_header and !content_started and chat_mod.museHeaderOpensReasoning(muse_head.items)) {
                    in_think_block = true;
                    saw_think_open = true;
                    skipped_think_open = true;
                    think_close_tag = "<|eom|>";
                }
                if (!muse_skip_header) muse_head.clearRetainingCapacity();
                continue;
            }
            // Skip Gemma 4 channel tags that leak after thinking blocks
            // A think block the MODEL opens: seeded like a template-injected one.
            // Checked before the marker skip, which would swallow a bare `<think>`.
            if (!content_started) if (chat_mod.modelThinkOpener(channel_armed, token_text)) |opener| {
                channel_armed = false;
                in_think_block = true;
                saw_think_open = true;
                try think_buf.appendSlice(allocator, opener);
                continue;
            };
            if (chat_mod.isChannelMarkerToken(token_text)) {
                channel_armed = !content_started and std.mem.eql(u8, token_text, "<|channel>");
                continue;
            }
            channel_armed = false;
            const vis_token_text = chat_mod.streamContentLead(token_text, content_started);
            if (vis_token_text.len == 0) {
                lps.dropPending();
            } else {
                content_started = true;
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = vis_token_text }, null, null, null, .{ .logprobs_json = try lps.take() });
            }
        }

        // Every branch above may have written NOTHING (tool-call detection and
        // unclosed thinking blocks buffer silently, sometimes for minutes on a
        // one-shot whole-file `write_file`). Tokens flowing is not bytes
        // flowing — without this the client's idle-body timeout kills the
        // stream mid-generation.
        beatStreamKeepalive(stream, .sse_comment) catch {
            log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
            running.?.stream.cancel();
            client_gone = true;
            break;
        };
    }

    if (!client_gone) {
        if (delivery) |*d| {
            d.clearOutput();
            try d.finish(allocator);
            try emitConstrainedChat(allocator, stream, chat_id, model_name, &lps, d);
        }
    }

    // Flush any remaining think buffer
    if (!client_gone and think_buf.items.len > 0) {
        // No close tag arrived. Reasoning ONLY with positive evidence a block
        // was open — the prompt injected the opener, or the model emitted one.
        // `in_think_block` alone is not that evidence: it STARTS true whenever
        // thinking was requested, so a Gemma turn answered directly (its
        // template renders a bare `<|turn>model` and lets the model decide)
        // shipped its whole answer as reasoning with EMPTY content, while
        // non-streaming returned it correctly.
        if (chat_mod.streamTailIsReasoning(in_think_block, prompt_opened_think, saw_think_open)) {
            if (!budget_exhausted) {
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items }, null, null, null, .{});
            }
        } else {
            const vis_tail = chat_mod.streamContentLead(think_buf.items, content_started);
            if (vis_tail.len > 0) {
                content_started = true;
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = vis_tail }, null, null, null, .{ .logprobs_json = try lps.take() });
            }
        }
    }

    // After generation: capture stats from whichever source was active
    // (Generator on legacy, Slot on scheduler).
    ts.finalize();

    // Check for tool calls in accumulated text
    var finish_reason: []const u8 = if (client_gone) "client_disconnect" else if (stopped) "stop" else ts.finish_reason;
    if (gated_stream and !client_gone) {
        log.debug("  checking {d} bytes of streamed text for tool calls\n", .{text_buf.items.len});
        if (log.isDebug() and text_buf.items.len > 0) {
            log.debug("  raw generated text before tool parse ({d}b): {s}\n", .{ text_buf.items.len, text_buf.items[0..@min(text_buf.items.len, 4000)] });
            // Corpus-harvest aid: the inline dump caps at 4KB, useless for
            // >30KB mega-tool-calls. Set MLX_SERVE_RAW_DUMP_FILE=<abs path> to
            // APPEND the FULL pre-parse buffer of every tools request, framed so
            // a harvester can slice each record exactly (the text is arbitrary
            // bytes, so the byte count — not a delimiter — defines the record).
            if (env.get(.MLX_SERVE_RAW_DUMP_FILE)) |dump_path| {
                appendRawToolDump(std.mem.span(dump_path), tools_json, text_buf.items);
            }
        }
        // Merge re-opened mid-text thought channels into the leading block so
        // the split/parse below never leaks raw tags (Gemma 12B tail behavior).
        const norm_owned = try chat_mod.normalizeEmbeddedThinkBlocks(allocator, text_buf.items);
        defer if (norm_owned) |n| allocator.free(n);
        const gen_text: []const u8 = norm_owned orelse text_buf.items;
        const found_calls = if (has_tools and shouldParseToolCalls(ts.finish_details)) try parseToolCallsForRequest(allocator, gen_text, tools_json, allow_parallel_tools) else null;
        if (found_calls) |tool_calls| {
            defer {
                for (tool_calls) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.arguments);
                }
                allocator.free(tool_calls);
            }

            // Emit reasoning_content before tool calls — whatever the model
            // generated is delivered, thinking flag or not.
            {
                const think_split = chat_mod.splitThinkBlock(gen_text, true, opens_think and !think_closed);
                // `budget_exhausted` means the cap was already streamed in full.
                if (chat_mod.unstreamedReasoning(if (budget_exhausted) "" else think_split.reasoning_content orelse "", reasoning_streamed)) |reasoning| {
                    // Apply reasoning budget truncation if set
                    const final_reasoning = if (reasoning_budget >= 0) blk: {
                        const r_ids = try tok.encode(allocator, reasoning);
                        defer allocator.free(r_ids);
                        const budget_usize: usize = @intCast(reasoning_budget);
                        if (r_ids.len > budget_usize) {
                            const truncated = try tok.decode(allocator, r_ids[0..budget_usize], false);
                            defer allocator.free(truncated);
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = truncated }, null, null, null, .{});
                            break :blk @as(?[]const u8, null);
                        }
                        break :blk @as(?[]const u8, reasoning);
                    } else @as(?[]const u8, reasoning);
                    if (final_reasoning) |r| {
                        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = r }, null, null, null, .{});
                    }
                }
            }

            // Emit tool call deltas in OpenAI streaming format
            for (tool_calls, 0..) |tc, i| {
                const tc_id = try std.fmt.allocPrint(allocator, "call_{d}_{d}", .{ chat_id, i });
                defer allocator.free(tc_id);
                const first_delta = try streamToolCallDelta(allocator, i, tc_id, tc.name, tc.arguments);
                defer allocator.free(first_delta);
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .tool_calls_json = first_delta }, null, null, null, .{ .logprobs_json = try lps.take() });
            }
            finish_reason = toolCallFinishReason(finish_reason);
        } else {
            // No tool calls found — flush buffered tokens, splitting thinking
            // from content. Runs regardless of the request's thinking flag:
            // reasoning the model produced ships as reasoning_content, and
            // split.content already passes trimLeakedToolMarkup, so unparsed
            // wreckage that held the buffer (markers the tokenizer split
            // across tokens — live, DSV4 `<` then `｜DSML｜`) is
            // still cut once, at the end.
            {
                // Concatenate all buffered tokens and split thinking from content
                var full_text = std.ArrayList(u8).empty;
                defer full_text.deinit(allocator);
                for (token_texts.items) |t| {
                    try full_text.appendSlice(allocator, t);
                }
                const flush_norm = try chat_mod.normalizeEmbeddedThinkBlocks(allocator, full_text.items);
                defer if (flush_norm) |n| allocator.free(n);
                const flush_text: []const u8 = flush_norm orelse full_text.items;
                const think_split = chat_mod.splitThinkBlock(flush_text, true, opens_think and !think_closed);
                // `budget_exhausted` means the cap was already streamed in full.
                if (chat_mod.unstreamedReasoning(if (budget_exhausted) "" else think_split.reasoning_content orelse "", reasoning_streamed)) |reasoning| {
                    // Apply reasoning budget truncation if set
                    const final_reasoning = if (reasoning_budget >= 0) blk: {
                        const r_ids = try tok.encode(allocator, reasoning);
                        defer allocator.free(r_ids);
                        const budget_usize: usize = @intCast(reasoning_budget);
                        if (r_ids.len > budget_usize) {
                            const truncated = try tok.decode(allocator, r_ids[0..budget_usize], false);
                            defer allocator.free(truncated);
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = truncated }, null, null, null, .{});
                            break :blk @as(?[]const u8, null);
                        }
                        break :blk @as(?[]const u8, reasoning);
                    } else @as(?[]const u8, reasoning);
                    if (final_reasoning) |r| {
                        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = r }, null, null, null, .{});
                    }
                }
                if (think_split.content.len > 0) {
                    // Locate against the RAW concatenation: it is always a true
                    // suffix of the generation, where `flush_text` may be a
                    // normalized rewrite. If the content cannot be found there
                    // (normalization moved it) nothing is skipped, which is the
                    // old behaviour rather than a wrong boundary.
                    lps.skipToContent(full_text.items, think_split.content);
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = think_split.content }, null, null, null, .{ .logprobs_json = try lps.take() });
                } else {
                    // All reasoning, no content: those entries describe text the
                    // client never received.
                    lps.dropPending();
                }
            }
        }
    }

    const total_prompt = ts.prompt_tokens;
    if (!client_gone) {
        // Final chunk with finish_reason
        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null }, .{ .reason = finish_reason, .details = ts.finish_details }, null, null, .{ .logprobs_json = try lps.take() });

        // Usage chunk (if requested via stream_options.include_usage). Scheduler
        // accounts for any prompt-cache hits in `ts.prompt_tokens` directly.
        // OpenAI ships this chunk with an EMPTY choices array — restating
        // finish_reason/finish_details here made every per-event client render
        // the ending twice ('s doubled truncation banner). Pending
        // logprobs all drained on the final chunk above.
        if (include_usage) {
            const usage_json = try formatChatUsage(allocator, total_prompt, ts.completion_tokens, ts.cached_tokens, "");
            defer allocator.free(usage_json);
            const timings_obj = try formatTimingsObject(allocator, total_prompt, ts.cached_tokens, ts.completion_tokens, ts.prefill_ns, ts.decode_ns, tokenize_ns);
            defer allocator.free(timings_obj);
            const timings_opt: ?[]const u8 = if (timings_obj.len > 0) timings_obj else null;
            try sendSSEUsageChunk(allocator, stream, chat_id, model_name, usage_json, timings_opt);
        }

        // Done sentinel
        logHttpSseData("[DONE]");
        try stream.writeAll("data: [DONE]\n\n");
    }

    // Per-slot timings come from the scheduler (ts.prefill_ns / ts.decode_ns,
    // populated in finalize); the bracket reports compute + any prefix reuse.
    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, ts.prompt_tokens, ts.cached_tokens, ts.completion_tokens, ts.prefill_ns, ts.decode_ns);
    log.info("  <- {d}+{d} tokens streamed [{s}] [{s}]\n", .{
        total_prompt, ts.completion_tokens, perf, finish_reason,
    });
}

// ── tests ──

test "a streamed turn's result keeps the prompt-cache hit and timings the slot measured" {
    const t = std.testing;
    const ts: StreamingTokenStream = .{ .stream = undefined, .mode = .regular, .eos_token_ids = &.{}, .prompt_tokens = 3016, .cached_tokens = 2985, .completion_tokens = 40, .prefill_ns = 90_000_000, .decode_ns = 1_200_000_000 };
    var text = "ok".*;
    var ids = [_]u32{ 7, 8 };
    const r = ts.generationResult(&text, &ids, "stop", null);
    try t.expectEqual(@as(u32, 2985), r.cached_tokens);
    try t.expectEqual(ts.prefill_ns, r.prefill_ns);
    try t.expectEqual(ts.decode_ns, r.decode_ns);
}
