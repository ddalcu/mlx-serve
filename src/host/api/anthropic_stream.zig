//! handleAnthropicStreaming, delta/thinking emitters

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const kv_quant = @import("../../sdk/blocks/attention/kv_quant.zig");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const request_mod = @import("../../sdk/request/request.zig");
const chat_mod = @import("../chat/chat.zig");
const rp_mod = @import("../../sdk/grammar/protocol.zig");
const log = @import("../../sdk/util/log.zig");
const MediaSpan = @import("../../sdk/kv/media_span.zig").MediaSpan;
const mlx_engine = @import("../engines/mlx.zig");
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
const shouldParseToolCalls = @import("../chat/tools/pipeline.zig").shouldParseToolCalls;
const appendRawToolDump = @import("../chat/tools/pipeline.zig").appendRawToolDump;
const parseToolCallsForRequest = @import("../chat/tools/pipeline.zig").parseToolCallsForRequest;
const StopStream = @import("../chat/stops.zig").StopStream;
const emitConstrainedAnthropic = @import("chat_completions.zig").emitConstrainedAnthropic;
const StreamingTokenStream = @import("chat_stream.zig").StreamingTokenStream;
const requestHasCompany = @import("request_options.zig").requestHasCompany;
const pickStreamMode = @import("request_options.zig").pickStreamMode;
const SSE_ALLOW_HEADERS_ANTHROPIC = @import("../http/respond.zig").SSE_ALLOW_HEADERS_ANTHROPIC;
const sendSseHeaders = @import("../http/respond.zig").sendSseHeaders;
const utf8TrailingIncomplete = @import("format.zig").utf8TrailingIncomplete;
const formatAnthropicUsage = @import("format.zig").formatAnthropicUsage;
const formatPerfBracket = @import("format.zig").formatPerfBracket;
const jsonEscape = @import("format.zig").jsonEscape;
const MropeData = @import("../media/encode.zig").MropeData;
const beatStreamKeepalive = @import("../http/sse.zig").beatStreamKeepalive;
const sendAnthropicEvent = @import("anthropic.zig").sendAnthropicEvent;
const anthropicStopReason = @import("anthropic.zig").anthropicStopReason;

pub fn handleAnthropicStreaming(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: request_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    has_tools: bool,
    /// OpenAI-shape tools JSON (for bare-args tool-call inference); null when
    /// the request defined no tools.
    tools_json: ?[]const u8,
    /// false = client set parallel_tool_calls:false (Anthropic:
    /// tool_choice.disable_parallel_tool_use) — at most one call per response.
    allow_parallel_tools: bool,
    enable_thinking: bool,
    reasoning_budget: i32,
    prompt_token_count: u32,
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    vision_embeddings: ?mlx.mlx_array,
    media: []const MediaSpan,
    mrope: MropeData,
    cache_key: u64,
    /// Wave 1.A: per-request KV-quant override.
    kv_quant_override: ?kv_quant.KVQuantConfig,
    kv_attn_explicit: ?bool,
    /// Iteration 1: tokenize_ns from parent handler. Anthropic streaming
    /// doesn't currently emit `timings` over SSE (spec doesn't model it),
    /// but plumbing the value through keeps the signature consistent with
    /// non-streaming and unblocks a future message_delta extension.
    tokenize_ns: u64,
) !void {
    _ = tokenize_ns; // reserved; see doc comment
    // Vision-array ownership: held by this handler on entry, transfers to
    // the slot on submit (slot.deinit frees). Nulled before transfer.
    var ve_local = vision_embeddings;
    defer {
        if (ve_local) |arr| _ = mlx.mlx_array_free(arr);
    }

    // Pick speculative-decoding mode (regular / PLD / drafter). The token-
    // stream adapter below feeds the per-token Anthropic state machine the
    // same way for all three modes.
    const config = lm.config.?;
    const stream_mode = pickStreamMode(enable_pld, enable_drafter, enable_mtp, lm.drafter != null, lm.dflash != null, mtpCapable(lm), sampling.shapesLogits(), 0, requestHasCompany());
    if (stream_mode == .pld) log.info("  pld=enabled (streaming, draft_len={d}, key_len={d})\n", .{ http_server.serverState().server_config.default_pld_draft_len, http_server.serverState().server_config.default_pld_key_len });
    if (stream_mode == .drafter) log.info("  drafter=enabled (streaming, block_size={d})\n", .{lm.drafter_block_size});
    if (stream_mode == .mtp) log.info("  mtp=enabled (streaming, depth={d})\n", .{lm.mtp_depth});

    var running: ?mlx_engine.Running = null;
    defer if (running) |r| r.done();

    // Transfer vision ownership into the slot.
    const slot_ve_anth = ve_local;
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
        .logprobs_n = 0,
        .vision_embeddings = slot_ve_anth,
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

    // SSE headers
    try sendSseHeaders(stream, "anthropic.messages", SSE_ALLOW_HEADERS_ANTHROPIC);

    // message_start
    {
        const data = try formatAnthropicMessageStart(allocator, nowMs(stream.io), model_name, prompt_token_count);
        defer allocator.free(data);
        try sendAnthropicEvent(stream, "message_start", data);
    }
    try sendAnthropicEvent(stream, "ping", "{\"type\":\"ping\"}");

    // State
    var block_index: u32 = 0;
    var text_block_open = false;
    var thinking_block_open = false;
    // Template-opened think (Qwen 3.5/3.6 render `…assistant\n<think>\n` into
    // the generation prompt): the output's thinking carries NO opener tag.
    // Needed up front by the tools branch — by the time the close tag is the
    // only evidence, visible-text flushing has already leaked the thoughts.
    // See the streaming chat site for why the prompt fact is NOT ANDed with
    // enable_thinking (LFM2.5's unconditional pre-opened `<think>`).
    const prompt_opened_think = promptOpensThink(allocator, lm, tok, prompt_ids);
    const opens_think = prompt_opened_think;
    // Prompt-opened reasoning is DELIVERED as thinking blocks even when the
    // request didn't ask for thinking — generated tokens are never dropped.
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
    // Set once the buffered think block has been split + emitted (tools
    // branch). Releases the buffer hold AND tells the end-of-stream split
    // that the remaining text has no template-opened semantics.
    var think_closed = false;
    var content_started = false;
    // Bytes of the leading thought the tools branch already streamed; every later emit site sends the remainder.
    var reasoning_streamed: usize = 0;
    // Muse-Glimmer plain-arm segment-header skip (<|start|>…<|message|>).
    var muse_skip_header = promptOpensMuseHeader(allocator, lm, tok, prompt_ids);
    var muse_head = std.ArrayList(u8).empty; // header bytes held while skipping
    defer muse_head.deinit(allocator);
    var think_buf = std.ArrayList(u8).empty;
    defer think_buf.deinit(allocator);
    var think_close_tag: []const u8 = "</think>";
    var skipped_think_open = false;
    var think_tokens: i32 = 0;
    var budget_exhausted = false;

    var text_buf = std.ArrayList(u8).empty;
    var stop_stream: StopStream = .{};
    defer stop_stream.deinit(allocator);
    defer text_buf.deinit(allocator);
    // Memoized marker scan for the think gate. Owned BESIDE text_buf and reset
    // with it — the gate is otherwise O(buffer) per token for as long as a
    // think block is open and unclosed (47.99 us/token at 113 KB).
    var think_scan: chat_mod.ThinkScan = .{};
    var token_texts = std.ArrayList([]const u8).empty;
    defer {
        for (token_texts.items) |t| allocator.free(t);
        token_texts.deinit(allocator);
    }
    var stopped = false;
    var matched_stop_seq: ?[]const u8 = null;
    var client_gone = false;
    var utf8_carry: [3]u8 = undefined;
    var utf8_carry_len: u8 = 0;

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
                sendAnthropicEvent(stream, "ping", "{\"type\":\"ping\"}") catch {
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
        // UTF-8 carry handling
        var token_text = if (next_id) |token_id| blk: {
            const strip = tok.tok_type == .sentencepiece_bpe;
            const raw_decoded = try decodeTokens(allocator, lm, tok, &[_]u32{token_id}, strip and false);
            if (delivery) |*d| d.noteToken(raw_decoded.len, running.?.stream.constraintPayloadStart());
            const with_carry = if (utf8_carry_len > 0) cc: {
                const combined = try allocator.alloc(u8, utf8_carry_len + raw_decoded.len);
                @memcpy(combined[0..utf8_carry_len], utf8_carry[0..utf8_carry_len]);
                @memcpy(combined[utf8_carry_len..], raw_decoded);
                allocator.free(raw_decoded);
                utf8_carry_len = 0;
                break :cc combined;
            } else raw_decoded;
            const tail = utf8TrailingIncomplete(with_carry);
            if (tail > 0) {
                @memcpy(utf8_carry[0..tail], with_carry[with_carry.len - tail ..]);
                utf8_carry_len = @intCast(tail);
            }
            if (with_carry.len == tail) {
                allocator.free(with_carry);
                continue;
            }
            if (tail > 0) {
                const trimmed = try allocator.dupe(u8, with_carry[0 .. with_carry.len - tail]);
                allocator.free(with_carry);
                break :blk trimmed;
            }
            break :blk with_carry;
        } else try stop_stream.takeHeld(allocator);

        // Stop sequences; remember WHICH one matched (reported as stop_reason
        // "stop_sequence" + the echoed `stop_sequence` field in message_delta).
        if (stop_sequences.len > 0) {
            const fed = try stop_stream.feed(allocator, &text_buf, token_text, stop_sequences, prompt_opened_think and !constrained_proto and !think_closed);
            allocator.free(token_text);
            token_text = fed.send;
            if (fed.stop) |m| {
                stopped = true;
                matched_stop_seq = m;
            }
            if (token_text.len == 0) {
                allocator.free(token_text);
                continue;
            }
        } else if (gated_stream) try text_buf.appendSlice(allocator, token_text);

        if (delivery) |*d| {
            defer allocator.free(token_text);
            try d.feed(allocator, token_text);
            try emitConstrainedAnthropic(allocator, stream, &block_index, &thinking_block_open, &text_block_open, d);
            try beatStreamKeepalive(stream, .anthropic_ping);
            continue;
        }

        if (gated_stream) {
            // Buffer for tool detection. Detection rules live in
            // `chat.streamShouldBufferForTools` — the SAME predicate as the
            // chat-completions stream (this path once carried its own inline
            // subset, which missed the Inkling invoke marker: the
            // NAME+JSON content leak, and before that the drift class the
            // think gate was unified for).
            try token_texts.append(allocator, token_text);
            const buf = text_buf.items;
            const maybe_tool = chat_mod.streamShouldBufferForTools(buf);

            if (!maybe_tool) {
                // Shared gate with the chat-completions stream (the two paths
                // drifted once: this one only recognized think openers present
                // in the OUTPUT, so Qwen-family template-opened thinking
                // streamed as visible text_deltas and a raw `</think>` leaked
                // into Claude Code transcripts). Hermetically pinned per
                // recorded model family by the corpus streaming-gate test.
                switch (chat_mod.streamThinkGateScan(buf, enable_thinking, think_closed, prompt_opened_think, &think_scan)) {
                    .hold_thinking => {
                        // Stream the leading thought as it arrives: the tool
                        // check above just passed, so these bytes are reasoning.
                        // A thought re-opened after visible text stays held — its
                        // split re-reads that text.
                        if (!budget_exhausted and !think_closed and !content_started) {
                            const so_far = chat_mod.splitThinkBlock(buf, true, opens_think);
                            if (so_far.reasoning_content) |split_rc| {
                                const rc = chat_mod.streamableReasoning(split_rc);
                                if (chat_mod.unstreamedReasoning(rc, reasoning_streamed)) |fresh| {
                                    try sendAnthropicThinking(allocator, stream, block_index, &thinking_block_open, fresh);
                                    reasoning_streamed = rc.len;
                                    think_tokens += 1;
                                    if (reasoning_budget >= 0 and think_tokens >= reasoning_budget) {
                                        budget_exhausted = true;
                                        try endAnthropicThinking(allocator, stream, &block_index, &thinking_block_open);
                                    }
                                }
                            }
                        }
                    },
                    .split_think => {
                        // Complete think block — split once: thinking block
                        // first, then the visible remainder as text. Reasoning
                        // ships regardless of the request's thinking flag.
                        const split = chat_mod.splitThinkBlock(buf, true, opens_think);
                        if (chat_mod.unstreamedReasoning(if (budget_exhausted) "" else split.reasoning_content orelse "", reasoning_streamed)) |rest| {
                            try sendAnthropicThinking(allocator, stream, block_index, &thinking_block_open, rest);
                        }
                        try endAnthropicThinking(allocator, stream, &block_index, &thinking_block_open);
                        reasoning_streamed = 0;
                        if (split.content.len > 0) {
                            if (!text_block_open) {
                                const sd = try std.fmt.allocPrint(allocator,
                                    \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                                , .{block_index});
                                defer allocator.free(sd);
                                try sendAnthropicEvent(stream, "content_block_start", sd);
                                text_block_open = true;
                            }
                            content_started = true;
                            try emitAnthropicTextDelta(allocator, stream, block_index, split.content);
                        }
                        for (token_texts.items) |tt| allocator.free(tt);
                        token_texts.clearRetainingCapacity();
                        text_buf.clearRetainingCapacity();
                        think_scan.reset();
                        think_closed = true;
                    },
                    .flush_text => {
                        // Freed here, not per item: see the chat stream's flush.
                        defer {
                            for (token_texts.items) |tt| allocator.free(tt);
                            token_texts.clearRetainingCapacity();
                        }
                        for (token_texts.items) |tt| {
                            // Skip bare think/channel tags that leak without a block
                            if (chat_mod.isChannelMarkerToken(tt)) {
                                continue;
                            }
                            const vis = chat_mod.streamContentLead(tt, content_started);
                            if (vis.len == 0) continue;
                            content_started = true;
                            if (!text_block_open) {
                                const sd = try std.fmt.allocPrint(allocator,
                                    \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                                , .{block_index});
                                defer allocator.free(sd);
                                try sendAnthropicEvent(stream, "content_block_start", sd);
                                text_block_open = true;
                            }
                            try emitAnthropicTextDelta(allocator, stream, block_index, vis);
                        }
                    },
                }
            }
        } else if (in_think_block) {
            // Thinking block handling. Also covers thinking-off +
            // template-opened (LFM2.5 class): the block streams as thinking
            // deltas instead of being paid for and dropped.
            defer allocator.free(token_text);
            try think_buf.appendSlice(allocator, token_text);
            think_tokens += 1;

            if (!skipped_think_open and (think_buf.items.len >= 7 or chat_mod.cannotOpenThink(think_buf.items))) {
                if (chat_mod.thinkOpenTagLenAt(think_buf.items)) |olen| {
                    var skip: usize = olen;
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                    if (!thinking_block_open) {
                        try openAnthropicThinkingBlock(allocator, stream, block_index);
                        thinking_block_open = true;
                    }
                } else if (std.mem.startsWith(u8, think_buf.items, "<|content_thinking|>")) {
                    // Inkling thinking message — closes at <|end_message|>.
                    think_close_tag = "<|end_message|>";
                    var skip: usize = "<|content_thinking|>".len;
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                    if (!thinking_block_open) {
                        try openAnthropicThinkingBlock(allocator, stream, block_index);
                        thinking_block_open = true;
                    }
                } else if (think_buf.items.len >= 17 and std.mem.startsWith(u8, think_buf.items, "<|channel>thought")) {
                    think_close_tag = "<channel|>";
                    var skip: usize = 17;
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                    if (!thinking_block_open) {
                        try openAnthropicThinkingBlock(allocator, stream, block_index);
                        thinking_block_open = true;
                    }
                } else if (chat_mod.harmonyThinkOpenerAt(think_buf.items) == .analysis) {
                    // gpt_oss: `[<|start|>assistant]<|channel|>analysis<|message|>`
                    // opens a reasoning segment that closes at <|end|>. Consume
                    // the header — every byte of it is ordinary text, and the
                    // close matcher below assumes reasoning starts at byte 0.
                    const skip = switch (chat_mod.harmonyThinkOpenerAt(think_buf.items)) {
                        .analysis => |hl| hl,
                        else => unreachable,
                    };
                    think_close_tag = "<|end|>";
                    var skip2 = skip;
                    while (skip2 < think_buf.items.len and think_buf.items[skip2] == '\n') skip2 += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip2..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                    if (!thinking_block_open) {
                        try openAnthropicThinkingBlock(allocator, stream, block_index);
                        thinking_block_open = true;
                    }
                } else if (chat_mod.harmonyThinkOpenerAt(think_buf.items) == .growing) {
                    // Harmony header still arriving — wait, or `<|channel|>anal`
                    // leaks as reasoning.
                } else if (chat_mod.museThinkOpenerAt(think_buf.items) == .self_opened) {
                    // Muse-Glimmer reasoning segment (` to=self<|message|>`) —
                    // closes at <|eom|>. The direct-answer header closes via
                    // the muse arm in the close matcher below.
                    const mskip = switch (chat_mod.museThinkOpenerAt(think_buf.items)) {
                        .self_opened => |hl| hl,
                        else => unreachable,
                    };
                    think_close_tag = "<|eom|>";
                    var skip: usize = mskip;
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                    if (!thinking_block_open) {
                        try openAnthropicThinkingBlock(allocator, stream, block_index);
                        thinking_block_open = true;
                    }
                } else if (chat_mod.museThinkOpenerAt(think_buf.items) == .growing) {
                    // Muse header still arriving token by token — wait.
                } else if (think_buf.items.len < 17 and std.mem.startsWith(u8, "<|channel>thought", think_buf.items)) {
                    // Partial prefix of `<|channel>thought` — wait for more tokens.
                } else if (think_buf.items.len < 32 and chat_mod.endsWithPartialThinkOpen(think_buf.items)) {
                    // Still a growing suffixed opener (`<think:opensou`) — wait.
                } else {
                    // No opener in the model's output — template injected one.
                    // Stay in the think block; close tag is detected dynamically.
                    skipped_think_open = true;
                    if (!thinking_block_open) {
                        try openAnthropicThinkingBlock(allocator, stream, block_index);
                        thinking_block_open = true;
                    }
                }
            }

            // A spent budget closes the DELIVERED thinking block; the tail is withheld, never text.
            if (!budget_exhausted and reasoning_budget >= 0 and think_tokens >= reasoning_budget and skipped_think_open) {
                budget_exhausted = true;
                if (thinking_block_open) {
                    try closeAnthropicThinkingBlock(allocator, stream, block_index);
                    thinking_block_open = false;
                    block_index += 1;
                }
            }

            // Check for the close tag — accept whichever appears first
            // (indexOfThinkCloseTag covers bare AND Hy3-suffixed </think…>).
            const think_match = chat_mod.indexOfThinkCloseTag(think_buf.items, 0);
            const channel_pos = std.mem.indexOf(u8, think_buf.items, "<channel|>");
            // Inkling: <|end_message|> closes the thinking message (gated on
            // the opener having switched think_close_tag); a leading
            // <|content_text|> is a DIRECT answer — empty reasoning.
            const inkling_pos: ?usize = blk_i: {
                if (std.mem.startsWith(u8, think_buf.items, "<|content_text|>")) break :blk_i 0;
                if (std.mem.eql(u8, think_close_tag, "<|end_message|>")) {
                    if (std.mem.indexOf(u8, think_buf.items, "<|end_message|>")) |p| break :blk_i p;
                }
                break :blk_i null;
            };
            const inkling_len: usize = if (inkling_pos != null and inkling_pos.? == 0 and std.mem.startsWith(u8, think_buf.items, "<|content_text|>")) "<|content_text|>".len else "<|end_message|>".len;
            // Muse: a resolved non-self header is a DIRECT answer — close at 0
            // with empty reasoning; a reasoning segment closes at its <|eom|>
            // (gated on the muse opener having latched think_close_tag).
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
                if (thinking_block_open and m.pos > 0) {
                    try emitAnthropicThinkingDelta(allocator, stream, block_index, think_buf.items[0..m.pos]);
                }
                if (thinking_block_open) {
                    try closeAnthropicThinkingBlock(allocator, stream, block_index);
                    thinking_block_open = false;
                    block_index += 1;
                }
                const after = m.pos + m.len;
                var content_after = std.mem.trimStart(u8, think_buf.items[after..], "\n ");
                if (std.mem.startsWith(u8, content_after, "<|channel>\n")) content_after = content_after[11..];
                if (std.mem.startsWith(u8, content_after, "<|channel>")) content_after = content_after[10..];
                if (std.mem.startsWith(u8, content_after, "<|message_model|>")) content_after = content_after["<|message_model|>".len..];
                if (std.mem.startsWith(u8, content_after, "<|content_text|>")) content_after = content_after["<|content_text|>".len..];
                // Harmony: the answer arrives as a whole new segment —
                // `<|start|>assistant<|channel|>final<|message|>` — and every
                // byte of that header is ordinary text that would otherwise
                // ride out as content.
                content_after = chat_mod.stripHarmonySegmentHeader(content_after);
                if (std.mem.indexOf(u8, content_after, "<|end_message|>")) |em| content_after = content_after[0..em];
                // Muse next-segment header: strip when complete, arm the
                // plain-arm skip when still arriving.
                if (chat_mod.museContentHeaderSkip(content_after)) |hl| {
                    content_after = content_after[hl..];
                } else if (std.mem.startsWith(u8, content_after, "<|start|>")) {
                    muse_skip_header = true;
                    content_after = "";
                }
                if (std.mem.indexOf(u8, content_after, "<|eot|>")) |et| content_after = content_after[0..et];
                content_after = std.mem.trimStart(u8, content_after, "\n ");
                if (content_after.len > 0) {
                    content_started = true;
                    if (!text_block_open) {
                        const sd = try std.fmt.allocPrint(allocator,
                            \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                        , .{block_index});
                        defer allocator.free(sd);
                        try sendAnthropicEvent(stream, "content_block_start", sd);
                        text_block_open = true;
                    }
                    try emitAnthropicTextDelta(allocator, stream, block_index, content_after);
                }
                think_buf.clearRetainingCapacity();
                in_think_block = false;
                think_close_tag = if (m.is_channel) "<channel|>" else "</think>";
            } else if (skipped_think_open) {
                // Hold back a still-growing partial close tag (suffixed forms
                // included), then back off to a UTF-8 boundary — a cut mid-
                // codepoint ships a lone continuation byte in the delta JSON.
                var safe_len = think_buf.items.len - chat_mod.partialThinkCloseSuffixLen(think_buf.items);
                while (safe_len > 0 and safe_len < think_buf.items.len and (think_buf.items[safe_len] & 0xC0) == 0x80) safe_len -= 1;
                if (safe_len > 0) {
                    if (thinking_block_open) try emitAnthropicThinkingDelta(allocator, stream, block_index, think_buf.items[0..safe_len]);
                    const remaining = try allocator.dupe(u8, think_buf.items[safe_len..]);
                    think_buf.clearRetainingCapacity();
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                }
            }
        } else {
            // Regular content token
            defer allocator.free(token_text);
            // Muse segment-header skip: <|start|>…<|message|> is never content.
            const was_skipping = muse_skip_header;
            muse_skip_header = chat_mod.museHeaderSkipNext(muse_skip_header, token_text);
            if (was_skipping or muse_skip_header) {
                if (muse_skip_header) try muse_head.appendSlice(allocator, token_text);
                // `<|message|>` resolved a header the model wrote: `to=self` is reasoning.
                if (was_skipping and !muse_skip_header and !content_started and chat_mod.museHeaderOpensReasoning(muse_head.items)) {
                    in_think_block = true;
                    skipped_think_open = true;
                    think_close_tag = "<|eom|>";
                    if (!thinking_block_open) {
                        try openAnthropicThinkingBlock(allocator, stream, block_index);
                        thinking_block_open = true;
                    }
                }
                if (!muse_skip_header) muse_head.clearRetainingCapacity();
                continue;
            }
            // A think block the MODEL opens: seeded like a template-injected one.
            // Checked before the marker skip, which would swallow a bare `<think>`.
            if (!content_started) if (chat_mod.modelThinkOpener(channel_armed, token_text)) |opener| {
                channel_armed = false;
                in_think_block = true;
                try think_buf.appendSlice(allocator, opener);
                continue;
            };
            if (chat_mod.isChannelMarkerToken(token_text)) {
                channel_armed = !content_started and std.mem.eql(u8, token_text, "<|channel>");
                continue;
            }
            channel_armed = false;
            const vis_token = chat_mod.streamContentLead(token_text, content_started);
            if (vis_token.len == 0) continue;
            content_started = true;
            if (!text_block_open) {
                const sd = try std.fmt.allocPrint(allocator,
                    \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                , .{block_index});
                defer allocator.free(sd);
                try sendAnthropicEvent(stream, "content_block_start", sd);
                text_block_open = true;
            }
            try emitAnthropicTextDelta(allocator, stream, block_index, vis_token);
        }

        // See the chat-completions loop: a buffered tool call / thinking block
        // emits no events at all, so liveness must be driven off socket
        // silence, not off token arrival.
        beatStreamKeepalive(stream, .anthropic_ping) catch {
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
            try emitConstrainedAnthropic(allocator, stream, &block_index, &thinking_block_open, &text_block_open, d);
        }
    }

    // Flush remaining think buffer
    if (!client_gone and thinking_block_open and think_buf.items.len > 0) {
        try emitAnthropicThinkingDelta(allocator, stream, block_index, think_buf.items);
    }
    // The tools branch closes its streamed block after the end-of-stream split below.
    if (!client_gone and !gated_stream and thinking_block_open) {
        try closeAnthropicThinkingBlock(allocator, stream, block_index);
        block_index += 1;
    }

    // Post-generation: snapshot stats from whichever source was active.
    ts.finalize();

    // Handle tool calls
    var finish_reason: []const u8 = if (client_gone) "client_disconnect" else if (stopped) "stop" else ts.finish_reason;

    if (gated_stream and !client_gone) {
        if (log.isDebug() and text_buf.items.len > 0) {
            log.debug("  raw generated text before tool parse ({d}b): {s}\n", .{ text_buf.items.len, text_buf.items[0..@min(text_buf.items.len, 4000)] });
            if (env.get(.MLX_SERVE_RAW_DUMP_FILE)) |dump_path| {
                appendRawToolDump(std.mem.span(dump_path), tools_json, text_buf.items);
            }
        }
        // Merge re-opened mid-text thought channels into the leading block so
        // the split/parse below never leaks raw tags (Gemma 12B re-opens
        // channels mid-turn — observed live via Claude Code on this surface).
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

            // Close any open text block FIRST, at its own index. The old
            // order emitted the thinking block's start while the text block
            // still held this index, then stopped the text block at a
            // never-started index — Claude Code aborted the turn with
            // "API Error: Content block not found".
            if (text_block_open) {
                const sd = try std.fmt.allocPrint(allocator, "{{\"type\":\"content_block_stop\",\"index\":{d}}}", .{block_index});
                defer allocator.free(sd);
                try sendAnthropicEvent(stream, "content_block_stop", sd);
                block_index += 1;
                text_block_open = false;
            }

            // Emit thinking from buffered text if present — delivery does not
            // key on the request's thinking flag. Once the in-loop split
            // already emitted it, the remaining buffer has no template-opened
            // semantics — passing `opens_think` unguarded would misfile the
            // visible tail as reasoning.
            {
                const think_split = chat_mod.splitThinkBlock(gen_text, true, opens_think and !think_closed);
                if (chat_mod.unstreamedReasoning(if (budget_exhausted) "" else think_split.reasoning_content orelse "", reasoning_streamed)) |reasoning| {
                    try sendAnthropicThinking(allocator, stream, block_index, &thinking_block_open, reasoning);
                }
                try endAnthropicThinking(allocator, stream, &block_index, &thinking_block_open);
            }

            for (tool_calls, 0..) |tc, i| {
                const tc_id = try std.fmt.allocPrint(allocator, "toolu_{d}_{d}", .{ nowMs(stream.io), i });
                defer allocator.free(tc_id);
                const esc_name = try jsonEscape(allocator, tc.name);
                defer allocator.free(esc_name);

                // content_block_start
                const start = try std.fmt.allocPrint(allocator,
                    \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"tool_use","id":"{s}","name":{s},"input":{{}}}}}}
                , .{ block_index, tc_id, esc_name });
                defer allocator.free(start);
                try sendAnthropicEvent(stream, "content_block_start", start);

                // input_json_delta
                const esc_args_full = try jsonEscape(allocator, tc.arguments);
                defer allocator.free(esc_args_full);
                const args_inner = esc_args_full[1 .. esc_args_full.len - 1];
                const delta = try std.fmt.allocPrint(allocator,
                    \\{{"type":"content_block_delta","index":{d},"delta":{{"type":"input_json_delta","partial_json":"{s}"}}}}
                , .{ block_index, args_inner });
                defer allocator.free(delta);
                try sendAnthropicEvent(stream, "content_block_delta", delta);

                // content_block_stop
                const stop = try std.fmt.allocPrint(allocator, "{{\"type\":\"content_block_stop\",\"index\":{d}}}", .{block_index});
                defer allocator.free(stop);
                try sendAnthropicEvent(stream, "content_block_stop", stop);
                block_index += 1;
            }
            finish_reason = toolCallFinishReason(finish_reason);
        } else {
            // No tool calls — flush buffered tokens, splitting thinking from
            // content regardless of the request's thinking flag (split.content
            // passes trimLeakedToolMarkup, so the unparsed-markup cut the old
            // thinking-off arm did still happens).
            {
                var full_text = std.ArrayList(u8).empty;
                defer full_text.deinit(allocator);
                for (token_texts.items) |t| try full_text.appendSlice(allocator, t);
                const flush_norm = try chat_mod.normalizeEmbeddedThinkBlocks(allocator, full_text.items);
                defer if (flush_norm) |n| allocator.free(n);
                const flush_text: []const u8 = flush_norm orelse full_text.items;
                const think_split = chat_mod.splitThinkBlock(flush_text, true, opens_think and !think_closed);
                if (chat_mod.unstreamedReasoning(if (budget_exhausted) "" else think_split.reasoning_content orelse "", reasoning_streamed)) |reasoning| {
                    try sendAnthropicThinking(allocator, stream, block_index, &thinking_block_open, reasoning);
                }
                try endAnthropicThinking(allocator, stream, &block_index, &thinking_block_open);
                if (think_split.content.len > 0) {
                    content_started = true;
                    if (!text_block_open) {
                        const sd2 = try std.fmt.allocPrint(allocator,
                            \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                        , .{block_index});
                        defer allocator.free(sd2);
                        try sendAnthropicEvent(stream, "content_block_start", sd2);
                        text_block_open = true;
                    }
                    try emitAnthropicTextDelta(allocator, stream, block_index, think_split.content);
                }
            }
        }
    }

    const total_prompt = ts.prompt_tokens;
    const stop_reason = anthropicStopReason(finish_reason, matched_stop_seq);
    const echo_stop_seq = std.mem.eql(u8, stop_reason, "stop_sequence");
    const stop_seq_json: []const u8 = if (echo_stop_seq) try jsonEscape(allocator, matched_stop_seq.?) else "null";
    defer if (echo_stop_seq) allocator.free(stop_seq_json);

    if (!client_gone) {
        // Close text block if open
        if (text_block_open) {
            const sd = try std.fmt.allocPrint(allocator, "{{\"type\":\"content_block_stop\",\"index\":{d}}}", .{block_index});
            defer allocator.free(sd);
            try sendAnthropicEvent(stream, "content_block_stop", sd);
        }

        // Ensure at least one content block
        if (!text_block_open and block_index == 0) {
            const sd = try std.fmt.allocPrint(allocator,
                \\{{"type":"content_block_start","index":0,"content_block":{{"type":"text","text":""}}}}
            , .{});
            defer allocator.free(sd);
            try sendAnthropicEvent(stream, "content_block_start", sd);
            const sd2 = "{\"type\":\"content_block_stop\",\"index\":0}";
            try sendAnthropicEvent(stream, "content_block_stop", sd2);
        }

        // Cache hits are known after prefill. Cumulative usage replaces the
        // provisional message_start counts with disjoint input buckets.
        {
            const md = try formatAnthropicMessageDelta(allocator, stop_reason, stop_seq_json, total_prompt, ts.completion_tokens, ts.cached_tokens);
            defer allocator.free(md);
            try sendAnthropicEvent(stream, "message_delta", md);
        }
        try sendAnthropicEvent(stream, "message_stop", "{\"type\":\"message_stop\"}");
    }

    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, ts.prompt_tokens, ts.cached_tokens, ts.completion_tokens, ts.prefill_ns, ts.decode_ns);
    log.info("  <- {d}+{d} tokens streamed [{s}] [{s}]\n", .{
        total_prompt, ts.completion_tokens, perf, stop_reason,
    });
}

fn formatAnthropicMessageStart(allocator: std.mem.Allocator, id: i64, model_name: []const u8, prompt_tokens: u32) ![]u8 {
    return try std.fmt.allocPrint(allocator,
        \\{{"type":"message_start","message":{{"id":"msg_{d}","type":"message","role":"assistant","content":[],"model":"{s}","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":{d},"output_tokens":1}}}}}}
    , .{ id, model_name, prompt_tokens });
}

fn formatAnthropicMessageDelta(allocator: std.mem.Allocator, stop_reason: []const u8, stop_seq_json: []const u8, prompt_tokens: u32, completion_tokens: u32, cached_tokens: u32) ![]u8 {
    const usage = try formatAnthropicUsage(allocator, prompt_tokens, completion_tokens, cached_tokens);
    defer allocator.free(usage);
    return try std.fmt.allocPrint(allocator,
        \\{{"type":"message_delta","delta":{{"stop_reason":"{s}","stop_sequence":{s}}},"usage":{s}}}
    , .{ stop_reason, stop_seq_json, usage });
}

/// Emit a text_delta event for Anthropic streaming.
pub fn emitAnthropicTextDelta(allocator: std.mem.Allocator, stream: *Conn, index: u32, text: []const u8) !void {
    const esc = try jsonEscape(allocator, text);
    defer allocator.free(esc);
    const inner = esc[1 .. esc.len - 1];
    const data = try std.fmt.allocPrint(allocator,
        \\{{"type":"content_block_delta","index":{d},"delta":{{"type":"text_delta","text":"{s}"}}}}
    , .{ index, inner });
    defer allocator.free(data);
    try sendAnthropicEvent(stream, "content_block_delta", data);
}

/// Emit a thinking_delta event for Anthropic streaming.
pub fn emitAnthropicThinkingDelta(allocator: std.mem.Allocator, stream: *Conn, index: u32, thinking: []const u8) !void {
    const esc = try jsonEscape(allocator, thinking);
    defer allocator.free(esc);
    const inner = esc[1 .. esc.len - 1];
    const data = try std.fmt.allocPrint(allocator,
        \\{{"type":"content_block_delta","index":{d},"delta":{{"type":"thinking_delta","thinking":"{s}"}}}}
    , .{ index, inner });
    defer allocator.free(data);
    try sendAnthropicEvent(stream, "content_block_delta", data);
}

/// Close a thinking block with a fake signature and content_block_stop.
fn openAnthropicThinkingBlock(allocator: std.mem.Allocator, stream: *Conn, block_index: u32) !void {
    const sd = try std.fmt.allocPrint(allocator,
        \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"thinking","thinking":"","signature":""}}}}
    , .{block_index});
    defer allocator.free(sd);
    try sendAnthropicEvent(stream, "content_block_start", sd);
}

pub fn closeAnthropicThinkingBlock(allocator: std.mem.Allocator, stream: *Conn, index: u32) !void {
    const sig = try std.fmt.allocPrint(allocator,
        \\{{"type":"content_block_delta","index":{d},"delta":{{"type":"signature_delta","signature":"mlx-serve-local"}}}}
    , .{index});
    defer allocator.free(sig);
    try sendAnthropicEvent(stream, "content_block_delta", sig);
    const stop = try std.fmt.allocPrint(allocator, "{{\"type\":\"content_block_stop\",\"index\":{d}}}", .{index});
    defer allocator.free(stop);
    try sendAnthropicEvent(stream, "content_block_stop", stop);
}

/// Stream reasoning into the thinking block at `index`, opening it on first use.
fn sendAnthropicThinking(allocator: std.mem.Allocator, stream: *Conn, index: u32, open: *bool, thinking: []const u8) !void {
    if (!open.*) {
        try openAnthropicThinkingBlock(allocator, stream, index);
        open.* = true;
    }
    try emitAnthropicThinkingDelta(allocator, stream, index, thinking);
}

/// Close the thinking block if one is open and move past its index.
fn endAnthropicThinking(allocator: std.mem.Allocator, stream: *Conn, index: *u32, open: *bool) !void {
    if (!open.*) return;
    try closeAnthropicThinkingBlock(allocator, stream, index.*);
    open.* = false;
    index.* += 1;
}

// ─── /v1/responses (OpenAI Responses API) ────────────────────────────────

// ── tests ──

test "Anthropic usage: message_delta overwrites provisional message_start counts" {
    for ([_]u32{ 0, 40000, 91863, 92934, 92935 }) |cached_tokens| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const start = try formatAnthropicMessageStart(a, 1, "mlx-serve", 92934);
        const delta = try formatAnthropicMessageDelta(a, "end_turn", "null", 92934, 126, cached_tokens);
        const start_event = try std.json.parseFromSliceLeaky(std.json.Value, a, start, .{});
        const delta_event = try std.json.parseFromSliceLeaky(std.json.Value, a, delta, .{});
        try testing.expectEqualStrings("message_start", start_event.object.get("type").?.string);
        try testing.expectEqualStrings("message_delta", delta_event.object.get("type").?.string);
        var usage = start_event.object.get("message").?.object.get("usage").?.object;
        try testing.expectEqual(@as(i64, 92934), usage.get("input_tokens").?.integer);
        var fields = delta_event.object.get("usage").?.object.iterator();
        while (fields.next()) |field| try usage.put(a, field.key_ptr.*, field.value_ptr.*);
        const input = usage.get("input_tokens").?.integer;
        const created = if (usage.get("cache_creation_input_tokens")) |v| v.integer else 0;
        const cached = usage.get("cache_read_input_tokens").?.integer;
        try testing.expectEqual(@as(i64, 92934), input + created + cached);
        try testing.expectEqual(@as(i64, 92934 -| cached_tokens), input);
        try testing.expectEqual(@as(i64, 0), created);
        try testing.expectEqual(@as(i64, @min(cached_tokens, 92934)), cached);
        try testing.expectEqual(@as(i64, 126), usage.get("output_tokens").?.integer);
    }
}
