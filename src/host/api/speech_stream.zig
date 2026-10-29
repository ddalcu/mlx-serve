//! A streamed spoken reply on `/v1/chat/completions`: OpenAI's `delta.audio`, the text as `transcript` deltas (the
//! frame slots and mode markers are not text) and the speech as 16-bit PCM `data` deltas as the codec makes it final.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const request_mod = @import("../../sdk/request/request.zig");
const log = @import("../../sdk/util/log.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const MediaSpan = @import("../../sdk/kv/media_span.zig").MediaSpan;
const mlx_engine = @import("../engines/mlx.zig");
const model_registry_mod = @import("../registry/registry.zig");
const http_server = @import("../http/server.zig");
const Conn = @import("../http/conn.zig").Conn;
const speech_api = @import("speech.zig");
const format = @import("format.zig");
const respond = @import("../http/respond.zig");
const sendSSEUsageChunk = @import("../http/sse.zig").sendSSEUsageChunk;
const sendStreamKeepalive = @import("../http/sse.zig").sendStreamKeepalive;
const slotFailure = @import("../http/errors.zig").slotFailure;
const decodeTokens = @import("tokenize.zig").decodeTokens;
const pipeline = @import("../chat/tools/pipeline.zig");
const MropeData = @import("../media/encode.zig").MropeData;

pub fn handle(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *model_registry_mod.LoadedModel,
    tok: *const tokenizer_mod.Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: request_mod.SamplingParams,
    eos_token_ids: []const u32,
    model_name: []const u8,
    include_usage: bool,
    /// The card's tools recipe (no prefill): text before the model's first speech marker is held, and a reply that
    /// never speaks is parsed for calls, sent as `tool_calls`.
    tools_json: ?[]const u8,
    allow_parallel_tools: bool,
    vision_embeddings: ?mlx.mlx_array,
    media: []const MediaSpan,
    cache_key: u64,
    mrope: MropeData,
    tokenize_ns: u64,
) !void {
    const f = lm.config.?.speech().?;
    const id = io_util.nowMs(stream.io);
    // The id a later history turn names this reply by (`speech.store`).
    const audio_id = speech_api.Store.newId(stream.io);
    const created = io_util.nowSecs(stream.io);
    const running = try mlx_engine.Running.submit(http_server.launch().global_scheduler.?, .{
        .model = lm,
        .prompt_ids = prompt_ids,
        .full_prompt = prompt_ids,
        .cached_tokens = 0,
        .has_tools = tools_json != null,
        .enable_thinking = false,
        .sampling = sampling,
        .eos_token_ids = eos_token_ids,
        .max_tokens = max_tokens,
        .timeout_ns = http_server.getTimeoutNs(),
        .enable_pld = false,
        .enable_drafter = false,
        .drafter = null,
        .dflash = null,
        .drafter_block_size = lm.drafter_block_size,
        .allow_batch_mtp = false,
        .enable_mtp = false,
        .mtp = null,
        .mtp_depth = lm.mtp_depth,
        .pld_draft_len = 0,
        .pld_key_len = 0,
        .kv_attn_fused = false,
        .vision_embeddings = vision_embeddings,
        .media = media,
        .cache_key = cache_key,
        .client = stream.client,
        .mrope_pos = mrope.pos,
        .mrope_total = mrope.total,
        .mrope_delta = mrope.delta,
        .speech = .{},
    });
    defer running.done();
    const s = running.stream;

    try respond.sendSseHeaders(stream, "chat.completions", respond.SSE_ALLOW_HEADERS_DEFAULT);
    try chunk(allocator, stream, id, created, model_name, "{\"role\":\"assistant\",\"content\":null}");

    // The reply's text, decoded a token at a time as it arrives (the main stream's way: never the whole reply again).
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    var sent_text: usize = 0;
    var sent_samples: usize = 0;
    var client_gone = false;
    // Without tools the turn is prefilled with `<|mixed_start|>`: it speaks from the first token.
    var spoke = tools_json == null;
    while (true) {
        const nr = s.waitNextTimeout(Conn.STREAM_KEEPALIVE_MS);
        if (stream.peerClosed()) {
            s.cancel();
            client_gone = true;
            break;
        }
        const done = if (nr) |r| switch (r) {
            .token => |t| blk: {
                if (t == f.mixed_start or t == f.audio_start) spoke = true;
                if (t != f.frame_token and t != f.audio_start and t != f.mixed_start and t != f.text_end) {
                    const piece = try decodeTokens(allocator, lm, tok, &.{t}, false);
                    defer allocator.free(piece);
                    try text.appendSlice(allocator, piece);
                }
                break :blk false;
            },
            .done => true,
            .err => return slotFailure(s),
        } else blk: {
            sendStreamKeepalive(stream) catch {
                s.cancel();
                client_gone = true;
                break;
            };
            break :blk false;
        };
        if (spoke) sent_text = try sendTranscript(allocator, stream, id, audio_id, created, model_name, text.items, sent_text, done);
        sent_samples = try sendAudio(allocator, stream, s, id, audio_id, created, model_name, sent_samples);
        if (done) break;
    }
    if (client_gone) return;
    const transcript = text.items;

    var reason: []const u8 = s.finish_reason;
    const calls = if (spoke) null else try pipeline.parseToolCallsForRequest(allocator, transcript, tools_json, allow_parallel_tools);
    if (calls) |tcs| {
        defer {
            for (tcs) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(tcs);
        }
        for (tcs, 0..) |tc, i| {
            const tc_id = try std.fmt.allocPrint(allocator, "call_{d}_{d}", .{ id, i });
            defer allocator.free(tc_id);
            const delta = try pipeline.streamToolCallDelta(allocator, i, tc_id, tc.name, tc.arguments);
            defer allocator.free(delta);
            const body = try std.fmt.allocPrint(allocator, "{{\"tool_calls\":{s}}}", .{delta});
            defer allocator.free(body);
            try chunk(allocator, stream, id, created, model_name, body);
        }
        reason = pipeline.toolCallFinishReason(reason);
    } else {
        // A reply that answered in text without speaking still reaches the client, as its transcript.
        if (!spoke) _ = try sendTranscript(allocator, stream, id, audio_id, created, model_name, text.items, sent_text, true);
        try speech_api.store.put(stream.io, audio_id, created, transcript);
    }
    const fin = try std.fmt.allocPrint(allocator, "{{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"{s}\",\"system_fingerprint\":\"mlx-serve\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{s}\"}}]}}", .{ id, created, model_name, reason });
    defer allocator.free(fin);
    try sendData(stream, fin);
    if (include_usage) {
        const details = try std.fmt.allocPrint(allocator, ",\"completion_tokens_details\":{{\"audio_tokens\":{d}}}", .{s.audio_codes.items.len / f.codebooks});
        defer allocator.free(details);
        const usage = try format.formatChatUsage(allocator, s.prompt_tokens, s.completion_tokens, s.cached_tokens, details);
        defer allocator.free(usage);
        const timings = try format.formatTimingsObject(allocator, s.prompt_tokens, s.cached_tokens, s.completion_tokens, s.prefill_ns, s.decode_ns, tokenize_ns);
        defer allocator.free(timings);
        try sendSSEUsageChunk(allocator, stream, id, model_name, usage, if (timings.len > 0) timings else null);
    }
    respond.logHttpSseData("[DONE]");
    try stream.writeAll("data: [DONE]\n\n");
    var perf_buf: [160]u8 = undefined;
    log.info("  <- {d}+{d} tokens streamed, {d} frames [{s}] [{s}]\n", .{ s.prompt_tokens, s.completion_tokens, s.audio_codes.items.len / f.codebooks, format.formatPerfBracket(&perf_buf, s.prompt_tokens, s.cached_tokens, s.completion_tokens, s.prefill_ns, s.decode_ns), reason });
}

/// The transcript past `sent` bytes, held back while its tail is an incomplete UTF-8 sequence (`final` sends all).
fn sendTranscript(a: std.mem.Allocator, stream: *Conn, id: i64, audio_id: speech_api.Id, created: i64, model: []const u8, text: []const u8, sent: usize, final: bool) !usize {
    if (text.len == 0) return sent;
    const end = if (final) text.len else text.len - format.utf8TrailingIncomplete(text);
    if (end <= sent) return sent;
    const esc = try format.jsonEscape(a, text[sent..end]);
    defer a.free(esc);
    const delta = try std.fmt.allocPrint(a, "{{\"audio\":{{\"id\":\"{s}\",\"transcript\":{s}}}}}", .{ &audio_id, esc });
    defer a.free(delta);
    try chunk(a, stream, id, created, model, delta);
    return end;
}

/// The samples the codec made final past `sent`, as one base64 16-bit PCM delta.
fn sendAudio(a: std.mem.Allocator, stream: *Conn, s: anytype, id: i64, audio_id: speech_api.Id, created: i64, model: []const u8, sent: usize) !usize {
    s.out_mu.lockUncancelable(s.io);
    const fresh = a.dupe(f32, s.audio_samples.items[@min(sent, s.audio_samples.items.len)..]) catch |e| {
        s.out_mu.unlock(s.io);
        return e;
    };
    s.out_mu.unlock(s.io);
    defer a.free(fresh);
    if (fresh.len == 0) return sent;
    const data = try speech_api.encode(a, fresh, 0, .pcm16);
    defer a.free(data);
    const delta = try std.fmt.allocPrint(a, "{{\"audio\":{{\"id\":\"{s}\",\"data\":\"{s}\",\"expires_at\":{d}}}}}", .{ &audio_id, data, created + speech_api.Store.TTL_SECS });
    defer a.free(delta);
    try chunk(a, stream, id, created, model, delta);
    return sent + fresh.len;
}

fn chunk(a: std.mem.Allocator, stream: *Conn, id: i64, created: i64, model: []const u8, delta: []const u8) !void {
    const body = try std.fmt.allocPrint(a, "{{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"{s}\",\"system_fingerprint\":\"mlx-serve\",\"choices\":[{{\"index\":0,\"delta\":{s},\"finish_reason\":null}}]}}", .{ id, created, model, delta });
    defer a.free(body);
    try sendData(stream, body);
}

fn sendData(stream: *Conn, body: []const u8) !void {
    respond.logHttpSseData(body);
    try stream.writeAllNoFlush("data: ");
    try stream.writeAllNoFlush(body);
    try stream.writeAllNoFlush("\n\n");
    try stream.flush();
}
