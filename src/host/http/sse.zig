//! DeltaFields, ChunkExtras; sendSSEChunk, usage chunk; keepalive

const std = @import("std");
const io_util = @import("../../sdk/util/io_util.zig");
const nowSecs = io_util.nowSecs;
const nowMsMonotonic = io_util.nowMsMonotonic;

const Conn = @import("conn.zig").Conn;
const finishDetailsField = @import("../chat/tools/pipeline.zig").finishDetailsField;
const logHttpSseData = @import("respond.zig").logHttpSseData;
const jsonEscape = @import("../api/format.zig").jsonEscape;
const sendAnthropicEvent = @import("../api/anthropic.zig").sendAnthropicEvent;

const DeltaFields = struct {
    role: ?[]const u8,
    content: ?[]const u8,
    reasoning_content: ?[]const u8 = null,
    tool_calls_json: ?[]const u8 = null,
};

/// Per-choice fields that sit BESIDE `delta`, not inside it. Defaulted so a
/// call site that has nothing to add passes `.{}` — every existing emitter
/// keeps its exact bytes, which is what makes this additive.
const ChunkExtras = struct {
    /// Already-rendered `{"content":[…]}` from `formatLogprobsObject`, or null
    /// to omit the key entirely.
    logprobs_json: ?[]const u8 = null,
};

/// How a stream ended: the wire `finish_reason` plus, when the
/// degenerate-tail guard cut it, the sibling cause (`finishDetailsField`).
const Finish = struct {
    reason: []const u8,
    details: ?[]const u8 = null,
};

pub fn sendSSEChunk(
    allocator: std.mem.Allocator,
    stream: *Conn,
    chat_id: i64,
    model_name: []const u8,
    delta: DeltaFields,
    /// `null` on every mid-stream delta; the two FINAL chunks pass the reason
    /// and, when the degenerate-tail guard cut the turn, its cause. Modelled
    /// as one value so a site cannot report the reason and forget the cause.
    finish: ?Finish,
    usage_json: ?[]const u8,
    timings_json: ?[]const u8,
    extras: ChunkExtras,
) !void {
    // Build the delta JSON object
    var delta_buf = std.ArrayList(u8).empty;
    defer delta_buf.deinit(allocator);

    try delta_buf.appendSlice(allocator, "{");
    var need_comma = false;

    if (delta.role) |role| {
        try delta_buf.appendSlice(allocator, "\"role\":\"");
        try delta_buf.appendSlice(allocator, role);
        try delta_buf.appendSlice(allocator, "\"");
        need_comma = true;
    }

    if (delta.content) |content| {
        if (need_comma) try delta_buf.appendSlice(allocator, ",");
        try delta_buf.appendSlice(allocator, "\"content\":");
        const escaped = try jsonEscape(allocator, content);
        defer allocator.free(escaped);
        try delta_buf.appendSlice(allocator, escaped);
        need_comma = true;
    }

    if (delta.reasoning_content) |reasoning| {
        if (need_comma) try delta_buf.appendSlice(allocator, ",");
        try delta_buf.appendSlice(allocator, "\"reasoning_content\":");
        const escaped_r = try jsonEscape(allocator, reasoning);
        defer allocator.free(escaped_r);
        try delta_buf.appendSlice(allocator, escaped_r);
        need_comma = true;
    }

    if (delta.tool_calls_json) |tc_json| {
        if (need_comma) try delta_buf.appendSlice(allocator, ",");
        try delta_buf.appendSlice(allocator, "\"tool_calls\":");
        try delta_buf.appendSlice(allocator, tc_json);
    }

    try delta_buf.appendSlice(allocator, "}");

    // Build the finish_reason field (+ its sibling cause, when there is one)
    var fr_buf: [64]u8 = undefined;
    const fr_str = if (finish) |f|
        std.fmt.bufPrint(&fr_buf, "\"{s}\"", .{f.reason}) catch "null"
    else
        "null";
    const fd_str = if (finish) |f| finishDetailsField(f.reason, f.details) else "";

    // `logprobs` is a sibling of `delta` on the choice, not a field inside it.
    // Emitted only when the request asked; OpenAI's own chunk shape carries
    // `"logprobs":null` otherwise, which is what the empty case renders.
    var lp_buf = std.ArrayList(u8).empty;
    defer lp_buf.deinit(allocator);
    if (extras.logprobs_json) |lp| {
        try lp_buf.appendSlice(allocator, ",\"logprobs\":");
        try lp_buf.appendSlice(allocator, lp);
    }

    // Build usage field
    const usage_str = if (usage_json) |u| u else "null";

    // Optional `timings` tail. Inserted as a top-level field next to `usage`
    // — matches the llama.cpp shape so clients that key off it work as-is.
    var timings_tail_buf = std.ArrayList(u8).empty;
    defer timings_tail_buf.deinit(allocator);
    if (timings_json) |t| {
        try timings_tail_buf.appendSlice(allocator, ",\"timings\":");
        try timings_tail_buf.appendSlice(allocator, t);
    }

    // Build the full SSE chunk
    const chunk = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"delta":{s},"finish_reason":{s}{s}{s}}}],"usage":{s}{s}}}
    , .{ chat_id, nowSecs(stream.io), model_name, delta_buf.items, fr_str, fd_str, lp_buf.items, usage_str, timings_tail_buf.items });
    defer allocator.free(chunk);

    // Write as SSE event
    logHttpSseData(chunk);
    try stream.writeAllNoFlush("data: ");
    try stream.writeAllNoFlush(chunk);
    try stream.writeAllNoFlush("\n\n");
    try stream.flush();
}

/// The `stream_options.include_usage` chunk: OpenAI's shape is an EMPTY
/// `choices` array beside the usage object — no delta, no finish_reason, no
/// logprobs (all per-choice fields; the final chunk already carried them).
pub fn sendSSEUsageChunk(
    allocator: std.mem.Allocator,
    stream: *Conn,
    chat_id: i64,
    model_name: []const u8,
    usage_json: []const u8,
    timings_json: ?[]const u8,
) !void {
    var timings_tail_buf = std.ArrayList(u8).empty;
    defer timings_tail_buf.deinit(allocator);
    if (timings_json) |t| {
        try timings_tail_buf.appendSlice(allocator, ",\"timings\":");
        try timings_tail_buf.appendSlice(allocator, t);
    }
    const chunk = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[],"usage":{s}{s}}}
    , .{ chat_id, nowSecs(stream.io), model_name, usage_json, timings_tail_buf.items });
    defer allocator.free(chunk);

    logHttpSseData(chunk);
    try stream.writeAllNoFlush("data: ");
    try stream.writeAllNoFlush(chunk);
    try stream.writeAllNoFlush("\n\n");
    try stream.flush();
}

// ── Shared utilities ──

/// SSE comment keepalive for the OpenAI-style streaming surfaces. Comments
/// are SSE-spec-legal and skipped by every SSE parser. No-op on the
/// WebSocket transport — a raw comment line would corrupt WS framing, and
/// WS has protocol-level liveness of its own.
pub fn sendStreamKeepalive(stream: *Conn) !void {
    if (stream.ws_mode != null) return;
    try stream.writeAll(": keepalive\n\n");
}

/// Which no-op event a surface uses to prove liveness.
const KeepaliveStyle = enum { sse_comment, anthropic_ping };

/// Emit a keepalive iff the socket has been SILENT for `STREAM_KEEPALIVE_MS`.
///
/// Every streaming token loop calls this once per iteration — not just on the
/// `.idle` (waiting-for-first-token) path. Those two are different facts: a
/// surface buffering tokens for tool-call or thinking detection is receiving
/// tokens steadily while writing nothing, and only bytes on the wire hold off
/// a client's idle-body timeout (Node's `fetch`/undici: 300 s, then
/// `TypeError: terminated`). A one-shot `write_file` of a whole source file
/// buffers for exactly that long.
pub fn beatStreamKeepalive(stream: *Conn, style: KeepaliveStyle) !void {
    if (!stream.keepaliveDue()) return;
    switch (style) {
        .sse_comment => try sendStreamKeepalive(stream),
        .anthropic_ping => try sendAnthropicEvent(stream, "ping", "{\"type\":\"ping\"}"),
    }
    // The WS transport no-ops both senders; stamp regardless so a bridged
    // connection doesn't re-evaluate the deadline on every token.
    stream.heartbeat.noteWrite(nowMsMonotonic(stream.io));
}
