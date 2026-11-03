//! WsBridge; WebSocket transport, body from ws request

const std = @import("std");
const log = @import("../../../sdk/util/log.zig");
const responses_mod = @import("model.zig");
const model_registry_mod = @import("../../registry/registry.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const ws_mod = @import("../../http/ws.zig");

const extractResponseId = @import("handler.zig").extractResponseId;
const extractCompletedStatus = @import("handler.zig").extractCompletedStatus;
const Conn = @import("../../http/conn.zig").Conn;
const getOrInitResponseStore = @import("store.zig").getOrInitResponseStore;
const MAX_JSON_DEPTH = @import("../../http/request.zig").MAX_JSON_DEPTH;
const jsonNestingDepth = @import("../../http/request.zig").jsonNestingDepth;
const mapGenerationError = @import("../../http/errors.zig").mapGenerationError;
const jsonEscape = @import("../format.zig").jsonEscape;
const handleResponses = @import("handler.zig").handleResponses;

/// Bridge that lets a Conn route OpenAI-Responses output through a WebSocket
/// transport instead of HTTP/SSE. When `Conn.ws_mode` is set, `sendResponse`
/// and `sendAnthropicEvent` send the JSON payload as a single WS text frame
/// instead of an HTTP body or SSE event line.
pub const WsBridge = struct {
    impl: *anyopaque,
    sendTextFn: *const fn (impl: *anyopaque, data: []const u8) anyerror!void,
    /// Captured response id from the first event seen this turn (for moving
    /// the entry between global and connection-local caches afterwards).
    /// Owned by `allocator`; freed by the WS handler at end of turn.
    captured_resp_id: ?[]u8 = null,
    allocator: ?std.mem.Allocator = null,
    /// Status emitted in the most recent `response.completed`-class event,
    /// or null if no terminal event was seen.
    captured_status: ?[]u8 = null,

    pub fn sendText(self: *WsBridge, data: []const u8) !void {
        if (self.captured_resp_id == null) {
            if (self.allocator) |a| {
                if (extractResponseId(a, data)) |id| self.captured_resp_id = id;
            }
        }
        if (self.allocator) |a| {
            if (extractCompletedStatus(a, data)) |st| {
                if (self.captured_status) |old| a.free(old);
                self.captured_status = st;
            }
        }
        try self.sendTextFn(self.impl, data);
    }

    pub fn reset(self: *WsBridge) void {
        if (self.allocator) |a| {
            if (self.captured_resp_id) |id| a.free(id);
            if (self.captured_status) |st| a.free(st);
        }
        self.captured_resp_id = null;
        self.captured_status = null;
    }
};

const WsConnT = ws_mod.WsConn(Conn);

/// Connection-local cache for `store: false` continuations on a WS session.
/// Each entry holds one reference (StoredResponse.release frees the last).
const WsLocalCache = struct {
    map: std.StringHashMapUnmanaged(*responses_mod.StoredResponse) = .{},
    gpa: std.mem.Allocator,

    fn put(self: *WsLocalCache, sr: *responses_mod.StoredResponse) !void {
        if (self.map.fetchRemove(sr.id)) |kv| kv.value.release();
        try self.map.put(self.gpa, sr.id, sr);
    }

    fn get(self: *WsLocalCache, id: []const u8) ?*responses_mod.StoredResponse {
        return self.map.get(id);
    }

    fn evict(self: *WsLocalCache, id: []const u8) void {
        if (self.map.fetchRemove(id)) |kv| kv.value.release();
    }

    fn deinit(self: *WsLocalCache) void {
        var it = self.map.valueIterator();
        while (it.next()) |sr_ptr| sr_ptr.*.release();
        self.map.deinit(self.gpa);
    }
};

/// Returns the call_id of the first `function_call_output` input item that
/// has no matching `function_call` in the prev response's history (or in
/// the same input list, for parallel tool-call replies). Returns null if
/// every output has a matching call.
fn orphanFunctionCallOutputId(
    input_val: std.json.Value,
    prev_id: ?[]const u8,
    local: *WsLocalCache,
    global: *responses_mod.ResponseStore,
) ?[]const u8 {
    if (input_val != .array) return null;
    // Collect all known call_ids from (a) prev history and (b) current input's
    // function_call items.
    var known: std.StringHashMapUnmanaged(void) = .{};
    defer known.deinit(std.heap.page_allocator);
    if (prev_id) |pid| blk: {
        const sr = local.get(pid) orelse global.get(pid) orelse break :blk;
        for (sr.history) |m| {
            if (m.tool_calls) |tcs| for (tcs) |tc| {
                _ = known.put(std.heap.page_allocator, tc.id, {}) catch {};
            };
        }
    }
    for (input_val.array.items) |item| {
        if (item != .object) continue;
        const t_v = item.object.get("type") orelse continue;
        if (t_v != .string or !std.mem.eql(u8, t_v.string, "function_call")) continue;
        const cid_v = item.object.get("call_id") orelse continue;
        if (cid_v == .string) {
            _ = known.put(std.heap.page_allocator, cid_v.string, {}) catch {};
        }
    }
    // Now check each function_call_output.
    for (input_val.array.items) |item| {
        if (item != .object) continue;
        const t_v = item.object.get("type") orelse continue;
        if (t_v != .string or !std.mem.eql(u8, t_v.string, "function_call_output")) continue;
        const cid_v = item.object.get("call_id") orelse return "";
        if (cid_v != .string) return "";
        if (!known.contains(cid_v.string)) return cid_v.string;
    }
    return null;
}

fn wsBridgeSend(impl: *anyopaque, data: []const u8) anyerror!void {
    const ws_conn: *WsConnT = @ptrCast(@alignCast(impl));
    try ws_conn.writeText(data);
}

/// Send a JSON error frame for a single WS turn. The compliance suite's
/// frame handler treats a `{"type":"error"}` text frame as terminal —
/// emitting a trailing `[DONE]` would land in the *next* turn's bucket
/// (same bug as success-path: see `handleResponsesWebSocket`).
fn wsSendErrorTurn(allocator: std.mem.Allocator, ws_conn: *WsConnT, status: u32, code: []const u8, message: []const u8) !void {
    const esc_code = try jsonEscape(allocator, code);
    defer allocator.free(esc_code);
    const esc_msg = try jsonEscape(allocator, message);
    defer allocator.free(esc_msg);
    const err_frame = try std.fmt.allocPrint(
        allocator,
        \\{{"type":"error","status":{d},"error":{{"code":{s},"message":{s}}}}}
    ,
        .{ status, esc_code, esc_msg },
    );
    defer allocator.free(err_frame);
    try ws_conn.writeText(err_frame);
}

/// Drive a WebSocket connection that bridges to /v1/responses.
///
/// Each text frame is a `response.create`-shaped JSON message. We translate
/// it into an HTTP-like body and reuse `handleResponses` (with a
/// `Conn.ws_mode` bridge) so all SSE events become WS text frames. After
/// each turn we emit `[DONE]` and wait for the next message on the same
/// connection, supporting chained `response.create` calls and
/// `previous_response_id` continuations.
pub fn handleResponsesWebSocket(
    allocator: std.mem.Allocator,
    stream: *Conn,
    headers: []const u8,
    lm: *LoadedModel,
) !void {
    ws_mod.handshake(stream, headers) catch |err| {
        log.warn("WS handshake failed: {s}\n", .{@errorName(err)});
        return;
    };
    log.info("WS /v1/responses connected\n", .{});

    var ws_conn = WsConnT.init(stream);
    defer ws_conn.deinit(allocator);

    var local_cache: WsLocalCache = .{ .gpa = allocator };
    defer local_cache.deinit();

    var bridge: WsBridge = .{ .impl = &ws_conn, .sendTextFn = &wsBridgeSend, .allocator = allocator };
    defer bridge.reset();

    // Frame loop — one iteration per inbound WS message.
    while (true) {
        const msg = ws_conn.readMessage(allocator) catch |err| switch (err) {
            error.WsClosed => return,
            error.WsProtocol => {
                ws_conn.writeClose(1002, "protocol error") catch {};
                return;
            },
            error.WsTooLarge => {
                ws_conn.writeClose(1009, "message too large") catch {};
                return;
            },
            else => return,
        };
        defer allocator.free(msg.payload);

        switch (msg.opcode) {
            .close => {
                ws_conn.writeClose(1000, "") catch {};
                return;
            },
            .ping => {
                try ws_conn.writePong(msg.payload);
                continue;
            },
            .pong => continue,
            .text => {},
            else => continue,
        }

        if (jsonNestingDepth(msg.payload) > MAX_JSON_DEPTH) {
            try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "Request JSON nests deeper than 4096 levels");
            continue;
        }
        // Parse the request payload — must be {"type":"response.create", ...}
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, msg.payload, .{}) catch {
            try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "Invalid JSON in request body");
            continue;
        };
        defer parsed.deinit();
        if (parsed.value != .object) {
            try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "Request must be a JSON object");
            continue;
        }
        const root = parsed.value.object;
        const type_val = root.get("type");
        if (type_val == null or type_val.? != .string or !std.mem.eql(u8, type_val.?.string, "response.create")) {
            try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "Expected type=response.create");
            continue;
        }
        // The plan disallows stream/stream_options/background here — WS is
        // inherently streaming and stateless (no background jobs).
        if (root.get("stream") != null or root.get("stream_options") != null or root.get("background") != null) {
            try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "stream/stream_options/background are not allowed over WebSocket");
            continue;
        }

        // ── Resolve previous_response_id against per-conn cache first.
        //    For store:false continuations on this connection, the chain
        //    root is in `local_cache` only. For store:true responses we
        //    fall back to the global store (lets stored chains survive
        //    across connections). On miss, evict any stale local entry
        //    before reporting the error.
        const prev_id_owned: ?[]const u8 = if (root.get("previous_response_id")) |v|
            (if (v == .string) v.string else null)
        else
            null;
        var prev_in_local: bool = false;
        if (prev_id_owned) |pid| {
            if (local_cache.get(pid) != null) {
                prev_in_local = true;
            } else {
                const store = getOrInitResponseStore(stream.io, allocator);
                if (store.get(pid)) |sr| sr.release() else {
                    local_cache.evict(pid);
                    try wsSendErrorTurn(allocator, &ws_conn, 404, "previous_response_not_found", "previous_response_id not found");
                    continue;
                }
            }
        }

        // Validate any function_call_output items reference real call_ids
        // in the prev response's history. An orphan output means the user
        // is trying to feed back a tool result for a call the model never
        // made — the turn must fail and (per compliance) evict the chain
        // root from the local cache.
        if (prev_id_owned) |pid| {
            if (root.get("input")) |input_v| {
                if (orphanFunctionCallOutputId(input_v, prev_id_owned, &local_cache, getOrInitResponseStore(stream.io, allocator))) |_| {
                    local_cache.evict(pid);
                    try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "function_call_output references a missing call_id");
                    continue;
                }
            }
        }

        // For store:false continuations on this connection, the chain root
        // lives in `local_cache` (not the global store). Move it into the
        // global store *just for this turn* so handleResponses' lookup at
        // line 4634 can find it. We restore the original location at end
        // of turn. (The kludge avoids modifying handleResponses.)
        const did_borrow_to_global = prev_id_owned != null and prev_in_local;
        if (did_borrow_to_global) {
            const sr = local_cache.map.get(prev_id_owned.?).?;
            // Remove from local without freeing.
            _ = local_cache.map.remove(prev_id_owned.?);
            const store = getOrInitResponseStore(stream.io, allocator);
            store.put(sr) catch {};
        }

        // ── Reserialize the request as an HTTP body, forcing
        //    `stream: true` (WS is inherently streaming) and `store: true`
        //    so handleResponses always persists the result. We move the
        //    entry to local_cache below if the user actually wanted
        //    `store: false`, achieving connection-scoped lifetime.
        const want_user_store: bool = if (root.get("store")) |v| (if (v == .bool) v.bool else true) else true;
        const body = try buildResponsesBodyFromWsRequest(allocator, root);
        defer allocator.free(body);

        // Reset sequence numbering per response per the OpenAI spec.
        // (handleResponses owns its own seq_num, fresh each call.)
        var seq: u64 = 0;
        _ = &seq;

        stream.ws_mode = &bridge;
        defer stream.ws_mode = null;

        bridge.reset();
        handleResponses(allocator, stream, body, lm) catch |err| {
            log.warn("WS handleResponses error: {s}\n", .{@errorName(err)});
            // Best-effort error frame; same mapping as every HTTP surface.
            var ws_err_buf: [320]u8 = undefined;
            const ws_wire = mapGenerationError(err, &ws_err_buf);
            wsSendErrorTurn(allocator, &ws_conn, ws_wire.code, ws_wire.openai_type, ws_wire.message) catch {};
            // Restore borrowed prev entry back to local cache on failure.
            if (did_borrow_to_global and prev_id_owned != null) {
                const store = getOrInitResponseStore(stream.io, allocator);
                if (store.take(prev_id_owned.?)) |taken| {
                    local_cache.map.put(local_cache.gpa, taken.id, taken) catch {};
                }
            }
            continue;
        };

        // Handle the borrowed prev: move it back from global to local
        // cache. On success, also do the eviction-on-failure check —
        // if the just-completed turn ended in a non-completed status
        // (failed/incomplete), evict the chain root from local cache.
        if (did_borrow_to_global and prev_id_owned != null) {
            const store = getOrInitResponseStore(stream.io, allocator);
            if (store.take(prev_id_owned.?)) |taken| {
                const turn_failed = bridge.captured_status != null and !std.mem.eql(u8, bridge.captured_status.?, "completed");
                if (turn_failed) {
                    // Compliance: a failed continuation evicts the chain root.
                    taken.release();
                } else {
                    local_cache.map.put(local_cache.gpa, taken.id, taken) catch {};
                }
            }
        }

        // For store:false on this WS, move the freshly-stored response
        // from the global store into the connection-local cache so it
        // (a) survives the user's `store: false` semantics within this
        // connection (allowing previous_response_id chains) and
        // (b) does NOT leak across reconnects (other WS connections
        // looking up this id should get previous_response_not_found).
        if (!want_user_store) {
            if (bridge.captured_resp_id) |rid| {
                const store = getOrInitResponseStore(stream.io, allocator);
                if (store.take(rid)) |taken| {
                    local_cache.map.put(local_cache.gpa, taken.id, taken) catch {};
                }
            }
        }

        // No `[DONE]` on the WS success path: over WebSocket the per-response
        // terminator is `response.completed` (or .failed/.incomplete), and the
        // compliance suite advances to the next turn the moment it sees one. A
        // trailing `[DONE]` would arrive *after* that advance and be misread
        // as the next turn's marker, killing chained sessions. HTTP SSE is the
        // opposite: one response per stream, and OpenAI ends it with
        // `data: [DONE]` (emitted in handleResponses, gated on ws_mode). We
        // still emit `[DONE]` for error fallbacks (see `wsSendErrorTurn`)
        // where no terminal event is sent.
    }
}

/// Reshape a WS `response.create` JSON object into an OpenAI-Responses HTTP
/// request body. We strip the `type` discriminator and force both
/// `stream: true` (WS is inherently streaming) and `store: true` (so
/// handleResponses persists the response in the global store, where the WS
/// handler can later move it into the connection-local cache when the user
/// requested `store: false`).
fn buildResponsesBodyFromWsRequest(allocator: std.mem.Allocator, root: std.json.ObjectMap) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try buf.append(allocator, '{');
    var first = true;
    var stream_seen = false;
    var store_seen = false;
    var iter = root.iterator();
    while (iter.next()) |entry| {
        const k = entry.key_ptr.*;
        if (std.mem.eql(u8, k, "type")) continue;
        if (std.mem.eql(u8, k, "stream")) stream_seen = true;
        if (std.mem.eql(u8, k, "store")) store_seen = true;
        if (!first) try buf.append(allocator, ',');
        first = false;
        const ek = try jsonEscape(allocator, k);
        defer allocator.free(ek);
        try buf.appendSlice(allocator, ek);
        try buf.append(allocator, ':');
        if (std.mem.eql(u8, k, "stream") or std.mem.eql(u8, k, "store")) {
            try buf.appendSlice(allocator, "true");
        } else {
            try responses_mod.serializeJsonValue(allocator, &buf, entry.value_ptr.*);
        }
    }
    if (!stream_seen) {
        if (!first) try buf.append(allocator, ',');
        first = false;
        try buf.appendSlice(allocator, "\"stream\":true");
    }
    if (!store_seen) {
        if (!first) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "\"store\":true");
    }
    try buf.append(allocator, '}');
    return try buf.toOwnedSlice(allocator);
}

// ─── Response envelope echo helpers ──────────────────────────────────────

// The OpenAI Responses API ResponseResource schema requires the response to

// echo most request configuration (tools, tool_choice, text.format, reasoning,

// metadata, etc.). These helpers re-render those values from the parsed

// request JSON in the exact shape the schema demands. Caller owns the result.
