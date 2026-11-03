//! response store globals; response store init/deinit

const std = @import("std");
const chat_mod = @import("../../chat/chat.zig");
const responses_mod = @import("model.zig");
const testing = std.testing;

const storeResponse = @import("envelope.zig").storeResponse;

/// In-memory store for OpenAI Responses API state (`store: true` requests).
/// Bounded LRU; lost on restart.
var global_response_store: ?responses_mod.ResponseStore = null;

var global_response_store_gpa: ?std.mem.Allocator = null;

const RESPONSE_STORE_CAP: usize = 256;

pub const DEFAULT_STRUCTURED_OUTPUT_MAX_TOKENS: u32 = 2048;

/// Connection threads race the first `/v1/responses` requests to the lazy init.
var global_response_store_init: std.Io.Mutex = .init;

pub fn getOrInitResponseStore(io: std.Io, gpa: std.mem.Allocator) *responses_mod.ResponseStore {
    global_response_store_init.lockUncancelable(io);
    defer global_response_store_init.unlock(io);
    if (global_response_store == null) {
        global_response_store = responses_mod.ResponseStore.init(io, gpa, RESPONSE_STORE_CAP);
        global_response_store_gpa = gpa;
    }
    return &global_response_store.?;
}

pub fn deinitGlobalResponseStore() void {
    if (global_response_store) |*store| {
        store.deinit();
        global_response_store = null;
        global_response_store_gpa = null;
    }
}

// ── tests ──

test "Responses history omits loop cuts while retaining the response and input" {
    deinitGlobalResponseStore();
    defer deinitGlobalResponseStore();
    const messages = [_]chat_mod.Message{.{ .role = "user", .content = "hi" }};
    const calls = [_]chat_mod.ToolCall{.{ .id = "call_1", .name = "writeFile", .arguments = "{}" }};
    try storeResponse(testing.io, testing.allocator, "loop", "model", "completed", "{\"status\":\"completed\"}", &messages, "loop fragment", "repeated thought", &calls, "repetition_loop");
    const stored = global_response_store.?.map.get("loop").?;
    try testing.expectEqualStrings("completed", stored.status);
    try testing.expectEqualStrings("{\"status\":\"completed\"}", stored.body_json);
    try testing.expectEqual(@as(usize, 1), stored.history.len);
    try testing.expectEqualStrings("user", stored.history[0].role);
    try testing.expectEqualStrings("hi", stored.history[0].content);

    // A continuation keeps earlier input and its healthy assistant output.
    try storeResponse(testing.io, testing.allocator, "next", "model", "completed", "{}", stored.history, "answer", null, null, null);
    const next = global_response_store.?.map.get("next").?;
    try testing.expectEqual(@as(usize, 2), next.history.len);
    try testing.expectEqualStrings("hi", next.history[0].content);
    try testing.expectEqualStrings("answer", next.history[1].content);
}

test "deinitGlobalResponseStore frees stored responses" {
    deinitGlobalResponseStore();
    defer deinitGlobalResponseStore();

    const messages = [_]chat_mod.Message{.{ .role = "user", .content = "hi" }};
    try storeResponse(testing.io, testing.allocator, "resp_test", "mlx-serve", "completed", "{}", &messages, "hello", null, null, null);

    if (global_response_store) |*store| {
        try testing.expectEqual(@as(usize, 1), store.map.count());
    } else {
        return error.TestUnexpectedResult;
    }

    deinitGlobalResponseStore();
    try testing.expect(global_response_store == null);
}
