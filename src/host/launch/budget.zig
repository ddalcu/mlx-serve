//! The budget an agent CLI declares for a model, from the context the server advertises: the response cap, the room
//! kept free before compacting, and the agents whose context floor the window misses. The one source: `/v1/models`
//! serves it on each chat row (`agent_budget`), and the app, the pi extension and `mlx-serve launch` read it.

const std = @import("std");

pub const Budget = struct { context: u64, output: u64 };

/// What a row with no advertised context (an unloaded stub with no readable config) declares.
pub const FALLBACK_BUDGET = Budget{ .context = 32768, .output = 8192 };

/// The advertised context verbatim (the server already reserved its headroom); the response cap is half of it, since
/// thinking shares it, bounded so a runaway generation stops.
pub fn budgetForContext(ctx: u64) Budget {
    if (ctx == 0) return FALLBACK_BUDGET;
    return .{ .context = ctx, .output = @min(65536, @max(1024, ctx / 2)) };
}

/// Room an agent keeps free before compacting, and what it keeps after: a quarter of the window, capped where pi's and
/// opencode2's own 20000-token defaults (sized for 200k windows) take over.
pub fn compactionReserve(ctx: u64) u64 {
    return @min(20000, @max(1024, ctx / 4));
}

/// Below its floor an agent's own fixed prompt leaves every turn compacting or truncated: Claude Code sends 40-70k
/// before the first word (tool + MCP schemas, skills catalogue), opencode ~8k, pi ~2k.
const Floor = struct { []const u8, u64 };
const floors = [_]Floor{
    .{ "claude", 65536 },
    .{ "opencode", 32768 },
    .{ "opencode2", 32768 },
};
pub const default_floor: u64 = 16384;

pub fn contextFloor(agent: []const u8) u64 {
    for (floors) |f| if (std.mem.eql(u8, f[0], agent)) return f[1];
    return default_floor;
}

/// `{"context":C,"output":O,"compaction_reserve":R,"too_small_for":{…}}` for a row advertising `ctx` (0: none).
/// `too_small_for` names each agent whose floor the declared window misses, with that floor; `"default"` stands for
/// every agent without a floor of its own.
pub fn json(buf: *[256]u8, ctx: u64) []const u8 {
    const b = budgetForContext(ctx);
    var w: std.Io.Writer = .fixed(buf);
    w.print("{{\"context\":{d},\"output\":{d},\"compaction_reserve\":{d},\"too_small_for\":{{", .{ b.context, b.output, compactionReserve(b.context) }) catch unreachable;
    var sep: []const u8 = "";
    for (floors ++ [_]Floor{.{ "default", default_floor }}) |f| if (b.context < f[1]) {
        w.print("{s}\"{s}\":{d}", .{ sep, f[0], f[1] }) catch unreachable;
        sep = ",";
    };
    w.writeAll("}}") catch unreachable;
    return w.buffered();
}

/// The context a `/v1/models` row advertises: top level first (#188), then `meta`; 0 when it names none.
pub fn advertisedContext(row: std.json.ObjectMap) u64 {
    for ([_][]const u8{ "context_length", "max_model_len" }) |k| if (row.get(k)) |v| if (v == .integer and v.integer > 0) return @intCast(v.integer);
    if (row.get("meta")) |m| if (m == .object) if (m.object.get("context_length")) |v| if (v == .integer and v.integer > 0) return @intCast(v.integer);
    return 0;
}

/// Gives a chat row from another server (a LAN peer on an older build) the `agent_budget` it did not carry. A row with
/// no capabilities is an old peer's chat model.
pub fn stampRow(arena: std.mem.Allocator, row: *std.json.ObjectMap) !void {
    if (row.get("agent_budget") != null) return;
    if (row.get("capabilities")) |caps| if (caps == .array and caps.array.items.len > 0) {
        for (caps.array.items) |c| {
            if (c == .string and std.mem.eql(u8, c.string, "chat")) break;
        } else return;
    };
    var buf: [256]u8 = undefined;
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena, json(&buf, advertisedContext(row.*)), .{ .allocate = .alloc_always });
    try row.put(arena, "agent_budget", v);
}

// ── tests ──

const t = std.testing;

test "budgetForContext: ctx/2 clamped to [1024, 65536], 0 = fallback" {
    // Thinking shares the response cap: a 24k window at ctx/4 gave pi 6144,
    // which one xhigh design turn on Qwen3.8 spent entirely on thinking.
    try t.expectEqual(FALLBACK_BUDGET, budgetForContext(0));
    try t.expectEqual(Budget{ .context = 4096, .output = 2048 }, budgetForContext(4096));
    try t.expectEqual(Budget{ .context = 2048, .output = 1024 }, budgetForContext(2048));
    try t.expectEqual(Budget{ .context = 24576, .output = 12288 }, budgetForContext(24576));
    try t.expectEqual(Budget{ .context = 90112, .output = 45056 }, budgetForContext(90112));
    try t.expectEqual(Budget{ .context = 1048576, .output = 65536 }, budgetForContext(1048576));
}

test "compactionReserve: a quarter of the window, capped where the agents' own defaults take over" {
    // pi keeps 20000 recent tokens and opencode2 reserves a 20000 buffer by
    // default; both assume a 200k window. A 24k window compacted before its
    // first reply (opencode2) or never (pi).
    try t.expectEqual(@as(u64, 6144), compactionReserve(24576));
    try t.expectEqual(@as(u64, 2048), compactionReserve(8192));
    try t.expectEqual(@as(u64, 1024), compactionReserve(2048));
    try t.expectEqual(@as(u64, 20000), compactionReserve(262144));
}

test "contextFloor: claude 64k, opencode 32k, every other agent 16k" {
    try t.expectEqual(@as(u64, 65536), contextFloor("claude"));
    try t.expectEqual(@as(u64, 32768), contextFloor("opencode"));
    try t.expectEqual(@as(u64, 32768), contextFloor("opencode2"));
    try t.expectEqual(@as(u64, 16384), contextFloor("pi"));
    try t.expectEqual(@as(u64, 16384), contextFloor("hermes"));
}

test "json: the served budget names the agents its window is too small for" {
    var buf: [256]u8 = undefined;
    try t.expectEqualStrings("{\"context\":24576,\"output\":12288,\"compaction_reserve\":6144,\"too_small_for\":{\"claude\":65536,\"opencode\":32768,\"opencode2\":32768}}", json(&buf, 24576));
    try t.expectEqualStrings("{\"context\":8192,\"output\":4096,\"compaction_reserve\":2048,\"too_small_for\":{\"claude\":65536,\"opencode\":32768,\"opencode2\":32768,\"default\":16384}}", json(&buf, 8192));
    try t.expectEqualStrings("{\"context\":32768,\"output\":8192,\"compaction_reserve\":8192,\"too_small_for\":{\"claude\":65536}}", json(&buf, 0));
    try t.expectEqualStrings("{\"context\":1048576,\"output\":65536,\"compaction_reserve\":20000,\"too_small_for\":{}}", json(&buf, 1048576));
}

test "stampRow: an older server's chat row gets the budget of its advertised context; others are left alone" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body =
        \\[{"id":"a","capabilities":["chat"],"meta":{"context_length":24576}},
        \\ {"id":"b","capabilities":[]},
        \\ {"id":"c","capabilities":["image"]},
        \\ {"id":"d","capabilities":["chat"],"agent_budget":{"context":1,"output":1,"compaction_reserve":1}}]
    ;
    const rows = try std.json.parseFromSliceLeaky(std.json.Value, a, body, .{});
    for (rows.array.items) |*r| try stampRow(a, &r.object);
    const got = try std.json.Stringify.valueAlloc(a, rows, .{});
    try t.expect(std.mem.indexOf(u8, got, "\"id\":\"a\",\"capabilities\":[\"chat\"],\"meta\":{\"context_length\":24576},\"agent_budget\":{\"context\":24576,\"output\":12288,\"compaction_reserve\":6144,") != null);
    try t.expect(std.mem.indexOf(u8, got, "\"id\":\"b\",\"capabilities\":[],\"agent_budget\":{\"context\":32768,\"output\":8192,\"compaction_reserve\":8192,") != null);
    try t.expect(std.mem.indexOf(u8, got, "\"id\":\"c\",\"capabilities\":[\"image\"]}") != null);
    try t.expect(std.mem.indexOf(u8, got, "\"agent_budget\":{\"context\":1,") != null);
}
