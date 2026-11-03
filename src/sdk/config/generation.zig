//! A pack's `generation_config.json` into `Shell.gen` and the stop set, read as the engine reads it: a value out of
//! range is dropped, the file's eos ids merge after config.json's without duplicates, a missing or broken file
//! changes nothing. MLX-free (a plugin's `config.zig` calls it from `parse`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");

pub fn apply(dir: *const abi.PackDir, shell: *abi.Shell) void {
    var file: [1 << 20]u8 = undefined;
    var why: abi.Verdict = .{};
    const bytes = dir.readFile("generation_config.json", &file, &why) orelse return;
    applyJson(bytes, shell);
}

fn num(v: std.json.Value) ?f32 {
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

/// `apply` over the file's bytes. The JSON value tree is bounded like the engine's (16 KiB).
pub fn applyJson(bytes: []const u8, shell: *abi.Shell) void {
    var buf: [16 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const parsed = std.json.parseFromSlice(std.json.Value, fba.allocator(), bytes, .{}) catch return;
    defer parsed.deinit();
    if (parsed.value != .object) return;
    const root = parsed.value.object;
    const g = &shell.gen;
    if (root.get("temperature")) |v| if (num(v)) |t| if (t >= 0.0 and t <= 2.0) {
        g.temperature = t;
    };
    if (root.get("top_p")) |v| if (num(v)) |p| if (p > 0.0 and p <= 1.0) {
        g.top_p = p;
    };
    if (root.get("top_k")) |v| if (v == .integer and v.integer > 0 and v.integer <= 1000) {
        g.top_k = @intCast(v.integer);
    };
    if (root.get("min_p")) |v| if (num(v)) |m| if (m >= 0.0 and m <= 1.0) {
        g.min_p = m;
    };
    if (root.get("eos_token_id")) |v| switch (v) {
        .integer => |i| if (i >= 0 and i <= std.math.maxInt(u32)) mergeEos(shell, @intCast(i)),
        .array => |arr| {
            var taken: usize = 0; // the engine reads at most eight valid ids
            for (arr.items) |item| if (taken < 8 and item == .integer and item.integer >= 0 and item.integer <= std.math.maxInt(u32)) {
                mergeEos(shell, @intCast(item.integer));
                taken += 1;
            };
        },
        else => {},
    };
    if (root.get("default_chat_template_kwargs")) |v| if (v == .object) {
        if (v.object.get("enable_thinking")) |et| if (et == .bool) {
            g.enable_thinking = if (et.bool) .on else .off;
        };
    };
}

/// Appends `id` to the stop set unless it is there already (or the set is full).
pub fn mergeEos(shell: *abi.Shell, id: u32) void {
    for (shell.eos[0..shell.eos_count]) |e| if (e == id) return;
    addEos(shell, id);
}

/// Appends `id` to the stop set (config.json's own list keeps its repeats, as the engine's does).
pub fn addEos(shell: *abi.Shell, id: u32) void {
    if (shell.eos_count >= abi.max_eos) return;
    shell.eos[shell.eos_count] = id;
    shell.eos_count += 1;
}

// ── tests ──

const testing = std.testing;

test "generation config: values in range are taken, the rest dropped; eos merges without repeats" {
    var s: abi.Shell = .{};
    addEos(&s, 2);
    applyJson(
        \\{"temperature":0.6,"top_p":1.5,"top_k":20,"min_p":0,"eos_token_id":[2,7],
        \\ "default_chat_template_kwargs":{"enable_thinking":false}}
    , &s);
    try testing.expectEqual(@as(f32, 0.6), s.gen.temperature);
    try testing.expect(std.math.isNan(s.gen.top_p));
    try testing.expectEqual(@as(i32, 20), s.gen.top_k);
    try testing.expectEqual(@as(f32, 0), s.gen.min_p);
    try testing.expectEqual(abi.Tri.off, s.gen.enable_thinking);
    try testing.expectEqualSlices(u32, &.{ 2, 7 }, s.eos[0..s.eos_count]);
}

test "generation config: a broken file changes nothing" {
    var s: abi.Shell = .{};
    applyJson("{\"temperature\":", &s);
    try testing.expect(std.math.isNan(s.gen.temperature) and s.eos_count == 0);
}
