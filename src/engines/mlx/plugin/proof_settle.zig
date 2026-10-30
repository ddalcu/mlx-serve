//! The per-slot plugin calls of a served model against a test table over the llama plugin: every forward carries the
//! slot's stable id, a deferred decode step leaves what the plugin says it owes and the engine settles it where the
//! legacy deferred-PLE path did (an owed input right after the build, owed history at the next forward or never), a
//! batched tick settles its slots in one call, the slot's meters are logged and its id released with the generator.
//! Decoded tokens stay bit-equal to the plain served model's.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const fixture = @import("../../../sdk/bases/proof/fixture.zig");
const llama = @import("../../../plugins/llama/plugin.zig");
const proof = @import("proof.zig");
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Generator = @import("../generate/generator.zig").Generator;
const Tokenizer = @import("../../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const Transformer = @import("../transformer.zig").Transformer;

const testing = std.testing;

/// What the test table saw.
var seen: struct {
    owes: abi.Owed = .none,
    /// One slot runs at a time: its id must not move.
    one_slot: bool = true,
    deferred: u32 = 0,
    slot: u32 = 0,
    flushes: u32 = 0,
    flushed_slots: u32 = 0,
    with_tokens: u32 = 0,
    discards: u32 = 0,
    released: u32 = 0,
    meters: u32 = 0,
} = .{};

fn tForward(m: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const slots = f.slots.slice();
    if (slots.len == 0 or slots[0] == 0) return abi.Verdict.refuse(.failed, "no slot id");
    if (seen.one_slot) {
        if (seen.slot != 0 and seen.slot != slots[0]) return abi.Verdict.refuse(.failed, "the slot id moved");
        seen.slot = slots[0];
    }
    const v = llama.table.forward(m, f, out);
    if (f.defer_host) {
        seen.deferred += 1;
        out.owed = seen.owes;
    }
    return v;
}

fn tSettle(_: *anyopaque, slots: abi.Span(u32), how: abi.Settle, tokens: *const abi.Span(u32)) callconv(.c) abi.Verdict {
    switch (how) {
        .flush => {
            seen.flushes += 1;
            seen.flushed_slots += @intCast(slots.len);
            if (tokens.len > 0) seen.with_tokens += 1;
        },
        .discard => seen.discards += 1,
    }
    return .ok;
}

fn tRelease(_: *anyopaque, slot: u32) callconv(.c) void {
    if (slot == seen.slot) seen.released += 1;
}

fn tMeters(_: *anyopaque, slot: u32, out: *abi.Str) callconv(.c) void {
    if (slot == seen.slot) seen.meters += 1;
    out.* = abi.Str.of("[qsa-arms] qsa=test calls=1\n");
}

const test_table: abi.ModelTable = blk: {
    var t = llama.table;
    t.forward = &tForward;
    t.forward_settle = &tSettle;
    t.release_slot = &tRelease;
    t.meters = &tMeters;
    break :blk t;
};

/// A fresh request on the model's own cache.
fn decode(x: *Transformer, out: []u32) !void {
    try x.resetCache();
    var tok: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5 };
    var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), testing.allocator, x, &tok, &prompt, @intCast(out.len), .{ .temperature = 0.0 }, &.{}, .{});
    defer gen.deinit(testing.allocator);
    for (out) |*t| t.* = (try gen.next(testing.allocator)).?;
    gen.logQsaArms();
}

// ── tests ──

test "plugin settle: deferred decode steps settle where the plain served path does, ids stay put, meters log, the slot is released" {
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x5e77);
    defer b.close();
    var want: [10]u32 = undefined;
    try decode(&b.xfm, &want);
    b.xfm.plugin.?.table = &test_table;

    // An owed input settles right after every deferred build, one slot at a time.
    seen = .{ .owes = .input };
    var got: [10]u32 = undefined;
    try decode(&b.xfm, &got);
    try testing.expectEqualSlices(u32, &want, &got);
    try testing.expect(seen.deferred > 0 and seen.slot != 0);
    try testing.expectEqual(seen.deferred, seen.flushes);
    try testing.expectEqual(seen.flushes, seen.flushed_slots);
    try testing.expectEqual(@as(u32, 0), seen.discards);
    try testing.expectEqual(@as(u32, 1), seen.meters);
    try testing.expectEqual(@as(u32, 1), seen.released);

    // Owed history waits for the plugin's next forward; the one the last step leaves dies with the slot.
    seen = .{ .owes = .history };
    try decode(&b.xfm, &got);
    try testing.expectEqualSlices(u32, &want, &got);
    try testing.expect(seen.deferred > 0);
    try testing.expectEqual(@as(u32, 0), seen.flushes);
    try testing.expectEqual(@as(u32, 1), seen.discards);
    try testing.expectEqual(@as(u32, 1), seen.released);
}

test "plugin settle: a batched tick settles its slots in one call" {
    const a = testing.allocator;
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x5e78);
    defer b.close();
    b.xfm.plugin.?.table = &test_table;
    seen = .{ .owes = .input, .one_slot = false };
    var slots: [2]proof.Slot = undefined;
    for (&slots) |*s| try s.init(b.layers);
    defer for (&slots) |*s| s.deinit();
    var g = ops.Graph.init(a, fixture.stream());
    defer g.deinit();
    for (&slots, [_][]const i32{ &.{ 4, 8, 15 }, &.{ 42, 7 } }) |*s, pr| _ = try g.adopt(try b.xfm.forwardWith(&s.ctx, try fixture.tokens(&g, pr)));
    try testing.expect(slots[0].ctx.plugin_slot != slots[1].ctx.plugin_slot);
    const ctxs = [_]*ForwardCtx{ &slots[0].ctx, &slots[1].ctx };
    const offs = [_]u32{ @intCast(slots[0].kv.step), @intCast(slots[1].kv.step) };
    const rows = try b.xfm.forwardBatchedDecode(&.{ 3, 9 }, &ctxs, &offs);
    defer a.free(rows);
    for (rows) |r| _ = try g.adopt(r);
    try testing.expectEqual(@as(u32, 1), seen.flushes);
    try testing.expectEqual(@as(u32, 2), seen.flushed_slots);
    for (ctxs) |c| try testing.expectEqual(abi.Owed.none, c.plugin_owed);
}
