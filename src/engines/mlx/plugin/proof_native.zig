//! The served model's own block drafter (`SpecHooks.native_drafter`) on the generator's DSpark rounds, against a
//! test drafter over the llama plugin that never drafts: each round forwards `t1` through the served trunk and hands
//! its argmax back, so the rounds decode the plain served stream. A forced commit (the reasoning budget's closer)
//! leaves its last token in the model's state: the next round publishes it and never feeds it to the drafter again.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const env = @import("../../../env.zig");
const driver = @import("driver.zig");
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

/// The served trunk the test drafter forwards on, and the request's context.
var trunk: struct { x: *Transformer = undefined, ctx: *ForwardCtx = undefined } = .{};

/// The stop ids the test drafter's last round was handed.
var seen_stops: [8]u32 = undefined;
var seen_stop_count: usize = 0;

fn nStep(_: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const stops = f.stop_ids.slice();
    seen_stop_count = @min(stops.len, seen_stops.len);
    @memcpy(seen_stops[0..seen_stop_count], stops[0..seen_stop_count]);
    plainRound(f, out) catch return abi.Verdict.refuse(.failed, "test native round");
    return .ok;
}

/// `t1` alone, verified by the trunk: no drafts, and the trunk's argmax after it is the next round's `t1`.
fn plainRound(f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const t1 = f.tokens.slice()[0];
    var g = ops.Graph.init(testing.allocator, trunk.x.s);
    defer g.deinit();
    const ids = [_]i32{@intCast(t1)};
    const logits = try g.adopt(try trunk.x.forwardWith(trunk.ctx, try fixture.tokens(&g, &ids)));
    const id = try g.argmax(logits, -1, false);
    try mlx.check(mlx.mlx_array_eval(id));
    var next: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&next, id));
    out.round = .{ .n = 1, .next_token = @intCast(next), .rows = 1, .position = f.positions.slice()[0] + 1 };
    out.round.tokens[0] = t1;
}

const drafter: abi.NativeDrafter = .{ .step = &nStep };
const spec_hooks: abi.SpecHooks = .{ .native_drafter = &drafter, .trunk_weight = llama.table.spec.?.trunk_weight };
const test_table: abi.ModelTable = blk: {
    var t = llama.table;
    t.spec = &spec_hooks;
    break :blk t;
};

/// A greedy request of `out.len` tokens on the model's own cache, `forced` committed after the first `at` the way the
/// reasoning budget commits its closer: plain steps, or the test drafter's rounds (every tick one, or under the
/// draft backoff `paced`, whose plain ticks ride the serial pipeline a token ahead).
fn decode(x: *Transformer, native: bool, paced: bool, at: usize, forced: []const u32, out: []u32) !void {
    const a = testing.allocator;
    try x.resetCache();
    var tok: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5 };
    var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, x, &tok, &prompt, @intCast(out.len), .{ .temperature = 0.0 }, &.{}, .{ .mtp_enabled = native });
    defer gen.deinit(a);
    try testing.expectEqual(native, gen.dspark_enabled);
    try testing.expectEqual(native, gen.dspark_pol != null);
    if (!paced) gen.dspark_pol = null;
    trunk = .{ .x = x, .ctx = &gen.ctx };
    var n: usize = 0;
    while (n < out.len) {
        if (n == at) {
            const r = try gen.commitForcedTokens(a, forced);
            defer a.free(r.emitted);
            try testing.expect(!r.stopped);
            @memcpy(out[n..][0..r.emitted.len], r.emitted);
            n += r.emitted.len;
        } else if (native) {
            const res = (try gen.nextDspark(a)) orelse break;
            defer a.free(res.tokens);
            @memcpy(out[n..][0..res.tokens.len], res.tokens);
            n += res.tokens.len;
            // The state holds the prompt and each published token once; `t1` waits outside it.
            if (!paced) try testing.expectEqual(prompt.len + gen.generated_ids.items.len, gen.ctx.cache.step);
        } else {
            out[n] = (try gen.next(a)) orelse break;
            n += 1;
        }
    }
    try testing.expectEqual(out.len, n);
}

// ── tests ──

test "plugin native drafter: the next round publishes a forced commit's last token, never feeds it to the drafter" {
    const saved = env.override(.MLX_SERVE_DSV4_DSPARK, null);
    defer saved.restore();
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x9a7e);
    defer b.close();
    const forced = [_]u32{ 40, 41 };
    var want: [10]u32 = undefined;
    try decode(&b.xfm, false, false, 3, &forced, &want);
    b.xfm.plugin.?.table = &test_table;
    b.xfm.config.plugin.?.parsed.spec.native_stages = 1;
    // Every tick a round; then under the backoff a greedy drafter gets, the commit landing among its plain ticks.
    for ([_]bool{ false, true }) |paced| {
        var got: [10]u32 = undefined;
        try decode(&b.xfm, true, paced, 3, &forced, &got);
        try testing.expectEqualSlices(u32, &want, &got);
    }
}

// Bar: a round is handed the request's stop ids, so the drafter never keeps a row past the end of the turn.
test "plugin native drafter: a round is handed the request's stop ids" {
    const saved = env.override(.MLX_SERVE_DSV4_DSPARK, null);
    defer saved.restore();
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x9a7e);
    defer b.close();
    b.xfm.plugin.?.table = &test_table;
    b.xfm.config.plugin.?.parsed.spec.native_stages = 1;
    const a = testing.allocator;
    try b.xfm.resetCache();
    var tok: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5 };
    const stops = [_]u32{ 61, 62 };
    var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, &b.xfm, &tok, &prompt, 4, .{ .temperature = 0.0 }, &stops, .{ .mtp_enabled = true });
    defer gen.deinit(a);
    gen.dspark_pol = null;
    trunk = .{ .x = &b.xfm, .ctx = &gen.ctx };
    seen_stop_count = 0;
    const res = (try gen.nextDspark(a)) orelse return error.TestUnexpectedStop;
    a.free(res.tokens);
    try testing.expect(gen.eos_token_ids.len > 0);
    try testing.expectEqualSlices(u32, gen.eos_token_ids, seen_stops[0..seen_stop_count]);
}

fn nInactive(_: *anyopaque) callconv(.c) bool {
    return false;
}
const inactive_drafter: abi.NativeDrafter = .{ .step = &nStep, .active = &nInactive };
const inactive_hooks: abi.SpecHooks = .{ .native_drafter = &inactive_drafter, .trunk_weight = llama.table.spec.?.trunk_weight };
const inactive_table: abi.ModelTable = blk: {
    var t = llama.table;
    t.spec = &inactive_hooks;
    break :blk t;
};

// Bar: a drafter its load left out (an M1, the model's kernels off) is no drafter: no stages, so no DSpark arm.
test "plugin native drafter: a load that left its drafter out arms no rounds" {
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x9a7e);
    defer b.close();
    b.xfm.config.plugin.?.parsed.spec.native_stages = 1;
    b.xfm.plugin.?.table = &test_table;
    try testing.expectEqual(@as(u32, 1), driver.nativeStages(&b.xfm));
    b.xfm.plugin.?.table = &inactive_table;
    try testing.expectEqual(@as(u32, 0), driver.nativeStages(&b.xfm));
    const a = testing.allocator;
    var tok: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1 };
    var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, &b.xfm, &tok, &prompt, 4, .{ .temperature = 0.0 }, &.{}, .{ .mtp_enabled = true });
    defer gen.deinit(a);
    try testing.expect(!gen.dspark_enabled);
}
