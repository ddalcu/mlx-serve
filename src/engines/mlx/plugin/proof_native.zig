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
const rounds = @import("../../../sdk/spec/rounds.zig");
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Generator = @import("../generate/generator.zig").Generator;
const Tokenizer = @import("../../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const Transformer = @import("../transformer.zig").Transformer;

const testing = std.testing;

/// The served trunk the test drafter forwards on, and the request's context.
var trunk: struct { x: *Transformer = undefined, ctx: *ForwardCtx = undefined } = .{};

/// The stop ids the test drafter's last round was handed, and the rounds it has run.
var seen_stops: [8]u32 = undefined;
var seen_stop_count: usize = 0;
var rounds_run: usize = 0;

/// A ladder on which a block that lands pays: a round costs little over a plain step.
fn cheapLadder() rounds.Ladder {
    var l: rounds.Ladder = .{ .plain = 10, .copies = undefined, .drafts = undefined, .block = 7 };
    for (&l.copies, &l.drafts, 0..) |*c, *d, k| {
        c.* = 11 + @as(f32, @floatFromInt(k));
        d.* = 12 + @as(f32, @floatFromInt(k));
    }
    return l;
}

fn nStep(_: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const stops = f.stop_ids.slice();
    seen_stop_count = @min(stops.len, seen_stops.len);
    @memcpy(seen_stops[0..seen_stop_count], stops[0..seen_stop_count]);
    rounds_run += 1;
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
    b.xfm.config.plugin.?.parsed.spec.native_block = 8;
    b.xfm.plugin.?.ladder = cheapLadder();
    // Every tick a round; then under the request's controller, whose rounds give way to plain ticks as the test
    // drafter's blocks keep verifying nothing, the commit landing among them.
    for ([_]bool{ false, true }) |paced| {
        var got: [10]u32 = undefined;
        rounds_run = 0;
        try decode(&b.xfm, true, paced, 3, &forced, &got);
        try testing.expectEqualSlices(u32, &want, &got);
        try testing.expect(rounds_run > 0 and (!paced or rounds_run < got.len - forced.len));
    }
}

// Bar: a model whose rounds are not timed yet runs none; the request its load times them with runs every cell of the
// ladder, ends by itself and leaves the ladder with the model, and the next request's controller reads it.
test "plugin native drafter: no rounds before the load's timing request, whose ladder the next request reads" {
    const saved = env.override(.MLX_SERVE_DSV4_DSPARK, null);
    defer saved.restore();
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x9a7e);
    defer b.close();
    b.xfm.plugin.?.table = &test_table;
    b.xfm.config.plugin.?.parsed.spec.native_stages = 1;
    b.xfm.config.plugin.?.parsed.spec.native_block = 8;
    const a = testing.allocator;
    var tok: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5 };
    const io = std.Io.Threaded.global_single_threaded.io();
    {
        try b.xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, &b.xfm, &tok, &prompt, 6, .{ .temperature = 0.0 }, &.{}, .{ .mtp_enabled = true });
        defer gen.deinit(a);
        try testing.expect(gen.dspark_pol.?.rounds == null);
        trunk = .{ .x = &b.xfm, .ctx = &gen.ctx };
        rounds_run = 0;
        while (try gen.nextDspark(a)) |res| a.free(res.tokens);
        try testing.expectEqual(@as(usize, 0), rounds_run);
    }
    {
        try b.xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, &b.xfm, &tok, &prompt, 2048, .{ .temperature = 0.0 }, &.{}, .{ .mtp_enabled = true });
        defer gen.deinit(a);
        gen.timeRounds();
        trunk = .{ .x = &b.xfm, .ctx = &gen.ctx };
        var ticks: usize = 0;
        while (try gen.nextDspark(a)) |res| : (ticks += 1) a.free(res.tokens);
        // Every cell for `REPS + 1` ticks (the first round's untimed one drains the plain steps' pipeline).
        try testing.expectEqual((rounds.Timing.PLAIN_CELLS + rounds.MAX_ROWS + 7) * (rounds.Timing.REPS + 1), ticks);
        try testing.expect(gen.done and gen.completion_tokens < 2048);
        gen.keepLadder();
    }
    const l = b.xfm.plugin.?.ladder.?;
    try testing.expectEqual(@as(u32, 7), l.block);
    for ([_]f32{l.plain} ++ l.copies ++ l.drafts[0..8].*) |ms| try testing.expect(std.math.isFinite(ms) and ms > 0);
    try b.xfm.resetCache();
    var gen = try Generator.initWithOptions(io, a, &b.xfm, &tok, &prompt, 6, .{ .temperature = 0.0 }, &.{}, .{ .mtp_enabled = true });
    defer gen.deinit(a);
    try testing.expectEqual(l.plain, gen.dspark_pol.?.rounds.?.ladder.plain);
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

/// The slot the model's forwards were asked to continue (`Forward.adopt_slot`), 0 when none was.
var seen_adopt: u32 = 0;
fn adoptSeeingForward(m: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    if (f.adopt_slot != 0) seen_adopt = f.adopt_slot;
    return llama.table.forward(m, f, out);
}
const adopt_table: abi.ModelTable = blk: {
    var t = test_table;
    t.forward = &adoptSeeingForward;
    break :blk t;
};

// Bar: a request that must answer the same twice (greedy, or sampled with a seed) takes neither the drafter state nor
// the answer of the request it repeats; an unseeded sampled one continues both.
test "plugin native drafter: only an unseeded sampled request continues an earlier one" {
    const saved = env.override(.MLX_SERVE_DSV4_DSPARK, null);
    defer saved.restore();
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x9a7e);
    defer b.close();
    b.xfm.plugin.?.table = &adopt_table;
    b.xfm.config.plugin.?.parsed.spec.native_stages = 1;
    const a = testing.allocator;
    var tok: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5 };
    const answer = [_]u32{ 9, 9, 4, 2 };
    driver.noteSequence(&b.xfm, 7, &prompt, &answer);
    const Case = struct { temperature: f32, seed: ?u64, slot: u32, indexed: usize };
    for ([_]Case{
        .{ .temperature = 0.0, .seed = null, .slot = 0, .indexed = prompt.len },
        .{ .temperature = 1.0, .seed = 42, .slot = 0, .indexed = prompt.len },
        .{ .temperature = 1.0, .seed = null, .slot = 7, .indexed = prompt.len + answer.len },
    }) |case| {
        try b.xfm.resetCache();
        seen_adopt = 0;
        var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, &b.xfm, &tok, &prompt, 4, .{ .temperature = case.temperature, .seed = case.seed }, &.{}, .{ .mtp_enabled = true });
        defer gen.deinit(a);
        try testing.expectEqual(case.slot, seen_adopt);
        try testing.expectEqual(case.indexed, (try gen.mtpLookupIndex(a)).toks.items.len);
    }
}
