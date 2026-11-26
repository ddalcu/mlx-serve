//! The served model's own block drafter (`SpecHooks.native_drafter`) on the generator's DSpark rounds, against a
//! test drafter over the llama plugin that never drafts: each round forwards `t1` through the served trunk and hands
//! its argmax back, so the rounds decode the plain served stream. A forced commit (the reasoning budget's closer)
//! leaves its last token in the model's state: the next round publishes it and never feeds it to the drafter again.
//! A second test drafter verifies several rows on the slot's own cache (`Forward.caches`) and leaves the rejected ones
//! there: the engine's settle drops them (`NativeDrafter`'s rows rule), so every layer holds the committed positions
//! after each round, on every tick, and in what the request leaves for the prefix cache. A third breaks the rule.

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
            // Every layer holds exactly the committed positions (`NativeDrafter`'s rows rule).
            try expectSettled(&gen.ctx);
        } else {
            out[n] = (try gen.next(a)) orelse break;
            n += 1;
        }
    }
    try testing.expectEqual(out.len, n);
}

/// Every written layer holds `cache.step` rows, and a recurrent model's forward position is the same.
fn expectSettled(ctx: *const ForwardCtx) !void {
    for (ctx.cache.entries) |e| if (e.initialized) try testing.expectEqual(ctx.cache.step, e.offset);
    try testing.expectEqual(ctx.cache.step, ctx.moe_seq_offset.*);
}

/// The plain stream the drafting test drafters copy their drafts from, after the prompt.
var oracle: struct { want: []const u32 = &.{}, prompt_len: u32 = 0 } = .{};

/// Up to three drafts after `t1` at `pos`, copied from the plain stream, the second one wrong on every other position.
fn oracleDrafts(t1: u32, pos: u64, ids: *[4]u32) u32 {
    const idx = pos - oracle.prompt_len;
    const k: u32 = @intCast(@min(3, oracle.want.len -| (idx + 1)));
    ids[0] = t1;
    for (0..k) |i| ids[1 + i] = oracle.want[idx + 1 + i];
    if (k > 1 and idx % 2 == 0) ids[2] = (ids[2] + 1) % 64;
    return k;
}

/// `ids` verified as one forward of the served llama plugin on the slot's own cache (`f.caches`): every row appended,
/// the logits `[rows, vocab]`.
fn verifyRows(m: *anyopaque, f: *const abi.Forward, ids: []const u32) !abi.Array {
    if (f.caches.len != 1 or f.cache_ops == null) return error.NoEngineCache;
    const f2: abi.Forward = .{ .mode = .verify, .logits = .all_rows, .rows = @intCast(ids.len), .tokens = .of(ids), .slots = f.slots, .positions = f.positions, .caches = f.caches, .cache_ops = f.cache_ops };
    var o2: abi.ForwardOut = .{};
    if (!llama.table.forward(m, &f2, &o2).isOk()) return error.VerifyRefused;
    return o2.logits;
}

fn dStep(m: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    draftingRound(m, f, out) catch return abi.Verdict.refuse(.failed, "test drafting round");
    return .ok;
}

/// A greedy round that keeps the drafts the verify rows' argmax agrees with and leaves every verified row in the cache.
fn draftingRound(m: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const pos = f.positions.slice()[0];
    var ids: [4]u32 = undefined;
    const k = oracleDrafts(f.tokens.slice()[0], pos, &ids);
    var g = ops.Graph.init(testing.allocator, trunk.x.s);
    defer g.deinit();
    const logits = try g.adopt(.{ .ctx = (try verifyRows(m, f, ids[0 .. k + 1])).ctx });
    const arg = try g.astype(try g.argmax(try g.reshape(logits, &.{ @intCast(k + 1), -1 }), -1, false), .int32);
    try mlx.check(mlx.mlx_array_eval(arg));
    const best = mlx.mlx_array_data_int32(arg) orelse return error.NoData;
    var a: u32 = 0;
    while (a < k and a < f.accept_cap and ids[1 + a] == @as(u32, @intCast(best[a]))) a += 1;
    out.round = .{ .n = 1 + a, .accepted = a, .next_token = @intCast(best[a]), .rows = k + 1, .position = pos + 1 + a };
    @memcpy(out.round.tokens[0 .. 1 + a], ids[0 .. 1 + a]);
}

const drafting: abi.NativeDrafter = .{ .step = &dStep };
const drafting_hooks: abi.SpecHooks = .{ .native_drafter = &drafting, .trunk_weight = llama.table.spec.?.trunk_weight };
const drafting_table: abi.ModelTable = blk: {
    var t = llama.table;
    t.spec = &drafting_hooks;
    break :blk t;
};

const every_round: abi.NativeDrafter = .{ .step = &dStep, .every_round = true };
const every_round_hooks: abi.SpecHooks = .{ .native_drafter = &every_round, .trunk_weight = llama.table.spec.?.trunk_weight };
const every_round_table: abi.ModelTable = blk: {
    var t = llama.table;
    t.spec = &every_round_hooks;
    break :blk t;
};

/// A sampled drafter's open verify: where it started and how many drafts it verified.
var open_verify: struct { p0: u64 = 0, drafts: u32 = 0 } = .{};

fn sVerify(m: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const pos = f.positions.slice()[0];
    var ids: [4]u32 = undefined;
    const k = @min(open_verify.drafts, oracleDrafts(f.tokens.slice()[0], pos, &ids));
    out.logits = verifyRows(m, f, ids[0 .. k + 1]) catch return abi.Verdict.refuse(.failed, "test verify");
    open_verify.p0 = pos;
    out.round = .{ .n = k + 1, .rows = k + 1, .position = pos };
    @memcpy(out.round.tokens[0 .. k + 1], ids[0 .. k + 1]);
    return .ok;
}

fn sFinish(_: *anyopaque, accepted: u32, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    out.round = .{ .position = open_verify.p0 + 1 + accepted };
    return .ok;
}

const sampled: abi.NativeDrafter = .{ .step = &dStep, .verify = &sVerify, .finish = &sFinish };
const sampled_hooks: abi.SpecHooks = .{ .native_drafter = &sampled, .trunk_weight = llama.table.spec.?.trunk_weight };
const sampled_table: abi.ModelTable = blk: {
    var t = llama.table;
    t.spec = &sampled_hooks;
    break :blk t;
};

/// A round that reports `t1` committed but appends nothing: it breaks the rows rule.
fn shortStep(_: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    out.round = .{ .n = 1, .rows = 1, .next_token = 1, .position = f.positions.slice()[0] + 1 };
    out.round.tokens[0] = f.tokens.slice()[0];
    return .ok;
}

const short: abi.NativeDrafter = .{ .step = &shortStep };
const short_hooks: abi.SpecHooks = .{ .native_drafter = &short, .trunk_weight = llama.table.spec.?.trunk_weight };
const short_table: abi.ModelTable = blk: {
    var t = llama.table;
    t.spec = &short_hooks;
    break :blk t;
};

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

// Bar: a drafter that leaves its rejected rows in the engine's cache decodes the plain stream, every round and every
// plain tick of the backoff after it reading only committed rows.
test "plugin native drafter: a drafter on the engine's cache verifies several rows and the rejected ones leave the cache" {
    const saved = env.override(.MLX_SERVE_DSV4_DSPARK, null);
    defer saved.restore();
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x9a7e);
    defer b.close();
    var want: [12]u32 = undefined;
    try decode(&b.xfm, false, false, want.len, &.{}, &want);
    oracle = .{ .want = &want, .prompt_len = 6 };
    defer oracle = .{};
    b.xfm.plugin.?.table = &drafting_table;
    b.xfm.config.plugin.?.parsed.spec.native_stages = 1;
    for ([_]bool{ false, true }) |paced| {
        var got: [12]u32 = undefined;
        try decode(&b.xfm, true, paced, got.len, &.{}, &got);
        try testing.expectEqualSlices(u32, &want, &got);
    }
}

// Bar: what a request leaves after drafted rounds (the prefix cache's snapshot) holds the committed rows only: restored
// into a fresh slot, it continues the plain stream.
test "plugin native drafter: the cache a drafted request leaves restores to the plain stream" {
    const saved = env.override(.MLX_SERVE_DSV4_DSPARK, null);
    defer saved.restore();
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x9a7e);
    defer b.close();
    var want: [10]u32 = undefined;
    try decode(&b.xfm, false, false, want.len, &.{}, &want);
    oracle = .{ .want = &want, .prompt_len = 6 };
    defer oracle = .{};
    b.xfm.plugin.?.table = &drafting_table;
    b.xfm.config.plugin.?.parsed.spec.native_stages = 1;
    const a = testing.allocator;
    try b.xfm.resetCache();
    var tok: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5 };
    var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, &b.xfm, &tok, &prompt, 8, .{ .temperature = 0.0 }, &.{}, .{ .mtp_enabled = true });
    defer gen.deinit(a);
    gen.dspark_pol = null;
    trunk = .{ .x = &b.xfm, .ctx = &gen.ctx };
    var n: usize = 0;
    while (n < 7) {
        const res = (try gen.nextDspark(a)) orelse break;
        defer a.free(res.tokens);
        try testing.expectEqualSlices(u32, want[n..][0..res.tokens.len], res.tokens);
        n += res.tokens.len;
    }
    try testing.expectEqual(prompt.len + n, gen.ctx.cache.step);
    var snap = try gen.ctx.cache.snapshot();
    defer snap.deinit();
    var s: proof.Slot = undefined;
    try s.init(b.layers);
    defer s.deinit();
    try s.kv.restore(&snap);
    s.off = s.kv.step;
    var g = ops.Graph.init(a, b.xfm.s);
    defer g.deinit();
    const ids = [_]i32{@intCast(want[n])};
    const id = try g.argmax(try g.adopt(try b.xfm.forwardWith(&s.ctx, try fixture.tokens(&g, &ids))), -1, false);
    try mlx.check(mlx.mlx_array_eval(id));
    var next: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&next, id));
    try testing.expectEqual(want[n + 1], @as(u32, @intCast(next)));
}

// Bar: a sampled round settles at its finish from where its verify started, whether the verify appended drafts the
// finish rejects or only `t1` (the verify's own appends moved `cache.step` before the finish).
test "plugin native drafter: a sampled round's finish leaves the committed rows only" {
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x9a7e);
    defer b.close();
    const a = testing.allocator;
    var want: [8]u32 = undefined;
    try decode(&b.xfm, false, false, want.len, &.{}, &want);
    oracle = .{ .want = &want, .prompt_len = 6 };
    defer oracle = .{};
    b.xfm.plugin.?.table = &sampled_table;
    const prompt = [_]i32{ 3, 7, 1, 12, 30, 5 };
    for ([_][2]u32{ .{ 3, 1 }, .{ 0, 0 } }) |c| {
        var s: proof.Slot = undefined;
        try s.init(b.layers);
        defer s.deinit();
        trunk = .{ .x = &b.xfm, .ctx = &s.ctx };
        var g = ops.Graph.init(a, b.xfm.s);
        defer g.deinit();
        _ = try g.adopt(try b.xfm.forwardWith(&s.ctx, try fixture.tokens(&g, &prompt)));
        open_verify = .{ .drafts = c[0] };
        // `t1` at the prompt's end; positions past the oracle's start draft from the plain stream.
        const v = try driver.nativeVerify(&b.xfm, &s.ctx, want[0], 4, &.{});
        _ = mlx.mlx_array_free(v.logits);
        try testing.expectEqual(@as(usize, prompt.len + 1 + c[0]), s.kv.step);
        try testing.expectEqual(@as(u64, prompt.len + 1 + c[1]), try driver.nativeFinish(&b.xfm, &s.ctx, c[1]));
        try testing.expectEqual(@as(usize, prompt.len + 1 + c[1]), s.kv.step);
        try expectSettled(&s.ctx);
    }
}

// Bar: a round that commits more than it left in the cache fails the request instead of reading a short history.
test "plugin native drafter: a round that appends less than it commits fails" {
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x9a7e);
    defer b.close();
    b.xfm.plugin.?.table = &short_table;
    var s: proof.Slot = undefined;
    try s.init(b.layers);
    defer s.deinit();
    var g = ops.Graph.init(testing.allocator, b.xfm.s);
    defer g.deinit();
    _ = try g.adopt(try b.xfm.forwardWith(&s.ctx, try fixture.tokens(&g, &.{ 3, 7, 1 })));
    try testing.expectError(error.PluginForwardFailed, driver.nativeRound(&b.xfm, &s.ctx, 12, 4, &.{}));
    try testing.expectEqual(@as(usize, 3), s.kv.step);
}

/// A greedy request of `out.len` tokens of DSpark rounds under the policy init gives `x`'s drafter; the plain ticks it ran.
fn decodeRounds(x: *Transformer, out: []u32) !u64 {
    const a = testing.allocator;
    try x.resetCache();
    var tok: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5 };
    var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, x, &tok, &prompt, @intCast(out.len), .{ .temperature = 0.0 }, &.{}, .{ .mtp_enabled = true });
    defer gen.deinit(a);
    try testing.expect(gen.dspark_enabled);
    trunk = .{ .x = x, .ctx = &gen.ctx };
    var n: usize = 0;
    while (n < out.len) {
        const res = (try gen.nextDspark(a)) orelse break;
        defer a.free(res.tokens);
        @memcpy(out[n..][0..res.tokens.len], res.tokens);
        n += res.tokens.len;
        try expectSettled(&gen.ctx);
    }
    try testing.expectEqual(out.len, n);
    return gen.dspark_plain;
}

// Bar: a drafter that declares every round (`NativeDrafter.every_round`) gets no drafting backoff, one that does not
// gets the engine's: init's policy is null for the first, the backoff for the second.
test "plugin native drafter: a drafter that drafts every round gets no backoff, another gets the engine's" {
    const saved = env.override(.MLX_SERVE_DSV4_DSPARK, null);
    defer saved.restore();
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x9a7e);
    defer b.close();
    b.xfm.config.plugin.?.parsed.spec.native_stages = 1;
    const a = testing.allocator;
    var tok: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1 };
    for ([_]bool{ true, false }) |every| {
        b.xfm.plugin.?.table = if (every) &every_round_table else &drafting_table;
        try testing.expectEqual(every, driver.nativeEveryRound(&b.xfm));
        try b.xfm.resetCache();
        var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, &b.xfm, &tok, &prompt, 4, .{ .temperature = 0.0 }, &.{}, .{ .mtp_enabled = true });
        defer gen.deinit(a);
        try testing.expect(gen.dspark_enabled);
        try testing.expectEqual(every, gen.dspark_pol == null);
    }
}

// Bar: an every-round drafter's request runs only rounds (no plain tick) and decodes the plain stream; the same drafter
// without the declaration runs the backoff's plain ticks (the red check) and decodes the same stream.
test "plugin native drafter: an every-round drafter runs no plain tick and decodes the plain stream" {
    const saved = env.override(.MLX_SERVE_DSV4_DSPARK, null);
    defer saved.restore();
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x9a7e);
    defer b.close();
    var want: [12]u32 = undefined;
    try decode(&b.xfm, false, false, want.len, &.{}, &want);
    oracle = .{ .want = &want, .prompt_len = 6 };
    defer oracle = .{};
    b.xfm.config.plugin.?.parsed.spec.native_stages = 1;
    for ([_]bool{ true, false }) |every| {
        b.xfm.plugin.?.table = if (every) &every_round_table else &drafting_table;
        var got: [12]u32 = undefined;
        const plain = try decodeRounds(&b.xfm, &got);
        try testing.expectEqualSlices(u32, &want, &got);
        if (every) try testing.expectEqual(@as(u64, 0), plain) else try testing.expect(plain > 0);
    }
}
