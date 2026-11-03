//! The sparse block at a second shape, without qwen4_exp: 128-key blocks, top 16 with one init and one local block,
//! `key_max` scores per index head (MiniMax-M3's sparse path), held to the float64 reference.

const std = @import("std");
const mlx = @import("../../../../mlx/ffi.zig");
const ops = @import("../../../../mlx/ops.zig");
const params = @import("params.zig");
const reference = @import("reference.zig");
const keyMaxScores = @import("score.zig").keyMaxScores;
const selectPicks = @import("select.zig").selectPicks;
const picksMask = @import("mask.zig").picksMask;
const qsaMaskArm = @import("mask.zig").qsaMaskArm;

const testing = std.testing;

/// Values k/64 in [-1, 1]: exact in TF32, so every q . k is exact and the scores equal float64's.
fn fill(rnd: std.Random, out: []f32) void {
    for (out) |*x| x.* = @as(f32, @floatFromInt(rnd.intRangeAtMost(i32, -64, 64))) / 64.0;
}

fn host(g: *ops.Graph, x: mlx.mlx_array) ![]const f32 {
    const f = try g.contiguous(try g.astype(x, .float32));
    try ops.eval(f);
    return (mlx.mlx_array_data_float32(f) orelse return error.Unreadable)[0..mlx.mlx_array_size(f)];
}

test "sparse block: 128-key blocks, top 16, one init and one local block, key max, against the float64 reference" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const p: params.Params = .{ .blocks = .{ .size = 128, .top_k = 16, .init = 1, .local = 1 }, .index_heads = 4, .index_dim = 64, .score = .key_max };
    try p.validate();
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const hq = 8;
    const hk = 4;
    const d = 64;
    const lists: usize = p.listsPerQuery();
    const scale: f32 = 0.125;
    // A partial last block, a cache ending on a block edge, and one with fewer blocks than top_k.
    const cases = [_]struct { kv: c_int, rows: c_int }{
        .{ .kv = 128 * 24 + 37, .rows = 1 }, .{ .kv = 128 * 24 + 37, .rows = 40 },
        .{ .kv = 128 * 24, .rows = 1 },      .{ .kv = 128 * 24, .rows = 40 },
        .{ .kv = 128 * 5 + 1, .rows = 24 },
    };
    var prng = std.Random.DefaultPrng.init(0x5BA2_5E);
    const rnd = prng.random();
    var checked_rows: usize = 0;
    for (cases) |c| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const kv: usize = @intCast(c.kv);
        const rows: usize = @intCast(c.rows);
        const row0 = c.kv - c.rows;
        const qi = try a.alloc(f32, lists * rows * d);
        defer a.free(qi);
        const ki = try a.alloc(f32, kv * d);
        defer a.free(ki);
        const q = try a.alloc(f32, hq * rows * d);
        defer a.free(q);
        const k = try a.alloc(f32, hk * kv * d);
        defer a.free(k);
        const v = try a.alloc(f32, hk * kv * d);
        defer a.free(v);
        for ([_][]f32{ qi, ki, q, k, v }) |x| fill(rnd, x);

        const scores = try g.adopt(try keyMaxScores(a, s, p.blocks, try g.fromSlice(f32, qi, &.{ 1, @intCast(lists), c.rows, d }), try g.fromSlice(f32, ki, &.{ 1, 1, c.kv, d }), row0));
        const nb: usize = @intCast(ops.shape(scores)[2]);
        try testing.expectEqual((kv + 127) / 128, nb);
        const got_scores = try host(&g, scores);
        const want_scores = try a.alloc(f64, nb);
        defer a.free(want_scores);
        const row_scores = try a.alloc(f64, nb);
        defer a.free(row_scores);
        const picks = try g.adopt(try selectPicks(a, s, p.blocks, scores, row0, c.rows));
        const kb: usize = @intCast(ops.shape(picks)[2]);
        try testing.expectEqual(@min(nb, 16), kb);
        try ops.eval(picks);
        const got_picks = (mlx.mlx_array_data_int32(picks) orelse return error.Unreadable)[0 .. lists * rows * kb];
        var want_picks: [16]i32 = undefined;
        for (0..lists) |l| for (0..rows) |r| {
            const pos = @as(usize, @intCast(row0)) + r;
            const row = l * rows + r;
            reference.keyMaxScores(want_scores, p.blocks, qi[row * d ..][0..d], ki, d, pos);
            for (want_scores, got_scores[row * nb ..][0..nb], row_scores) |w, gv, *rs| {
                try testing.expectEqual(w, @as(f64, gv));
                rs.* = gv;
            }
            const n = reference.select(p.blocks, row_scores, pos, &want_picks);
            const got = got_picks[row * kb ..][0..kb];
            try testing.expectEqualSlices(i32, want_picks[0..n], got[0..n]);
            for (got[n..]) |x| try testing.expectEqual(@as(i32, std.math.maxInt(i32)), x);
        };

        const mask = try g.adopt(try picksMask(a, s, p.blocks, picks, c.rows, c.kv, hq));
        const qa = try g.fromSlice(f32, q, &.{ 1, hq, c.rows, d });
        const ka = try g.fromSlice(f32, k, &.{ 1, hk, c.kv, d });
        const va = try g.fromSlice(f32, v, &.{ 1, hk, c.kv, d });
        const out = try host(&g, try g.adopt(try qsaMaskArm(s, qa, ka, va, scale, mask)));
        var want: [d]f64 = undefined;
        for (0..hq) |h| for (0..rows) |r| {
            const l = h / (hq / lists);
            const kvh = h / (hq / hk);
            const pos = @as(usize, @intCast(row0)) + r;
            const row = l * rows + r;
            var n: usize = 0;
            while (n < kb and got_picks[row * kb + n] != std.math.maxInt(i32)) n += 1;
            reference.attend(&want, p.blocks, q[(h * rows + r) * d ..][0..d], k[kvh * kv * d ..][0 .. kv * d], v[kvh * kv * d ..][0 .. kv * d], d, scale, got_picks[row * kb ..][0..n], pos);
            for (want, out[(h * rows + r) * d ..][0..d]) |w, o| try testing.expectApproxEqAbs(w, @as(f64, o), 2e-3);
            checked_rows += 1;
        };
    }
    try testing.expectEqual(@as(usize, 8 * (1 + 40 + 1 + 40 + 24)), checked_rows);
}
