//! GLM-5.3's indexer top-k on the GPU: each row's `k` largest scores as MLX's `argpartition(scores, n - k)[..., n - k:]`
//! gives them, in one launch. MLX partitions by a stable merge sort of the whole row: the last `k` of the stable
//! ascending order, so the `k` largest values, ties at the threshold taken from the highest indices, in ascending
//! (value, index) order. The kernel finds the `k`-th largest of the composite key (the value's order key, then the
//! index) by a radix select over the row, keeps the `k` elements at or above it, and sorts them ascending: the same ids
//! in the same order. Zeros of either sign tie, as they compare equal; a NaN (the indexer writes none) sorts as its
//! bits do. Not the host's glm5_next TOPK_EXPAND: that selects one decode query's pooled scores and writes its pools
//! in chronological order (ties to the lower index); the prompt needs argpartition's ids, in its order, for every row.
//!
//! The texts (one threadgroup of 1024 threads per row, `K` a template argument; the one-buffer form reads its rows from
//! `scores` at one length `params[0]`, the rows form row `r` from its own input `s<r>` at length `params[r]`) are the
//! pinned registry's (`trunk_kernels`: kernels/glm_moe_dsa/), built and run once at the module's construction.

const std = @import("std");
const env = @import("mlx_host").env;
const host_caps = @import("../../sdk/weights/streamed/host_caps.zig");
const mlx = @import("sdk").mlx;
const ops = @import("../../sdk/weights/streamed/ops.zig");
const trunk = @import("trunk_kernels.zig");

const G = ops.MlxOps;
const T = G.T;

/// The most rows `topkRows` takes in one launch: its lengths ride in `params`' eight slots (a verify has at most
/// `max_route_ids / n_experts_per_tok` = 6 rows).
pub const max_rows = trunk.max_topk_rows;

/// Whether the kernel serves a top `k`: an instantiation the registry registers (`trunk_kernels.topk_instances`; the
/// memory bill charges the selection by it).
pub fn serves(k: u32) bool {
    return trunk.registersTopk(k);
}

/// The top-k route of one load over the trunk set's kernels (`trunk_kernels`), which the module built and ran once at the
/// model's top-k at its construction (the acceptance), never inside a prompt. The module installs it only where the
/// route takes it (a GPU stream, a top-k the registry serves, `serves`); elsewhere the graph's route is argpartition.
pub const Kernels = struct {
    bound: *const trunk.Bound,

    /// The ids `[1, rows, k]` (int32) of each row's `k` largest scores (`scores` `[1, rows, n]` f32, row-contiguous,
    /// `k < n`), as `argpartition(scores, n - k)[..., n - k:]` orders them. The route's invariants (a GPU stream, `k`
    /// served, f32 scores of one leading row, more keys than `k`) are the callers': asserted, never re-checked.
    pub fn topk(ks: *const Kernels, g: *G, scores: T, k: u32) !T {
        const sh = g.shapeOf(scores);
        std.debug.assert(sh.n == 3 and sh.dim(0) == 1 and g.dtypeOf(scores) == .float32 and serves(k) and sh.dim(2) > k);
        const rows = sh.dim(1);
        // Eight slots: an input of fewer is bound in the constant address space.
        const pv = [8]i32{ sh.dim(2), 0, 0, 0, 0, 0, 0, 0 };
        const params = try g.hostArray(std.mem.sliceAsBytes(&pv), &.{8}, .int32);
        return launch(g, ks.bound, .glm_moe_dsa_topk, &.{ scores, params }, trunk.topkLaunch(@intCast(rows), k, false));
    }

    /// The ids `[rows.len, k]` (int32) of each row's `k` largest scores, `rows[r]` one row of `n_r > k` f32 scores
    /// (`[..., n_r]`, every leading dim 1; the lengths may differ), as `topk` gives them, every row in one launch (a row's
    /// threadgroup reads its own input). The invariants are the callers', asserted.
    pub fn topkRows(ks: *const Kernels, g: *G, rows: []const T, k: u32) !T {
        std.debug.assert(rows.len > 0 and rows.len <= max_rows and serves(k));
        var pv: [8]i32 = @splat(0);
        for (rows, 0..) |r, i| {
            const sh = g.shapeOf(r);
            const n = sh.dim(-1);
            std.debug.assert(sh.numel() == n and n > k and g.dtypeOf(r) == .float32);
            pv[i] = n;
        }
        const params = try g.hostArray(std.mem.sliceAsBytes(&pv), &.{8}, .int32);
        var inputs: [max_rows + 1]T = undefined;
        @memcpy(inputs[0..rows.len], rows);
        inputs[rows.len] = params;
        return launch(g, ks.bound, trunk.topkRowsKernel(rows.len), inputs[0 .. rows.len + 1], trunk.topkLaunch(@intCast(rows.len), k, true));
    }
};

/// One launch of the registry's kernel `kern` over `inputs` at `cfg`: its int32 ids, adopted by `g`.
fn launch(g: *G, bound: *const trunk.Bound, kern: trunk.Kernel, inputs: []const T, cfg: trunk.LaunchConfig) !T {
    var outs: [1]mlx.mlx_array = undefined;
    try bound.apply(kern, inputs, &cfg, &outs);
    return g.adopt(outs[0]);
}

const testing = std.testing;

test "glm topk: each row's k largest scores in argpartition's ids and order, ties, both zeros and masked tails included" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    const set = try trunk.openForTest(a, s);
    defer set.deinit();
    const ks: Kernels = .{ .bound = &set.bound };
    var prng = std.Random.DefaultPrng.init(17);
    const rand = prng.random();
    var checked: usize = 0;
    // (rows, n, k, case): distinct values; values on a coarse grid (many ties at the threshold); a masked tail of -1e30
    // past each row's causal keys (fewer than k for some rows); zeros of both signs.
    const cases = [_]struct { rows: u32, n: u32, k: u32, kind: u8 }{
        .{ .rows = 33, .n = 5000, .k = 2048, .kind = 0 },
        .{ .rows = 17, .n = 3000, .k = 2048, .kind = 1 },
        .{ .rows = 40, .n = 4100, .k = 2048, .kind = 2 },
        .{ .rows = 9, .n = 2500, .k = 2048, .kind = 3 },
        .{ .rows = 64, .n = 300, .k = 8, .kind = 1 },
        .{ .rows = 5, .n = 70000, .k = 2048, .kind = 0 },
    };
    for (cases) |cs| {
        const m = g.mark();
        defer g.resetTo(m);
        const v = try a.alloc(f32, cs.rows * cs.n);
        defer a.free(v);
        for (v, 0..) |*x, i| {
            const row = i / cs.n;
            const col = i % cs.n;
            x.* = switch (cs.kind) {
                0 => rand.floatNorm(f32),
                1 => @floor(rand.floatNorm(f32) * 4) / 4,
                2 => if (col > 1800 + row * 37) -1e30 else rand.floatNorm(f32),
                else => switch (rand.uintLessThan(u8, 4)) {
                    0 => 0.0,
                    1 => -0.0,
                    else => rand.floatNorm(f32),
                },
            };
        }
        const sc = try g.hostArray(std.mem.sliceAsBytes(v), &.{ 1, @intCast(cs.rows), @intCast(cs.n) }, .float32);
        const got = try ks.topk(&g, sc, cs.k);
        const kth: c_int = @intCast(cs.n - cs.k);
        const want = try g.astype(try lastSlice(&g, try g.argpartition(sc, kth, -1), kth, @intCast(cs.n)), .int32);
        const gv = try a.alloc(u32, cs.rows * cs.k);
        defer a.free(gv);
        const wv = try a.alloc(u32, cs.rows * cs.k);
        defer a.free(wv);
        _ = try g.hostU32(try g.astype(got, .uint32), gv);
        _ = try g.hostU32(try g.astype(want, .uint32), wv);
        try testing.expectEqualSlices(u32, wv, gv);
        checked += gv.len;
    }
    std.debug.print("glm topk: {d} ids equal argpartition's, in its order\n", .{checked});
}

test "glm topk: rows of their own lengths in one launch, each its k largest in argpartition's ids and order (ties, both zeros)" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    const set = try trunk.openForTest(a, s);
    defer set.deinit();
    const ks: Kernels = .{ .bound = &set.bound };
    var prng = std.Random.DefaultPrng.init(19);
    const rand = prng.random();
    const k: u32 = 2048;
    // A verify's lengths (one key more per row), and rows of other lengths on a coarse grid (ties at the threshold)
    // with zeros of both signs; one row up to max_rows rows.
    const sets = [_][]const u32{ &.{ 16900, 16901, 16902, 16903 }, &.{ 2049, 3000, 5000, 2100, 70000, 4097, 2500, 9000 }, &.{6000} };
    var checked: usize = 0;
    for (sets, 0..) |lens, si| {
        const m = g.mark();
        defer g.resetTo(m);
        var rows: [max_rows]T = undefined;
        var want: [max_rows]T = undefined;
        for (lens, 0..) |n, r| {
            const v = try a.alloc(f32, n);
            defer a.free(v);
            for (v) |*x| x.* = switch (si) {
                0 => rand.floatNorm(f32),
                else => switch (rand.uintLessThan(u8, 4)) {
                    0 => 0.0,
                    1 => -0.0,
                    else => @floor(rand.floatNorm(f32) * 4) / 4,
                },
            };
            rows[r] = try g.hostArray(std.mem.sliceAsBytes(v), &.{ 1, 1, 1, @intCast(n) }, .float32);
            const kth: c_int = @intCast(n - k);
            want[r] = try g.astype(try lastSlice(&g, try g.reshape(try g.argpartition(rows[r], kth, -1), &.{ 1, 1, @intCast(n) }), kth, @intCast(n)), .uint32);
        }
        const got = try ks.topkRows(&g, rows[0..lens.len], k);
        try testing.expect(g.dtypeOf(got) == .int32);
        const gv = try a.alloc(u32, lens.len * k);
        defer a.free(gv);
        _ = try g.hostU32(try g.astype(got, .uint32), gv);
        for (0..lens.len) |r| {
            var wv: [2048]u32 = undefined;
            _ = try g.hostU32(want[r], &wv);
            try testing.expectEqualSlices(u32, &wv, gv[r * k ..][0..k]);
            checked += k;
        }
    }
    std.debug.print("glm topk rows: {d} ids equal argpartition's, in its order\n", .{checked});
}

fn lastSlice(g: *G, x: T, lo: c_int, hi: c_int) !T {
    const s = g.shapeOf(x);
    return g.slice(x, &.{ 0, 0, lo }, &.{ s.dim(0), s.dim(1), hi }, &.{ 1, 1, 1 });
}
