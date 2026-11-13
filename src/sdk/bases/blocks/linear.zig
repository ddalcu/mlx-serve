//! copied_from: sdk/blocks/linear/qmatmul.zig (`qmatmul`, `qmatmulMaybeBias` without the row-exact arm).
//! A projection over its own quant geometry, and a row-joined group of them.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const qmatmulBits = @import("../../blocks/linear/qmatmul.zig").qmatmulBits;
const sliceQkvPart = @import("../../blocks/linear/fused_rows.zig").sliceQkvPart;
const qmv2 = @import("../../blocks/qmm/qmv2.zig");
const qmv_nax2 = @import("../../blocks/qmm/qmv_nax2.zig");
const qmm_int8 = @import("../../blocks/qmm/qmm_int8.zig");
const rht = @import("../../quant/hadamard/rht.zig");
const outside = @import("../../quant/outside.zig");

const Array = ops.Array;
const Graph = ops.Graph;

/// `x @ w^T` (+ an additive `bias`). A dense `w` is stored pre-transposed `[in, out]` with null scales.
pub const Linear = struct {
    w: Array = .{},
    s: Array = .{},
    b: Array = .{},
    /// The additive projection bias (Qwen2's q/k/v, BERT); null = none.
    bias: Array = .{},
    bits: u32 = 0,
    group_size: u32 = 0,
    mode: QuantMode = .affine,
    /// A ternary 2-bit pack (every bias exactly -scale): decode and verify rows take qmv2's kernels, as the engine's.
    ternary: bool = false,
    /// A Hadamard (Prism) pack's registry: every projection of the pack takes the rotated arm, reading its input as
    /// `H_block(signs * x)` through the registry's memo (`signs` pooled there; null = an unrotated weight).
    rht: ?*rht.Registry = null,
    signs: Array = .{},
    /// A projection an outside quant plugin loaded (`sdk/quant/outside_model.zig`): its table runs the matmul. `w` is
    /// then the plugin's first array, so a presence check holds; `s` stays unset.
    ext: ?outside.Weight = null,

    pub fn forward(l: *const Linear, g: *Graph, x: Array) !Array {
        if (l.bias.ctx == null) return g.adopt(try l.matmul(g, x));
        var t = g.scope();
        defer t.deinit();
        return g.keep(try t.add(try t.adopt(try l.matmul(g, x)), l.bias));
    }

    /// Rows one call takes with each row given its one-row call's bytes: the outside table's declared width for a
    /// projection it runs (`QuantTable.exact_rows`), else one (MLX picks a matmul kernel by the row count).
    pub fn exactRows(l: *const Linear) c_int {
        const n: u32 = if (l.ext) |*e| e.exactRows() else 1;
        return @intCast(@min(n, 1 << 16));
    }

    /// `forward` with each row of one sequence's `x` `[1, S, in]` given its own one-row call's bytes: calls of
    /// `exactRows` rows.
    pub fn forwardRows(l: *const Linear, g: *Graph, x: Array) !Array {
        const Call = struct {
            l: *const Linear,
            g: *Graph,
            fn run(c: @This(), part: Array) anyerror!Array {
                return c.l.matmul(c.g, part);
            }
        };
        const y = try g.adopt(try eachRows(x, l.exactRows(), g.s, Call{ .l = l, .g = g }, Call.run));
        return if (l.bias.ctx == null) y else g.add(y, l.bias);
    }

    fn matmul(l: *const Linear, g: *Graph, x: Array) !Array {
        if (l.ext) |*e| return e.matmul(x, g.s);
        if (l.rht) |reg| return l.rotated(reg, x, g.s);
        if (l.ternary) {
            if (try qmv_nax2.qmm(x, l.w, l.s, l.b, l.bits, l.group_size, g.s)) |y| return y;
            if (try qmv2.qmm(x, l.w, l.s, l.b, l.bits, l.group_size, true, false, g.s)) |y| return y;
        }
        return qmatmulBits(x, l.w, l.s, l.b, l.bits, l.group_size, l.mode, g.s);
    }

    /// The engine's rotated arm: the 2-bit kernels first (each declines what it does not serve), then the opt-in int8
    /// prompt route, then stock qmm.
    fn rotated(l: *const Linear, reg: *rht.Registry, x: Array, s: ops.Stream) !Array {
        // With int8 on, a prompt-width rotation rides the quantizer: the rotated activation is never written.
        if (reg.int8) |ic| if (l.signs.ctx != null) if (try qmm_int8.qmmRotated(ic, x, l.signs, reg.block, l.w, l.s, l.b, l.bits, l.group_size, reg.bias_is_neg_scale, s)) |y| return y;
        const xr = if (l.signs.ctx != null) try reg.applyIn(x, l.signs, s) else x;
        defer if (l.signs.ctx != null) {
            _ = mlx.mlx_array_free(xr);
        };
        if (try qmv_nax2.qmm(xr, l.w, l.s, l.b, l.bits, l.group_size, s)) |y| return y;
        if (try qmv2.qmm(xr, l.w, l.s, l.b, l.bits, l.group_size, l.signs.ctx != null and reg.bias_is_neg_scale, true, s)) |y| return y;
        if (reg.int8) |ic| if (try qmm_int8.qmm(ic, xr, l.w, l.s, l.b, l.bits, l.group_size, l.signs.ctx != null and reg.bias_is_neg_scale, s)) |y| return y;
        return qmatmulBits(xr, l.w, l.s, l.b, l.bits, l.group_size, l.mode, s);
    }
};

/// `f(ctx, part)` over the rows of `x` (axis -2) in parts of at most `width` rows, joined on that axis: a new array.
pub fn eachRows(x: Array, width: c_int, s: ops.Stream, ctx: anytype, comptime f: fn (@TypeOf(ctx), Array) anyerror!Array) !Array {
    const sh = mlx.getShape(x);
    const nd = sh.len;
    const rows = sh[nd - 2];
    if (rows <= width) return f(ctx, x);
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    var lo: [8]c_int = @splat(0);
    var hi: [8]c_int = undefined;
    const ones: [8]c_int = @splat(1);
    @memcpy(hi[0..nd], sh);
    var at: c_int = 0;
    while (at < rows) : (at += width) {
        lo[nd - 2] = at;
        hi[nd - 2] = @min(rows, at + width);
        var part = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(part);
        try mlx.check(mlx.mlx_slice(&part, x, &lo, nd, &hi, nd, &ones, nd, s));
        const y = try f(ctx, part);
        defer _ = mlx.mlx_array_free(y);
        _ = mlx.mlx_vector_array_append_value(parts, y);
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, parts, @intCast(nd - 2), s));
    return out;
}

/// Projections that read one input joined along their output rows (one matmul at decode width); null `lin.w` =
/// not built.
pub const Joined = struct {
    lin: Linear = .{},
    widths: [4]c_int = @splat(0),

    pub fn built(j: *const Joined) bool {
        return j.lin.w.ctx != null;
    }

    /// Columns of part `i` of a joined matmul result `y`.
    pub fn part(j: *const Joined, g: *Graph, y: Array, i: usize) !Array {
        var start: c_int = 0;
        for (j.widths[0..i]) |w| start += w;
        var out = mlx.mlx_array_new();
        sliceQkvPart(&out, y, start, j.widths[i], g.s) catch |e| {
            _ = mlx.mlx_array_free(out);
            return e;
        };
        return g.adopt(out);
    }
};

// ── tests ──

test "linear: forwardRows gives each row of an affine projection its own one-row forward's bytes" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const testing = std.testing;
    const s = mlx.gpuStream();
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    const in: usize = 512;
    const rows = 6;
    var prng = std.Random.DefaultPrng.init(11);
    const wbuf = try testing.allocator.alloc(f32, 256 * in);
    defer testing.allocator.free(wbuf);
    for (wbuf) |*v| v.* = prng.random().floatNorm(f32) * 0.05;
    const dense = try g.astype(try g.adopt(mlx.mlx_array_new_data(wbuf.ptr, &[_]c_int{ 256, @intCast(in) }, 2, .float32)), .bfloat16);
    var q = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(q);
    try mlx.check(mlx.mlx_quantize(&q, dense, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
    var parts: [3]Array = undefined;
    for (&parts, 0..) |*p, i| {
        var t = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&t, q, i));
        p.* = try g.adopt(t);
    }
    const l: Linear = .{ .w = parts[0], .s = parts[1], .b = parts[2], .bits = 4, .group_size = 64 };
    try testing.expectEqual(@as(c_int, 1), l.exactRows());
    const xbuf = try testing.allocator.alloc(f32, rows * in);
    defer testing.allocator.free(xbuf);
    for (xbuf) |*v| v.* = prng.random().floatNorm(f32);
    const x = try g.astype(try g.adopt(mlx.mlx_array_new_data(xbuf.ptr, &[_]c_int{ 1, rows, @intCast(in) }, 3, .float32)), .bfloat16);
    // Each row from a buffer of its own, as a serial decode step hands it over.
    var want: [rows]Array = undefined;
    for (&want, 0..) |*w, r| {
        const one = try g.astype(try g.adopt(mlx.mlx_array_new_data(xbuf[r * in ..].ptr, &[_]c_int{ 1, 1, @intCast(in) }, 3, .float32)), .bfloat16);
        w.* = try l.forward(&g, one);
    }
    try ops.expectBitEqual(try l.forwardRows(&g, x), try g.concat(&want, 1));
}
