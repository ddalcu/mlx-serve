//! copied_from: sdk/blocks/linear/qmatmul.zig (`qmatmul`, `qmatmulMaybeBias` without the row-exact arm).
//! A projection over its own quant geometry, and a row-joined group of them.

const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const qmatmulBits = @import("../../blocks/linear/qmatmul.zig").qmatmulBits;
const sliceQkvPart = @import("../../blocks/linear/fused_rows.zig").sliceQkvPart;
const qmv2 = @import("../../blocks/qmm/qmv2.zig");
const qmv_nax2 = @import("../../blocks/qmm/qmv_nax2.zig");
const qmm_int8 = @import("../../blocks/qmm/qmm_int8.zig");
const rht = @import("../../quant/hadamard/rht.zig");

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

    pub fn forward(l: *const Linear, g: *Graph, x: Array) !Array {
        if (l.bias.ctx == null) return g.adopt(try l.matmul(g, x));
        var t = g.scope();
        defer t.deinit();
        return g.keep(try t.add(try t.adopt(try l.matmul(g, x)), l.bias));
    }

    fn matmul(l: *const Linear, g: *Graph, x: Array) !Array {
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
