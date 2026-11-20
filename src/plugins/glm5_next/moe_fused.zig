//! GLM-5.3-Flash's one-kernel MoE decode (`glm5.moeDecode`, affine banks, up to `MOE_DECODE_MAX_ROWS` rows), the
//! shared `Moe` block's `fused` arm on GLM's routed layers (the trunk's `Hooks.fused_decode`).

const mlx = @import("../../mlx/ffi.zig");
const moe = @import("../../sdk/bases/blocks/moe.zig");
const Linear = @import("../../sdk/bases/blocks/linear.zig").Linear;
const glm5 = @import("glm5_next.zig");

const Array = mlx.mlx_array;

/// GLM's routed + shared experts in two dispatches; null (another quant mode, geometry or device) takes the block's
/// arms.
pub fn fusedDecode(f: *const moe.Fused, w: *const moe.Moe.Weights, s: mlx.mlx_stream, x: Array, inds: Array, scores: Array) !?Array {
    const se = &(w.shared orelse return null);
    inline for (.{ w.gate.qp, w.up.qp, w.down.qp }) |qp| if (qp.mode != .affine) return null;
    inline for (.{ se.gate, se.up, se.down }) |l| if (l.mode != .affine or l.s.ctx == null) return null;
    const k: c_int = @intCast(mlx.mlx_array_size(inds));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, inds, &[_]c_int{k}, 1, s));
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_astype(&ids, flat, .uint32, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_reshape(&sc, scores, &[_]c_int{k}, 1, s));
    const y = (try glm5.moeDecode(s, x, ids, sc, qb(w.gate), qb(w.up), qb(w.down), ql(se.gate), ql(se.up), ql(se.down), f.limit)) orelse return null;
    defer _ = mlx.mlx_array_free(y);
    const xs = mlx.getShape(x);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, xs.ptr, xs.len, s));
    return out;
}

fn qb(b: moe.Bank) glm5.QBank {
    return .{ .w = b.w, .s = b.s, .b = b.b, .bits = b.qp.bits, .gs = b.qp.group_size };
}

fn ql(l: Linear) glm5.QBank {
    return .{ .w = l.w, .s = l.s, .b = l.b, .bits = l.bits, .gs = l.group_size };
}
