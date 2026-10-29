//! The Hadamard format (Prism packs): affine weights stored in a rotated basis, each `<linear>.signs` read as
//! `H_block(signs * x)` before the matmul (`rht.zig`). Loads through the MLX-native path; the matmul is the engine's
//! rotated arm (`Transformer.qmatmul` with a registry bound).

const abi = @import("../../../abi/quant.zig");
const mlx = @import("../../../mlx/ffi.zig");
const native = @import("../native.zig");
const rht = @import("rht.zig");
const qmv2 = @import("../../blocks/qmm/qmv2.zig");
const qmv_nax2 = @import("../../blocks/qmm/qmv_nax2.zig");
const qmatmulBits = @import("../../blocks/linear/qmatmul.zig").qmatmulBits;

const Str = abi.Str;
const Verdict = abi.Verdict;
const arr = native.arr;
const wrap = native.wrap;

/// `Weight.flags`: the biases are exactly -scales (Prism's codec), the 2-bit kernels' fast path.
pub const NEG_SCALE: u8 = 1;

fn loadTensor(ctx: *const abi.LoadCtx, name: Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    load(ctx, name.slice(), in_dim, out) catch |e| return native.failed(e);
    return .ok;
}

fn load(ctx: *const abi.LoadCtx, base: []const u8, in_dim: u32, out: *abi.Weight) !void {
    try native.loadInto(ctx, base, in_dim, out);
    errdefer native.release(out);
    const signs = native.tensor(ctx, base, "signs");
    if (signs.ctx == null) return;
    if (out.scales.ctx == null) return error.HadamardSignsOnDenseWeight;
    out.block = native.state(ctx).?.block;
    var h = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&h, signs));
    out.side = wrap(h);
    if (out.biases.ctx != null and try biasIsNegScale(arr(out.scales), arr(out.biases), .{ .ctx = ctx.stream.ctx })) out.flags |= NEG_SCALE;
}

/// `max |scales + biases| == 0`, as the engine's load probes each rotated weight.
fn biasIsNegScale(sc: mlx.mlx_array, bi: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, sc, bi, s));
    var mag = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mag);
    try mlx.check(mlx.mlx_abs(&mag, sum, s));
    var peak = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(peak);
    try mlx.check(mlx.mlx_max(&peak, mag, false, s));
    var p32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p32);
    try mlx.check(mlx.mlx_astype(&p32, peak, .float32, s));
    try mlx.check(mlx.mlx_array_eval(p32));
    var v: f32 = 1;
    try mlx.check(mlx.mlx_array_item_float32(&v, p32));
    return v == 0;
}

fn matmul(w: *const abi.Weight, x: abi.Array, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    out.* = wrap(rotated(w, arr(x), .{ .ctx = s.ctx }) catch |e| return native.failed(e));
    return .ok;
}

/// The engine's rotated arm: the 2-bit kernels first (each declines what it does not serve), then stock qmm.
fn rotated(w: *const abi.Weight, x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const signs = arr(w.side);
    const xr = if (signs.ctx != null) try rht.transform(x, signs, @intCast(w.block), false, s) else x;
    defer if (signs.ctx != null) {
        _ = mlx.mlx_array_free(xr);
    };
    const W = arr(w.w);
    const sc = arr(w.scales);
    const bi = arr(w.biases);
    if (try qmv_nax2.qmm(xr, W, sc, bi, w.bits, w.group_size, s)) |y| return y;
    if (try qmv2.qmm(xr, W, sc, bi, w.bits, w.group_size, signs.ctx != null and w.flags & NEG_SCALE != 0, true, s)) |y| return y;
    return qmatmulBits(xr, W, sc, bi, w.bits, w.group_size, native.modeOf(w), s);
}

fn noBank(_: *const abi.LoadCtx, _: Str, _: u32, _: *abi.Weight) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "hadamard: rotated expert banks");
}
fn noMoe(_: *const abi.Weight, _: abi.Array, _: abi.Array, _: abi.Array, _: bool, _: abi.Stream, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "hadamard: rotated expert banks");
}

pub const table: abi.QuantTable = blk: {
    var t = native.table("hadamard", &native.billStored);
    t.load_tensor = &loadTensor;
    t.load_bank = &noBank;
    t.matmul = &matmul;
    t.moe = &noMoe;
    break :blk t;
};
