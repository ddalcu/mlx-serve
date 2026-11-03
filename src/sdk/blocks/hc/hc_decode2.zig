//! Hyper-connection read at ONE decode row in two dispatches instead of three. The stream norm
//! folds into the down launch (the same idea as oMLX's two-launch decode, Apache-2.0): the down
//! projection's K slice for stream s is exactly stream s, so a threadgroup normalizes only its
//! own stream and multiplies the slice, with its weight loads issued before the norm finishes.
//! The slice partial sums meet in the up/mix launch, which sums them into the activation once per
//! threadgroup. Rounding sites are the three-kernel path's (xn, silu, up, gate, mix); only the
//! f32 accumulation order differs.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");

/// A deferred hcWrite the norm launch applies itself: stream + out * inj.
pub const Pending = struct { out: mlx.mlx_array, inj: mlx.mlx_array };

/// Flat results: mixed [hidden], inj [hc] (null-ctx without inject weights), stream [hc*hidden]
/// (null-ctx without a pending write).
pub const Out = struct { mixed: mlx.mlx_array, inj: mlx.mlx_array, stream: mlx.mlx_array };

/// Down rows a simdgroup owns (a threadgroup has 8 simdgroups): one row each spreads the launch over
/// the most threadgroups, where four rows left cores idle.
const RPS: c_int = 1;
const ROWS_TG: c_int = 8 * RPS;
/// Up columns a threadgroup owns, one per simdgroup.
const COLS_TG: c_int = 16;

pub const ND_SOURCE = kernelSource("kernels/hc/hc_decode2_nd.metal");

pub const U2_SOURCE = kernelSource("kernels/hc/hc_decode2_u2.metal");

/// The up/mix launch for 1..16 rows at hc 4, 8-bit. Lanes split
/// [part:2][stream:2][column:1], so one lane holds one stream's slice of a weight row in registers (16-byte loads issued
/// before any math) and every row's activations are applied to it; the activations (the stream partials summed, then
/// silu) are staged once per threadgroup. Rounding sites are U2_SOURCE's; only the f32 summation order differs.
pub const U2V_SOURCE = kernelSource("kernels/hc/hc_decode2_u2v.metal");

var engaged = false;
/// Reads served by the two launches (the tests assert the path engaged).
pub var served: u64 = 0;
/// Of those, reads whose up launch took the vectorized kernel.
pub var served_vec: u64 = 0;

pub var rows_override: ?c_int = null;
/// Widest read the two launches serve (the verify widths and batched slots).
pub fn maxRows() c_int {
    return rows_override orelse 16;
}

/// Values per pack the HC kernels read: one word at 2/4/8 bits, 8 values in BITS bytes at 3/5/6.
pub fn packValues(bits: u32) ?c_int {
    return switch (bits) {
        2, 4, 8, 16 => @intCast(32 / bits),
        3, 5, 6 => 8,
        else => null,
    };
}

/// Eight B-bit values packed in B bytes (3/5/6-bit), little-endian as mx.quantize lays them out.
/// `hc_val`: a packed code as a float, or (16-bit dense) the raw weight's bits.
pub const PACK8_HEADER = kernelSource("kernels/hc/hc_decode2_pack8_header.metal");

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, PACK8_HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

/// A launch config bakes grid, outputs and template values, which a layer's geometry fixes.
fn config(outs: []const struct { c_int, mlx.mlx_dtype }, grid: [3]c_int, tg: c_int, tmpl: []const struct { [*:0]const u8, c_int }, dt: mlx.mlx_dtype) !mlx.mlx_fast_metal_kernel_config {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{o[0]}, 1, o[1]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
    for (tmpl) |a| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, a[0], a[1]));
    return c;
}

fn run(kernel: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, inputs: []const mlx.mlx_array, s: mlx.mlx_stream, res: []mlx.mlx_array) !void {
    const v = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel, v, cfg, s));
    if (mlx.mlx_vector_array_size(o) != res.len) return error.MetalKernelBadOutputCount;
    var got: usize = 0;
    errdefer for (res[0..got]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (res, 0..) |*r, i| {
        r.* = mlx.mlx_array_new();
        got = i + 1;
        try mlx.check(mlx.mlx_vector_array_get(r, o, i));
    }
}

const Key = struct { hc: c_int, h: c_int, r: c_int, inj: c_int, wr: c_int, bits: u32, gs: c_int, dtype: mlx.mlx_dtype, rows: c_int };
const Armed = struct { key: Key, nd: mlx.mlx_fast_metal_kernel_config, up: mlx.mlx_fast_metal_kernel_config, upv: ?mlx.mlx_fast_metal_kernel_config };
var nd_kernel: ?mlx.mlx_fast_metal_kernel = null;
var up_kernel: ?mlx.mlx_fast_metal_kernel = null;
var up_v_kernel: ?mlx.mlx_fast_metal_kernel = null;
/// A mixed pack's per-layer widths each keep their configs, so no layer re-arms; round-robin eviction.
var armed: [16]?Armed = @splat(null);
var armed_next: usize = 0;

/// Rows the vectorized up launch serves (hc 4, 8-bit weights, 1..16 rows): its threadgroup holds every row's activations.
/// One row takes it too, so a row sums in the same order at every row count and a grouped verify stays byte-identical to a
/// solo one.
pub fn upVectorized(rows: c_int, hc: c_int, r: c_int, bits: u32, gs: c_int) bool {
    if (rows < 1 or rows > 16 or hc != 4 or bits != 8 or @rem(r, 64) != 0 or r > 512 or @rem(gs, 16) != 0) return false;
    return true;
}

/// Simdgroups per threadgroup of the vectorized up launch, two columns each. One row has the least work per column, so the
/// widest threadgroup hides the weight loads best there; more rows already fill the time. The arithmetic order does not
/// depend on this, so a row sums the same at every row count whichever width serves it.
fn upvSgs(rows: c_int) c_int {
    return if (rows == 1) 16 else 8;
}

/// The up launch's 16-byte weight loads need a 16-byte-aligned buffer. A lazy weight has no data pointer yet, so it is
/// evaluated here: the launch must not depend on whether an earlier read already did, or one row would sum in
/// a different order from the same row read before the weights were resident.
fn aligned16(a: mlx.mlx_array) bool {
    var avail = false;
    if (mlx._mlx_array_is_available(&avail, a) != 0) return false;
    if (!avail) mlx.check(mlx.mlx_array_eval(a)) catch return false;
    const p = mlx.mlx_array_data_uint32(a) orelse return false;
    return @intFromPtr(p) % 16 == 0;
}

/// The two-launch read of `rows` rows, or null outside its envelope (the caller keeps the
/// unfused chain). `dw` [R, K/vpw] and `uw` [K, R/vpw] are the packed projections with
/// scales and biases in x's dtype, `iw` the dense [K, hc] inject weight or null-ctx.
pub fn read(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    nw: mlx.mlx_array,
    dw: mlx.mlx_array,
    ds: mlx.mlx_array,
    db: mlx.mlx_array,
    uw: mlx.mlx_array,
    us: mlx.mlx_array,
    ub: mlx.mlx_array,
    iw: mlx.mlx_array,
    eps: mlx.mlx_array,
    rows: c_int,
    hc: c_int,
    hidden: c_int,
    bits: u32,
    group_size: u32,
    pend: ?Pending,
) !?Out {
    if (!mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    const dense = bits == 16;
    if (nw.ctx == null or mlx.mlx_array_dtype(nw) != dt) return null;
    if (!dense) inline for (.{ ds, db, us, ub }) |a| {
        if (a.ctx == null or mlx.mlx_array_dtype(a) != dt) return null;
    };
    if (dense and dt != .bfloat16) return null;
    if (iw.ctx != null and mlx.mlx_array_dtype(iw) != dt) return null;
    const vpw = packValues(bits) orelse return null;
    const b: c_int = @intCast(bits);
    const gs: c_int = @intCast(group_size);
    const K = hc * hidden;
    if (hc < 1 or hc > 8 or @rem(hidden, 256) != 0 or @rem(gs, 16) != 0 or @rem(hidden, gs) != 0) return null;
    const dsh = mlx.getShape(dw);
    const ush = mlx.getShape(uw);
    if (dsh.len != 2 or ush.len != 2 or dsh[1] * 32 != K * b or ush[0] != K) return null;
    const R = dsh[0];
    if (ush[1] * 32 != R * b or @rem(R, vpw) != 0 or @rem(R, ROWS_TG) != 0 or @rem(R, gs) != 0 or @rem(hidden, COLS_TG) != 0) return null;
    // A lane's 16 values are whole words, and a down row's slice splits into whole slots.
    if (@rem(16, vpw) != 0 or @rem(hidden, 16) != 0) return null;
    const inj: c_int = @intFromBool(iw.ctx != null);
    const wr: c_int = @intFromBool(pend != null);
    if (rows < 1 or mlx.mlx_array_size(x) != @as(usize, @intCast(rows * K)) or mlx.mlx_array_size(nw) != @as(usize, @intCast(K))) return null;

    const key = Key{ .hc = hc, .h = hidden, .r = R, .inj = inj, .wr = wr, .bits = bits, .gs = gs, .dtype = dt, .rows = rows };
    if (nd_kernel == null) {
        const ins = [_][*:0]const u8{ "x_in", "nw", "iw", "eps", "wo_in", "wi_in", "dw_q", "dw_s", "dw_b" };
        const outs = [_][*:0]const u8{ "xn_out", "parts_out", "xs_out" };
        nd_kernel = try makeKernel("mlxserve_hc_read_nd", &ins, &outs, ND_SOURCE);
    }
    if (up_kernel == null) {
        const ins = [_][*:0]const u8{ "xn_in", "parts_in", "uw_q", "uw_s", "uw_b" };
        const outs = [_][*:0]const u8{ "mixed_out", "inj_out" };
        up_kernel = try makeKernel("mlxserve_hc_read_u2", &ins, &outs, U2_SOURCE);
    }
    if (up_v_kernel == null) {
        const ins = [_][*:0]const u8{ "xn_in", "parts_in", "uw_q", "uw_s", "uw_b" };
        const outs = [_][*:0]const u8{ "mixed_out", "inj_out" };
        up_v_kernel = try makeKernel("mlxserve_hc_read_u2v", &ins, &outs, U2V_SOURCE);
    }
    const pr: c_int = R + hc;
    const cfgs: Armed = for (armed) |a| {
        if (a != null and std.meta.eql(a.?.key, key)) break a.?;
    } else blk: {
        const groups = @divExact(R, ROWS_TG) + inj;
        const nd = try config(&.{ .{ rows * K, dt }, .{ rows * hc * pr, .float32 }, .{ if (wr == 1) rows * K else 1, dt } }, .{ 256, groups * hc, rows }, 256, &.{ .{ "HC", hc }, .{ "H", hidden }, .{ "R", R }, .{ "GS", gs }, .{ "BITS", @intCast(bits) }, .{ "INJ", inj }, .{ "WR", wr }, .{ "RPS", RPS }, .{ "ROWS_TG", ROWS_TG } }, dt);
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(nd);
        const up = try config(&.{ .{ rows * hidden, dt }, .{ rows * hc, dt } }, .{ 32 * COLS_TG, @divExact(hidden, COLS_TG), rows }, 32 * COLS_TG, &.{ .{ "HC", hc }, .{ "H", hidden }, .{ "R", R }, .{ "GS", gs }, .{ "BITS", @intCast(bits) }, .{ "INJ", inj }, .{ "COLS", COLS_TG } }, dt);
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(up);
        const upv: ?mlx.mlx_fast_metal_kernel_config = if (upVectorized(rows, hc, R, bits, gs))
            try config(&.{ .{ rows * hidden, dt }, .{ rows * hc, dt } }, .{ 32 * upvSgs(rows) * @divExact(hidden, 2 * upvSgs(rows)), 1, 1 }, 32 * upvSgs(rows), &.{ .{ "HC", hc }, .{ "H", hidden }, .{ "R", R }, .{ "GS", gs }, .{ "BITS", @intCast(bits) }, .{ "INJ", inj }, .{ "ROWS", rows }, .{ "SGS", upvSgs(rows) } }, dt)
        else
            null;
        if (armed[armed_next]) |old| {
            _ = mlx.mlx_fast_metal_kernel_config_free(old.nd);
            _ = mlx.mlx_fast_metal_kernel_config_free(old.up);
            if (old.upv) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        }
        const fresh: Armed = .{ .key = key, .nd = nd, .up = up, .upv = upv };
        armed[armed_next] = fresh;
        armed_next = (armed_next + 1) % armed.len;
        break :blk fresh;
    };
    var nd_out: [3]mlx.mlx_array = undefined;
    const wo = if (pend) |p| p.out else nw;
    const wi = if (pend) |p| p.inj else nw;
    // Dense weights carry no scales; the kernel never reads the operands bound in their place.
    const dsb: [4]mlx.mlx_array = if (dense) .{ nw, nw, nw, nw } else .{ ds, db, us, ub };
    try run(nd_kernel.?, cfgs.nd, &.{ x, nw, if (inj == 1) iw else nw, eps, wo, wi, dw, dsb[0], dsb[1] }, s, &nd_out);
    defer _ = mlx.mlx_array_free(nd_out[0]);
    defer _ = mlx.mlx_array_free(nd_out[1]);
    errdefer _ = mlx.mlx_array_free(nd_out[2]);
    var upmix: [2]mlx.mlx_array = undefined;
    if (cfgs.upv != null and aligned16(uw)) {
        try run(up_v_kernel.?, cfgs.upv.?, &.{ nd_out[0], nd_out[1], uw, dsb[2], dsb[3] }, s, &upmix);
        served_vec += 1;
    } else {
        try run(up_kernel.?, cfgs.up, &.{ nd_out[0], nd_out[1], uw, dsb[2], dsb[3] }, s, &upmix);
    }
    var stream = nd_out[2];
    if (wr == 0) {
        _ = mlx.mlx_array_free(stream);
        stream = .{ .ctx = null };
    }
    var inj_out = upmix[1];
    if (inj == 0) {
        _ = mlx.mlx_array_free(inj_out);
        inj_out = .{ .ctx = null };
    }
    if (!engaged) {
        engaged = true;
        log.info("[qwen4] two-launch hyper-connection read engaged: hc={d} hidden={d} lowrank={d} {d}-bit g{d}\n", .{ hc, hidden, R, bits, group_size });
    }
    served += 1;
    return .{ .mixed = upmix[0], .inj = inj_out, .stream = stream };
}
