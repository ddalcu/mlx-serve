//! MoE decode over fp expert banks (nvfp4, mxfp4, mxfp8) on MLX's CUDA backend, in two launches:
//! gate+up+SwiGLU (with a per-expert gate scale), then down with the routing-weighted sum. The dot
//! products follow `row_dot`'s lane split and reduction tree (patches/mlx-cuda-row-dot-matvec.patch,
//! which serves these gathers on the composed chain), and the activation and weighted sum use MLX's
//! own functors in the chain's order, so a decode row reproduces the chain's rounding sites.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");

pub const Format = enum(c_int) { mxfp8 = 1, mxfp4 = 2, nvfp4 = 3 };

/// The banks of one projection: packed codes [E, N, K*bits/32] and uint8 scales [E, N, K/group].
pub const Bank = struct { w: mlx.mlx_array, s: mlx.mlx_array };

const HEADER = kernelSource("kernels/moe/moe_cuda_header.cu");

// Grid (I/8 blocks, B*TOPK): one warp per intermediate row, gate and up rows of one expert.
const GU_SOURCE = kernelSource("kernels/moe/moe_cuda_gu.cu");

// Grid (D/DPB blocks, B): 8-lane groups own one (output, expert) down dot; the weighted sum over
// experts is the chain's: Multiply by the routing weight, then a sequential Sum in T.
const DR_SOURCE = kernelSource("kernels/moe/moe_cuda_dr.cu");

var gu_kernel: ?mlx.mlx_fast_cuda_kernel = null;
var dr_kernel: ?mlx.mlx_fast_cuda_kernel = null;
var engaged = false;
/// Decodes served by the two launches (the tests assert the path engaged).
pub var served: u64 = 0;

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, source: [*:0]const u8, out: [*:0]const u8) !mlx.mlx_fast_cuda_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const outs = [_][*:0]const u8{out};
    const out_vec = mlx.mlx_vector_string_new_data(&outs, 1);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_cuda_kernel_new(name, in_vec, out_vec, source, HEADER, true, 0);
    if (k.ctx == null) return error.CudaKernelCreateFailed;
    return k;
}

fn launch(kernel: mlx.mlx_fast_cuda_kernel, out_size: c_int, dt: mlx.mlx_dtype, grid: [3]c_int, tmpl: []const struct { [*:0]const u8, c_int }, inputs: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const c = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(c, &[_]c_int{out_size}, 1, dt));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(c, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(c, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(c, "T", dt));
    for (tmpl) |a| try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, a[0], a[1]));
    const v = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&o, kernel, v, c, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    return y;
}

/// sum_k scores[b,k] * down_e(silu(gate_e(x_b) * gscale_e) * up_e(x_b)), e = inds[b,k]: [B, D] in x's
/// dtype, or null outside the kernels' envelope. x [B, D]; inds uint32 and scores [B, TOPK].
pub fn decode(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Bank, up: Bank, down: Bank, gate_scale: ?mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array, fmt: Format) !?mlx.mlx_array {
    if (!mlx.cudaAvailable() or !mlx.streamOnGpuDevice(s)) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    if (mlx.mlx_array_dtype(scores) != dt or mlx.mlx_array_dtype(inds) != .uint32) return null;
    if (gate_scale) |g| if (mlx.mlx_array_dtype(g) != dt) return null;
    const bits: c_int = if (fmt == .mxfp8) 8 else 4;
    const group: c_int = if (fmt == .nvfp4) 16 else 32;
    const xs = mlx.getShape(x);
    const is = mlx.getShape(inds);
    if (xs.len != 2 or is.len != 2 or is[0] != xs[0]) return null;
    const rows = xs[0];
    const D = xs[1];
    const topk = is[1];
    if (rows < 1 or rows > 16 or topk < 1 or topk > 32) return null;
    const gsh = mlx.getShape(gate.w);
    const ush = mlx.getShape(up.w);
    const dsh = mlx.getShape(down.w);
    if (gsh.len != 3 or ush.len != 3 or dsh.len != 3) return null;
    const I = gsh[1];
    if (gsh[2] * 32 != D * bits or !std.mem.eql(c_int, gsh, ush) or dsh[0] != gsh[0] or dsh[1] != D or dsh[2] * 32 != I * bits) return null;
    // 16-byte chunks per row, and nvfp4's scale pairs, need these widths.
    if (@rem(D * bits, 128) != 0 or @rem(I * bits, 128) != 0 or @rem(I, 8) != 0) return null;
    inline for (.{ gate, up, down }) |bank| {
        if (mlx.mlx_array_dtype(bank.w) != .uint32 or mlx.mlx_array_dtype(bank.s) != .uint8) return null;
    }
    if (mlx.getShape(gate.s)[2] * group != D or mlx.getShape(down.s)[2] * group != I) return null;

    if (gu_kernel == null) gu_kernel = try makeKernel("msv_moe_gate_up", &.{ "x", "gw", "gs", "uw", "us", "inds", "gscale" }, GU_SOURCE, "act");
    if (dr_kernel == null) dr_kernel = try makeKernel("msv_moe_down_sum", &.{ "act", "dw", "ds", "inds", "scores" }, DR_SOURCE, "y");
    const tmpl = [_]struct { [*:0]const u8, c_int }{ .{ "FMT", @intFromEnum(fmt) }, .{ "D", D }, .{ "I", I }, .{ "TOPK", topk }, .{ "GSCALE", @intFromBool(gate_scale != null) } };
    const act = try launch(gu_kernel.?, rows * topk * I, dt, .{ 256 * @divExact(I, 8), rows * topk, 1 }, &tmpl, &.{ x, gate.w, gate.s, up.w, up.s, inds, gate_scale orelse scores }, s);
    defer _ = mlx.mlx_array_free(act);

    const dpb = @divTrunc(32, topk);
    const y = try launch(dr_kernel.?, rows * D, dt, .{ 256 * @divTrunc(D + dpb - 1, dpb), rows, 1 }, &tmpl, &.{ act, down.w, down.s, inds, scores }, s);
    if (!engaged) {
        engaged = true;
        log.info("[moe] CUDA fused decode engaged: {s} top-{d} D={d} I={d}\n", .{ @tagName(fmt), topk, D, I });
    }
    served += 1;
    return y;
}

const testing = std.testing;

/// The arrays of one test; `free` releases every one.
const Bag = struct {
    items: [128]mlx.mlx_array = undefined,
    n: usize = 0,
    fn keep(b: *Bag, a: mlx.mlx_array) mlx.mlx_array {
        b.items[b.n] = a;
        b.n += 1;
        return a;
    }
    fn free(b: *Bag) void {
        for (b.items[0..b.n]) |a| _ = mlx.mlx_array_free(a);
    }
};

fn rand(b: *Bag, s: mlx.mlx_stream, key: u64, shape: []const c_int, lo: f32, hi: f32, dt: mlx.mlx_dtype) !mlx.mlx_array {
    var k = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_key(&k, key));
    _ = b.keep(k);
    var lo_a = b.keep(mlx.mlx_array_new_float(lo));
    var hi_a = b.keep(mlx.mlx_array_new_float(hi));
    _ = &lo_a;
    _ = &hi_a;
    var u = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_uniform(&u, lo_a, hi_a, shape.ptr, shape.len, .float32, k, s));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, b.keep(u), dt, s));
    return b.keep(o);
}

/// The production decode chain: per-projection unsorted gather_qmm, the gate scale picked by
/// expert, silu(g) * u, down, the routing weight, then the sum over experts.
fn chain(b: *Bag, s: mlx.mlx_stream, x: mlx.mlx_array, g: [2]mlx.mlx_array, u: [2]mlx.mlx_array, d: [2]mlx.mlx_array, gscale: mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array, mode: [*:0]const u8, group: c_int, bits: c_int) !mlx.mlx_array {
    const O = struct {
        fn r(bb: *Bag, comptime f: anytype, args: anytype) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            try mlx.check(@call(.auto, f, .{&o} ++ args));
            return bb.keep(o);
        }
    };
    const none: mlx.mlx_array = .{ .ctx = null };
    const xs = mlx.getShape(x);
    const rows = xs[0];
    const topk = mlx.getShape(inds)[1];
    const gi = mlx.mlx_optional_int.some(group);
    const bi = mlx.mlx_optional_int.some(bits);
    const x_5 = try O.r(b, mlx.mlx_reshape, .{ x, &[_]c_int{ rows, 1, 1, 1, xs[1] }, 5, s });
    const ind3 = try O.r(b, mlx.mlx_reshape, .{ inds, &[_]c_int{ rows, 1, topk }, 3, s });
    const g_5 = try O.r(b, mlx.mlx_gather_qmm, .{ x_5, g[0], g[1], none, none, ind3, true, gi, bi, mode, false, s });
    const gq = try O.r(b, mlx.mlx_reshape, .{ g_5, &[_]c_int{ rows, 1, topk, -1 }, 4, s });
    const pick = try O.r(b, mlx.mlx_take_axis, .{ gscale, ind3, 0, s });
    const gsc = try O.r(b, mlx.mlx_multiply, .{ gq, try O.r(b, mlx.mlx_expand_dims, .{ pick, -1, s }), s });
    const sl = try O.r(b, mlx.mlx_multiply, .{ gsc, try O.r(b, mlx.mlx_sigmoid, .{ gsc, s }), s });
    const u_5 = try O.r(b, mlx.mlx_gather_qmm, .{ x_5, u[0], u[1], none, none, ind3, true, gi, bi, mode, false, s });
    const act = try O.r(b, mlx.mlx_multiply, .{ sl, try O.r(b, mlx.mlx_reshape, .{ u_5, &[_]c_int{ rows, 1, topk, -1 }, 4, s }), s });

    const d_5 = try O.r(b, mlx.mlx_gather_qmm, .{ try O.r(b, mlx.mlx_expand_dims, .{ act, -2, s }), d[0], d[1], none, none, ind3, true, gi, bi, mode, false, s });
    const dq = try O.r(b, mlx.mlx_reshape, .{ d_5, &[_]c_int{ rows, 1, topk, -1 }, 4, s });
    const sc = try O.r(b, mlx.mlx_reshape, .{ scores, &[_]c_int{ rows, 1, topk, 1 }, 4, s });
    const w = try O.r(b, mlx.mlx_multiply, .{ dq, sc, s });
    return O.r(b, mlx.mlx_reshape, .{ try O.r(b, mlx.mlx_sum_axis, .{ w, 2, false, s }), &[_]c_int{ rows, xs[1] }, 2, s });
}

test "moe_cuda: decode rows are bit-identical to the composed gather chain (nvfp4, mxfp8; 1 and 3 rows)" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const E: c_int = 16;
    const D: c_int = 2560;
    const I: c_int = 640;
    const K: c_int = 10;
    const cases = [_]struct { fmt: Format, mode: [*:0]const u8, group: c_int, bits: c_int }{ .{ .fmt = .nvfp4, .mode = "nvfp4", .group = 16, .bits = 4 }, .{ .fmt = .mxfp8, .mode = "mxfp8", .group = 32, .bits = 8 } };
    for (cases) |c| {
        for ([_]c_int{ 1, 3 }) |rows| {
            var b: Bag = .{};
            defer b.free();
            var banks: [3][2]mlx.mlx_array = undefined;
            for (&banks, [3][3]c_int{ .{ E, I, D }, .{ E, I, D }, .{ E, D, I } }, 0..) |*bank, sh, i| {
                const w = try rand(&b, s, 10 + i, &sh, -0.05, 0.05, .bfloat16);
                var tri = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(tri);
                try mlx.check(mlx.mlx_quantize(&tri, w, mlx.mlx_optional_int.some(c.group), mlx.mlx_optional_int.some(c.bits), c.mode, .{}, s));
                for (bank, 0..) |*a, j| {
                    a.* = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_vector_array_get(a, tri, j));
                    _ = b.keep(a.*);
                }
            }
            const x = try rand(&b, s, 1, &.{ rows, D }, -2, 2, .bfloat16);
            const gscale = try rand(&b, s, 2, &.{E}, 0.5, 1.5, .bfloat16);
            const scores = try rand(&b, s, 3, &.{ rows, K }, 0.0, 0.3, .bfloat16);
            var idata_buf: [3 * 32]u32 = undefined;
            const idata = idata_buf[0..@intCast(rows * K)];
            for (idata, 0..) |*v, i| v.* = @intCast((i * 7 + i / 5) % @as(usize, E));
            const inds = b.keep(mlx.mlx_array_new_data(idata.ptr, &[_]c_int{ rows, K }, 2, .uint32));
            const before = served;
            const got = b.keep((try decode(s, x, .{ .w = banks[0][0], .s = banks[0][1] }, .{ .w = banks[1][0], .s = banks[1][1] }, .{ .w = banks[2][0], .s = banks[2][1] }, gscale, inds, scores, c.fmt)) orelse return error.Declined);
            try testing.expectEqual(before + 1, served);
            const ref = try chain(&b, s, x, .{ banks[0][0], banks[0][1] }, .{ banks[1][0], banks[1][1] }, .{ banks[2][0], banks[2][1] }, gscale, inds, scores, c.mode, c.group, c.bits);
            var got2 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&got2, got, &[_]c_int{ rows, D }, 2, s));
            _ = b.keep(got2);
            var eq = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_equal(&eq, got2, ref, false, s));
            try mlx.check(mlx.mlx_array_eval(b.keep(eq)));
            var same = false;
            try mlx.check(mlx.mlx_array_item_bool(&same, eq));
            if (!same) {
                var eqe = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_equal(&eqe, got2, ref, s));
                var diff = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_logical_not(&diff, b.keep(eqe), s));
                var cnt = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_sum(&cnt, b.keep(diff), false, s));
                var n: f32 = 0;
                var cf = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_astype(&cf, b.keep(cnt), .float32, s));
                try mlx.check(mlx.mlx_array_eval(b.keep(cf)));
                try mlx.check(mlx.mlx_array_item_float32(&n, cf));
                std.debug.print("moe_cuda {s} rows={d}: {d} of {d} outputs differ from the chain\n", .{ c.mode, rows, n, rows * D });
            }
            try testing.expect(same);
        }
    }
}
