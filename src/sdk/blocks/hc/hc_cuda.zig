//! `hc_decode2`'s two-launch hyper-connection read on MLX's CUDA backend: the stream norm (and a
//! pending hcWrite) folds into the down launch, the stream partials meet in the up/mix launch.
//! Rounding sites are the composed chain's (stream', xn, silu, up, gate, mix); only the f32
//! accumulation order differs. Dense bf16 and 2/4/8-bit affine weights.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const hc_decode2 = @import("hc_decode2.zig");

/// Down rows a block owns, one per warp.
const ROWS_TG: c_int = 8;
/// Up columns a block owns, one per warp.
const COLS_TG: c_int = 16;

const HEADER = kernelSource("kernels/hc/hc_cuda_header.cu");

const ND_SOURCE = kernelSource("kernels/hc/hc_cuda_nd.cu");

const UP_SOURCE = kernelSource("kernels/hc/hc_cuda_up.cu");

var nd_kernel: ?mlx.mlx_fast_cuda_kernel = null;
var up_kernel: ?mlx.mlx_fast_cuda_kernel = null;
var engaged = false;
/// Reads served by the two launches (the tests assert the path engaged).
pub var served: u64 = 0;

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_cuda_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_cuda_kernel_new(name, in_vec, out_vec, source, HEADER, true, 0);
    if (k.ctx == null) return error.CudaKernelCreateFailed;
    return k;
}

fn launch(kernel: mlx.mlx_fast_cuda_kernel, outs: []const struct { c_int, mlx.mlx_dtype }, grid: [3]c_int, tg: c_int, tmpl: []const struct { [*:0]const u8, c_int }, dt: mlx.mlx_dtype, inputs: []const mlx.mlx_array, s: mlx.mlx_stream, res: []mlx.mlx_array) !void {
    const c = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(c);
    for (outs) |o| try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(c, &[_]c_int{o[0]}, 1, o[1]));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(c, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(c, tg, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(c, "T", dt));
    for (tmpl) |a| try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, a[0], a[1]));
    const v = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&o, kernel, v, c, s));
    if (mlx.mlx_vector_array_size(o) != res.len) return error.CudaKernelBadOutputCount;
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

/// `hc_decode2.read`'s contract on CUDA, or null outside its envelope (the caller keeps the
/// unfused chain).
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
    pend: ?hc_decode2.Pending,
) !?hc_decode2.Out {
    if (!mlx.cudaAvailable() or !mlx.streamOnGpuDevice(s)) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    const dense = bits == 16;
    if (dense and dt != .bfloat16) return null;
    if (bits != 2 and bits != 4 and bits != 8 and !dense) return null;
    if (nw.ctx == null or mlx.mlx_array_dtype(nw) != dt) return null;
    if (!dense) inline for (.{ ds, db, us, ub }) |a| {
        if (a.ctx == null or mlx.mlx_array_dtype(a) != dt) return null;
    };
    if (iw.ctx != null and mlx.mlx_array_dtype(iw) != dt) return null;
    const vpw: c_int = @intCast(32 / bits);
    const b: c_int = @intCast(bits);
    const gs: c_int = @intCast(group_size);
    const K = hc * hidden;
    if (hc < 1 or hc > 8 or @rem(hidden, 256) != 0 or @rem(gs, 16) != 0 or @rem(hidden, gs) != 0) return null;
    const dsh = mlx.getShape(dw);
    const ush = mlx.getShape(uw);
    if (dsh.len != 2 or ush.len != 2 or dsh[1] * 32 != K * b or ush[0] != K) return null;
    const R = dsh[0];
    if (ush[1] * 32 != R * b or @rem(R, vpw) != 0 or @rem(R, ROWS_TG) != 0 or @rem(R, gs) != 0 or @rem(hidden, COLS_TG) != 0) return null;
    const inj: c_int = @intFromBool(iw.ctx != null);
    const wr: c_int = @intFromBool(pend != null);
    if (rows < 1 or rows > 16 or mlx.mlx_array_size(x) != @as(usize, @intCast(rows * K)) or mlx.mlx_array_size(nw) != @as(usize, @intCast(K))) return null;
    if (pend) |p| {
        if (mlx.mlx_array_dtype(p.out) != dt or mlx.mlx_array_dtype(p.inj) != dt) return null;
    }

    if (nd_kernel == null) nd_kernel = try makeKernel("msv_hc_read_nd", &.{ "x_in", "nw", "iw", "eps", "wo_in", "wi_in", "dw_q", "dw_s", "dw_b" }, &.{ "xn_out", "parts_out", "xs_out" }, ND_SOURCE);
    if (up_kernel == null) up_kernel = try makeKernel("msv_hc_read_up", &.{ "xn_in", "parts_in", "uw_q", "uw_s", "uw_b" }, &.{ "mixed_out", "inj_out" }, UP_SOURCE);

    const pr = R + hc;
    const groups = @divExact(R, ROWS_TG) + inj;
    const tmpl = [_]struct { [*:0]const u8, c_int }{ .{ "HC", hc }, .{ "H", hidden }, .{ "R", R }, .{ "GS", gs }, .{ "BITS", b }, .{ "INJ", inj }, .{ "WR", wr }, .{ "COLS", COLS_TG } };
    const wo = if (pend) |p| p.out else nw;
    const wi = if (pend) |p| p.inj else nw;
    // Dense weights carry no scales; the kernel never reads the operands bound in their place.
    const dsb: [4]mlx.mlx_array = if (dense) .{ nw, nw, nw, nw } else .{ ds, db, us, ub };
    var nd_out: [3]mlx.mlx_array = undefined;
    try launch(nd_kernel.?, &.{ .{ rows * K, dt }, .{ rows * hc * pr, .float32 }, .{ if (wr == 1) rows * K else 1, dt } }, .{ 256, groups * hc, 1 }, 256, &tmpl, dt, &.{ x, nw, if (inj == 1) iw else nw, eps, wo, wi, dw, dsb[0], dsb[1] }, s, &nd_out);
    defer _ = mlx.mlx_array_free(nd_out[0]);
    defer _ = mlx.mlx_array_free(nd_out[1]);
    errdefer _ = mlx.mlx_array_free(nd_out[2]);
    var upmix: [2]mlx.mlx_array = undefined;
    try launch(up_kernel.?, &.{ .{ rows * hidden, dt }, .{ rows * hc, dt } }, .{ 32 * COLS_TG, @divExact(hidden, COLS_TG), 1 }, 32 * COLS_TG, &tmpl, dt, &.{ nd_out[0], nd_out[1], uw, dsb[2], dsb[3] }, s, &upmix);
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
        log.info("[qwen4] CUDA two-launch hyper-connection read engaged: hc={d} hidden={d} lowrank={d} {d}-bit\n", .{ hc, hidden, R, bits });
    }
    served += 1;
    return .{ .mixed = upmix[0], .inj = inj_out, .stream = stream };
}

const testing = std.testing;

pub fn randArr(rnd: std.Random, shape: []const c_int, scale: f32, offset: f32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(buf);
    for (buf) |*v| v.* = (rnd.float(f32) - 0.5) * scale + offset;
    const f = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, dt, s));
    return out;
}

/// The arrays of one test case; `free` releases every one.
pub const Bag = struct {
    items: [256]mlx.mlx_array = undefined,
    n: usize = 0,
    pub fn keep(b: *Bag, a: mlx.mlx_array) mlx.mlx_array {
        b.items[b.n] = a;
        b.n += 1;
        return a;
    }
    pub fn free(b: *Bag) void {
        for (b.items[0..b.n]) |a| _ = mlx.mlx_array_free(a);
    }
};

/// The composed hyper-connection read in dtype `dt`: stream' = x + out*inj, n = rms_norm(stream')*nw,
/// mixed = mean_s(sigmoid(up(silu(down(n)))) * n), inj = 2*sigmoid(n @ iw). `down` [R,K], `up` [K,R].
pub fn chain(b: *Bag, s: mlx.mlx_stream, dt: mlx.mlx_dtype, x: mlx.mlx_array, nw: mlx.mlx_array, down: mlx.mlx_array, up: mlx.mlx_array, iw: mlx.mlx_array, wo: ?mlx.mlx_array, wi: ?mlx.mlx_array, hc: c_int, h: c_int, eps: f32) ![3]mlx.mlx_array {
    const O = struct {
        fn cast(bb: *Bag, st: mlx.mlx_stream, a: mlx.mlx_array, d: mlx.mlx_dtype) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&o, a, d, st));
            return bb.keep(o);
        }
        fn bin(bb: *Bag, st: mlx.mlx_stream, comptime f: anytype, a: mlx.mlx_array, c: mlx.mlx_array) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            try mlx.check(f(&o, a, c, st));
            return bb.keep(o);
        }
        fn shape(bb: *Bag, st: mlx.mlx_stream, a: mlx.mlx_array, sh: []const c_int) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&o, a, sh.ptr, sh.len, st));
            return bb.keep(o);
        }
        fn sig(bb: *Bag, st: mlx.mlx_stream, a: mlx.mlx_array) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_sigmoid(&o, a, st));
            return bb.keep(o);
        }
        fn t(bb: *Bag, st: mlx.mlx_stream, a: mlx.mlx_array) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_transpose(&o, a, st));
            return bb.keep(o);
        }
    };
    const rows = @divExact(@as(c_int, @intCast(mlx.mlx_array_size(x))), hc * h);
    var stream = try O.cast(b, s, x, dt);
    if (wo) |o| {
        const o4 = try O.shape(b, s, try O.cast(b, s, o, dt), &.{ 1, rows, 1, h });
        const prod = try O.bin(b, s, mlx.mlx_multiply, o4, try O.cast(b, s, wi.?, dt));
        stream = try O.bin(b, s, mlx.mlx_add, try O.shape(b, s, stream, &.{ 1, rows, hc, h }), prod);
    }
    const x4 = try O.shape(b, s, stream, &.{ 1, rows, hc, h });
    var prng = std.Random.DefaultPrng.init(0);
    const ones = try O.cast(b, s, b.keep(try randArr(prng.random(), &.{h}, 0.0, 1.0, .float32, s)), dt);
    var nr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&nr, x4, ones, eps, s));
    const n4 = try O.bin(b, s, mlx.mlx_multiply, b.keep(nr), try O.cast(b, s, nw, dt));
    const nf = try O.shape(b, s, n4, &.{ 1, rows, hc * h });
    const d = try O.bin(b, s, mlx.mlx_matmul, nf, try O.t(b, s, try O.cast(b, s, down, dt)));
    const act = try O.bin(b, s, mlx.mlx_multiply, d, try O.sig(b, s, d));
    const u = try O.bin(b, s, mlx.mlx_matmul, act, try O.t(b, s, try O.cast(b, s, up, dt)));
    const prod = try O.bin(b, s, mlx.mlx_multiply, try O.sig(b, s, try O.shape(b, s, u, &.{ 1, rows, hc, h })), n4);
    var mixed = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_mean_axis(&mixed, prod, 2, false, s));
    const isg = try O.sig(b, s, try O.bin(b, s, mlx.mlx_matmul, nf, try O.cast(b, s, iw, dt)));
    var two = mlx.mlx_array_new_float(2.0);
    two = try O.cast(b, s, b.keep(two), dt);
    const inj = try O.bin(b, s, mlx.mlx_multiply, isg, two);
    return .{ b.keep(mixed), inj, try O.shape(b, s, stream, &.{ 1, rows, hc * h }) };
}

pub fn rmsDiff(b: *Bag, s: mlx.mlx_stream, a: mlx.mlx_array, ref: mlx.mlx_array) !f32 {
    var af = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&af, a, .float32, s));
    var rf = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&rf, ref, .float32, s));
    var rr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&rr, rf, &[_]c_int{-1}, 1, s));
    var ar = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&ar, af, &[_]c_int{-1}, 1, s));
    var d = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&d, ar, rr, s));
    var sq = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_square(&sq, d, s));
    var m = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_mean(&m, sq, false, s));
    inline for (.{ af, rf, rr, ar, d, sq }) |x| _ = b.keep(x);
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_eval(b.keep(m)));
    try mlx.check(mlx.mlx_array_item_float32(&v, m));
    return @sqrt(v);
}

/// Every element of `got` within two bf16 ulps of an O(1) operand of `ref` (the Metal read's bar): the
/// same rounding sites, only the f32 summation order differs, and the mean over streams cancels.
pub fn expectClose(b: *Bag, s: mlx.mlx_stream, got: mlx.mlx_array, ref: mlx.mlx_array) !void {
    const n = mlx.mlx_array_size(ref);
    try testing.expectEqual(n, mlx.mlx_array_size(got));
    var vals: [2][]const f32 = undefined;
    for ([_]mlx.mlx_array{ got, ref }, 0..) |a, i| {
        var f = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&f, a, .float32, s));
        try mlx.check(mlx.mlx_array_eval(b.keep(f)));
        vals[i] = (mlx.mlx_array_data_float32(f) orelse return error.Unreadable)[0..n];
    }
    for (vals[0], vals[1]) |g, r| {
        try testing.expect(std.math.isFinite(g));
        try testing.expect(@abs(g - r) <= 0.02 * @abs(r) + 0.008);
    }
}
