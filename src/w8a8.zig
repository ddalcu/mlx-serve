//! Qwen-Image w8a8 on M5 NAX, with one scale per weight/activation row.
//! Weights are packed [K/32,N,32] at load; activations remain [M,K].
//! Opt-in with --w8a8; dense bf16/f16 DiT weights only.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;
const A = mlx.mlx_array;

/// Symmetric int8 maps a row's absmax to 127.
const Q_MAX: f32 = 127.0;
/// A zero row would divide by zero; this keeps the scale finite and the row zero.
const Q_MIN_SCALE: f32 = 1e-8;
/// The balanced NAX tile consumes each weight K block as one contiguous row.
const WEIGHT_K_BLOCK: usize = 32;

pub const default_on: bool = false;

pub var override: ?bool = null;

pub fn optionValue(arg: []const u8) ?bool {
    if (std.mem.eql(u8, arg, "--w8a8")) return true;
    if (std.mem.eql(u8, arg, "--no-w8a8")) return false;
    return null;
}

pub fn enabled() bool {
    return if (override) |v| v else default_on;
}

var logged_engagement = false;

/// The NAX int8 tile consumes K in 32-wide steps, so K must be a multiple of 32.
pub fn eligible(n: u32, k: u32, dtype: mlx.mlx_dtype) bool {
    return dtype == .bfloat16 and k >= 32 and k % 32 == 0 and n >= 1;
}

fn available(s: S) bool {
    return mlx.streamIsGpu(s) and @import("transformer.zig").verifyQmmNaxAvailable();
}

var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const names = [_][*:0]const u8{ "xq", "wq", "sx", "sw" };
    const outs = [_][*:0]const u8{"y"};
    const ins = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(ins);
    const outv = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(outv);
    const k = mlx.mlx_fast_metal_kernel_new(
        "msv_w8a8",
        ins,
        outv,
        @embedFile("kernels/w8a8.metal"),
        @embedFile("kernels/w8a8_header.metal"),
        true,
        false,
    );
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// `y[rows, n]` (bf16) from row-quantized operands. `wq` is packed
/// [k/32, n, 32]; xq remains row-major [rows, k]. Caller owns the result.
fn gemm(xq: A, wq: A, sx: A, sw: A, rows: c_int, n: c_int, s: S) !A {
    if (!available(s)) return error.W8a8Ineligible;
    const kern = try getKernel();
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const y_shape = [_]c_int{ rows, n };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, .bfloat16));
    // The grid is in THREADS (mlx dispatches with `dispatch_threads`), so one
    // 128x128 tile per 32x8 threadgroup: tg_x * 32 x tg_y * 8.
    const tg_x = @divTrunc(n + 127, 128);
    const tg_y = @divTrunc(rows + 127, 128);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, tg_x * 32, tg_y * 8, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 8, 1));
    const arrays = [_]A{ xq, wq, sx, sw };
    const ins = mlx.mlx_vector_array_new_data(&arrays, arrays.len);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, kern, ins, cfg, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    return y;
}

fn scalarArray(v: f32, s: S) !A {
    var buf: [1]f32 = .{v};
    const sh = [_]c_int{1};
    const raw = mlx.mlx_array_new_data(&buf, &sh, 1, .float32);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, raw, .float32, s));
    return out;
}

const Quantized = struct { q: A, scale: A };
var quant_kernel: ?mlx.mlx_fast_metal_kernel = null;
var quantize_calls: u64 = 0;

fn quantizeRows(x: A, rows: c_int, k: c_int, s: S) !Quantized {
    if (@import("builtin").is_test) quantize_calls += 1;
    if (mlx.streamIsGpu(s) and mlx.mlx_array_dtype(x) == .bfloat16)
        return quantizeRowsFused(x, rows, k, s);
    return quantizeRowsOps(x, rows, k, s);
}

fn quantizeRowsFused(x: A, rows: c_int, k: c_int, s: S) !Quantized {
    if (!mlx.streamIsGpu(s) or mlx.mlx_array_dtype(x) != .bfloat16)
        return error.W8a8Ineligible;
    if (quant_kernel == null) {
        const ins = mlx.mlx_vector_string_new_data(&[_][*:0]const u8{"x"}, 1);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&[_][*:0]const u8{ "q", "scale" }, 2);
        defer _ = mlx.mlx_vector_string_free(outs);
        const kernel_handle = mlx.mlx_fast_metal_kernel_new(
            "msv_w8a8_quant",
            ins,
            outs,
            @embedFile("kernels/w8a8_quant.metal"),
            "",
            true,
            false,
        );
        if (kernel_handle.ctx == null) return error.MetalKernelCompileFailed;
        quant_kernel = kernel_handle;
    }
    var matrix = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(matrix);
    try mlx.check(mlx.mlx_reshape(&matrix, x, &[_]c_int{ rows, k }, 2, s));
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, k }, 2, .int8));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, 1 }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, rows * 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    const ins = mlx.mlx_vector_array_new_data(&[_]A{matrix}, 1);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, quant_kernel.?, ins, cfg, s));
    var q = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(q);
    var scale = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(scale);
    try mlx.check(mlx.mlx_vector_array_get(&q, outs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&scale, outs, 1));
    return .{ .q = q, .scale = scale };
}

/// Per-row symmetric int8 quantization of `x[..., k]`: the packed int8 [rows, k]
/// and one f32 scale per row. A all-zero row quantizes to zeros.
fn quantizeRowsOps(x: A, rows: c_int, k: c_int, s: S) !Quantized {
    // The activation stays in its own dtype (bf16 in production): a f32
    // round-trip doubles every pass over a [rows x 4096] tensor, and the scale
    // is the only value that needs the extra exponent range.
    const dt = mlx.mlx_array_dtype(x);
    var m2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m2);
    try mlx.check(mlx.mlx_reshape(&m2, x, &[_]c_int{ rows, k }, 2, s));
    var ax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ax);
    try mlx.check(mlx.mlx_abs(&ax, m2, s));
    var amax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(amax);
    try mlx.check(mlx.mlx_max_axis(&amax, ax, 1, true, s)); // [rows, 1]
    var amaxf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(amaxf);
    if (dt == .float32) {
        try mlx.check(mlx.mlx_array_set(&amaxf, amax));
    } else {
        try mlx.check(mlx.mlx_astype(&amaxf, amax, .float32, s));
    }
    const lo = try scalarArray(Q_MIN_SCALE, s);
    defer _ = mlx.mlx_array_free(lo);
    var floored = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(floored);
    try mlx.check(mlx.mlx_maximum(&floored, amaxf, lo, s));
    const qmax = try scalarArray(Q_MAX, s);
    defer _ = mlx.mlx_array_free(qmax);
    var scale = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(scale);
    try mlx.check(mlx.mlx_divide(&scale, floored, qmax, s));
    var scale_div = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scale_div);
    if (dt == .float32) {
        try mlx.check(mlx.mlx_array_set(&scale_div, scale));
    } else {
        try mlx.check(mlx.mlx_astype(&scale_div, scale, dt, s));
        var narrowed = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(narrowed);
        try mlx.check(mlx.mlx_astype(&narrowed, scale_div, .float32, s));
        _ = mlx.mlx_array_free(scale);
        scale = narrowed;
    }
    var normed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normed);
    try mlx.check(mlx.mlx_divide(&normed, m2, scale_div, s));
    var rounded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rounded);
    try mlx.check(mlx.mlx_round(&rounded, normed, 0, s));
    const lo_q = try scalarArray(-Q_MAX, s);
    defer _ = mlx.mlx_array_free(lo_q);
    var clipped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(clipped);
    try mlx.check(mlx.mlx_clip(&clipped, rounded, lo_q, qmax, s));
    var q = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_astype(&q, clipped, .int8, s));
    return .{ .q = q, .scale = scale };
}

/// Materialize row-major [n, k] int8 weights as K-block-major [k/32, n, 32].
/// The contiguous result owns its storage, so callers can release the
/// row-major quantized graph immediately after this returns.
fn packWeightArray(q: A, n: c_int, k: c_int, s: S) !A {
    if (n <= 0 or k <= 0 or @mod(k, @as(c_int, @intCast(WEIGHT_K_BLOCK))) != 0)
        return error.W8a8ShapeMismatch;
    const blocks: c_int = @divExact(k, @as(c_int, @intCast(WEIGHT_K_BLOCK)));
    var grouped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grouped);
    try mlx.check(mlx.mlx_reshape(
        &grouped,
        q,
        &[_]c_int{ n, blocks, @intCast(WEIGHT_K_BLOCK) },
        3,
        s,
    ));
    var transposed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(transposed);
    try mlx.check(mlx.mlx_transpose_axes(
        &transposed,
        grouped,
        &[_]c_int{ 1, 0, 2 },
        3,
        s,
    ));
    var tiled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(tiled);
    // The kernel uses raw row-major byte offsets into this rank-3 payload.
    try mlx.check(mlx.mlx_contiguous(&tiled, transposed, false, s));
    try mlx.check(mlx.mlx_array_eval(tiled));
    return tiled;
}

/// Decode the physical packed weight layout for an independent reference.
/// This is deliberately a transpose/reshape path, not the production GEMM.
fn unpackWeightArray(tiled: A, n: c_int, k: c_int, s: S) !A {
    if (n <= 0 or k <= 0 or @mod(k, @as(c_int, @intCast(WEIGHT_K_BLOCK))) != 0)
        return error.W8a8ShapeMismatch;
    const blocks: c_int = @divExact(k, @as(c_int, @intCast(WEIGHT_K_BLOCK)));
    const shape = mlx.getShape(tiled);
    if (shape.len != 3 or shape[0] != blocks or shape[1] != n or
        shape[2] != @as(c_int, @intCast(WEIGHT_K_BLOCK)))
        return error.W8a8ShapeMismatch;
    var transposed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(transposed);
    try mlx.check(mlx.mlx_transpose_axes(
        &transposed,
        tiled,
        &[_]c_int{ 1, 0, 2 },
        3,
        s,
    ));
    var transposed_contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(transposed_contig);
    try mlx.check(mlx.mlx_contiguous(&transposed_contig, transposed, false, s));
    var row_major = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(row_major);
    try mlx.check(mlx.mlx_reshape(&row_major, transposed_contig, &[_]c_int{ n, k }, 2, s));
    return row_major;
}

/// Owned row-quantized activation that can be borrowed by several linears with
/// the same input. The original shape is retained because GEMM flattens rows.
pub const Prepared = struct {
    q: A,
    scale: A,
    shape: [4]c_int,
    rank: u8,

    pub fn deinit(self: *Prepared) void {
        _ = mlx.mlx_array_free(self.q);
        _ = mlx.mlx_array_free(self.scale);
    }

    pub fn dims(self: *const Prepared) []const c_int {
        return self.shape[0..self.rank];
    }
};

/// Quantize an activation once for reuse by `Lin.forwardQuantized`.
pub fn prepare(x: A, s: S) !Prepared {
    const shape = mlx.getShape(x);
    if (shape.len == 0) return error.W8a8BadInputRank;
    const k = shape[shape.len - 1];
    if (k <= 0) return error.W8a8ShapeMismatch;
    const ku: usize = @intCast(k);
    const size = mlx.mlx_array_size(x);
    if (size % ku != 0) return error.W8a8ShapeMismatch;
    const rows: c_int = @intCast(size / ku);
    if (shape.len > 4) return error.W8a8BadInputRank;
    const q = try quantizeRows(x, rows, k, s);
    errdefer _ = mlx.mlx_array_free(q.q);
    errdefer _ = mlx.mlx_array_free(q.scale);
    var owned_shape: [4]c_int = undefined;
    @memcpy(owned_shape[0..shape.len], shape);
    return .{ .q = q.q, .scale = q.scale, .shape = owned_shape, .rank = @intCast(shape.len) };
}

/// Reject a loaded shape whose packed GEMM disagrees with dequantized MLX matmul.
fn probeLin(l: *const Lin, s: S) !void {
    const rows: c_int = 9;
    const k: usize = @intCast(l.k);
    const x = std.heap.page_allocator.alloc(f32, @as(usize, @intCast(rows)) * k) catch return error.SkipProbe;
    defer std.heap.page_allocator.free(x);
    var st: u32 = 0x9E3779B9;
    for (x, 0..) |*v, i| {
        st = st *% 1664525 +% 1013904223;
        const u: f32 = @as(f32, @floatFromInt(st >> 8)) / 16777216.0;
        v.* = (u - 0.5) * (if (i % 7 == 0) @as(f32, 8.0) else @as(f32, 1.0));
    }
    const host = mlx.mlx_array_new_data(x.ptr, &[_]c_int{ rows, @intCast(k) }, 2, .float32);
    defer _ = mlx.mlx_array_free(host);
    var xb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xb);
    try mlx.check(mlx.mlx_astype(&xb, host, .bfloat16, s));

    const q = try quantizeRows(xb, rows, l.k, s);
    defer _ = mlx.mlx_array_free(q.q);
    defer _ = mlx.mlx_array_free(q.scale);
    const y = try gemm(q.q, l.wq, q.scale, l.sw, rows, l.n, s);
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));

    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, q.q, .float32, s));
    var xr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xr);
    try mlx.check(mlx.mlx_multiply(&xr, xf, q.scale, s)); // [rows, k]
    const row_major_w = try unpackWeightArray(l.wq, l.n, l.k, s);
    defer _ = mlx.mlx_array_free(row_major_w);
    var wf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wf);
    try mlx.check(mlx.mlx_astype(&wf, row_major_w, .float32, s));
    var wr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wr);
    try mlx.check(mlx.mlx_multiply(&wr, wf, l.sw, s)); // [n, k]
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, wr, s)); // [k, n]
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_matmul(&ref, xr, wt, s));
    try mlx.check(mlx.mlx_array_eval(ref));
    var yf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(yf);
    try mlx.check(mlx.mlx_astype(&yf, y, .float32, s));
    var diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_subtract(&diff, yf, ref, s));
    var adiff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(adiff);
    try mlx.check(mlx.mlx_abs(&adiff, diff, s));
    var worst = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(worst);
    try mlx.check(mlx.mlx_max(&worst, adiff, false, s));
    var aref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aref);
    try mlx.check(mlx.mlx_abs(&aref, ref, s));
    var rmax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rmax);
    try mlx.check(mlx.mlx_max(&rmax, aref, false, s));
    try mlx.check(mlx.mlx_array_eval(worst));
    try mlx.check(mlx.mlx_array_eval(rmax));
    var d: f32 = 0;
    var r: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&d, worst));
    try mlx.check(mlx.mlx_array_item_float32(&r, rmax));
    if (!(d <= 0.02 * r + 1e-3)) {
        log.err("[w8a8] parity probe FAILED: n={d} k={d} worst={e} ref={e} (arm off)\n", .{ l.n, l.k, d, r });
        return error.W8a8ParityFailed;
    }
}

/// Test seam: `Lin.load` returns this fault before touching weights.
pub const LoadFault = enum { none, parity, skip_probe, compile };
pub var test_load_fault: LoadFault = .none;

pub const Lin = struct {
    wq: A, // int8 [k/32, n, 32], K-block-major and row-major within each block
    sw: A, // f32 [n, 1]
    n: c_int,
    k: c_int,

    /// `prefix.weight` of a dense checkpoint. An affine source is refused
    /// (`W8a8AffineSource`): requantizing it would stack a second rounding on the pack's.
    pub fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, dtype: mlx.mlx_dtype, s: S) !Lin {
        if (@import("builtin").is_test) switch (test_load_fault) {
            .none => {},
            .parity => return error.W8a8ParityFailed,
            .skip_probe => return error.SkipProbe,
            .compile => return error.MetalKernelCompileFailed,
        };
        if (!available(s)) return error.W8a8Ineligible;
        const raw = try loadSource(w, a, prefix, s);
        defer _ = mlx.mlx_array_free(raw);
        const shape = mlx.getShape(raw);
        if (shape.len != 2) return error.W8a8BadWeightRank;
        const n = shape[0];
        const k = shape[1];
        if (!eligible(@intCast(n), @intCast(k), dtype)) return error.W8a8Ineligible;
        var q = try quantizeRows(raw, n, k, s);
        const tiled = packWeightArray(q.q, n, k, s) catch |err| {
            _ = mlx.mlx_array_free(q.q);
            _ = mlx.mlx_array_free(q.scale);
            return err;
        };
        _ = mlx.mlx_array_free(q.q);
        q.q = tiled;
        errdefer _ = mlx.mlx_array_free(q.q);
        errdefer _ = mlx.mlx_array_free(q.scale);
        try mlx.check(mlx.mlx_array_eval(q.scale));
        return .{ .wq = q.q, .sw = q.scale, .n = n, .k = k };
    }

    /// Test seam: quantize an in-memory f32 [n, k] weight.
    pub fn fromHost(vals: []const f32, n: c_int, k: c_int, s: S) !Lin {
        const host = mlx.mlx_array_new_data(vals.ptr, &[_]c_int{ n, k }, 2, .float32);
        defer _ = mlx.mlx_array_free(host);
        var q = try quantizeRows(host, n, k, s);
        const tiled = packWeightArray(q.q, n, k, s) catch |err| {
            _ = mlx.mlx_array_free(q.q);
            _ = mlx.mlx_array_free(q.scale);
            return err;
        };
        _ = mlx.mlx_array_free(q.q);
        q.q = tiled;
        errdefer _ = mlx.mlx_array_free(q.q);
        errdefer _ = mlx.mlx_array_free(q.scale);
        try mlx.check(mlx.mlx_array_eval(q.scale));
        return .{ .wq = q.q, .sw = q.scale, .n = n, .k = k };
    }

    pub fn deinit(self: *Lin) void {
        _ = mlx.mlx_array_free(self.wq);
        _ = mlx.mlx_array_free(self.sw);
    }

    fn forwardRows(self: *const Lin, q: A, scale: A, rows: c_int, shape: []const c_int, bias: ?A, s: S) !A {
        var y = try gemm(q, self.wq, scale, self.sw, rows, self.n, s);
        defer _ = mlx.mlx_array_free(y);
        if (bias) |b| {
            var biased = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(biased);
            try mlx.check(mlx.mlx_add(&biased, y, b, s));
            _ = mlx.mlx_array_free(y);
            y = biased;
        }
        if (shape.len == 0 or shape.len > 4) return error.W8a8BadInputRank;
        var want: [4]c_int = undefined;
        @memcpy(want[0..shape.len], shape);
        want[shape.len - 1] = self.n;
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_reshape(&out, y, &want, shape.len, s));
        if (!logged_engagement) {
            logged_engagement = true;
            log.info("[w8a8] engaged: NAX int8xint8, per-row scales (n={d}, k={d})\n", .{ self.n, self.k });
        }
        return out;
    }

    /// x[..., k] -> y[..., n], bf16. `bias` is added after the dequantizing scale.
    pub fn forward(self: *const Lin, x: A, bias: ?A, s: S) !A {
        const shape = mlx.getShape(x);
        if (shape.len == 0) return error.W8a8BadInputRank;
        if (shape[shape.len - 1] != self.k) return error.W8a8ShapeMismatch;
        const rows: c_int = @intCast(mlx.mlx_array_size(x) / @as(usize, @intCast(self.k)));
        const q = try quantizeRows(x, rows, self.k, s);
        defer _ = mlx.mlx_array_free(q.q);
        defer _ = mlx.mlx_array_free(q.scale);
        return self.forwardRows(q.q, q.scale, rows, shape, bias, s);
    }

    /// Apply this linear to an activation prepared by `prepare`.
    pub fn forwardQuantized(self: *const Lin, prepared: *const Prepared, bias: ?A, s: S) !A {
        const shape = prepared.dims();
        if (shape.len == 0 or shape[shape.len - 1] != self.k)
            return error.W8a8ShapeMismatch;
        const ku: usize = @intCast(self.k);
        const q_size = mlx.mlx_array_size(prepared.q);
        if (ku == 0 or q_size % ku != 0) return error.W8a8ShapeMismatch;
        const rows: c_int = @intCast(q_size / ku);
        return self.forwardRows(prepared.q, prepared.scale, rows, shape, bias, s);
    }
};

/// Read `prefix` as f32 [n,k], dequantizing affine weights with their own geometry.
fn loadSource(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !A {
    const wk = try std.fmt.allocPrint(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const weight = w.get(wk) orelse {
        log.err("[w8a8] missing weight: {s}\n", .{wk});
        return error.MissingWeight;
    };
    const sk = try std.fmt.allocPrint(a, "{s}.scales", .{prefix});
    defer a.free(sk);
    if (w.get(sk) != null) return error.W8a8AffineSource;
    const dt = mlx.mlx_array_dtype(weight);
    if (dt != .bfloat16 and dt != .float16 and dt != .float32) return error.W8a8Ineligible;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, weight, .float32, s));
    return out;
}

const testing = std.testing;

test "w8a8: CLI flags are the only switch" {
    try testing.expectEqual(@as(?bool, true), optionValue("--w8a8"));
    try testing.expectEqual(@as(?bool, false), optionValue("--no-w8a8"));
    try testing.expectEqual(@as(?bool, null), optionValue("--w8a8-extra"));
    const saved = override;
    defer override = saved;
    override = null;
    try testing.expect(!default_on);
    try testing.expectEqual(default_on, enabled());
    override = optionValue("--w8a8");
    try testing.expect(enabled());
    override = optionValue("--no-w8a8");
    try testing.expect(!enabled());
}

test "w8a8: unsupported NAX declines before loading weights" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const transformer = @import("transformer.zig");
    const saved = transformer.vqmm_nax_probe_override;
    defer transformer.vqmm_nax_probe_override = saved;
    transformer.vqmm_nax_probe_override = false;
    var weights = Weights.init(testing.allocator);
    defer weights.deinit();
    try testing.expectError(error.W8a8Ineligible, Lin.load(&weights, testing.allocator, "missing", .bfloat16, mlx.gpuStream()));
}

test "w8a8: fused bf16 row quantization preserves the op-chain bytes" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    for ([_]usize{ 32, 96, 4096, 12288 }) |k| {
        const rows = 19;
        const values = try testing.allocator.alloc(u16, rows * k);
        defer testing.allocator.free(values);
        var rng = std.Random.DefaultPrng.init(0x8178);
        for (values, 0..) |*v, i| {
            const value: f32 = if (i < k) 0 else if (i < 2 * k) 1e-12 else rng.random().float(f32) * 32 - 16;
            v.* = @intCast(@as(u32, @bitCast(value)) >> 16);
        }
        const input = mlx.mlx_array_new_data(values.ptr, &[_]c_int{ rows, @intCast(k) }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(input);
        const ref = try quantizeRowsOps(input, rows, @intCast(k), s);
        defer _ = mlx.mlx_array_free(ref.q);
        defer _ = mlx.mlx_array_free(ref.scale);
        const fused = try quantizeRowsFused(input, rows, @intCast(k), s);
        defer _ = mlx.mlx_array_free(fused.q);
        defer _ = mlx.mlx_array_free(fused.scale);
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, ref.q, fused.q, false, s));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        try testing.expect(same);
        try mlx.check(mlx.mlx_array_eval(ref.scale));
        try mlx.check(mlx.mlx_array_eval(fused.scale));
        const rs = mlx.mlx_array_data_float32(ref.scale).?;
        const fs = mlx.mlx_array_data_float32(fused.scale).?;
        for (0..rows) |row| {
            try testing.expect(std.math.isFinite(fs[row]));
            try testing.expectApproxEqRel(rs[row], fs[row], 1e-6);
        }
    }
}

/// Symmetric per-row int8 quantization on the host, the reference for `quantizeRows`.
fn hostQuantizeRow(row: []const f32, q: []i8, scale: *f32) void {
    var amax: f32 = 0;
    for (row) |v| amax = @max(amax, @abs(v));
    const sc = @max(amax, Q_MIN_SCALE) / Q_MAX;
    scale.* = sc;
    for (row, 0..) |v, i| {
        const r = std.math.round(v / sc);
        q[i] = @intFromFloat(@max(-Q_MAX, @min(Q_MAX, r)));
    }
}

fn unpackWeightHost(comptime T: type, tiled: []const T, row_major: []T, n: usize, k: usize) !void {
    const total = std.math.mul(usize, n, k) catch return error.W8a8ShapeMismatch;
    if (n == 0 or k == 0 or k % WEIGHT_K_BLOCK != 0 or tiled.len < total or row_major.len < total)
        return error.W8a8ShapeMismatch;
    for (0..k / WEIGHT_K_BLOCK) |block| {
        for (0..n) |row| {
            for (0..WEIGHT_K_BLOCK) |in_block| {
                row_major[row * k + block * WEIGHT_K_BLOCK + in_block] =
                    tiled[(block * n + row) * WEIGHT_K_BLOCK + in_block];
            }
        }
    }
}

test "w8a8: MLX weight prepack has the physical bytes and shape contract" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const n: usize = 5;
    const k: usize = 96;
    var source: [n * k]i8 = undefined;
    for (&source, 0..) |*v, i| {
        const raw: i32 = @intCast((i * 29 + 5) % 255);
        v.* = @intCast(raw - 127);
    }
    const row_shape = [_]c_int{ @intCast(n), @intCast(k) };
    const row_major = mlx.mlx_array_new_data(&source, &row_shape, 2, .int8);
    defer _ = mlx.mlx_array_free(row_major);
    const tiled = try packWeightArray(row_major, @intCast(n), @intCast(k), s);
    defer _ = mlx.mlx_array_free(tiled);
    try testing.expectEqual(mlx.mlx_dtype.int8, mlx.mlx_array_dtype(tiled));
    try testing.expectEqualSlices(
        c_int,
        &[_]c_int{ @intCast(k / WEIGHT_K_BLOCK), @intCast(n), @intCast(WEIGHT_K_BLOCK) },
        mlx.getShape(tiled),
    );

    var packed_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(packed_i);
    try mlx.check(mlx.mlx_astype(&packed_i, tiled, .int32, s));
    try mlx.check(mlx.mlx_array_eval(packed_i));
    const actual = mlx.mlx_array_data_int32(packed_i) orelse return error.NoData;
    for (0..k / WEIGHT_K_BLOCK) |block| {
        for (0..n) |row| {
            for (0..WEIGHT_K_BLOCK) |in_block| {
                const packed_i0 = (block * n + row) * WEIGHT_K_BLOCK + in_block;
                const source_i = row * k + block * WEIGHT_K_BLOCK + in_block;
                try testing.expectEqual(@as(i32, @intCast(source[source_i])), actual[packed_i0]);
            }
        }
    }

    const unpacked = try unpackWeightArray(tiled, @intCast(n), @intCast(k), s);
    defer _ = mlx.mlx_array_free(unpacked);
    var unpacked_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(unpacked_i);
    try mlx.check(mlx.mlx_astype(&unpacked_i, unpacked, .int32, s));
    try mlx.check(mlx.mlx_array_eval(unpacked_i));
    const roundtrip = mlx.mlx_array_data_int32(unpacked_i) orelse return error.NoData;
    for (source, 0..) |v, i| try testing.expectEqual(@as(i32, @intCast(v)), roundtrip[i]);
}

test "w8a8: per-row int8 GEMM matches the dequantized f32 reference" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!available(s)) return error.SkipZigTest;
    const n: usize = 128;
    const k: usize = 96;
    const rows: usize = 40;

    var prng = std.Random.DefaultPrng.init(0x8A8);
    const rnd = prng.random();
    var w: [n * k]f32 = undefined;
    var x: [rows * k]f32 = undefined;
    for (&w) |*v| v.* = rnd.float(f32) * 2 - 1;
    for (&x) |*v| v.* = rnd.float(f32) * 2 - 1;
    // A row with a single large spike, the shape that makes per-row scales pay.
    for (0..k) |i| x[7 * k + i] = 0;
    x[7 * k + 3] = 9.5;

    var l = try Lin.fromHost(&w, @intCast(n), @intCast(k), s);
    defer l.deinit();
    try testing.expectEqualSlices(
        c_int,
        &[_]c_int{ @intCast(k / WEIGHT_K_BLOCK), @intCast(n), @intCast(WEIGHT_K_BLOCK) },
        mlx.getShape(l.wq),
    );
    try testing.expectEqualSlices(c_int, &[_]c_int{ @intCast(n), 1 }, mlx.getShape(l.sw));

    const xv = mlx.mlx_array_new_data(&x, &[_]c_int{ @intCast(rows), @intCast(k) }, 2, .float32);
    defer _ = mlx.mlx_array_free(xv);
    const got = try l.forward(xv, null, s);
    defer _ = mlx.mlx_array_free(got);
    try mlx.check(mlx.mlx_array_eval(got));
    const gp = mlx.mlx_array_data_bfloat16(got) orelse return error.NoData;

    var wq: [n * k]i8 = undefined;
    var sw: [n]f32 = undefined;
    for (0..n) |r| hostQuantizeRow(w[r * k ..][0..k], wq[r * k ..][0..k], &sw[r]);
    var xq: [rows * k]i8 = undefined;
    var sx: [rows]f32 = undefined;
    for (0..rows) |r| hostQuantizeRow(x[r * k ..][0..k], xq[r * k ..][0..k], &sx[r]);

    var worst: f32 = 0;
    for (0..rows) |m| {
        for (0..n) |c| {
            var acc: f64 = 0;
            for (0..k) |kk| acc += @as(f64, @floatFromInt(xq[m * k + kk])) * @as(f64, @floatFromInt(wq[c * k + kk]));
            const want: f32 = @floatCast(acc * @as(f64, sx[m]) * @as(f64, sw[c]));
            const g: f32 = @bitCast(@as(u32, gp[m * n + c]) << 16);
            try testing.expect(std.math.isFinite(g));
            const denom = @max(@abs(want), 1e-3);
            worst = @max(worst, @abs(g - want) / denom);
        }
    }
    try testing.expect(worst < 2e-2);
}

test "w8a8: bf16 3-D activation and a bf16 weight" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!available(s)) return error.SkipZigTest;
    const n: usize = 259;
    const k: usize = 64;
    const rows: usize = 300;

    var prng = std.Random.DefaultPrng.init(0x3D);
    const rnd = prng.random();
    var w: [n * k]f32 = undefined;
    var x: [rows * k]f32 = undefined;
    for (&w) |*v| v.* = rnd.float(f32) * 2 - 1;
    for (&x) |*v| v.* = rnd.float(f32) * 2 - 1;

    // bf16 host weight, as the loader gets it from a dense checkpoint.
    const wsh = [_]c_int{ @intCast(n), @intCast(k) };
    const wf = mlx.mlx_array_new_data(&w, &wsh, 2, .float32);
    defer _ = mlx.mlx_array_free(wf);
    var wb = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&wb, wf, .bfloat16, s));
    defer _ = mlx.mlx_array_free(wb);
    var wmap = model_mod.Weights.init(testing.allocator);
    defer wmap.deinit();
    const key = try testing.allocator.dupe(u8, "t.weight");
    var handle = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&handle, wb));
    try wmap.map.put(key, handle);
    var l = try Lin.load(&wmap, testing.allocator, "t", .bfloat16, s);
    defer l.deinit();
    try testing.expectEqualSlices(
        c_int,
        &[_]c_int{ @intCast(k / WEIGHT_K_BLOCK), @intCast(n), @intCast(WEIGHT_K_BLOCK) },
        mlx.getShape(l.wq),
    );

    // [1, rows, k] bf16 activation, the shape a DiT linear sees.
    const xsh = [_]c_int{ 1, @intCast(rows), @intCast(k) };
    const xf = mlx.mlx_array_new_data(&x, &xsh, 3, .float32);
    defer _ = mlx.mlx_array_free(xf);
    var xb = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&xb, xf, .bfloat16, s));
    defer _ = mlx.mlx_array_free(xb);

    const got = try l.forward(xb, null, s);
    defer _ = mlx.mlx_array_free(got);
    try mlx.check(mlx.mlx_array_eval(got));
    try testing.expectEqual(@as(c_int, n), mlx.getShape(got)[2]);
    const gp = mlx.mlx_array_data_bfloat16(got) orelse return error.NoData;

    // GEMM truth uses the quantized operands, not the pre-bf16 random inputs.
    const quant = try quantizeRows(xb, rows, k, s);
    defer _ = mlx.mlx_array_free(quant.q);
    defer _ = mlx.mlx_array_free(quant.scale);
    try mlx.check(mlx.mlx_array_eval(quant.q));
    try mlx.check(mlx.mlx_array_eval(quant.scale));
    var wi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wi);
    var xi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xi);
    try mlx.check(mlx.mlx_astype(&wi, l.wq, .int32, s));
    try mlx.check(mlx.mlx_astype(&xi, quant.q, .int32, s));
    try mlx.check(mlx.mlx_array_eval(wi));
    try mlx.check(mlx.mlx_array_eval(xi));
    const packed_wq = mlx.mlx_array_data_int32(wi) orelse return error.NoData;
    var wq: [n * k]i32 = undefined;
    try unpackWeightHost(i32, packed_wq[0 .. n * k], &wq, n, k);
    const sw = mlx.mlx_array_data_float32(l.sw) orelse return error.NoData;
    const xq = mlx.mlx_array_data_int32(xi) orelse return error.NoData;
    const sx = mlx.mlx_array_data_float32(quant.scale) orelse return error.NoData;
    var worst: f32 = 0;
    for (0..rows) |m| {
        for (0..n) |c| {
            var acc: f64 = 0;
            for (0..k) |kk| acc += @as(f64, @floatFromInt(xq[m * k + kk])) * @as(f64, @floatFromInt(wq[c * k + kk]));
            const want: f32 = @floatCast(acc * @as(f64, sx[m]) * @as(f64, sw[c]));
            const g: f32 = @bitCast(@as(u32, gp[m * n + c]) << 16);
            try testing.expect(std.math.isFinite(g));
            worst = @max(worst, @abs(g - want) / @max(@abs(want), 1e-3));
        }
    }
    try testing.expect(worst < 2e-2);
}

test "w8a8: prepared activation reuses quantization and preserves rank and bias" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!available(s)) return error.SkipZigTest;
    const n: usize = 40;
    const k: usize = 64;
    const rows: usize = 6;

    var w: [n * k]f32 = undefined;
    for (&w, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i % 17) + 1)) / 17.0;
    var l = try Lin.fromHost(&w, @intCast(n), @intCast(k), s);
    defer l.deinit();

    var x: [rows * k]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 23)) / 11.0 - 1.0;
    const x_shape = [_]c_int{ 2, 3, @intCast(k) };
    const host = mlx.mlx_array_new_data(&x, &x_shape, x_shape.len, .float32);
    defer _ = mlx.mlx_array_free(host);
    var input = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(input);
    try mlx.check(mlx.mlx_astype(&input, host, .bfloat16, s));
    for (&w) |*v| v.* = -v.*;
    var second = try Lin.fromHost(&w, @intCast(n), @intCast(k), s);
    defer second.deinit();

    var bias_values: [n]f32 = undefined;
    for (&bias_values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 100.0;
    const bias_shape = [_]c_int{@intCast(n)};
    const bias = mlx.mlx_array_new_data(&bias_values, &bias_shape, bias_shape.len, .float32);
    defer _ = mlx.mlx_array_free(bias);

    const before = quantize_calls;
    var prepared = try prepare(input, s);
    defer prepared.deinit();
    try testing.expectEqual(before + 1, quantize_calls);

    const reused = try l.forwardQuantized(&prepared, bias, s);
    defer _ = mlx.mlx_array_free(reused);
    const reused_second = try second.forwardQuantized(&prepared, bias, s);
    defer _ = mlx.mlx_array_free(reused_second);
    try testing.expectEqual(before + 1, quantize_calls);
    try testing.expectEqualSlices(c_int, x_shape[0..], prepared.dims());
    try testing.expectEqualSlices(c_int, &[_]c_int{ 2, 3, @intCast(n) }, mlx.getShape(reused));

    const independent = try l.forward(input, bias, s);
    defer _ = mlx.mlx_array_free(independent);
    try testing.expectEqual(before + 2, quantize_calls);
    try mlx.check(mlx.mlx_array_eval(reused));
    try mlx.check(mlx.mlx_array_eval(independent));
    var equal = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(equal);
    try mlx.check(mlx.mlx_array_equal(&equal, reused, independent, false, s));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, equal));
    try testing.expect(same);

    const independent_second = try second.forward(input, bias, s);
    defer _ = mlx.mlx_array_free(independent_second);
    try mlx.check(mlx.mlx_array_equal(&equal, reused_second, independent_second, false, s));
    try mlx.check(mlx.mlx_array_item_bool(&same, equal));
    try testing.expect(same);
    try testing.expectEqual(before + 3, quantize_calls);

    const wrong_shape = [_]c_int{ 2, 3, 32 };
    var wrong_values: [rows * 32]f32 = undefined;
    for (&wrong_values) |*v| v.* = 1.0;
    const wrong_input = mlx.mlx_array_new_data(&wrong_values, &wrong_shape, wrong_shape.len, .float32);
    defer _ = mlx.mlx_array_free(wrong_input);
    var wrong = try prepare(wrong_input, s);
    defer wrong.deinit();
    try testing.expectError(error.W8a8ShapeMismatch, l.forwardQuantized(&wrong, null, s));
    try testing.expectEqual(before + 4, quantize_calls);
}

fn expectStoredScaleIsDivisor(input: A, quant: Quantized, rows: usize, k: c_int, s: S) !void {
    try mlx.check(mlx.mlx_array_eval(quant.scale));
    var narrow = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(narrow);
    try mlx.check(mlx.mlx_astype(&narrow, quant.scale, .bfloat16, s));
    var back = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(back);
    try mlx.check(mlx.mlx_astype(&back, narrow, .float32, s));
    try mlx.check(mlx.mlx_array_eval(back));
    const got = mlx.mlx_array_data_float32(quant.scale) orelse return error.NoData;
    const roundtrip = mlx.mlx_array_data_float32(back) orelse return error.NoData;
    for (0..rows) |row| {
        if (std.math.isNan(got[row])) {
            try testing.expect(std.math.isNan(roundtrip[row]));
            continue;
        }
        try testing.expectEqual(got[row], roundtrip[row]);
    }
    var matrix = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(matrix);
    try mlx.check(mlx.mlx_reshape(&matrix, input, &[_]c_int{ @intCast(rows), k }, 2, s));
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, matrix, .float32, s));
    var quot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(quot);
    try mlx.check(mlx.mlx_divide(&quot, xf, quant.scale, s));
    var qb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qb);
    try mlx.check(mlx.mlx_astype(&qb, quot, .bfloat16, s));
    var qf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qf);
    try mlx.check(mlx.mlx_astype(&qf, qb, .float32, s));
    var rounded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rounded);
    try mlx.check(mlx.mlx_round(&rounded, qf, 0, s));
    const lo_q = try scalarArray(-Q_MAX, s);
    defer _ = mlx.mlx_array_free(lo_q);
    const hi_q = try scalarArray(Q_MAX, s);
    defer _ = mlx.mlx_array_free(hi_q);
    var clipped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(clipped);
    try mlx.check(mlx.mlx_clip(&clipped, rounded, lo_q, hi_q, s));
    var expect_q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(expect_q);
    try mlx.check(mlx.mlx_astype(&expect_q, clipped, .int8, s));
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, quant.q, expect_q, false, s));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    try testing.expect(same);
}

test "w8a8: stored scale is the divisor the row was quantized with" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const rows: usize = 3;
    const k: usize = 64;
    const values = try testing.allocator.alloc(u16, rows * k);
    defer testing.allocator.free(values);
    @memset(values[0..k], 0x4040);
    for (values[k .. 2 * k], 0..) |*v, i| v.* = if (i == 0) 0x4040 else 0x3f80;
    @memset(values[2 * k ..], 0);
    const input = mlx.mlx_array_new_data(values.ptr, &[_]c_int{ @intCast(rows), @intCast(k) }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(input);
    const ops = try quantizeRowsOps(input, @intCast(rows), @intCast(k), s);
    defer _ = mlx.mlx_array_free(ops.q);
    defer _ = mlx.mlx_array_free(ops.scale);
    const fused = try quantizeRowsFused(input, @intCast(rows), @intCast(k), s);
    defer _ = mlx.mlx_array_free(fused.q);
    defer _ = mlx.mlx_array_free(fused.scale);
    try expectStoredScaleIsDivisor(input, ops, rows, @intCast(k), s);
    try expectStoredScaleIsDivisor(input, fused, rows, @intCast(k), s);
}

test "w8a8: affine 8-bit g64 weights stay packed and dense bf16 or f16 load" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const n: c_int = 4;
    const k: c_int = 128;
    var dense: [4 * 128]f32 = undefined;
    for (&dense, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 9)) * 0.01 - 0.03;
    const host = mlx.mlx_array_new_data(&dense, &[_]c_int{ n, k }, 2, .float32);
    defer _ = mlx.mlx_array_free(host);
    var vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_quantize(&vec, host, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, s));
    var qw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qw);
    var qs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qs);
    var qb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qb);
    try mlx.check(mlx.mlx_vector_array_get(&qw, vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&qs, vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&qb, vec, 2));
    const qp = @import("transformer.zig").affineParamsFromGeometry(qw, qs, @intCast(k)) orelse return error.NoGeometry;
    try testing.expectEqual(@as(u32, 8), qp.bits);
    try testing.expectEqual(@as(u32, 64), qp.group_size);
    var affine_w = model_mod.Weights.init(testing.allocator);
    defer affine_w.deinit();
    const wk = try testing.allocator.dupe(u8, "blk.weight");
    const sk = try testing.allocator.dupe(u8, "blk.scales");
    const bk = try testing.allocator.dupe(u8, "blk.biases");
    var hw = mlx.mlx_array_new();
    var hs = mlx.mlx_array_new();
    var hb = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&hw, qw));
    try mlx.check(mlx.mlx_array_set(&hs, qs));
    try mlx.check(mlx.mlx_array_set(&hb, qb));
    try affine_w.map.put(wk, hw);
    try affine_w.map.put(sk, hs);
    try affine_w.map.put(bk, hb);
    try testing.expectError(error.W8a8AffineSource, Lin.load(&affine_w, testing.allocator, "blk", .bfloat16, s));

    var bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bf);
    try mlx.check(mlx.mlx_astype(&bf, host, .bfloat16, s));
    var half = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(half);
    try mlx.check(mlx.mlx_astype(&half, host, .float16, s));
    try expectDenseLoad("bf", bf, n, k, s);
    try expectDenseLoad("half", half, n, k, s);
}

fn expectDenseLoad(prefix: []const u8, weight: A, n: c_int, k: c_int, s: S) !void {
    var weights = model_mod.Weights.init(testing.allocator);
    defer weights.deinit();
    const key = try std.fmt.allocPrint(testing.allocator, "{s}.weight", .{prefix});
    var handle = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&handle, weight));
    try weights.map.put(key, handle);
    if (available(s)) {
        var l = try Lin.load(&weights, testing.allocator, prefix, .bfloat16, s);
        defer l.deinit();
        try testing.expectEqual(n, l.n);
        try testing.expectEqual(k, l.k);
    } else {
        try testing.expectError(error.W8a8Ineligible, Lin.load(&weights, testing.allocator, prefix, .bfloat16, s));
    }
}

test "w8a8: kernel matches the dequantized matmul on DiT shapes" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!available(s)) return error.SkipZigTest;
    const shapes = [_][2]usize{
        .{ 4096, 4096 },
        .{ 12288, 4096 },
        .{ 4096, 12288 },
    };
    for (shapes, 0..) |sh, si| {
        const n = sh[0];
        const kk = sh[1];
        const vals = try testing.allocator.alloc(f32, n * kk);
        defer testing.allocator.free(vals);
        var prng = std.Random.DefaultPrng.init(0xD17 + si);
        const rnd = prng.random();
        for (vals) |*v| v.* = rnd.float(f32) * 0.2 - 0.1;
        var l = try Lin.fromHost(vals, @intCast(n), @intCast(kk), s);
        defer l.deinit();
        try probeLin(&l, s);
    }
}

test "w8a8: fused quantizer matches op reference for NaN rows" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const rows: usize = 4;
    const k: usize = 512;
    const values = try testing.allocator.alloc(u16, rows * k);
    defer testing.allocator.free(values);
    for (values) |*v| v.* = 0x3f80; // 1.0 in bf16
    values[0] = 0x7fc1;
    values[3 * k - 1] = 0x7fc1;
    @memset(values[3 * k ..], 0x7fc1);
    const input = mlx.mlx_array_new_data(values.ptr, &[_]c_int{ @intCast(rows), @intCast(k) }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(input);

    const ref = try quantizeRowsOps(input, @intCast(rows), @intCast(k), s);
    defer _ = mlx.mlx_array_free(ref.q);
    defer _ = mlx.mlx_array_free(ref.scale);
    const fused = try quantizeRowsFused(input, @intCast(rows), @intCast(k), s);
    defer _ = mlx.mlx_array_free(fused.q);
    defer _ = mlx.mlx_array_free(fused.scale);
    try mlx.check(mlx.mlx_array_eval(ref.scale));
    try mlx.check(mlx.mlx_array_eval(fused.scale));
    const ref_scales = mlx.mlx_array_data_float32(ref.scale) orelse return error.NoData;
    const fused_scales = mlx.mlx_array_data_float32(fused.scale) orelse return error.NoData;
    for ([_]usize{ 0, 2, 3 }) |row| {
        try testing.expect(std.math.isNan(ref_scales[row]));
        try testing.expect(std.math.isNan(fused_scales[row]));
    }
    try testing.expectApproxEqRel(ref_scales[1], fused_scales[1], 1e-6);
}
