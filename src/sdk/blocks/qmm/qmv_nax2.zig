//! 2-bit tensor-unit route for Prism packs on M5, verify widths 6 to 16.
//!
//! `qmv2` covers 1..8 rows with ALU FMAs and is the faster of the two below 6,
//! so this route is tried first and declines there;
//! from 6 up it wins, and past 8 a Bonsai pack has nothing but stock
//! `quantized_matmul` (`verifyQmm`'s NAX lane starts at 4-bit). This route is near flat in rows, so a wider
//! verify stops costing proportionally more: 1.0-1.6x stock at 6 rows, 1.3-2.8x
//! at 8 to 16. Prompt width stays on stock — see `formFor` for the prompt-form
//! measurement and why it is not here.
//!
//! Every 16-k step dequantizes straight into the tensor unit's right operand
//! and runs one 16x32x16 multiply-accumulate, so the codes never land in
//! memory; the affine scale and bias are folded into the operand in f32, which
//! is where stock puts them too.
//!
//! Ported from the Bonsai speedup engine's few-row 2-bit matmul
//! (Layr-Labs/mlxfast-bonsai2-27b-engine, MIT). See NOTICE.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = std.log.scoped(.qmv_nax2);

/// Rows below this stay on qmv2's ALU kernels, which win at narrow widths.
/// qmv2 covers 1..8, so this route is tried FIRST and this number is what
/// decides the overlap: MEASURED on the 27B's four projection shapes (M5 Max,
/// f16), qmv2 and stock are 1.1-1.4x ahead at 4 and 5 rows, this route is
/// ahead on all four at 6, and from 7 it is 1.3-2.4x ahead and near flat in M.
pub const MIN_ROWS: c_int = 6;
/// The widest call this route takes. The kernel tiles rows in 16-high blocks,
/// so it is not a tile height but a measured ceiling: each block re-reads the
/// weights, so past some width stock's own tiling wins.
pub const NARROW_MAX_ROWS_DEFAULT: c_int = 16;
/// Test seam: pin the ceiling so the row tiling is covered whatever ships.
pub var max_rows_override: ?c_int = null;

pub fn narrowMaxRows() c_int {
    return max_rows_override orelse NARROW_MAX_ROWS_DEFAULT;
}

/// Test seam: pin whether the tensor unit is there (null = the device's).
pub var nax_override: ?bool = null;

fn naxHere() bool {
    return nax_override orelse @import("../../../mlx/device.zig").naxAvailable();
}

/// `#include` for the tensor-op primitives; the JIT compiles at Metal 4 on a
/// macOS 26 host, which is the only place `formFor` lets this route run.
pub const HEADER = kernelSource("kernels/common/mpp_tensor.metal");

/// 9..16 activation rows, one 64-column block per threadgroup: simdgroups
/// (0, 1) take its low 32 columns and (2, 3) its high 32, each pair splitting
/// K in half and summing through `red_all`.
///
/// A 256-k block is one contiguous 64-byte line of each weight row, so the
/// four lanes of a fragment quad load it between them and take each other's
/// words back with `simd_shuffle`; one block stays in flight ahead of the one
/// being consumed. Every 16-k step dequantizes into the tensor unit's right
/// operand and runs a 16x32x16 multiply-accumulate, so the codes never land in
/// memory and the affine scale is already in the product.
pub const NARROW_SOURCE = kernelSource("kernels/qmm/qmv_nax2_narrow.metal");

var narrow_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// f16 packs dequantize in half2 (output-identical to the f32 form); the
/// f32 form stays for bf16 and as the test's reference.
pub var half_dq: bool = true;

const NarrowKey = struct { n: c_int, m: c_int, dt: mlx.mlx_dtype, hdq: c_int };
var narrow_cfg: std.AutoHashMapUnmanaged(NarrowKey, mlx.mlx_fast_metal_kernel_config) = .{};

fn narrowConfig(key: NarrowKey) !mlx.mlx_fast_metal_kernel_config {
    if (narrow_cfg.get(key)) |c| return c;
    const config = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const out_shape = [_]c_int{ key.m, key.n };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 2, key.dt));
    // 4 simdgroups per threadgroup, one 64-column block each.
    const tiles = @divTrunc(key.n + 63, 64);
    const row_blocks = @divTrunc(key.m + 15, 16);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 128, tiles, row_blocks));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HDQ", key.hdq));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", key.dt));
    try narrow_cfg.put(std.heap.c_allocator, key, config);
    return config;
}

fn narrowKernel() !mlx.mlx_fast_metal_kernel {
    if (narrow_kernel) |k| return k;
    const in_names = [_][*:0]const u8{ "x", "w", "scales", "biases" };
    const out_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new("nax2_qmm_m16", in_vec, out_vec, NARROW_SOURCE, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    narrow_kernel = k;
    return k;
}

var engaged_logged = false;

fn logEngagedOnce(form: Form, m: c_int, n: c_int, k: c_int) void {
    if (engaged_logged) return;
    engaged_logged = true;
    log.info("[nax2] engaged: {s} form, first call M={d} N={d} K={d}\n", .{ @tagName(form), m, n, k });
}

fn launchNarrow(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, m: c_int, n: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(x);
    const inputs = [_]mlx.mlx_array{ x, w, sc, bi };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    logEngagedOnce(.narrow, m, n, mlx.getShape(x)[mlx.getShape(x).len - 1]);
    const cfg = try narrowConfig(.{ .n = n, .m = m, .dt = dt, .hdq = @intFromBool(dt == .float16 and half_dq) });
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try narrowKernel(), in_vec, cfg, s));
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    const xs = mlx.getShape(x);
    var out_shape: [8]c_int = undefined;
    @memcpy(out_shape[0..xs.len], xs);
    out_shape[xs.len - 1] = n;
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&r, y, &out_shape, xs.len, s));
    return r;
}

pub const Form = enum { decline, narrow };

/// Whether the narrow form serves `m` rows of a `[n, k]` 2-bit group-128
/// weight. Shape rules follow the kernel: K a whole number of 128-groups, N a
/// whole number of the 32-column block it writes.
///
/// Above 16 rows this declines and stock keeps prompt width. A 64x64-tile
/// prompt form was built and MEASURED at 0.87-0.93x stock on all four
/// projection shapes at M 128..2048 (0.55-0.75x with two row subtiles per
/// threadgroup), so it is not in the tree. The reason is the toolchain: this
/// host's Metal has `uint4b_format` but no `uint2b_format`, so a prompt tile
/// has to expand the stored 2-bit words to 4-bit codes in threadgroup memory
/// first, and that costs more than stock's own tiled `qmm` saves. The engine
/// this route came from avoids the expansion with `uint2b_format`, and its
/// expanding variant pays for the staging by quantizing the activation to
/// int8 -- which is not lossless. Re-test if `uint2b_format` ever appears.
pub fn formFor(nax: bool, bits: u32, group_size: u32, m: c_int, n: c_int, k: c_int) Form {
    if (bits != 2 or group_size != 128) return .decline;
    if (!nax) return .decline;
    if (m < MIN_ROWS) return .decline;
    if (m > narrowMaxRows()) return .decline;
    if (@rem(k, 128) != 0 or @rem(n, 32) != 0) return .decline;
    return .narrow;
}

/// The route's entry. Returns null when it declines; the caller then keeps
/// whatever it would have done.
pub fn qmm(
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    s: mlx.mlx_stream,
) !?mlx.mlx_array {
    if (bi.ctx == null or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .float16 and dt != .bfloat16) return null;
    if (mlx.mlx_array_dtype(sc) != dt or mlx.mlx_array_dtype(bi) != dt) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len == 0 or xs.len > 8 or ws.len != 2) return null;
    var m: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| m *= d;
    const k = xs[xs.len - 1];
    const n = ws[0];
    // The kernels read the packed row as uint32 words and index scales by
    // group, so the pack has to be the plain [n, k / 16] affine layout.
    if (ws[1] * 16 != k) return null;
    switch (formFor(naxHere(), bits, group_size, m, n, k)) {
        .decline => return null,
        .narrow => return try launchNarrow(x, w, sc, bi, m, n, s),
    }
}

test "qmv_nax2.formFor: routes 6..16 narrow and declines everything else" {
    const K: c_int = 5120;
    const N: c_int = 17408;
    // Below 6 rows qmv2 owns it.
    for ([_]c_int{ 1, 4, 5 }) |m| try std.testing.expectEqual(Form.decline, formFor(true, 2, 128, m, N, K));
    // 6..16 is the narrow tile; above it stock keeps prompt width.
    for ([_]c_int{ 6, 9, 12, 16 }) |m| try std.testing.expectEqual(Form.narrow, formFor(true, 2, 128, m, N, K));
    for ([_]c_int{ 17, 64, 512, 2048 }) |m| try std.testing.expectEqual(Form.decline, formFor(true, 2, 128, m, N, K));
    // Not our format, and no tensor unit.
    try std.testing.expectEqual(Form.decline, formFor(true, 4, 128, 12, N, K));
    try std.testing.expectEqual(Form.decline, formFor(true, 2, 64, 12, N, K));
    try std.testing.expectEqual(Form.decline, formFor(false, 2, 128, 12, N, K));
    // Shapes the kernel cannot tile.
    try std.testing.expectEqual(Form.decline, formFor(true, 2, 128, 12, N, 5120 + 64));
    try std.testing.expectEqual(Form.decline, formFor(true, 2, 128, 12, 17408 + 16, K));
}

test "qmv_nax2: a CPU-stream call declines" {
    const s = mlx.mlx_default_cpu_stream_new();
    nax_override = true;
    defer nax_override = null;
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_zeros(&x, &[_]c_int{ 8, 256 }, 2, .float16, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_zeros(&w, &[_]c_int{ 64, 16 }, 2, .uint32, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_ones(&sc, &[_]c_int{ 64, 2 }, 2, .float16, s));
    try std.testing.expectEqual(@as(?mlx.mlx_array, null), try qmm(x, w, sc, sc, 2, 128, s));
}

test "qmv_nax2: every real Bonsai projection shape routes at verify width" {
    // K 5120 -> N {17408, 34816, 6144, 1024, 248320} and K 17408 -> N 5120.
    const shapes = [_][2]c_int{
        .{ 17408, 5120 }, .{ 34816, 5120 },  .{ 6144, 5120 },
        .{ 1024, 5120 },  .{ 248320, 5120 }, .{ 5120, 17408 },
    };
    for (shapes) |nk| {
        for ([_]c_int{ 6, 9, 12, 16 }) |m| {
            const f = formFor(true, 2, 128, m, nk[0], nk[1]);
            try std.testing.expect(f != .decline);
        }
    }
}

const RmsMax = struct { rms: f32, max: f32 };

fn errVsTruth(got: mlx.mlx_array, truth: []const f32, m: usize, n: usize, s: mlx.mlx_stream) ![]RmsMax {
    var g32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g32);
    try mlx.check(mlx.mlx_astype(&g32, got, .float32, s));
    try mlx.check(mlx.mlx_array_eval(g32));
    const g = mlx.mlx_array_data_float32(g32).?;
    const out = try std.testing.allocator.alloc(RmsMax, m);
    for (0..m) |r| {
        var ss: f64 = 0;
        var mx: f32 = 0;
        for (0..n) |c| {
            const d = g[r * n + c] - truth[r * n + c];
            try std.testing.expect(std.math.isFinite(g[r * n + c]));
            ss += d * d;
            mx = @max(mx, @abs(d));
        }
        out[r] = .{ .rms = @floatCast(@sqrt(ss / @as(f64, @floatFromInt(n)))), .max = mx };
    }
    return out;
}

/// The kernel is Metal 4 tensor ops: off a tensor unit it compiles and
/// answers garbage, so a GPU dispatch is only a test where the hardware is.
fn skipWithoutNax() !void {
    if (!@import("../../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
}

fn parityAgainstStock(widths: []const c_int) !void {
    try skipWithoutNax();
    const s = mlx.gpuStream();
    const n: c_int = 1024;
    const k: c_int = 1536; // 12 whole 128-groups
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const nu: usize = @intCast(n);
    const ku: usize = @intCast(k);

    const codes = try std.testing.allocator.alloc(u32, nu * ku / 16);
    defer std.testing.allocator.free(codes);
    for (codes) |*wd| {
        var v: u32 = 0;
        for (0..16) |j| v |= @as(u32, rnd.uintLessThan(u32, 3)) << @intCast(2 * j);
        wd.* = v;
    }
    const sc32 = try std.testing.allocator.alloc(f32, nu * ku / 128);
    defer std.testing.allocator.free(sc32);
    for (sc32) |*e| e.* = 0.005 + 0.02 * rnd.float(f32);
    const wq = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ n, @divExact(k, 16) }, 2, .uint32);
    defer _ = mlx.mlx_array_free(wq);
    const sc_f = mlx.mlx_array_new_data(sc32.ptr, &[_]c_int{ n, @divExact(k, 128) }, 2, .float32);
    defer _ = mlx.mlx_array_free(sc_f);

    const dt: mlx.mlx_dtype = .float16; // the pack is served in f16
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_astype(&sc, sc_f, dt, s));
    var bi = mlx.mlx_array_new(); // ternary: bias == -scale
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_negative(&bi, sc, s));

    // f32 weights from the SAME dt-rounded scales and biases: the truth.
    var sc_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc_t);
    var bi_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi_t);
    try mlx.check(mlx.mlx_astype(&sc_t, sc, .float32, s));
    try mlx.check(mlx.mlx_astype(&bi_t, bi, .float32, s));
    var w_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_t);
    try mlx.check(mlx.mlx_dequantize(&w_t, wq, sc_t, bi_t, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(2), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    var w_tt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_tt);
    try mlx.check(mlx.mlx_transpose(&w_tt, w_t, s));

    nax_override = true;
    defer nax_override = null;

    for (widths) |m| {
        const mu: usize = @intCast(m);
        const xv = try std.testing.allocator.alloc(f32, mu * ku);
        defer std.testing.allocator.free(xv);
        for (xv, 0..) |*e, i| e.* = if (i % 97 == 0) 3.0e3 * rnd.floatNorm(f32) else if (i % 13 == 0) 1e-3 * rnd.floatNorm(f32) else rnd.floatNorm(f32);
        const x32 = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ 1, m, k }, 3, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, dt, s));
        var xt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xt);
        try mlx.check(mlx.mlx_astype(&xt, x, .float32, s));
        var truth_a = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(truth_a);
        try mlx.check(mlx.mlx_matmul(&truth_a, xt, w_tt, s));
        try mlx.check(mlx.mlx_array_eval(truth_a));
        const truth = mlx.mlx_array_data_float32(truth_a).?[0 .. mu * nu];

        var stock = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stock);
        try mlx.check(mlx.mlx_quantized_matmul(&stock, x, wq, sc, bi, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(2), "affine", s));
        const es = try errVsTruth(stock, truth, mu, nu, s);
        defer std.testing.allocator.free(es);

        // The route claims this width, so it must produce a result.
        const got = (try qmm(x, wq, sc, bi, 2, 128, s)) orelse {
            std.debug.print("[qmv_nax2] declined at M={d} while formFor says {s}\n", .{ m, @tagName(formFor(true, 2, 128, m, n, k)) });
            return error.RouteDeclined;
        };
        defer _ = mlx.mlx_array_free(got);
        try std.testing.expectEqual(dt, mlx.mlx_array_dtype(got));
        try std.testing.expectEqualSlices(c_int, mlx.getShape(stock), mlx.getShape(got));
        const eg = try errVsTruth(got, truth, mu, nu, s);
        defer std.testing.allocator.free(eg);
        for (es, eg) |a, b| {
            try std.testing.expect(b.rms <= 1.05 * a.rms);
            try std.testing.expect(b.max <= 1.05 * a.max);
        }
    }
}

test "qmv_nax2 narrow: no worse than stock quantized_matmul against f32 truth (M 6..16)" {
    try parityAgainstStock(&.{ 6, 9, 12, 16 });
}

// Rows tile in 16-high blocks, so a width past one block and a short last
// block are their own cases: 17 is 16 + 1, 32 two whole blocks, 40 two plus 8.
test "qmv_nax2 narrow: row tiling holds across and past a 16-row block" {
    max_rows_override = 64;
    defer max_rows_override = null;
    try parityAgainstStock(&.{ 17, 24, 32, 40, 64 });
}

test "qmv_nax2 narrow: the half dequant is bit-identical to the f32 dequant" {
    const s = mlx.gpuStream();
    const n: c_int = 1024;
    const k: c_int = 1536;
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    const nu: usize = @intCast(n);
    const ku: usize = @intCast(k);
    const codes = try std.testing.allocator.alloc(u32, nu * ku / 16);
    defer std.testing.allocator.free(codes);
    for (codes) |*wd| wd.* = rnd.int(u32);
    // General affine, not just ternary: independent scales and biases.
    const sb = try std.testing.allocator.alloc(f32, 2 * nu * ku / 128);
    defer std.testing.allocator.free(sb);
    for (sb, 0..) |*e, i| e.* = if (i % 2 == 0) 1e-4 + 0.05 * rnd.float(f32) else 0.1 * rnd.floatNorm(f32);
    const wq = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ n, @divExact(k, 16) }, 2, .uint32);
    defer _ = mlx.mlx_array_free(wq);
    const g = @divExact(k, 128);
    const sbf = mlx.mlx_array_new_data(sb.ptr, &[_]c_int{ 2, n, g }, 3, .float32);
    defer _ = mlx.mlx_array_free(sbf);
    var sbh = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sbh);
    try mlx.check(mlx.mlx_astype(&sbh, sbf, .float16, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_slice(&sc, sbh, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, n, g }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    try mlx.check(mlx.mlx_slice(&bi, sbh, &[_]c_int{ 1, 0, 0 }, 3, &[_]c_int{ 2, n, g }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var sc2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc2);
    var bi2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi2);
    try mlx.check(mlx.mlx_reshape(&sc2, sc, &[_]c_int{ n, g }, 2, s));
    try mlx.check(mlx.mlx_reshape(&bi2, bi, &[_]c_int{ n, g }, 2, s));
    var scc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scc);
    var bic = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bic);
    try mlx.check(mlx.mlx_contiguous(&scc, sc2, false, s));
    try mlx.check(mlx.mlx_contiguous(&bic, bi2, false, s));
    try skipWithoutNax();
    nax_override = true;
    defer nax_override = null;
    defer half_dq = true;
    for ([_]c_int{ 6, 12, 16 }) |m| {
        const mu: usize = @intCast(m);
        const xv = try std.testing.allocator.alloc(f32, mu * ku);
        defer std.testing.allocator.free(xv);
        for (xv) |*e| e.* = rnd.floatNorm(f32);
        const x32 = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ m, k }, 2, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, .float16, s));
        var outs: [2][]u16 = undefined;
        for ([_]bool{ false, true }, 0..) |h, i| {
            half_dq = h;
            const o = (try qmm(x, wq, scc, bic, 2, 128, s)) orelse return error.RouteDeclined;
            defer _ = mlx.mlx_array_free(o);
            try mlx.check(mlx.mlx_array_eval(o));
            const p16: [*]const u16 = @ptrCast(@alignCast(mlx.mlx_array_data_float16(o).?));
            outs[i] = try std.testing.allocator.dupe(u16, p16[0 .. mu * nu]);
        }
        defer for (outs) |o| std.testing.allocator.free(o);
        try std.testing.expectEqualSlices(u16, outs[0], outs[1]);
    }
}
