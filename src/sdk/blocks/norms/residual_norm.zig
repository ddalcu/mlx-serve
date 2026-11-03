//! sdk/blocks/norms/residual_norm.
//! Owns add_norm_max_rows, RESIDUAL_NORM_HEADER, RESIDUAL_NORM_SOURCE, residual_norm_kernel, residual_norm_engaged, add_rmsnorm_override, and 12 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const mamba2_decode = @import("../ssm/mamba2_decode.zig");
const add_norm = @import("add_norm.zig");
const log = @import("../../util/log.zig");
const testing = std.testing;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const ShapeKey = @import("../moe/router_kernel.zig").ShapeKey;

pub var add_norm_max_rows: c_int = mamba2_decode.MAX_ROWS; // test seam

// ── Fused residual add + RMSNorm ──

/// One dispatch for the residual seam between blocks:
///   n1  = PRE1 ? w1 * rms(b1, eps1) : b1
///   n2  = B2 ? (PRE2 ? w2 * rms(b2, eps2) : b2) : -
///   c   = B2 ? n1 + n2 : n1
///   c   = PREC ? wc * rms(c, epsc) : c
///   res = (a + c) [* scale]
///   out = POST ? wp * rms(res, eps) : -,  out2 = POST2 ? wp2 * rms(res, eps) : -
/// Every op rounds to T exactly where MLX's separate kernels do, and each
/// reduction keeps MLX's `rms_single_row` geometry (ceil(AXIS/4) threads,
/// four consecutive elements each, simd_sum, a zeroed partials plane, one
/// simd_sum over it) so the tree is MLX's tree, not merely its total.
/// `metal::precise::rsqrt` as upstream.
pub const RESIDUAL_NORM_HEADER = kernelSource("kernels/norms/residual_norm_header.metal");
pub const RESIDUAL_NORM_SOURCE = kernelSource("kernels/norms/residual_norm.metal");

var residual_norm_kernel: ?mlx.mlx_fast_metal_kernel = null;
var residual_norm_engaged: bool = false;

fn getResidualNormKernel() !mlx.mlx_fast_metal_kernel {
    if (residual_norm_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "a", "b1", "b2", "w1", "eps1", "w2", "eps2", "wc", "epsc", "scale", "wp", "wp2", "eps" };
    const output_names = [_][*:0]const u8{ "res", "out", "out2" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_residual_norm",
        in_vec,
        out_vec,
        RESIDUAL_NORM_SOURCE,
        RESIDUAL_NORM_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    residual_norm_kernel = kernel;
    return kernel;
}

/// The stages of `fusedResidualNorm`; a null-ctx weight switches its stage
/// off. Eps arrays are 0-dim f32 scalars; `eps` serves both post-norms.
pub const ResidualNormOpts = struct {
    b2: mlx.mlx_array = .{ .ctx = null },
    w1: mlx.mlx_array = .{ .ctx = null },
    eps1: mlx.mlx_array = .{ .ctx = null },
    w2: mlx.mlx_array = .{ .ctx = null },
    eps2: mlx.mlx_array = .{ .ctx = null },
    wc: mlx.mlx_array = .{ .ctx = null },
    epsc: mlx.mlx_array = .{ .ctx = null },
    scale: mlx.mlx_array = .{ .ctx = null },
    w_post: mlx.mlx_array = .{ .ctx = null },
    w_post2: mlx.mlx_array = .{ .ctx = null },
    eps: mlx.mlx_array = .{ .ctx = null },
};

const ResidualNormCfgKey = struct { shape: ShapeKey, dtype: mlx.mlx_dtype, flags: u8 };
var residual_norm_cfg: [8]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var residual_norm_cfg_key: [8]ResidualNormCfgKey = @splat(std.mem.zeroes(ResidualNormCfgKey));
var residual_norm_cfg_next: usize = 0;

pub const AddNormResult = struct { sum: mlx.mlx_array, normed: mlx.mlx_array, normed2: mlx.mlx_array = .{ .ctx = null } };

/// The residual seam in one dispatch (RESIDUAL_NORM_SOURCE); `normed` /
/// `normed2` are null-ctx when their post-norm is off. Null → caller keeps
/// the composed ops.
pub fn fusedResidualNorm(s: mlx.mlx_stream, a: mlx.mlx_array, b1: mlx.mlx_array, o: ResidualNormOpts) !?AddNormResult {
    if (!mlx.streamIsGpu(s)) return null;
    const b2 = o.b2.ctx != null;
    const pre1 = o.w1.ctx != null;
    const pre2 = b2 and o.w2.ctx != null;
    const prec = o.wc.ctx != null;
    const scaled = o.scale.ctx != null;
    const post = o.w_post.ctx != null;
    const post2 = o.w_post2.ctx != null;
    if (!pre1 and !b2 and !prec and !scaled and !post and !post2) return null;
    if ((pre1 and o.eps1.ctx == null) or (pre2 and o.eps2.ctx == null) or (prec and o.epsc.ctx == null) or ((post or post2) and o.eps.ctx == null)) return null;
    const dt = mlx.mlx_array_dtype(a);
    if (dt != .bfloat16 and dt != .float16 and dt != .float32) return null;
    const ash = mlx.getShape(a);
    if (ash.len == 0 or ash.len > 4) return null;
    for ([_]mlx.mlx_array{ b1, o.b2 }) |b| {
        if (b.ctx == null) continue;
        if (mlx.mlx_array_dtype(b) != dt) return null;
        const bsh = mlx.getShape(b);
        if (bsh.len != ash.len) return null;
        for (ash, bsh) |x, y| if (x != y) return null; // no broadcasting
    }
    const axis = ash[ash.len - 1];
    // Above RMS_LOOPED_LIMIT mlx switches to a different kernel with a
    // different reduction shape; decline rather than guess.
    if (axis <= 0 or axis > 4096) return null;
    for ([_]mlx.mlx_array{ o.w1, o.w2, o.wc, o.w_post, o.w_post2 }) |w| {
        if (w.ctx == null) continue;
        const wsh = mlx.getShape(w);
        if (wsh.len != 1 or wsh[0] != axis or mlx.mlx_array_dtype(w) != dt) return null;
    }
    if (scaled) {
        const ssh = mlx.getShape(o.scale);
        if (ssh.len != 1 or ssh[0] != 1 or mlx.mlx_array_dtype(o.scale) != dt) return null;
    }

    var rows: c_int = 1;
    for (ash[0 .. ash.len - 1]) |d| rows *= d;
    if (rows < 1) return null;

    // MLX's own launch: ceil(axis / N_READS) threads, rounded up to simdgroups.
    const tg: c_int = @divTrunc(@divTrunc(axis + 3, 4) + 31, 32) * 32;
    if (tg > 1024) return null;

    const flags: u8 = @as(u8, @intFromBool(pre1)) | @as(u8, @intFromBool(b2)) << 1 | @as(u8, @intFromBool(pre2)) << 2 |
        @as(u8, @intFromBool(prec)) << 3 | @as(u8, @intFromBool(scaled)) << 4 | @as(u8, @intFromBool(post)) << 5 | @as(u8, @intFromBool(post2)) << 6;
    const key = ResidualNormCfgKey{ .shape = ShapeKey.from(ash), .dtype = dt, .flags = flags };
    var slot: ?usize = null;
    for (residual_norm_cfg, 0..) |c, k| {
        if (c != null and std.meta.eql(residual_norm_cfg_key[k], key)) slot = k;
    }
    if (slot == null) {
        const k = residual_norm_cfg_next;
        residual_norm_cfg_next = (k + 1) % residual_norm_cfg.len;
        if (residual_norm_cfg[k]) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, ash.ptr, ash.len, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "AXIS", axis));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "PRE1", pre1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "B2", b2));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "PRE2", pre2));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "PREC", prec));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "SCALE", scaled));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "POST", post));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "POST2", post2));
        residual_norm_cfg[k] = config;
        residual_norm_cfg_key[k] = key;
        slot = k;
    }

    // Stages that are off still need a bound input; they never read it.
    const some_eps = if (o.eps.ctx != null) o.eps else if (o.eps1.ctx != null) o.eps1 else if (o.eps2.ctx != null) o.eps2 else o.epsc;
    const pick = struct {
        fn f(x: mlx.mlx_array, fallback: mlx.mlx_array) mlx.mlx_array {
            return if (x.ctx != null) x else fallback;
        }
    }.f;
    const inputs_arr = [_]mlx.mlx_array{
        a,
        b1,
        pick(o.b2, b1),
        pick(o.w1, b1),
        pick(o.eps1, some_eps),
        pick(o.w2, b1),
        pick(o.eps2, some_eps),
        pick(o.wc, b1),
        pick(o.epsc, some_eps),
        pick(o.scale, b1),
        pick(o.w_post, b1),
        pick(o.w_post2, b1),
        some_eps,
    };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getResidualNormKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, residual_norm_cfg[slot.?].?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 3) return error.MetalKernelBadOutputCount;
    var outs: [3]mlx.mlx_array = undefined;
    for (&outs, 0..) |*oa, k| {
        oa.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(oa, outputs_vec, k));
    }
    if (!post) {
        _ = mlx.mlx_array_free(outs[1]);
        outs[1] = .{ .ctx = null };
    }
    if (!post2) {
        _ = mlx.mlx_array_free(outs[2]);
        outs[2] = .{ .ctx = null };
    }
    if (!residual_norm_engaged) {
        residual_norm_engaged = true;
        log.info("[layer] fused residual+RMSNorm kernel engaged: axis={d} tg={d}\n", .{ axis, tg });
    }
    return .{ .sum = outs[0], .normed = outs[1], .normed2 = outs[2] };
}

/// `sum = a + b` and `normed = rms_norm(sum, w, eps)` in one dispatch, bit-identical to the composed pair; null →
/// the caller keeps the two ops. Pays only where a frame loop is per-NODE CPU-bound (Qwen3-TTS); on a decoder layer it
/// measured ~1% negative (one saved 4 KB read does not pay for leaving the metallib norm kernel). `eps_arr` is a 0-dim
/// f32 scalar.
pub fn fusedAddRmsNorm(
    s: mlx.mlx_stream,
    a: mlx.mlx_array,
    b: mlx.mlx_array,
    w: mlx.mlx_array,
    eps_arr: mlx.mlx_array,
) !?AddNormResult {
    if (w.ctx == null) return null;
    return fusedResidualNorm(s, a, b, .{ .w_post = w, .eps = eps_arr });
}

// ── tests ──

// Bar: the fused residual add + next norm is BIT-EQUAL to add -> fast rms_norm,
// on the plain arm and on the MoE-combine arm (with and without a shared expert).
test "add_norm: plain and MoE-combine arms match the composed ops" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const R: c_int = 3;
    const D: c_int = 2688;
    const K: c_int = 3;
    const mk = struct {
        fn f(shape: []const c_int, seed: f32, dt: mlx.mlx_dtype, st: mlx.mlx_stream) !mlx.mlx_array {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            var buf: [3 * 3 * 2688]f32 = undefined;
            for (buf[0..n], 0..) |*v, i| v.* = @sin(seed + 0.29 * @as(f32, @floatFromInt(i)));
            const src = mlx.mlx_array_new_data(&buf, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(src);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&out, src, dt, st));
            return out;
        }
    }.f;
    const cmp = struct {
        fn f(a: mlx.mlx_array, b: mlx.mlx_array, st: mlx.mlx_stream, tol: f32) !void {
            var af = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(af);
            var bf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bf);
            try mlx.check(mlx.mlx_astype(&af, a, .float32, st));
            try mlx.check(mlx.mlx_astype(&bf, b, .float32, st));
            try mlx.check(mlx.mlx_array_eval(af));
            try mlx.check(mlx.mlx_array_eval(bf));
            const pa = mlx.mlx_array_data_float32(af) orelse return error.NoData;
            const pb = mlx.mlx_array_data_float32(bf) orelse return error.NoData;
            const n: usize = @intCast(mlx.mlx_array_size(af));
            var max_ref: f32 = 0;
            var max_err: f32 = 0;
            for (0..n) |i| {
                try std.testing.expect(std.math.isFinite(pa[i]) and std.math.isFinite(pb[i]));
                max_ref = @max(max_ref, @abs(pa[i]));
                max_err = @max(max_err, @abs(pa[i] - pb[i]));
            }
            if (!(max_ref > 0.01) or max_err > tol * max_ref) {
                std.debug.print("add_norm mismatch: max_err={d:.5} max_ref={d:.5}\n", .{ max_err, max_ref });
                return error.AddNormMismatch;
            }
        }
    }.f;
    const h = try mk(&.{ 1, R, D }, 0.1, .bfloat16, s);
    defer _ = mlx.mlx_array_free(h);
    const x = try mk(&.{ 1, R, D }, 0.7, .bfloat16, s);
    defer _ = mlx.mlx_array_free(x);
    const w = try mk(&.{D}, 1.3, .bfloat16, s);
    defer _ = mlx.mlx_array_free(w);
    const eps = mlx.mlx_array_new_float(1e-5);
    defer _ = mlx.mlx_array_free(eps);

    // Plain arm.
    {
        const got = (try add_norm.addNorm(h, x, w, eps, s)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got.h);
        defer _ = mlx.mlx_array_free(got.normed);
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, h, x, s));
        var normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed);
        try mlx.check(mlx.mlx_fast_rms_norm(&normed, sum, w, 1e-5, s));
        try cmp(sum, got.h, s, 0);
        try cmp(normed, got.normed, s, 0);
    }
    // MoE-combine arm: f32 weighted sum over K slots, then the shared expert
    // (and once without one).
    for ([_]bool{ true, false }) |with_shared| {
        const y = try mk(&.{ R, K, D }, 2.1, .bfloat16, s);
        defer _ = mlx.mlx_array_free(y);
        const wt = try mk(&.{ R, K }, 0.4, .float32, s);
        defer _ = mlx.mlx_array_free(wt);
        const xs = try mk(&.{ R, D }, 1.9, .bfloat16, s);
        defer _ = mlx.mlx_array_free(xs);
        const got = (try add_norm.moeCombineAddNorm(h, y, wt, if (with_shared) xs else null, w, eps, s)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got.h);
        defer _ = mlx.mlx_array_free(got.normed);
        var wt_e = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt_e);
        try mlx.check(mlx.mlx_expand_dims(&wt_e, wt, -1, s));
        var prod = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(prod);
        try mlx.check(mlx.mlx_multiply(&prod, y, wt_e, s));
        var summed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(summed);
        try mlx.check(mlx.mlx_sum_axis(&summed, prod, 1, false, s));
        var routed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(routed);
        try mlx.check(mlx.mlx_astype(&routed, summed, .bfloat16, s));
        var delta = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(delta);
        if (with_shared) try mlx.check(mlx.mlx_add(&delta, routed, xs, s)) else try mlx.check(mlx.mlx_array_set(&delta, routed));
        var delta3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(delta3);
        try mlx.check(mlx.mlx_reshape(&delta3, delta, &[_]c_int{ 1, R, D }, 3, s));
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, h, delta3, s));
        var normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed);
        try mlx.check(mlx.mlx_fast_rms_norm(&normed, sum, w, 1e-5, s));
        try cmp(sum, got.h, s, 0);
        try cmp(normed, got.normed, s, 0);
    }
}

test "fused residual norm chain is bit-identical to the composed ops in every stage combination" {
    // Bar: pre-norm / scale / post-norm on and off in all eight combinations,
    // aligned and ragged widths, rows 1 and 3, bf16 and f32, byte for byte
    // against rms_norm -> add -> multiply -> rms_norm as separate MLX ops.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EA3);
    const rnd = prng.random();
    const EPS_PRE: f32 = 1e-5;
    const EPS: f32 = 1e-6;
    const eps_pre = mlx.mlx_array_new_float(EPS_PRE);
    defer _ = mlx.mlx_array_free(eps_pre);
    const eps = mlx.mlx_array_new_float(EPS);
    defer _ = mlx.mlx_array_free(eps);
    const rand = struct {
        fn f(a: std.mem.Allocator, r: std.Random, st: mlx.mlx_stream, shape: []const c_int, scale: f32, offset: f32, dt: mlx.mlx_dtype) !mlx.mlx_array {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const buf = try a.alloc(f32, n);
            defer a.free(buf);
            for (buf) |*v| v.* = (r.float(f32) - 0.5) * scale + offset;
            const a32 = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(a32);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&out, a32, dt, st));
            return out;
        }
    }.f;
    for ([_]c_int{ 512, 2050, 2560 }) |axis| {
        for ([_]c_int{ 1, 3 }) |rows| {
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const sh = [_]c_int{ rows, axis };
                const a = try rand(allocator, rnd, s, &sh, 6.0, 0.0, dt);
                defer _ = mlx.mlx_array_free(a);
                const b = try rand(allocator, rnd, s, &sh, 6.0, 0.0, dt);
                defer _ = mlx.mlx_array_free(b);
                const wsh = [_]c_int{axis};
                const w_pre = try rand(allocator, rnd, s, &wsh, 1.0, 1.0, dt);
                defer _ = mlx.mlx_array_free(w_pre);
                const w_post = try rand(allocator, rnd, s, &wsh, 1.0, 1.0, dt);
                defer _ = mlx.mlx_array_free(w_post);
                const ssh = [_]c_int{1};
                const scale = try rand(allocator, rnd, s, &ssh, 0.5, 0.9, dt);
                defer _ = mlx.mlx_array_free(scale);
                const none = mlx.mlx_array{ .ctx = null };
                for (0..8) |combo| {
                    const pre = combo & 1 != 0;
                    const scl = combo & 2 != 0;
                    const post = combo & 4 != 0;
                    if (!pre and !scl and !post) continue;
                    // Reference chain.
                    var nb = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(nb);
                    if (pre) try mlx.check(mlx.mlx_fast_rms_norm(&nb, b, w_pre, EPS_PRE, s)) else _ = mlx.mlx_array_set(&nb, b);
                    var sum = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(sum);
                    try mlx.check(mlx.mlx_add(&sum, a, nb, s));
                    var res = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(res);
                    if (scl) try mlx.check(mlx.mlx_multiply(&res, sum, scale, s)) else _ = mlx.mlx_array_set(&res, sum);
                    var ref_norm = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(ref_norm);
                    if (post) try mlx.check(mlx.mlx_fast_rms_norm(&ref_norm, res, w_post, EPS, s));

                    const got = (try fusedResidualNorm(s, a, b, .{ .w1 = if (pre) w_pre else none, .eps1 = eps_pre, .scale = if (scl) scale else none, .w_post = if (post) w_post else none, .eps = eps })) orelse return error.FusedResidualNormDeclined;
                    defer _ = mlx.mlx_array_free(got.sum);
                    defer if (post) {
                        _ = mlx.mlx_array_free(got.normed);
                    };
                    try testing.expect((got.normed.ctx != null) == post);

                    const cnt: usize = @intCast(rows * axis);
                    const h1 = try allocator.alloc(f32, cnt);
                    defer allocator.free(h1);
                    const h2 = try allocator.alloc(f32, cnt);
                    defer allocator.free(h2);
                    try testReadF32(res, h1, s);
                    try testReadF32(got.sum, h2, s);
                    try testing.expectEqualSlices(f32, h1, h2);
                    if (post) {
                        try testReadF32(ref_norm, h1, s);
                        try testReadF32(got.normed, h2, s);
                        try testing.expectEqualSlices(f32, h1, h2);
                    }
                }
            }
        }
    }
}

test "fused residual norm chain: two normed branches, combine norm and both post-norms" {
    // The Gemma MoE tail shape: res = (a + wc*rms(w1*rms(b1) + w2*rms(b2))) * s,
    // out = wp*rms(res), out2 = wp2*rms(res); byte for byte against the ops.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x2B2B);
    const rnd = prng.random();
    const EPS: f32 = 1e-6;
    const eps = mlx.mlx_array_new_float(EPS);
    defer _ = mlx.mlx_array_free(eps);
    const axis: c_int = 2816;
    const rows: c_int = 1;
    const sh = [_]c_int{ rows, axis };
    const wsh = [_]c_int{axis};
    const ssh = [_]c_int{1};
    const mk = struct {
        fn f(a: std.mem.Allocator, r: std.Random, st: mlx.mlx_stream, shape: []const c_int, scale: f32, offset: f32) !mlx.mlx_array {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const buf = try a.alloc(f32, n);
            defer a.free(buf);
            for (buf) |*v| v.* = (r.float(f32) - 0.5) * scale + offset;
            const a32 = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(a32);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&out, a32, .bfloat16, st));
            return out;
        }
    }.f;
    const a = try mk(allocator, rnd, s, &sh, 6.0, 0.0);
    defer _ = mlx.mlx_array_free(a);
    const b1 = try mk(allocator, rnd, s, &sh, 6.0, 0.0);
    defer _ = mlx.mlx_array_free(b1);
    const b2 = try mk(allocator, rnd, s, &sh, 6.0, 0.0);
    defer _ = mlx.mlx_array_free(b2);
    var ws: [5]mlx.mlx_array = undefined;
    for (&ws) |*w| w.* = try mk(allocator, rnd, s, &wsh, 1.0, 1.0);
    defer for (ws) |w| {
        _ = mlx.mlx_array_free(w);
    };
    const scale = try mk(allocator, rnd, s, &ssh, 0.5, 0.9);
    defer _ = mlx.mlx_array_free(scale);

    var n1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n1);
    try mlx.check(mlx.mlx_fast_rms_norm(&n1, b1, ws[0], EPS, s));
    var n2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n2);
    try mlx.check(mlx.mlx_fast_rms_norm(&n2, b2, ws[1], EPS, s));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_add(&c, n1, n2, s));
    var nc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nc);
    try mlx.check(mlx.mlx_fast_rms_norm(&nc, c, ws[2], EPS, s));
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, a, nc, s));
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);
    try mlx.check(mlx.mlx_multiply(&res, sum, scale, s));
    var o1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o1);
    try mlx.check(mlx.mlx_fast_rms_norm(&o1, res, ws[3], EPS, s));
    var o2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o2);
    try mlx.check(mlx.mlx_fast_rms_norm(&o2, res, ws[4], EPS, s));

    const got = (try fusedResidualNorm(s, a, b1, .{ .b2 = b2, .w1 = ws[0], .eps1 = eps, .w2 = ws[1], .eps2 = eps, .wc = ws[2], .epsc = eps, .scale = scale, .w_post = ws[3], .w_post2 = ws[4], .eps = eps })) orelse return error.FusedResidualNormDeclined;
    defer _ = mlx.mlx_array_free(got.sum);
    defer _ = mlx.mlx_array_free(got.normed);
    defer _ = mlx.mlx_array_free(got.normed2);
    const cnt: usize = @intCast(rows * axis);
    const h1 = try allocator.alloc(f32, cnt);
    defer allocator.free(h1);
    const h2 = try allocator.alloc(f32, cnt);
    defer allocator.free(h2);
    const pairs = [_][2]mlx.mlx_array{ .{ res, got.sum }, .{ o1, got.normed }, .{ o2, got.normed2 } };
    for (pairs) |pr| {
        try testReadF32(pr[0], h1, s);
        try testReadF32(pr[1], h2, s);
        try testing.expectEqualSlices(f32, h1, h2);
    }
}

test "fused residual+RMSNorm is bit-identical to mlx_add + mlx_fast_rms_norm" {
    // Same bar as the other fusions, and the same reason it is reachable: the
    // kernel keeps MLX's launch geometry, so the reduction is the same TREE and
    // not merely the same sum. Widths chosen to hit the aligned path (2048,
    // 512), the ragged tail (2050 — not a multiple of 4), and a non-power-of-2
    // axis (5120) where `acc / axis_size` could differ if a compiler turned the
    // divide into a reciprocal multiply.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x0DDBA11);
    const rnd = prng.random();

    const EPS: f32 = 1e-6;
    const eps_arr = mlx.mlx_array_new_float(EPS);
    defer _ = mlx.mlx_array_free(eps_arr);

    for ([_]c_int{ 512, 1536, 2048, 2050 }) |axis| {
        for ([_]c_int{ 1, 3 }) |rows| {
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const cnt: usize = @intCast(rows * axis);
                const abuf = try allocator.alloc(f32, cnt);
                defer allocator.free(abuf);
                const bbuf = try allocator.alloc(f32, cnt);
                defer allocator.free(bbuf);
                const wbuf = try allocator.alloc(f32, @intCast(axis));
                defer allocator.free(wbuf);
                for (abuf) |*v| v.* = (rnd.float(f32) - 0.5) * 6.0;
                for (bbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 6.0;
                for (wbuf) |*v| v.* = 0.5 + rnd.float(f32);

                const sh = [_]c_int{ rows, axis };
                const wsh = [_]c_int{axis};
                const a32 = mlx.mlx_array_new_data(abuf.ptr, &sh, 2, .float32);
                defer _ = mlx.mlx_array_free(a32);
                const b32 = mlx.mlx_array_new_data(bbuf.ptr, &sh, 2, .float32);
                defer _ = mlx.mlx_array_free(b32);
                const w32 = mlx.mlx_array_new_data(wbuf.ptr, &wsh, 1, .float32);
                defer _ = mlx.mlx_array_free(w32);
                var a = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(a);
                try mlx.check(mlx.mlx_astype(&a, a32, dt, s));
                var b = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(b);
                try mlx.check(mlx.mlx_astype(&b, b32, dt, s));
                var w = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(w);
                try mlx.check(mlx.mlx_astype(&w, w32, dt, s));

                var ref_sum = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ref_sum);
                try mlx.check(mlx.mlx_add(&ref_sum, a, b, s));
                var ref_norm = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ref_norm);
                try mlx.check(mlx.mlx_fast_rms_norm(&ref_norm, ref_sum, w, EPS, s));

                const got = (try fusedAddRmsNorm(s, a, b, w, eps_arr)) orelse return error.FusedAddNormDeclined;
                defer _ = mlx.mlx_array_free(got.sum);
                defer _ = mlx.mlx_array_free(got.normed);

                const h1 = try allocator.alloc(f32, cnt);
                defer allocator.free(h1);
                const h2 = try allocator.alloc(f32, cnt);
                defer allocator.free(h2);
                try testReadF32(ref_sum, h1, s);
                try testReadF32(got.sum, h2, s);
                try testing.expectEqualSlices(f32, h1, h2);
                try testReadF32(ref_norm, h1, s);
                try testReadF32(got.normed, h2, s);
                try testing.expectEqualSlices(f32, h1, h2);
            }
        }
    }
}

test "fused residual+RMSNorm declines what it cannot reproduce" {
    const s = mlx.gpuStream();
    // mlx_array_new_data COPIES shape-worth of bytes — the buffer must really
    // hold 8192 floats (a 16-float buffer here "passed" by overreading the
    // stack until a layout shift put the guard page in range).
    var buf: [8192]f32 = @splat(0.25);
    // Above RMS_LOOPED_LIMIT mlx switches to a different kernel with a
    // different reduction shape.
    const big = [_]c_int{ 1, 8192 };
    const wbig = [_]c_int{8192};
    const a = mlx.mlx_array_new_data(&buf, &big, 2, .float32);
    defer _ = mlx.mlx_array_free(a);
    const w = mlx.mlx_array_new_data(&buf, &wbig, 1, .float32);
    defer _ = mlx.mlx_array_free(w);
    const eps = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps);
    try testing.expect((try fusedAddRmsNorm(s, a, a, w, eps)) == null);
    // Mismatched shapes must decline rather than read out of bounds.
    const s2 = [_]c_int{ 1, 4 };
    const small = mlx.mlx_array_new_data(&buf, &s2, 2, .float32);
    defer _ = mlx.mlx_array_free(small);
    try testing.expect((try fusedAddRmsNorm(s, a, small, w, eps)) == null);
}
