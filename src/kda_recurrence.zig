//! Prefill recurrence of the vector-gated delta rule (KDA) with the bounded-sigmoid gate,
//! one threadgroup per GPU core: oMLX's per-core kernel (`kernels/kda_percore.metal`, see
//! NOTICE). The gate `exp(lb * sigmoid(exp(A_log) * (a + dt_bias)))` is evaluated while
//! staging, so the f32 [T, H, Dk] gate never reaches memory; the f32 state stays in registers.
const std = @import("std");
const mlx = @import("mlx.zig");
const status = @import("status.zig");

/// Value rows one threadgroup carries: 8 lanes each, at most two heads.
const MAX_ROWS: c_int = 128;

var kernel: ?mlx.mlx_fast_metal_kernel = null;
/// Whether this GPU launches the kernel at its thread count (probed once per geometry).
var launchable: ?bool = null;
var engaged = false;

const Plan = struct { ntg: c_int, rows: c_int };

/// Threadgroups (one per core) and value rows per threadgroup for `h * dv` rows, or null.
fn planFor(h: c_int, dv: c_int, cores: c_int) ?Plan {
    const total = h * dv;
    if (cores < 1 or total > cores * MAX_ROWS) return null;
    const ntg = @min(cores, @divTrunc(total, 2));
    if (ntg < 1) return null;
    const rows = 2 * @divTrunc(@divTrunc(total, 2) + ntg - 1, ntg);
    return if (rows <= MAX_ROWS) .{ .ntg = ntg, .rows = rows } else null;
}

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const ins = [_][*:0]const u8{ "q", "k", "v", "a", "beta", "A_log", "dt_bias", "lower_bound", "state_in", "T" };
    const outs = [_][*:0]const u8{ "y", "state_out" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("msv_kda_percore", iv, ov, @embedFile("kernels/kda_percore.metal"), "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

pub const Out = struct { y: mlx.mlx_array, state: mlx.mlx_array };

/// Whether `recur` serves this geometry: q/k/v heads alike, 128-wide, 16-bit activations,
/// an f32 state, and a GPU whose core count spreads the value rows at most 128 per core.
pub fn serves(s: mlx.mlx_stream, hk: c_int, hv: c_int, dk: c_int, dv: c_int, act: mlx.mlx_dtype, state: mlx.mlx_dtype) bool {
    if (!mlx.streamIsGpu(s) or hk != hv or dk != 128 or dv != 128) return false;
    if ((act != .bfloat16 and act != .float16) or state != .float32) return false;
    const plan = planFor(hv, dv, @intCast(status.gpuCoreCount())) orelse return false;
    if (launchable == null) launchable = probe(s, hv, act, plan) catch false;
    return launchable.?;
}

/// One tiny launch: Metal caps a kernel's threadgroup by its register use, and MLX only
/// reports that at evaluation.
fn probe(s: mlx.mlx_stream, h: c_int, act: mlx.mlx_dtype, plan: Plan) !bool {
    var arrs: [6]mlx.mlx_array = undefined;
    const shapes = [_][]const c_int{ &.{ 1, 1, h, 128 }, &.{ 1, 1, h }, &.{h}, &.{ h, 128 }, &.{ 1, h, 128, 128 }, &.{1} };
    const dts = [_]mlx.mlx_dtype{ act, act, .float32, .float32, .float32, .float32 };
    for (&arrs, shapes, dts) |*a, sh, dt| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(a, sh.ptr, sh.len, dt, s));
    }
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const r = try launch(s, arrs[0], arrs[0], arrs[0], arrs[0], arrs[1], arrs[2], arrs[3], arrs[5], arrs[4], plan);
    defer {
        _ = mlx.mlx_array_free(r.y);
        _ = mlx.mlx_array_free(r.state);
    }
    mlx.check(mlx.mlx_array_eval(r.state)) catch |e| {
        if (e == error.MlxError and mlx.takeErrorIf("Thread group size")) return false;
        return e;
    };
    return true;
}

fn launch(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, a: mlx.mlx_array, beta: mlx.mlx_array, a_log: mlx.mlx_array, dt_bias: mlx.mlx_array, lower_bound: mlx.mlx_array, state: mlx.mlx_array, plan: Plan) !Out {
    const qs = mlx.getShape(q);
    const b = qs[0];
    const t = qs[1];
    const h = qs[2];
    const dt = mlx.mlx_array_dtype(q);
    const t_arr = mlx.mlx_array_new_int(t);
    defer _ = mlx.mlx_array_free(t_arr);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, t, h, 128 }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, h, 128, 128 }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, plan.rows * 8 * plan.ntg, 1, b));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, plan.rows * 8, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt));
    inline for (.{ .{ "Dk", 128 }, .{ "Dv", 128 }, .{ "H", h }, .{ "NTG", plan.ntg }, .{ "MR", plan.rows } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const inputs = [_]mlx.mlx_array{ q, k, v, a, beta, a_log, dt_bias, lower_bound, state, t_arr };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try getKernel(), iv, cfg, s));
    var out: Out = .{ .y = mlx.mlx_array_new(), .state = mlx.mlx_array_new() };
    errdefer {
        _ = mlx.mlx_array_free(out.y);
        _ = mlx.mlx_array_free(out.state);
    }
    try mlx.check(mlx.mlx_vector_array_get(&out.y, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.state, ov, 1));
    return out;
}

/// The recurrence over a chunk: q, k [B, T, H, 128] normalized and scaled, v [B, T, H, 128],
/// `a` the gate pre-activation [B, T, H*128] in the activation dtype, `beta` sigmoid(b)
/// [B, T, H], `a_log` [H] (any float), `dt_bias` [H*128], f32 `state` [B, H, 128, 128].
/// Returns y [B, T, H, 128] in q's dtype and the new f32 state. Call only where `serves`.
pub fn recur(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, a: mlx.mlx_array, beta: mlx.mlx_array, a_log: mlx.mlx_array, dt_bias: mlx.mlx_array, lower_bound: f32, state: mlx.mlx_array) !Out {
    const h = mlx.getShape(q)[2];
    const plan = planFor(h, 128, @intCast(status.gpuCoreCount())).?;
    const dt = mlx.mlx_array_dtype(q);
    var a_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_c);
    try mlx.check(mlx.mlx_astype(&a_c, a, dt, s));
    var beta_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(beta_c);
    try mlx.check(mlx.mlx_astype(&beta_c, beta, dt, s));
    var alog = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(alog);
    try mlx.check(mlx.mlx_astype(&alog, a_log, .float32, s));
    var dtb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dtb);
    try mlx.check(mlx.mlx_astype(&dtb, dt_bias, .float32, s));
    const lbv = [1]f32{lower_bound};
    const lb = mlx.mlx_array_new_data(&lbv, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(lb);
    const out = try launch(s, q, k, v, a_c, beta_c, alog, dtb, lb, state, plan);
    if (!engaged) {
        engaged = true;
        @import("log.zig").info("[kda] per-core prefill recurrence engaged ({d} threadgroups x {d} rows)\n", .{ plan.ntg, plan.rows });
    }
    return out;
}

// ── decode ─────────────────────────────────────────────────────────────

/// oMLX's KDA decode step (`kernels/kda_decode.metal`): one threadgroup per head runs the
/// short conv, the low-rank gate projections, the q/k l2 norms, the gate, the delta rule and
/// the gated RMSNorm for T <= 8 tokens of one sequence.
var decode_kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };
/// Threads per threadgroup; a test raises it past any GPU's cap.
var decode_threads: c_int = 1024;
/// Whether this GPU launches the decode kernel at that count, per [gate slot][tokens][8-bit gates]:
/// Metal caps a kernel's threadgroup by its register use and MLX reports it only at evaluation.
var decode_launchable: [2][9][2]?bool = @splat(@splat(@splat(null)));

pub const Low = struct { w: mlx.mlx_array, s: mlx.mlx_array, b: mlx.mlx_array, bits: u32, gs: u32 };
/// `state_seq` f32 [T, H, 128, 128], the state after each row, only under `capture`.
pub const DecodeOut = struct { y: mlx.mlx_array, conv_state: mlx.mlx_array, state: mlx.mlx_array, state_seq: mlx.mlx_array = .{ .ctx = null } };

/// `proj` [1, T, W] is the row-joined input projection: q|k|v at 0, then the first gate stages
/// and b at `off_ga`, `off_fa`, `off_b`. `f_b`/`g_b` are the second gate stages ([H*Dk, Dk]),
/// `a_log` f32 [H], `dt_bias` f32 [H*Dk]. Null outside the shapes served.
pub fn decodeStep(s: mlx.mlx_stream, proj: mlx.mlx_array, off_ga: c_int, off_fa: c_int, off_b: c_int, heads: c_int, conv_state: mlx.mlx_array, conv_w: mlx.mlx_array, a_log: mlx.mlx_array, dt_bias: mlx.mlx_array, state: mlx.mlx_array, norm_w: mlx.mlx_array, f_b: Low, g_b: Low, lower_bound: f32, norm_eps: f32, capture: bool) !?DecodeOut {
    if (!mlx.streamIsGpu(s)) return null;
    const ps = mlx.getShape(proj);
    const dt = mlx.mlx_array_dtype(proj);
    if (ps.len != 3 or ps[0] != 1 or ps[1] < 1 or ps[1] > 8 or (dt != .bfloat16 and dt != .float16)) return null;
    const t = ps[1];
    const qkv = heads * 128;
    if (mlx.mlx_array_dtype(conv_w) != dt or mlx.mlx_array_dtype(norm_w) != dt or mlx.mlx_array_size(norm_w) != 128) return null;
    if (mlx.mlx_array_dtype(a_log) != .float32 or mlx.mlx_array_size(a_log) != @as(usize, @intCast(heads))) return null;
    if (mlx.mlx_array_dtype(dt_bias) != .float32 or mlx.mlx_array_size(dt_bias) != @as(usize, @intCast(qkv))) return null;
    if (!std.mem.eql(c_int, mlx.getShape(conv_state), &[_]c_int{ 1, 3, 3 * qkv }) or mlx.mlx_array_dtype(conv_state) != dt) return null;
    if (!std.mem.eql(c_int, mlx.getShape(state), &[_]c_int{ 1, heads, 128, 128 }) or mlx.mlx_array_dtype(state) != .float32) return null;
    if (f_b.bits != g_b.bits or f_b.gs != g_b.gs or (f_b.gs != 32 and f_b.gs != 64 and f_b.gs != 128)) return null;
    // 4/8 bits: MLX's qmv_quad rows; 5 bits: the one-row qmv.
    const gate5 = f_b.bits == 5;
    if (!(f_b.bits == 4 or f_b.bits == 8 or (gate5 and t == 1))) return null;
    if (f_b.s.ctx == null or f_b.b.ctx == null or g_b.s.ctx == null or g_b.b.ctx == null) return null;
    const ws = [_]c_int{ qkv, @intCast(128 * f_b.bits / 32) };
    if (!std.mem.eql(c_int, mlx.getShape(f_b.w), &ws) or !std.mem.eql(c_int, mlx.getShape(g_b.w), &ws)) return null;
    if (mlx.mlx_array_dtype(f_b.s) != dt or mlx.mlx_array_dtype(g_b.s) != dt) return null;

    const slot: usize = @intFromBool(gate5);
    const launchable_at = &decode_launchable[slot][@intCast(t)][@intFromBool(f_b.bits == 8)];
    if (launchable_at.* == false) return null;
    const kern = decode_kernels[slot] orelse blk: {
        const ins = [_][*:0]const u8{ "proj", "conv_w", "a_log", "dt_bias", "norm_w", "consts", "conv_state", "state_in", "fb_w", "fb_s", "fb_b", "gb_w", "gb_s", "gb_b" };
        const outs = [_][*:0]const u8{ "y", "conv_state_out", "state_out", "state_seq" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const body = @embedFile("kernels/kda_decode.metal");
        const source = if (gate5) "#define GATE5 1\n" ++ body else "#define GATE5 0\n" ++ body;
        const k = mlx.mlx_fast_metal_kernel_new(if (gate5) "msv_kda_decode_q5" else "msv_kda_decode", iv, ov, source, @embedFile("kernels/glm5_qmv_header.metal"), true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        decode_kernels[slot] = k;
        break :blk k;
    };
    const cv = [5]f32{ 1.0 / @sqrt(128.0), 1e-6, norm_eps, lower_bound, 1.0 / 128.0 };
    const consts = mlx.mlx_array_new_data(&cv, &[_]c_int{5}, 1, .float32);
    defer _ = mlx.mlx_array_free(consts);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, t, qkv }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 3, 3 * qkv }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, heads, 128, 128 }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, if (capture) &[_]c_int{ t, heads, 128, 128 } else &[_]c_int{ 1, 1, 1, 1 }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, decode_threads * heads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, decode_threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    inline for (.{ .{ "TOK", t }, .{ "DK", 128 }, .{ "QKV", qkv }, .{ "PROJ_W", ps[2] }, .{ "OFF_FA", off_fa }, .{ "OFF_GA", off_ga }, .{ "OFF_B", off_b }, .{ "BITS", @as(c_int, @intCast(f_b.bits)) }, .{ "GS", @as(c_int, @intCast(f_b.gs)) }, .{ "CAP", @as(c_int, @intFromBool(capture)) } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const inputs = [_]mlx.mlx_array{ proj, conv_w, a_log, dt_bias, norm_w, consts, conv_state, state, f_b.w, f_b.s, f_b.b, g_b.w, g_b.s, g_b.b };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out: DecodeOut = .{ .y = mlx.mlx_array_new(), .conv_state = mlx.mlx_array_new(), .state = mlx.mlx_array_new() };
    errdefer inline for (.{ out.y, out.conv_state, out.state }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    try mlx.check(mlx.mlx_vector_array_get(&out.y, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.conv_state, ov, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.state, ov, 2));
    if (capture) {
        out.state_seq = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&out.state_seq, ov, 3));
    }
    if (launchable_at.* == null) {
        mlx.check(mlx.mlx_array_eval(out.state)) catch |e| {
            if (e != error.MlxError or !mlx.takeErrorIf("Thread group size")) return e;
            launchable_at.* = false;
            inline for (.{ out.y, out.conv_state, out.state, out.state_seq }) |a| {
                _ = mlx.mlx_array_free(a);
            }
            @import("log.zig").info("[kda] decode step declined: this GPU launches fewer than {d} threads per threadgroup\n", .{decode_threads});
            return null;
        };
        launchable_at.* = true;
    }
    if (!decode_engaged) {
        decode_engaged = true;
        @import("log.zig").info("[kda] decode step engaged (T={d}, gate bits {d})\n", .{ t, f_b.bits });
    }
    return out;
}
var decode_engaged = false;

// ── tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "planFor spreads the value rows one threadgroup per core" {
    // GLM-5.3 (64 heads x 128) on 80 cores: 104 rows each; 40 cores would need 205.
    try testing.expectEqual(Plan{ .ntg = 80, .rows = 104 }, planFor(64, 128, 80).?);
    try testing.expect(planFor(64, 128, 40) == null);
    try testing.expect(planFor(64, 128, 0) == null);
}

test "kda per-core recurrence follows the bounded-gate delta rule" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const h: c_int = 64;
    const t: c_int = 29; // two full 12-token blocks and a tail
    if (!serves(s, h, h, 128, 128, .bfloat16, .float32)) return error.SkipZigTest;
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const rand = prng.random();
    const n_qk: usize = @intCast(t * h * 128);
    const bf = struct {
        fn round(x: f32) f32 {
            const u: u32 = @bitCast(x);
            return @bitCast((u +% 0x7fff +% ((u >> 16) & 1)) & 0xffff0000);
        }
    };
    const qv = try alloc.alloc(f32, n_qk);
    defer alloc.free(qv);
    const kv = try alloc.alloc(f32, n_qk);
    defer alloc.free(kv);
    const vv = try alloc.alloc(f32, n_qk);
    defer alloc.free(vv);
    const av = try alloc.alloc(f32, n_qk);
    defer alloc.free(av);
    const bv = try alloc.alloc(f32, @intCast(t * h));
    defer alloc.free(bv);
    const alog = try alloc.alloc(f32, @intCast(h));
    defer alloc.free(alog);
    const dtb = try alloc.alloc(f32, @intCast(h * 128));
    defer alloc.free(dtb);
    const st0 = try alloc.alloc(f32, @intCast(h * 128 * 128));
    defer alloc.free(st0);
    // q/k at unit norm like the normalized inputs; values bf16-exact so host and GPU agree on inputs.
    for (qv) |*x| x.* = bf.round(rand.floatNorm(f32) / 11.3);
    for (kv) |*x| x.* = bf.round(rand.floatNorm(f32) / 11.3);
    for (vv) |*x| x.* = bf.round(rand.floatNorm(f32));
    for (av) |*x| x.* = bf.round(rand.floatNorm(f32));
    for (bv) |*x| x.* = bf.round(rand.float(f32));
    for (alog) |*x| x.* = @log(1.0 + 15.0 * rand.float(f32));
    for (dtb) |*x| x.* = 0.1 * rand.floatNorm(f32);
    for (st0) |*x| x.* = 0.1 * rand.floatNorm(f32);
    const lb: f32 = -5.0;

    const mk = struct {
        fn arr(data: []const f32, shape: []const c_int, dt: mlx.mlx_dtype, st: mlx.mlx_stream) !mlx.mlx_array {
            const f = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f);
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&o, f, dt, st));
            return o;
        }
    };
    const q = try mk.arr(qv, &.{ 1, t, h, 128 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try mk.arr(kv, &.{ 1, t, h, 128 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try mk.arr(vv, &.{ 1, t, h, 128 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(v);
    const a = try mk.arr(av, &.{ 1, t, h * 128 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(a);
    const beta = try mk.arr(bv, &.{ 1, t, h }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(beta);
    const a_log = try mk.arr(alog, &.{h}, .float32, s);
    defer _ = mlx.mlx_array_free(a_log);
    const dt_bias = try mk.arr(dtb, &.{h * 128}, .float32, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    const state = try mk.arr(st0, &.{ 1, h, 128, 128 }, .float32, s);
    defer _ = mlx.mlx_array_free(state);
    const out = try recur(s, q, k, v, a, beta, a_log, dt_bias, lb, state);
    defer {
        _ = mlx.mlx_array_free(out.y);
        _ = mlx.mlx_array_free(out.state);
    }
    var y32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y32);
    try mlx.check(mlx.mlx_astype(&y32, out.y, .float32, s));
    try mlx.check(mlx.mlx_array_eval(y32));
    try mlx.check(mlx.mlx_array_eval(out.state));
    const yg = mlx.mlx_array_data_float32(y32).?;
    const sg = mlx.mlx_array_data_float32(out.state).?;

    // Host reference in f64, per head.
    var max_y: f64 = 0;
    var max_s: f64 = 0;
    const S = try alloc.alloc(f64, 128 * 128);
    defer alloc.free(S);
    for (0..@intCast(h)) |hh| {
        for (S, st0[hh * 128 * 128 ..][0 .. 128 * 128]) |*d, x| d.* = x;
        for (0..@intCast(t)) |tt| {
            const base = (tt * @as(usize, @intCast(h)) + hh) * 128;
            var g: [128]f64 = undefined;
            for (0..128) |c| {
                const x = @exp(@as(f64, alog[hh])) * (@as(f64, av[base + c]) + dtb[hh * 128 + c]);
                g[c] = @exp(lb * (1.0 / (1.0 + @exp(-x))));
            }
            const bt: f64 = bv[tt * @as(usize, @intCast(h)) + hh];
            for (0..128) |r| {
                const row = S[r * 128 ..][0..128];
                var p: f64 = 0;
                for (0..128) |c| {
                    row[c] *= g[c];
                    p += row[c] * kv[base + c];
                }
                const delta = (vv[base + r] - p) * bt;
                var o: f64 = 0;
                for (0..128) |c| {
                    row[c] += kv[base + c] * delta;
                    o += row[c] * qv[base + c];
                }
                max_y = @max(max_y, @abs(o - yg[base + r]));
            }
        }
        for (S, sg[hh * 128 * 128 ..][0 .. 128 * 128]) |want, got| max_s = @max(max_s, @abs(want - got));
    }
    // y leaves in bf16 (|y| < ~4: half an ulp is <= 2^-7); the f32 state tracks f64 closely.
    try testing.expect(max_y < 0.02);
    try testing.expect(max_s < 1e-4);
}

test "kda decode step over T rows equals T one-row steps, bit for bit, at the gate widths a verify window serves" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    const heads: c_int = 64;
    const qkv: c_int = heads * 128;
    const rows: c_int = 4;
    const off_ga: c_int = 3 * qkv;
    const off_fa = off_ga + 128;
    const off_b = off_fa + 128;
    const w_cols = off_b + heads;
    var prng = std.Random.DefaultPrng.init(11);
    const rand = prng.random();
    const alloc = testing.allocator;
    const mk = struct {
        fn f32s(a: std.mem.Allocator, r: std.Random, shape: []const c_int, scale: f32, dt: mlx.mlx_dtype, st: mlx.mlx_stream) !mlx.mlx_array {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const buf = try a.alloc(f32, n);
            defer a.free(buf);
            for (buf) |*x| x.* = scale * r.floatNorm(f32);
            const f = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f);
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&o, f, dt, st));
            return o;
        }
        fn words(a: std.mem.Allocator, r: std.Random, rows_: c_int, cols: c_int) !mlx.mlx_array {
            const n: usize = @intCast(rows_ * cols);
            const buf = try a.alloc(u32, n);
            defer a.free(buf);
            for (buf) |*x| x.* = r.int(u32);
            return mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ rows_, cols }, 2, .uint32);
        }
        fn equal(a: mlx.mlx_array, b: mlx.mlx_array, st: mlx.mlx_stream) !bool {
            var e = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(e);
            try mlx.check(mlx.mlx_array_equal(&e, a, b, false, st));
            try mlx.check(mlx.mlx_array_eval(e));
            var same = false;
            try mlx.check(mlx.mlx_array_item_bool(&same, e));
            return same;
        }
    };
    const proj = try mk.f32s(alloc, rand, &.{ 1, rows, w_cols }, 1.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(proj);
    const conv_w = try mk.f32s(alloc, rand, &.{ 3 * qkv, 4 }, 0.5, .bfloat16, s);
    defer _ = mlx.mlx_array_free(conv_w);
    const conv0 = try mk.f32s(alloc, rand, &.{ 1, 3, 3 * qkv }, 1.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(conv0);
    const alog_host = try alloc.alloc(f32, @intCast(heads));
    defer alloc.free(alog_host);
    for (alog_host) |*x| x.* = @log(1.0 + 15.0 * rand.float(f32));
    const a_log = mlx.mlx_array_new_data(alog_host.ptr, &[_]c_int{heads}, 1, .float32);
    defer _ = mlx.mlx_array_free(a_log);
    const dt_bias = try mk.f32s(alloc, rand, &.{qkv}, 0.1, .float32, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    const state0 = try mk.f32s(alloc, rand, &.{ 1, heads, 128, 128 }, 0.1, .float32, s);
    defer _ = mlx.mlx_array_free(state0);
    const norm_w = try mk.f32s(alloc, rand, &.{128}, 1.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(norm_w);

    for ([_]u32{ 4, 8 }) |bits| {
        const cols: c_int = @intCast(128 * bits / 32);
        const fw = try mk.words(alloc, rand, qkv, cols);
        defer _ = mlx.mlx_array_free(fw);
        const gw = try mk.words(alloc, rand, qkv, cols);
        defer _ = mlx.mlx_array_free(gw);
        const fs = try mk.f32s(alloc, rand, &.{ qkv, 2 }, 0.02, .bfloat16, s);
        defer _ = mlx.mlx_array_free(fs);
        const fbias = try mk.f32s(alloc, rand, &.{ qkv, 2 }, 0.02, .bfloat16, s);
        defer _ = mlx.mlx_array_free(fbias);
        const gs_ = try mk.f32s(alloc, rand, &.{ qkv, 2 }, 0.02, .bfloat16, s);
        defer _ = mlx.mlx_array_free(gs_);
        const gbias = try mk.f32s(alloc, rand, &.{ qkv, 2 }, 0.02, .bfloat16, s);
        defer _ = mlx.mlx_array_free(gbias);
        const f_b: Low = .{ .w = fw, .s = fs, .b = fbias, .bits = bits, .gs = 64 };
        const g_b: Low = .{ .w = gw, .s = gs_, .b = gbias, .bits = bits, .gs = 64 };

        // A threadgroup past the GPU's cap declines and leaves no latched error behind.
        const launchable_at = &decode_launchable[0][@intCast(rows)][@intFromBool(bits == 8)];
        {
            decode_threads = 2048;
            defer decode_threads = 1024;
            const too_wide = try decodeStep(s, proj, off_ga, off_fa, off_b, heads, conv0, conv_w, a_log, dt_bias, state0, norm_w, f_b, g_b, -5.0, 1e-6, true);
            try testing.expect(too_wide == null and launchable_at.* == false and !mlx.errorPending());
        }
        launchable_at.* = null;

        const multi = (try decodeStep(s, proj, off_ga, off_fa, off_b, heads, conv0, conv_w, a_log, dt_bias, state0, norm_w, f_b, g_b, -5.0, 1e-6, true)) orelse {
            if (launchable_at.* == false) return error.SkipZigTest; // a GPU that cannot launch the kernel
            std.debug.print("decodeStep declined {d} rows at {d}-bit gates\n", .{ rows, bits });
            return error.Declined;
        };
        defer {
            _ = mlx.mlx_array_free(multi.y);
            _ = mlx.mlx_array_free(multi.conv_state);
            _ = mlx.mlx_array_free(multi.state);
            _ = mlx.mlx_array_free(multi.state_seq);
        }
        var conv = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&conv, conv0);
        var st = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&st, state0);
        defer _ = mlx.mlx_array_free(conv);
        defer _ = mlx.mlx_array_free(st);
        for (0..@intCast(rows)) |t| {
            const ti: c_int = @intCast(t);
            var p1 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(p1);
            try mlx.check(mlx.mlx_slice(&p1, proj, &[_]c_int{ 0, ti, 0 }, 3, &[_]c_int{ 1, ti + 1, w_cols }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            const one = (try decodeStep(s, p1, off_ga, off_fa, off_b, heads, conv, conv_w, a_log, dt_bias, st, norm_w, f_b, g_b, -5.0, 1e-6, false)) orelse return error.Declined;
            defer _ = mlx.mlx_array_free(one.y);
            _ = mlx.mlx_array_free(conv);
            conv = one.conv_state;
            _ = mlx.mlx_array_free(st);
            st = one.state;
            var y_row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(y_row);
            try mlx.check(mlx.mlx_slice(&y_row, multi.y, &[_]c_int{ 0, ti, 0 }, 3, &[_]c_int{ 1, ti + 1, qkv }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            var seq_row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(seq_row);
            try mlx.check(mlx.mlx_slice(&seq_row, multi.state_seq, &[_]c_int{ ti, 0, 0, 0 }, 4, &[_]c_int{ ti + 1, heads, 128, 128 }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
            var st4 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(st4);
            try mlx.check(mlx.mlx_reshape(&st4, st, &[_]c_int{ 1, heads, 128, 128 }, 4, s));
            if (!try mk.equal(y_row, one.y, s)) std.debug.print("{d}-bit gates: row {d} output differs\n", .{ bits, t });
            try testing.expect(try mk.equal(y_row, one.y, s));
            if (!try mk.equal(seq_row, st4, s)) std.debug.print("{d}-bit gates: row {d} captured state differs\n", .{ bits, t });
            try testing.expect(try mk.equal(seq_row, st4, s));
        }
        try testing.expect(try mk.equal(multi.conv_state, conv, s));
        try testing.expect(try mk.equal(multi.state, st, s));
    }
}
