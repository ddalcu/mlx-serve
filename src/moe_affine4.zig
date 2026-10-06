//! MoE decode over 4-bit AFFINE expert banks (Qwen3.8-Flash-Next's routed experts), read in
//! place: one dispatch for gate+up+SwiGLU across the top-k experts, one for down and the
//! score-weighted sum. The shape is `moe_fp4`'s: a simdgroup owns ROWS output rows and keeps
//! its 16 values of x in registers across them, loads are 8 bytes a lane, no threadgroup
//! memory, and the down kernel accumulates every expert of a token in registers. The affine
//! bias rides a per-group sum of x: `sum((s*q + b) * x) = s * dot(q, x) + b * sum(x)`.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

/// One output row a simdgroup: more rows share the x registers but their live state cuts occupancy.
const ROWS: c_int = 1;
const SGS: c_int = 2;

const DOT16 =
    \\inline float mlxserve_a4_dot16(uint2 w, thread const float* x) {
    \\  float a = 0.0f, b = 0.0f;
    \\  for (int j = 0; j < 8; ++j) {
    \\    a += x[j] * float((w.x >> (4 * j)) & 0xFu);
    \\    b += x[8 + j] * float((w.y >> (4 * j)) & 0xFu);
    \\  }
    \\  return a + b;
    \\}
;

// grid (32, N/ROWS, TOPK), threadgroup (32, SGS, 1): one token, slot e = its e-th expert.
// GS >= 16 and 16 | GS, so a lane's 16 values share one scale and bias.
const GATEUP_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int row0 = int(thread_position_in_grid.y) * ROWS;
    \\const uint e = thread_position_in_grid.z;
    \\constexpr int KW2 = K / 16;
    \\constexpr int KG = K / GS;
    \\const size_t row = size_t(inds[e]) * N + row0;
    \\const device uint2* gw = (const device uint2*)wg_q + row * KW2 + lane;
    \\const device uint2* uw = (const device uint2*)wu_q + row * KW2 + lane;
    \\float ag[ROWS] = {0.0f};
    \\float au[ROWS] = {0.0f};
    \\for (int k = 0; k < K; k += 512) {
    \\  const int kk = k + int(lane) * 16;
    \\  if (kk < K) {
    \\    float xr[16];
    \\    float xs = 0.0f;
    \\    for (int i = 0; i < 16; ++i) { xr[i] = float(x[kk + i]); xs += xr[i]; }
    \\    const int gi = kk / GS;
    \\    for (int r = 0; r < ROWS; ++r) {
    \\      const size_t sg = (row + r) * KG + gi;
    \\      ag[r] += float(g_scales[sg]) * mlxserve_a4_dot16(gw[r * KW2 + k / 16], xr) + float(g_biases[sg]) * xs;
    \\      au[r] += float(u_scales[sg]) * mlxserve_a4_dot16(uw[r * KW2 + k / 16], xr) + float(u_biases[sg]) * xs;
    \\    }
    \\  }
    \\}
    \\for (int r = 0; r < ROWS; ++r) {
    \\  const float g = simd_sum(ag[r]);
    \\  const float u = simd_sum(au[r]);
    \\  if (lane == 0) {
    \\    const T gt = T(g);
    \\    y[size_t(e) * N + row0 + r] = (gt * sigtab[as_type<ushort>(gt)]) * T(u);
    \\  }
    \\}
;

// grid (32, H/ROWS, 1), threadgroup (32, SGS, 1): every expert's down row, weighted by its
// routing score, accumulated in the same registers. SHARED: the gated shared expert's output
// (`sdown`, gate logit `glog`) is added in the epilogue.
const DOWNRED_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int row0 = int(thread_position_in_grid.y) * ROWS;
    \\constexpr int KW2 = I / 16;
    \\constexpr int KG = I / GS;
    \\float acc[ROWS] = {0.0f};
    \\for (int e = 0; e < TOPK; ++e) {
    \\  const size_t row = size_t(inds[e]) * H + row0;
    \\  const device uint2* dw = (const device uint2*)wd_q + row * KW2 + lane;
    \\  const float sc = float(scores[e]);
    \\  for (int k = 0; k < I; k += 512) {
    \\    const int kk = k + int(lane) * 16;
    \\    if (kk < I) {
    \\      float xr[16];
    \\      float xs = 0.0f;
    \\      for (int i = 0; i < 16; ++i) { xr[i] = float(act[size_t(e) * I + kk + i]); xs += xr[i]; }
    \\      const int gi = kk / GS;
    \\      for (int r = 0; r < ROWS; ++r) {
    \\        const size_t sg = (row + r) * KG + gi;
    \\        acc[r] += sc * (float(d_scales[sg]) * mlxserve_a4_dot16(dw[r * KW2 + k / 16], xr) + float(d_biases[sg]) * xs);
    \\      }
    \\    }
    \\  }
    \\}
    \\for (int r = 0; r < ROWS; ++r) {
    \\  const float v = simd_sum(acc[r]);
    \\  if (lane == 0) {
    \\    T out = T(v);
    \\    if (SHARED != 0) {
    \\      // The gated shared expert joins as the chain does it: sigmoid in T (MLX's Sigmoid op), the
    \\      // product rounded to T, the sum rounded to T.
    \\      const T gl = glog[0];
    \\      auto sy = 1 / (1 + metal::precise::exp(metal::abs(gl)));
    \\      const T sig = (gl < 0) ? sy : 1 - sy;
    \\      const T gated = T(float(sig) * float(sdown[row0 + r]));
    \\      out = T(float(out) + float(gated));
    \\    }
    \\    y[row0 + r] = out;
    \\  }
    \\}
;

var gateup_kernel: ?mlx.mlx_fast_metal_kernel = null;
var downred_kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;
var env_enabled: ?bool = null;
pub var override: ?bool = null;

/// `MLX_SERVE_MOE_AFFINE4=0` keeps the per-slot gather kernels.
fn enabled() bool {
    if (override) |v| return v;
    if (env_enabled) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_MOE_AFFINE4");
    env_enabled = raw == null or raw.?[0] != '0';
    return env_enabled.?;
}

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const outs = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, DOT16, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn apply(kernel: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, out_shape: []const c_int, dt: mlx.mlx_dtype, grid: [3]c_int, tmpl: []const struct { [*:0]const u8, c_int }, s: mlx.mlx_stream) !mlx.mlx_array {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, out_shape.ptr, out_shape.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, SGS, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ROWS", ROWS));
    for (tmpl) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], t[1]));
    const v = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel, v, c, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    return y;
}

pub const Bank = struct { w: mlx.mlx_array, s: mlx.mlx_array, b: mlx.mlx_array };

/// One-expert views of a layer's banks, bound in place of them: MLX bills a command buffer for the
/// `data_size` of every array a kernel binds, and a whole bank (419 MB) forced a commit per kernel.
/// The kernels address experts from the buffer pointer, so the view only changes the billing.
pub const Views = struct { gate: Bank, up: Bank, down: Bank };

var views_enabled: ?bool = null;

/// `MLX_SERVE_MOE_BANK_VIEWS=0` binds the whole banks.
fn viewsOn() bool {
    if (views_enabled) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_MOE_BANK_VIEWS");
    views_enabled = raw == null or raw.?[0] != '0';
    return views_enabled.?;
}

fn oneExpert(a: mlx.mlx_array, owned: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(a);
    if (sh.len != 3) return a;
    var v = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_slice(&v, a, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    try owned.append(allocator, v);
    return v;
}

/// Views for banks that carry scales and biases (null otherwise); the caller keeps `owned` alive
/// as long as the banks.
pub fn bindViews(gate: Bank, up: Bank, down: Bank, owned: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !?Views {
    if (!viewsOn()) return null;
    inline for (.{ gate, up, down }) |bk| {
        if (bk.s.ctx == null or bk.b.ctx == null) return null;
    }
    var out: Views = undefined;
    inline for (.{ "gate", "up", "down" }, .{ gate, up, down }) |name, bk| {
        @field(out, name) = .{ .w = try oneExpert(bk.w, owned, allocator, s), .s = try oneExpert(bk.s, owned, allocator, s), .b = try oneExpert(bk.b, owned, allocator, s) };
    }
    return out;
}

/// The shared expert's down output [hidden] and its gate logit [1], both in x's dtype: the down
/// kernel adds `sigmoid(logit) * down` to the routed sum.
pub const Shared = struct { down: mlx.mlx_array, logit: mlx.mlx_array };

/// y [hidden] = sum_k scores[k] * down_k(silu(gate_k(x)) * up_k(x)) for ONE token. `x` [hidden]
/// bf16/f16, banks [E, out, in/8] u32 with scales and biases [E, out, in/GS] in x's dtype,
/// `inds` [TOPK] u32, `scores` [TOPK] in x's dtype, `sigtab` the SwiGLU table. Null outside
/// the kernels' set (caller keeps its path). `views` (of the same banks) are what the kernels bind.
pub fn decode(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Bank, up: Bank, down: Bank, views: ?Views, shared: ?Shared, inds: mlx.mlx_array, scores: mlx.mlx_array, sigtab: mlx.mlx_array, group_size: u32) !?mlx.mlx_array {
    if (!enabled() or !mlx.streamIsGpu(s)) return null;
    if (group_size < 16 or group_size % 16 != 0) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32 or mlx.mlx_array_dtype(scores) != dt) return null;
    inline for (.{ gate, up, down }) |bk| {
        if (bk.s.ctx == null or bk.b.ctx == null or mlx.mlx_array_dtype(bk.s) != dt or mlx.mlx_array_dtype(bk.b) != dt or mlx.mlx_array_dtype(bk.w) != .uint32) return null;
    }
    const gsh = mlx.getShape(gate.w);
    const dsh = mlx.getShape(down.w);
    if (gsh.len != 3 or dsh.len != 3 or !std.mem.eql(c_int, gsh, mlx.getShape(up.w))) return null;
    const inter = gsh[1];
    const hidden = gsh[2] * 8;
    const gs: c_int = @intCast(group_size);
    if (dsh[1] != hidden or dsh[2] * 8 != inter) return null;
    if (@rem(hidden, gs) != 0 or @rem(inter, gs) != 0 or @rem(inter, ROWS * SGS) != 0 or @rem(hidden, ROWS * SGS) != 0) return null;
    if (@rem(hidden, 16) != 0 or @rem(inter, 16) != 0) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len != 1 or xsh[0] != hidden) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 1 or !std.mem.eql(c_int, ish, mlx.getShape(scores))) return null;
    const topk: c_int = ish[0];
    if (shared) |sh| {
        if (mlx.mlx_array_dtype(sh.down) != dt or mlx.mlx_array_dtype(sh.logit) != dt or mlx.mlx_array_size(sh.logit) != 1) return null;
        if (mlx.mlx_array_size(sh.down) != @as(usize, @intCast(hidden))) return null;
    }

    if (gateup_kernel == null) {
        const ins = [_][*:0]const u8{ "x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases", "inds", "sigtab" };
        gateup_kernel = try makeKernel("mlxserve_moe_a4_gateup", &ins, GATEUP_SOURCE);
    }
    if (downred_kernel == null) {
        const ins = [_][*:0]const u8{ "act", "wd_q", "d_scales", "d_biases", "inds", "scores", "sdown", "glog" };
        downred_kernel = try makeKernel("mlxserve_moe_a4_downred", &ins, DOWNRED_SOURCE);
    }
    const bound = views orelse Views{ .gate = gate, .up = up, .down = down };
    const act = try apply(gateup_kernel.?, &.{ x, bound.gate.w, bound.gate.s, bound.gate.b, bound.up.w, bound.up.s, bound.up.b, inds, sigtab }, &.{ topk, inter }, dt, .{ 32, @divExact(inter, ROWS), topk }, &.{ .{ "K", hidden }, .{ "N", inter }, .{ "GS", gs } }, s);
    defer _ = mlx.mlx_array_free(act);
    // Without a shared expert the two operands are never read; any array of the right type binds.
    const sh = shared orelse Shared{ .down = scores, .logit = scores };
    const y = try apply(downred_kernel.?, &.{ act, bound.down.w, bound.down.s, bound.down.b, inds, scores, sh.down, sh.logit }, &.{hidden}, dt, .{ 32, @divExact(hidden, ROWS), 1 }, &.{ .{ "I", inter }, .{ "H", hidden }, .{ "GS", gs }, .{ "TOPK", topk }, .{ "SHARED", @intFromBool(shared != null) } }, s);
    if (!engaged) {
        engaged = true;
        log.info("[moe] affine-4 decode kernels engaged: topk={d} inter={d} hidden={d} gs={d} (MLX_SERVE_MOE_AFFINE4=0 restores the per-slot gather kernels)\n", .{ topk, inter, hidden, gs });
    }
    return y;
}

const testing = std.testing;

fn randBf16(rnd: std.Random, shape: []const c_int, scale: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(buf);
    for (buf) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0 * scale;
    const f = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

fn quant(w: mlx.mlx_array, s: mlx.mlx_stream) !struct { b: Bank, deq: mlx.mlx_array } {
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, w, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var b: Bank = .{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new(), .b = mlx.mlx_array_new() };
    try mlx.check(mlx.mlx_vector_array_get(&b.w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b.s, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&b.b, triple, 2));
    var deq = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_dequantize(&deq, b.w, b.s, b.b, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, .{ .value = .float32, .has_value = true }, s));
    return .{ .b = b, .deq = deq };
}

fn freeBank(b: Bank) void {
    _ = mlx.mlx_array_free(b.w);
    _ = mlx.mlx_array_free(b.s);
    _ = mlx.mlx_array_free(b.b);
}

fn readF32(a: mlx.mlx_array, s: mlx.mlx_stream) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, a, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const n = mlx.mlx_array_size(f);
    const out = try testing.allocator.alloc(f32, n);
    @memcpy(out, (mlx.mlx_array_data_float32(f) orelse return error.Unreadable)[0..n]);
    return out;
}

test "moe affine-4 decode: no worse than the per-slot gather kernels against the f32 truth (Flash Next geometry, K not a whole block)" {
    const s = mlx.gpuStream();
    const xfm = @import("transformer.zig");
    var prng = std.Random.DefaultPrng.init(0xA4E0);
    const rnd = prng.random();
    override = true;
    defer override = null;
    xfm.gqmv_gateup_override = true;
    defer xfm.gqmv_gateup_override = null;
    xfm.downred_override = true;
    defer xfm.downred_override = null;

    const E: c_int = 16;
    const H: c_int = 2560; // hidden: 5 whole blocks
    const I: c_int = 640; // intermediate: 1.25 blocks, the predicated tail
    const TOPK: c_int = 10;
    const gw = try randBf16(rnd, &.{ E, I, H }, 0.05, s);
    defer _ = mlx.mlx_array_free(gw);
    const uw = try randBf16(rnd, &.{ E, I, H }, 0.05, s);
    defer _ = mlx.mlx_array_free(uw);
    const dw = try randBf16(rnd, &.{ E, H, I }, 0.05, s);
    defer _ = mlx.mlx_array_free(dw);
    const g = try quant(gw, s);
    defer {
        freeBank(g.b);
        _ = mlx.mlx_array_free(g.deq);
    }
    const u = try quant(uw, s);
    defer {
        freeBank(u.b);
        _ = mlx.mlx_array_free(u.deq);
    }
    const d = try quant(dw, s);
    defer {
        freeBank(d.b);
        _ = mlx.mlx_array_free(d.deq);
    }
    const x = try randBf16(rnd, &.{H}, 1.0, s);
    defer _ = mlx.mlx_array_free(x);
    var idx: [10]u32 = .{ 3, 15, 0, 7, 9, 1, 12, 5, 14, 2 };
    const inds = mlx.mlx_array_new_data(&idx, &[_]c_int{TOPK}, 1, .uint32);
    defer _ = mlx.mlx_array_free(inds);
    const scf = [_]f32{ 0.3, 0.2, 0.1, 0.1, 0.08, 0.07, 0.05, 0.05, 0.03, 0.02 };
    const sc32 = mlx.mlx_array_new_data(&scf, &[_]c_int{TOPK}, 1, .float32);
    defer _ = mlx.mlx_array_free(sc32);
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_astype(&scores, sc32, .bfloat16, s));
    const sigtab = try xfm.swigluSigTable(s, .bfloat16, std.heap.c_allocator);

    const ours = (try decode(s, x, g.b, u.b, d.b, null, null, inds, scores, sigtab, 64)) orelse return error.KernelDeclined;
    defer _ = mlx.mlx_array_free(ours);
    // The shipped per-slot kernels on the same inputs.
    const act = (try xfm.gatherQmvGateUp(s, x, g.b.w, g.b.s, g.b.b, u.b.w, u.b.s, u.b.b, inds, 4, 64, .affine, 0)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(act);
    const stock = (try xfm.gatherQmvDownReduce(s, act, d.b.w, d.b.s, d.b.b, inds, scores, 4, 64, .affine)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(stock);

    // f32 truth over the dequantized experts.
    const xh = try readF32(x, s);
    defer testing.allocator.free(xh);
    const gd = try readF32(g.deq, s);
    defer testing.allocator.free(gd);
    const ud = try readF32(u.deq, s);
    defer testing.allocator.free(ud);
    const dd = try readF32(d.deq, s);
    defer testing.allocator.free(dd);
    const truth = try testing.allocator.alloc(f64, @intCast(H));
    defer testing.allocator.free(truth);
    @memset(truth, 0);
    const act_e = try testing.allocator.alloc(f64, @intCast(I));
    defer testing.allocator.free(act_e);
    for (idx, scf) |e, w| {
        for (0..@intCast(I)) |n| {
            var gs: f64 = 0;
            var us: f64 = 0;
            for (0..@intCast(H)) |k| {
                const base = (e * @as(usize, @intCast(I)) + n) * @as(usize, @intCast(H)) + k;
                gs += @as(f64, gd[base]) * xh[k];
                us += @as(f64, ud[base]) * xh[k];
            }
            act_e[n] = gs / (1.0 + @exp(-gs)) * us;
        }
        for (0..@intCast(H)) |h| {
            var acc: f64 = 0;
            for (0..@intCast(I)) |n| acc += @as(f64, dd[(e * @as(usize, @intCast(H)) + h) * @as(usize, @intCast(I)) + n]) * act_e[n];
            truth[h] += w * acc;
        }
    }
    const ho = try readF32(ours, s);
    defer testing.allocator.free(ho);
    const hs = try readF32(stock, s);
    defer testing.allocator.free(hs);
    var se_o: f64 = 0;
    var se_s: f64 = 0;
    for (ho, hs, truth) |o, st, tr| {
        try testing.expect(std.math.isFinite(o));
        se_o += (o - tr) * (o - tr);
        se_s += (st - tr) * (st - tr);
    }
    const rms_o = @sqrt(se_o / @as(f64, @floatFromInt(H)));
    const rms_s = @sqrt(se_s / @as(f64, @floatFromInt(H)));
    testing.expect(rms_o <= rms_s * 1.15 + 1e-6) catch |e| {
        std.debug.print("\n[moe-a4] ours_rms={e:.4} per-slot_rms={e:.4}\n", .{ rms_o, rms_s });
        return e;
    };

    // The shared-expert epilogue is the chain's own rounding: sigmoid, product and sum each in bf16.
    const sd = try randBf16(rnd, &.{H}, 1.0, s);
    defer _ = mlx.mlx_array_free(sd);
    const logit = try randBf16(rnd, &.{1}, 4.0, s);
    defer _ = mlx.mlx_array_free(logit);
    const joined = (try decode(s, x, g.b, u.b, d.b, null, .{ .down = sd, .logit = logit }, inds, scores, sigtab, 64)) orelse return error.KernelDeclined;
    defer _ = mlx.mlx_array_free(joined);
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, logit, s));
    var gated = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gated);
    try mlx.check(mlx.mlx_multiply(&gated, sig, sd, s));
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_add(&want, ours, gated, s));
    const hj = try readF32(joined, s);
    defer testing.allocator.free(hj);
    const hw = try readF32(want, s);
    defer testing.allocator.free(hw);
    try testing.expectEqualSlices(f32, hw, hj);

    // The views production binds: one expert wide, the banks' own buffers, the same bits out.
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (owned.items) |a| _ = mlx.mlx_array_free(a);
        owned.deinit(testing.allocator);
    }
    const views = (try bindViews(g.b, u.b, d.b, &owned, testing.allocator, s)) orelse return error.NoViews;
    for (owned.items) |a| try mlx.check(mlx.mlx_array_eval(a));
    inline for (.{ .{ views.gate, g.b }, .{ views.up, u.b }, .{ views.down, d.b } }) |pair| {
        inline for (.{ "w", "s", "b" }) |f| {
            const v = @field(pair[0], f);
            const bank = @field(pair[1], f);
            try testing.expectEqual(mlx.mlx_array_size(bank) / @as(usize, @intCast(E)), mlx.mlx_array_size(v));
            try testing.expectEqual(@intFromPtr(mlx.mlx_array_data_uint32(bank)), @intFromPtr(mlx.mlx_array_data_uint32(v)));
        }
    }
    const viewed = (try decode(s, x, g.b, u.b, d.b, views, null, inds, scores, sigtab, 64)) orelse return error.KernelDeclined;
    defer _ = mlx.mlx_array_free(viewed);
    const hv = try readF32(viewed, s);
    defer testing.allocator.free(hv);
    try testing.expectEqualSlices(f32, ho, hv);
}
