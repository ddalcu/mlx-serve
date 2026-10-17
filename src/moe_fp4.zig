//! MoE decode over fp4 expert banks (MXFP4 / NVFP4), read in place: one
//! dispatch for gate+up+SwiGLU across the top-k experts, one for down and the
//! score-weighted sum. Each simdgroup owns ROWS output rows and keeps its 16
//! values of x in registers across them (MLX's `fp_qmv_fast` shape), so x is
//! read once per row group rather than once per row. ModelOpt NVFP4 banks add a
//! per-expert gate scale (before the SiLU); their up x down scales ride the scores.
const std = @import("std");
const mlx = @import("mlx.zig");
const xfm = @import("transformer.zig");
const moe_affine4 = @import("moe_affine4.zig");

const ROWS: c_int = 4;
const SGS: c_int = 2;
/// Widest token count worth sending here: a row re-reads its own experts, so
/// past two rows MLX's sorted gather (one read per expert per row group) wins.
pub const MAX_ROWS: c_int = 2;

/// e2m1 nibble -> its value times 2^-14, by bit placement into a half.
pub const E2M1_HEADER =
    \\inline float mlxserve_e2m1(uint c) {
    \\  return float(as_type<half>(ushort(((c & 0x7u) << 9) | ((c & 0x8u) << 12))));
    \\}
    \\
;
/// Group scale -> its value times 2^-8: e4m3 (NVFP4) by bit placement into a
/// half, e8m0 (MXFP4) as MLX's `fp8_e8m0` reads it (the byte is a float
/// exponent, 0 is 2^-127). With the e2m1 2^-14, one 2^22 multiply restores both.
const NV_SCALE =
    \\inline float mlxserve_fp4_scale(uint b) {
    \\  return float(as_type<half>(ushort(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8))));
    \\}
    \\
;
const MX_SCALE =
    \\inline float mlxserve_fp4_scale(uint b) {
    \\  return as_type<float>(b == 0u ? 0x400000u : (b << 23)) * 0.00390625f;
    \\}
    \\
;

/// Nibbles j and j+4 of a word decode together: one shift puts them in the two halves of a
/// half2, then the e2m1 bit placement serves both lanes.
pub const E2M1X2_HEADER =
    \\inline float2 mlxserve_e2m1x2(uint w, int j) {
    \\  const uint v = w >> (4 * j);
    \\  return float2(as_type<half2>(((v & 0x70007u) << 9) | ((v & 0x80008u) << 12)));
    \\}
    \\
;

const DOT16 = E2M1X2_HEADER ++
    \\inline float mlxserve_fp4_dot16(uint2 w, thread const float* x) {
    \\  float a = 0.0f, b = 0.0f;
    \\  for (int j = 0; j < 4; ++j) {
    \\    const float2 p = mlxserve_e2m1x2(w.x, j);
    \\    const float2 q = mlxserve_e2m1x2(w.y, j);
    \\    a += x[j] * p.x + x[j + 4] * p.y;
    \\    b += x[8 + j] * q.x + x[12 + j] * q.y;
    \\  }
    \\  return a + b;
    \\}
;

// grid (32, N/ROWS, R*TOPK), threadgroup (32, SGS, 1): slot e is token e / TOPK's
// e % TOPK-th expert. USCALE and CLAMP are the per-expert up scale and GLM's whole-number SwiGLU clamp. The e2m1 and scale decodes each scale by a power of two;
// 2^22 folds both back once per row.
const GATEUP_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int row0 = int(thread_position_in_grid.y) * ROWS;
    \\const uint e = thread_position_in_grid.z;
    \\const device auto* xt = x + size_t(e / TOPK) * K;
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
    \\    for (int i = 0; i < 16; ++i) xr[i] = float(xt[kk + i]);
    \\    const int gi = kk / GS;
    \\    for (int r = 0; r < ROWS; ++r) {
    \\      const size_t sg = (row + r) * KG + gi;
    \\      ag[r] += mlxserve_fp4_dot16(gw[r * KW2 + k / 16], xr) * mlxserve_fp4_scale(uint(g_scales[sg]));
    \\      au[r] += mlxserve_fp4_dot16(uw[r * KW2 + k / 16], xr) * mlxserve_fp4_scale(uint(u_scales[sg]));
    \\    }
    \\  }
    \\}
    \\for (int r = 0; r < ROWS; ++r) {
    \\  const float g = simd_sum(ag[r]) * 4194304.0f;
    \\  const float u = simd_sum(au[r]) * 4194304.0f;
    \\  if (lane == 0) {
    \\    T gt = GSCALE != 0 ? T(T(g) * gsc[inds[e]]) : T(g);
    \\    T ut = USCALE != 0 ? T(T(u) * usc[inds[e]]) : T(u);
    \\    if (CLAMP != 0) {
    \\      const float hi = float(T(float(CLAMP)));
    \\      gt = T(min(float(gt), hi));
    \\      ut = T(clamp(float(ut), -hi, hi));
    \\    }
    \\    y[size_t(e) * N + row0 + r] = (gt * sigtab[as_type<ushort>(gt)]) * ut;
    \\  }
    \\}
;

// grid (32, H/ROWS, R), threadgroup (32, SGS, 1): every expert's down row for
// token tok, weighted by its routing score, accumulated in the same registers.
const DOWNRED_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int row0 = int(thread_position_in_grid.y) * ROWS;
    \\const int tok = int(thread_position_in_grid.z);
    \\constexpr int KW2 = I / 16;
    \\constexpr int KG = I / GS;
    \\float acc[ROWS] = {0.0f};
    \\for (int e = tok * TOPK; e < (tok + 1) * TOPK; ++e) {
    \\  const size_t row = size_t(inds[e]) * H + row0;
    \\  const device uint2* dw = (const device uint2*)wd_q + row * KW2 + lane;
    \\  const float sc = float(scores[e]);
    \\  for (int k = 0; k < I; k += 512) {
    \\    const int kk = k + int(lane) * 16;
    \\    if (kk < I) {
    \\      float xr[16];
    \\      for (int i = 0; i < 16; ++i) xr[i] = float(act[size_t(e) * I + kk + i]);
    \\      const int gi = kk / GS;
    \\      for (int r = 0; r < ROWS; ++r)
    \\        acc[r] += sc * mlxserve_fp4_dot16(dw[r * KW2 + k / 16], xr) * mlxserve_fp4_scale(uint(d_scales[(row + r) * KG + gi]));
    \\    }
    \\  }
    \\}
    \\for (int r = 0; r < ROWS; ++r) {
    \\  const float v = simd_sum(acc[r]) * 4194304.0f;
    \\  if (lane == 0) {
    \\    T out = T(v);
    \\    if (SHARED != 0) {
    \\      // The gated shared expert joins as `moe_affine4` adds it: sigmoid in T, product and sum rounded to T.
    \\      const T gl = glog[0];
    \\      auto sy = 1 / (1 + metal::precise::exp(metal::abs(gl)));
    \\      const T sig = (gl < 0) ? sy : 1 - sy;
    \\      out = T(float(out) + float(T(float(sig) * float(sdown[row0 + r]))));
    \\    }
    \\    y[size_t(tok) * H + row0 + r] = out;
    \\  }
    \\}
;

const Kernels = struct { gateup: ?mlx.mlx_fast_metal_kernel = null, downred: ?mlx.mlx_fast_metal_kernel = null };
/// Indexed by format: nvfp4, mxfp4.
var kernels: [2]Kernels = .{ .{}, .{} };
var engaged = false;

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const outs = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn apply(kernel: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, out_shape: []const c_int, dt: mlx.mlx_dtype, grid: [3]c_int, rows: c_int, tmpl: []const struct { [*:0]const u8, c_int }, s: mlx.mlx_stream) !mlx.mlx_array {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, out_shape.ptr, out_shape.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, SGS, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ROWS", rows));
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

pub const Bank = struct { w: mlx.mlx_array, s: mlx.mlx_array };
/// Per-expert [E] scales in x's dtype on the gate and up outputs, then `clamp` (0 = none, else
/// GLM's whole-number limit: silu(min(g, c)) * clip(u, -c, c)).
pub const Scales = struct { gate: ?mlx.mlx_array = null, up: ?mlx.mlx_array = null, clamp: u32 = 0 };

/// y [R, hidden] = sum_k scores[r,k] * down_k(silu(gate_k(x_r)) * up_k(x_r)) for
/// R tokens. `x` [R, hidden] bf16/f16, banks [E, out, in/8] u32 + [E, out, in/GS]
/// u8, `inds` [R, TOPK] u32, `scores` [R, TOPK] in x's dtype, `sigtab` the SwiGLU
/// table. `views` (of the same banks) are what the kernels bind; `scales` [E] in
/// x's dtype multiply each expert's gate (before the SiLU) and up (before the clamp); `shared` (one token only)
/// adds the gated shared expert in the epilogue. Null outside the kernels' set.
pub fn decode(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Bank, up: Bank, down: Bank, views: ?moe_affine4.Views, scales: Scales, shared: ?moe_affine4.Shared, inds: mlx.mlx_array, scores: mlx.mlx_array, sigtab: mlx.mlx_array, mode: @import("model.zig").QuantMode, group_size: u32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const ki: usize = switch (mode) {
        .nvfp4 => if (group_size == 16) 0 else return null,
        .mxfp4 => if (group_size == 32) 1 else return null,
        else => return null,
    };
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32 or mlx.mlx_array_dtype(scores) != dt) return null;
    const gsh = mlx.getShape(gate.w);
    const dsh = mlx.getShape(down.w);
    if (gsh.len != 3 or dsh.len != 3 or !std.mem.eql(c_int, gsh, mlx.getShape(up.w))) return null;
    const inter = gsh[1];
    const hidden = gsh[2] * 8;
    const rows_per_sg = ROWS;
    if (dsh[1] != hidden or dsh[2] * 8 != inter) return null;
    if (@rem(hidden, 16) != 0 or @rem(inter, 16) != 0 or @rem(inter, rows_per_sg * SGS) != 0 or @rem(hidden, rows_per_sg * SGS) != 0) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len != 2 or xsh[1] != hidden) return null;
    const rows = xsh[0];
    const ish = mlx.getShape(inds);
    if (ish.len != 2 or ish[0] != rows or !std.mem.eql(c_int, ish, mlx.getShape(scores))) return null;
    const topk: c_int = ish[1];
    inline for (.{ scales.gate, scales.up }) |sc| if (sc) |a| if (mlx.mlx_array_dtype(a) != dt) return null;
    if (shared) |sh| {
        if (rows != 1 or mlx.mlx_array_dtype(sh.down) != dt or mlx.mlx_array_dtype(sh.logit) != dt or mlx.mlx_array_size(sh.logit) != 1 or mlx.mlx_array_size(sh.down) != @as(usize, @intCast(hidden))) return null;
    }

    const full_header = [2][:0]const u8{ E2M1_HEADER ++ NV_SCALE ++ DOT16, E2M1_HEADER ++ MX_SCALE ++ DOT16 };
    if (kernels[ki].gateup == null) {
        const names = [2][*:0]const u8{ "mlxserve_moe_fp4_gateup_nv", "mlxserve_moe_fp4_gateup_mx" };
        const ins = [_][*:0]const u8{ "x", "wg_q", "g_scales", "wu_q", "u_scales", "inds", "sigtab", "gsc", "usc" };
        kernels[ki].gateup = try makeKernel(names[ki], &ins, GATEUP_SOURCE, full_header[ki].ptr);
    }
    if (kernels[ki].downred == null) {
        const names = [2][*:0]const u8{ "mlxserve_moe_fp4_downred_nv", "mlxserve_moe_fp4_downred_mx" };
        const ins = [_][*:0]const u8{ "act", "wd_q", "d_scales", "inds", "scores", "sdown", "glog" };
        kernels[ki].downred = try makeKernel(names[ki], &ins, DOWNRED_SOURCE, full_header[ki].ptr);
    }
    const gs: c_int = @intCast(group_size);
    const b = if (views) |v| [3]Bank{ .{ .w = v.gate.w, .s = v.gate.s }, .{ .w = v.up.w, .s = v.up.s }, .{ .w = v.down.w, .s = v.down.s } } else [3]Bank{ gate, up, down };
    // Unused operands bind any array of the right type; the template flag keeps them unread.
    const gsc = scales.gate orelse scores;
    const usc = scales.up orelse scores;
    const act = try apply(kernels[ki].gateup.?, &.{ x, b[0].w, b[0].s, b[1].w, b[1].s, inds, sigtab, gsc, usc }, &.{ rows * topk, inter }, dt, .{ 32, @divExact(inter, rows_per_sg), rows * topk }, rows_per_sg, &.{ .{ "K", hidden }, .{ "N", inter }, .{ "GS", gs }, .{ "TOPK", topk }, .{ "GSCALE", @intFromBool(scales.gate != null) }, .{ "USCALE", @intFromBool(scales.up != null) }, .{ "CLAMP", @intCast(scales.clamp) } }, s);
    defer _ = mlx.mlx_array_free(act);
    const sh = shared orelse moe_affine4.Shared{ .down = scores, .logit = scores };
    const y = try apply(kernels[ki].downred.?, &.{ act, b[2].w, b[2].s, inds, scores, sh.down, sh.logit }, &.{ rows, hidden }, dt, .{ 32, @divExact(hidden, rows_per_sg), rows }, rows_per_sg, &.{ .{ "I", inter }, .{ "H", hidden }, .{ "GS", gs }, .{ "TOPK", topk }, .{ "SHARED", @intFromBool(shared != null) } }, s);
    if (!engaged) {
        engaged = true;
        @import("log.zig").info("[moe] fp4 decode kernels engaged: {s} rows={d} topk={d} inter={d} hidden={d} gate_scale={} shared={} views={}\n", .{ @tagName(mode), rows, topk, inter, hidden, scales.gate != null, shared != null, views != null });
    }
    return y;
}

const testing = std.testing;

fn randBf16(rnd: std.Random, shape: []const c_int, scale: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(buf);
    for (buf) |*v| v.* = (rnd.float(f32) - 0.5) * scale;
    const f = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

fn quant(w: mlx.mlx_array, gs: c_int, mode: [:0]const u8, s: mlx.mlx_stream) !struct { b: Bank, deq: mlx.mlx_array } {
    var pair = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(pair);
    try mlx.check(mlx.mlx_quantize(&pair, w, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(4), mode, .{}, s));
    var b: Bank = .{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new() };
    try mlx.check(mlx.mlx_vector_array_get(&b.w, pair, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b.s, pair, 1));
    var deq = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_dequantize(&deq, b.w, b.s, .{ .ctx = null }, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(4), mode, .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    return .{ .b = b, .deq = deq };
}

/// f32 truth for one token over the dequantized banks:
/// y = sum_k s_k * Wd_k (silu(Wg_k x) * Wu_k x), host copy [H].
fn truthRow(deq: [3]mlx.mlx_array, x: mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array, clamp: u32, s: mlx.mlx_stream) ![]f32 {
    var tr = std.ArrayList(mlx.mlx_array).empty;
    defer {
        for (tr.items) |a| _ = mlx.mlx_array_free(a);
        tr.deinit(testing.allocator);
    }
    const P = struct {
        fn keep(list: *std.ArrayList(mlx.mlx_array), a: mlx.mlx_array) !mlx.mlx_array {
            try list.append(testing.allocator, a);
            return a;
        }
    };
    const h: c_int = @intCast(mlx.mlx_array_size(x));
    const topk: c_int = @intCast(mlx.mlx_array_size(inds));
    var a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&a, x, .float32, s));
    const x32 = try P.keep(&tr, a);
    a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&a, x32, &.{ 1, h, 1 }, 3, s));
    const xc = try P.keep(&tr, a);
    var mats: [3]mlx.mlx_array = undefined;
    for (&mats, deq) |*m, d| {
        a = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_take_axis(&a, d, inds, 0, s));
        m.* = try P.keep(&tr, a);
    }
    a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&a, mats[0], xc, s));
    const g0 = try P.keep(&tr, a);
    a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&a, mats[1], xc, s));
    var u = try P.keep(&tr, a);
    var g = g0;
    if (clamp > 0) {
        const hi = mlx.mlx_array_new_float(@floatFromInt(clamp));
        defer _ = mlx.mlx_array_free(hi);
        const lo = mlx.mlx_array_new_float(-@as(f32, @floatFromInt(clamp)));
        defer _ = mlx.mlx_array_free(lo);
        a = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_minimum(&a, g0, hi, s));
        g = try P.keep(&tr, a);
        a = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_clip(&a, u, lo, hi, s));
        u = try P.keep(&tr, a);
    }
    a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&a, g, s));
    const sg = try P.keep(&tr, a);
    a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&a, g, sg, s));
    const gs = try P.keep(&tr, a);
    a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&a, gs, u, s));
    const act = try P.keep(&tr, a);
    a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&a, mats[2], act, s));
    const d = try P.keep(&tr, a);
    a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&a, scores, .float32, s));
    const s32 = try P.keep(&tr, a);
    a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&a, s32, &.{ topk, 1, 1 }, 3, s));
    const s3 = try P.keep(&tr, a);
    a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&a, d, s3, s));
    const wd = try P.keep(&tr, a);
    a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sum_axis(&a, wd, 0, false, s));
    const y = try P.keep(&tr, a);
    try mlx.check(mlx.mlx_array_eval(y));
    const out = try testing.allocator.alloc(f32, @intCast(h));
    @memcpy(out, mlx.mlx_array_data_float32(y).?[0..@intCast(h)]);
    return out;
}

fn row(a: mlx.mlx_array, r: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(a);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, a, &.{ r, 0 }, 2, &.{ r + 1, sh[1] }, 2, &.{ 1, 1 }, 2, s));
    var flat = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&flat, out, &.{sh[1]}, 1, s));
    _ = mlx.mlx_array_free(out);
    return flat;
}

test "moe fp4 decode: gate+up+SwiGLU and the weighted down match fp32 truth per token (mxfp4, nvfp4; widths 640 and a tensor rank's 320, per-expert gate scale)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    try fp4DecodeCase(640);
    try fp4DecodeCase(320);
}

fn fp4DecodeCase(I: c_int) !void {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x4D1E0F4);
    const rnd = prng.random();
    const E: c_int = 16;
    const H: c_int = 1536;
    const R: c_int = 3;
    const TOPK: c_int = 8;
    const Fmt = struct { name: [:0]const u8, gs: c_int, mode: @import("model.zig").QuantMode };
    for ([_]Fmt{ .{ .name = "mxfp4", .gs = 32, .mode = .mxfp4 }, .{ .name = "nvfp4", .gs = 16, .mode = .nvfp4 } }) |f| {
        var banks: [3]Bank = undefined;
        var deq: [3]mlx.mlx_array = undefined;
        const shapes = [3][3]c_int{ .{ E, I, H }, .{ E, I, H }, .{ E, H, I } };
        for (&banks, &deq, shapes) |*b, *d, sh| {
            const w = try randBf16(rnd, &sh, 0.2, s);
            defer _ = mlx.mlx_array_free(w);
            const q = try quant(w, f.gs, f.name, s);
            b.* = q.b;
            d.* = q.deq;
        }
        defer for (banks, deq) |b, d| {
            _ = mlx.mlx_array_free(b.w);
            _ = mlx.mlx_array_free(b.s);
            _ = mlx.mlx_array_free(d);
        };
        const x = try randBf16(rnd, &.{ R, H }, 2.0, s);
        defer _ = mlx.mlx_array_free(x);
        const idx = [3 * 8]u32{ 3, 0, 15, 7, 9, 1, 12, 5, 2, 3, 4, 5, 6, 7, 8, 9, 15, 14, 13, 12, 0, 1, 10, 11 };
        const inds = mlx.mlx_array_new_data(&idx, &[_]c_int{ R, TOPK }, 2, .uint32);
        defer _ = mlx.mlx_array_free(inds);
        const scores = try randBf16(rnd, &.{ R, TOPK }, 0.5, s);
        defer _ = mlx.mlx_array_free(scores);
        const sigtab = try xfm.swigluSigTable(s, .bfloat16, std.heap.c_allocator);
        // A gate scale multiplies expert e's gate output, so the truth scales its gate rows.
        const gsc = try randBf16(rnd, &.{E}, 1.0, s);
        defer _ = mlx.mlx_array_free(gsc);
        var gsc3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gsc3);
        try mlx.check(mlx.mlx_reshape(&gsc3, gsc, &.{ E, 1, 1 }, 3, s));
        var gate_t = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&gate_t, deq[0], gsc3, s));
        _ = mlx.mlx_array_free(deq[0]);
        deq[0] = gate_t;

        const ours = (try decode(s, x, banks[0], banks[1], banks[2], null, .{ .gate = gsc }, null, inds, scores, sigtab, f.mode, @intCast(f.gs))) orelse return error.KernelDeclined;
        defer _ = mlx.mlx_array_free(ours);
        var o32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o32);
        try mlx.check(mlx.mlx_astype(&o32, ours, .float32, s));
        try mlx.check(mlx.mlx_array_eval(o32));
        const po = mlx.mlx_array_data_float32(o32).?;
        for (0..@intCast(R)) |r| {
            const ri: c_int = @intCast(r);
            const xr = try row(x, ri, s);
            defer _ = mlx.mlx_array_free(xr);
            const ir = try row(inds, ri, s);
            defer _ = mlx.mlx_array_free(ir);
            const sr = try row(scores, ri, s);
            defer _ = mlx.mlx_array_free(sr);
            const pt = try truthRow(deq, xr, ir, sr, 0, s);
            defer testing.allocator.free(pt);
            var se: f64 = 0;
            var st: f64 = 0;
            for (pt, 0..) |t, i| {
                const o = po[r * @as(usize, @intCast(H)) + i];
                se += (o - t) * (o - t);
                st += t * t;
            }
            const rel = @sqrt(se / st);
            std.debug.print("[moe-fp4] {s} token {d}: rel rms err vs fp32 truth {d:.5}\n", .{ f.name, r, rel });
            try testing.expect(rel < 0.02);
        }
    }
}

/// [E] scales in [0.5, 1.5) in bf16: big enough that the clamp below bites, none near zero.
fn scaleVec(rnd: std.Random, n: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const v = try randBf16(rnd, &.{n}, 1.0, s);
    defer _ = mlx.mlx_array_free(v);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, v, one, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, sum, .bfloat16, s));
    return out;
}

fn scaledBank(deq: mlx.mlx_array, sc: mlx.mlx_array, e: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var sc3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc3);
    try mlx.check(mlx.mlx_reshape(&sc3, sc, &.{ e, 1, 1 }, 3, s));
    var sc32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc32);
    try mlx.check(mlx.mlx_astype(&sc32, sc3, .float32, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, deq, sc32, s));
    return out;
}

test "moe fp4 decode: a clamped SwiGLU with per-expert gate and up scales matches fp32 truth (mxfp4, nvfp4; one and two tokens)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xC1A4F4);
    const rnd = prng.random();
    const E: c_int = 16;
    const I: c_int = 640;
    const H: c_int = 1536;
    const TOPK: c_int = 8;
    const CLAMP: u32 = 2;
    const Fmt = struct { name: [:0]const u8, gs: c_int, mode: @import("model.zig").QuantMode };
    for ([_]Fmt{ .{ .name = "mxfp4", .gs = 32, .mode = .mxfp4 }, .{ .name = "nvfp4", .gs = 16, .mode = .nvfp4 } }) |f| {
        var banks: [3]Bank = undefined;
        var deq: [3]mlx.mlx_array = undefined;
        const shapes = [3][3]c_int{ .{ E, I, H }, .{ E, I, H }, .{ E, H, I } };
        for (&banks, &deq, shapes) |*b, *d, sh| {
            const w = try randBf16(rnd, &sh, 0.2, s);
            defer _ = mlx.mlx_array_free(w);
            const q = try quant(w, f.gs, f.name, s);
            b.* = q.b;
            d.* = q.deq;
        }
        defer for (banks, deq) |b, d| {
            _ = mlx.mlx_array_free(b.w);
            _ = mlx.mlx_array_free(b.s);
            _ = mlx.mlx_array_free(d);
        };
        const gsc = try scaleVec(rnd, E, s);
        defer _ = mlx.mlx_array_free(gsc);
        const usc = try scaleVec(rnd, E, s);
        defer _ = mlx.mlx_array_free(usc);
        // The scales multiply each expert's projection output, so the truth scales the dequantized rows.
        const scaled = [3]mlx.mlx_array{ try scaledBank(deq[0], gsc, E, s), try scaledBank(deq[1], usc, E, s), deq[2] };
        defer for (scaled[0..2]) |a| {
            _ = mlx.mlx_array_free(a);
        };

        const R: c_int = 2;
        const x = try randBf16(rnd, &.{ R, H }, 8.0, s);
        defer _ = mlx.mlx_array_free(x);
        const idx = [2 * 8]u32{ 3, 0, 15, 7, 9, 1, 12, 5, 2, 3, 4, 5, 6, 7, 8, 9 };
        const inds = mlx.mlx_array_new_data(&idx, &[_]c_int{ R, TOPK }, 2, .uint32);
        defer _ = mlx.mlx_array_free(inds);
        const scores = try randBf16(rnd, &.{ R, TOPK }, 0.5, s);
        defer _ = mlx.mlx_array_free(scores);
        const sigtab = try xfm.swigluSigTable(s, .bfloat16, std.heap.c_allocator);

        const ours = (try decode(s, x, banks[0], banks[1], banks[2], null, .{ .gate = gsc, .up = usc, .clamp = CLAMP }, null, inds, scores, sigtab, f.mode, @intCast(f.gs))) orelse return error.KernelDeclined;
        defer _ = mlx.mlx_array_free(ours);
        var o32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o32);
        try mlx.check(mlx.mlx_astype(&o32, ours, .float32, s));
        try mlx.check(mlx.mlx_array_eval(o32));
        const po = mlx.mlx_array_data_float32(o32).?;
        for (0..@intCast(R)) |r| {
            const ri: c_int = @intCast(r);
            const xr = try row(x, ri, s);
            defer _ = mlx.mlx_array_free(xr);
            const ir = try row(inds, ri, s);
            defer _ = mlx.mlx_array_free(ir);
            const sr = try row(scores, ri, s);
            defer _ = mlx.mlx_array_free(sr);
            const pt = try truthRow(scaled, xr, ir, sr, CLAMP, s);
            defer testing.allocator.free(pt);
            // The clamp must change the answer, or the check below proves nothing about it.
            const open = try truthRow(scaled, xr, ir, sr, 0, s);
            defer testing.allocator.free(open);
            var se: f64 = 0;
            var st: f64 = 0;
            var sc: f64 = 0;
            for (pt, open, 0..) |t, u, i| {
                const o = po[r * @as(usize, @intCast(H)) + i];
                se += (o - t) * (o - t);
                st += t * t;
                sc += (t - u) * (t - u);
            }
            const rel = @sqrt(se / st);
            std.debug.print("[moe-fp4] {s} clamp token {d}: rel rms err vs fp32 truth {d:.5} (clamp moves the answer by {d:.3})\n", .{ f.name, r, rel, @sqrt(sc / st) });
            try testing.expect(@sqrt(sc / st) > 0.05);
            try testing.expect(rel < 0.02);
        }
    }
}

test "moe fp4 decode: one-expert bank views and the fused shared expert match fp32 truth (mxfp4, nvfp4; one token)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E0F4);
    const rnd = prng.random();
    const E: c_int = 16;
    const I: c_int = 640;
    const H: c_int = 1536;
    const TOPK: c_int = 8;
    const Fmt = struct { name: [:0]const u8, gs: c_int, mode: @import("model.zig").QuantMode };
    for ([_]Fmt{ .{ .name = "mxfp4", .gs = 32, .mode = .mxfp4 }, .{ .name = "nvfp4", .gs = 16, .mode = .nvfp4 } }) |f| {
        var banks: [3]Bank = undefined;
        var deq: [3]mlx.mlx_array = undefined;
        const shapes = [3][3]c_int{ .{ E, I, H }, .{ E, I, H }, .{ E, H, I } };
        for (&banks, &deq, shapes) |*b, *d, sh| {
            const w = try randBf16(rnd, &sh, 0.2, s);
            defer _ = mlx.mlx_array_free(w);
            const q = try quant(w, f.gs, f.name, s);
            b.* = q.b;
            d.* = q.deq;
        }
        defer for (banks, deq) |b, d| {
            _ = mlx.mlx_array_free(b.w);
            _ = mlx.mlx_array_free(b.s);
            _ = mlx.mlx_array_free(d);
        };
        var owned: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (owned.items) |a| _ = mlx.mlx_array_free(a);
            owned.deinit(testing.allocator);
        }
        const A4 = moe_affine4.Bank;
        const views = (try moe_affine4.bindViews(A4{ .w = banks[0].w, .s = banks[0].s, .b = .{ .ctx = null } }, A4{ .w = banks[1].w, .s = banks[1].s, .b = .{ .ctx = null } }, A4{ .w = banks[2].w, .s = banks[2].s, .b = .{ .ctx = null } }, &owned, testing.allocator, s)) orelse return error.SkipZigTest;

        const x = try randBf16(rnd, &.{ 1, H }, 2.0, s);
        defer _ = mlx.mlx_array_free(x);
        const idx = [8]u32{ 3, 0, 15, 7, 9, 1, 12, 5 };
        const inds = mlx.mlx_array_new_data(&idx, &[_]c_int{ 1, TOPK }, 2, .uint32);
        defer _ = mlx.mlx_array_free(inds);
        const scores = try randBf16(rnd, &.{ 1, TOPK }, 0.5, s);
        defer _ = mlx.mlx_array_free(scores);
        const sdown = try randBf16(rnd, &.{H}, 1.0, s);
        defer _ = mlx.mlx_array_free(sdown);
        const logit = try randBf16(rnd, &.{1}, 1.0, s);
        defer _ = mlx.mlx_array_free(logit);
        const sigtab = try xfm.swigluSigTable(s, .bfloat16, std.heap.c_allocator);

        const plain = (try decode(s, x, banks[0], banks[1], banks[2], null, .{}, null, inds, scores, sigtab, f.mode, @intCast(f.gs))) orelse return error.KernelDeclined;
        defer _ = mlx.mlx_array_free(plain);
        const viewed = (try decode(s, x, banks[0], banks[1], banks[2], views, .{}, null, inds, scores, sigtab, f.mode, @intCast(f.gs))) orelse return error.KernelDeclined;
        defer _ = mlx.mlx_array_free(viewed);
        const fused = (try decode(s, x, banks[0], banks[1], banks[2], views, .{}, .{ .down = sdown, .logit = logit }, inds, scores, sigtab, f.mode, @intCast(f.gs))) orelse return error.KernelDeclined;
        defer _ = mlx.mlx_array_free(fused);
        // Views bind the same bytes: the output is bit-identical.
        try testing.expectEqual(@as(f32, 0), try xfm.maxAbsDiffF32(plain, viewed, s));

        var f32o = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f32o);
        try mlx.check(mlx.mlx_astype(&f32o, fused, .float32, s));
        var sd32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sd32);
        try mlx.check(mlx.mlx_astype(&sd32, sdown, .float32, s));
        var lg32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lg32);
        try mlx.check(mlx.mlx_astype(&lg32, logit, .float32, s));
        try mlx.check(mlx.mlx_array_eval(f32o));
        try mlx.check(mlx.mlx_array_eval(sd32));
        try mlx.check(mlx.mlx_array_eval(lg32));
        const po = mlx.mlx_array_data_float32(f32o).?;
        const psd = mlx.mlx_array_data_float32(sd32).?;
        const gate = 1.0 / (1.0 + @exp(-mlx.mlx_array_data_float32(lg32).?[0]));
        const xr = try row(x, 0, s);
        defer _ = mlx.mlx_array_free(xr);
        const ir = try row(inds, 0, s);
        defer _ = mlx.mlx_array_free(ir);
        const sr = try row(scores, 0, s);
        defer _ = mlx.mlx_array_free(sr);
        const pt = try truthRow(deq, xr, ir, sr, 0, s);
        defer testing.allocator.free(pt);
        var se: f64 = 0;
        var st: f64 = 0;
        for (pt, 0..) |t0, i| {
            const t = t0 + gate * psd[i];
            se += (po[i] - t) * (po[i] - t);
            st += t * t;
        }
        const rel = @sqrt(se / st);
        std.debug.print("[moe-fp4] {s} views+shared: rel rms err vs fp32 truth {d:.5}\n", .{ f.name, rel });
        try testing.expect(rel < 0.02);
    }
}
