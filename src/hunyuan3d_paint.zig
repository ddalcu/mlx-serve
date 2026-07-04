//! Hunyuan3D-2.1 PAINT engine (texture stage): multiview PBR texture
//! generation for meshes produced by the shape engine (src/hunyuan3d.zig,
//! `generateMeshRaw` seam). Ported from the Tencent reference `hy3dpaint/`
//! (textureGenPipeline.py + hunyuanpaintpbr/) to mlx-c FFI.
//!
//! Pipeline: xatlas UV unwrap (src/uvwrap.zig) → 6 fixed-view ortho geometry
//! renders (src/rasterize.zig + src/bake.zig) → SD-VAE encode of ref photo +
//! per-view normal/position maps → 2.5D UNet multiview denoise (2 PBR
//! materials × 6 views, DDIM v-pred/zero-SNR/trailing) → VAE decode →
//! back-project bake (cos⁴ blend) → vertex-graph inpaint (src/texinpaint.zig)
//! → textured GLB (src/glb.zig).
//!
//! Parity facts honored here (verified against reference sources 2026-07-04):
//! - Runtime scheduler in the reference is UniPCMultistep FROM the shipped
//!   DDIM config; we implement DDIM (same config: v-prediction,
//!   rescale_betas_zero_snr, trailing spacing) first — fixtures are dumped
//!   with the scheduler the Zig side implements. UniPC-bh2 15-step is a
//!   follow-up perf task.
//! - The alphas-cumprod table is computed in f32 exactly like diffusers
//!   (linspace of sqrt-betas in f32, square, cumprod, zero-SNR rescale) —
//!   NOT fp16 (fp16 collapses the terminal near-zero alphas).
//! - Trailing timesteps: round(arange(T, 0, -T/steps)) - 1 with numpy's
//!   ties-to-even rounding (libc rint), descending.
//! - DDIM step: eta 0, clip_sample false, set_alpha_to_one (prev_t < 0 →
//!   final alpha 1.0), steps_offset unused by trailing spacing.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;
/// LayerNorm/GroupNorm eps for DINO + VAE (SD2.x VAE GroupNorm is 1e-6; DINOv2
/// LayerNorm is 1e-6; the ImageProjModel's nn.LayerNorm uses the 1e-5 default).
const PAINT_LN_EPS: f32 = 1e-6;

// ════════════════════════════════════════════════════════════════════════
// DDIM scheduler (v-prediction, zero-SNR, trailing) — pure f32 math.
// ════════════════════════════════════════════════════════════════════════

pub const DDIM_TRAIN_STEPS: usize = 1000;
pub const DDIM_BETA_START: f32 = 0.00085;
pub const DDIM_BETA_END: f32 = 0.012;

extern "c" fn rint(x: f64) f64; // ties-to-even, matches numpy round()

pub const Scheduler = struct {
    allocator: std.mem.Allocator,
    /// f32 alphas-cumprod table after zero-SNR rescale; [999] is exactly 0.
    alphas_cumprod: []f32,
    /// Descending trailing-spaced timesteps (len = num_steps).
    timesteps: []u32,
    num_steps: usize,

    pub fn init(allocator: std.mem.Allocator, num_steps: usize) !Scheduler {
        std.debug.assert(num_steps > 0 and num_steps <= DDIM_TRAIN_STEPS);
        const ac = try allocator.alloc(f32, DDIM_TRAIN_STEPS);
        errdefer allocator.free(ac);

        // scaled_linear betas: linspace(sqrt(b0), sqrt(b1), T)² — f32 like diffusers.
        const s0: f32 = std.math.sqrt(DDIM_BETA_START);
        const s1: f32 = std.math.sqrt(DDIM_BETA_END);
        var cum: f32 = 1.0;
        for (0..DDIM_TRAIN_STEPS) |i| {
            const frac: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(DDIM_TRAIN_STEPS - 1));
            const sb = s0 + (s1 - s0) * frac;
            const beta = sb * sb;
            cum *= 1.0 - beta;
            ac[i] = cum;
        }

        // rescale_betas_zero_snr (Lin et al.): shift+scale sqrt(alphas_cumprod)
        // so the terminal SNR is exactly zero. Same op order as diffusers.
        {
            const f0: f32 = std.math.sqrt(ac[0]);
            const t_last: f32 = std.math.sqrt(ac[DDIM_TRAIN_STEPS - 1]);
            const scale = f0 / (f0 - t_last);
            for (ac) |*v| {
                const sq = (std.math.sqrt(v.*) - t_last) * scale;
                v.* = sq * sq;
            }
        }

        // Trailing spacing: round(arange(T, 0, -T/steps)) - 1, descending.
        const ts = try allocator.alloc(u32, num_steps);
        errdefer allocator.free(ts);
        const ratio: f64 = @as(f64, @floatFromInt(DDIM_TRAIN_STEPS)) / @as(f64, @floatFromInt(num_steps));
        for (0..num_steps) |i| {
            const x: f64 = @as(f64, @floatFromInt(DDIM_TRAIN_STEPS)) - @as(f64, @floatFromInt(i)) * ratio;
            ts[i] = @intCast(@as(i64, @intFromFloat(rint(x))) - 1);
        }

        return .{ .allocator = allocator, .alphas_cumprod = ac, .timesteps = ts, .num_steps = num_steps };
    }

    pub fn deinit(self: *Scheduler) void {
        self.allocator.free(self.alphas_cumprod);
        self.allocator.free(self.timesteps);
        self.* = undefined;
    }

    /// One DDIM step (v-prediction, eta 0, no clipping), in place on `sample`.
    /// `model_output` is the UNet's v prediction at timestep `t`.
    pub fn step(self: *const Scheduler, model_output: []const f32, t: u32, sample: []f32) void {
        std.debug.assert(model_output.len == sample.len);
        const prev_i: i64 = @as(i64, t) - @as(i64, @intCast(DDIM_TRAIN_STEPS / self.num_steps));
        const alpha_t: f32 = self.alphas_cumprod[t];
        const alpha_prev: f32 = if (prev_i >= 0) self.alphas_cumprod[@intCast(prev_i)] else 1.0; // set_alpha_to_one
        const beta_t: f32 = 1.0 - alpha_t;
        const sa = std.math.sqrt(alpha_t);
        const sb = std.math.sqrt(beta_t);
        const sap = std.math.sqrt(alpha_prev);
        const sbp = std.math.sqrt(1.0 - alpha_prev);
        for (sample, model_output) |*x, v| {
            const pred_x0 = sa * x.* - sb * v;
            const pred_eps = sa * v + sb * x.*;
            x.* = sap * pred_x0 + sbp * pred_eps;
        }
    }
};

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic first; golden values generated from diffusers
// DDIMScheduler with the SHIPPED hunyuan3d-paintpbr-v2-1 scheduler config
// (see the dump in tests/hy3d_paint_weights_contract.md workflow).
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "hy3d-paint scheduler: zero-SNR alphas table matches diffusers" {
    const a = testing.allocator;
    var s = try Scheduler.init(a, 30);
    defer s.deinit();
    // Golden from diffusers 0.x with the shipped config (f32 table).
    try testing.expectApproxEqRel(@as(f32, 0.9991499781608582), s.alphas_cumprod[0], 2e-6);
    try testing.expectApproxEqRel(@as(f32, 0.2423589825630188), s.alphas_cumprod[499], 2e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.9679627882851491e-07), s.alphas_cumprod[998], 1e-9);
    try testing.expectEqual(@as(f32, 0.0), s.alphas_cumprod[999]); // terminal SNR exactly zero
}

test "hy3d-paint scheduler: trailing timesteps match diffusers (30 steps)" {
    const a = testing.allocator;
    var s = try Scheduler.init(a, 30);
    defer s.deinit();
    const expected = [_]u32{ 999, 966, 932, 899, 866, 832, 799, 766, 732, 699, 666, 632, 599, 566, 532, 499, 466, 432, 399, 366, 332, 299, 266, 232, 199, 166, 132, 99, 66, 32 };
    try testing.expectEqualSlices(u32, &expected, s.timesteps);
}

test "hy3d-paint scheduler: v-pred DDIM step matches diffusers goldens" {
    const a = testing.allocator;
    var s = try Scheduler.init(a, 30);
    defer s.deinit();

    var sample: [8]f32 = undefined;
    var model_output: [8]f32 = undefined;
    for (0..8) |i| {
        const f: f32 = @as(f32, @floatFromInt(i)) / 7.0;
        sample[i] = -1.0 + 2.0 * f; // linspace(-1, 1, 8)
        model_output[i] = 0.5 - 1.0 * f; // linspace(0.5, -0.5, 8)
    }

    // t=999 (first step)
    var x = sample;
    s.step(&model_output, 999, &x);
    const g999 = [_]f32{ -1.0077428817749023, -0.7198163270950317, -0.43188977241516113, -0.14396321773529053, 0.14396321773529053, 0.43188977241516113, 0.7198163270950317, 1.0077428817749023 };
    for (0..8) |i| try testing.expectApproxEqAbs(g999[i], x[i], 2e-6);

    // t=499 (mid)
    x = sample;
    s.step(&model_output, 499, &x);
    const g499 = [_]f32{ -1.0248523950576782, -0.7320374846458435, -0.43922245502471924, -0.14640744030475616, 0.14640744030475616, 0.43922245502471924, 0.7320374846458435, 1.0248523950576782 };
    for (0..8) |i| try testing.expectApproxEqAbs(g499[i], x[i], 2e-6);

    // t=32 (last; prev_t < 0 → final alpha 1.0)
    x = sample;
    s.step(&model_output, 32, &x);
    const g32 = [_]f32{ -1.0735630989074707, -0.7668308019638062, -0.4600984454154968, -0.1533661186695099, 0.1533661186695099, 0.4600984454154968, 0.7668308019638062, 1.0735630989074707 };
    for (0..8) |i| try testing.expectApproxEqAbs(g32[i], x[i], 2e-6);
}

// ════════════════════════════════════════════════════════════════════════
// Low-level mlx helpers — file-local primitives (self-contained sibling of
// flux.zig / hunyuan3d.zig; dense dtype fp16, SD2.x / DINOv2-giant shapes).
// hunyuan3d.zig's equivalents are file-private, so they are cloned here per
// the flux/krea self-contained-sibling convention.
// ════════════════════════════════════════════════════════════════════════

inline fn matmul(x: mlx.mlx_array, w_t: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, x, w_t, s));
    return o;
}
inline fn addA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, a, b, s));
    return o;
}
inline fn mulA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, a, b, s));
    return o;
}
inline fn subA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&o, a, b, s));
    return o;
}
inline fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}
inline fn transpose(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}
inline fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}
inline fn contig(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}
inline fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, s));
    return mulA(x, sig, s);
}
inline fn layerNorm(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_layer_norm(&o, x, w, b, eps, s));
    return o;
}
fn scalarF(v: f32) mlx.mlx_array {
    return mlx.mlx_array_new_float(v);
}
fn concat(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}
/// Slice [start,stop) on `axis` of an N-D array (N ≤ 8).
fn sliceAxis(x: mlx.mlx_array, axis: usize, start: c_int, stop: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const nd = sh.len;
    var lo: [8]c_int = undefined;
    var hi: [8]c_int = undefined;
    var st: [8]c_int = undefined;
    for (0..nd) |i| {
        lo[i] = 0;
        hi[i] = sh[i];
        st[i] = 1;
    }
    lo[axis] = start;
    hi[axis] = stop;
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, lo[0..nd].ptr, nd, hi[0..nd].ptr, nd, st[0..nd].ptr, nd, s));
    return o;
}
/// SDPA without mask: q/k/v [B,H,L,hd].
fn sdpa(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    const null_a = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, "", null_a, null_a, s));
    return o;
}
/// [B,L,H*hd] → [B,H,L,hd]
fn splitHeads(x: mlx.mlx_array, heads: c_int, hd: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const x4 = try reshape(x, &[_]c_int{ sh[0], sh[1], heads, hd }, s);
    defer _ = mlx.mlx_array_free(x4);
    return transpose(x4, &[_]c_int{ 0, 2, 1, 3 }, s);
}
/// [B,H,L,hd] → [B,L,H*hd]
fn mergeHeads(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [B,H,L,hd]
    const t = try transpose(x, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(t);
    return reshape(t, &[_]c_int{ sh[0], sh[2], sh[1] * sh[3] }, s);
}

// ── Weight-map helpers (mirror hunyuan3d.zig) ──

fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const a = w.get(key) orelse {
        log.err("[hy3d-paint] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingPaintWeight;
    };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, a));
    return o;
}
fn ownOpt(w: *const Weights, key: []const u8) ?mlx.mlx_array {
    const a = w.get(key) orelse return null;
    var o = mlx.mlx_array_new();
    mlx.check(mlx.mlx_array_set(&o, a)) catch return null;
    return o;
}
fn fmtKey(a: std.mem.Allocator, comptime f: []const u8, args: anytype) ![]u8 {
    return std.fmt.allocPrint(a, f, args);
}
/// Own a tensor and cast it to fp16 (the engine's dense dtype). Used for
/// convs/norms/biases/embeddings so a future fp32 build loads unchanged.
fn ownF16(w: *const Weights, key: []const u8, s: S) !mlx.mlx_array {
    const raw = try ownWeight(w, key);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, .float16, s);
}
fn keyF16(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, sub: []const u8, s: S) !mlx.mlx_array {
    const k = try fmtKey(a, "{s}.{s}", .{ pfx, sub });
    defer a.free(k);
    return ownF16(w, k, s);
}
fn optF16(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, sub: []const u8, s: S) ?mlx.mlx_array {
    const k = fmtKey(a, "{s}.{s}", .{ pfx, sub }) catch return null;
    defer a.free(k);
    const raw = ownOpt(w, k) orelse return null;
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, .float16, s) catch null;
}
/// Load a norm vector (weight or bias) as f32 (mlx fast norms upcast the
/// activations internally; f32 params close fp16 rounding drift for free).
fn normVec(w: *const Weights, key: []const u8, s: S) !mlx.mlx_array {
    const raw = try ownWeight(w, key);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, .float32, s);
}
fn normVecKey(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, sub: []const u8, s: S) !mlx.mlx_array {
    const k = try fmtKey(a, "{s}.{s}", .{ pfx, sub });
    defer a.free(k);
    return normVec(w, k, s);
}

/// Load ONE safetensors file into a Weights map (CPU stream — Load::eval_gpu is
/// Not Implemented). The iterator hands a +1 reference in `value`; transfer it
/// straight into the map (the model.zig pattern) — copying+dropping leaks.
fn loadFileWeights(allocator: std.mem.Allocator, model_dir: []const u8, file: []const u8) !Weights {
    var w = Weights.init(allocator);
    errdefer w.deinit();
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, file }, 0);
    defer allocator.free(path);

    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path, cpu_s));

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();
        const rc = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (rc != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }
        const owned_key = try allocator.dupe(u8, std.mem.span(key.?));
        errdefer allocator.free(owned_key);
        try w.map.put(owned_key, value);
    }
    log.info("[hy3d-paint] loaded {d} tensors from {s}\n", .{ w.count(), file });
    return w;
}

// ════════════════════════════════════════════════════════════════════════
// MixedLinear — fp16 OR affine-quantized, bits/group_size inferred from tensor
// geometry (the hunyuan3d.zig primitive, cloned; dense dtype float16). VAE
// linears are always dense (never quantized); DINO/ImageProj linears may be
// 8-bit affine gs64 in the shipping build.
// ════════════════════════════════════════════════════════════════════════

const MixedLinear = struct {
    quantized: bool,
    w: mlx.mlx_array, // quantized: packed u32 [out, in*bits/32]; dense: pre-transposed [in,out] f16
    scales: mlx.mlx_array = .{ .ctx = null },
    biases: mlx.mlx_array = .{ .ctx = null },
    add_bias: ?mlx.mlx_array = null,
    bits: u32 = 0,
    group_size: u32 = 0,

    fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, in_features: u32, s: S) !MixedLinear {
        const wk = try fmtKey(a, "{s}.weight", .{prefix});
        defer a.free(wk);
        const sk = try fmtKey(a, "{s}.scales", .{prefix});
        defer a.free(sk);
        const bk = try fmtKey(a, "{s}.biases", .{prefix});
        defer a.free(bk);
        const ak = try fmtKey(a, "{s}.bias", .{prefix});
        defer a.free(ak);

        if (ownOpt(w, sk)) |scales| {
            const weight = try ownWeight(w, wk);
            const biases = try ownWeight(w, bk);
            const w_cols: u32 = @intCast(mlx.getShape(weight)[1]); // in*bits/32
            const s_cols: u32 = @intCast(mlx.getShape(scales)[1]); // in/group_size
            const bits: u32 = @intCast(@divExact(32 * w_cols, in_features));
            const gs: u32 = @intCast(@divExact(in_features, s_cols));
            return .{
                .quantized = true,
                .w = weight,
                .scales = scales,
                .biases = biases,
                .add_bias = ownOpt(w, ak),
                .bits = bits,
                .group_size = gs,
            };
        }
        // Dense: pre-transpose [out,in] → [in,out], materialize, cast f16.
        const raw = try ownWeight(w, wk);
        defer _ = mlx.mlx_array_free(raw);
        const t = try transpose(raw, &[_]c_int{ 1, 0 }, s);
        defer _ = mlx.mlx_array_free(t);
        const tc = try contig(t, s);
        defer _ = mlx.mlx_array_free(tc);
        const wt = try astype(tc, .float16, s);
        return .{ .quantized = false, .w = wt, .add_bias = ownOpt(w, ak) };
    }

    fn deinit(self: *MixedLinear) void {
        _ = mlx.mlx_array_free(self.w);
        if (self.quantized) {
            _ = mlx.mlx_array_free(self.scales);
            _ = mlx.mlx_array_free(self.biases);
        }
        if (self.add_bias) |b| _ = mlx.mlx_array_free(b);
    }

    fn forward(self: *const MixedLinear, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const xh = try astype(x, .float16, s);
        defer _ = mlx.mlx_array_free(xh);
        var o = mlx.mlx_array_new();
        if (self.quantized) {
            try mlx.check(mlx.mlx_quantized_matmul(&o, xh, self.w, self.scales, self.biases, true, mlx.mlx_optional_int.some(@intCast(self.group_size)), mlx.mlx_optional_int.some(@intCast(self.bits)), "affine", s));
        } else {
            try mlx.check(mlx.mlx_matmul(&o, xh, self.w, s));
        }
        if (self.add_bias) |b| {
            const r = try addA(o, b, s);
            _ = mlx.mlx_array_free(o);
            o = r;
        }
        return o;
    }
};

// ════════════════════════════════════════════════════════════════════════
// PaintConfig — the fields this engine reads out of the synthesized paint
// config.json (contract §3). VAE / DINO / ImageProjModel only; the UNet fields
// belong to P2-7/P2-8.
// ════════════════════════════════════════════════════════════════════════

pub const PaintConfig = struct {
    // VAE (SD2.x AutoencoderKL)
    vae_scaling_factor: f32 = 0.18215,
    vae_latent_channels: u32 = 4,
    vae_norm_groups: u32 = 32,
    // DINOv2-giant
    dino_hidden: u32 = 1536,
    dino_layers: u32 = 40,
    dino_heads: u32 = 24,
    dino_head_dim: u32 = 64,
    dino_patch: u32 = 14,
    dino_image_size: u32 = 518,
    dino_intermediate: u32 = 4096,
    // ImageProjModel (lives in unet.safetensors)
    imageproj_clip_dim: u32 = 1536,
    imageproj_cross_dim: u32 = 1024,
    imageproj_num_tokens: u32 = 4,

    /// pos-embed patch grid side = image_size / patch (37 for 518/14).
    pub fn posEmbedGrid(self: PaintConfig) u32 {
        return self.dino_image_size / self.dino_patch;
    }
    /// stored pos-embed token count = grid² + 1 CLS (1370).
    pub fn posEmbedTokens(self: PaintConfig) u32 {
        const g = self.posEmbedGrid();
        return g * g + 1;
    }
    pub fn dinoHeadDim(self: PaintConfig) u32 {
        return self.dino_hidden / self.dino_heads; // 64
    }
    /// DINO token count for an input side (grid² + 1; 257 for a 224 input).
    pub fn dinoTokensFor(self: PaintConfig, input_side: u32) u32 {
        const g = input_side / self.dino_patch;
        return g * g + 1;
    }
};

fn cfgU32(obj: std.json.ObjectMap, key: []const u8, dflt: u32) u32 {
    if (obj.get(key)) |v| {
        if (v == .integer) return @intCast(v.integer);
    }
    return dflt;
}
fn cfgF32(obj: std.json.ObjectMap, key: []const u8, dflt: f32) f32 {
    if (obj.get(key)) |v| {
        if (v == .float) return @floatCast(v.float);
        if (v == .integer) return @floatFromInt(v.integer);
    }
    return dflt;
}

/// Parse the synthesized paint config.json text (pure — hermetically tested).
pub fn parseConfigText(allocator: std.mem.Allocator, text: []const u8) !PaintConfig {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadPaintConfig;
    const obj = parsed.value.object;
    const mt = obj.get("model_type") orelse return error.BadPaintConfig;
    if (mt != .string or !std.mem.startsWith(u8, mt.string, "hunyuan3d")) return error.BadPaintConfig;
    var cfg = PaintConfig{};
    if (obj.get("vae")) |v| {
        if (v == .object) {
            cfg.vae_scaling_factor = cfgF32(v.object, "scaling_factor", cfg.vae_scaling_factor);
            cfg.vae_latent_channels = cfgU32(v.object, "latent_channels", cfg.vae_latent_channels);
            cfg.vae_norm_groups = cfgU32(v.object, "norm_num_groups", cfg.vae_norm_groups);
        }
    }
    if (obj.get("dino")) |v| {
        if (v == .object) {
            cfg.dino_hidden = cfgU32(v.object, "hidden_size", cfg.dino_hidden);
            cfg.dino_layers = cfgU32(v.object, "num_layers", cfg.dino_layers);
            cfg.dino_heads = cfgU32(v.object, "num_heads", cfg.dino_heads);
            cfg.dino_head_dim = cfgU32(v.object, "head_dim", cfg.dino_head_dim);
            cfg.dino_patch = cfgU32(v.object, "patch_size", cfg.dino_patch);
            cfg.dino_image_size = cfgU32(v.object, "image_size", cfg.dino_image_size);
            cfg.dino_intermediate = cfgU32(v.object, "intermediate_size", cfg.dino_intermediate);
        }
    }
    if (obj.get("image_proj")) |v| {
        if (v == .object) {
            cfg.imageproj_clip_dim = cfgU32(v.object, "clip_embeddings_dim", cfg.imageproj_clip_dim);
            cfg.imageproj_cross_dim = cfgU32(v.object, "cross_attention_dim", cfg.imageproj_cross_dim);
            cfg.imageproj_num_tokens = cfgU32(v.object, "num_context_tokens", cfg.imageproj_num_tokens);
        }
    }
    return cfg;
}

fn readConfigFile(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !PaintConfig {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return error.BadPaintConfig;
    const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(path);
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = try rs.interface.allocRemaining(allocator, .limited(1024 * 1024));
    defer allocator.free(content);
    return parseConfigText(allocator, content);
}

// ════════════════════════════════════════════════════════════════════════
// SD 2.x AutoencoderKL (encoder + decoder). Convs are fp16 OHWI [O,kH,kW,I];
// data flows NHWC inside each conv/groupnorm. block_out_channels [128,256,512,
// 512]; mid-block single-head self-attention uses the LEGACY diffusers naming
// (query/key/value/proj_attn + group_norm). GroupNorm eps 1e-6. Structural
// mirror of flux.zig's VAE, adapted to SD2.x channels + fp16 dtype.
// ════════════════════════════════════════════════════════════════════════

const VAE_GROUPS: c_int = 32;
const VAE_GN_EPS: f32 = 1e-6;
const VAE_MID_CH: u32 = 512; // block_out_channels last

/// conv2d on NHWC input with fp16 OHWI weight [out,kh,kw,in] + bias [out].
fn conv2d(x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, pad: c_int, s: S) !mlx.mlx_array {
    // Materialize: mlx_conv2d silently miscomputes on strided/lazy-view inputs.
    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    try mlx.check(mlx.mlx_contiguous(&xc, x, false, s));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, 1, 1, pad, pad, 1, 1, 1, s));
    const r = try addA(o, bias, s);
    _ = mlx.mlx_array_free(o);
    return r;
}

/// PyTorch-compatible GroupNorm on NHWC [1,H,W,C], fp32 internal, + affine,
/// returns fp16.
fn groupNorm(x: mlx.mlx_array, weight: mlx.mlx_array, bias: mlx.mlx_array, groups: c_int, eps: f32, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,H,W,C]
    const H = sh[1];
    const Wd = sh[2];
    const C = sh[3];
    const cg = @divExact(C, groups);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    // [1,H,W,C] -> [1, H*W, groups, cg] -> [1, groups, H*W, cg] -> [1, groups, H*W*cg]
    const r1 = try reshape(xf, &[_]c_int{ 1, H * Wd, groups, cg }, s);
    defer _ = mlx.mlx_array_free(r1);
    const t1 = try transpose(r1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(t1);
    const flat = try reshape(t1, &[_]c_int{ 1, groups, H * Wd * cg }, s);
    defer _ = mlx.mlx_array_free(flat);
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, flat, -1, true, s));
    const xc = try subA(flat, mean, s);
    defer _ = mlx.mlx_array_free(xc);
    const sq = try mulA(xc, xc, s);
    defer _ = mlx.mlx_array_free(sq);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_mean_axis(&v, sq, -1, true, s));
    const epsa = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(epsa);
    var ve = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ve);
    try mlx.check(mlx.mlx_add(&ve, v, epsa, s));
    var rsq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rsq);
    try mlx.check(mlx.mlx_rsqrt(&rsq, ve, s));
    const norm = try mulA(xc, rsq, s);
    defer _ = mlx.mlx_array_free(norm);
    // back to NHWC
    const b1 = try reshape(norm, &[_]c_int{ 1, groups, H * Wd, cg }, s);
    defer _ = mlx.mlx_array_free(b1);
    const b2 = try transpose(b1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(b2);
    const b3 = try reshape(b2, &[_]c_int{ 1, H, Wd, C }, s);
    defer _ = mlx.mlx_array_free(b3);
    const wf = try astype(weight, .float32, s);
    defer _ = mlx.mlx_array_free(wf);
    const bf = try astype(bias, .float32, s);
    defer _ = mlx.mlx_array_free(bf);
    const sc = try mulA(b3, wf, s);
    defer _ = mlx.mlx_array_free(sc);
    const out = try addA(sc, bf, s);
    defer _ = mlx.mlx_array_free(out);
    return astype(out, .float16, s);
}

/// Asymmetric (0,1,0,1) zero-pad + 3x3 stride-2 valid conv on NHWC (diffusers
/// Downsample2D — pads RIGHT/BOTTOM only, so an odd input side downsamples one
/// smaller than a symmetric pad would).
fn conv2dDown(x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, s: S) !mlx.mlx_array {
    const axes = [_]c_int{ 1, 2 };
    const low = [_]c_int{ 0, 0 };
    const high = [_]c_int{ 1, 1 };
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p);
    try mlx.check(mlx.mlx_pad(&p, x, &axes, 2, &low, 2, &high, 2, zero, "constant", s));
    var pc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pc);
    try mlx.check(mlx.mlx_contiguous(&pc, p, false, s));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, pc, w, 2, 2, 0, 0, 1, 1, 1, s));
    const r = try addA(o, bias, s);
    _ = mlx.mlx_array_free(o);
    return r;
}

/// Nearest 2x upsample (repeat) + 3x3 pad-1 conv (diffusers Upsample2D).
fn upsampleConv(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var r1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r1);
    try mlx.check(mlx.mlx_repeat_axis(&r1, x, 2, 1, s));
    var r2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r2);
    try mlx.check(mlx.mlx_repeat_axis(&r2, r1, 2, 2, s));
    return conv2d(r2, w, b, 1, s);
}

const Resnet = struct {
    n1w: mlx.mlx_array,
    n1b: mlx.mlx_array,
    c1w: mlx.mlx_array,
    c1b: mlx.mlx_array,
    n2w: mlx.mlx_array,
    n2b: mlx.mlx_array,
    c2w: mlx.mlx_array,
    c2b: mlx.mlx_array,
    sw: ?mlx.mlx_array = null, // conv_shortcut (1x1) where channels change
    sb: ?mlx.mlx_array = null,
    fn deinit(self: *Resnet) void {
        inline for (.{ "n1w", "n1b", "c1w", "c1b", "n2w", "n2b", "c2w", "c2b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        if (self.sw) |x| _ = mlx.mlx_array_free(x);
        if (self.sb) |x| _ = mlx.mlx_array_free(x);
    }
    fn forward(self: *const Resnet, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const h0 = try groupNorm(x, self.n1w, self.n1b, VAE_GROUPS, VAE_GN_EPS, s);
        defer _ = mlx.mlx_array_free(h0);
        const a0 = try silu(h0, s);
        defer _ = mlx.mlx_array_free(a0);
        const c1 = try conv2d(a0, self.c1w, self.c1b, 1, s);
        defer _ = mlx.mlx_array_free(c1);
        const h1 = try groupNorm(c1, self.n2w, self.n2b, VAE_GROUPS, VAE_GN_EPS, s);
        defer _ = mlx.mlx_array_free(h1);
        const a1 = try silu(h1, s);
        defer _ = mlx.mlx_array_free(a1);
        const c2 = try conv2d(a1, self.c2w, self.c2b, 1, s);
        defer _ = mlx.mlx_array_free(c2);
        if (self.sw) |sw| {
            const sc = try conv2d(x, sw, self.sb.?, 0, s);
            defer _ = mlx.mlx_array_free(sc);
            return addA(c2, sc, s);
        }
        return addA(c2, x, s);
    }
};

const VaeAttn = struct {
    gnw: mlx.mlx_array,
    gnb: mlx.mlx_array,
    q: MixedLinear,
    k: MixedLinear,
    v: MixedLinear,
    o: MixedLinear,
    fn deinit(self: *VaeAttn) void {
        _ = mlx.mlx_array_free(self.gnw);
        _ = mlx.mlx_array_free(self.gnb);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
    }
    fn forward(self: *const VaeAttn, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const sh = mlx.getShape(x);
        const H = sh[1];
        const Wd = sh[2];
        const C = sh[3];
        const normed = try groupNorm(x, self.gnw, self.gnb, VAE_GROUPS, VAE_GN_EPS, s);
        defer _ = mlx.mlx_array_free(normed);
        const q = try self.q.forward(normed, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try self.k.forward(normed, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try self.v.forward(normed, s);
        defer _ = mlx.mlx_array_free(v);
        // [1,H,W,C] -> [1, 1, H*W, C] (single head)
        const qr = try reshape(q, &[_]c_int{ 1, 1, H * Wd, C }, s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try reshape(k, &[_]c_int{ 1, 1, H * Wd, C }, s);
        defer _ = mlx.mlx_array_free(kr);
        const vr = try reshape(v, &[_]c_int{ 1, 1, H * Wd, C }, s);
        defer _ = mlx.mlx_array_free(vr);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(C)));
        const attn = try sdpa(qr, kr, vr, scale, s);
        defer _ = mlx.mlx_array_free(attn);
        const ar = try reshape(attn, &[_]c_int{ 1, H, Wd, C }, s);
        defer _ = mlx.mlx_array_free(ar);
        const ao = try self.o.forward(ar, s);
        defer _ = mlx.mlx_array_free(ao);
        return addA(x, ao, s);
    }
};

fn loadResnet(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, s: S) !Resnet {
    return .{
        .n1w = try keyF16(w, a, pfx, "norm1.weight", s),
        .n1b = try keyF16(w, a, pfx, "norm1.bias", s),
        .c1w = try keyF16(w, a, pfx, "conv1.weight", s),
        .c1b = try keyF16(w, a, pfx, "conv1.bias", s),
        .n2w = try keyF16(w, a, pfx, "norm2.weight", s),
        .n2b = try keyF16(w, a, pfx, "norm2.bias", s),
        .c2w = try keyF16(w, a, pfx, "conv2.weight", s),
        .c2b = try keyF16(w, a, pfx, "conv2.bias", s),
        .sw = optF16(w, a, pfx, "conv_shortcut.weight", s),
        .sb = optF16(w, a, pfx, "conv_shortcut.bias", s),
    };
}

fn loadVaeAttn(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, s: S) !VaeAttn {
    const kq = try fmtKey(a, "{s}.query", .{pfx});
    defer a.free(kq);
    const kk = try fmtKey(a, "{s}.key", .{pfx});
    defer a.free(kk);
    const kv = try fmtKey(a, "{s}.value", .{pfx});
    defer a.free(kv);
    const ko = try fmtKey(a, "{s}.proj_attn", .{pfx});
    defer a.free(ko);
    return .{
        .gnw = try keyF16(w, a, pfx, "group_norm.weight", s),
        .gnb = try keyF16(w, a, pfx, "group_norm.bias", s),
        .q = try MixedLinear.load(w, a, kq, VAE_MID_CH, s),
        .k = try MixedLinear.load(w, a, kk, VAE_MID_CH, s),
        .v = try MixedLinear.load(w, a, kv, VAE_MID_CH, s),
        .o = try MixedLinear.load(w, a, ko, VAE_MID_CH, s),
    };
}

pub const SdVae = struct {
    allocator: std.mem.Allocator,
    s: S,
    scaling_factor: f32,
    // encoder
    enc_conv_in_w: mlx.mlx_array,
    enc_conv_in_b: mlx.mlx_array,
    down_resnets: [4][2]Resnet,
    down_conv_w: [3]mlx.mlx_array,
    down_conv_b: [3]mlx.mlx_array,
    enc_mid_r0: Resnet,
    enc_mid_attn: VaeAttn,
    enc_mid_r1: Resnet,
    enc_norm_out_w: mlx.mlx_array,
    enc_norm_out_b: mlx.mlx_array,
    enc_conv_out_w: mlx.mlx_array, // 512 -> 8 (mean+logvar)
    enc_conv_out_b: mlx.mlx_array,
    quant_w: mlx.mlx_array, // 1x1 8->8
    quant_b: mlx.mlx_array,
    // decoder
    post_quant_w: mlx.mlx_array, // 1x1 4->4
    post_quant_b: mlx.mlx_array,
    dec_conv_in_w: mlx.mlx_array, // 4 -> 512
    dec_conv_in_b: mlx.mlx_array,
    dec_mid_r0: Resnet,
    dec_mid_attn: VaeAttn,
    dec_mid_r1: Resnet,
    up_resnets: [4][3]Resnet,
    up_conv_w: [3]mlx.mlx_array,
    up_conv_b: [3]mlx.mlx_array,
    dec_norm_out_w: mlx.mlx_array,
    dec_norm_out_b: mlx.mlx_array,
    dec_conv_out_w: mlx.mlx_array, // 128 -> 3
    dec_conv_out_b: mlx.mlx_array,

    pub fn deinit(self: *SdVae) void {
        inline for (.{ "enc_conv_in_w", "enc_conv_in_b", "enc_norm_out_w", "enc_norm_out_b", "enc_conv_out_w", "enc_conv_out_b", "quant_w", "quant_b", "post_quant_w", "post_quant_b", "dec_conv_in_w", "dec_conv_in_b", "dec_norm_out_w", "dec_norm_out_b", "dec_conv_out_w", "dec_conv_out_b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        for (&self.down_resnets) |*blk| for (blk) |*r| r.deinit();
        for (&self.up_resnets) |*blk| for (blk) |*r| r.deinit();
        self.enc_mid_r0.deinit();
        self.enc_mid_attn.deinit();
        self.enc_mid_r1.deinit();
        self.dec_mid_r0.deinit();
        self.dec_mid_attn.deinit();
        self.dec_mid_r1.deinit();
        for (0..3) |i| {
            _ = mlx.mlx_array_free(self.down_conv_w[i]);
            _ = mlx.mlx_array_free(self.down_conv_b[i]);
            _ = mlx.mlx_array_free(self.up_conv_w[i]);
            _ = mlx.mlx_array_free(self.up_conv_b[i]);
        }
    }

    /// x_nchw [1,3,H,W] f32 in [-1,1] → latent [1,4,H/8,W/8] f32 = posterior
    /// mean · scaling_factor.
    pub fn encode(self: *SdVae, x_nchw: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const t = try transpose(x_nchw, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer _ = mlx.mlx_array_free(t);
        const tc = try contig(t, s);
        defer _ = mlx.mlx_array_free(tc);
        var h = try astype(tc, .float16, s);
        {
            const nh = try conv2d(h, self.enc_conv_in_w, self.enc_conv_in_b, 1, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        for (0..4) |bi| {
            for (0..2) |ri| {
                const nh = try self.down_resnets[bi][ri].forward(h, s);
                _ = mlx.mlx_array_free(h);
                h = nh;
            }
            if (bi < 3) {
                const nh = try conv2dDown(h, self.down_conv_w[bi], self.down_conv_b[bi], s);
                _ = mlx.mlx_array_free(h);
                h = nh;
            }
        }
        {
            const nh = try self.enc_mid_r0.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try self.enc_mid_attn.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try self.enc_mid_r1.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try groupNorm(h, self.enc_norm_out_w, self.enc_norm_out_b, VAE_GROUPS, VAE_GN_EPS, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try silu(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try conv2d(h, self.enc_conv_out_w, self.enc_conv_out_b, 1, s); // -> 8
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try conv2d(h, self.quant_w, self.quant_b, 0, s); // 1x1 8->8
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        // mean = first `latent_channels` channels (NHWC last axis); ×scaling_factor.
        {
            const nh = try sliceAxis(h, 3, 0, @intCast(self.latentChannels()), s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try astype(h, .float32, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const sf = scalarF(self.scaling_factor);
            defer _ = mlx.mlx_array_free(sf);
            const nh = try mulA(h, sf, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try transpose(h, &[_]c_int{ 0, 3, 1, 2 }, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try contig(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        return h;
    }

    /// z_nchw [1,4,h,w] f32 (latents ALREADY including the ·scaling_factor
    /// scale) → image [1,3,H,W] f32 in [-1,1].
    pub fn decode(self: *SdVae, z_nchw: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const inv = scalarF(1.0 / self.scaling_factor);
        defer _ = mlx.mlx_array_free(inv);
        var h = try mulA(z_nchw, inv, s); // f32 / scaling_factor
        {
            const nh = try transpose(h, &[_]c_int{ 0, 2, 3, 1 }, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try contig(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try astype(h, .float16, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try conv2d(h, self.post_quant_w, self.post_quant_b, 0, s); // 1x1 4->4
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try conv2d(h, self.dec_conv_in_w, self.dec_conv_in_b, 1, s); // 4->512
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try self.dec_mid_r0.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try self.dec_mid_attn.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try self.dec_mid_r1.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        for (0..4) |bi| {
            for (0..3) |ri| {
                const nh = try self.up_resnets[bi][ri].forward(h, s);
                _ = mlx.mlx_array_free(h);
                h = nh;
            }
            if (bi < 3) {
                const nh = try upsampleConv(h, self.up_conv_w[bi], self.up_conv_b[bi], s);
                _ = mlx.mlx_array_free(h);
                h = nh;
            }
        }
        {
            const nh = try groupNorm(h, self.dec_norm_out_w, self.dec_norm_out_b, VAE_GROUPS, VAE_GN_EPS, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try silu(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try conv2d(h, self.dec_conv_out_w, self.dec_conv_out_b, 1, s); // ->3
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try transpose(h, &[_]c_int{ 0, 3, 1, 2 }, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try contig(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try astype(h, .float32, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        return h;
    }

    fn latentChannels(self: *const SdVae) u32 {
        _ = self;
        return 4;
    }
};

pub fn loadSdVae(allocator: std.mem.Allocator, cfg: PaintConfig, model_dir: []const u8, s: S) !SdVae {
    var w = try loadFileWeights(allocator, model_dir, "vae.safetensors");
    defer w.deinit();
    var v: SdVae = undefined;
    v.allocator = allocator;
    v.s = s;
    v.scaling_factor = cfg.vae_scaling_factor;

    // encoder
    v.enc_conv_in_w = try ownF16(&w, "encoder.conv_in.weight", s);
    v.enc_conv_in_b = try ownF16(&w, "encoder.conv_in.bias", s);
    for (0..4) |bi| {
        for (0..2) |ri| {
            const pfx = try fmtKey(allocator, "encoder.down_blocks.{d}.resnets.{d}", .{ bi, ri });
            defer allocator.free(pfx);
            v.down_resnets[bi][ri] = try loadResnet(&w, allocator, pfx, s);
        }
    }
    for (0..3) |bi| {
        const wk = try fmtKey(allocator, "encoder.down_blocks.{d}.downsamplers.0.conv.weight", .{bi});
        defer allocator.free(wk);
        const bk = try fmtKey(allocator, "encoder.down_blocks.{d}.downsamplers.0.conv.bias", .{bi});
        defer allocator.free(bk);
        v.down_conv_w[bi] = try ownF16(&w, wk, s);
        v.down_conv_b[bi] = try ownF16(&w, bk, s);
    }
    v.enc_mid_r0 = try loadResnet(&w, allocator, "encoder.mid_block.resnets.0", s);
    v.enc_mid_attn = try loadVaeAttn(&w, allocator, "encoder.mid_block.attentions.0", s);
    v.enc_mid_r1 = try loadResnet(&w, allocator, "encoder.mid_block.resnets.1", s);
    v.enc_norm_out_w = try ownF16(&w, "encoder.conv_norm_out.weight", s);
    v.enc_norm_out_b = try ownF16(&w, "encoder.conv_norm_out.bias", s);
    v.enc_conv_out_w = try ownF16(&w, "encoder.conv_out.weight", s);
    v.enc_conv_out_b = try ownF16(&w, "encoder.conv_out.bias", s);
    v.quant_w = try ownF16(&w, "quant_conv.weight", s);
    v.quant_b = try ownF16(&w, "quant_conv.bias", s);

    // decoder
    v.post_quant_w = try ownF16(&w, "post_quant_conv.weight", s);
    v.post_quant_b = try ownF16(&w, "post_quant_conv.bias", s);
    v.dec_conv_in_w = try ownF16(&w, "decoder.conv_in.weight", s);
    v.dec_conv_in_b = try ownF16(&w, "decoder.conv_in.bias", s);
    v.dec_mid_r0 = try loadResnet(&w, allocator, "decoder.mid_block.resnets.0", s);
    v.dec_mid_attn = try loadVaeAttn(&w, allocator, "decoder.mid_block.attentions.0", s);
    v.dec_mid_r1 = try loadResnet(&w, allocator, "decoder.mid_block.resnets.1", s);
    for (0..4) |bi| {
        for (0..3) |ri| {
            const pfx = try fmtKey(allocator, "decoder.up_blocks.{d}.resnets.{d}", .{ bi, ri });
            defer allocator.free(pfx);
            v.up_resnets[bi][ri] = try loadResnet(&w, allocator, pfx, s);
        }
    }
    for (0..3) |bi| {
        const wk = try fmtKey(allocator, "decoder.up_blocks.{d}.upsamplers.0.conv.weight", .{bi});
        defer allocator.free(wk);
        const bk = try fmtKey(allocator, "decoder.up_blocks.{d}.upsamplers.0.conv.bias", .{bi});
        defer allocator.free(bk);
        v.up_conv_w[bi] = try ownF16(&w, wk, s);
        v.up_conv_b[bi] = try ownF16(&w, bk, s);
    }
    v.dec_norm_out_w = try ownF16(&w, "decoder.conv_norm_out.weight", s);
    v.dec_norm_out_b = try ownF16(&w, "decoder.conv_norm_out.bias", s);
    v.dec_conv_out_w = try ownF16(&w, "decoder.conv_out.weight", s);
    v.dec_conv_out_b = try ownF16(&w, "decoder.conv_out.bias", s);
    log.info("[hy3d-paint] SD2.x VAE ready (scaling_factor {d})\n", .{cfg.vae_scaling_factor});
    return v;
}

// ════════════════════════════════════════════════════════════════════════
// Image preprocessing (DINOv2-giant transform, contract §7 tap point). The
// oracle taps AFTER this stage (fixtures are post-preprocess pixels), so the
// resize kernel is decoupled from parity.
// ════════════════════════════════════════════════════════════════════════

const IMAGENET_MEAN = [3]f32{ 0.485, 0.456, 0.406 };
const IMAGENET_STD = [3]f32{ 0.229, 0.224, 0.225 };
const DINO_RESIZE_SHORT: usize = 256;
const DINO_CROP: usize = 224;

/// Bilinear resize of HWC f32 (align_corners=False). Constant-preserving.
fn bilinearResize(src: []const f32, sw: usize, sh: usize, dst: []f32, dw: usize, dh: usize) void {
    const fx = @as(f32, @floatFromInt(sw)) / @as(f32, @floatFromInt(dw));
    const fy = @as(f32, @floatFromInt(sh)) / @as(f32, @floatFromInt(dh));
    for (0..dh) |oy| {
        var syf = (@as(f32, @floatFromInt(oy)) + 0.5) * fy - 0.5;
        if (syf < 0) syf = 0;
        const y0 = @min(@as(usize, @intFromFloat(@floor(syf))), sh - 1);
        const y1 = @min(y0 + 1, sh - 1);
        const wy = syf - @floor(syf);
        for (0..dw) |ox| {
            var sxf = (@as(f32, @floatFromInt(ox)) + 0.5) * fx - 0.5;
            if (sxf < 0) sxf = 0;
            const x0 = @min(@as(usize, @intFromFloat(@floor(sxf))), sw - 1);
            const x1 = @min(x0 + 1, sw - 1);
            const wx = sxf - @floor(sxf);
            inline for (0..3) |c| {
                const p00 = src[(y0 * sw + x0) * 3 + c];
                const p01 = src[(y0 * sw + x1) * 3 + c];
                const p10 = src[(y1 * sw + x0) * 3 + c];
                const p11 = src[(y1 * sw + x1) * 3 + c];
                const top = p00 * (1 - wx) + p01 * wx;
                const bot = p10 * (1 - wx) + p11 * wx;
                dst[(oy * dw + ox) * 3 + c] = top * (1 - wy) + bot * wy;
            }
        }
    }
}

/// Straight-alpha RGBA8 → CHW f32 [1,3,224,224], ImageNet-normalized: composite
/// over white, resize shorter side to 256 (bilinear), center-crop 224, then
/// (x-mean)/std per channel. Caller frees.
pub fn preprocessImagePaint(allocator: std.mem.Allocator, rgba: []const u8, w: u32, h: u32) ![]f32 {
    if (rgba.len < @as(usize, w) * h * 4 or w == 0 or h == 0) return error.BadImage;
    const sw: usize = w;
    const sh: usize = h;
    // composite over white → HWC f32 [0,1]
    const rgb = try allocator.alloc(f32, sw * sh * 3);
    defer allocator.free(rgb);
    for (0..sh) |y| {
        for (0..sw) |x| {
            const idx = (y * sw + x) * 4;
            const a: f32 = @as(f32, @floatFromInt(rgba[idx + 3])) / 255.0;
            inline for (0..3) |c| {
                const v: f32 = @as(f32, @floatFromInt(rgba[idx + c])) / 255.0;
                rgb[(y * sw + x) * 3 + c] = v * a + (1.0 - a);
            }
        }
    }
    // resize shorter side to 256, preserve aspect (floor at the crop size).
    const scale: f32 = @as(f32, @floatFromInt(DINO_RESIZE_SHORT)) / @as(f32, @floatFromInt(@min(sw, sh)));
    const rw: usize = @max(@as(usize, @intFromFloat(@round(@as(f32, @floatFromInt(sw)) * scale))), DINO_CROP);
    const rh: usize = @max(@as(usize, @intFromFloat(@round(@as(f32, @floatFromInt(sh)) * scale))), DINO_CROP);
    const resized = try allocator.alloc(f32, rw * rh * 3);
    defer allocator.free(resized);
    bilinearResize(rgb, sw, sh, resized, rw, rh);
    // center-crop 224 → CHW, ImageNet norm
    const y0 = (rh - DINO_CROP) / 2;
    const x0 = (rw - DINO_CROP) / 2;
    const out = try allocator.alloc(f32, 3 * DINO_CROP * DINO_CROP);
    for (0..DINO_CROP) |y| {
        for (0..DINO_CROP) |x| {
            inline for (0..3) |c| {
                const v = resized[((y0 + y) * rw + (x0 + x)) * 3 + c];
                out[c * DINO_CROP * DINO_CROP + y * DINO_CROP + x] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Bicubic position-embedding resample (separable, align_corners=False,
// a=-0.75). DINOv2-giant stores pos_embed for a 37×37 patch grid (518 input);
// a 224 input yields a 16×16 grid, so the patch pos-embeds are interpolated
// (HF Dinov2 interpolate_pos_encoding). Identity when out==in.
//
// PARITY NOTE for P2-13: this is bicubic WITHOUT antialias, matching the
// `size=`-target HF path. If DINO-feature parity misses on the downsample, the
// suspects are (a) HF's older `scale_factor`+`+0.1` variant, (b) antialias=True
// — both localized to buildBicubicMatrix.
// ════════════════════════════════════════════════════════════════════════

fn cubicKernel(x: f32) f32 {
    const a: f32 = -0.75;
    const ax = @abs(x);
    if (ax <= 1.0) return ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0;
    if (ax < 2.0) return (((ax - 5.0) * ax + 8.0) * ax - 4.0) * a;
    return 0.0;
}

/// Row-major [out_n, in_n] separable bicubic resample matrix. Each row sums to
/// 1 (cubic-convolution weights sum to 1); identity when out_n == in_n. Caller
/// frees.
fn buildBicubicMatrix(allocator: std.mem.Allocator, out_n: usize, in_n: usize) ![]f32 {
    const m = try allocator.alloc(f32, out_n * in_n);
    @memset(m, 0);
    const scale: f32 = @as(f32, @floatFromInt(in_n)) / @as(f32, @floatFromInt(out_n));
    for (0..out_n) |o| {
        const src = (@as(f32, @floatFromInt(o)) + 0.5) * scale - 0.5;
        const ifloor = @floor(src);
        const base: i64 = @intFromFloat(ifloor);
        const t = src - ifloor;
        const wts = [4]f32{ cubicKernel(1.0 + t), cubicKernel(t), cubicKernel(1.0 - t), cubicKernel(2.0 - t) };
        for (0..4) |k| {
            var idx: i64 = base - 1 + @as(i64, @intCast(k));
            if (idx < 0) idx = 0;
            if (idx > @as(i64, @intCast(in_n)) - 1) idx = @as(i64, @intCast(in_n)) - 1;
            m[o * in_n + @as(usize, @intCast(idx))] += wts[k];
        }
    }
    return m;
}

// ════════════════════════════════════════════════════════════════════════
// DINOv2-giant image encoder (40 layers, hidden 1536, 24 heads, SwiGLU FFN,
// LayerScale, qkv-bias). Output [1, grid²+1, 1536] (both CLS + patch tokens).
// ════════════════════════════════════════════════════════════════════════

/// SwiGLU: h [.., 2·inner] → silu(FIRST half) · SECOND half. The activation is
/// on the FIRST half (HF Dinov2SwiGLUFFN — the OPPOSITE half of GeGLU).
fn swiglu(h: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(h);
    const last = sh.len - 1;
    const total = sh[last];
    const inner = @divExact(total, 2);
    const x1 = try sliceAxis(h, last, 0, inner, s); // FIRST half → SiLU
    defer _ = mlx.mlx_array_free(x1);
    const x2 = try sliceAxis(h, last, inner, total, s); // SECOND half → gate
    defer _ = mlx.mlx_array_free(x2);
    const a = try silu(x1, s);
    defer _ = mlx.mlx_array_free(a);
    return mulA(a, x2, s);
}

const PaintDinoLayer = struct {
    ln1_w: mlx.mlx_array,
    ln1_b: mlx.mlx_array,
    q: MixedLinear,
    k: MixedLinear,
    v: MixedLinear,
    o: MixedLinear,
    ls1: mlx.mlx_array,
    ln2_w: mlx.mlx_array,
    ln2_b: mlx.mlx_array,
    w_in: MixedLinear,
    w_out: MixedLinear,
    ls2: mlx.mlx_array,
    fn deinit(self: *PaintDinoLayer) void {
        _ = mlx.mlx_array_free(self.ln1_w);
        _ = mlx.mlx_array_free(self.ln1_b);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        _ = mlx.mlx_array_free(self.ls1);
        _ = mlx.mlx_array_free(self.ln2_w);
        _ = mlx.mlx_array_free(self.ln2_b);
        self.w_in.deinit();
        self.w_out.deinit();
        _ = mlx.mlx_array_free(self.ls2);
    }
};

pub const DinoGiant = struct {
    cfg: PaintConfig,
    allocator: std.mem.Allocator,
    s: S,
    cls_token: mlx.mlx_array, // [1,1,1536] f16
    pos_embed: mlx.mlx_array, // [1,1370,1536] f16
    patch_w: mlx.mlx_array, // [1536,14,14,3] f16 (OHWI)
    patch_b: mlx.mlx_array, // [1536] f16
    layers: []PaintDinoLayer,
    norm_w: mlx.mlx_array,
    norm_b: mlx.mlx_array,

    pub fn deinit(self: *DinoGiant) void {
        _ = mlx.mlx_array_free(self.cls_token);
        _ = mlx.mlx_array_free(self.pos_embed);
        _ = mlx.mlx_array_free(self.patch_w);
        _ = mlx.mlx_array_free(self.patch_b);
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        _ = mlx.mlx_array_free(self.norm_w);
        _ = mlx.mlx_array_free(self.norm_b);
    }

    /// pixels [1,3,S,S] f32 (post-preprocess) → features [1, (S/14)²+1, 1536] f16.
    pub fn encode(self: *DinoGiant, pixels: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const c = self.cfg;
        const heads: c_int = @intCast(c.dino_heads);
        const hd: c_int = @intCast(c.dinoHeadDim());
        const H: c_int = @intCast(c.dino_hidden);
        const side_u: u32 = @intCast(mlx.getShape(pixels)[2]);
        const grid: u32 = side_u / c.dino_patch;
        const gc: c_int = @intCast(grid);
        const p: c_int = @intCast(c.dino_patch);

        // NCHW → NHWC f16, patch conv (stride = patch) → [1, grid², H].
        const nhwc0 = try transpose(pixels, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer _ = mlx.mlx_array_free(nhwc0);
        const nhwc1 = try contig(nhwc0, s);
        defer _ = mlx.mlx_array_free(nhwc1);
        const nhwc = try astype(nhwc1, .float16, s);
        defer _ = mlx.mlx_array_free(nhwc);
        var convd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(convd);
        try mlx.check(mlx.mlx_conv2d(&convd, nhwc, self.patch_w, p, p, 0, 0, 1, 1, 1, s));
        const flat = try reshape(convd, &[_]c_int{ 1, gc * gc, H }, s);
        defer _ = mlx.mlx_array_free(flat);
        const patched = try addA(flat, self.patch_b, s);
        defer _ = mlx.mlx_array_free(patched);

        // CLS first, + (interpolated) learned pos embed.
        const with_cls = try concat(&[_]mlx.mlx_array{ self.cls_token, patched }, 1, s);
        defer _ = mlx.mlx_array_free(with_cls);
        const pos = try self.interpolatePosEmbed(grid);
        defer _ = mlx.mlx_array_free(pos);
        var x = try addA(with_cls, pos, s);

        for (self.layers) |*layer| {
            const nx = try self.layerForward(x, layer, heads, hd, s);
            _ = mlx.mlx_array_free(x);
            x = nx;
        }
        defer _ = mlx.mlx_array_free(x);
        return layerNorm(x, self.norm_w, self.norm_b, PAINT_LN_EPS, s);
    }

    fn layerForward(self: *DinoGiant, x: mlx.mlx_array, layer: *const PaintDinoLayer, heads: c_int, hd: c_int, s: S) !mlx.mlx_array {
        _ = self;
        const n1 = try layerNorm(x, layer.ln1_w, layer.ln1_b, PAINT_LN_EPS, s);
        defer _ = mlx.mlx_array_free(n1);
        const q0 = try layer.q.forward(n1, s);
        defer _ = mlx.mlx_array_free(q0);
        const k0 = try layer.k.forward(n1, s);
        defer _ = mlx.mlx_array_free(k0);
        const v0 = try layer.v.forward(n1, s);
        defer _ = mlx.mlx_array_free(v0);
        const q = try splitHeads(q0, heads, hd, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try splitHeads(k0, heads, hd, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try splitHeads(v0, heads, hd, s);
        defer _ = mlx.mlx_array_free(v);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(@as(u32, @intCast(hd)))));
        const attn = try sdpa(q, k, v, scale, s);
        defer _ = mlx.mlx_array_free(attn);
        const merged = try mergeHeads(attn, s);
        defer _ = mlx.mlx_array_free(merged);
        const o = try layer.o.forward(merged, s);
        defer _ = mlx.mlx_array_free(o);
        const o_ls = try mulA(o, layer.ls1, s);
        defer _ = mlx.mlx_array_free(o_ls);
        const h1 = try addA(x, o_ls, s);
        defer _ = mlx.mlx_array_free(h1);
        const n2 = try layerNorm(h1, layer.ln2_w, layer.ln2_b, PAINT_LN_EPS, s);
        defer _ = mlx.mlx_array_free(n2);
        const f1 = try layer.w_in.forward(n2, s);
        defer _ = mlx.mlx_array_free(f1);
        const g = try swiglu(f1, s);
        defer _ = mlx.mlx_array_free(g);
        const f2 = try layer.w_out.forward(g, s);
        defer _ = mlx.mlx_array_free(f2);
        const f_ls = try mulA(f2, layer.ls2, s);
        defer _ = mlx.mlx_array_free(f_ls);
        return addA(h1, f_ls, s);
    }

    /// Interpolate the stored [1,pg²+1,1536] pos-embed to a `grid`×`grid` patch
    /// grid + CLS → [1, grid²+1, 1536] f16. Identity fast-path when grid == pg.
    fn interpolatePosEmbed(self: *DinoGiant, grid: u32) !mlx.mlx_array {
        const s = self.s;
        const pg = self.cfg.posEmbedGrid();
        if (grid == pg) {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&o, self.pos_embed));
            return o;
        }
        const Hdim: c_int = @intCast(self.cfg.dino_hidden);
        const pgc: c_int = @intCast(pg);
        const gc: c_int = @intCast(grid);
        const cls_pos = try sliceAxis(self.pos_embed, 1, 0, 1, s); // [1,1,H] f16
        defer _ = mlx.mlx_array_free(cls_pos);
        const patch_pos = try sliceAxis(self.pos_embed, 1, 1, 1 + pgc * pgc, s); // [1,pg²,H]
        defer _ = mlx.mlx_array_free(patch_pos);
        const pf = try astype(patch_pos, .float32, s);
        defer _ = mlx.mlx_array_free(pf);
        const pr = try reshape(pf, &[_]c_int{ pgc, pgc, Hdim }, s); // [pg,pg,H]
        defer _ = mlx.mlx_array_free(pr);
        const pt = try transpose(pr, &[_]c_int{ 2, 0, 1 }, s); // [H,pg,pg]
        defer _ = mlx.mlx_array_free(pt);
        const ptc = try contig(pt, s);
        defer _ = mlx.mlx_array_free(ptc);
        // resample matrix W [grid, pg] f32 (mlx_array_new_data copies the buffer).
        const wbuf = try buildBicubicMatrix(self.allocator, grid, pg);
        defer self.allocator.free(wbuf);
        const wsh = [_]c_int{ gc, pgc };
        const wmat = mlx.mlx_array_new_data(wbuf.ptr, &wsh, 2, .float32);
        defer _ = mlx.mlx_array_free(wmat);
        const wt = try transpose(wmat, &[_]c_int{ 1, 0 }, s); // [pg, grid]
        defer _ = mlx.mlx_array_free(wt);
        const tmp = try matmul(wmat, ptc, s); // [H, grid, pg]
        defer _ = mlx.mlx_array_free(tmp);
        const out = try matmul(tmp, wt, s); // [H, grid, grid]
        defer _ = mlx.mlx_array_free(out);
        const outt = try transpose(out, &[_]c_int{ 1, 2, 0 }, s); // [grid,grid,H]
        defer _ = mlx.mlx_array_free(outt);
        const outc = try contig(outt, s);
        defer _ = mlx.mlx_array_free(outc);
        const outr = try reshape(outc, &[_]c_int{ 1, gc * gc, Hdim }, s); // [1,grid²,H]
        defer _ = mlx.mlx_array_free(outr);
        const outh = try astype(outr, .float16, s);
        defer _ = mlx.mlx_array_free(outh);
        return concat(&[_]mlx.mlx_array{ cls_pos, outh }, 1, s); // [1, grid²+1, H]
    }
};

pub fn loadDinoGiant(allocator: std.mem.Allocator, cfg: PaintConfig, model_dir: []const u8, s: S) !DinoGiant {
    var w = try loadFileWeights(allocator, model_dir, "dino.safetensors");
    defer w.deinit();
    var d: DinoGiant = undefined;
    d.cfg = cfg;
    d.allocator = allocator;
    d.s = s;
    d.cls_token = try ownF16(&w, "cls_token", s);
    d.pos_embed = try ownF16(&w, "pos_embed", s);
    if (mlx.getShape(d.pos_embed)[1] != @as(c_int, @intCast(cfg.posEmbedTokens()))) {
        log.err("[hy3d-paint] DINO pos_embed token count {d} != expected {d}\n", .{ mlx.getShape(d.pos_embed)[1], cfg.posEmbedTokens() });
        return error.BadPaintWeights;
    }
    d.patch_w = try ownF16(&w, "patch_embed.weight", s); // already OHWI in the converted layout
    d.patch_b = try ownF16(&w, "patch_embed.bias", s);
    d.layers = try allocator.alloc(PaintDinoLayer, cfg.dino_layers);
    const H = cfg.dino_hidden;
    const inter = cfg.dino_intermediate; // 4096; w_out in-features
    for (d.layers, 0..) |*layer, i| {
        const pfx = try fmtKey(allocator, "layers.{d}", .{i});
        defer allocator.free(pfx);
        const kq = try fmtKey(allocator, "{s}.attn.q", .{pfx});
        defer allocator.free(kq);
        const kk = try fmtKey(allocator, "{s}.attn.k", .{pfx});
        defer allocator.free(kk);
        const kv = try fmtKey(allocator, "{s}.attn.v", .{pfx});
        defer allocator.free(kv);
        const ko = try fmtKey(allocator, "{s}.attn.out", .{pfx});
        defer allocator.free(ko);
        const kwi = try fmtKey(allocator, "{s}.mlp.w_in", .{pfx});
        defer allocator.free(kwi);
        const kwo = try fmtKey(allocator, "{s}.mlp.w_out", .{pfx});
        defer allocator.free(kwo);
        layer.* = .{
            .ln1_w = try normVecKey(&w, allocator, pfx, "norm1.weight", s),
            .ln1_b = try normVecKey(&w, allocator, pfx, "norm1.bias", s),
            .q = try MixedLinear.load(&w, allocator, kq, H, s),
            .k = try MixedLinear.load(&w, allocator, kk, H, s),
            .v = try MixedLinear.load(&w, allocator, kv, H, s),
            .o = try MixedLinear.load(&w, allocator, ko, H, s),
            .ls1 = try keyF16(&w, allocator, pfx, "ls1", s),
            .ln2_w = try normVecKey(&w, allocator, pfx, "norm2.weight", s),
            .ln2_b = try normVecKey(&w, allocator, pfx, "norm2.bias", s),
            .w_in = try MixedLinear.load(&w, allocator, kwi, H, s),
            .w_out = try MixedLinear.load(&w, allocator, kwo, inter, s),
            .ls2 = try keyF16(&w, allocator, pfx, "ls2", s),
        };
    }
    d.norm_w = try normVec(&w, "norm.weight", s);
    d.norm_b = try normVec(&w, "norm.bias", s);
    log.info("[hy3d-paint] DINOv2-giant conditioner ready ({d} layers)\n", .{cfg.dino_layers});
    return d;
}

// ════════════════════════════════════════════════════════════════════════
// ImageProjModel — DINO features [1,T,1536] → proj (1536→4·1024) → reshape
// [1, T·4, 1024] → LayerNorm(1024). Weights live in unet.safetensors
// (image_proj_model_dino.{proj,norm}). num_context_tokens = 4.
// ════════════════════════════════════════════════════════════════════════

pub const ImageProjModel = struct {
    proj: MixedLinear,
    norm_w: mlx.mlx_array,
    norm_b: mlx.mlx_array,
    num_tokens: u32,
    cross_dim: u32,

    pub fn deinit(self: *ImageProjModel) void {
        self.proj.deinit();
        _ = mlx.mlx_array_free(self.norm_w);
        _ = mlx.mlx_array_free(self.norm_b);
    }

    /// feats [1,T,clip_dim] → context [1, T·num_tokens, cross_dim] f16.
    /// nn.LayerNorm default eps 1e-5.
    pub fn forward(self: *const ImageProjModel, feats: mlx.mlx_array, s: S) !mlx.mlx_array {
        const proj = try self.proj.forward(feats, s); // [1,T, num_tokens·cross_dim]
        defer _ = mlx.mlx_array_free(proj);
        const sh = mlx.getShape(proj);
        const T: c_int = sh[1];
        const cross: c_int = @intCast(self.cross_dim);
        const ctx = try reshape(proj, &[_]c_int{ 1, T * @as(c_int, @intCast(self.num_tokens)), cross }, s);
        defer _ = mlx.mlx_array_free(ctx);
        return layerNorm(ctx, self.norm_w, self.norm_b, 1e-5, s);
    }
};

/// Loads the ImageProjModel out of unet.safetensors (§4a). NOTE: this loads the
/// whole UNet file to extract 4 tensors — heavy but correct; the shared
/// conditioner is env-gated (oracle) / one-shot (paint run) so it's acceptable.
pub fn loadImageProj(allocator: std.mem.Allocator, cfg: PaintConfig, model_dir: []const u8, s: S) !ImageProjModel {
    var w = try loadFileWeights(allocator, model_dir, "unet.safetensors");
    defer w.deinit();
    return .{
        .proj = try MixedLinear.load(&w, allocator, "image_proj_model_dino.proj", cfg.imageproj_clip_dim, s),
        .norm_w = try normVec(&w, "image_proj_model_dino.norm.weight", s),
        .norm_b = try normVec(&w, "image_proj_model_dino.norm.bias", s),
        .num_tokens = cfg.imageproj_num_tokens,
        .cross_dim = cfg.imageproj_cross_dim,
    };
}

/// The paint image conditioner: DINOv2-giant + ImageProjModel. `encodeDino`
/// returns BOTH the raw DINO features [1,257,1536] and the projected cross-attn
/// context [1,1028,1024] (caller frees both).
pub const PaintConditioner = struct {
    dino: DinoGiant,
    proj: ImageProjModel,
    s: S,

    pub fn deinit(self: *PaintConditioner) void {
        self.dino.deinit();
        self.proj.deinit();
    }

    pub const DinoOut = struct { feats: mlx.mlx_array, context: mlx.mlx_array };

    /// pixels [1,3,S,S] f32 (post-preprocess) → { feats, context }.
    pub fn encodeDino(self: *PaintConditioner, pixels: mlx.mlx_array) !DinoOut {
        const feats = try self.dino.encode(pixels);
        errdefer _ = mlx.mlx_array_free(feats);
        const context = try self.proj.forward(feats, self.s);
        return .{ .feats = feats, .context = context };
    }
};

pub fn loadPaintConditioner(allocator: std.mem.Allocator, cfg: PaintConfig, model_dir: []const u8, s: S) !PaintConditioner {
    var dino = try loadDinoGiant(allocator, cfg, model_dir, s);
    errdefer dino.deinit();
    const proj = try loadImageProj(allocator, cfg, model_dir, s);
    return .{ .dino = dino, .proj = proj, .s = s };
}

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic (no weights) + env-gated cos oracles (fixtures from
// tests/dump_hunyuan3d_paint_fixtures.py against the fp16 build).
// ════════════════════════════════════════════════════════════════════════

test "hy3d-paint PaintConfig parse reads nested vae/dino/image_proj" {
    const a = testing.allocator;
    const cfg = try parseConfigText(a,
        \\{"model_type":"hunyuan3d_2_1_paint","quant":"fp16",
        \\ "vae":{"in_channels":3,"latent_channels":4,"norm_num_groups":32,
        \\        "block_out_channels":[128,256,512,512],"scaling_factor":0.18215},
        \\ "dino":{"hidden_size":1536,"num_layers":40,"num_heads":24,"head_dim":64,
        \\         "patch_size":14,"image_size":518,"num_tokens":1370,"mlp":"swiglu",
        \\         "intermediate_size":4096,"layer_norm_eps":1e-06,"qkv_bias":true},
        \\ "image_proj":{"clip_embeddings_dim":1536,"cross_attention_dim":1024,
        \\               "num_context_tokens":4}}
    );
    try testing.expectApproxEqAbs(@as(f32, 0.18215), cfg.vae_scaling_factor, 1e-9);
    try testing.expectEqual(@as(u32, 4), cfg.vae_latent_channels);
    try testing.expectEqual(@as(u32, 1536), cfg.dino_hidden);
    try testing.expectEqual(@as(u32, 40), cfg.dino_layers);
    try testing.expectEqual(@as(u32, 24), cfg.dino_heads);
    try testing.expectEqual(@as(u32, 64), cfg.dinoHeadDim());
    try testing.expectEqual(@as(u32, 14), cfg.dino_patch);
    try testing.expectEqual(@as(u32, 4096), cfg.dino_intermediate);
    try testing.expectEqual(@as(u32, 37), cfg.posEmbedGrid());
    try testing.expectEqual(@as(u32, 1370), cfg.posEmbedTokens());
    try testing.expectEqual(@as(u32, 257), cfg.dinoTokensFor(224));
    try testing.expectEqual(@as(u32, 4), cfg.imageproj_num_tokens);
    try testing.expectEqual(@as(u32, 1024), cfg.imageproj_cross_dim);
    // Wrong model_type must be rejected.
    try testing.expectError(error.BadPaintConfig, parseConfigText(a,
        \\{"model_type":"flux2"}
    ));
    // Defaults when nested objects are absent.
    const bare = try parseConfigText(a,
        \\{"model_type":"hunyuan3d_2_1_paint"}
    );
    try testing.expectEqual(@as(u32, 40), bare.dino_layers);
    try testing.expectApproxEqAbs(@as(f32, 0.18215), bare.vae_scaling_factor, 1e-9);
}

test "hy3d-paint SwiGLU applies SiLU to the FIRST half (opposite of GeGLU)" {
    const s = mlx.mlx_default_gpu_stream_new();
    // h = [x1_0, x1_1, x2_0, x2_1]; out = silu(x1) * x2 (x1 = first half).
    const hv = [_]f32{ 1.0, -2.0, 3.0, 5.0 };
    const hsh = [_]c_int{ 1, 1, 4 };
    const h = mlx.mlx_array_new_data(&hv, &hsh, 3, .float32);
    defer _ = mlx.mlx_array_free(h);
    const out = try swiglu(h, s);
    defer _ = mlx.mlx_array_free(out);
    const of = try astype(out, .float32, s);
    defer _ = mlx.mlx_array_free(of);
    _ = mlx.mlx_array_eval(of);
    try testing.expectEqual(@as(usize, 2), @as(usize, @intCast(mlx.mlx_array_size(of))));
    const d = mlx.mlx_array_data_float32(of) orelse return error.NoData;
    const sig = struct {
        fn f(x: f32) f32 {
            return x / (1.0 + @exp(-x));
        }
    }.f;
    // FIRST half {1,-2} gets SiLU, SECOND half {3,5} multiplies.
    try testing.expectApproxEqAbs(sig(1.0) * 3.0, d[0], 1e-3);
    try testing.expectApproxEqAbs(sig(-2.0) * 5.0, d[1], 1e-3);
    // A swapped (SiLU-on-second-half) impl would give silu(3)*1 for d[0] — a
    // clearly different value, so the split order is pinned (the exact asserts
    // above already catch a swap; this documents the distinguishing margin).
    try testing.expect(@abs(d[0] - sig(3.0) * 1.0) > 0.3);
}

test "hy3d-paint groupNorm matches hand-computed per-group normalization" {
    const s = mlx.mlx_default_gpu_stream_new();
    const a = testing.allocator;
    // NHWC [1,2,2,4], groups=2 (cg=2). Group g normalizes over 4 spatial × 2 ch.
    const N = 2 * 2 * 4;
    const xv = try a.alloc(f32, N);
    defer a.free(xv);
    for (xv, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % 7)) * 0.5 - 1.0 + @as(f32, @floatFromInt(i)) * 0.13;
    const xsh = [_]c_int{ 1, 2, 2, 4 };
    const x = mlx.mlx_array_new_data(xv.ptr, &xsh, 4, .float32);
    defer _ = mlx.mlx_array_free(x);
    const ones = [_]f32{ 1, 1, 1, 1 };
    const zeros = [_]f32{ 0, 0, 0, 0 };
    const csh = [_]c_int{4};
    const wv = mlx.mlx_array_new_data(&ones, &csh, 1, .float32);
    defer _ = mlx.mlx_array_free(wv);
    const bv = mlx.mlx_array_new_data(&zeros, &csh, 1, .float32);
    defer _ = mlx.mlx_array_free(bv);
    const gn = try groupNorm(x, wv, bv, 2, VAE_GN_EPS, s);
    defer _ = mlx.mlx_array_free(gn);
    const gnf = try astype(gn, .float32, s);
    defer _ = mlx.mlx_array_free(gnf);
    _ = mlx.mlx_array_eval(gnf);
    const gd = mlx.mlx_array_data_float32(gnf) orelse return error.NoData;

    // Reference: for each group, collect its 8 values, mean/var (population),
    // normalize (weight 1, bias 0). Index (p,c) = p*4 + c; group = c/2.
    for (0..2) |g| {
        var sum: f64 = 0;
        for (0..4) |p| for (0..2) |cc| {
            sum += xv[p * 4 + g * 2 + cc];
        };
        const mean: f64 = sum / 8.0;
        var vs: f64 = 0;
        for (0..4) |p| for (0..2) |cc| {
            const dd = xv[p * 4 + g * 2 + cc] - mean;
            vs += dd * dd;
        };
        const variance: f64 = vs / 8.0;
        const inv = 1.0 / std.math.sqrt(variance + VAE_GN_EPS);
        for (0..4) |p| for (0..2) |cc| {
            const expect: f32 = @floatCast((xv[p * 4 + g * 2 + cc] - mean) * inv);
            try testing.expectApproxEqAbs(expect, gd[p * 4 + g * 2 + cc], 3e-3);
        };
    }
}

test "hy3d-paint encoder downsample uses asymmetric (0,1,0,1) pad on odd dims" {
    const s = mlx.mlx_default_gpu_stream_new();
    const a = testing.allocator;
    // Odd input side 15: asymmetric down-right pad → 7 (a symmetric pad-1 → 8).
    const Cc = 4;
    const N = 15;
    const xv = try a.alloc(f32, 1 * N * N * Cc);
    defer a.free(xv);
    for (xv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) * 0.1;
    const xsh = [_]c_int{ 1, N, N, Cc };
    const x = mlx.mlx_array_new_data(xv.ptr, &xsh, 4, .float32);
    defer _ = mlx.mlx_array_free(x);
    const wv = try a.alloc(f32, Cc * 3 * 3 * Cc); // OHWI [4,3,3,4]
    defer a.free(wv);
    for (wv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 3)) * 0.05;
    const wsh = [_]c_int{ Cc, 3, 3, Cc };
    const wa = mlx.mlx_array_new_data(wv.ptr, &wsh, 4, .float32);
    defer _ = mlx.mlx_array_free(wa);
    const bz = [_]f32{ 0, 0, 0, 0 };
    const bsh = [_]c_int{Cc};
    const ba = mlx.mlx_array_new_data(&bz, &bsh, 1, .float32);
    defer _ = mlx.mlx_array_free(ba);
    const out = try conv2dDown(x, wa, ba, s);
    defer _ = mlx.mlx_array_free(out);
    _ = mlx.mlx_array_eval(out);
    const osh = mlx.getShape(out);
    try testing.expectEqual(@as(c_int, 1), osh[0]);
    try testing.expectEqual(@as(c_int, 7), osh[1]); // NOT 8 — asymmetric pad
    try testing.expectEqual(@as(c_int, 7), osh[2]);
    try testing.expectEqual(@as(c_int, Cc), osh[3]);
}

test "hy3d-paint preprocessImagePaint shape + ImageNet-normalized solid color" {
    const a = testing.allocator;
    // Solid opaque gray 300x300 (alpha 255 → compositing is identity).
    const w: u32 = 300;
    const h: u32 = 260;
    const rgba = try a.alloc(u8, @as(usize, w) * h * 4);
    defer a.free(rgba);
    for (0..@as(usize, w) * h) |i| {
        rgba[i * 4 + 0] = 128;
        rgba[i * 4 + 1] = 128;
        rgba[i * 4 + 2] = 128;
        rgba[i * 4 + 3] = 255;
    }
    const out = try preprocessImagePaint(a, rgba, w, h);
    defer a.free(out);
    try testing.expectEqual(@as(usize, 3 * 224 * 224), out.len);
    const g: f32 = 128.0 / 255.0;
    // Center pixel per channel = (g - mean)/std (constant image → uniform).
    const cy = 112;
    const cx = 112;
    inline for (0..3) |c| {
        const expect = (g - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        try testing.expectApproxEqAbs(expect, out[c * 224 * 224 + cy * 224 + cx], 1e-4);
    }
}

test "hy3d-paint bicubic matrix is identity for equal grids, row-normalized for downsample" {
    const a = testing.allocator;
    // Identity when out == in (align_corners=False, integer sample points).
    const eye = try buildBicubicMatrix(a, 37, 37);
    defer a.free(eye);
    for (0..37) |o| {
        for (0..37) |i| {
            const expect: f32 = if (o == i) 1.0 else 0.0;
            try testing.expectApproxEqAbs(expect, eye[o * 37 + i], 1e-5);
        }
    }
    // Downsample 37 → 16: every row sums to ~1 (cubic weights sum to 1).
    const down = try buildBicubicMatrix(a, 16, 37);
    defer a.free(down);
    for (0..16) |o| {
        var sum: f32 = 0;
        for (0..37) |i| sum += down[o * 37 + i];
        try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-4);
    }
}

// ── Oracle tests (env-gated; fixtures from tests/dump_hunyuan3d_paint_fixtures.py) ──

fn readF32(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]f32 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(a, .limited(1024 * 1024 * 1024));
    defer a.free(bytes);
    const n = bytes.len / 4;
    const out = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
}
fn cosine(data: []const f32, ref: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..data.len) |i| {
        dot += @as(f64, data[i]) * ref[i];
        na += @as(f64, data[i]) * data[i];
        nb += @as(f64, ref[i]) * ref[i];
    }
    return dot / (std.math.sqrt(na) * std.math.sqrt(nb));
}
fn arrayCosine(arr: mlx.mlx_array, ref: []const f32, s: S) !f64 {
    const f = try astype(arr, .float32, s);
    defer _ = mlx.mlx_array_free(f);
    _ = mlx.mlx_array_eval(f);
    const n: usize = @intCast(mlx.mlx_array_size(f));
    try testing.expectEqual(ref.len, n);
    const d = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return cosine(d[0..n], ref);
}

// Oracle 1: VAE encode. HY3DP_TEST_MODEL, HY3DP_VAE_ENC_IN ([1,3,512,512] f32,
// post-preprocess (x-0.5)*2 pixels), HY3DP_VAE_ENC_OUT ([1,4,64,64] mean·0.18215).
test "hy3d-paint oracle: VAE encode matches reference" {
    const model_dir = std.mem.span(std.c.getenv("HY3DP_TEST_MODEL") orelse return error.SkipZigTest);
    const in_p = std.mem.span(std.c.getenv("HY3DP_VAE_ENC_IN") orelse return error.SkipZigTest);
    const out_p = std.mem.span(std.c.getenv("HY3DP_VAE_ENC_OUT") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const in_d = try readF32(io, a, in_p);
    defer a.free(in_d);
    const ref = try readF32(io, a, out_p);
    defer a.free(ref);
    const s = mlx.mlx_default_gpu_stream_new();
    const cfg = try readConfigFile(io, a, model_dir);
    var vae = try loadSdVae(a, cfg, model_dir, s);
    defer vae.deinit();
    const side: c_int = @intCast(std.math.sqrt(in_d.len / 3));
    const ish = [_]c_int{ 1, 3, side, side };
    const xf = mlx.mlx_array_new_data(in_d.ptr, &ish, 4, .float32);
    defer _ = mlx.mlx_array_free(xf);
    const lat = try vae.encode(xf);
    defer _ = mlx.mlx_array_free(lat);
    const corr = try arrayCosine(lat, ref, s);
    std.debug.print("[hy3d-paint-vae-enc] corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.999);
}

// Oracle 2: VAE decode. HY3DP_VAE_DEC_IN ([1,4,64,64] scaled latents),
// HY3DP_VAE_DEC_OUT ([1,3,512,512]).
test "hy3d-paint oracle: VAE decode matches reference" {
    const model_dir = std.mem.span(std.c.getenv("HY3DP_TEST_MODEL") orelse return error.SkipZigTest);
    const in_p = std.mem.span(std.c.getenv("HY3DP_VAE_DEC_IN") orelse return error.SkipZigTest);
    const out_p = std.mem.span(std.c.getenv("HY3DP_VAE_DEC_OUT") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const in_d = try readF32(io, a, in_p);
    defer a.free(in_d);
    const ref = try readF32(io, a, out_p);
    defer a.free(ref);
    const s = mlx.mlx_default_gpu_stream_new();
    const cfg = try readConfigFile(io, a, model_dir);
    var vae = try loadSdVae(a, cfg, model_dir, s);
    defer vae.deinit();
    const lside: c_int = @intCast(std.math.sqrt(in_d.len / 4));
    const zsh = [_]c_int{ 1, 4, lside, lside };
    const zf = mlx.mlx_array_new_data(in_d.ptr, &zsh, 4, .float32);
    defer _ = mlx.mlx_array_free(zf);
    const img = try vae.decode(zf);
    defer _ = mlx.mlx_array_free(img);
    const corr = try arrayCosine(img, ref, s);
    std.debug.print("[hy3d-paint-vae-dec] corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.999);
}

// Oracle 3: DINO features + ImageProjModel context. HY3DP_DINO_IN
// ([1,3,224,224] post-preprocess), HY3DP_DINO_OUT ([1,257,1536]),
// HY3DP_DINO_PROJ ([1,1028,1024]).
test "hy3d-paint oracle: DINO features + ImageProjModel context match reference" {
    const model_dir = std.mem.span(std.c.getenv("HY3DP_TEST_MODEL") orelse return error.SkipZigTest);
    const in_p = std.mem.span(std.c.getenv("HY3DP_DINO_IN") orelse return error.SkipZigTest);
    const out_p = std.mem.span(std.c.getenv("HY3DP_DINO_OUT") orelse return error.SkipZigTest);
    const proj_p = std.mem.span(std.c.getenv("HY3DP_DINO_PROJ") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const in_d = try readF32(io, a, in_p);
    defer a.free(in_d);
    const ref_feats = try readF32(io, a, out_p);
    defer a.free(ref_feats);
    const ref_proj = try readF32(io, a, proj_p);
    defer a.free(ref_proj);
    const s = mlx.mlx_default_gpu_stream_new();
    const cfg = try readConfigFile(io, a, model_dir);
    var cond = try loadPaintConditioner(a, cfg, model_dir, s);
    defer cond.deinit();
    const side: c_int = @intCast(std.math.sqrt(in_d.len / 3));
    const ish = [_]c_int{ 1, 3, side, side };
    const xf = mlx.mlx_array_new_data(in_d.ptr, &ish, 4, .float32);
    defer _ = mlx.mlx_array_free(xf);
    const out = try cond.encodeDino(xf);
    defer _ = mlx.mlx_array_free(out.feats);
    defer _ = mlx.mlx_array_free(out.context);
    const cf = try arrayCosine(out.feats, ref_feats, s);
    const cp = try arrayCosine(out.context, ref_proj, s);
    std.debug.print("[hy3d-paint-dino] feats_corr={d:.6} proj_corr={d:.6}\n", .{ cf, cp });
    try testing.expect(cf > 0.999);
    try testing.expect(cp > 0.999);
}

// ════════════════════════════════════════════════════════════════════════
// PaintEngine — P2-12 orchestration: raw shape mesh + reference photo →
// textured GLB. Composes uvwrap → bake renders → VAE/DINO conditioning →
// 2.5D UNet DDIM denoise (2-batch CFG) → VAE decode → back-project bake →
// vertex-graph inpaint → writeGlbTextured. All stages progress-labeled
// "paint-*" so the SSE stream distinguishes the texture phase (the video
// two-stage precedent).
// ════════════════════════════════════════════════════════════════════════

const uvwrap = @import("uvwrap.zig");
const bake = @import("bake.zig");
const texinpaint = @import("texinpaint.zig");
const glb_mod = @import("glb.zig");
const png_mod = @import("png.zig");
const mc = @import("marching_cubes.zig");
const punet = @import("hunyuan3d_paint_unet.zig");
const sse = @import("gen_sse.zig");

pub const PaintOpts = struct {
    steps: u32 = 30,
    guidance: f32 = 3.0,
    seed: u64 = 0,
    /// Bake atlas resolution (exported as-is; the reference bakes 4096 → ÷2
    /// export. v1 bakes 2048 directly — no RealESRGAN super-res yet, the
    /// 512² views are used at native res, quality-flagged in the plan §5.1H).
    texture_size: u32 = 2048,
    view_size: u32 = 512,
};

/// glTF texcoords: v grows DOWNWARD while xatlas/reference uvs grow upward,
/// and the bake atlas row = (1−v)·(H−1) — so the glTF uv is exactly (u, 1−v).
pub fn gltfUvsFromAtlas(allocator: std.mem.Allocator, uvs: []const f32) ![]f32 {
    const out = try allocator.alloc(f32, uvs.len);
    var i: usize = 0;
    while (i < uvs.len) : (i += 2) {
        out[i] = uvs[i];
        out[i + 1] = 1.0 - uvs[i + 1];
    }
    return out;
}

/// Generated "mr" image channels are R=metallic, G=roughness; glTF's
/// metallicRoughness texture packs G=roughness, B=metallic. Remap [0,1] f32
/// HWC → RGB8 bytes with the two swaps (R left 0).
pub fn mrChannelsToGltf(allocator: std.mem.Allocator, mr_img: []const f32) ![]u8 {
    const texels = mr_img.len / 3;
    const out = try allocator.alloc(u8, mr_img.len);
    for (0..texels) |i| {
        const metallic = mr_img[i * 3 + 0];
        const roughness = mr_img[i * 3 + 1];
        out[i * 3 + 0] = 0;
        out[i * 3 + 1] = quant8(roughness);
        out[i * 3 + 2] = quant8(metallic);
    }
    return out;
}

fn quant8(v: f32) u8 {
    const c = std.math.clamp(v, 0.0, 1.0);
    return @intFromFloat(@round(c * 255.0));
}

fn rgbF32ToBytes(allocator: std.mem.Allocator, img: []const f32) ![]u8 {
    const out = try allocator.alloc(u8, img.len);
    for (img, out) |v, *o| o.* = quant8(v);
    return out;
}

/// Owned components of a painted mesh (post-unwrap vertex set + encoded PNG
/// textures) — what the rig stage composes with a skeleton.
pub const PaintedMesh = struct {
    positions: []f32, // N*3, original world coords (seam-duplicated)
    normals: []f32, // N*3
    uvs: []f32, // N*2, glTF convention
    indices: []u32, // M*3
    albedo_png: []u8,
    mr_png: []u8,

    pub fn textured(self: *const PaintedMesh) glb_mod.TexturedMesh {
        return .{
            .positions = self.positions,
            .normals = self.normals,
            .uvs = self.uvs,
            .indices = self.indices,
            .albedo_png = self.albedo_png,
            .mr_png = self.mr_png,
        };
    }

    pub fn deinit(self: *PaintedMesh, alloc: std.mem.Allocator) void {
        alloc.free(self.positions);
        alloc.free(self.normals);
        alloc.free(self.uvs);
        alloc.free(self.indices);
        alloc.free(self.albedo_png);
        alloc.free(self.mr_png);
        self.* = undefined;
    }
};

pub const PaintEngine = struct {
    allocator: std.mem.Allocator,
    s: S,
    cfg: PaintConfig,
    vae: SdVae,
    conditioner: PaintConditioner,
    unet: *punet.PaintUnet,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*PaintEngine {
        const self = try allocator.create(PaintEngine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.s = mlx.mlx_default_gpu_stream_new();
        self.cfg = try readConfigFile(io, allocator, model_dir);
        self.vae = try loadSdVae(allocator, self.cfg, model_dir, self.s);
        errdefer self.vae.deinit();
        self.conditioner = try loadPaintConditioner(allocator, self.cfg, model_dir, self.s);
        errdefer self.conditioner.deinit();
        self.unet = try punet.PaintUnet.load(allocator, model_dir, self.s);
        log.info("[hy3d-paint] paint engine ready (VAE + DINO-giant + 2.5D UNet)\n", .{});
        return self;
    }

    pub fn deinit(self: *PaintEngine) void {
        self.unet.deinit();
        self.conditioner.deinit();
        self.vae.deinit();
        self.allocator.destroy(self);
    }

    /// The full texture stage. `mesh` is the raw shape-stage output (world
    /// coords, CCW-outward — the marching-cubes contract); `image_rgba` is the
    /// SAME straight-alpha reference photo the shape stage consumed. Returns
    /// the textured-mesh COMPONENTS (caller frees via deinit) — the rig stage
    /// composes them with a skeleton; `paintMeshToGlb` wraps them into a GLB.
    pub fn paintMesh(self: *PaintEngine, alloc: std.mem.Allocator, mesh: *const mc.Mesh, image_rgba: []const u8, img_w: u32, img_h: u32, opts: PaintOpts, progress: ?sse.Progress) !PaintedMesh {
        const s = self.s;
        const vsz = opts.view_size;

        // ── 1. UV unwrap (xatlas), seam-duplicate vertex attributes ──
        if (progress) |p| p.emit("paint-unwrap", 0, 1);
        var uw = try uvwrap.parametrize(alloc, mesh.vertices, mesh.indices);
        defer uw.deinit(alloc);
        const n_new = uw.vmapping.len;
        const positions_u = try alloc.alloc(f32, n_new * 3);
        defer alloc.free(positions_u);
        const normals_u = try alloc.alloc(f32, n_new * 3);
        defer alloc.free(normals_u);
        for (uw.vmapping, 0..) |orig, i| {
            @memcpy(positions_u[i * 3 ..][0..3], mesh.vertices[orig * 3 ..][0..3]);
            @memcpy(normals_u[i * 3 ..][0..3], mesh.normals[orig * 3 ..][0..3]);
        }
        if (progress) |p| {
            if (p.cancelled()) return error.Cancelled;
            p.emit("paint-unwrap", 1, 1);
        }

        // ── 2. Bake-space geometry + 6 fixed-view geometry renders ──
        var geom = try bake.prepareMesh(alloc, positions_u, uw.uvs, uw.indices, uw.indices, 1.15);
        defer geom.deinit();
        const views = bake.defaultViews();
        const rcfg = bake.RenderConfig{};

        // Per-view CHW position maps (VoxelRopes + VAE) and normal maps (VAE).
        const plane: usize = @as(usize, vsz) * @as(usize, vsz);
        const posmaps_chw = try alloc.alloc(f32, 6 * 3 * plane);
        defer alloc.free(posmaps_chw);
        const nrmmaps_chw = try alloc.alloc(f32, 6 * 3 * plane);
        defer alloc.free(nrmmaps_chw);
        for (views, 0..) |view, vi| {
            var maps = try bake.renderGeometryMaps(alloc, &geom, view, vsz, rcfg);
            defer maps.deinit();
            // HWC → CHW.
            for (0..plane) |pix| {
                for (0..3) |c| {
                    posmaps_chw[(vi * 3 + c) * plane + pix] = maps.position[pix * 3 + c];
                    nrmmaps_chw[(vi * 3 + c) * plane + pix] = maps.normal[pix * 3 + c];
                }
            }
            if (progress) |p| {
                if (p.cancelled()) return error.Cancelled;
                p.emit("paint-render", @intCast(vi + 1), 6);
            }
        }

        // ── 3. Conditioning: ref VAE latent, view-map VAE latents, DINO ──
        if (progress) |p| p.emit("paint-encode", 0, 14);
        const ref_chw = try refImageChw(alloc, image_rgba, img_w, img_h, vsz);
        defer alloc.free(ref_chw);
        const ref_latents = try self.encodeChw(ref_chw, vsz);
        defer rel_(ref_latents);
        if (progress) |p| p.emit("paint-encode", 1, 14);

        var embeds_normal = mlx.mlx_array_new();
        defer rel_(embeds_normal);
        var embeds_position = mlx.mlx_array_new();
        defer rel_(embeds_position);
        try self.encodeViewMaps(alloc, nrmmaps_chw, vsz, &embeds_normal, progress, 1);
        try self.encodeViewMaps(alloc, posmaps_chw, vsz, &embeds_position, progress, 7);

        const dino_pix = try preprocessImagePaint(alloc, image_rgba, img_w, img_h);
        defer alloc.free(dino_pix);
        const dsh = [_]c_int{ 1, 3, 224, 224 };
        const dino_arr = mlx.mlx_array_new_data(dino_pix.ptr, &dsh, 4, .float32);
        defer rel_(dino_arr);
        const dino = try self.conditioner.encodeDino(dino_arr);
        defer rel_(dino.feats);
        defer rel_(dino.context);
        if (progress) |p| {
            if (p.cancelled()) return error.Cancelled;
            p.emit("paint-encode", 14, 14);
        }

        // ── 4. PoseRoPE tables + dual-UNet reference cache ──
        var ropes = try punet.VoxelRopes.build(alloc, posmaps_chw, s);
        defer ropes.deinit();
        var ref_cache = try self.unet.buildRefCache(ref_latents);
        defer ref_cache.deinit();

        // ── 5. DDIM denoise, 2-batch CFG (reference 3-batch collapses) ──
        var sch = try Scheduler.init(alloc, opts.steps);
        defer sch.deinit();
        const lat_n: usize = 12 * 4 * 64 * 64;
        const lat_cpu = try alloc.alloc(f32, lat_n);
        defer alloc.free(lat_cpu);
        {
            // Seeded gaussian init (any deterministic normal source is valid —
            // the CFG/scheduler math is pinned by oracles, not the noise).
            var key = mlx.mlx_array_new();
            defer rel_(key);
            try mlx.check(mlx.mlx_random_key(&key, opts.seed));
            var noise = mlx.mlx_array_new();
            defer rel_(noise);
            const nsh = [_]c_int{ 12, 4, 64, 64 };
            try mlx.check(mlx.mlx_random_normal(&noise, &nsh, 4, .float32, 0.0, 1.0, key, s));
            _ = mlx.mlx_array_eval(noise);
            const nd = mlx.mlx_array_data_float32(noise) orelse return error.NoData;
            @memcpy(lat_cpu, nd[0..lat_n]);
        }
        const v_cpu = try alloc.alloc(f32, lat_n);
        defer alloc.free(v_cpu);
        const va_cpu = try alloc.alloc(f32, lat_n);
        defer alloc.free(va_cpu);
        for (sch.timesteps, 0..) |t, si| {
            const lsh = [_]c_int{ 12, 4, 64, 64 };
            const lat_mx = mlx.mlx_array_new_data(lat_cpu.ptr, &lsh, 4, .float32);
            defer rel_(lat_mx);
            const tf: f32 = @floatFromInt(t);
            // Batch A: uncond (ref off, dino zeros). Batch B: full conditioning.
            const va = try self.unet.forward(lat_mx, tf, embeds_normal, embeds_position, null, &ref_cache, 0.0, &ropes);
            defer rel_(va);
            const vb = try self.unet.forward(lat_mx, tf, embeds_normal, embeds_position, dino.context, &ref_cache, 1.0, &ropes);
            defer rel_(vb);
            _ = mlx.mlx_array_eval(va);
            _ = mlx.mlx_array_eval(vb);
            const da = mlx.mlx_array_data_float32(va) orelse return error.NoData;
            const db = mlx.mlx_array_data_float32(vb) orelse return error.NoData;
            @memcpy(va_cpu, da[0..lat_n]);
            punet.cfgCombine(va_cpu, db[0..lat_n], opts.guidance, v_cpu);
            sch.step(v_cpu, t, lat_cpu);
            if (progress) |p| {
                if (p.cancelled()) return error.Cancelled;
                p.emit("paint-denoise", @intCast(si + 1), @intCast(sch.timesteps.len));
            }
        }

        // ── 6. VAE decode 12 views → f32 HWC [0,1] images ──
        var view_imgs: [12][]f32 = undefined;
        var decoded: usize = 0;
        defer for (0..decoded) |i| alloc.free(view_imgs[i]);
        for (0..12) |vi| {
            const zsh = [_]c_int{ 1, 4, 64, 64 };
            const z = mlx.mlx_array_new_data(lat_cpu.ptr + vi * 4 * 64 * 64, &zsh, 4, .float32);
            defer rel_(z);
            const img = try self.vae.decode(z); // [1,3,H,W] f32 in [-1,1]
            defer rel_(img);
            _ = mlx.mlx_array_eval(img);
            const d = mlx.mlx_array_data_float32(img) orelse return error.NoData;
            const hw = @as(usize, vsz) * @as(usize, vsz);
            const out = try alloc.alloc(f32, hw * 3);
            for (0..hw) |pix| {
                for (0..3) |c| {
                    out[pix * 3 + c] = std.math.clamp(d[c * hw + pix] / 2.0 + 0.5, 0.0, 1.0);
                }
            }
            view_imgs[vi] = out;
            decoded += 1;
            if (progress) |p| {
                if (p.cancelled()) return error.Cancelled;
                p.emit("paint-decode", @intCast(vi + 1), 12);
            }
        }

        // ── 7. Back-project bake (albedo = views 0..5, mr = 6..11) ──
        var textiles = try bake.extractTextiles(alloc, &geom, opts.texture_size);
        defer textiles.deinit();
        var albedo_bake = try self.bakeMaterial(alloc, &geom, &textiles, views, view_imgs[0..6], vsz, rcfg, opts.texture_size, progress, "paint-bake");
        defer albedo_bake.deinit();
        var mr_bake = try self.bakeMaterial(alloc, &geom, &textiles, views, view_imgs[6..12], vsz, rcfg, opts.texture_size, progress, "paint-bake-mr");
        defer mr_bake.deinit();

        // ── 8. Inpaint both atlases (vertex-graph + diffusion fill) ──
        if (progress) |p| p.emit("paint-inpaint", 0, 2);
        const tex = opts.texture_size;
        const texels = @as(usize, tex) * @as(usize, tex);
        const mask = try alloc.alloc(u8, texels);
        defer alloc.free(mask);
        for (albedo_bake.mask, mask) |m, *o| o.* = if (m) 1 else 0;
        // inpaintAtlas does its own internal v-flip — pass the UNflipped uvs.
        try texinpaint.inpaintAtlas(alloc, albedo_bake.atlas, mask, tex, tex, geom.positions, uw.uvs, geom.indices);
        if (progress) |p| p.emit("paint-inpaint", 1, 2);
        for (mr_bake.mask, mask) |m, *o| o.* = if (m) 1 else 0;
        try texinpaint.inpaintAtlas(alloc, mr_bake.atlas, mask, tex, tex, geom.positions, uw.uvs, geom.indices);
        if (progress) |p| {
            if (p.cancelled()) return error.Cancelled;
            p.emit("paint-inpaint", 2, 2);
        }

        // ── 9. PNG encode + textured GLB ──
        const albedo_rgb = try rgbF32ToBytes(alloc, albedo_bake.atlas);
        defer alloc.free(albedo_rgb);
        const albedo_png = try png_mod.encodeRgb(alloc, albedo_rgb, tex, tex);
        defer alloc.free(albedo_png);
        const mr_rgb = try mrChannelsToGltf(alloc, mr_bake.atlas);
        defer alloc.free(mr_rgb);
        const mr_png = try png_mod.encodeRgb(alloc, mr_rgb, tex, tex);
        defer alloc.free(mr_png);
        const gltf_uvs = try gltfUvsFromAtlas(alloc, uw.uvs);
        defer alloc.free(gltf_uvs);

        // Transfer components out (dupes are trivial next to the GPU work).
        var out = PaintedMesh{
            .positions = try alloc.dupe(f32, positions_u), // ORIGINAL world coords
            .normals = undefined,
            .uvs = undefined,
            .indices = undefined,
            .albedo_png = undefined,
            .mr_png = undefined,
        };
        errdefer alloc.free(out.positions);
        out.normals = try alloc.dupe(f32, normals_u);
        errdefer alloc.free(out.normals);
        out.uvs = try alloc.dupe(f32, gltf_uvs);
        errdefer alloc.free(out.uvs);
        out.indices = try alloc.dupe(u32, uw.indices);
        errdefer alloc.free(out.indices);
        out.albedo_png = try alloc.dupe(u8, albedo_png);
        errdefer alloc.free(out.albedo_png);
        out.mr_png = try alloc.dupe(u8, mr_png);
        return out;
    }

    /// Convenience: paint + serialize to a textured GLB (the non-rig path).
    pub fn paintMeshToGlb(self: *PaintEngine, alloc: std.mem.Allocator, mesh: *const mc.Mesh, image_rgba: []const u8, img_w: u32, img_h: u32, opts: PaintOpts, progress: ?sse.Progress) ![]u8 {
        var pm = try self.paintMesh(alloc, mesh, image_rgba, img_w, img_h, opts, progress);
        defer pm.deinit(alloc);
        const tm = pm.textured();
        return glb_mod.writeGlbTextured(alloc, &tm);
    }

    fn bakeMaterial(self: *PaintEngine, alloc: std.mem.Allocator, geom: *const bake.MeshGeom, textiles: *const bake.Textiles, views: [6]bake.View, imgs: []const []f32, vsz: u32, rcfg: bake.RenderConfig, tex_size: u32, progress: ?sse.Progress, stage: []const u8) !bake.BakeResult {
        _ = self;
        var vps: [6]bake.ViewProjection = undefined;
        var built: usize = 0;
        defer for (0..built) |i| vps[i].deinit();
        for (views, 0..) |view, vi| {
            vps[vi] = try bake.backProject(alloc, geom, textiles, view, imgs[vi], vsz, vsz, 3, rcfg);
            built += 1;
            if (progress) |p| {
                if (p.cancelled()) return error.Cancelled;
                p.emit(stage, @intCast(vi + 1), 6);
            }
        }
        return bake.bakeBlend(alloc, vps[0..built], 4.0, tex_size, 3);
    }

    /// VAE-encode one CHW [3,size,size] f32 image in [0,1]: (x−0.5)·2 →
    /// posterior mean·0.18215 → [1,4,size/8,size/8].
    fn encodeChw(self: *PaintEngine, chw: []const f32, size: u32) !mlx.mlx_array {
        const scaled = try self.allocator.alloc(f32, chw.len);
        defer self.allocator.free(scaled);
        for (chw, scaled) |v, *o| o.* = (v - 0.5) * 2.0;
        const sh = [_]c_int{ 1, 3, @intCast(size), @intCast(size) };
        const arr = mlx.mlx_array_new_data(scaled.ptr, &sh, 4, .float32);
        defer rel_(arr);
        return self.vae.encode(arr);
    }

    /// Encode 6 per-view CHW maps → concat [6,4,size/8,size/8].
    fn encodeViewMaps(self: *PaintEngine, alloc: std.mem.Allocator, maps_chw: []const f32, size: u32, out: *mlx.mlx_array, progress: ?sse.Progress, prog_base: u32) !void {
        _ = alloc;
        const plane = @as(usize, size) * @as(usize, size) * 3;
        var lats: [6]mlx.mlx_array = undefined;
        var made: usize = 0;
        defer for (0..made) |i| rel_(lats[i]);
        for (0..6) |vi| {
            lats[vi] = try self.encodeChw(maps_chw[vi * plane ..][0..plane], size);
            made += 1;
            if (progress) |p| {
                if (p.cancelled()) return error.Cancelled;
                p.emit("paint-encode", prog_base + @as(u32, @intCast(vi)), 14);
            }
        }
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (0..6) |i| _ = mlx.mlx_vector_array_append_value(vec, lats[i]);
        try mlx.check(mlx.mlx_concatenate_axis(out, vec, 0, self.s));
        _ = mlx.mlx_array_eval(out.*);
    }
};

inline fn rel_(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

/// Reference photo → CHW [3,size,size] f32 in [0,1]: straight-alpha composite
/// on WHITE, then bilinear stretch to size² (the reference resizes the prompt
/// image squarely to the view resolution).
pub fn refImageChw(allocator: std.mem.Allocator, rgba: []const u8, w: u32, h: u32, size: u32) ![]f32 {
    const out = try allocator.alloc(f32, 3 * @as(usize, size) * @as(usize, size));
    const plane = @as(usize, size) * @as(usize, size);
    const sx = @as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(size));
    const sy = @as(f32, @floatFromInt(h)) / @as(f32, @floatFromInt(size));
    for (0..size) |oy| {
        for (0..size) |ox| {
            const fx = (@as(f32, @floatFromInt(ox)) + 0.5) * sx - 0.5;
            const fy = (@as(f32, @floatFromInt(oy)) + 0.5) * sy - 0.5;
            const x0f = @floor(fx);
            const y0f = @floor(fy);
            const tx = fx - x0f;
            const ty = fy - y0f;
            const x0: i64 = @intFromFloat(x0f);
            const y0: i64 = @intFromFloat(y0f);
            var acc = [3]f32{ 0, 0, 0 };
            inline for (.{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ 1, 1 } }) |d| {
                const xx: usize = @intCast(std.math.clamp(x0 + d[0], 0, @as(i64, @intCast(w)) - 1));
                const yy: usize = @intCast(std.math.clamp(y0 + d[1], 0, @as(i64, @intCast(h)) - 1));
                const wgt = (if (d[0] == 0) 1.0 - tx else tx) * (if (d[1] == 0) 1.0 - ty else ty);
                const pi = (yy * w + xx) * 4;
                const a = @as(f32, @floatFromInt(rgba[pi + 3])) / 255.0;
                for (0..3) |c| {
                    const src = @as(f32, @floatFromInt(rgba[pi + c])) / 255.0;
                    acc[c] += wgt * (src * a + (1.0 - a)); // composite on white
                }
            }
            for (0..3) |c| out[c * plane + oy * size + ox] = acc[c];
        }
    }
    return out;
}

test "hy3d-paint gltf uv flip + mr channel remap helpers" {
    const a = testing.allocator;
    const uvs = [_]f32{ 0.25, 0.75, 1.0, 0.0 };
    const flipped = try gltfUvsFromAtlas(a, &uvs);
    defer a.free(flipped);
    try testing.expectApproxEqAbs(@as(f32, 0.25), flipped[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.25), flipped[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), flipped[3], 1e-6);

    // R=metallic 1.0, G=roughness 0.5 → glTF G=roughness(128), B=metallic(255).
    const mr = [_]f32{ 1.0, 0.5, 0.0 };
    const remapped = try mrChannelsToGltf(a, &mr);
    defer a.free(remapped);
    try testing.expectEqual(@as(u8, 0), remapped[0]);
    try testing.expectEqual(@as(u8, 128), remapped[1]);
    try testing.expectEqual(@as(u8, 255), remapped[2]);
}

test "hy3d-paint refImageChw composites alpha on white and stretches" {
    const a = testing.allocator;
    // 2x2 fully-transparent image → all white regardless of RGB.
    const rgba = [_]u8{ 10, 20, 30, 0, 40, 50, 60, 0, 70, 80, 90, 0, 100, 110, 120, 0 };
    const chw = try refImageChw(a, &rgba, 2, 2, 4);
    defer a.free(chw);
    for (chw) |v| try testing.expectApproxEqAbs(@as(f32, 1.0), v, 1e-6);
}
