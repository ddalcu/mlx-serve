//! Hunyuan3D-2.1 paint 2.5D UNet (main + dual reference stream) — the P2-7/8
//! denoiser for src/hunyuan3d_paint.zig. Ported from the Tencent reference
//! per tests/hy3d_paint_unet_dossier.md (source-verified facts only; read it
//! before touching the forward).
//!
//! Parity traps honored here:
//! - conv_in is 12-channel on the MAIN UNet (config lies with 4), 4 on dual.
//! - Block order inside Basic2p5DTransformerBlock: MDA → RA → MA → attn2 →
//!   DINO → FF; RA/MA reuse norm1's output, DINO reuses norm2's output.
//! - RA: query = ALBEDO slice only; V = concat(to_v, to_v_mr) per head;
//!   per-material to_out/to_out_mr; residual × ref_scale.
//! - MA: (n·l)-token cross-view self-attn with 3D PoseRoPE — interleaved-pair
//!   rotation, axis dims 3/8·3/8·2/8 of head_dim, f32 tables, voxel grids
//!   keyed by sequence length. Fused SDPA mandatory (24,576-token seqs).
//! - GeGLU: value = FIRST half of proj rows, gate = SECOND half.
//! - Timestep embedding: diffusers Timesteps(320, flip_sin_to_cos=true,
//!   downscale_freq_shift=0) — COS half first.
//! - CFG runs as 2 batches (A = {ref_scale 0, dino ZEROS}, B = {ref_scale 1,
//!   dino real}); the reference's 3-batch middle term cancels because 2.1
//!   never passes camera_azims (view scale ≡ 1). Zero-DINO batches still add
//!   attn_dino.to_out.bias — identical in A and B, so the algebra holds.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const hy3d = @import("hunyuan3d.zig");

const S = mlx.mlx_stream;
const MixedLinear = hy3d.MixedLinear;

// ════════════════════════════════════════════════════════════════════════
// Pure math (CPU) — timestep embedding, PoseRoPE tables, voxel indices, CFG.
// ════════════════════════════════════════════════════════════════════════

/// diffusers Timesteps(dim, flip_sin_to_cos=true, downscale_freq_shift=0):
/// half = dim/2; freq_i = exp(-ln(10000)·i/half); output [cos(t·f) | sin(t·f)].
pub fn timestepEmbed(allocator: std.mem.Allocator, t: f32, dim: u32) ![]f32 {
    const half = dim / 2;
    const out = try allocator.alloc(f32, dim);
    for (0..half) |i| {
        const freq = std.math.exp(-std.math.log(f32, std.math.e, 10000.0) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
        const arg = t * freq;
        out[i] = std.math.cos(arg); // flip_sin_to_cos → cos half FIRST
        out[half + i] = std.math.sin(arg);
    }
    return out;
}

/// 1D RoPE table pair for one axis: freqs = theta^-(2i/dim), i in [0,dim/2);
/// cos/sin(pos·freq) with each frequency REPEATED ×2 (interleaved pairs).
/// `dim` must be even. Rows = voxel_resolution positions. f32 (parity trap).
pub fn ropeAxisTables(allocator: std.mem.Allocator, dim: usize, resolution: usize) !struct { cos: []f32, sin: []f32 } {
    const half = dim / 2;
    const cos = try allocator.alloc(f32, resolution * dim);
    errdefer allocator.free(cos);
    const sin = try allocator.alloc(f32, resolution * dim);
    for (0..resolution) |p| {
        for (0..half) |i| {
            const freq = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dim)));
            const arg = @as(f32, @floatFromInt(p)) * freq;
            const c = std.math.cos(arg);
            const sn = std.math.sin(arg);
            cos[p * dim + 2 * i] = c;
            cos[p * dim + 2 * i + 1] = c;
            sin[p * dim + 2 * i] = sn;
            sin[p * dim + 2 * i + 1] = sn;
        }
    }
    return .{ .cos = cos, .sin = sin };
}

/// Per-token 3D RoPE cos/sin [L, head_dim] from voxel indices [L,3]:
/// x/y axes share a dim_xy = head_dim/8·3 table, z uses dim_z = head_dim/8·2;
/// concatenated x|y|z along the last dim.
pub const Rope3d = struct {
    cos: []f32, // [L, head_dim]
    sin: []f32,
    head_dim: usize,

    pub fn deinit(self: *Rope3d, allocator: std.mem.Allocator) void {
        allocator.free(self.cos);
        allocator.free(self.sin);
        self.* = undefined;
    }
};

pub fn rope3dFromVoxels(allocator: std.mem.Allocator, voxels: []const u32, head_dim: usize, voxel_resolution: usize) !Rope3d {
    const l = voxels.len / 3;
    const dim_xy = head_dim / 8 * 3;
    const dim_z = head_dim / 8 * 2;
    var xy = try ropeAxisTables(allocator, dim_xy, voxel_resolution);
    defer {
        allocator.free(xy.cos);
        allocator.free(xy.sin);
    }
    var zt = try ropeAxisTables(allocator, dim_z, voxel_resolution);
    defer {
        allocator.free(zt.cos);
        allocator.free(zt.sin);
    }
    const cos = try allocator.alloc(f32, l * head_dim);
    errdefer allocator.free(cos);
    const sin = try allocator.alloc(f32, l * head_dim);
    for (0..l) |i| {
        const vx: usize = voxels[i * 3 + 0];
        const vy: usize = voxels[i * 3 + 1];
        const vz: usize = voxels[i * 3 + 2];
        const dst_c = cos[i * head_dim ..][0..head_dim];
        const dst_s = sin[i * head_dim ..][0..head_dim];
        @memcpy(dst_c[0..dim_xy], xy.cos[vx * dim_xy ..][0..dim_xy]);
        @memcpy(dst_s[0..dim_xy], xy.sin[vx * dim_xy ..][0..dim_xy]);
        @memcpy(dst_c[dim_xy .. 2 * dim_xy], xy.cos[vy * dim_xy ..][0..dim_xy]);
        @memcpy(dst_s[dim_xy .. 2 * dim_xy], xy.sin[vy * dim_xy ..][0..dim_xy]);
        @memcpy(dst_c[2 * dim_xy ..][0..dim_z], zt.cos[vz * dim_z ..][0..dim_z]);
        @memcpy(dst_s[2 * dim_xy ..][0..dim_z], zt.sin[vz * dim_z ..][0..dim_z]);
    }
    return .{ .cos = cos, .sin = sin, .head_dim = head_dim };
}

/// Interleaved-pair rotary application on one token vector (f32):
/// out = x·cos + rotate(x)·sin where rotate maps (x0,x1) → (−x1,x0) per pair.
pub fn ropeApplyToken(x: []const f32, cos: []const f32, sin: []const f32, out: []f32) void {
    var i: usize = 0;
    while (i < x.len) : (i += 2) {
        out[i] = x[i] * cos[i] + (-x[i + 1]) * sin[i];
        out[i + 1] = x[i + 1] * cos[i + 1] + x[i] * sin[i + 1];
    }
}

/// Port of compute_discrete_voxel_indice (modules.py:196-250) for ONE view set:
/// position maps [n, 3, H, W] f32 (bg EXACTLY 1.0) → voxel indices
/// [n · grid · grid × 3] u32 in view-major, row-major cell order (the
/// reference's "b n c h w -> b (n h w) c" flatten).
pub fn voxelIndices(allocator: std.mem.Allocator, maps: []const f32, n_views: usize, h: usize, w: usize, grid: usize, voxel_res: usize) ![]u32 {
    const cell_h = h / grid;
    const cell_w = w / grid;
    const thres: usize = (cell_h * cell_w) / 16; // (H/g)·(W/g) // (4·4)
    const out = try allocator.alloc(u32, n_views * grid * grid * 3);
    for (0..n_views) |v| {
        const base = v * 3 * h * w;
        for (0..grid) |gy| {
            for (0..grid) |gx| {
                var sum = [3]f64{ 0, 0, 0 };
                var count: usize = 0;
                for (0..cell_h) |cy| {
                    for (0..cell_w) |cx| {
                        const y = gy * cell_h + cy;
                        const x = gx * cell_w + cx;
                        const px = maps[base + 0 * h * w + y * w + x];
                        const py = maps[base + 1 * h * w + y * w + x];
                        const pz = maps[base + 2 * h * w + y * w + x];
                        // valid iff NO channel equals exactly 1 in fp16 terms:
                        // reference checks (position != 1).all(channel).
                        const valid = px != 1.0 and py != 1.0 and pz != 1.0;
                        if (valid) {
                            sum[0] += px;
                            sum[1] += py;
                            sum[2] += pz;
                            count += 1;
                        }
                    }
                }
                const denom: f64 = @floatFromInt(@max(count, 1));
                const oi = (v * grid * grid + gy * grid + gx) * 3;
                for (0..3) |c| {
                    var mean: f64 = sum[c] / denom;
                    if (count < thres) mean = 0;
                    mean = std.math.clamp(mean, 0.0, 1.0);
                    const idx = @round(mean * @as(f64, @floatFromInt(voxel_res - 1)));
                    out[oi + c] = @intFromFloat(idx);
                }
            }
        }
    }
    return out;
}

/// 2-batch CFG combine (the reference's 3-batch collapses; dossier "CFG"):
/// v = a + g·(b − a), elementwise in f32.
pub fn cfgCombine(a: []const f32, b: []const f32, g: f32, out: []f32) void {
    for (out, a, b) |*o, av, bv| o.* = av + g * (bv - av);
}

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic goldens generated from the reference implementation
// (diffusers Timesteps + hy3dpaint RotaryEmbedding/compute_discrete_voxel_indice).
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "hy3d-paint-unet: timestep embedding is cos-first (diffusers goldens)" {
    const a = testing.allocator;
    const e999 = try timestepEmbed(a, 999.0, 320);
    defer a.free(e999);
    const g_cos = [_]f32{ 0.9996498227119446, 0.8026775121688843, -0.2780621349811554, 0.17680281400680542 };
    const g_sin = [_]f32{ -0.02646075189113617, 0.5964133143424988, -0.9605631232261658, -0.9842463135719299 };
    for (0..4) |i| {
        try testing.expectApproxEqAbs(g_cos[i], e999[i], 2e-4);
        try testing.expectApproxEqAbs(g_sin[i], e999[160 + i], 2e-4);
    }
    const e0 = try timestepEmbed(a, 0.0, 320);
    defer a.free(e0);
    try testing.expectEqual(@as(f32, 1.0), e0[0]);
    try testing.expectEqual(@as(f32, 0.0), e0[160]);
}

test "hy3d-paint-unet: 3D PoseRoPE tables + interleaved rotation match reference" {
    const a = testing.allocator;
    // Token at voxel (3,7,11), head_dim 64, resolution 64.
    const vox = [_]u32{ 3, 7, 11 };
    var r = try rope3dFromVoxels(a, &vox, 64, 64);
    defer r.deinit(a);
    // x-axis (dim 24) leading entries at pos 3: cos(3·f0)= -0.98999, pairs repeat.
    const g_cos6 = [_]f32{ -0.9899924993515015, -0.9899924993515015, 0.17737612128257751, 0.17737612128257751, 0.798299252986908, 0.798299252986908 };
    const g_sin6 = [_]f32{ 0.14112000167369843, 0.14112000167369843, 0.9841431379318237, 0.9841431379318237, 0.6022610068321228, 0.6022610068321228 };
    for (0..6) |i| {
        try testing.expectApproxEqAbs(g_cos6[i], r.cos[i], 1e-5);
        try testing.expectApproxEqAbs(g_sin6[i], r.sin[i], 1e-5);
    }
    // y-axis starts at 24 (pos 7), z-axis at 48 (pos 11).
    try testing.expectApproxEqAbs(@as(f32, 0.7539022564888), r.cos[24], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.004425697959959507), r.cos[48], 1e-5);

    // apply_rotary_emb golden: x = arange(64)/64.
    var x: [64]f32 = undefined;
    for (0..64) |i| x[i] = @as(f32, @floatFromInt(i)) / 64.0;
    var out: [64]f32 = undefined;
    ropeApplyToken(&x, r.cos[0..64], r.sin[0..64], &out);
    const g_out = [_]f32{ -0.002205000026151538, -0.01546863280236721, -0.040588703006505966, 0.039068978279829025, 0.002842061221599579, 0.10000844299793243 };
    for (0..6) |i| try testing.expectApproxEqAbs(g_out[i], out[i], 1e-5);
}

test "hy3d-paint-unet: voxel indices match reference (mean, threshold, background)" {
    const a = testing.allocator;
    // One view, 3x8x8 maps, all background except a 4x4 patch at 0.5.
    var maps = [_]f32{1.0} ** (3 * 8 * 8);
    for (0..3) |c| for (0..4) |y| for (0..4) |x| {
        maps[c * 64 + y * 8 + x] = 0.5;
    };
    const vi = try voxelIndices(a, &maps, 1, 8, 8, 2, 16);
    defer a.free(vi);
    // grid 2x2: cell(0,0) mean 0.5 → round(0.5·15)=8; others empty → 0.
    const expected = [_]u32{ 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    try testing.expectEqualSlices(u32, &expected, vi);

    // Single valid texel (count 1 ≥ thres 16/16=1): mean 0.25 → round(3.75)=4.
    var maps2 = [_]f32{1.0} ** (3 * 8 * 8);
    for (0..3) |c| maps2[c * 64] = 0.25;
    const vi2 = try voxelIndices(a, &maps2, 1, 8, 8, 2, 16);
    defer a.free(vi2);
    try testing.expectEqual(@as(u32, 4), vi2[0]);
    try testing.expectEqual(@as(u32, 4), vi2[1]);
    try testing.expectEqual(@as(u32, 4), vi2[2]);
}

test "hy3d-paint-unet: 2-batch CFG equals the reference 3-batch collapse" {
    // Reference: v = uncond + g·(ref − uncond) + g·(full − ref) with view
    // scale 1 ≡ uncond + g·(full − uncond). Our 2-batch drops the middle.
    var a3: [8]f32 = undefined; // uncond
    var b3: [8]f32 = undefined; // ref (arbitrary — must cancel)
    var c3: [8]f32 = undefined; // full
    var prng = std.Random.DefaultPrng.init(42);
    for (0..8) |i| {
        a3[i] = prng.random().float(f32) - 0.5;
        b3[i] = prng.random().float(f32) - 0.5;
        c3[i] = prng.random().float(f32) - 0.5;
    }
    const g: f32 = 3.0;
    var ours: [8]f32 = undefined;
    cfgCombine(&a3, &c3, g, &ours);
    for (0..8) |i| {
        const reference = a3[i] + g * (b3[i] - a3[i]) + g * (c3[i] - b3[i]);
        try testing.expectApproxEqAbs(reference, ours[i], 1e-5);
    }
}

// ════════════════════════════════════════════════════════════════════════
// GPU forward — SD2.1 base UNet skeleton shared by main + dual, plus the
// main UNet's 2.5D additions (MDA / RA / MA+PoseRoPE / DINO). All dense fp16.
// One code path loads both UNets (§5 of the contract: byte-identical base
// namespaces); `is_main` gates the 12-vs-4-ch conv_in and the 2.5D modules.
// ════════════════════════════════════════════════════════════════════════

const Weights = @import("model.zig").Weights;

/// head_dim is 64 everywhere → SDPA scale is a constant 1/sqrt(64).
const ATTN_SCALE: f32 = 0.125;
const HEAD_DIM: c_int = 64;

inline fn rel(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
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
inline fn expandDims(x: mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_expand_dims(&o, x, axis, s));
    return o;
}
inline fn broadcastTo(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_broadcast_to(&o, x, shape.ptr, shape.len, s));
    return o;
}
/// New handle sharing an existing array (+1 ref) — the ownWeight mechanism.
/// Lets us snapshot a residual into a stack while `h` continues to be replaced.
inline fn cloneRef(a: mlx.mlx_array) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, a));
    return o;
}
fn concat(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}
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
inline fn scalarMul(x: mlx.mlx_array, v: f32, s: S) !mlx.mlx_array {
    const sc = mlx.mlx_array_new_float(v);
    defer rel(sc);
    return mulA(x, sc, s);
}
fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer rel(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, s));
    return mulA(x, sig, s);
}
/// Exact (erf) GELU — diffusers GEGLU uses F.gelu approximate="none".
fn geluErf(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const inv_sqrt2 = mlx.mlx_array_new_float(0.7071067811865476);
    defer rel(inv_sqrt2);
    const xs = try mulA(x, inv_sqrt2, s);
    defer rel(xs);
    var e = mlx.mlx_array_new();
    defer rel(e);
    try mlx.check(mlx.mlx_erf(&e, xs, s));
    const one = mlx.mlx_array_new_float(1.0);
    defer rel(one);
    const opt = try addA(e, one, s);
    defer rel(opt);
    const half = mlx.mlx_array_new_float(0.5);
    defer rel(half);
    const hx = try mulA(x, half, s);
    defer rel(hx);
    return mulA(hx, opt, s);
}
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
    defer rel(x4);
    return transpose(x4, &[_]c_int{ 0, 2, 1, 3 }, s);
}
/// [B,H,L,hd] → [B,L,H*hd]
fn mergeHeads(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const t = try transpose(x, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer rel(t);
    return reshape(t, &[_]c_int{ sh[0], sh[2], sh[1] * sh[3] }, s);
}
/// Zero-pad the last axis by `extra` on the high side (RA q/k → 128, so a fused
/// SDPA with a 128-wide value never needs an unequal-dim kernel; padding adds 0
/// to every q·k so the scaled dot product is unchanged).
fn padLastDim(x: mlx.mlx_array, extra: c_int, s: S) !mlx.mlx_array {
    const nd = mlx.getShape(x).len;
    const axis = [_]c_int{@intCast(nd - 1)};
    const low = [_]c_int{0};
    const high = [_]c_int{extra};
    const zero = mlx.mlx_array_new_float(0.0);
    defer rel(zero);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_pad(&o, x, &axis, 1, &low, 1, &high, 1, zero, "constant", s));
    return o;
}
/// conv2d on NHWC data with OHWI f16 weight [O,kH,kW,I] + bias [O]. Materialize
/// first — mlx_conv2d miscomputes on strided/lazy-view inputs (flux gotcha).
fn conv2d(x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, stride: c_int, pad: c_int, s: S) !mlx.mlx_array {
    const xf = try astype(x, .float16, s);
    defer rel(xf);
    const xc = try contig(xf, s);
    defer rel(xc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, stride, stride, pad, pad, 1, 1, 1, s));
    const r = try addA(o, bias, s);
    rel(o);
    return r;
}
/// PyTorch GroupNorm on NHWC [B,H,W,C], computed in f32, affine params per-channel.
fn groupNorm(x: mlx.mlx_array, weight: mlx.mlx_array, bias: mlx.mlx_array, groups: c_int, eps: f32, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const B = sh[0];
    const H = sh[1];
    const Wd = sh[2];
    const C = sh[3];
    const cg = @divExact(C, groups);
    const xf = try astype(x, .float32, s);
    defer rel(xf);
    const r1 = try reshape(xf, &[_]c_int{ B, H * Wd, groups, cg }, s);
    defer rel(r1);
    const t1 = try transpose(r1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer rel(t1);
    const flat = try reshape(t1, &[_]c_int{ B, groups, H * Wd * cg }, s);
    defer rel(flat);
    var mean = mlx.mlx_array_new();
    defer rel(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, flat, -1, true, s));
    const xc = try subA(flat, mean, s);
    defer rel(xc);
    const sq = try mulA(xc, xc, s);
    defer rel(sq);
    var v = mlx.mlx_array_new();
    defer rel(v);
    try mlx.check(mlx.mlx_mean_axis(&v, sq, -1, true, s));
    const epsa = mlx.mlx_array_new_float(eps);
    defer rel(epsa);
    var ve = mlx.mlx_array_new();
    defer rel(ve);
    try mlx.check(mlx.mlx_add(&ve, v, epsa, s));
    var rsq = mlx.mlx_array_new();
    defer rel(rsq);
    try mlx.check(mlx.mlx_rsqrt(&rsq, ve, s));
    const norm = try mulA(xc, rsq, s);
    defer rel(norm);
    const b1 = try reshape(norm, &[_]c_int{ B, groups, H * Wd, cg }, s);
    defer rel(b1);
    const b2 = try transpose(b1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer rel(b2);
    const b3 = try reshape(b2, &[_]c_int{ B, H, Wd, C }, s);
    defer rel(b3);
    const wf = try astype(weight, .float32, s);
    defer rel(wf);
    const bf = try astype(bias, .float32, s);
    defer rel(bf);
    const sc = try mulA(b3, wf, s);
    defer rel(sc);
    const out = try addA(sc, bf, s);
    defer rel(out);
    return astype(out, .float16, s);
}
/// LayerNorm (weight+bias, eps 1e-5) via the fast kernel (upcasts internally).
fn layerNorm(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_layer_norm(&o, x, w, b, 1e-5, s));
    return o;
}
/// Upload a host f32 slice into a self-contained mlx array (add-zero + eval
/// decouples it from the host buffer, which the caller may then free).
fn ownHostF32(data: []const f32, shape: []const c_int, s: S) !mlx.mlx_array {
    const arr = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer rel(arr);
    const zero = mlx.mlx_array_new_float(0.0);
    defer rel(zero);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, arr, zero, s));
    _ = mlx.mlx_array_eval(o);
    return o;
}
/// Interleaved-pair rotary application: out = x·cos + rotate(x)·sin where rotate
/// maps (x0,x1)→(−x1,x0). cos/sin are f32 [1,1,L,head_dim]; math in f32.
fn ropeApply(x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [B,H,L,D]
    const B = sh[0];
    const Hn = sh[1];
    const L = sh[2];
    const D = sh[3];
    const xf = try astype(x, .float32, s);
    defer rel(xf);
    const x2 = try reshape(xf, &[_]c_int{ B, Hn, L, @divExact(D, 2), 2 }, s);
    defer rel(x2);
    const even = try sliceAxis(x2, 4, 0, 1, s); // (x0)
    defer rel(even);
    const odd = try sliceAxis(x2, 4, 1, 2, s); // (x1)
    defer rel(odd);
    var neg_odd = mlx.mlx_array_new();
    defer rel(neg_odd);
    try mlx.check(mlx.mlx_negative(&neg_odd, odd, s));
    const rot5 = try concat(&[_]mlx.mlx_array{ neg_odd, even }, 4, s); // (−x1,x0)
    defer rel(rot5);
    const rot = try reshape(rot5, &[_]c_int{ B, Hn, L, D }, s);
    defer rel(rot);
    const t1 = try mulA(xf, cos, s);
    defer rel(t1);
    const t2 = try mulA(rot, sin, s);
    defer rel(t2);
    const out = try addA(t1, t2, s);
    defer rel(out);
    return astype(out, .float16, s);
}

// ── Weight-holding structs ────────────────────────────────────────────────

const Conv = struct {
    w: mlx.mlx_array,
    b: mlx.mlx_array,
    fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8) !Conv {
        const wk = try hy3d.fmtKey(a, "{s}.weight", .{prefix});
        defer a.free(wk);
        const bk = try hy3d.fmtKey(a, "{s}.bias", .{prefix});
        defer a.free(bk);
        return .{ .w = try hy3d.ownWeight(w, wk), .b = try hy3d.ownWeight(w, bk) };
    }
    fn loadOpt(w: *const Weights, a: std.mem.Allocator, prefix: []const u8) ?Conv {
        const wk = hy3d.fmtKey(a, "{s}.weight", .{prefix}) catch return null;
        defer a.free(wk);
        const bk = hy3d.fmtKey(a, "{s}.bias", .{prefix}) catch return null;
        defer a.free(bk);
        const cw = hy3d.ownOpt(w, wk) orelse return null;
        const cb = hy3d.ownOpt(w, bk) orelse {
            rel(cw);
            return null;
        };
        return .{ .w = cw, .b = cb };
    }
    fn deinit(self: *Conv) void {
        rel(self.w);
        rel(self.b);
    }
};

const Resnet = struct {
    n1w: mlx.mlx_array,
    n1b: mlx.mlx_array,
    conv1: Conv,
    temb: MixedLinear, // time_emb_proj, in=1280
    n2w: mlx.mlx_array,
    n2b: mlx.mlx_array,
    conv2: Conv,
    shortcut: ?Conv,

    fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !Resnet {
        const g = struct {
            fn norm(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8) !mlx.mlx_array {
                const kk = try hy3d.fmtKey(aa, "{s}.{s}", .{ p, sub });
                defer aa.free(kk);
                return hy3d.ownWeight(ww, kk);
            }
        };
        const tp = try hy3d.fmtKey(a, "{s}.time_emb_proj", .{prefix});
        defer a.free(tp);
        const c1p = try hy3d.fmtKey(a, "{s}.conv1", .{prefix});
        defer a.free(c1p);
        const c2p = try hy3d.fmtKey(a, "{s}.conv2", .{prefix});
        defer a.free(c2p);
        const scp = try hy3d.fmtKey(a, "{s}.conv_shortcut", .{prefix});
        defer a.free(scp);
        return .{
            .n1w = try g.norm(w, a, prefix, "norm1.weight"),
            .n1b = try g.norm(w, a, prefix, "norm1.bias"),
            .conv1 = try Conv.load(w, a, c1p),
            .temb = try MixedLinear.load(w, a, tp, 1280, s),
            .n2w = try g.norm(w, a, prefix, "norm2.weight"),
            .n2b = try g.norm(w, a, prefix, "norm2.bias"),
            .conv2 = try Conv.load(w, a, c2p),
            .shortcut = Conv.loadOpt(w, a, scp),
        };
    }
    fn deinit(self: *Resnet) void {
        rel(self.n1w);
        rel(self.n1b);
        self.conv1.deinit();
        self.temb.deinit();
        rel(self.n2w);
        rel(self.n2b);
        self.conv2.deinit();
        if (self.shortcut) |*c| c.deinit();
    }
    /// x [B,H,W,Cin] → [B,H,W,Cout]. temb [1,1280] shared across streams.
    fn forward(self: *const Resnet, x: mlx.mlx_array, temb: mlx.mlx_array, s: S) !mlx.mlx_array {
        const h0 = try groupNorm(x, self.n1w, self.n1b, 32, 1e-5, s);
        defer rel(h0);
        const a0 = try silu(h0, s);
        defer rel(a0);
        var h = try conv2d(a0, self.conv1.w, self.conv1.b, 1, 1, s);
        // + time_emb_proj(silu(temb)) broadcast over H,W (and streams)
        {
            const st = try silu(temb, s);
            defer rel(st);
            const tp = try self.temb.forward(st, s); // [1,Cout]
            defer rel(tp);
            const csh = mlx.getShape(tp);
            const tp4 = try reshape(tp, &[_]c_int{ 1, 1, 1, csh[csh.len - 1] }, s);
            defer rel(tp4);
            const nh = try addA(h, tp4, s);
            rel(h);
            h = nh;
        }
        {
            const nh = try groupNorm(h, self.n2w, self.n2b, 32, 1e-5, s);
            rel(h);
            h = nh;
        }
        {
            const nh = try silu(h, s);
            rel(h);
            h = nh;
        }
        {
            const nh = try conv2d(h, self.conv2.w, self.conv2.b, 1, 1, s);
            rel(h);
            h = nh;
        }
        if (self.shortcut) |sc| {
            const scv = try conv2d(x, sc.w, sc.b, 1, 0, s); // 1x1, pad 0
            defer rel(scv);
            const out = try addA(h, scv, s);
            rel(h);
            return out;
        }
        const out = try addA(h, x, s);
        rel(h);
        return out;
    }
};

/// The base `BasicTransformerBlock` fields (present in main + dual) plus the
/// 2.5D siblings (main only). heads = C/64; cache_idx addresses the RefCache.
const TransformerBlock = struct {
    norm1w: mlx.mlx_array,
    norm1b: mlx.mlx_array,
    norm2w: mlx.mlx_array,
    norm2b: mlx.mlx_array,
    norm3w: mlx.mlx_array,
    norm3b: mlx.mlx_array,
    // attn1 (self / MDA albedo)
    a1_q: MixedLinear,
    a1_k: MixedLinear,
    a1_v: MixedLinear,
    a1_out: MixedLinear,
    // MDA mr projections (main only)
    a1_q_mr: ?MixedLinear = null,
    a1_k_mr: ?MixedLinear = null,
    a1_v_mr: ?MixedLinear = null,
    a1_out_mr: ?MixedLinear = null,
    // attn2 (cross to 77 text tokens)
    a2_q: MixedLinear,
    a2_k: MixedLinear,
    a2_v: MixedLinear,
    a2_out: MixedLinear,
    // ff (GeGLU)
    ff0: MixedLinear,
    ff2: MixedLinear,
    // MA (attn_multiview) — main only
    mv_q: ?MixedLinear = null,
    mv_k: ?MixedLinear = null,
    mv_v: ?MixedLinear = null,
    mv_out: ?MixedLinear = null,
    // RA (attn_refview) — main only; to_v_mr/to_out_mr are the per-material extras
    rv_q: ?MixedLinear = null,
    rv_k: ?MixedLinear = null,
    rv_v: ?MixedLinear = null,
    rv_out: ?MixedLinear = null,
    rv_v_mr: ?MixedLinear = null,
    rv_out_mr: ?MixedLinear = null,
    // DINO cross-attn — main only
    dino_q: ?MixedLinear = null,
    dino_k: ?MixedLinear = null,
    dino_v: ?MixedLinear = null,
    dino_out: ?MixedLinear = null,

    heads: c_int,
    cache_idx: usize,

    fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, C: c_int, heads: c_int, is_main: bool, cache_idx: usize, s: S) !TransformerBlock {
        const inner4: c_int = 4 * C; // ff.net.2 in-features (FF mult 4)
        const L = struct {
            fn lin(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8, in_f: c_int, ss: S) !MixedLinear {
                const kk = try hy3d.fmtKey(aa, "{s}.{s}", .{ p, sub });
                defer aa.free(kk);
                return MixedLinear.load(ww, aa, kk, @intCast(in_f), ss);
            }
            fn optLin(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8, in_f: c_int, ss: S) !MixedLinear {
                return lin(ww, aa, p, sub, in_f, ss);
            }
            fn nrm(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8, ss: S) !mlx.mlx_array {
                const kk = try hy3d.fmtKey(aa, "{s}.{s}", .{ p, sub });
                defer aa.free(kk);
                return hy3d.normVec(ww, kk, ss); // f32
            }
        };
        var tb: TransformerBlock = .{
            .norm1w = try L.nrm(w, a, prefix, "norm1.weight", s),
            .norm1b = try L.nrm(w, a, prefix, "norm1.bias", s),
            .norm2w = try L.nrm(w, a, prefix, "norm2.weight", s),
            .norm2b = try L.nrm(w, a, prefix, "norm2.bias", s),
            .norm3w = try L.nrm(w, a, prefix, "norm3.weight", s),
            .norm3b = try L.nrm(w, a, prefix, "norm3.bias", s),
            .a1_q = try L.lin(w, a, prefix, "attn1.to_q", C, s),
            .a1_k = try L.lin(w, a, prefix, "attn1.to_k", C, s),
            .a1_v = try L.lin(w, a, prefix, "attn1.to_v", C, s),
            .a1_out = try L.lin(w, a, prefix, "attn1.to_out", C, s),
            .a2_q = try L.lin(w, a, prefix, "attn2.to_q", C, s),
            .a2_k = try L.lin(w, a, prefix, "attn2.to_k", 1024, s),
            .a2_v = try L.lin(w, a, prefix, "attn2.to_v", 1024, s),
            .a2_out = try L.lin(w, a, prefix, "attn2.to_out", C, s),
            .ff0 = try L.lin(w, a, prefix, "ff.net.0.proj", C, s),
            .ff2 = try L.lin(w, a, prefix, "ff.net.2", inner4, s),
            .heads = heads,
            .cache_idx = cache_idx,
        };
        if (is_main) {
            tb.a1_q_mr = try L.optLin(w, a, prefix, "attn1.to_q_mr", C, s);
            tb.a1_k_mr = try L.optLin(w, a, prefix, "attn1.to_k_mr", C, s);
            tb.a1_v_mr = try L.optLin(w, a, prefix, "attn1.to_v_mr", C, s);
            tb.a1_out_mr = try L.optLin(w, a, prefix, "attn1.to_out_mr", C, s);
            tb.mv_q = try L.optLin(w, a, prefix, "attn_multiview.to_q", C, s);
            tb.mv_k = try L.optLin(w, a, prefix, "attn_multiview.to_k", C, s);
            tb.mv_v = try L.optLin(w, a, prefix, "attn_multiview.to_v", C, s);
            tb.mv_out = try L.optLin(w, a, prefix, "attn_multiview.to_out", C, s);
            tb.rv_q = try L.optLin(w, a, prefix, "attn_refview.to_q", C, s);
            tb.rv_k = try L.optLin(w, a, prefix, "attn_refview.to_k", C, s);
            tb.rv_v = try L.optLin(w, a, prefix, "attn_refview.to_v", C, s);
            tb.rv_out = try L.optLin(w, a, prefix, "attn_refview.to_out", C, s);
            tb.rv_v_mr = try L.optLin(w, a, prefix, "attn_refview.to_v_mr", C, s);
            tb.rv_out_mr = try L.optLin(w, a, prefix, "attn_refview.to_out_mr", C, s);
            tb.dino_q = try L.optLin(w, a, prefix, "attn_dino.to_q", C, s);
            tb.dino_k = try L.optLin(w, a, prefix, "attn_dino.to_k", 1024, s);
            tb.dino_v = try L.optLin(w, a, prefix, "attn_dino.to_v", 1024, s);
            tb.dino_out = try L.optLin(w, a, prefix, "attn_dino.to_out", C, s);
        }
        return tb;
    }
    fn deinit(self: *TransformerBlock) void {
        inline for (.{ "norm1w", "norm1b", "norm2w", "norm2b", "norm3w", "norm3b" }) |f| rel(@field(self, f));
        inline for (.{ "a1_q", "a1_k", "a1_v", "a1_out", "a2_q", "a2_k", "a2_v", "a2_out", "ff0", "ff2" }) |f| @field(self, f).deinit();
        inline for (.{ "a1_q_mr", "a1_k_mr", "a1_v_mr", "a1_out_mr", "mv_q", "mv_k", "mv_v", "mv_out", "rv_q", "rv_k", "rv_v", "rv_out", "rv_v_mr", "rv_out_mr", "dino_q", "dino_k", "dino_v", "dino_out" }) |f| {
            if (@field(self, f)) |*m| m.deinit();
        }
    }
};

/// diffusers Transformer2DModel wrapper (use_linear_projection): GroupNorm(1e-6)
/// → Linear proj_in → transformer block → Linear proj_out → residual add.
const AttnModule = struct {
    normw: mlx.mlx_array,
    normb: mlx.mlx_array,
    proj_in: MixedLinear,
    proj_out: MixedLinear,
    block: TransformerBlock,

    fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, C: c_int, heads: c_int, is_main: bool, cache_idx: usize, s: S) !AttnModule {
        const nw = try hy3d.fmtKey(a, "{s}.norm.weight", .{prefix});
        defer a.free(nw);
        const nb = try hy3d.fmtKey(a, "{s}.norm.bias", .{prefix});
        defer a.free(nb);
        const pin = try hy3d.fmtKey(a, "{s}.proj_in", .{prefix});
        defer a.free(pin);
        const pout = try hy3d.fmtKey(a, "{s}.proj_out", .{prefix});
        defer a.free(pout);
        const tbp = try hy3d.fmtKey(a, "{s}.transformer_blocks.0", .{prefix});
        defer a.free(tbp);
        return .{
            .normw = try hy3d.ownWeight(w, nw),
            .normb = try hy3d.ownWeight(w, nb),
            .proj_in = try MixedLinear.load(w, a, pin, @intCast(C), s),
            .proj_out = try MixedLinear.load(w, a, pout, @intCast(C), s),
            .block = try TransformerBlock.load(w, a, tbp, C, heads, is_main, cache_idx, s),
        };
    }
    fn deinit(self: *AttnModule) void {
        rel(self.normw);
        rel(self.normb);
        self.proj_in.deinit();
        self.proj_out.deinit();
        self.block.deinit();
    }
};

const DownBlock = struct {
    resnets: [2]Resnet,
    attns: ?[2]AttnModule,
    downsampler: ?Conv,
    fn deinit(self: *DownBlock) void {
        for (&self.resnets) |*r| r.deinit();
        if (self.attns) |*arr| for (arr) |*am| am.deinit();
        if (self.downsampler) |*c| c.deinit();
    }
};
const UpBlock = struct {
    resnets: [3]Resnet,
    attns: ?[3]AttnModule,
    upsampler: ?Conv,
    fn deinit(self: *UpBlock) void {
        for (&self.resnets) |*r| r.deinit();
        if (self.attns) |*arr| for (arr) |*am| am.deinit();
        if (self.upsampler) |*c| c.deinit();
    }
};
const MidBlock = struct {
    r0: Resnet,
    attn: AttnModule,
    r1: Resnet,
    fn deinit(self: *MidBlock) void {
        self.r0.deinit();
        self.attn.deinit();
        self.r1.deinit();
    }
};

pub const BaseUnet = struct {
    conv_in: Conv,
    te1: MixedLinear,
    te2: MixedLinear,
    downs: [4]DownBlock,
    mid: MidBlock,
    ups: [4]UpBlock,
    norm_out_w: mlx.mlx_array,
    norm_out_b: mlx.mlx_array,
    conv_out: Conv,

    fn load(w: *const Weights, a: std.mem.Allocator, is_main: bool, s: S) !BaseUnet {
        var u: BaseUnet = undefined;
        u.conv_in = try Conv.load(w, a, "conv_in");
        u.te1 = try MixedLinear.load(w, a, "time_embedding.linear_1", 320, s);
        u.te2 = try MixedLinear.load(w, a, "time_embedding.linear_2", 1280, s);

        const down_ch = [4]c_int{ 320, 640, 1280, 1280 };
        const down_attn = [4]bool{ true, true, true, false };
        for (0..4) |bi| {
            var db: DownBlock = .{ .resnets = undefined, .attns = null, .downsampler = null };
            for (0..2) |ri| {
                const p = try hy3d.fmtKey(a, "down_blocks.{d}.resnets.{d}", .{ bi, ri });
                defer a.free(p);
                db.resnets[ri] = try Resnet.load(w, a, p, s);
            }
            if (down_attn[bi]) {
                var attns: [2]AttnModule = undefined;
                for (0..2) |ai| {
                    const p = try hy3d.fmtKey(a, "down_blocks.{d}.attentions.{d}", .{ bi, ai });
                    defer a.free(p);
                    attns[ai] = try AttnModule.load(w, a, p, down_ch[bi], @divExact(down_ch[bi], HEAD_DIM), is_main, bi * 2 + ai, s);
                }
                db.attns = attns;
            }
            if (bi < 3) {
                const p = try hy3d.fmtKey(a, "down_blocks.{d}.downsamplers.0.conv", .{bi});
                defer a.free(p);
                db.downsampler = try Conv.load(w, a, p);
            }
            u.downs[bi] = db;
        }

        u.mid = .{
            .r0 = try Resnet.load(w, a, "mid_block.resnets.0", s),
            .attn = try AttnModule.load(w, a, "mid_block.attentions.0", 1280, 20, is_main, 6, s),
            .r1 = try Resnet.load(w, a, "mid_block.resnets.1", s),
        };

        const up_ch = [4]c_int{ 1280, 1280, 640, 320 };
        const up_attn = [4]bool{ false, true, true, true };
        for (0..4) |bi| {
            var ub: UpBlock = .{ .resnets = undefined, .attns = null, .upsampler = null };
            for (0..3) |ri| {
                const p = try hy3d.fmtKey(a, "up_blocks.{d}.resnets.{d}", .{ bi, ri });
                defer a.free(p);
                ub.resnets[ri] = try Resnet.load(w, a, p, s);
            }
            if (up_attn[bi]) {
                var attns: [3]AttnModule = undefined;
                for (0..3) |ai| {
                    const p = try hy3d.fmtKey(a, "up_blocks.{d}.attentions.{d}", .{ bi, ai });
                    defer a.free(p);
                    attns[ai] = try AttnModule.load(w, a, p, up_ch[bi], @divExact(up_ch[bi], HEAD_DIM), is_main, 7 + (bi - 1) * 3 + ai, s);
                }
                ub.attns = attns;
            }
            if (bi < 3) {
                const p = try hy3d.fmtKey(a, "up_blocks.{d}.upsamplers.0.conv", .{bi});
                defer a.free(p);
                ub.upsampler = try Conv.load(w, a, p);
            }
            u.ups[bi] = ub;
        }

        u.norm_out_w = try hy3d.ownWeight(w, "conv_norm_out.weight");
        u.norm_out_b = try hy3d.ownWeight(w, "conv_norm_out.bias");
        u.conv_out = try Conv.load(w, a, "conv_out");
        return u;
    }
    fn deinit(self: *BaseUnet) void {
        self.conv_in.deinit();
        self.te1.deinit();
        self.te2.deinit();
        for (&self.downs) |*d| d.deinit();
        self.mid.deinit();
        for (&self.ups) |*u| u.deinit();
        rel(self.norm_out_w);
        rel(self.norm_out_b);
        self.conv_out.deinit();
    }
};

// ── RefCache: 16 dual-UNet norm1 caches keyed by (section, block, attn) ────

/// Canonical index order: down_{0,1,2}_{0,1}=0..5, mid_0=6, up_{1,2,3}_{0,1,2}=7..15.
pub const RefCache = struct {
    entries: [16]?mlx.mlx_array = .{null} ** 16,

    fn set(self: *RefCache, idx: usize, arr: mlx.mlx_array) void {
        if (self.entries[idx]) |old| rel(old);
        self.entries[idx] = arr;
    }
    fn get(self: *const RefCache, idx: usize) mlx.mlx_array {
        return self.entries[idx].?;
    }
    /// Look up a canary by its dossier name ("down_0_0_0"/"mid_0_0"/"up_3_2_0").
    pub fn getByName(self: *const RefCache, name: []const u8) ?mlx.mlx_array {
        const idx = idxFromName(name) orelse return null;
        return self.entries[idx];
    }
    pub fn deinit(self: *RefCache) void {
        for (&self.entries) |*e| if (e.*) |arr| {
            rel(arr);
            e.* = null;
        };
    }
};

/// Parse "{down|mid|up}_{block}_{attn}_0" → canonical index.
fn idxFromName(name: []const u8) ?usize {
    var it = std.mem.splitScalar(u8, name, '_');
    const sec = it.next() orelse return null;
    if (std.mem.eql(u8, sec, "mid")) return 6;
    const b = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
    const at = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
    if (std.mem.eql(u8, sec, "down")) return b * 2 + at;
    if (std.mem.eql(u8, sec, "up")) return 7 + (b - 1) * 3 + at;
    return null;
}

// ── VoxelRopes: per-level 3D PoseRoPE cos/sin tables (built once/generation) ──

pub const VoxelRopes = struct {
    allocator: std.mem.Allocator,
    cos_mx: [4]mlx.mlx_array, // [1,1,6·grid²,64] f32, levels res 64/32/16/8
    sin_mx: [4]mlx.mlx_array,

    /// posmaps: [6,3,512,512] f32 (bg exactly 1.0), row-major view-then-CHW.
    pub fn build(allocator: std.mem.Allocator, posmaps: []const f32, s: S) !VoxelRopes {
        const grids = [4]usize{ 64, 32, 16, 8 };
        const vres = [4]usize{ 512, 256, 128, 64 };
        var vr: VoxelRopes = undefined;
        vr.allocator = allocator;
        for (0..4) |lv| {
            const vox = try voxelIndices(allocator, posmaps, 6, 512, 512, grids[lv], vres[lv]);
            defer allocator.free(vox);
            var rope = try rope3dFromVoxels(allocator, vox, @intCast(HEAD_DIM), vres[lv]);
            defer rope.deinit(allocator);
            const L: c_int = @intCast(vox.len / 3);
            vr.cos_mx[lv] = try ownHostF32(rope.cos, &[_]c_int{ 1, 1, L, HEAD_DIM }, s);
            errdefer rel(vr.cos_mx[lv]);
            vr.sin_mx[lv] = try ownHostF32(rope.sin, &[_]c_int{ 1, 1, L, HEAD_DIM }, s);
        }
        return vr;
    }
    fn forL(self: *const VoxelRopes, l: usize) struct { cos: mlx.mlx_array, sin: mlx.mlx_array } {
        const lv: usize = levelForL(l);
        return .{ .cos = self.cos_mx[lv], .sin = self.sin_mx[lv] };
    }
    pub fn deinit(self: *VoxelRopes) void {
        for (0..4) |lv| {
            rel(self.cos_mx[lv]);
            rel(self.sin_mx[lv]);
        }
    }
};

/// Latent spatial size l=H·W → pyramid level (64²→0, 32²→1, 16²→2, 8²→3).
fn levelForL(l: usize) usize {
    return switch (l) {
        4096 => 0,
        1024 => 1,
        256 => 2,
        64 => 3,
        else => unreachable,
    };
}

// ── Forward context + attention helpers ───────────────────────────────────

const Mode = enum { main, dual };

const FwdCtx = struct {
    mode: Mode,
    text: mlx.mlx_array, // [B_stream,77,1024] encoder text
    dino: mlx.mlx_array, // [B_stream,1028,1024] (main only)
    ref_cache: *RefCache,
    ref_scale: f32,
    ropes: ?*const VoxelRopes,
    n_pbr: c_int,
    n_views: c_int,
    s: S,
};

/// Cross-attention: q from `x`, k/v from `tokens`; head_dim 64.
fn crossAttn(q_lin: *const MixedLinear, k_lin: *const MixedLinear, v_lin: *const MixedLinear, out_lin: *const MixedLinear, x: mlx.mlx_array, tokens: mlx.mlx_array, heads: c_int, s: S) !mlx.mlx_array {
    const q = try q_lin.forward(x, s);
    defer rel(q);
    const k = try k_lin.forward(tokens, s);
    defer rel(k);
    const v = try v_lin.forward(tokens, s);
    defer rel(v);
    const qh = try splitHeads(q, heads, HEAD_DIM, s);
    defer rel(qh);
    const kh = try splitHeads(k, heads, HEAD_DIM, s);
    defer rel(kh);
    const vh = try splitHeads(v, heads, HEAD_DIM, s);
    defer rel(vh);
    const attn = try sdpa(qh, kh, vh, ATTN_SCALE, s);
    defer rel(attn);
    const merged = try mergeHeads(attn, s);
    defer rel(merged);
    return out_lin.forward(merged, s);
}

/// GeGLU: proj → chunk2 (value=first half, gate=second half) → value·gelu(gate).
fn ffForward(tb: *const TransformerBlock, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const p = try tb.ff0.forward(x, s); // [B,l,8C]
    defer rel(p);
    const nd = mlx.getShape(p).len;
    const two_inner = mlx.getShape(p)[nd - 1];
    const inner = @divExact(two_inner, 2);
    const val = try sliceAxis(p, nd - 1, 0, inner, s);
    defer rel(val);
    const gate = try sliceAxis(p, nd - 1, inner, two_inner, s);
    defer rel(gate);
    const g = try geluErf(gate, s);
    defer rel(g);
    const vg = try mulA(val, g, s);
    defer rel(vg);
    return tb.ff2.forward(vg, s);
}

/// MDA: per material, per view self-attn (seq = l only). albedo→attn1.to_*,
/// mr→attn1.to_*_mr. nh [n_pbr·n, l, c] → residual delta [n_pbr·n, l, c].
fn mdaForward(tb: *const TransformerBlock, nh: mlx.mlx_array, n_pbr: c_int, n_views: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(nh);
    const l = sh[1];
    const c = sh[2];
    const r4 = try reshape(nh, &[_]c_int{ n_pbr, n_views, l, c }, s);
    defer rel(r4);
    var outs: [2]mlx.mlx_array = undefined;
    var built: usize = 0;
    errdefer for (0..built) |i| rel(outs[i]);
    var m: c_int = 0;
    while (m < n_pbr) : (m += 1) {
        const nm4 = try sliceAxis(r4, 0, m, m + 1, s); // [1,n,l,c]
        defer rel(nm4);
        const nm = try reshape(nm4, &[_]c_int{ n_views, l, c }, s); // [n,l,c]
        defer rel(nm);
        const q_lin = if (m == 0) &tb.a1_q else &tb.a1_q_mr.?;
        const k_lin = if (m == 0) &tb.a1_k else &tb.a1_k_mr.?;
        const v_lin = if (m == 0) &tb.a1_v else &tb.a1_v_mr.?;
        const out_lin = if (m == 0) &tb.a1_out else &tb.a1_out_mr.?;
        outs[@intCast(m)] = try crossAttn(q_lin, k_lin, v_lin, out_lin, nm, nm, tb.heads, s);
        built += 1;
    }
    defer for (0..built) |i| rel(outs[i]);
    return concat(outs[0..@intCast(n_pbr)], 0, s); // material-major [n_pbr·n, l, c]
}

/// RA read: query = albedo slice [1,(n·l),c]; enc = ref cache [1,l,c]. Per-head
/// value = [to_v_h | to_v_mr_h] (128 wide, materials concat on the HEAD_DIM axis
/// AFTER head-splitting each — NOT a channel concat); q/k zero-padded to 128 for a
/// single equal-dim SDPA; the SDPA output splits back into the two 64-wide halves
/// per head → to_out (albedo) / to_out_mr (mr).
fn raForward(tb: *const TransformerBlock, nh: mlx.mlx_array, enc: mlx.mlx_array, n_pbr: c_int, n_views: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(nh);
    const l = sh[1];
    const c = sh[2];
    const heads = tb.heads;
    const nl = n_views * l;
    const r4 = try reshape(nh, &[_]c_int{ n_pbr, n_views, l, c }, s);
    defer rel(r4);
    const alb4 = try sliceAxis(r4, 0, 0, 1, s); // [1,n,l,c]
    defer rel(alb4);
    const query = try reshape(alb4, &[_]c_int{ 1, nl, c }, s); // [1,n·l,c]
    defer rel(query);

    const q = try tb.rv_q.?.forward(query, s);
    defer rel(q);
    const k = try tb.rv_k.?.forward(enc, s);
    defer rel(k);
    const v_alb = try tb.rv_v.?.forward(enc, s); // [1,l,c]
    defer rel(v_alb);
    const v_mr = try tb.rv_v_mr.?.forward(enc, s); // [1,l,c]
    defer rel(v_mr);

    const qh0 = try splitHeads(q, heads, HEAD_DIM, s); // [1,heads,n·l,64]
    defer rel(qh0);
    const kh0 = try splitHeads(k, heads, HEAD_DIM, s); // [1,heads,l,64]
    defer rel(kh0);
    // Value layout: CHANNEL-concat [to_v | to_v_mr] then a STRAIGHT head view
    // (reference reshape_qkv: value.view(B,L,H, value_width//H=128).transpose).
    // Head h therefore spans concat COLUMNS [128h, 128h+128) — with an odd
    // head count the middle head even straddles the material boundary. Looks
    // scrambled, but it is exactly the reshape training saw. Do NOT "fix"
    // this into a per-head [to_v_h|to_v_mr_h] interleave: that pairs
    // different value columns with each attention head than the trained
    // weights expect (adjudicated by oracle 5; see the raForward war-story in
    // tests/hy3d_paint_unet_dossier.md).
    const v_full = try concat(&[_]mlx.mlx_array{ v_alb, v_mr }, 2, s); // [1,l,2c]
    defer rel(v_full);
    const vh = try splitHeads(v_full, heads, 2 * HEAD_DIM, s); // [1,heads,l,128]
    defer rel(vh);
    const qh = try padLastDim(qh0, HEAD_DIM, s); // [1,heads,n·l,128]
    defer rel(qh);
    const kh = try padLastDim(kh0, HEAD_DIM, s);
    defer rel(kh);
    const attn = try sdpa(qh, kh, vh, ATTN_SCALE, s); // [1,heads,n·l,128]
    defer rel(attn);
    // Split each head's 128-wide value output into [0:64]=albedo, [64:128]=mr
    // BEFORE merging heads. The reference does `torch.split(out, head_dim, dim=-1)`
    // then transpose+reshape per half; merging first would interleave the two
    // materials across head boundaries (merged[0:c] mixes head0's mr with head1's
    // albedo for heads>1), scrambling the per-material to_out inputs.
    const alb_heads = try sliceAxis(attn, 3, 0, HEAD_DIM, s); // [1,heads,n·l,64]
    defer rel(alb_heads);
    const mr_heads = try sliceAxis(attn, 3, HEAD_DIM, 2 * HEAD_DIM, s);
    defer rel(mr_heads);
    const alb_merged = try mergeHeads(alb_heads, s); // [1,n·l,c]
    defer rel(alb_merged);
    const mr_merged = try mergeHeads(mr_heads, s);
    defer rel(mr_merged);
    const alb_out = try tb.rv_out.?.forward(alb_merged, s); // [1,n·l,c]
    defer rel(alb_out);
    const mr_out = try tb.rv_out_mr.?.forward(mr_merged, s);
    defer rel(mr_out);
    const st = try concat(&[_]mlx.mlx_array{ alb_out, mr_out }, 0, s); // [n_pbr,n·l,c]
    defer rel(st);
    const r4o = try reshape(st, &[_]c_int{ n_pbr, n_views, l, c }, s);
    defer rel(r4o);
    return reshape(r4o, &[_]c_int{ n_pbr * n_views, l, c }, s); // [n_pbr·n, l, c]
}

/// MA: cross-view self-attn per material. seq [n_pbr,(n·l),c] with 3D PoseRoPE
/// on q,k. nh [n_pbr·n, l, c] → delta [n_pbr·n, l, c].
fn maForward(tb: *const TransformerBlock, nh: mlx.mlx_array, n_pbr: c_int, n_views: c_int, rope: anytype, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(nh);
    const l = sh[1];
    const c = sh[2];
    const heads = tb.heads;
    const r4 = try reshape(nh, &[_]c_int{ n_pbr, n_views, l, c }, s);
    defer rel(r4);
    const seq = try reshape(r4, &[_]c_int{ n_pbr, n_views * l, c }, s); // [n_pbr,n·l,c]
    defer rel(seq);
    const q = try tb.mv_q.?.forward(seq, s);
    defer rel(q);
    const k = try tb.mv_k.?.forward(seq, s);
    defer rel(k);
    const v = try tb.mv_v.?.forward(seq, s);
    defer rel(v);
    const qh0 = try splitHeads(q, heads, HEAD_DIM, s);
    defer rel(qh0);
    const kh0 = try splitHeads(k, heads, HEAD_DIM, s);
    defer rel(kh0);
    const vh = try splitHeads(v, heads, HEAD_DIM, s);
    defer rel(vh);
    const qh = try ropeApply(qh0, rope.cos, rope.sin, s);
    defer rel(qh);
    const kh = try ropeApply(kh0, rope.cos, rope.sin, s);
    defer rel(kh);
    const attn = try sdpa(qh, kh, vh, ATTN_SCALE, s);
    defer rel(attn);
    const merged = try mergeHeads(attn, s); // [n_pbr,n·l,c]
    defer rel(merged);
    const out = try tb.mv_out.?.forward(merged, s);
    defer rel(out);
    const r4o = try reshape(out, &[_]c_int{ n_pbr, n_views, l, c }, s);
    defer rel(r4o);
    return reshape(r4o, &[_]c_int{ n_pbr * n_views, l, c }, s);
}

/// Main 2.5D transformer block (modules.py:472-707 order): MDA → RA → MA →
/// attn2 → DINO → FF. RA/MA reuse norm1's output; DINO reuses norm2's output.
fn transformerBlockMain(tb: *const TransformerBlock, h_in: mlx.mlx_array, ctx: *const FwdCtx, l: usize, s: S) !mlx.mlx_array {
    var h = try cloneRef(h_in);
    errdefer rel(h);
    const nh = try layerNorm(h_in, tb.norm1w, tb.norm1b, s);
    defer rel(nh);
    // MDA (attn1 as multi-material diffusion attention)
    {
        const mda = try mdaForward(tb, nh, ctx.n_pbr, ctx.n_views, s);
        defer rel(mda);
        const nx = try addA(h, mda, s);
        rel(h);
        h = nx;
    }
    // RA read (reference attention), scaled by ref_scale
    {
        const enc = ctx.ref_cache.get(tb.cache_idx);
        const ra = try raForward(tb, nh, enc, ctx.n_pbr, ctx.n_views, s);
        defer rel(ra);
        const scaled = try scalarMul(ra, ctx.ref_scale, s);
        defer rel(scaled);
        const nx = try addA(h, scaled, s);
        rel(h);
        h = nx;
    }
    // MA (multiview attention with PoseRoPE); mva_scale 1.0
    {
        const rope = ctx.ropes.?.forL(l);
        const ma = try maForward(tb, nh, ctx.n_pbr, ctx.n_views, rope, s);
        defer rel(ma);
        const nx = try addA(h, ma, s);
        rel(h);
        h = nx;
    }
    // norm2 → attn2 (cross to 77 text) → DINO (same norm2 output)
    const nh2 = try layerNorm(h, tb.norm2w, tb.norm2b, s);
    defer rel(nh2);
    {
        const a2 = try crossAttn(&tb.a2_q, &tb.a2_k, &tb.a2_v, &tb.a2_out, nh2, ctx.text, tb.heads, s);
        defer rel(a2);
        const nx = try addA(h, a2, s);
        rel(h);
        h = nx;
    }
    {
        const d = try crossAttn(&tb.dino_q.?, &tb.dino_k.?, &tb.dino_v.?, &tb.dino_out.?, nh2, ctx.dino, tb.heads, s);
        defer rel(d);
        const nx = try addA(h, d, s);
        rel(h);
        h = nx;
    }
    // norm3 → FF
    const nh3 = try layerNorm(h, tb.norm3w, tb.norm3b, s);
    defer rel(nh3);
    {
        const ff = try ffForward(tb, nh3, s);
        defer rel(ff);
        const nx = try addA(h, ff, s);
        rel(h);
        h = nx;
    }
    return h;
}

/// Dual (reference-stream) block: plain BasicTransformerBlock + the RA-write
/// cache of norm1's output (mode "w").
fn transformerBlockDual(tb: *const TransformerBlock, h_in: mlx.mlx_array, ctx: *const FwdCtx, s: S) !mlx.mlx_array {
    var h = try cloneRef(h_in);
    errdefer rel(h);
    const nh = try layerNorm(h_in, tb.norm1w, tb.norm1b, s);
    defer rel(nh);
    // RA write: condition_embed_dict[layer] = rearrange(nh,"(b n) l c -> b (n l) c")
    // with n=1 → identical [1,l,c]. Materialize so it survives this forward.
    {
        const entry = try contig(nh, s);
        _ = mlx.mlx_array_eval(entry);
        ctx.ref_cache.set(tb.cache_idx, entry);
    }
    // attn1 self-attention
    {
        const a1 = try crossAttn(&tb.a1_q, &tb.a1_k, &tb.a1_v, &tb.a1_out, nh, nh, tb.heads, s);
        defer rel(a1);
        const nx = try addA(h, a1, s);
        rel(h);
        h = nx;
    }
    const nh2 = try layerNorm(h, tb.norm2w, tb.norm2b, s);
    defer rel(nh2);
    {
        const a2 = try crossAttn(&tb.a2_q, &tb.a2_k, &tb.a2_v, &tb.a2_out, nh2, ctx.text, tb.heads, s);
        defer rel(a2);
        const nx = try addA(h, a2, s);
        rel(h);
        h = nx;
    }
    const nh3 = try layerNorm(h, tb.norm3w, tb.norm3b, s);
    defer rel(nh3);
    {
        const ff = try ffForward(tb, nh3, s);
        defer rel(ff);
        const nx = try addA(h, ff, s);
        rel(h);
        h = nx;
    }
    return h;
}

/// Transformer2DModel wrapper forward around a stream batch [B,H,W,C].
fn attnModuleForward(am: *const AttnModule, x: mlx.mlx_array, ctx: *const FwdCtx, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const B = sh[0];
    const H = sh[1];
    const Wd = sh[2];
    const C = sh[3];
    const normed = try groupNorm(x, am.normw, am.normb, 32, 1e-6, s);
    defer rel(normed);
    const flat = try reshape(normed, &[_]c_int{ B, H * Wd, C }, s);
    defer rel(flat);
    var hh = try am.proj_in.forward(flat, s); // [B,HW,C]
    {
        const t = switch (ctx.mode) {
            .main => try transformerBlockMain(&am.block, hh, ctx, @intCast(H * Wd), s),
            .dual => try transformerBlockDual(&am.block, hh, ctx, s),
        };
        rel(hh);
        hh = t;
    }
    {
        const po = try am.proj_out.forward(hh, s);
        rel(hh);
        hh = po;
    }
    const back = try reshape(hh, &[_]c_int{ B, H, Wd, C }, s);
    rel(hh);
    defer rel(back);
    return addA(back, x, s);
}

fn upsample(x: mlx.mlx_array, cv: Conv, s: S) !mlx.mlx_array {
    var r1 = mlx.mlx_array_new();
    defer rel(r1);
    try mlx.check(mlx.mlx_repeat_axis(&r1, x, 2, 1, s));
    var r2 = mlx.mlx_array_new();
    defer rel(r2);
    try mlx.check(mlx.mlx_repeat_axis(&r2, r1, 2, 2, s));
    return conv2d(r2, cv.w, cv.b, 1, 1, s);
}

/// Full base UNet forward on NHWC input [B,H,W,Cin] → NHWC output [B,H,W,4].
fn baseForward(a: std.mem.Allocator, unet: *const BaseUnet, x0: mlx.mlx_array, temb: mlx.mlx_array, ctx: *const FwdCtx, s: S) !mlx.mlx_array {
    var h = try conv2d(x0, unet.conv_in.w, unet.conv_in.b, 1, 1, s);
    errdefer rel(h);

    var res: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (res.items) |r| rel(r);
        res.deinit(a);
    }
    try res.append(a, try cloneRef(h)); // residual[0] = conv_in output

    // down blocks
    for (0..4) |bi| {
        const db = &unet.downs[bi];
        for (0..2) |ri| {
            {
                const nh = try db.resnets[ri].forward(h, temb, s);
                rel(h);
                h = nh;
            }
            if (db.attns) |*attns| {
                const nh = try attnModuleForward(&attns[ri], h, ctx, s);
                rel(h);
                h = nh;
            }
            try res.append(a, try cloneRef(h));
        }
        if (db.downsampler) |ds| {
            const nh = try conv2d(h, ds.w, ds.b, 2, 1, s); // stride 2, pad 1
            rel(h);
            h = nh;
            try res.append(a, try cloneRef(h));
        }
    }

    // mid block
    {
        const nh = try unet.mid.r0.forward(h, temb, s);
        rel(h);
        h = nh;
    }
    {
        const nh = try attnModuleForward(&unet.mid.attn, h, ctx, s);
        rel(h);
        h = nh;
    }
    {
        const nh = try unet.mid.r1.forward(h, temb, s);
        rel(h);
        h = nh;
    }

    // up blocks (pop residuals LIFO, concat on channel before each resnet)
    for (0..4) |bi| {
        const ub = &unet.ups[bi];
        for (0..3) |ri| {
            const r = res.pop().?;
            {
                const cat = try concat(&[_]mlx.mlx_array{ h, r }, 3, s);
                rel(h);
                rel(r);
                h = cat;
            }
            {
                const nh = try ub.resnets[ri].forward(h, temb, s);
                rel(h);
                h = nh;
            }
            if (ub.attns) |*attns| {
                const nh = try attnModuleForward(&attns[ri], h, ctx, s);
                rel(h);
                h = nh;
            }
        }
        if (ub.upsampler) |us| {
            const nh = try upsample(h, us, s);
            rel(h);
            h = nh;
        }
    }

    // conv_norm_out → silu → conv_out
    {
        const nh = try groupNorm(h, unet.norm_out_w, unet.norm_out_b, 32, 1e-5, s);
        rel(h);
        h = nh;
    }
    {
        const nh = try silu(h, s);
        rel(h);
        h = nh;
    }
    {
        const nh = try conv2d(h, unet.conv_out.w, unet.conv_out.b, 1, 1, s);
        rel(h);
        h = nh;
    }
    return h;
}

// ── PaintUnet: the public engine ──────────────────────────────────────────

pub const PaintUnet = struct {
    allocator: std.mem.Allocator,
    s: S,
    main: BaseUnet,
    dual: BaseUnet,
    learned_albedo: mlx.mlx_array, // [77,1024]
    learned_mr: mlx.mlx_array,
    learned_ref: mlx.mlx_array,

    pub fn load(allocator: std.mem.Allocator, model_dir: []const u8, s: S) !*PaintUnet {
        const self = try allocator.create(PaintUnet);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.s = s;

        var wm = try hy3d.loadFileWeights(allocator, model_dir, "unet.safetensors");
        defer wm.deinit();
        self.main = try BaseUnet.load(&wm, allocator, true, s);
        self.learned_albedo = try hy3d.ownWeight(&wm, "learned_text_clip_albedo");
        self.learned_mr = try hy3d.ownWeight(&wm, "learned_text_clip_mr");
        self.learned_ref = try hy3d.ownWeight(&wm, "learned_text_clip_ref");

        var wd = try hy3d.loadFileWeights(allocator, model_dir, "unet_dual.safetensors");
        defer wd.deinit();
        self.dual = try BaseUnet.load(&wd, allocator, false, s);
        return self;
    }

    pub fn deinit(self: *PaintUnet) void {
        self.main.deinit();
        self.dual.deinit();
        rel(self.learned_albedo);
        rel(self.learned_mr);
        rel(self.learned_ref);
        self.allocator.destroy(self);
    }

    fn timeEmbedding(self: *PaintUnet, unet: *const BaseUnet, t: f32) !mlx.mlx_array {
        const emb = try timestepEmbed(self.allocator, t, 320);
        defer self.allocator.free(emb);
        const arr = try ownHostF32(emb, &[_]c_int{ 1, 320 }, self.s);
        defer rel(arr);
        var h = try unet.te1.forward(arr, self.s);
        {
            const nh = try silu(h, self.s);
            rel(h);
            h = nh;
        }
        {
            const nh = try unet.te2.forward(h, self.s);
            rel(h);
            h = nh;
        }
        return h; // [1,1280]
    }

    /// Run the dual UNet once (t=0, encoder = learned_text_clip_ref) over the
    /// reference latents [1,4,64,64] NCHW to fill the 16-entry RefCache.
    pub fn buildRefCache(self: *PaintUnet, ref_latents: mlx.mlx_array) !RefCache {
        const s = self.s;
        var cache = RefCache{};
        errdefer cache.deinit();
        const nhwc = try transpose(ref_latents, &[_]c_int{ 0, 2, 3, 1 }, s); // [1,64,64,4]
        defer rel(nhwc);
        const temb = try self.timeEmbedding(&self.dual, 0.0);
        defer rel(temb);
        const text = try expandDims(self.learned_ref, 0, s); // [1,77,1024]
        defer rel(text);
        var ctx = FwdCtx{ .mode = .dual, .text = text, .dino = undefined, .ref_cache = &cache, .ref_scale = 0, .ropes = null, .n_pbr = 1, .n_views = 1, .s = s };
        const v = try baseForward(self.allocator, &self.dual, nhwc, temb, &ctx, s);
        rel(v); // output discarded; the cache is the product
        return cache;
    }

    /// One 2.5D denoising step. latents [12,4,64,64] NCHW (material-major:
    /// material 0 views 0-5, then material 1). embeds_normal/position [6,4,64,64]
    /// (per view, shared across materials). dino_proj [1,1028,1024] (null→zeros).
    /// Returns v [12,4,64,64] NCHW f32.
    pub fn forward(self: *PaintUnet, latents: mlx.mlx_array, t: f32, embeds_normal: mlx.mlx_array, embeds_position: mlx.mlx_array, dino_proj: ?mlx.mlx_array, ref_cache: *const RefCache, ref_scale: f32, voxel_ropes: *const VoxelRopes) !mlx.mlx_array {
        const s = self.s;
        // Build the 12-ch input: [latent | normal | position] per stream; normal
        // and position are tiled across the two materials (material-major).
        const nt = try concat(&[_]mlx.mlx_array{ embeds_normal, embeds_normal }, 0, s); // [12,4,64,64]
        defer rel(nt);
        const pt = try concat(&[_]mlx.mlx_array{ embeds_position, embeds_position }, 0, s);
        defer rel(pt);
        const x12 = try concat(&[_]mlx.mlx_array{ latents, nt, pt }, 1, s); // [12,12,64,64]
        defer rel(x12);
        const nhwc = try transpose(x12, &[_]c_int{ 0, 2, 3, 1 }, s); // [12,64,64,12]
        defer rel(nhwc);

        const temb = try self.timeEmbedding(&self.main, t);
        defer rel(temb);
        const text = try self.buildText(s);
        defer rel(text);
        const dino = try self.buildDino(dino_proj, s);
        defer rel(dino);

        var ctx = FwdCtx{ .mode = .main, .text = text, .dino = dino, .ref_cache = @constCast(ref_cache), .ref_scale = ref_scale, .ropes = voxel_ropes, .n_pbr = 2, .n_views = 6, .s = s };
        const out_nhwc = try baseForward(self.allocator, &self.main, nhwc, temb, &ctx, s); // [12,64,64,4]
        defer rel(out_nhwc);
        const nchw = try transpose(out_nhwc, &[_]c_int{ 0, 3, 1, 2 }, s);
        defer rel(nchw);
        const nchw_c = try contig(nchw, s);
        defer rel(nchw_c);
        const f32out = try astype(nchw_c, .float32, s);
        _ = mlx.mlx_array_eval(f32out);
        return f32out;
    }

    /// Encoder text [12,77,1024]: streams 0-5 albedo, 6-11 mr (material-major).
    fn buildText(self: *PaintUnet, s: S) !mlx.mlx_array {
        const alb1 = try expandDims(self.learned_albedo, 0, s);
        defer rel(alb1);
        const alb6 = try broadcastTo(alb1, &[_]c_int{ 6, 77, 1024 }, s);
        defer rel(alb6);
        const mr1 = try expandDims(self.learned_mr, 0, s);
        defer rel(mr1);
        const mr6 = try broadcastTo(mr1, &[_]c_int{ 6, 77, 1024 }, s);
        defer rel(mr6);
        const cat = try concat(&[_]mlx.mlx_array{ alb6, mr6 }, 0, s);
        defer rel(cat);
        return contig(cat, s);
    }

    /// DINO tokens [12,1028,1024]: the projected image tokens repeated per stream,
    /// or zeros for the unconditional CFG batch.
    fn buildDino(self: *PaintUnet, dino_proj: ?mlx.mlx_array, s: S) !mlx.mlx_array {
        _ = self;
        if (dino_proj) |dp| {
            const b = try broadcastTo(dp, &[_]c_int{ 12, 1028, 1024 }, s);
            defer rel(b);
            return contig(b, s);
        }
        const zf = mlx.mlx_array_new_float(0.0);
        defer rel(zf);
        const zh = try astype(zf, .float16, s);
        defer rel(zh);
        const b = try broadcastTo(zh, &[_]c_int{ 12, 1028, 1024 }, s);
        defer rel(b);
        return contig(b, s);
    }
};

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic (GeGLU order, cache/level maps) + env-gated oracles.
// ════════════════════════════════════════════════════════════════════════

test "hy3d-paint-unet: cache index + level maps (residual/attention order)" {
    try testing.expectEqual(@as(?usize, 0), idxFromName("down_0_0_0"));
    try testing.expectEqual(@as(?usize, 5), idxFromName("down_2_1_0"));
    try testing.expectEqual(@as(?usize, 6), idxFromName("mid_0_0"));
    try testing.expectEqual(@as(?usize, 7), idxFromName("up_1_0_0"));
    try testing.expectEqual(@as(?usize, 15), idxFromName("up_3_2_0"));
    try testing.expectEqual(@as(usize, 0), levelForL(4096));
    try testing.expectEqual(@as(usize, 1), levelForL(1024));
    try testing.expectEqual(@as(usize, 2), levelForL(256));
    try testing.expectEqual(@as(usize, 3), levelForL(64));
}

test "hy3d-paint-unet: GeGLU applies the activation to the SECOND half (gate)" {
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    // ff.net.0.proj as identity [8,4] (value rows first, gate rows second) so the
    // FF's chunk splits a known [val | gate] vector; ff.net.2 as identity [4,4].
    const C: c_int = 4;
    const inner: c_int = 4; // 2*inner = 8
    // identity weight [out,in]: proj is [8,4] identity-ish -> we want proj(x)=x
    // padded to 8 by stacking [x ; x] so value=x, gate=x, out should be x*gelu(x).
    // Build ff0 weight [8,4] = [I4 ; I4]; ff2 weight [4,4] = I4.
    var proj = [_]f32{0} ** 32;
    for (0..4) |i| {
        proj[i * 4 + i] = 1.0; // top I4 (value rows)
        proj[(i + 4) * 4 + i] = 1.0; // bottom I4 (gate rows)
    }
    var eye = [_]f32{0} ** 16;
    for (0..4) |i| eye[i * 4 + i] = 1.0;

    var w = Weights.init(a);
    defer w.deinit();
    const pw = mlx.mlx_array_new_data(&proj, &[_]c_int{ 8, 4 }, 2, .float32);
    const pwh = try astype(pw, .float16, s);
    _ = mlx.mlx_array_eval(pwh);
    rel(pw);
    const ew = mlx.mlx_array_new_data(&eye, &[_]c_int{ 4, 4 }, 2, .float32);
    const ewh = try astype(ew, .float16, s);
    _ = mlx.mlx_array_eval(ewh);
    rel(ew);
    try w.map.put(try a.dupe(u8, "ff.net.0.proj.weight"), pwh);
    try w.map.put(try a.dupe(u8, "ff.net.2.weight"), ewh);

    const ff0 = try MixedLinear.load(&w, a, "ff.net.0.proj", @intCast(C), s);
    const ff2 = try MixedLinear.load(&w, a, "ff.net.2", @intCast(inner), s);
    var tb: TransformerBlock = undefined;
    tb.ff0 = ff0;
    tb.ff2 = ff2;
    defer {
        tb.ff0.deinit();
        tb.ff2.deinit();
    }

    var xv = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    const xa = mlx.mlx_array_new_data(&xv, &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer rel(xa);
    const out = try ffForward(&tb, xa, s);
    defer rel(out);
    const of = try astype(out, .float32, s);
    defer rel(of);
    _ = mlx.mlx_array_eval(of);
    const od = mlx.mlx_array_data_float32(of) orelse return error.NoData;
    for (0..4) |i| {
        const g = 0.5 * xv[i] * (1.0 + erff(xv[i] * 0.7071067811865476)); // gelu(gate=x)
        const expect = xv[i] * g; // value(=x) * gelu(gate=x)
        try testing.expectApproxEqAbs(expect, od[i], 2e-3);
    }
}

fn erff(x: f32) f32 {
    // Abramowitz-Stegun 7.1.26 — reference for the GeGLU test only.
    const t = 1.0 / (1.0 + 0.3275911 * @abs(x));
    const y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * @exp(-x * x);
    return if (x < 0) -y else y;
}

test "hy3d-paint-unet: RA splits per-head value output before merging heads" {
    // Regression guard for the reference-attention material split. The reference
    // does `torch.split(sdpa_out, head_dim=64, dim=-1)` (per head) then transpose+
    // reshape each half — so the albedo/mr separation happens BEFORE heads merge.
    // A merge-first-then-slice impl interleaves head boundaries and scrambles the
    // per-material to_out inputs (invisible to finiteness checks; only a numeric
    // oracle or this test catches it). heads=2, C=128, L_ref=1 (softmax≡1 → the RA
    // output equals the split value), n_pbr=2, n_views=1, l=1.
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    var eye = [_]f32{0} ** (128 * 128);
    var two_eye = [_]f32{0} ** (128 * 128);
    for (0..128) |i| {
        eye[i * 128 + i] = 1.0;
        two_eye[i * 128 + i] = 2.0;
    }
    var w = Weights.init(a);
    defer w.deinit();
    const put = struct {
        fn f(ww: *Weights, al: std.mem.Allocator, key: []const u8, data: []const f32, ss: S) !void {
            const arr = mlx.mlx_array_new_data(data.ptr, &[_]c_int{ 128, 128 }, 2, .float32);
            const h = try astype(arr, .float16, ss);
            _ = mlx.mlx_array_eval(h);
            rel(arr);
            try ww.map.put(try al.dupe(u8, key), h);
        }
    }.f;
    // rv_v = I → albedo value = enc (ones); rv_v_mr = 2·I → mr value = 2·enc. With
    // heads=2 the value reshape makes head0=to_v, head1=to_v_mr; the correct split
    // gives albedo = [to_v[0:64] | to_v_mr[0:64]] = [1×64, 2×64] (a merge-first impl
    // would give pure to_v = [1×128]). rv_q/rv_k/rv_out/rv_out_mr are identity.
    try put(&w, a, "rv.to_q.weight", &eye, s);
    try put(&w, a, "rv.to_k.weight", &eye, s);
    try put(&w, a, "rv.to_v.weight", &eye, s);
    try put(&w, a, "rv.to_v_mr.weight", &two_eye, s);
    try put(&w, a, "rv.to_out.weight", &eye, s);
    try put(&w, a, "rv.to_out_mr.weight", &eye, s);

    var tb: TransformerBlock = undefined;
    tb.heads = 2;
    tb.rv_q = try MixedLinear.load(&w, a, "rv.to_q", 128, s);
    tb.rv_k = try MixedLinear.load(&w, a, "rv.to_k", 128, s);
    tb.rv_v = try MixedLinear.load(&w, a, "rv.to_v", 128, s);
    tb.rv_v_mr = try MixedLinear.load(&w, a, "rv.to_v_mr", 128, s);
    tb.rv_out = try MixedLinear.load(&w, a, "rv.to_out", 128, s);
    tb.rv_out_mr = try MixedLinear.load(&w, a, "rv.to_out_mr", 128, s);
    defer {
        tb.rv_q.?.deinit();
        tb.rv_k.?.deinit();
        tb.rv_v.?.deinit();
        tb.rv_v_mr.?.deinit();
        tb.rv_out.?.deinit();
        tb.rv_out_mr.?.deinit();
    }

    var ones_nh = [_]f32{1} ** (2 * 128); // nh [n_pbr·n_views, l, c] = [2,1,128]
    const nh = mlx.mlx_array_new_data(&ones_nh, &[_]c_int{ 2, 1, 128 }, 3, .float32);
    defer rel(nh);
    var ones_enc = [_]f32{1} ** 128; // enc [1, l_ref, c] = [1,1,128]
    const enc = mlx.mlx_array_new_data(&ones_enc, &[_]c_int{ 1, 1, 128 }, 3, .float32);
    defer rel(enc);

    const out = try raForward(&tb, nh, enc, 2, 1, s); // [n_pbr·n_views, l, c] = [2,1,128]
    defer rel(out);
    const of = try astype(out, .float32, s);
    defer rel(of);
    _ = mlx.mlx_array_eval(of);
    const d = mlx.mlx_array_data_float32(of) orelse return error.NoData;
    // Row 0 = albedo out: to_v[0:64]=1 then to_v_mr[0:64]=2 (merge-first bug → all 1).
    try testing.expectApproxEqAbs(@as(f32, 1.0), d[0], 1e-2);
    try testing.expectApproxEqAbs(@as(f32, 2.0), d[64], 1e-2);
    // Row 1 = mr out: to_v[64:128]=1 then to_v_mr[64:128]=2.
    try testing.expectApproxEqAbs(@as(f32, 1.0), d[128], 1e-2);
    try testing.expectApproxEqAbs(@as(f32, 2.0), d[128 + 64], 1e-2);
}

// ── Env-gated oracles (fixtures from dump_hunyuan3d_paint_fixtures.py --with-unet) ──

fn oracleIo() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}
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
fn cosineArr(arr: mlx.mlx_array, ref: []const f32, s: S) !f64 {
    const f = try astype(arr, .float32, s);
    defer rel(f);
    _ = mlx.mlx_array_eval(f);
    const n: usize = @intCast(mlx.mlx_array_size(f));
    try testing.expectEqual(ref.len, n);
    const d = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..n) |i| {
        dot += @as(f64, d[i]) * ref[i];
        na += @as(f64, d[i]) * d[i];
        nb += @as(f64, ref[i]) * ref[i];
    }
    return dot / (@sqrt(na) * @sqrt(nb));
}

test "hy3d-paint-unet: smoke-load both UNets (fp16 shapes)" {
    const model_dir = std.mem.span(std.c.getenv("HY3DP_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    var pu = try PaintUnet.load(a, model_dir, s);
    defer pu.deinit();
    // conv_in channel count differs main (12) vs dual (4); OHWI [O,kH,kW,I].
    try testing.expectEqualSlices(c_int, &[_]c_int{ 320, 3, 3, 12 }, mlx.getShape(pu.main.conv_in.w));
    try testing.expectEqualSlices(c_int, &[_]c_int{ 320, 3, 3, 4 }, mlx.getShape(pu.dual.conv_in.w));
    try testing.expectEqualSlices(c_int, &[_]c_int{ 77, 1024 }, mlx.getShape(pu.learned_albedo));
    // main has the 2.5D siblings; dual does not.
    try testing.expect(pu.main.downs[0].attns.?[0].block.mv_q != null);
    try testing.expect(pu.dual.downs[0].attns.?[0].block.mv_q == null);
    std.debug.print("[hy3d-paint-unet] smoke-load OK (main+dual, fp16)\n", .{});
}

test "hy3d-paint-unet oracle: dual-UNet ref cache matches reference" {
    const model_dir = std.mem.span(std.c.getenv("HY3DP_TEST_MODEL") orelse return error.SkipZigTest);
    const in_p = std.mem.span(std.c.getenv("HY3DP_REF_IN") orelse return error.SkipZigTest);
    const down_p = std.mem.span(std.c.getenv("HY3DP_REF_DOWN") orelse return error.SkipZigTest);
    const mid_p = std.mem.span(std.c.getenv("HY3DP_REF_MID") orelse return error.SkipZigTest);
    const up_p = std.mem.span(std.c.getenv("HY3DP_REF_UP") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = oracleIo();
    const s = mlx.mlx_default_gpu_stream_new();
    var pu = try PaintUnet.load(a, model_dir, s);
    defer pu.deinit();

    const ref_in = try readF32(io, a, in_p);
    defer a.free(ref_in);
    const g_down = try readF32(io, a, down_p);
    defer a.free(g_down);
    const g_mid = try readF32(io, a, mid_p);
    defer a.free(g_mid);
    const g_up = try readF32(io, a, up_p);
    defer a.free(g_up);

    const ri = mlx.mlx_array_new_data(ref_in.ptr, &[_]c_int{ 1, 4, 64, 64 }, 4, .float32);
    defer rel(ri);
    var cache = try pu.buildRefCache(ri);
    defer cache.deinit();

    const c_down = try cosineArr(cache.getByName("down_0_0_0").?, g_down, s);
    const c_mid = try cosineArr(cache.getByName("mid_0_0").?, g_mid, s);
    const c_up = try cosineArr(cache.getByName("up_3_2_0").?, g_up, s);
    std.debug.print("[hy3d-paint-unet] ref-cache cos down={d:.5} mid={d:.5} up={d:.5}\n", .{ c_down, c_mid, c_up });
    try testing.expect(c_down > 0.995);
    try testing.expect(c_mid > 0.995);
    try testing.expect(c_up > 0.995);
}

test "hy3d-paint-unet oracle: full 2.5D step matches reference" {
    const model_dir = std.mem.span(std.c.getenv("HY3DP_TEST_MODEL") orelse return error.SkipZigTest);
    const in_p = std.mem.span(std.c.getenv("HY3DP_REF_IN") orelse return error.SkipZigTest);
    const lat_p = std.mem.span(std.c.getenv("HY3DP_UNET_LATENTS") orelse return error.SkipZigTest);
    const nrm_p = std.mem.span(std.c.getenv("HY3DP_UNET_NORMAL") orelse return error.SkipZigTest);
    const pos_p = std.mem.span(std.c.getenv("HY3DP_UNET_POSITION") orelse return error.SkipZigTest);
    const pmap_p = std.mem.span(std.c.getenv("HY3DP_UNET_POSMAPS") orelse return error.SkipZigTest);
    const dino_p = std.mem.span(std.c.getenv("HY3DP_DINO_PROJ") orelse return error.SkipZigTest);
    const v_p = std.mem.span(std.c.getenv("HY3DP_UNET_V") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = oracleIo();
    const s = mlx.mlx_default_gpu_stream_new();
    var pu = try PaintUnet.load(a, model_dir, s);
    defer pu.deinit();

    const ref_in = try readF32(io, a, in_p);
    defer a.free(ref_in);
    const lat = try readF32(io, a, lat_p); // [1,2,6,4,64,64] → [12,4,64,64]
    defer a.free(lat);
    const nrm = try readF32(io, a, nrm_p); // [1,6,4,64,64] → [6,4,64,64]
    defer a.free(nrm);
    const pos = try readF32(io, a, pos_p);
    defer a.free(pos);
    const pmap = try readF32(io, a, pmap_p); // [1,6,3,512,512] → [6,3,512,512]
    defer a.free(pmap);
    const dproj = try readF32(io, a, dino_p); // [1,1028,1024]
    defer a.free(dproj);
    const g_v = try readF32(io, a, v_p); // [12,4,64,64]
    defer a.free(g_v);

    const ri = mlx.mlx_array_new_data(ref_in.ptr, &[_]c_int{ 1, 4, 64, 64 }, 4, .float32);
    defer rel(ri);
    var cache = try pu.buildRefCache(ri);
    defer cache.deinit();

    var ropes = try VoxelRopes.build(a, pmap, s);
    defer ropes.deinit();

    const la = mlx.mlx_array_new_data(lat.ptr, &[_]c_int{ 12, 4, 64, 64 }, 4, .float32);
    defer rel(la);
    const na = mlx.mlx_array_new_data(nrm.ptr, &[_]c_int{ 6, 4, 64, 64 }, 4, .float32);
    defer rel(na);
    const pa = mlx.mlx_array_new_data(pos.ptr, &[_]c_int{ 6, 4, 64, 64 }, 4, .float32);
    defer rel(pa);
    const da = mlx.mlx_array_new_data(dproj.ptr, &[_]c_int{ 1, 1028, 1024 }, 3, .float32);
    defer rel(da);

    const v = try pu.forward(la, 999.0, na, pa, da, &cache, 1.0, &ropes);
    defer rel(v);
    const cos = try cosineArr(v, g_v, s);
    std.debug.print("[hy3d-paint-unet] full-step cos={d:.5}\n", .{cos});
    try testing.expect(cos > 0.995);
}

// Wiring smoke: no golden — runs buildRefCache + a full 2.5D forward on
// synthetic inputs against real weights, asserting the output shape [12,4,64,64]
// and finiteness. Catches shape/reshape/attention-wiring bugs the hermetic
// tests can't (they never touch the GPU forward). Opt-in via HY3DP_FWD_SMOKE.
test "hy3d-paint-unet: full forward runs end-to-end (shape + finite)" {
    const model_dir = std.mem.span(std.c.getenv("HY3DP_TEST_MODEL") orelse return error.SkipZigTest);
    if (std.c.getenv("HY3DP_FWD_SMOKE") == null) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    var pu = try PaintUnet.load(a, model_dir, s);
    defer pu.deinit();

    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const mk = struct {
        fn buf(al: std.mem.Allocator, r: std.Random, n: usize, scale: f32) ![]f32 {
            const b = try al.alloc(f32, n);
            for (b) |*x| x.* = (r.float(f32) - 0.5) * scale;
            return b;
        }
    };
    const lat = try mk.buf(a, rnd, 12 * 4 * 64 * 64, 1.0);
    defer a.free(lat);
    const nrm = try mk.buf(a, rnd, 6 * 4 * 64 * 64, 0.36);
    defer a.free(nrm);
    const pos = try mk.buf(a, rnd, 6 * 4 * 64 * 64, 0.36);
    defer a.free(pos);
    const dproj = try mk.buf(a, rnd, 1 * 1028 * 1024, 1.0);
    defer a.free(dproj);
    const refi = try mk.buf(a, rnd, 1 * 4 * 64 * 64, 0.36);
    defer a.free(refi);
    // Position maps: constant 0.5 (valid, not the bg sentinel 1.0) → every cell
    // averages to a real voxel index, exercising the full PoseRoPE path.
    const pmap = try a.alloc(f32, 6 * 3 * 512 * 512);
    defer a.free(pmap);
    @memset(pmap, 0.5);

    const ri = mlx.mlx_array_new_data(refi.ptr, &[_]c_int{ 1, 4, 64, 64 }, 4, .float32);
    defer rel(ri);
    var cache = try pu.buildRefCache(ri);
    defer cache.deinit();
    var ropes = try VoxelRopes.build(a, pmap, s);
    defer ropes.deinit();

    const la = mlx.mlx_array_new_data(lat.ptr, &[_]c_int{ 12, 4, 64, 64 }, 4, .float32);
    defer rel(la);
    const na = mlx.mlx_array_new_data(nrm.ptr, &[_]c_int{ 6, 4, 64, 64 }, 4, .float32);
    defer rel(na);
    const pa = mlx.mlx_array_new_data(pos.ptr, &[_]c_int{ 6, 4, 64, 64 }, 4, .float32);
    defer rel(pa);
    const da = mlx.mlx_array_new_data(dproj.ptr, &[_]c_int{ 1, 1028, 1024 }, 3, .float32);
    defer rel(da);

    const v = try pu.forward(la, 999.0, na, pa, da, &cache, 1.0, &ropes);
    defer rel(v);
    try testing.expectEqualSlices(c_int, &[_]c_int{ 12, 4, 64, 64 }, mlx.getShape(v));
    const n: usize = @intCast(mlx.mlx_array_size(v));
    const d = mlx.mlx_array_data_float32(v) orelse return error.NoData;
    var mean_abs: f64 = 0;
    for (0..n) |i| {
        try testing.expect(std.math.isFinite(d[i]));
        mean_abs += @abs(d[i]);
    }
    std.debug.print("[hy3d-paint-unet] fwd-smoke OK: v[12,4,64,64] finite, mean|v|={d:.4}\n", .{mean_abs / @as(f64, @floatFromInt(n))});
}
