//! LTX-2.5 DiffVAE — fused 3D neighborhood-attention Metal kernel.
//!
//! The DiffVAE decoder is nothing but NA: 16 deterministic blocks upsampling the
//! latent, then 8 diffusion blocks × 2 steps over the full pixel-token volume
//! (2.38M tokens at 768×512×97, 12.7M at 1920×1088). Materializing a
//! `[tokens, kernel_volume]` score tensor is not an option at that size, so the
//! whole window — score, softmax and the value accumulate — lives in one kernel,
//! one thread per (query, head), with a 64-float accumulator and an online
//! softmax in registers.
//!
//! K/V are read THROUGH the cache rather than staged in threadgroup memory: a
//! useful query tile's halo blows past Metal's 32 KB long before it pays
//! (a (1,4,32) tile under (3,7,7) needs (3,10,38) keys = 145 KB of K+V), and
//! neighbouring queries share ~95% of their window anyway. Threads walk W first
//! so a simdgroup's 32 windows overlap.
//!
//! NATTEN SHIFTS its window inward at a boundary — it does not clamp-and-mask.
//! `ltx_diffvae.naWindowStart` is the one definition of that; getting it wrong
//! is invisible except in a `kernel/2`-wide frame around every edge (and, once
//! tiling is on, around every tile). Reference: `Lightricks/LTX-2`
//! `video_vae/transformer/fallback_na/eager.py`.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const geom = @import("ltx_diffvae.zig");

const S = mlx.mlx_stream;

/// Window kernels this file ships a specialization for. Templating on the
/// window is free (it is checkpoint geometry, fixed per stage); templating on
/// anything that varies per CALL would be a fresh Metal JIT per value.
pub const VARIANTS = [_]geom.Kernel{
    .{ 3, 7, 7 }, // det stages 1-2 + the diffusion stage
    .{ 3, 5, 5 }, // det stages 3-4
    .{ 3, 3, 3 }, // reference stage-5 default; tests
};

var cached: [VARIANTS.len]?mlx.mlx_fast_metal_kernel = @splat(null);
var engaged_logged: [VARIANTS.len]bool = @splat(false);

const HEADER =
    \\inline int ltx_na_start(int length, int kern, int i) {
    \\    const int k = kern < length ? kern : length;
    \\    const int lo = length - k;
    \\    const int mid = k / 2;
    \\    const int c = i >= mid ? (i - mid) : 0;
    \\    return c < lo ? c : lo;
    \\}
    \\
;

// One thread = one (t, h, w, head) query. Q/K/V arrive [1,T,H,W,NH,HD], already
// q/k-normed, Q pre-scaled by HD^-0.5 and RoPE'd; the output is [1,T,H,W,NH*HD].
//
// The online-softmax rescale is BRANCHED on a new maximum instead of applied
// every key: the correction is 1.0 for all but O(log n) of the ~147 keys, and
// the unconditional form spends a second 64-wide pass per key doing nothing.
const SOURCE =
    \\const int Nt = q_shape[1];
    \\const int Nh = q_shape[2];
    \\const int Nw = q_shape[3];
    \\const int NH = q_shape[4];
    \\const int w = int(thread_position_in_grid.x);
    \\const int h = int(thread_position_in_grid.y);
    \\const int z = int(thread_position_in_grid.z);
    \\const int t = z / NH;
    \\const int nh = z - t * NH;
    \\if (w >= Nw || h >= Nh || t >= Nt) return;
    \\
    \\const int kt = KT < Nt ? KT : Nt;
    \\const int kh = KH < Nh ? KH : Nh;
    \\const int kw = KW < Nw ? KW : Nw;
    \\const int t0 = ltx_na_start(Nt, KT, t);
    \\const int h0 = ltx_na_start(Nh, KH, h);
    \\const int w0 = ltx_na_start(Nw, KW, w);
    \\
    \\const size_t w_stride = (size_t)NH * HD;
    \\const size_t qoff = ((((size_t)t * Nh + h) * Nw + w) * NH + nh) * HD;
    \\
    \\T qv[HD];
    \\for (int d = 0; d < HD; ++d) qv[d] = q[qoff + d];
    \\float acc[HD];
    \\for (int d = 0; d < HD; ++d) acc[d] = 0.0f;
    \\// Finite sentinel, not -INFINITY: the first key's correction is then a
    \\// plain exp() that underflows to 0 instead of an inf-minus-inf NaN.
    \\float m = -1e30f;
    \\float lsum = 0.0f;
    \\
    \\for (int a = 0; a < kt; ++a) {
    \\    for (int b = 0; b < kh; ++b) {
    \\        size_t kv = ((((size_t)(t0 + a) * Nh + (h0 + b)) * Nw + w0) * NH + nh) * HD;
    \\        for (int c = 0; c < kw; ++c, kv += w_stride) {
    \\            float sc = 0.0f;
    \\            for (int d = 0; d < HD; ++d) sc += float(qv[d]) * float(k[kv + d]);
    \\            float p;
    \\            if (sc > m) {
    \\                const float corr = metal::precise::exp(m - sc);
    \\                m = sc;
    \\                lsum *= corr;
    \\                for (int d = 0; d < HD; ++d) acc[d] *= corr;
    \\                p = 1.0f;
    \\            } else {
    \\                p = metal::precise::exp(sc - m);
    \\            }
    \\            lsum += p;
    \\            for (int d = 0; d < HD; ++d) acc[d] += p * float(v[kv + d]);
    \\        }
    \\    }
    \\}
    \\
    \\const float inv = 1.0f / lsum;
    \\for (int d = 0; d < HD; ++d) out[qoff + d] = static_cast<T>(acc[d] * inv);
;

fn variantIndex(kern: geom.Kernel) ?usize {
    for (VARIANTS, 0..) |v, i| {
        if (std.meta.eql(v, kern)) return i;
    }
    return null;
}

fn getKernel(idx: usize) !mlx.mlx_fast_metal_kernel {
    if (cached[idx]) |k| return k;
    const input_names = [_][*:0]const u8{ "q", "k", "v" };
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    // Distinct NAME per window: two specializations sharing one name bind the
    // wrong binary out of mlx's library cache.
    const name: [*:0]const u8 = switch (idx) {
        0 => "ltx_na3d_k3_7_7",
        1 => "ltx_na3d_k3_5_5",
        2 => "ltx_na3d_k3_3_3",
        else => return error.UnsupportedNaKernel,
    };
    const kernel = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, SOURCE, HEADER, true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    cached[idx] = kernel;
    return kernel;
}

/// 3D neighborhood attention. `q`/`k`/`v` are `[1,T,H,W,NH,HD]` (Q already
/// scaled and RoPE'd); the result is `[1,T,H,W,NH*HD]`, caller owns.
pub fn na3d(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, kern: geom.Kernel, s: S) !mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return error.NaKernelNeedsGpuStream;
    const idx = variantIndex(kern) orelse return error.UnsupportedNaKernel;

    const sh = mlx.getShape(q);
    if (sh.len != 6) return error.NaKernelBadRank;
    const nt = sh[1];
    const nh_dim = sh[2];
    const nw = sh[3];
    const heads = sh[4];
    const hd = sh[5];
    if (nt < 1 or nh_dim < 1 or nw < 1) return error.NaKernelEmptyVolume;

    const dtype = mlx.mlx_array_dtype(q);
    const out_shape = [_]c_int{ sh[0], nt, nh_dim, nw, heads * hd };

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, out_shape.len, dtype));
    // W fastest so a simdgroup's 32 queries share their K/V window through L1.
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, nw, nh_dim, nt * heads));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KT", @intCast(kern[0])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KH", @intCast(kern[1])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KW", @intCast(kern[2])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HD", hd));

    const inputs_arr = [_]mlx.mlx_array{ q, k, v };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getKernel(idx);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;

    if (!engaged_logged[idx]) {
        engaged_logged[idx] = true;
        log.info("[diffvae] NA kernel engaged: window {d}x{d}x{d} head_dim {d}\n", .{ kern[0], kern[1], kern[2], hd });
    }

    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

// ── CPU ground truth (tests + the Phase-1 bisector) ──────────────────────

pub const Dims = struct { t: u32, h: u32, w: u32, nh: u32, hd: u32 };

/// Straight-from-the-definition NA in f32: every query attends exactly
/// `min(K, len)` keys per axis, starting at `geom.naWindowStart`. Not a
/// serving path — it is what the kernel is proven against.
pub fn naNaiveF32(
    alloc: std.mem.Allocator,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    d: Dims,
    kern: geom.Kernel,
) ![]f32 {
    const n = d.t * d.h * d.w * d.nh * d.hd;
    std.debug.assert(q.len == n and k.len == n and v.len == n);
    const out = try alloc.alloc(f32, n);
    errdefer alloc.free(out);

    const kt = @min(kern[0], d.t);
    const kh = @min(kern[1], d.h);
    const kw = @min(kern[2], d.w);
    const scores = try alloc.alloc(f32, kt * kh * kw);
    defer alloc.free(scores);

    const w_stride: usize = d.nh * d.hd;
    for (0..d.t) |t| {
        const t0 = geom.naWindowStart(d.t, kern[0], @intCast(t));
        for (0..d.h) |h| {
            const h0 = geom.naWindowStart(d.h, kern[1], @intCast(h));
            for (0..d.w) |w| {
                const w0 = geom.naWindowStart(d.w, kern[2], @intCast(w));
                for (0..d.nh) |nh| {
                    const qoff = ((((t * d.h + h) * d.w + w) * d.nh) + nh) * d.hd;
                    var max: f32 = -std.math.inf(f32);
                    var i: usize = 0;
                    for (0..kt) |a| {
                        for (0..kh) |b| {
                            for (0..kw) |c| {
                                const koff = (((((t0 + a) * d.h + (h0 + b)) * d.w + w0 + c) * d.nh) + nh) * d.hd;
                                var sc: f32 = 0;
                                for (0..d.hd) |e| sc += q[qoff + e] * k[koff + e];
                                scores[i] = sc;
                                max = @max(max, sc);
                                i += 1;
                            }
                        }
                    }
                    var sum: f32 = 0;
                    for (scores) |*sc| {
                        sc.* = @exp(sc.* - max);
                        sum += sc.*;
                    }
                    @memset(out[qoff .. qoff + d.hd], 0);
                    i = 0;
                    for (0..kt) |a| {
                        for (0..kh) |b| {
                            var voff = (((((t0 + a) * d.h + (h0 + b)) * d.w + w0) * d.nh) + nh) * d.hd;
                            for (0..kw) |_| {
                                const p = scores[i] / sum;
                                for (0..d.hd) |e| out[qoff + e] += p * v[voff + e];
                                voff += w_stride;
                                i += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}

// ── tests ────────────────────────────────────────────────────────────────

const testing = std.testing;

fn fillRandom(buf: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (buf) |*x| x.* = r.floatNorm(f32);
}

/// Kernel vs the f32 definition. The bar is fp32 ground truth, not another
/// kernel, and BOTH sides run in f32 so any gap is the window semantics rather
/// than rounding. Boundary positions are the whole risk surface — a
/// clamp-and-mask port passes an interior-only check.
fn naParityCase(d: Dims, kern: geom.Kernel) !void {
    const alloc = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    const n = d.t * d.h * d.w * d.nh * d.hd;
    const qb = try alloc.alloc(f32, n);
    defer alloc.free(qb);
    const kb = try alloc.alloc(f32, n);
    defer alloc.free(kb);
    const vb = try alloc.alloc(f32, n);
    defer alloc.free(vb);
    fillRandom(qb, 0x51ce);
    fillRandom(kb, 0x9e37);
    fillRandom(vb, 0xbeef);
    // Pre-scale Q exactly like the forward does, so the scores this exercises
    // are the magnitudes the softmax actually sees.
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d.hd)));
    for (qb) |*x| x.* *= scale;

    const ref = try naNaiveF32(alloc, qb, kb, vb, d, kern);
    defer alloc.free(ref);

    const shape = [_]c_int{ 1, @intCast(d.t), @intCast(d.h), @intCast(d.w), @intCast(d.nh), @intCast(d.hd) };
    const qa = mlx.mlx_array_new_data(qb.ptr, &shape, shape.len, .float32);
    defer _ = mlx.mlx_array_free(qa);
    const ka = mlx.mlx_array_new_data(kb.ptr, &shape, shape.len, .float32);
    defer _ = mlx.mlx_array_free(ka);
    const va = mlx.mlx_array_new_data(vb.ptr, &shape, shape.len, .float32);
    defer _ = mlx.mlx_array_free(va);

    const out = try na3d(qa, ka, va, kern, s);
    defer _ = mlx.mlx_array_free(out);
    _ = mlx.mlx_array_eval(out);
    try testing.expectEqual(@as(usize, n), @as(usize, @intCast(mlx.mlx_array_size(out))));

    const got = mlx.mlx_array_data_float32(out).?;
    var max_err: f64 = 0;
    for (0..n) |i| max_err = @max(max_err, @abs(@as(f64, got[i]) - @as(f64, ref[i])));
    try testing.expect(std.math.isFinite(max_err));
    try testing.expect(max_err < 1e-5);
}

test "diffvae NA kernel matches the f32 definition in the interior" {
    // Every axis clears its window with room to spare, so most queries sit in
    // the shift-free interior.
    try naParityCase(.{ .t = 5, .h = 9, .w = 9, .nh = 2, .hd = 64 }, .{ 3, 7, 7 });
}

test "diffvae NA kernel matches the f32 definition at the boundaries" {
    // T=3 against K_t=3 and H=7 against K_h=7 make EVERY position a boundary
    // position: the window has nowhere to centre, so a clamp-and-mask port
    // disagrees on the whole volume rather than in a frame around it.
    try naParityCase(.{ .t = 3, .h = 7, .w = 8, .nh = 2, .hd = 64 }, .{ 3, 7, 7 });
    // Axes SHORTER than their kernel: NA degenerates to "attend everything".
    try naParityCase(.{ .t = 2, .h = 4, .w = 3, .nh = 3, .hd = 64 }, .{ 3, 7, 7 });
    // The narrow window, and a W long enough to cross a threadgroup boundary.
    try naParityCase(.{ .t = 4, .h = 5, .w = 37, .nh = 1, .hd = 64 }, .{ 3, 3, 3 });
    try naParityCase(.{ .t = 6, .h = 6, .w = 6, .nh = 2, .hd = 64 }, .{ 3, 5, 5 });
}

test "every window the production config asks for has a kernel variant" {
    for (geom.production.stages) |st| {
        try testing.expect(variantIndex(st.kernel) != null);
    }
    try testing.expect(variantIndex(geom.production.stage5_kernel) != null);
}
