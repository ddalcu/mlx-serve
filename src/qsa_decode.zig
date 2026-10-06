//! QSA attention for ONE decode token over dense bf16 KV: the indexer's sorted block picks go
//! straight to a split pass and a merge pass, with no mask, gather or SDPA between them.
//!
//! Split pass: a simdgroup owns one q head and `CH` of the visible keys, eight keys at a time with
//! the head dim spread over the lanes. The eight partial dots reduce in one butterfly that leaves
//! key `lane >> 2` on lanes `4k..4k+3` (9 shuffles, not 8 x 5). More heads per simdgroup lose to
//! register pressure.
const std = @import("std");
const mlx = @import("mlx.zig");

const CH: c_int = 32; // visible keys per simdgroup, a multiple of 8 and of RATIO
const BD: c_int = 256;

const HEADER =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\inline float msv_bf_lo(uint w) { return as_type<float>(w << 16); }
    \\inline float msv_bf_hi(uint w) { return as_type<float>(w & 0xffff0000u); }
    \\inline void msv_unpack8(uint4 r, thread float* f) {
    \\  f[0] = msv_bf_lo(r.x); f[1] = msv_bf_hi(r.x); f[2] = msv_bf_lo(r.y); f[3] = msv_bf_hi(r.y);
    \\  f[4] = msv_bf_lo(r.z); f[5] = msv_bf_hi(r.z); f[6] = msv_bf_lo(r.w); f[7] = msv_bf_hi(r.w);
    \\}
    \\// v[i] is this lane's partial dot for key i. Each stage keeps half of the keys on the lane
    \\// and trades the other half with the partner; the key left on lane l is l >> 2.
    \\inline float msv_reduce8(thread float* v, ushort lane) {
    \\  const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
    \\  float w[4];
    \\  #pragma unroll
    \\  for (int i = 0; i < 4; ++i) w[i] = (b4 ? v[i + 4] : v[i]) + simd_shuffle_xor(b4 ? v[i] : v[i + 4], 16);
    \\  float x[2];
    \\  #pragma unroll
    \\  for (int i = 0; i < 2; ++i) x[i] = (b3 ? w[i + 2] : w[i]) + simd_shuffle_xor(b3 ? w[i] : w[i + 2], 8);
    \\  float y = (b2 ? x[1] : x[0]) + simd_shuffle_xor(b2 ? x[0] : x[1], 4);
    \\  y += simd_shuffle_xor(y, 1);
    \\  return y + simd_shuffle_xor(y, 2);
    \\}
    \\inline int msv_dec_pos(const device int* blk, int j, int sel_len, int tail_start, int ratio) {
    \\  const int b = j / ratio;
    \\  return (j < sel_len) ? (blk[b] * ratio + (j - b * ratio)) : (tail_start + (j - sel_len));
    \\}
    \\
;

const PARTS_SOURCE =
    \\const int kL = k_shape[2];
    \\const int split = int(threadgroup_position_in_grid.x);
    \\const int hk = int(threadgroup_position_in_grid.y);
    \\const ushort lane = ushort(thread_index_in_simdgroup);
    \\const int head = hk * GQA + int(simdgroup_index_in_threadgroup);
    \\
    \\const int nb = kL / RATIO;
    \\const int sel_len = metal::min(nb, KB) * RATIO;
    \\const int tail_start = nb * RATIO;
    \\const int L = sel_len + (kL - tail_start);
    \\const int j_lo = split * CH;
    \\const int j_hi = metal::min(j_lo + CH, L);
    \\const float scale_log2e = scl[0] * 1.44269504088896340736f;
    \\
    \\const device T* Kp = k + (long)hk * k_strides[1];
    \\const device T* Vp = v + (long)hk * v_strides[1];
    \\
    \\// The chunk's block bases, loaded up front so no key waits on a block id.
    \\int bp[CH / RATIO];
    \\#pragma unroll
    \\for (int i = 0; i < CH / RATIO; ++i) {
    \\  const int b = j_lo / RATIO + i;
    \\  bp[i] = (b * RATIO < sel_len) ? blocks[b] * RATIO : 0;
    \\}
    \\float qf[8];
    \\msv_unpack8(*((const device uint4*)(q + (long)head * q_strides[1]) + lane), qf);
    \\#pragma unroll
    \\for (int d = 0; d < 8; ++d) qf[d] *= scale_log2e;
    \\float m = -3.0e38f, l = 0.0f, acc[8];
    \\#pragma unroll
    \\for (int d = 0; d < 8; ++d) acc[d] = 0.0f;
    \\
    \\#pragma unroll
    \\for (int bi = 0; bi < CH / 8; ++bi) {
    \\  const int jb = j_lo + bi * 8;
    \\  if (jb < j_hi) {
    \\    uint4 kr[8], vr[8];
    \\    #pragma unroll
    \\    for (int i = 0; i < 8; ++i) {
    \\      const int jj = bi * 8 + i;
    \\      const int j = j_lo + jj;
    \\      const int pos = (j >= j_hi) ? 0 : ((j < sel_len) ? bp[jj / RATIO] + jj % RATIO : tail_start + (j - sel_len));
    \\      kr[i] = *((const device uint4*)(Kp + (long)pos * k_strides[2]) + lane);
    \\      vr[i] = *((const device uint4*)(Vp + (long)pos * v_strides[2]) + lane);
    \\    }
    \\    float part[8];
    \\    #pragma unroll
    \\    for (int i = 0; i < 8; ++i) {
    \\      float kf[8];
    \\      msv_unpack8(kr[i], kf);
    \\      float a0 = qf[0] * kf[0], a1 = qf[1] * kf[1];
    \\      a0 = fma(qf[2], kf[2], a0); a1 = fma(qf[3], kf[3], a1);
    \\      a0 = fma(qf[4], kf[4], a0); a1 = fma(qf[5], kf[5], a1);
    \\      a0 = fma(qf[6], kf[6], a0); a1 = fma(qf[7], kf[7], a1);
    \\      part[i] = a0 + a1;
    \\    }
    \\    const float total = msv_reduce8(part, lane);
    \\    const float y = (jb + int(lane >> 2) < j_hi) ? total : -INFINITY;
    \\    float bm = metal::max(y, simd_shuffle_xor(y, 4));
    \\    bm = metal::max(bm, simd_shuffle_xor(bm, 8));
    \\    bm = metal::max(bm, simd_shuffle_xor(bm, 16));
    \\    const float m_new = metal::max(m, bm);
    \\    const float factor = metal::exp2(m - m_new);
    \\    const float p = metal::exp2(y - m_new);
    \\    float lb = p + simd_shuffle_xor(p, 4);
    \\    lb += simd_shuffle_xor(lb, 8);
    \\    lb += simd_shuffle_xor(lb, 16);
    \\    l = l * factor + lb;
    \\    m = m_new;
    \\    #pragma unroll
    \\    for (int d = 0; d < 8; ++d) acc[d] *= factor;
    \\    #pragma unroll
    \\    for (int i = 0; i < 8; ++i) {
    \\      float vf[8];
    \\      msv_unpack8(vr[i], vf);
    \\      const float pi = simd_shuffle(p, ushort(4 * i));
    \\      #pragma unroll
    \\      for (int d = 0; d < 8; ++d) acc[d] = fma(pi, vf[d], acc[d]);
    \\    }
    \\  }
    \\}
    \\
    \\const long row = (long)head * NS + split;
    \\device float* pa = pacc + row * 256 + lane * 8;
    \\*((device float4*)pa) = float4(acc[0], acc[1], acc[2], acc[3]);
    \\*((device float4*)(pa + 4)) = float4(acc[4], acc[5], acc[6], acc[7]);
    \\if (lane == 0) {
    \\  pml[row * 2] = m;
    \\  pml[row * 2 + 1] = l;
    \\}
;

// One threadgroup a head. An empty split writes l = 0, the skip flag.
const MERGE_SOURCE =
    \\const int head = int(threadgroup_position_in_grid.x);
    \\const int d = int(thread_position_in_threadgroup.x);
    \\const ushort lane = ushort(thread_index_in_simdgroup);
    \\threadgroup float wts[NS];
    \\threadgroup float zsum[1];
    \\if (simdgroup_index_in_threadgroup == 0) {
    \\  constexpr int SLOTS = (NS + 31) / 32;
    \\  float ms[SLOTS], ls[SLOTS];
    \\  float M = -3.0e38f;
    \\  #pragma unroll
    \\  for (int s = 0; s < SLOTS; ++s) {
    \\    const int j = lane + s * 32;
    \\    const long r = (long)head * NS + j;
    \\    ms[s] = (j < NS) ? pml[r * 2] : -3.0e38f;
    \\    ls[s] = (j < NS) ? pml[r * 2 + 1] : 0.0f;
    \\    if (ls[s] > 0.0f) M = metal::max(M, ms[s]);
    \\  }
    \\  M = simd_max(M);
    \\  float Z = 0.0f;
    \\  #pragma unroll
    \\  for (int s = 0; s < SLOTS; ++s) {
    \\    const int j = lane + s * 32;
    \\    const float w = (ls[s] > 0.0f) ? metal::exp2(ms[s] - M) : 0.0f;
    \\    if (j < NS) wts[j] = w;
    \\    Z += w * ls[s];
    \\  }
    \\  Z = simd_sum(Z);
    \\  if (lane == 0) zsum[0] = Z;
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\const device float* pa = pacc + (long)head * NS * 256 + d;
    \\float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    \\#pragma unroll
    \\for (int j = 0; j < NS; ++j) a[j & 3] = fma(wts[j], pa[(long)j * 256], a[j & 3]);
    \\out[(long)head * 256 + d] = T(((a[0] + a[1]) + (a[2] + a[3])) / zsum[0]);
;

var parts_kernel: ?mlx.mlx_fast_metal_kernel = null;
var merge_kernel: ?mlx.mlx_fast_metal_kernel = null;

const Key = struct { hq: c_int, hk: c_int, kb: c_int, ratio: c_int };
const Cfgs = struct { key: Key, parts: mlx.mlx_fast_metal_kernel_config, merge: mlx.mlx_fast_metal_kernel_config };
var cfgs: ?Cfgs = null;

var env_enabled: ?bool = null;
pub var override: ?bool = null;

/// `MLX_SERVE_QSA_DEC_KERNEL=0` keeps the mask / gather arms.
pub fn enabled() bool {
    if (override) |v| return v;
    if (env_enabled) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_QSA_DEC_KERNEL");
    env_enabled = raw == null or raw.?[0] != '0';
    return env_enabled.?;
}

fn newKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    // K/V are cache views: the kernel reads their strides.
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, HEADER, false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn configsFor(key: Key, ns: c_int) !Cfgs {
    if (cfgs) |c| {
        if (std.meta.eql(c.key, key)) return c;
        _ = mlx.mlx_fast_metal_kernel_config_free(c.parts);
        _ = mlx.mlx_fast_metal_kernel_config_free(c.merge);
        cfgs = null;
    }
    const gqa = @divExact(key.hq, key.hk);
    const parts = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(parts);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(parts, &[_]c_int{ key.hq, ns, BD }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(parts, &[_]c_int{ key.hq, ns, 2 }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(parts, ns * gqa * 32, key.hk, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(parts, gqa * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(parts, "T", .bfloat16));
    inline for (.{ .{ "GQA", gqa }, .{ "KB", key.kb }, .{ "RATIO", key.ratio }, .{ "CH", CH }, .{ "NS", ns } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(parts, kv[0], kv[1]));
    }
    const merge = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(merge);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(merge, &[_]c_int{ 1, key.hq, 1, BD }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(merge, BD * key.hq, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(merge, BD, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(merge, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge, "NS", ns));
    cfgs = .{ .key = key, .parts = parts, .merge = merge };
    return cfgs.?;
}

/// `[1, Hq, 1, 256]` bf16 attention of one query row `q` over the keys `blocks` `[1, 1, kb]` int32
/// selects (RATIO tokens each) plus the incomplete tail; `k`/`v` are the dense cache views
/// `[1, Hk, kv, 256]`. Null when the geometry is outside the kernel's set (the caller keeps its arm).
pub fn attend(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, blocks: mlx.mlx_array, ratio: c_int, attn_scale: f32) !?mlx.mlx_array {
    if (!enabled() or !mlx.streamIsGpu(s) or q.ctx == null or k.ctx == null or v.ctx == null or blocks.ctx == null) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(blocks) != 3) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16 or mlx.mlx_array_dtype(k) != .bfloat16 or mlx.mlx_array_dtype(v) != .bfloat16) return null;
    if (mlx.mlx_array_dtype(blocks) != .int32) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const bs = mlx.getShape(blocks);
    if (!std.mem.eql(c_int, ks, mlx.getShape(v))) return null;
    if (qs[0] != 1 or qs[2] != 1 or qs[3] != BD or ks[0] != 1 or ks[3] != BD or bs[0] != 1 or bs[1] != 1) return null;
    const hq = qs[1];
    const hk = ks[1];
    // An input under 8 elements binds in `constant` space, where the kernel's `device int*` fails to compile.
    if (hk <= 0 or ratio <= 0 or bs[2] < 8 or @rem(hq, hk) != 0 or @divTrunc(hq, hk) > 32 or @rem(CH, ratio) != 0) return null;
    if (mlx.mlx_array_strides(q)[3] != 1 or mlx.mlx_array_strides(blocks)[2] != 1) return null;

    if (parts_kernel == null) {
        parts_kernel = try newKernel("msv_qsa_dec_parts", &.{ "q", "scl", "blocks", "k", "v" }, &.{ "pacc", "pml" }, PARTS_SOURCE);
        merge_kernel = try newKernel("msv_qsa_dec_merge", &.{ "pacc", "pml" }, &.{"out"}, MERGE_SOURCE);
    }
    const ns = @divTrunc(bs[2] * ratio + ratio - 1 + CH - 1, CH);
    const c = try configsFor(.{ .hq = hq, .hk = hk, .kb = bs[2], .ratio = ratio }, ns);

    const scl_data = [_]f32{attn_scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const ins = [_]mlx.mlx_array{ q, scl, blocks, k, v };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&parts, parts_kernel.?, in_vec, c.parts, s));
    if (mlx.mlx_vector_array_size(parts) != 2) return error.MetalKernelBadOutputCount;
    var pacc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pacc);
    var pml = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pml);
    try mlx.check(mlx.mlx_vector_array_get(&pacc, parts, 0));
    try mlx.check(mlx.mlx_vector_array_get(&pml, parts, 1));

    const merge_in = [_]mlx.mlx_array{ pacc, pml };
    const merge_vec = mlx.mlx_vector_array_new_data(&merge_in, merge_in.len);
    defer _ = mlx.mlx_vector_array_free(merge_vec);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, merge_kernel.?, merge_vec, c.merge, s));
    if (mlx.mlx_vector_array_size(o) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, o, 0));
    return out;
}

const testing = std.testing;

fn randomBf16(s: mlx.mlx_stream, shape: []const c_int, seed: u64) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_random_normal(&f, shape.ptr, shape.len, .float32, 0.0, 1.0, key, s));
    var b = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_astype(&b, f, .bfloat16, s));
    return b;
}

fn hostF32(s: mlx.mlx_stream, a: mlx.mlx_array) !struct { arr: mlx.mlx_array, data: []const f32 } {
    var f = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, a, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const p = mlx.mlx_array_data_float32(f) orelse return error.Unreadable;
    return .{ .arr = f, .data = p[0..mlx.mlx_array_size(f)] };
}

test "qsa decode: the one-token kernel is no worse than the masked SDPA against the f64 truth" {
    const xfm = @import("transformer.zig");
    defer override = null;
    const s = mlx.gpuStream();
    const hq: usize = 24;
    const hk: usize = 2;
    const gqa = hq / hk;
    const kb: usize = 512;
    const ratio: usize = 4;
    const cap: usize = 12288;
    const scale: f32 = 1.0 / 16.0;
    var prng = std.Random.DefaultPrng.init(0xDEC0DE);
    const rnd = prng.random();
    const kvs = [_]usize{ 2052, 2055, 4099, 8076, 8103, 11000 };
    for (kvs, 0..) |kv, case| {
        const q = try randomBf16(s, &[_]c_int{ 1, @intCast(hq), 1, BD }, 100 + case);
        defer _ = mlx.mlx_array_free(q);
        const kfull = try randomBf16(s, &[_]c_int{ 1, @intCast(hk), @intCast(cap), BD }, 200 + case);
        defer _ = mlx.mlx_array_free(kfull);
        const vfull = try randomBf16(s, &[_]c_int{ 1, @intCast(hk), @intCast(cap), BD }, 300 + case);
        defer _ = mlx.mlx_array_free(vfull);
        // The cache views a decode step sees: a leading slice of a larger buffer.
        var kview = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kview);
        var vview = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(vview);
        const lo = [_]c_int{ 0, 0, 0, 0 };
        const hi = [_]c_int{ 1, @intCast(hk), @intCast(kv), BD };
        const st = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&kview, kfull, &lo, 4, &hi, 4, &st, 4, s));
        try mlx.check(mlx.mlx_slice(&vview, vfull, &lo, 4, &hi, 4, &st, 4, s));

        const nb = kv / ratio;
        const pool = try testing.allocator.alloc(i32, nb);
        defer testing.allocator.free(pool);
        for (pool, 0..) |*p, i| p.* = @intCast(i);
        rnd.shuffle(i32, pool);
        const sel = try testing.allocator.dupe(i32, pool[0..kb]);
        defer testing.allocator.free(sel);
        std.mem.sort(i32, sel, {}, std.sort.asc(i32));
        const blocks = mlx.mlx_array_new_data(sel.ptr, &[_]c_int{ 1, 1, @intCast(kb) }, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);

        // f64 truth over the same bf16 values.
        const qh = try hostF32(s, q);
        defer _ = mlx.mlx_array_free(qh.arr);
        const kh = try hostF32(s, kview);
        defer _ = mlx.mlx_array_free(kh.arr);
        const vh = try hostF32(s, vview);
        defer _ = mlx.mlx_array_free(vh.arr);
        var pos_list: std.ArrayList(usize) = .empty;
        defer pos_list.deinit(testing.allocator);
        for (sel) |b| for (0..ratio) |r| try pos_list.append(testing.allocator, @as(usize, @intCast(b)) * ratio + r);
        var t = nb * ratio;
        while (t < kv) : (t += 1) try pos_list.append(testing.allocator, t);
        const truth = try testing.allocator.alloc(f64, hq * BD);
        defer testing.allocator.free(truth);
        const sc = try testing.allocator.alloc(f64, pos_list.items.len);
        defer testing.allocator.free(sc);
        for (0..hq) |h| {
            const g = h / gqa;
            var mx: f64 = -std.math.inf(f64);
            for (pos_list.items, 0..) |pos, i| {
                var dot: f64 = 0;
                for (0..BD) |d| dot += @as(f64, qh.data[h * BD + d]) * @as(f64, kh.data[(g * kv + pos) * BD + d]);
                sc[i] = dot * scale;
                mx = @max(mx, sc[i]);
            }
            var z: f64 = 0;
            for (sc) |*x| {
                x.* = @exp(x.* - mx);
                z += x.*;
            }
            for (0..BD) |d| {
                var o: f64 = 0;
                for (pos_list.items, 0..) |pos, i| o += sc[i] * @as(f64, vh.data[(g * kv + pos) * BD + d]);
                truth[h * BD + d] = o / z;
            }
        }

        override = true;
        const got = (try attend(s, q, kview, vview, blocks, @intCast(ratio), scale)) orelse return error.KernelDeclined;
        defer _ = mlx.mlx_array_free(got);
        const mask = try xfm.qsaMaskFromBlocks(s, blocks, @intCast(kv), @intCast(ratio));
        defer _ = mlx.mlx_array_free(mask);
        const stock = try xfm.qsaMaskArm(s, q, kview, vview, scale, mask);
        defer _ = mlx.mlx_array_free(stock);
        const gh = try hostF32(s, got);
        defer _ = mlx.mlx_array_free(gh.arr);
        const sh = try hostF32(s, stock);
        defer _ = mlx.mlx_array_free(sh.arr);
        try testing.expectEqual(hq * BD, gh.data.len);
        var e_got: f64 = 0;
        var e_stock: f64 = 0;
        var n_truth: f64 = 0;
        for (truth, 0..) |tv, i| {
            try testing.expect(std.math.isFinite(gh.data[i]));
            e_got += (@as(f64, gh.data[i]) - tv) * (@as(f64, gh.data[i]) - tv);
            e_stock += (@as(f64, sh.data[i]) - tv) * (@as(f64, sh.data[i]) - tv);
            n_truth += tv * tv;
        }
        const rms_got = @sqrt(e_got / @as(f64, @floatFromInt(truth.len)));
        const rms_stock = @sqrt(e_stock / @as(f64, @floatFromInt(truth.len)));
        const rms_truth = @sqrt(n_truth / @as(f64, @floatFromInt(truth.len)));
        std.debug.print("[qsa-dec parity] kv={d} rms_err kernel={e:.3} stock={e:.3} (truth rms {e:.3})\n", .{ kv, rms_got, rms_stock, rms_truth });
        try testing.expect(rms_got <= rms_stock * 1.15);
        try testing.expect(rms_got < rms_truth * 0.02);
    }
}
