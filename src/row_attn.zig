//! Decode attention whose every row has the bits of the one-row step at its
//! position, ported from TensorFold's `row_attention.py` (MIT, see NOTICE).
//! One threadgroup per (kv head, window row, key chunk). Keys are cut into
//! CK-position chunks counted from 0 and a chunk's keys
//! interleave over SPLIT simdgroups; per simdgroup an online softmax in
//! ascending position, the simdgroups merged in order, then the chunks. What
//! rides in the window changes nothing a row computes, so serial decoding
//! through this kernel is the reference a verify window reproduces.
const std = @import("std");
const mlx = @import("mlx.zig");

const CK = 128;
const SPLIT = 4;
const BLK = 4;
pub const MAX_ROWS = 16;

// K/V are the KV cache's views [1, HKV, L, D]: indexed through their strides,
// never copied. The window's rows sit at cache rows P .. P + W - 1 (P = L - W);
// row w is at position P + depth[w] and its keys past P are the window rows
// path[w][0 .. depth[w]] (a chain: depth w, path 0..w; a tree: its ancestors).
const PARTIAL =
    \\  const uint lane = thread_index_in_simdgroup;
    \\  const uint sgi = simdgroup_index_in_threadgroup;
    \\  const int g = int(sgi) / SPLIT, s = int(sgi) % SPLIT;
    \\  const int c = int(threadgroup_position_in_grid.y);
    \\  const int P = dims[0], W = dims[1], NCH = dims[2], MAXD = dims[3];
    \\  const int h = int(threadgroup_position_in_grid.z) / W, w = int(threadgroup_position_in_grid.z) % W;
    \\  constexpr int DPL = D / 32;
    \\  const int qh = h * G + g;
    \\  const size_t kh = size_t(h) * size_t(K_strides[1]), kp = size_t(K_strides[2]);
    \\  const size_t vh = size_t(h) * size_t(V_strides[1]), vp = size_t(V_strides[2]);
    \\  threadgroup float sm[G * SPLIT], sl[G * SPLIT];
    \\  threadgroup float so[G * SPLIT][D];
    \\  {
    \\    const int last = P + depth[w];
    \\    const int k0 = c * CK;
    \\    const int k1 = min(k0 + CK, last + 1);
    \\    float q[DPL], o[DPL];
    \\    for (int i = 0; i < DPL; i++) {
    \\      q[i] = float(Q[((qh * W) + w) * D + int(lane) * DPL + i]);
    \\      o[i] = 0.0f;
    \\    }
    \\    float m = -INFINITY, l = 0.0f;
    \\    for (int base = k0 + s; base < k1; base += SPLIT * BLK) {
    \\      float sc[BLK];
    \\      int rows[BLK];
    \\      float bm = -INFINITY;
    \\      for (int j = 0; j < BLK; j++) {
    \\        const int pos = base + j * SPLIT;
    \\        rows[j] = pos < k1 ? (pos < P ? pos : P + path[w * MAXD + (pos - P)]) : -1;
    \\        float d = 0.0f;
    \\        if (rows[j] >= 0) {
    \\          const device bfloat* kr = K + kh + size_t(rows[j]) * kp + int(lane) * DPL;
    \\          for (int i = 0; i < DPL; i++) d = fma(q[i], float(kr[i]), d);
    \\        }
    \\        sc[j] = simd_sum(d) * scale[0];
    \\        if (rows[j] >= 0) bm = metal::max(bm, sc[j]);
    \\      }
    \\      const float mn = metal::max(m, bm);
    \\      const float a = metal::exp(m - mn);
    \\      l *= a;
    \\      for (int i = 0; i < DPL; i++) o[i] *= a;
    \\      for (int j = 0; j < BLK; j++) {
    \\        if (rows[j] < 0) continue;
    \\        const float b = metal::exp(sc[j] - mn);
    \\        l += b;
    \\        const device bfloat* vr = V + vh + size_t(rows[j]) * vp + int(lane) * DPL;
    \\        for (int i = 0; i < DPL; i++) o[i] = fma(b, float(vr[i]), o[i]);
    \\      }
    \\      m = mn;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (lane == 0) { sm[sgi] = m; sl[sgi] = l; }
    \\    for (int i = 0; i < DPL; i++) so[sgi][int(lane) * DPL + i] = o[i];
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (s == 0) {
    \\      float mx_ = -INFINITY;
    \\      for (int t = 0; t < SPLIT; t++) mx_ = metal::max(mx_, sm[g * SPLIT + t]);
    \\      float lsum = 0.0f, acc[DPL];
    \\      for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
    \\      for (int t = 0; t < SPLIT; t++) {
    \\        const float e = sl[g * SPLIT + t] > 0.0f ? metal::exp(sm[g * SPLIT + t] - mx_) : 0.0f;
    \\        lsum = fma(sl[g * SPLIT + t], e, lsum);
    \\        for (int i = 0; i < DPL; i++) acc[i] = fma(so[g * SPLIT + t][int(lane) * DPL + i], e, acc[i]);
    \\      }
    \\      const int slot = (qh * W + w) * NCH + c;
    \\      if (lane == 0) { PM[slot] = mx_; PL[slot] = lsum; }
    \\      for (int i = 0; i < DPL; i++) PO[size_t(slot) * D + int(lane) * DPL + i] = acc[i];
    \\    }
    \\  }
;

const MERGE =
    \\  const uint lane = thread_index_in_simdgroup;
    \\  const int qh = int(threadgroup_position_in_grid.y);
    \\  const int w = int(threadgroup_position_in_grid.z);
    \\  const int W = dims[1], NCH = dims[2];
    \\  constexpr int DPL = D / 32;
    \\  const int base = (qh * W + w) * NCH;
    \\  float mx_ = -INFINITY;
    \\  for (int c = 0; c < NCH; c++) mx_ = metal::max(mx_, PM[base + c]);
    \\  float lsum = 0.0f, acc[DPL];
    \\  for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
    \\  for (int c = 0; c < NCH; c++) {
    \\    const float e = PL[base + c] > 0.0f ? metal::exp(PM[base + c] - mx_) : 0.0f;
    \\    lsum = fma(PL[base + c], e, lsum);
    \\    for (int i = 0; i < DPL; i++) acc[i] = fma(PO[size_t(base + c) * D + int(lane) * DPL + i], e, acc[i]);
    \\  }
    \\  for (int i = 0; i < DPL; i++) OUT[((qh * W) + w) * D + int(lane) * DPL + i] = bfloat(acc[i] / lsum);
;

var partial_kernel: ?mlx.mlx_fast_metal_kernel = null;
var merge_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn kernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8, row_contig: bool) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, src, "", row_contig, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// Whether the kernel serves this attention: bf16 q/k/v, head dim a multiple
/// of 32, whole kv-head groups in one threadgroup, 1..MAX_ROWS query rows.
pub fn fits(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array) bool {
    for ([_]mlx.mlx_array{ q, k, v }) |a| if (mlx.mlx_array_dtype(a) != .bfloat16) return false;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    if (qs.len != 4 or ks.len != 4 or qs[0] != 1 or ks[0] != 1) return false;
    if (@rem(qs[3], 32) != 0 or ks[3] != qs[3] or @rem(qs[1], ks[1]) != 0) return false;
    const g = @divExact(qs[1], ks[1]);
    return g * SPLIT * 32 <= 1024 and qs[2] >= 1 and qs[2] <= MAX_ROWS and ks[2] >= qs[2];
}

/// A draft tree over a window's rows: `depth [W]` int32 and `path [W, MAXD]`
/// int32 (row w's window rows at depths 0 .. depth[w]), `max_depth` = MAXD - 1.
pub const Tree = struct { depth: mlx.mlx_array, path: mlx.mlx_array, max_depth: c_int };

/// q [1, H, W, D] of a window whose rows sit at the last W positions of k/v
/// [1, HKV, L, D] (the cache views after this step's update) -> [1, H, W, D].
/// `tree` null: a chain.
pub fn sdpa(out: *mlx.mlx_array, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, tree: ?Tree, s: mlx.mlx_stream) !bool {
    if (!mlx.streamIsGpu(s) or !fits(q, k, v)) return false;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const h = qs[1];
    const w = qs[2];
    const d = qs[3];
    const hkv = ks[1];
    const g = @divExact(h, hkv);
    const p = ks[2] - w;
    const nch = @divTrunc(ks[2] + CK - 1, CK);

    const pk = try kernel(&partial_kernel, "msv_row_attn_partial", &.{ "Q", "K", "V", "depth", "path", "scale", "dims" }, &.{ "PM", "PL", "PO" }, PARTIAL, false);
    const mk = try kernel(&merge_kernel, "msv_row_attn_merge", &.{ "PM", "PL", "PO", "dims" }, &.{"OUT"}, MERGE, true);

    var chain: [MAX_ROWS * MAX_ROWS]i32 = undefined;
    var chain_depth: [MAX_ROWS]i32 = undefined;
    const t = tree orelse blk: {
        const wu: usize = @intCast(w);
        for (0..wu) |r| {
            chain_depth[r] = @intCast(r);
            for (0..wu) |i| chain[r * wu + i] = @intCast(i);
        }
        break :blk Tree{
            .depth = mlx.mlx_array_new_data(&chain_depth, &[_]c_int{w}, 1, .int32),
            .path = mlx.mlx_array_new_data(&chain, &[_]c_int{ w, w }, 2, .int32),
            .max_depth = w - 1,
        };
    };
    defer if (tree == null) {
        _ = mlx.mlx_array_free(t.depth);
        _ = mlx.mlx_array_free(t.path);
    };
    const dims_v = [_]i32{ p, w, nch, t.max_depth + 1 };
    const dims = mlx.mlx_array_new_data(&dims_v, &[_]c_int{4}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    const scale_arr = mlx.mlx_array_new_data(&scale, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(scale_arr);
    var qc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qc);
    try mlx.check(mlx.mlx_contiguous(&qc, q, false, s));

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const slots = h * w * nch;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{slots}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{slots}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ slots, d }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * g * SPLIT, nch, hkv * w));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * g * SPLIT, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", d));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "G", g));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "CK", CK));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SPLIT", SPLIT));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BLK", BLK));
    const ins = [_]mlx.mlx_array{ qc, k, v, t.depth, t.path, scale_arr, dims };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&parts, pk, in_vec, cfg, s));

    const mcfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(mcfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mcfg, &[_]c_int{ 1, h, w, d }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mcfg, 32, h, w));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mcfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mcfg, "D", d));
    var pm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pm);
    var pl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pl);
    var po = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(po);
    try mlx.check(mlx.mlx_vector_array_get(&pm, parts, 0));
    try mlx.check(mlx.mlx_vector_array_get(&pl, parts, 1));
    try mlx.check(mlx.mlx_vector_array_get(&po, parts, 2));
    const mins = [_]mlx.mlx_array{ pm, pl, po, dims };
    const mvec = mlx.mlx_vector_array_new_data(&mins, mins.len);
    defer _ = mlx.mlx_vector_array_free(mvec);
    var res = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(res);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&res, mk, mvec, mcfg, s));
    try mlx.check(mlx.mlx_vector_array_get(out, res, 0));
    return true;
}

// ── tests ──

const testing = std.testing;

fn randBf16(shape: []const c_int, seed: u64, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_random_normal(&f, shape.ptr, shape.len, .float32, 0.0, 1.0, key, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

fn slice4(a: mlx.mlx_array, axis: usize, lo: c_int, hi: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(a);
    var start = [_]c_int{ 0, 0, 0, 0 };
    var stop = [_]c_int{ sh[0], sh[1], sh[2], sh[3] };
    start[axis] = lo;
    stop[axis] = hi;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, a, &start, 4, &stop, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
    return out;
}

test "row_attn: every window row equals the one-row step at its position, and matches MLX's sdpa" {
    const s = mlx.gpuStream();
    // Qwen3.8-27B: 24 query heads over 4 kv heads, head dim 256; a cache
    // buffer longer than the keys, read as a view like the live cache.
    const H: c_int = 24;
    const HKV: c_int = 4;
    const D: c_int = 256;
    const CAP: c_int = 512;
    const L: c_int = 300;
    const W: c_int = 8;
    const q = try randBf16(&.{ 1, H, W, D }, 1, s);
    defer _ = mlx.mlx_array_free(q);
    const kbuf = try randBf16(&.{ 1, HKV, CAP, D }, 2, s);
    defer _ = mlx.mlx_array_free(kbuf);
    const vbuf = try randBf16(&.{ 1, HKV, CAP, D }, 3, s);
    defer _ = mlx.mlx_array_free(vbuf);
    const k = try slice4(kbuf, 2, 0, L, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try slice4(vbuf, 2, 0, L, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0;

    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try testing.expect(try sdpa(&all, q, k, v, scale, null, s));
    var r: c_int = 0;
    while (r < W) : (r += 1) {
        const keys = L - W + r + 1;
        const qr = try slice4(q, 2, r, r + 1, s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try slice4(k, 2, 0, keys, s);
        defer _ = mlx.mlx_array_free(kr);
        const vr = try slice4(v, 2, 0, keys, s);
        defer _ = mlx.mlx_array_free(vr);
        var one = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(one);
        try testing.expect(try sdpa(&one, qr, kr, vr, scale, null, s));
        const want = try slice4(all, 2, r, r + 1, s);
        defer _ = mlx.mlx_array_free(want);
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, one, want, false, s));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        try testing.expect(same);

        // Against MLX's one-query sdpa over the same keys.
        const none = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(none);
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, qr, kr, vr, scale, "", none, .{ .ctx = null }, false, s));
        var d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d);
        var a32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a32);
        var b32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(b32);
        try mlx.check(mlx.mlx_astype(&a32, one, .float32, s));
        try mlx.check(mlx.mlx_astype(&b32, ref, .float32, s));
        try mlx.check(mlx.mlx_subtract(&d, a32, b32, s));
        var ad = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ad);
        try mlx.check(mlx.mlx_abs(&ad, d, s));
        var mx_ = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mx_);
        try mlx.check(mlx.mlx_max(&mx_, ad, false, s));
        var worst: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&worst, mx_));
        try testing.expect(worst < 2e-2);
    }
}

test "row_attn: a tree node equals the one-row step over its own path's keys" {
    const s = mlx.gpuStream();
    const H: c_int = 24;
    const HKV: c_int = 4;
    const D: c_int = 256;
    const P: c_int = 150; // committed keys
    const W: c_int = 6;
    // rows: 0 root, 1 and 2 children of 0, 3 child of 1, 4 child of 2, 5 child of 3
    const parents = [_]i32{ -1, 0, 0, 1, 2, 3 };
    var depth: [W]i32 = undefined;
    const MAXD: c_int = 4;
    var path: [W * MAXD]i32 = @splat(0);
    for (0..W) |r| {
        depth[r] = if (parents[r] < 0) 0 else depth[@intCast(parents[r])] + 1;
        var cur: i32 = @intCast(r);
        var d = depth[r];
        while (cur >= 0) : (d -= 1) {
            path[r * MAXD + @as(usize, @intCast(d))] = cur;
            cur = parents[@intCast(cur)];
        }
    }
    const q = try randBf16(&.{ 1, H, W, D }, 11, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try randBf16(&.{ 1, HKV, P + W, D }, 12, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try randBf16(&.{ 1, HKV, P + W, D }, 13, s);
    defer _ = mlx.mlx_array_free(v);
    const tree = Tree{
        .depth = mlx.mlx_array_new_data(&depth, &[_]c_int{W}, 1, .int32),
        .path = mlx.mlx_array_new_data(&path, &[_]c_int{ W, MAXD }, 2, .int32),
        .max_depth = MAXD - 1,
    };
    defer _ = mlx.mlx_array_free(tree.depth);
    defer _ = mlx.mlx_array_free(tree.path);
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try testing.expect(try sdpa(&all, q, k, v, 1.0 / 16.0, tree, s));
    for (0..W) |r| {
        // The node's serial step: committed keys, then its path's rows in depth order.
        const du: usize = @intCast(depth[r]);
        var idx: [P + MAXD]i32 = undefined;
        for (0..P) |i| idx[i] = @intCast(i);
        for (0..du + 1) |i| idx[P + i] = P + path[r * MAXD + i];
        const ia = mlx.mlx_array_new_data(&idx, &[_]c_int{@intCast(P + du + 1)}, 1, .int32);
        defer _ = mlx.mlx_array_free(ia);
        var kr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kr);
        try mlx.check(mlx.mlx_take_axis(&kr, k, ia, 2, s));
        var vr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(vr);
        try mlx.check(mlx.mlx_take_axis(&vr, v, ia, 2, s));
        const ri: c_int = @intCast(r);
        const qr = try slice4(q, 2, ri, ri + 1, s);
        defer _ = mlx.mlx_array_free(qr);
        var one = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(one);
        try testing.expect(try sdpa(&one, qr, kr, vr, 1.0 / 16.0, null, s));
        const want = try slice4(all, 2, ri, ri + 1, s);
        defer _ = mlx.mlx_array_free(want);
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, one, want, false, s));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        try testing.expect(same);
    }
}
