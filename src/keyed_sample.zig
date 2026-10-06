//! Keyed Gumbel-max sampling on the GPU, ported from TensorFold's `gpu_sampling.py` (MIT, see NOTICE).
//! The token at absolute position p is argmax over the top-k/top-p candidates of v_i/T + g(seed, p, i),
//! g from a splitmix64 hash: a function of the row's own logits and its position only, so a verify row
//! draws the token the serial step draws, and a draft over the head's logits shares that noise.
const std = @import("std");
const mlx = @import("mlx.zig");

/// Most candidates a row sorts; tokens within NEAR (in logits / T) of the max are gathered directly.
const CANDIDATES = 1024;
const NEAR: f32 = 20.0;

const HEADER =
    \\inline uint tf_key(float v) { uint b = as_type<uint>(v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
    \\inline float tf_val(uint k) { uint b = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k; return as_type<float>(b); }
    \\inline ulong tf_mix(ulong x) {
    \\  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9UL; x ^= x >> 27; x *= 0x94D049BB133111EBUL; return x ^ (x >> 31);
    \\}
    \\inline float tf_uniform(ulong seed, uint pos, uint id) {
    \\  ulong x = tf_mix(seed + 0x9E3779B97F4A7C15UL);
    \\  x = tf_mix(x ^ (ulong(pos) * 0xD1B54A32D192ED03UL));
    \\  x = tf_mix(x ^ ulong(id));
    \\  return (float(uint(x >> 40)) + 0.5f) * (1.0f / 16777216.0f);
    \\}
;

const SOURCE =
    \\  constexpr uint TG = 1024;
    \\  constexpr uint NSG = TG / 32;
    \\  const uint t = thread_position_in_threadgroup.x;
    \\  const uint row = threadgroup_position_in_grid.x;
    \\  const uint lane = thread_index_in_simdgroup;
    \\  const uint sg = simdgroup_index_in_threadgroup;
    \\  const size_t base = size_t(row) * V;
    \\  const float inv_t = cfg[0];
    \\  const float top_p = cfg[1];
    \\  const float min_p = cfg[3];
    \\  const uint cap = (kcap[0] == 0u || kcap[0] > C) ? C : kcap[0];
    \\  const ulong seed = ulong(seeds[2 * row]) | (ulong(seeds[2 * row + 1]) << 32);
    \\  const uint position = positions[row];
    \\
    \\  threadgroup float fsh[NSG];
    \\  threadgroup float fsh2[NSG];
    \\  threadgroup uint ush[NSG];
    \\  threadgroup atomic_uint hist[256];
    \\  threadgroup uint st[4];
    \\  threadgroup uint ck[C];
    \\  threadgroup uint ci[C];
    \\  threadgroup atomic_uint fill_hi;
    \\  threadgroup atomic_uint fill_tie;
    \\
    \\  // row max (fixed order: each thread's stride, simd reduction, simdgroups in order)
    \\  float lm = -INFINITY;
    \\  for (uint i = t; i < V; i += TG) lm = max(lm, float(L[base + i]) * inv_t);
    \\  lm = simd_max(lm);
    \\  if (lane == 0) fsh[sg] = lm;
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  float m = -INFINITY;
    \\  for (uint s = 0; s < NSG; s++) m = max(m, fsh[s]);
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  // normalizer, and the count and mass of the tokens within NEAR, NEAR / 2 and NEAR / 4 of the max
    \\  float ls = 0.0f, lnear[3] = {0.0f, 0.0f, 0.0f};
    \\  uint near_count[3] = {0u, 0u, 0u};
    \\  for (uint i = t; i < V; i += TG) {
    \\    const float v = float(L[base + i]) * inv_t;
    \\    const float e = metal::exp(v - m);
    \\    ls += e;
    \\    for (int w = 0; w < 3; w++) {
    \\      if (v >= m - cfg[2] / float(1 << w)) { lnear[w] += e; near_count[w]++; }
    \\    }
    \\  }
    \\  ls = simd_sum(ls);
    \\  if (lane == 0) fsh[sg] = ls;
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  float z = 0.0f;
    \\  for (uint s = 0; s < NSG; s++) z += fsh[s];
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  // the widest window whose tokens are at most C and hold what the rule reads (top_p of the mass with a
    \\  // margin, or the top_k tokens): those tokens are a top segment of the (value desc, id asc) order
    \\  int window = -1;
    \\  uint offset = 0u, n_near = 0u;
    \\  for (int w = 0; w < 3; w++) {
    \\    const float znear_part = simd_sum(lnear[w]);
    \\    const uint before_in_simd = simd_prefix_exclusive_sum(near_count[w]);
    \\    const uint simd_count = simd_sum(near_count[w]);
    \\    if (lane == 0) { fsh2[sg] = znear_part; ush[sg] = simd_count; }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    float znear = 0.0f;
    \\    uint off = before_in_simd, count = 0u;
    \\    for (uint s = 0; s < NSG; s++) {
    \\      znear += fsh2[s];
    \\      if (s < sg) off += ush[s];
    \\      count += ush[s];
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    const bool holds = count <= C && (kcap[0] == 0u
    \\        ? (top_p > 0.0f && top_p < 1.0f && znear >= (top_p + 1e-4f) * z)
    \\        : count >= min(kcap[0], uint(C)));
    \\    if (window < 0 && holds) { window = w; offset = off; n_near = count; }
    \\  }
    \\  const float floor_v = window < 0 ? INFINITY : m - cfg[2] / float(1 << window);
    \\
    \\  // When a window holds the rule's candidates they are gathered at offsets from the prefix sum above (no
    \\  // atomics), then sorted; otherwise the radix path finds the C largest. Both give the same candidates for
    \\  // the rule, so the same token.
    \\  const bool fast = window >= 0;
    \\  uint n_cand = 0u;
    \\  uint sort_n = C;
    \\  ck[t] = 0u;
    \\  ci[t] = 0xFFFFFFFFu;
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (fast) {
    \\    uint at = offset;
    \\    for (uint i = t; i < V; i += TG) {
    \\      const float v = float(L[base + i]) * inv_t;
    \\      if (v >= floor_v) { ck[at] = tf_key(v); ci[at] = i; at++; }
    \\    }
    \\    n_cand = n_near;
    \\    sort_n = 32u;
    \\    while (sort_n < n_near) sort_n <<= 1;
    \\  } else {
    \\    // key of the C-th largest value: radix select over the key's bytes, most significant first
    \\    uint prefix = 0u, pmask = 0u, need = min(uint(C), uint(V)), above = 0u, ties = 0u;
    \\    for (int shift = 24; shift >= 0; shift -= 8) {
    \\      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\      for (uint i = t; i < V; i += TG) {
    \\        const uint k = tf_key(float(L[base + i]) * inv_t);
    \\        if ((k & pmask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    \\      }
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\      if (t == 0) {
    \\        uint cum = 0u;
    \\        int b = 255;
    \\        for (; b > 0; b--) {
    \\          const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
    \\          if (cum + c >= need) break;
    \\          cum += c;
    \\        }
    \\        st[0] = uint(b); st[1] = cum; st[2] = atomic_load_explicit(&hist[b], memory_order_relaxed);
    \\      }
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\      prefix |= st[0] << shift; pmask |= 255u << shift;
    \\      above += st[1]; need -= st[1]; ties = st[2];
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    // more ties at that value than places left: the lowest ids (a second radix select, over ids)
    \\    uint idcut = 0xFFFFFFFFu;
    \\    if (ties > need) {
    \\      uint ipre = 0u, imask = 0u, ineed = need;
    \\      for (int shift = 16; shift >= 0; shift -= 8) {
    \\        if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint i = t; i < V; i += TG) {
    \\          if (tf_key(float(L[base + i]) * inv_t) == prefix && (i & imask) == ipre)
    \\            atomic_fetch_add_explicit(&hist[(i >> shift) & 255u], 1u, memory_order_relaxed);
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        if (t == 0) {
    \\          uint cum = 0u;
    \\          uint b = 0u;
    \\          for (; b < 255u; b++) {
    \\            const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
    \\            if (cum + c >= ineed) break;
    \\            cum += c;
    \\          }
    \\          st[0] = b; st[1] = cum;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        ipre |= st[0] << shift; imask |= 255u << shift; ineed -= st[1];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\      }
    \\      idcut = ipre;
    \\    }
    \\    if (t == 0) {
    \\      atomic_store_explicit(&fill_hi, 0u, memory_order_relaxed);
    \\      atomic_store_explicit(&fill_tie, 0u, memory_order_relaxed);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint i = t; i < V; i += TG) {
    \\      const uint k = tf_key(float(L[base + i]) * inv_t);
    \\      if (k > prefix) {
    \\        const uint s = atomic_fetch_add_explicit(&fill_hi, 1u, memory_order_relaxed);
    \\        ck[s] = k; ci[s] = i;
    \\      } else if (k == prefix && i <= idcut) {
    \\        const uint s = above + atomic_fetch_add_explicit(&fill_tie, 1u, memory_order_relaxed);
    \\        ck[s] = k; ci[s] = i;
    \\      }
    \\    }
    \\    n_cand = above + need;
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\  // bitonic sort by (value desc, id asc): the order does not depend on how the candidates were gathered
    \\  for (uint k = 2; k <= sort_n; k <<= 1) {
    \\    for (uint j = k >> 1; j > 0; j >>= 1) {
    \\      const uint p = t ^ j;
    \\      if (p > t && p < sort_n) {
    \\        const uint ka = ck[t], kb = ck[p], ia = ci[t], ib = ci[p];
    \\        const bool a_first = ka > kb || (ka == kb && ia < ib);
    \\        if (a_first != ((t & k) == 0)) { ck[t] = kb; ck[p] = ka; ci[t] = ib; ci[p] = ia; }
    \\      }
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\  }
    \\
    \\  // the nucleus: softmax over the whole row (top_k 0) or over the top_k, cut where it reaches top_p
    \\  if (t == 0) {
    \\    const uint n = min(n_cand, cap);
    \\    float norm = z;
    \\    if (kcap[0] != 0u) {
    \\      norm = 0.0f;
    \\      for (uint j = 0; j < n; j++) norm += metal::exp(tf_val(ck[j]) - m);
    \\    }
    \\    uint keep = n;
    \\    if (top_p > 0.0f && top_p < 1.0f) {
    \\      float cum = 0.0f;
    \\      for (uint j = 0; j < n; j++) {
    \\        cum += metal::exp(tf_val(ck[j]) - m) / norm;
    \\        if (cum >= top_p) { keep = j + 1; break; }
    \\      }
    \\    }
    \\    if (min_p > 0.0f) {
    \\      const float floor_v = m + metal::log(min_p);
    \\      for (uint j = 0; j < keep; j++) {
    \\        if (tf_val(ck[j]) < floor_v) { keep = j; break; }
    \\      }
    \\    }
    \\    st[0] = keep;
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  const uint keep = st[0];
    \\
    \\  // Gumbel-max over the kept candidates; ties go to the earlier candidate
    \\  float score = -INFINITY;
    \\  uint best = 0xFFFFFFFFu;
    \\  if (t < keep) {
    \\    score = tf_val(ck[t]) - metal::log(-metal::log(tf_uniform(seed, position, ci[t])));
    \\    best = t;
    \\  }
    \\  const float sm = simd_max(score);
    \\  const uint pick = simd_min(score == sm ? best : 0xFFFFFFFFu);
    \\  if (lane == 0) { fsh[sg] = sm; ush[sg] = pick; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (t == 0) {
    \\    float bs = -INFINITY;
    \\    uint bj = 0xFFFFFFFFu;
    \\    for (uint s = 0; s < NSG; s++) {
    \\      if (fsh[s] > bs || (fsh[s] == bs && ush[s] < bj)) { bs = fsh[s]; bj = ush[s]; }
    \\    }
    \\    TOK[row] = ci[bj];
    \\  }
;

var kernel: ?mlx.mlx_fast_metal_kernel = null;
const CfgKey = struct { rows: c_int, vocab: c_int };
var cfg_cache: std.AutoHashMapUnmanaged(CfgKey, mlx.mlx_fast_metal_kernel_config) = .{};

fn config(key: CfgKey) !mlx.mlx_fast_metal_kernel_config {
    if (cfg_cache.get(key)) |c| return c;
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    const out_shape = [_]c_int{key.rows};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 1024 * key.rows, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "V", key.vocab));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "C", CANDIDATES));
    try cfg_cache.put(std.heap.c_allocator, key, c);
    return c;
}

fn mix(v: u64) u64 {
    var x = v;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

/// The kernel's Gumbel noise for token `id` at `position`, on the host.
pub fn gumbel(seed: u64, position: u64, id: u32) f32 {
    var x = mix(seed +% 0x9E3779B97F4A7C15);
    x = mix(x ^ (@as(u64, @as(u32, @truncate(position))) *% 0xD1B54A32D192ED03));
    x = mix(x ^ id);
    const u = (@as(f32, @floatFromInt(@as(u32, @truncate(x >> 40)))) + 0.5) * (1.0 / 16777216.0);
    return -@log(-@log(u));
}

pub const Params = struct { seed: u64, temperature: f32, top_p: f32, top_k: u32, min_p: f32 = 0 };

/// Tokens `[R]` uint32 (lazy) for `logits` `[..., V]` (R rows), row r at absolute position `positions[r]`.
pub fn sample(logits: mlx.mlx_array, p: Params, positions: []const u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(logits);
    const vocab = sh[sh.len - 1];
    const rows: c_int = @intCast(positions.len);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, logits, &[_]c_int{ rows, vocab }, 2, s));

    var seed_buf: [2 * 64]u32 = undefined;
    if (rows > 64) return error.TooManyRows;
    for (0..@intCast(rows)) |r| {
        seed_buf[2 * r] = @truncate(p.seed);
        seed_buf[2 * r + 1] = @truncate(p.seed >> 32);
    }
    const seeds = mlx.mlx_array_new_data(&seed_buf, &[_]c_int{2 * rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(seeds);
    const pos = mlx.mlx_array_new_data(positions.ptr, &[_]c_int{rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(pos);
    const cfg_v = [_]f32{ 1.0 / @max(p.temperature, 1e-6), p.top_p, NEAR, p.min_p };
    const cfg = mlx.mlx_array_new_data(&cfg_v, &[_]c_int{4}, 1, .float32);
    defer _ = mlx.mlx_array_free(cfg);
    const kcap = mlx.mlx_array_new_data(&p.top_k, &[_]c_int{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(kcap);

    if (kernel == null) {
        const names_in = [_][*:0]const u8{ "L", "seeds", "positions", "cfg", "kcap" };
        const names_out = [_][*:0]const u8{"TOK"};
        const in_n = mlx.mlx_vector_string_new_data(&names_in, names_in.len);
        defer _ = mlx.mlx_vector_string_free(in_n);
        const out_n = mlx.mlx_vector_string_new_data(&names_out, names_out.len);
        defer _ = mlx.mlx_vector_string_free(out_n);
        const k = mlx.mlx_fast_metal_kernel_new("msv_keyed_sample", in_n, out_n, SOURCE, HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel = k;
    }
    const inputs = [_]mlx.mlx_array{ flat, seeds, pos, cfg, kcap };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, kernel.?, in_vec, try config(.{ .rows = rows, .vocab = vocab }), s));
    var tok = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&tok, outs, 0));
    return tok;
}

const testing = std.testing;

test "keyed_sample: draws TensorFold's reference tokens, a row alone or among others" {
    const s = mlx.gpuStream();
    const V = 3000;
    var v: [V]f32 = undefined;
    for (&v, 0..) |*x, i| {
        const f: f32 = @floatFromInt(i);
        x.* = 6.0 * @sin(f * 0.37) * @cos(f * 0.011) - @as(f32, @floatFromInt(i % 13)) * 0.25;
    }
    // Expected tokens from tensorfold.engine.gpu_sampling.reference on the same row.
    const Case = struct { p: Params, positions: []const u32, want: []const u32 };
    const cases = [_]Case{
        .{ .p = .{ .seed = 1234, .temperature = 1.0, .top_p = 0.95, .top_k = 20 }, .positions = &.{ 31, 32, 200 }, .want = &.{ 1158, 2263, 2314 } },
        .{ .p = .{ .seed = 99, .temperature = 1.0, .top_p = 0.95, .top_k = 20 }, .positions = &.{ 31, 32, 200 }, .want = &.{ 1703, 1159, 1405 } },
        .{ .p = .{ .seed = 99, .temperature = 0.7, .top_p = 0.9, .top_k = 0 }, .positions = &.{ 31, 32, 200 }, .want = &.{ 861, 2577, 2891 } },
        .{ .p = .{ .seed = 1234, .temperature = 2.0, .top_p = 1.0, .top_k = 0 }, .positions = &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, .want = &.{ 316, 1822, 4, 2718, 599, 616, 2576, 1756 } },
    };
    var rows_buf: [8 * V]f32 = undefined;
    for (cases) |c| {
        const n = c.positions.len;
        for (0..n) |r| @memcpy(rows_buf[r * V ..][0..V], &v);
        const logits = mlx.mlx_array_new_data(&rows_buf, &[_]c_int{ @intCast(n), V }, 2, .float32);
        defer _ = mlx.mlx_array_free(logits);
        const all = try sample(logits, c.p, c.positions, s);
        defer _ = mlx.mlx_array_free(all);
        try mlx.check(mlx.mlx_array_eval(all));
        try testing.expectEqualSlices(u32, c.want, mlx.mlx_array_data_uint32(all).?[0..n]);
        const one_row = mlx.mlx_array_new_data(&v, &[_]c_int{ 1, 1, V }, 3, .float32);
        defer _ = mlx.mlx_array_free(one_row);
        for (c.positions, c.want) |pos, want| {
            const one = try sample(one_row, c.p, &.{pos}, s);
            defer _ = mlx.mlx_array_free(one);
            try mlx.check(mlx.mlx_array_eval(one));
            try testing.expectEqual(want, mlx.mlx_array_data_uint32(one).?[0]);
        }
    }
}

test "keyed_sample.gumbel: the host noise picks the kernel's token" {
    const s = mlx.gpuStream();
    const V = 3000;
    var v: [V]f32 = undefined;
    for (&v, 0..) |*x, i| x.* = 6.0 * @sin(@as(f32, @floatFromInt(i)) * 0.37);
    var ids: [V]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast(i);
    std.mem.sort(u32, &ids, &v, struct {
        fn lt(vals: *const [V]f32, a: u32, b: u32) bool {
            return vals[a] > vals[b] or (vals[a] == vals[b] and a < b);
        }
    }.lt);
    const logits = mlx.mlx_array_new_data(&v, &[_]c_int{ 1, V }, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    for (0..32) |pos| {
        const p: Params = .{ .seed = 1234, .temperature = 1.0, .top_p = 1.0, .top_k = 20 };
        const tok = try sample(logits, p, &.{@intCast(pos)}, s);
        defer _ = mlx.mlx_array_free(tok);
        try mlx.check(mlx.mlx_array_eval(tok));
        var best: u32 = ids[0];
        var best_score = -std.math.inf(f32);
        for (ids[0..20]) |id| {
            const score = v[id] + gumbel(p.seed, pos, id);
            if (score > best_score) {
                best_score = score;
                best = id;
            }
        }
        try testing.expectEqual(best, mlx.mlx_array_data_uint32(tok).?[0]);
    }
}

test "keyed_sample min_p excludes low logits after temperature and top_k" {
    const s = mlx.gpuStream();
    const raw = [_]f32{ 3, 2, 0, -5 };
    const logits = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const p: Params = .{ .seed = 9, .temperature = 1, .top_p = 1, .top_k = 2, .min_p = 0.5 };
    for (0..8) |draw| {
        const pos = [_]u32{@intCast(draw)};
        const tok = try sample(logits, p, &pos, s);
        defer _ = mlx.mlx_array_free(tok);
        try mlx.check(mlx.mlx_array_eval(tok));
        try testing.expectEqual(@as(u32, 0), mlx.mlx_array_data_uint32(tok).?[0]);
    }
}

test "keyed_sample min_p retains the temperature-scaled nucleus" {
    const s = mlx.gpuStream();
    const raw = [_]f32{ 3, 2.8, 0, -5 };
    const logits = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const p: Params = .{ .seed = 1234, .temperature = 2, .top_p = 0.9, .top_k = 3, .min_p = 0.5 };
    for (0..32) |draw| {
        const pos: u32 = @intCast(draw);
        const tok = try sample(logits, p, &.{pos}, s);
        defer _ = mlx.mlx_array_free(tok);
        try mlx.check(mlx.mlx_array_eval(tok));
        const score0 = raw[0] / p.temperature + gumbel(p.seed, pos, 0);
        const score1 = raw[1] / p.temperature + gumbel(p.seed, pos, 1);
        try testing.expectEqual(@as(u32, if (score0 >= score1) 0 else 1), mlx.mlx_array_data_uint32(tok).?[0]);
    }
}
