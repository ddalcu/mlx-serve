//! Hyper-connection read at ONE decode row in two dispatches instead of three. The stream norm
//! folds into the down launch (the same idea as oMLX's two-launch decode, Apache-2.0): the down
//! projection's K slice for stream s is exactly stream s, so a threadgroup normalizes only its
//! own stream and multiplies the slice, with its weight loads issued before the norm finishes.
//! The slice partial sums meet in the up/mix launch, which sums them into the activation once per
//! threadgroup. Rounding sites are the three-kernel path's (xn, silu, up, gate, mix); only the
//! f32 accumulation order differs.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

/// A deferred hcWrite the norm launch applies itself: stream + out * inj.
pub const Pending = struct { out: mlx.mlx_array, inj: mlx.mlx_array };

/// Flat results: mixed [hidden], inj [hc] (null-ctx without inject weights), stream [hc*hidden]
/// (null-ctx without a pending write).
pub const Out = struct { mixed: mlx.mlx_array, inj: mlx.mlx_array, stream: mlx.mlx_array };

/// Down rows a simdgroup owns (a threadgroup has 8 simdgroups): one row each spreads the launch over
/// the most threadgroups, where four rows left cores idle.
const RPS: c_int = 1;
const ROWS_TG: c_int = 8 * RPS;
/// Up columns a threadgroup owns, one per simdgroup.
const COLS_TG: c_int = 16;

const ND_SOURCE =
    \\const uint t = thread_index_in_threadgroup;
    \\const int row = int(threadgroup_position_in_grid.z);
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const int s = int(threadgroup_position_in_grid.y) % HC;
    \\const int g = int(threadgroup_position_in_grid.y) / HC;
    \\constexpr int PER = H / 256;
    \\constexpr int K = HC * H;
    \\// A slot's 16 values are WPL words at 2/4/8 bits, else two 8-value packs of BITS bytes.
    \\constexpr bool WORD = (32 % BITS) == 0;
    \\constexpr int VPW = WORD ? 32 / BITS : 8;
    \\constexpr int WPL = 16 / VPW;
    \\constexpr int KW = K / VPW;
    \\constexpr int KG = K / GS;
    \\constexpr int NSLOT = H / 16;
    \\constexpr int SITER = (NSLOT + 31) / 32;
    \\constexpr int PR = R + HC;
    \\constexpr int NDG = R / ROWS_TG;
    \\constexpr uint MASK = (1u << BITS) - 1u;
    \\threadgroup T xs[H];
    \\threadgroup float npart[8];
    \\threadgroup float tgi[8 * HC];
    \\const int base = s * H;
    \\const size_t xrow = (size_t)row * K;
    \\const bool down = g < NDG;
    \\
    \\// The down weights do not wait for the norm: every load of this simdgroup's rows goes out first.
    \\uint pw[RPS][SITER][WORD ? WPL : 1];
    \\ulong pr[RPS][SITER][WORD ? 1 : WPL];
    \\float ps[RPS][SITER];
    \\float pb[RPS][SITER];
    \\const int n0 = g * ROWS_TG + int(sg) * RPS;
    \\if (down) {
    \\  for (int r = 0; r < RPS; ++r) {
    \\    for (int i = 0; i < SITER; ++i) {
    \\      const int slot = i * 32 + int(lane);
    \\      if (slot < NSLOT) {
    \\        if constexpr (WORD) {
    \\          const device uint32_t* wp = dw_q + (size_t)(n0 + r) * KW + (size_t)(s * (H / VPW) + slot * WPL);
    \\          for (int e = 0; e < WPL; ++e) pw[r][i][e] = wp[e];
    \\        } else {
    \\          const device uchar* wb = (const device uchar*)dw_q + (size_t)(n0 + r) * (size_t)(K * BITS / 8) + (size_t)(base + slot * 16) * BITS / 8;
    \\          for (int e = 0; e < WPL; ++e) pr[r][i][e] = hc_pack8<BITS>(wb + e * BITS);
    \\        }
    \\        const size_t gi = (size_t)(n0 + r) * KG + (size_t)((base + slot * 16) / GS);
    \\        ps[r][i] = float(dw_s[gi]);
    \\        pb[r][i] = float(dw_b[gi]);
    \\      }
    \\    }
    \\  }
    \\}
    \\
    \\float wv[PER];
    \\for (int i = 0; i < PER; ++i) wv[i] = float(nw[base + int(t) + 256 * i]);
    \\float xv[PER];
    \\T written[PER];
    \\if (WR) {
    \\  // Pending hcWrite: stream' = T(stream + T(out * inj)), the chain's two roundings.
    \\  const float gate = float(wi_in[row * HC + s]);
    \\  for (int i = 0; i < PER; ++i) {
    \\    const int k = int(t) + 256 * i;
    \\    const T v = T(float(x_in[xrow + base + k]) + float(T(float(wo_in[(size_t)row * H + k]) * gate)));
    \\    written[i] = v;
    \\    xv[i] = float(v);
    \\  }
    \\} else {
    \\  for (int i = 0; i < PER; ++i) xv[i] = float(x_in[xrow + base + int(t) + 256 * i]);
    \\}
    \\float a = 0.0f;
    \\for (int i = 0; i < PER; ++i) a += xv[i] * xv[i];
    \\a = simd_sum(a);
    \\if (lane == 0) npart[sg] = a;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float tot = 0.0f;
    \\for (int q = 0; q < 8; ++q) tot += npart[q];
    \\const float rsh = rsqrt(tot / float(H) + eps[0]);
    \\T xnv[PER];
    \\for (int i = 0; i < PER; ++i) {
    \\  xnv[i] = T(float(T(xv[i] * rsh)) * wv[i]);
    \\  xs[int(t) + 256 * i] = xnv[i];
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\device float* pp = parts_out + (size_t)(row * HC + s) * PR;
    \\if (down) {
    \\  float acc[RPS];
    \\  for (int r = 0; r < RPS; ++r) acc[r] = 0.0f;
    \\  for (int i = 0; i < SITER; ++i) {
    \\    const int slot = i * 32 + int(lane);
    \\    if (slot < NSLOT) {
    \\      float xr[16];
    \\      float xsum = 0.0f;
    \\      for (int e = 0; e < 16; ++e) { xr[e] = float(xs[slot * 16 + e]); xsum += xr[e]; }
    \\      for (int r = 0; r < RPS; ++r) {
    \\        float d = 0.0f;
    \\        for (int w = 0; w < WPL; ++w) {
    \\          if constexpr (WORD) {
    \\            for (int j = 0; j < VPW; ++j) d += xr[w * VPW + j] * float((pw[r][i][w] >> (j * BITS)) & MASK);
    \\          } else {
    \\            for (int j = 0; j < VPW; ++j) d += xr[w * VPW + j] * float(uint(pr[r][i][w] >> (j * BITS)) & MASK);
    \\          }
    \\        }
    \\        acc[r] += ps[r][i] * d + pb[r][i] * xsum;
    \\      }
    \\    }
    \\  }
    \\  for (int r = 0; r < RPS; ++r) {
    \\    const float v = simd_sum(acc[r]);
    \\    if (lane == 0) pp[n0 + r] = v;
    \\  }
    \\} else if (INJ != 0) {
    \\  // This stream's partial of the inject matvec; the up launch adds the streams.
    \\  float ip[HC];
    \\  for (int c = 0; c < HC; ++c) ip[c] = 0.0f;
    \\  for (int i = 0; i < PER; ++i) {
    \\    const size_t k = (size_t)(base + int(t) + 256 * i);
    \\    for (int c = 0; c < HC; ++c) ip[c] += float(xnv[i]) * float(iw[k * HC + c]);
    \\  }
    \\  for (int c = 0; c < HC; ++c) {
    \\    const float pa = simd_sum(ip[c]);
    \\    if (lane == 0) tgi[sg * HC + c] = pa;
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (t < uint(HC)) {
    \\    float tt = 0.0f;
    \\    for (int q = 0; q < 8; ++q) tt += tgi[q * HC + t];
    \\    pp[R + t] = tt;
    \\  }
    \\}
    \\// Device stores come last: a later device load waits on every earlier store it cannot prove disjoint.
    \\if (g == 0) {
    \\  for (int i = 0; i < PER; ++i) {
    \\    xn_out[xrow + base + int(t) + 256 * i] = xnv[i];
    \\    if (WR) xs_out[xrow + base + int(t) + 256 * i] = written[i];
    \\  }
    \\}
;

const U2_SOURCE =
    \\const uint t = thread_index_in_threadgroup;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const int row = int(threadgroup_position_in_grid.z);
    \\const int j = int(threadgroup_position_in_grid.y) * COLS + int(sg);
    \\constexpr bool WORD = (32 % BITS) == 0;
    \\constexpr int VPW = WORD ? 32 / BITS : 8;
    \\constexpr int R_by_p = R / VPW;
    \\constexpr int R_by_gs = R / GS;
    \\constexpr int RIT = (R_by_p + 31) / 32;
    \\constexpr int PR = R + HC;
    \\constexpr uint MASK = (1u << BITS) - 1u;
    \\threadgroup T acts[R];
    \\
    \\// This column's up rows (one per stream) do not wait for the activation.
    \\uint pw[HC][WORD ? RIT : 1];
    \\ulong pr[HC][WORD ? 1 : RIT];
    \\float ps[HC][RIT];
    \\float pb[HC][RIT];
    \\float xnv[HC];
    \\for (int h = 0; h < HC; ++h) {
    \\  const size_t hrow = (size_t)h * H + (size_t)j;
    \\  xnv[h] = float(xn_in[(size_t)row * HC * H + hrow]);
    \\  for (int i = 0; i < RIT; ++i) {
    \\    const int pack = int(lane) + 32 * i;
    \\    if (pack < R_by_p) {
    \\      if constexpr (WORD) pw[h][i] = uw_q[hrow * R_by_p + (size_t)pack];
    \\      else pr[h][i] = hc_pack8<BITS>((const device uchar*)uw_q + hrow * (size_t)(R * BITS / 8) + (size_t)pack * BITS);
    \\      const size_t gi = hrow * R_by_gs + (size_t)((pack * VPW) / GS);
    \\      ps[h][i] = float(uw_s[gi]);
    \\      pb[h][i] = float(uw_b[gi]);
    \\    }
    \\  }
    \\}
    \\// act = silu(T(sum of the stream partials)), as the three-kernel down launch ends.
    \\for (int i = int(t); i < R; i += COLS * 32) {
    \\  float tsum = 0.0f;
    \\  for (int s = 0; s < HC; ++s) tsum += parts_in[(row * HC + s) * PR + i];
    \\  const T v = T(tsum);
    \\  const T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
    \\  acts[i] = v * sig;
    \\}
    \\if (INJ != 0 && threadgroup_position_in_grid.y == 0 && t < uint(HC)) {
    \\  float tsum = 0.0f;
    \\  for (int s = 0; s < HC; ++s) tsum += parts_in[(row * HC + s) * PR + R + int(t)];
    \\  const T v = T(tsum);
    \\  const T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
    \\  inj_out[row * HC + t] = sig * T(2.0f);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\float sum = 0.0f;
    \\for (int h = 0; h < HC; ++h) {
    \\  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    \\  for (int i = 0; i < RIT; ++i) {
    \\    const int pack = int(lane) + 32 * i;
    \\    if (pack < R_by_p) {
    \\      const int k_base = pack * VPW;
    \\      for (int ki = 0; ki < VPW; ki += 4) {
    \\        const int k = k_base + ki;
    \\        // Values ki..ki+3 of this pack; a 3/5/6-bit pack is 8 values in a ulong.
    \\        const uint q = WORD ? (pw[h][i] >> (ki * BITS)) : uint(pr[h][i] >> (ki * BITS));
    \\        a0 += float(acts[k + 0]) * (float((q >> (0 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
    \\        a1 += float(acts[k + 1]) * (float((q >> (1 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
    \\        a2 += float(acts[k + 2]) * (float((q >> (2 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
    \\        a3 += float(acts[k + 3]) * (float((q >> (3 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
    \\      }
    \\    }
    \\  }
    \\  const float acc = simd_sum((a0 + a1) + (a2 + a3));
    \\  const T u = T(acc);
    \\  const T sgv = T(1.0f / (1.0f + metal::exp(-float(u))));
    \\  sum += float(T(float(sgv) * xnv[h]));
    \\}
    \\if (lane == 0) mixed_out[(size_t)row * H + j] = T(float(T(sum)) * float(T(1.0f / float(HC))));
;

/// The up/mix launch for 1..16 rows at hc 4, 8-bit. Lanes split
/// [part:2][stream:2][column:1], so one lane holds one stream's slice of a weight row in registers (16-byte loads issued
/// before any math) and every row's activations are applied to it; the activations (the stream partials summed, then
/// silu) are staged once per threadgroup. Rounding sites are U2_SOURCE's; only the f32 summation order differs.
const U2V_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint tid = thread_index_in_threadgroup;
    \\const uint sgi = simdgroup_index_in_threadgroup;
    \\const uint part = lane & 3u;
    \\const uint h = (lane >> 2) & 3u;
    \\const uint cl = lane >> 4;
    \\const uint j = threadgroup_position_in_grid.x * uint(SGS * 2) + sgi * 2u + cl;
    \\constexpr int VPW = 32 / BITS;
    \\constexpr int R_by_p = R / VPW;
    \\constexpr int R_by_gs = R / GS;
    \\constexpr int PV = R / 4;
    \\constexpr int PU4 = PV / (4 * VPW);
    \\constexpr int PR = R + HC;
    \\threadgroup float act_tg[ROWS * R];
    \\const size_t wrow = (size_t)h * (size_t)H + (size_t)j;
    \\const device uint4* wq4 = (const device uint4*)(uw_q + wrow * (size_t)R_by_p) + part * PU4;
    \\uint4 wv[PU4];
    \\for (int i = 0; i < PU4; ++i) wv[i] = wq4[i];
    \\float sc[PU4], bi[PU4];
    \\for (int i = 0; i < PU4; ++i) {
    \\  const int k0 = int(part) * PV + i * 4 * VPW;
    \\  sc[i] = float(uw_s[wrow * (size_t)R_by_gs + (size_t)(k0 / GS)]);
    \\  bi[i] = float(uw_b[wrow * (size_t)R_by_gs + (size_t)(k0 / GS)]);
    \\}
    \\float xnv[ROWS];
    \\for (int r = 0; r < ROWS; ++r) xnv[r] = float(xn_in[(size_t)r * (size_t)(HC * H) + wrow]);
    \\for (uint e = tid; e < uint(ROWS * R); e += uint(SGS * 32)) {
    \\  const int r = int(e) / R;
    \\  const int k = int(e) % R;
    \\  float tsum = 0.0f;
    \\  for (int s = 0; s < HC; ++s) tsum += parts_in[(size_t)(r * HC + s) * PR + k];
    \\  const T v = T(tsum);
    \\  const T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
    \\  act_tg[e] = float(v * sig);
    \\}
    \\if (INJ != 0 && threadgroup_position_in_grid.x == 0 && tid < uint(ROWS * HC)) {
    \\  const int r = int(tid) / HC;
    \\  const int c = int(tid) % HC;
    \\  float tsum = 0.0f;
    \\  for (int s = 0; s < HC; ++s) tsum += parts_in[(size_t)(r * HC + s) * PR + R + c];
    \\  const T v = T(tsum);
    \\  const T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
    \\  inj_out[tid] = sig * T(2.0f);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float acc[ROWS];
    \\for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    \\for (int i = 0; i < PU4; ++i) {
    \\  const int k0 = int(part) * PV + i * 4 * VPW;
    \\  const uint words[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
    \\  for (int wd = 0; wd < 4; ++wd) {
    \\    for (int b = 0; b < VPW; ++b) {
    \\      const float wf = float((words[wd] >> (b * BITS)) & ((1u << BITS) - 1u)) * sc[i] + bi[i];
    \\      const int k = k0 + wd * VPW + b;
    \\      for (int r = 0; r < ROWS; ++r) acc[r] += act_tg[r * R + k] * wf;
    \\    }
    \\  }
    \\}
    \\float mix[ROWS];
    \\for (int r = 0; r < ROWS; ++r) {
    \\  float a = acc[r];
    \\  a += simd_shuffle_xor(a, 1);
    \\  a += simd_shuffle_xor(a, 2);
    \\  const T u = T(a);
    \\  const T sgv = T(1.0f / (1.0f + metal::exp(-float(u))));
    \\  float m = float(T(float(sgv) * xnv[r]));
    \\  m += simd_shuffle_xor(m, 4);
    \\  m += simd_shuffle_xor(m, 8);
    \\  mix[r] = m;
    \\}
    \\if ((lane & 15u) == 0) {
    \\  for (int r = 0; r < ROWS; ++r) mixed_out[(size_t)r * (size_t)H + (size_t)j] = T(float(T(mix[r])) * float(T(1.0f / float(HC))));
    \\}
;

var env_enabled: ?bool = null;
pub var override: ?bool = null;
var engaged = false;
/// Reads served by the two launches (the tests assert the path engaged).
pub var served: u64 = 0;
/// Of those, reads whose up launch took the vectorized kernel.
pub var served_vec: u64 = 0;

/// `MLX_SERVE_HC_DECODE2=0` keeps the unfused chain.
pub fn enabled() bool {
    if (override) |v| return v;
    if (env_enabled) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_HC_DECODE2");
    env_enabled = raw == null or raw.?[0] != '0';
    return env_enabled.?;
}

var env_rows: ?c_int = null;
pub var rows_override: ?c_int = null;
/// Widest read the two launches serve (the verify widths and batched slots); `MLX_SERVE_HC_DECODE2_ROWS=1` limits it to one row.
pub fn maxRows() c_int {
    if (rows_override) |v| return v;
    if (env_rows) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_HC_DECODE2_ROWS");
    const parsed = if (raw) |r| std.fmt.parseInt(c_int, std.mem.span(r), 10) catch 16 else 16;
    env_rows = std.math.clamp(parsed, 1, 16);
    return env_rows.?;
}

/// Values per pack the HC kernels read: one word at 2/4/8 bits, 8 values in BITS bytes at 3/5/6.
pub fn packValues(bits: u32) ?c_int {
    return switch (bits) {
        2, 4, 8 => @intCast(32 / bits),
        3, 5, 6 => 8,
        else => null,
    };
}

/// Eight B-bit values packed in B bytes (3/5/6-bit), little-endian as mx.quantize lays them out.
pub const PACK8_HEADER =
    \\template <int B>
    \\inline ulong hc_pack8(const device uchar* p) {
    \\  ulong r = 0;
    \\  for (int b = 0; b < B; ++b) r |= ulong(p[b]) << (8 * b);
    \\  return r;
    \\}
;

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, PACK8_HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

/// A launch config bakes grid, outputs and template values, which a layer's geometry fixes.
fn config(outs: []const struct { c_int, mlx.mlx_dtype }, grid: [3]c_int, tg: c_int, tmpl: []const struct { [*:0]const u8, c_int }, dt: mlx.mlx_dtype) !mlx.mlx_fast_metal_kernel_config {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{o[0]}, 1, o[1]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
    for (tmpl) |a| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, a[0], a[1]));
    return c;
}

fn run(kernel: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, inputs: []const mlx.mlx_array, s: mlx.mlx_stream, res: []mlx.mlx_array) !void {
    const v = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel, v, cfg, s));
    if (mlx.mlx_vector_array_size(o) != res.len) return error.MetalKernelBadOutputCount;
    var got: usize = 0;
    errdefer for (res[0..got]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (res, 0..) |*r, i| {
        r.* = mlx.mlx_array_new();
        got = i + 1;
        try mlx.check(mlx.mlx_vector_array_get(r, o, i));
    }
}

const Key = struct { hc: c_int, h: c_int, r: c_int, inj: c_int, wr: c_int, bits: u32, gs: c_int, dtype: mlx.mlx_dtype, rows: c_int };
const Armed = struct { key: Key, nd: mlx.mlx_fast_metal_kernel_config, up: mlx.mlx_fast_metal_kernel_config, upv: ?mlx.mlx_fast_metal_kernel_config };
var nd_kernel: ?mlx.mlx_fast_metal_kernel = null;
var up_kernel: ?mlx.mlx_fast_metal_kernel = null;
var up_v_kernel: ?mlx.mlx_fast_metal_kernel = null;
/// A mixed pack's per-layer widths each keep their configs, so no layer re-arms; round-robin eviction.
var armed: [16]?Armed = @splat(null);
var armed_next: usize = 0;

/// Rows the vectorized up launch serves (hc 4, 8-bit weights, 1..16 rows): its threadgroup holds every row's activations.
/// One row takes it too, so a row sums in the same order at every row count and a grouped verify stays byte-identical to a
/// solo one.
fn upVectorized(rows: c_int, hc: c_int, r: c_int, bits: u32, gs: c_int) bool {
    if (rows < 1 or rows > 16 or hc != 4 or bits != 8 or @rem(r, 64) != 0 or r > 512 or @rem(gs, 16) != 0) return false;
    return true;
}

/// Simdgroups per threadgroup of the vectorized up launch, two columns each. One row has the least work per column, so the
/// widest threadgroup hides the weight loads best there; more rows already fill the time. The arithmetic order does not
/// depend on this, so a row sums the same at every row count whichever width serves it.
fn upvSgs(rows: c_int) c_int {
    return if (rows == 1) 16 else 8;
}

/// The up launch's 16-byte weight loads need a 16-byte-aligned buffer. A lazy weight has no data pointer yet, so it is
/// evaluated here: the launch must not depend on whether an earlier read already did, or one row would sum in
/// a different order from the same row read before the weights were resident.
fn aligned16(a: mlx.mlx_array) bool {
    var avail = false;
    if (mlx._mlx_array_is_available(&avail, a) != 0) return false;
    if (!avail) mlx.check(mlx.mlx_array_eval(a)) catch return false;
    const p = mlx.mlx_array_data_uint32(a) orelse return false;
    return @intFromPtr(p) % 16 == 0;
}

/// The two-launch read of `rows` rows, or null outside its envelope (the caller keeps the
/// unfused chain). `dw` [R, K/vpw] and `uw` [K, R/vpw] are the packed projections with
/// scales and biases in x's dtype, `iw` the dense [K, hc] inject weight or null-ctx.
pub fn read(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    nw: mlx.mlx_array,
    dw: mlx.mlx_array,
    ds: mlx.mlx_array,
    db: mlx.mlx_array,
    uw: mlx.mlx_array,
    us: mlx.mlx_array,
    ub: mlx.mlx_array,
    iw: mlx.mlx_array,
    eps: mlx.mlx_array,
    rows: c_int,
    hc: c_int,
    hidden: c_int,
    bits: u32,
    group_size: u32,
    pend: ?Pending,
) !?Out {
    if (!enabled() or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    inline for (.{ nw, ds, db, us, ub }) |a| {
        if (a.ctx == null or mlx.mlx_array_dtype(a) != dt) return null;
    }
    if (iw.ctx != null and mlx.mlx_array_dtype(iw) != dt) return null;
    const vpw = packValues(bits) orelse return null;
    const b: c_int = @intCast(bits);
    const gs: c_int = @intCast(group_size);
    const K = hc * hidden;
    if (hc < 1 or hc > 8 or @rem(hidden, 256) != 0 or @rem(gs, 16) != 0 or @rem(hidden, gs) != 0) return null;
    const dsh = mlx.getShape(dw);
    const ush = mlx.getShape(uw);
    if (dsh.len != 2 or ush.len != 2 or dsh[1] * 32 != K * b or ush[0] != K) return null;
    const R = dsh[0];
    if (ush[1] * 32 != R * b or @rem(R, vpw) != 0 or @rem(R, ROWS_TG) != 0 or @rem(R, gs) != 0 or @rem(hidden, COLS_TG) != 0) return null;
    // A lane's 16 values are whole words, and a down row's slice splits into whole slots.
    if (@rem(16, vpw) != 0 or @rem(hidden, 16) != 0) return null;
    const inj: c_int = @intFromBool(iw.ctx != null);
    const wr: c_int = @intFromBool(pend != null);
    if (rows < 1 or mlx.mlx_array_size(x) != @as(usize, @intCast(rows * K)) or mlx.mlx_array_size(nw) != @as(usize, @intCast(K))) return null;

    const key = Key{ .hc = hc, .h = hidden, .r = R, .inj = inj, .wr = wr, .bits = bits, .gs = gs, .dtype = dt, .rows = rows };
    if (nd_kernel == null) {
        const ins = [_][*:0]const u8{ "x_in", "nw", "iw", "eps", "wo_in", "wi_in", "dw_q", "dw_s", "dw_b" };
        const outs = [_][*:0]const u8{ "xn_out", "parts_out", "xs_out" };
        nd_kernel = try makeKernel("mlxserve_hc_read_nd", &ins, &outs, ND_SOURCE);
    }
    if (up_kernel == null) {
        const ins = [_][*:0]const u8{ "xn_in", "parts_in", "uw_q", "uw_s", "uw_b" };
        const outs = [_][*:0]const u8{ "mixed_out", "inj_out" };
        up_kernel = try makeKernel("mlxserve_hc_read_u2", &ins, &outs, U2_SOURCE);
    }
    if (up_v_kernel == null) {
        const ins = [_][*:0]const u8{ "xn_in", "parts_in", "uw_q", "uw_s", "uw_b" };
        const outs = [_][*:0]const u8{ "mixed_out", "inj_out" };
        up_v_kernel = try makeKernel("mlxserve_hc_read_u2v", &ins, &outs, U2V_SOURCE);
    }
    const pr: c_int = R + hc;
    const cfgs: Armed = for (armed) |a| {
        if (a != null and std.meta.eql(a.?.key, key)) break a.?;
    } else blk: {
        const groups = @divExact(R, ROWS_TG) + inj;
        const nd = try config(&.{ .{ rows * K, dt }, .{ rows * hc * pr, .float32 }, .{ if (wr == 1) rows * K else 1, dt } }, .{ 256, groups * hc, rows }, 256, &.{ .{ "HC", hc }, .{ "H", hidden }, .{ "R", R }, .{ "GS", gs }, .{ "BITS", @intCast(bits) }, .{ "INJ", inj }, .{ "WR", wr }, .{ "RPS", RPS }, .{ "ROWS_TG", ROWS_TG } }, dt);
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(nd);
        const up = try config(&.{ .{ rows * hidden, dt }, .{ rows * hc, dt } }, .{ 32 * COLS_TG, @divExact(hidden, COLS_TG), rows }, 32 * COLS_TG, &.{ .{ "HC", hc }, .{ "H", hidden }, .{ "R", R }, .{ "GS", gs }, .{ "BITS", @intCast(bits) }, .{ "INJ", inj }, .{ "COLS", COLS_TG } }, dt);
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(up);
        const upv: ?mlx.mlx_fast_metal_kernel_config = if (upVectorized(rows, hc, R, bits, gs))
            try config(&.{ .{ rows * hidden, dt }, .{ rows * hc, dt } }, .{ 32 * upvSgs(rows) * @divExact(hidden, 2 * upvSgs(rows)), 1, 1 }, 32 * upvSgs(rows), &.{ .{ "HC", hc }, .{ "H", hidden }, .{ "R", R }, .{ "GS", gs }, .{ "BITS", @intCast(bits) }, .{ "INJ", inj }, .{ "ROWS", rows }, .{ "SGS", upvSgs(rows) } }, dt)
        else
            null;
        if (armed[armed_next]) |old| {
            _ = mlx.mlx_fast_metal_kernel_config_free(old.nd);
            _ = mlx.mlx_fast_metal_kernel_config_free(old.up);
            if (old.upv) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        }
        const fresh: Armed = .{ .key = key, .nd = nd, .up = up, .upv = upv };
        armed[armed_next] = fresh;
        armed_next = (armed_next + 1) % armed.len;
        break :blk fresh;
    };
    var nd_out: [3]mlx.mlx_array = undefined;
    const wo = if (pend) |p| p.out else nw;
    const wi = if (pend) |p| p.inj else nw;
    try run(nd_kernel.?, cfgs.nd, &.{ x, nw, if (inj == 1) iw else nw, eps, wo, wi, dw, ds, db }, s, &nd_out);
    defer _ = mlx.mlx_array_free(nd_out[0]);
    defer _ = mlx.mlx_array_free(nd_out[1]);
    errdefer _ = mlx.mlx_array_free(nd_out[2]);
    var upmix: [2]mlx.mlx_array = undefined;
    if (cfgs.upv != null and aligned16(uw)) {
        try run(up_v_kernel.?, cfgs.upv.?, &.{ nd_out[0], nd_out[1], uw, us, ub }, s, &upmix);
        served_vec += 1;
    } else {
        try run(up_kernel.?, cfgs.up, &.{ nd_out[0], nd_out[1], uw, us, ub }, s, &upmix);
    }
    var stream = nd_out[2];
    if (wr == 0) {
        _ = mlx.mlx_array_free(stream);
        stream = .{ .ctx = null };
    }
    var inj_out = upmix[1];
    if (inj == 0) {
        _ = mlx.mlx_array_free(inj_out);
        inj_out = .{ .ctx = null };
    }
    if (!engaged) {
        engaged = true;
        log.info("[qwen4] two-launch hyper-connection read engaged: hc={d} hidden={d} lowrank={d} {d}-bit g{d} (MLX_SERVE_HC_DECODE2=0 restores the unfused chain)\n", .{ hc, hidden, R, bits, group_size });
    }
    served += 1;
    return .{ .mixed = upmix[0], .inj = inj_out, .stream = stream };
}

const testing = std.testing;

fn randBf16(rnd: std.Random, shape: []const c_int, scale: f32, offset: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(buf);
    for (buf) |*v| v.* = (rnd.float(f32) - 0.5) * scale + offset;
    const f = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

const Q = struct {
    w: mlx.mlx_array,
    s: mlx.mlx_array,
    b: mlx.mlx_array,
    fn deinit(q: Q) void {
        _ = mlx.mlx_array_free(q.w);
        _ = mlx.mlx_array_free(q.s);
        _ = mlx.mlx_array_free(q.b);
    }
};

fn quantRandom(rnd: std.Random, rows: c_int, cols: c_int, bits: u32, gs: u32, s: mlx.mlx_stream) !Q {
    const w = try randBf16(rnd, &.{ rows, cols }, 0.2, 0.0, s);
    defer _ = mlx.mlx_array_free(w);
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, w, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
    var q: Q = .{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new(), .b = mlx.mlx_array_new() };
    try mlx.check(mlx.mlx_vector_array_get(&q.w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q.s, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&q.b, triple, 2));
    return q;
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

/// Per element within a couple of bf16 ulps (the f32 accumulation order is the only difference), on every output.
fn expectClose(ref: mlx.mlx_array, got: mlx.mlx_array, s: mlx.mlx_stream) !void {
    const rh = try readF32(ref, s);
    defer testing.allocator.free(rh);
    const gh = try readF32(got, s);
    defer testing.allocator.free(gh);
    try testing.expectEqual(rh.len, gh.len);
    var dot: f64 = 0;
    var nr: f64 = 0;
    var ng: f64 = 0;
    for (rh, gh) |r, g| {
        try testing.expect(std.math.isFinite(g));
        dot += @as(f64, r) * g;
        nr += @as(f64, r) * r;
        ng += @as(f64, g) * g;
        try testing.expect(@abs(r - g) <= 0.02 * @abs(r) + 0.008);
    }
    try testing.expect(dot / @sqrt(nr * ng) >= 0.9999);
}

/// Row `row` of `a`, whose dim 1 holds the rows (rank 3: [1,rows,n], rank 4: [1,rows,n,1]).
fn sliceRow(a: mlx.mlx_array, row: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(a);
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = @splat(1);
    @memcpy(stop[0..shape.len], shape);
    start[1] = row;
    stop[1] = row + 1;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, a, &start, shape.len, &stop, shape.len, &[_]c_int{ 1, 1, 1, 1 }, shape.len, s));
    return out;
}

/// A `rows`-row read agrees, row by row, with each row read alone through the one-row launches.
/// A row read among others is byte-identical to the same row read alone: a grouped verify must not change a row's output.
fn expectEqualBits(ref: mlx.mlx_array, got: mlx.mlx_array, s: mlx.mlx_stream) !void {
    const rh = try readF32(ref, s);
    defer testing.allocator.free(rh);
    const gh = try readF32(got, s);
    defer testing.allocator.free(gh);
    try testing.expectEqualSlices(f32, rh, gh);
}

fn parityRows(h: c_int, r: c_int, bits: u32, gs: u32, seed: u64, rows: c_int) !void {
    const xfm = @import("transformer.zig");
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    xfm.hc_fused_override = true;
    defer xfm.hc_fused_override = null;
    rows_override = rows;
    defer rows_override = null;
    const hc: c_int = 4;
    const k = hc * h;
    const down = try quantRandom(rnd, r, k, bits, gs, s);
    defer down.deinit();
    const up = try quantRandom(rnd, k, r, bits, gs, s);
    defer up.deinit();
    // The weight stays lazy on purpose: the launch choice must not depend on whether an earlier read evaluated it.
    const x = try randBf16(rnd, &.{ 1, rows, k }, 4.0, 0.0, s);
    defer _ = mlx.mlx_array_free(x);
    const nw = try randBf16(rnd, &.{ hc, h }, 1.0, 1.0, s);
    defer _ = mlx.mlx_array_free(nw);
    const iw = try randBf16(rnd, &.{ k, hc }, 0.1, 0.0, s);
    defer _ = mlx.mlx_array_free(iw);
    const wo = try randBf16(rnd, &.{ 1, rows, h }, 2.0, 0.0, s);
    defer _ = mlx.mlx_array_free(wo);
    const wi = try randBf16(rnd, &.{ 1, rows, hc, 1 }, 1.0, 1.0, s);
    defer _ = mlx.mlx_array_free(wi);

    const cases = [_]struct { inject: bool, pending: bool }{ .{ .inject = true, .pending = false }, .{ .inject = true, .pending = true }, .{ .inject = false, .pending = false } };
    for (cases) |c| {
        const inj_w: mlx.mlx_array = if (c.inject) iw else .{ .ctx = null };
        const pend: ?xfm.HcPending = if (c.pending) .{ .out = wo, .inj = wi } else null;
        const before = served;
        const before_vec = served_vec;
        const got = (try xfm.hcReadFused(s, x, 1, rows, nw, down.w, down.s, down.b, up.w, up.s, up.b, inj_w, 1e-6, hc, h, bits, gs, pend)) orelse return error.HcFusedDeclined;
        defer {
            _ = mlx.mlx_array_free(got.mixed);
            if (got.inj.ctx != null) _ = mlx.mlx_array_free(got.inj);
            if (got.stream.ctx != null) _ = mlx.mlx_array_free(got.stream);
        }
        try testing.expectEqual(before + 1, served);
        try testing.expectEqual(before_vec + @intFromBool(upVectorized(rows, 4, r, bits, @intCast(gs))), served_vec);
        try testing.expect((got.inj.ctx != null) == c.inject);
        try testing.expect((got.stream.ctx != null) == c.pending);
        var row: c_int = 0;
        while (row < rows) : (row += 1) {
            const xr = try sliceRow(x, row, s);
            defer _ = mlx.mlx_array_free(xr);
            const wor = try sliceRow(wo, row, s);
            defer _ = mlx.mlx_array_free(wor);
            const wir = try sliceRow(wi, row, s);
            defer _ = mlx.mlx_array_free(wir);
            const one_pend: ?xfm.HcPending = if (c.pending) .{ .out = wor, .inj = wir } else null;
            const one = (try xfm.hcReadFused(s, xr, 1, 1, nw, down.w, down.s, down.b, up.w, up.s, up.b, inj_w, 1e-6, hc, h, bits, gs, one_pend)) orelse return error.HcFusedDeclined;
            defer {
                _ = mlx.mlx_array_free(one.mixed);
                if (one.inj.ctx != null) _ = mlx.mlx_array_free(one.inj);
                if (one.stream.ctx != null) _ = mlx.mlx_array_free(one.stream);
            }
            inline for (.{ "mixed", "inj", "stream" }) |name| {
                if (@field(got, name).ctx != null) {
                    const part = try sliceRow(@field(got, name), row, s);
                    defer _ = mlx.mlx_array_free(part);
                    try expectEqualBits(@field(one, name), part, s);
                }
            }
        }
    }
}

test "hc decode2: a multi-row read matches each row read alone (Flash Next geometry, 8-bit, vectorized up launch)" {
    for ([_]c_int{ 2, 6, 9, 16 }) |rows| try parityRows(2560, 320, 8, 64, 0x2D0DE9 + @as(u64, @intCast(rows)), rows);
}

test "hc decode2: a multi-row read matches each row read alone (4/5/6-bit, per-row up launch)" {
    try parityRows(512, 64, 4, 64, 0x2D0DE4, 3);
    for ([_]u32{ 5, 6 }) |bits| try parityRows(2560, 320, bits, 64, 0x2D0DE5 + bits, 5);
}
