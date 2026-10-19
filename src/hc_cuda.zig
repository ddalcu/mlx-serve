//! `hc_decode2`'s two-launch hyper-connection read on MLX's CUDA backend: the stream norm (and a
//! pending hcWrite) folds into the down launch, the stream partials meet in the up/mix launch.
//! Rounding sites are the composed chain's (stream', xn, silu, up, gate, mix); only the f32
//! accumulation order differs. Dense bf16 and 2/4/8-bit affine weights.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const hc_decode2 = @import("hc_decode2.zig");

/// Down rows a block owns, one per warp.
const ROWS_TG: c_int = 8;
/// Up columns a block owns, one per warp.
const COLS_TG: c_int = 16;

const HEADER =
    \\template <bool DENSE>
    \\__device__ __forceinline__ float hc_val(unsigned q) {
    \\  if constexpr (DENSE) return __bfloat162float(__ushort_as_bfloat16((unsigned short)q));
    \\  else return (float)q;
    \\}
    \\__device__ __forceinline__ float hc_warp_sum(float v) {
    \\  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    \\  return v;
    \\}
    \\__device__ __forceinline__ float hc_sig(float v) { return 1.0f / (1.0f + expf(-v)); }
;

const ND_SOURCE =
    \\const int t = threadIdx.x;
    \\const int lane = t & 31;
    \\const int sg = t >> 5;
    \\const int s = blockIdx.y % HC;
    \\const int g = blockIdx.y / HC;
    \\constexpr int PER = H / 256;
    \\constexpr int K = HC * H;
    \\constexpr bool DENSE = BITS == 16;
    \\constexpr int VPW = 32 / BITS;
    \\constexpr int WPL = 16 / VPW;
    \\constexpr int KW = K / VPW;
    \\constexpr int KG = K / GS;
    \\constexpr int NSLOT = H / 16;
    \\constexpr int SITER = (NSLOT + 31) / 32;
    \\constexpr int PR = R + HC;
    \\constexpr int NDG = R / 8;
    \\constexpr unsigned MASK = (1u << BITS) - 1u;
    \\__shared__ __align__(16) T xs[H];
    \\__shared__ float npart[8];
    \\__shared__ float tgi[8 * HC];
    \\const int base = s * H;
    \\const bool down = g < NDG;
    \\// The down weights do not wait for the norm: every load of this warp's row goes out first, once
    \\// for all NR rows.
    \\unsigned pw[SITER][WPL];
    \\float ps[SITER];
    \\float pb[SITER];
    \\const int n0 = g * 8 + sg;
    \\if (down) {
    \\  for (int i = 0; i < SITER; ++i) {
    \\    const int slot = i * 32 + lane;
    \\    if (slot < NSLOT) {
    \\      const uint32_t* wp = dw_q + (size_t)n0 * KW + (size_t)(s * (H / VPW) + slot * WPL);
    \\      for (int e = 0; e < WPL; ++e) pw[i][e] = wp[e];
    \\      const size_t gi = (size_t)n0 * KG + (size_t)((base + slot * 16) / GS);
    \\      ps[i] = DENSE ? 1.0f : static_cast<float>(dw_s[DENSE ? 0 : gi]);
    \\      pb[i] = DENSE ? 0.0f : static_cast<float>(dw_b[DENSE ? 0 : gi]);
    \\    }
    \\  }
    \\}
    \\float wv[PER];
    \\for (int i = 0; i < PER; ++i) wv[i] = static_cast<float>(nw[base + t + 256 * i]);
    \\// The row count is read off the input, not templated: one compiled kernel serves every width.
    \\int NR = 1;
    \\for (int d = 0; d < x_in_ndim; ++d) NR *= x_in_shape[d];
    \\NR /= K;
    \\for (int row = 0; row < NR; ++row) {
    \\const size_t xrow = (size_t)row * K;
    \\float xv[PER];
    \\T written[PER];
    \\if (WR) {
    \\  // Pending hcWrite: stream' = T(stream + T(out * inj)), the chain's two roundings.
    \\  const float gate = static_cast<float>(wi_in[row * HC + s]);
    \\  for (int i = 0; i < PER; ++i) {
    \\    const int k = t + 256 * i;
    \\    const T prod = T(static_cast<float>(wo_in[(size_t)row * H + k]) * gate);
    \\    const T v = T(static_cast<float>(x_in[xrow + base + k]) + static_cast<float>(prod));
    \\    written[i] = v;
    \\    xv[i] = static_cast<float>(v);
    \\  }
    \\} else {
    \\  for (int i = 0; i < PER; ++i) xv[i] = static_cast<float>(x_in[xrow + base + t + 256 * i]);
    \\}
    \\float a = 0.0f;
    \\for (int i = 0; i < PER; ++i) a += xv[i] * xv[i];
    \\a = hc_warp_sum(a);
    \\if (lane == 0) npart[sg] = a;
    \\__syncthreads();
    \\float tot = 0.0f;
    \\for (int q = 0; q < 8; ++q) tot += npart[q];
    \\const float rsh = rsqrtf(tot / float(H) + eps[0]);
    \\T xnv[PER];
    \\for (int i = 0; i < PER; ++i) {
    \\  xnv[i] = T(static_cast<float>(T(xv[i] * rsh)) * wv[i]);
    \\  xs[t + 256 * i] = xnv[i];
    \\}
    \\__syncthreads();
    \\float* pp = parts_out + (size_t)(row * HC + s) * PR;
    \\if (down) {
    \\  float acc = 0.0f;
    \\  for (int i = 0; i < SITER; ++i) {
    \\    const int slot = i * 32 + lane;
    \\    if (slot < NSLOT) {
    \\      float xr[16];
    \\      float xsum = 0.0f;
    \\      // Two 16-byte loads: one bf16 at a time, lanes 32 bytes apart conflict 8 ways.
    \\      uint4 xq[2] = {reinterpret_cast<const uint4*>(xs + slot * 16)[0], reinterpret_cast<const uint4*>(xs + slot * 16)[1]};
    \\      const T* xe = reinterpret_cast<const T*>(xq);
    \\      for (int e = 0; e < 16; ++e) { xr[e] = static_cast<float>(xe[e]); xsum += xr[e]; }
    \\      float d = 0.0f;
    \\      for (int w = 0; w < WPL; ++w) {
    \\        for (int j = 0; j < VPW; ++j) d += xr[w * VPW + j] * hc_val<DENSE>((pw[i][w] >> (j * BITS)) & MASK);
    \\      }
    \\      acc += ps[i] * d + pb[i] * xsum;
    \\    }
    \\  }
    \\  const float v = hc_warp_sum(acc);
    \\  if (lane == 0) pp[n0] = v;
    \\} else if (INJ != 0) {
    \\  // This stream's partial of the inject matvec; the up launch adds the streams.
    \\  float ip[HC];
    \\  for (int c = 0; c < HC; ++c) ip[c] = 0.0f;
    \\  for (int i = 0; i < PER; ++i) {
    \\    const size_t k = (size_t)(base + t + 256 * i);
    \\    for (int c = 0; c < HC; ++c) ip[c] += static_cast<float>(xnv[i]) * static_cast<float>(iw[k * HC + c]);
    \\  }
    \\  for (int c = 0; c < HC; ++c) {
    \\    const float pa = hc_warp_sum(ip[c]);
    \\    if (lane == 0) tgi[sg * HC + c] = pa;
    \\  }
    \\  __syncthreads();
    \\  if (t < HC) {
    \\    float tt = 0.0f;
    \\    for (int q = 0; q < 8; ++q) tt += tgi[q * HC + t];
    \\    pp[R + t] = tt;
    \\  }
    \\}
    \\if (g == 0) {
    \\  for (int i = 0; i < PER; ++i) {
    \\    xn_out[xrow + base + t + 256 * i] = xnv[i];
    \\    if (WR) xs_out[xrow + base + t + 256 * i] = written[i];
    \\  }
    \\}
    \\__syncthreads();
    \\}
;

const UP_SOURCE =
    \\const int t = threadIdx.x;
    \\const int lane = t & 31;
    \\const int sg = t >> 5;
    \\const int j = blockIdx.y * COLS + sg;
    \\constexpr bool DENSE = BITS == 16;
    \\constexpr int VPW = 32 / BITS;
    \\constexpr int R_by_p = R / VPW;
    \\constexpr int R_by_gs = R / GS;
    \\constexpr int RIT = (R_by_p + 31) / 32;
    \\constexpr int PR = R + HC;
    \\constexpr unsigned MASK = (1u << BITS) - 1u;
    \\__shared__ T acts[R];
    \\// This column's up rows (one per stream) do not wait for the activation, loaded once for all NR rows.
    \\unsigned pw[HC][RIT];
    \\float ps[HC][RIT];
    \\float pb[HC][RIT];
    \\for (int h = 0; h < HC; ++h) {
    \\  const size_t hrow = (size_t)h * H + (size_t)j;
    \\  for (int i = 0; i < RIT; ++i) {
    \\    const int pack = lane + 32 * i;
    \\    if (pack < R_by_p) {
    \\      pw[h][i] = uw_q[hrow * R_by_p + (size_t)pack];
    \\      const size_t gi = hrow * R_by_gs + (size_t)((pack * VPW) / GS);
    \\      ps[h][i] = DENSE ? 1.0f : static_cast<float>(uw_s[DENSE ? 0 : gi]);
    \\      pb[h][i] = DENSE ? 0.0f : static_cast<float>(uw_b[DENSE ? 0 : gi]);
    \\    }
    \\  }
    \\}
    \\const int NR = xn_in_shape[0] / (HC * H);
    \\for (int row = 0; row < NR; ++row) {
    \\float xnv[HC];
    \\for (int h = 0; h < HC; ++h) xnv[h] = static_cast<float>(xn_in[(size_t)row * HC * H + (size_t)h * H + (size_t)j]);
    \\// act = silu(T(sum of the stream partials)), as the composed down projection ends.
    \\for (int i = t; i < R; i += COLS * 32) {
    \\  float tsum = 0.0f;
    \\  for (int s = 0; s < HC; ++s) tsum += parts_in[(row * HC + s) * PR + i];
    \\  const T v = T(tsum);
    \\  const T sig = T(hc_sig(static_cast<float>(v)));
    \\  acts[i] = T(static_cast<float>(v) * static_cast<float>(sig));
    \\}
    \\if (INJ != 0 && blockIdx.y == 0 && t < HC) {
    \\  float tsum = 0.0f;
    \\  for (int s = 0; s < HC; ++s) tsum += parts_in[(row * HC + s) * PR + R + t];
    \\  const T v = T(tsum);
    \\  const T sig = T(hc_sig(static_cast<float>(v)));
    \\  inj_out[row * HC + t] = T(static_cast<float>(sig) * 2.0f);
    \\}
    \\__syncthreads();
    \\float sum = 0.0f;
    \\for (int h = 0; h < HC; ++h) {
    \\  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    \\  for (int i = 0; i < RIT; ++i) {
    \\    const int pack = lane + 32 * i;
    \\    if (pack >= R_by_p) continue;
    \\    const int k = pack * VPW;
    \\    if constexpr (DENSE) {
    \\      a0 += static_cast<float>(acts[k]) * hc_val<true>(pw[h][i] & 0xFFFFu);
    \\      a1 += static_cast<float>(acts[k + 1]) * hc_val<true>(pw[h][i] >> 16);
    \\    } else {
    \\      for (int ki = 0; ki < VPW; ki += 4) {
    \\        const unsigned q = pw[h][i] >> (ki * BITS);
    \\        a0 += static_cast<float>(acts[k + ki + 0]) * (float((q >> (0 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
    \\        a1 += static_cast<float>(acts[k + ki + 1]) * (float((q >> (1 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
    \\        a2 += static_cast<float>(acts[k + ki + 2]) * (float((q >> (2 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
    \\        a3 += static_cast<float>(acts[k + ki + 3]) * (float((q >> (3 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
    \\      }
    \\    }
    \\  }
    \\  const float acc = hc_warp_sum((a0 + a1) + (a2 + a3));
    \\  const T u = T(acc);
    \\  const T sgv = T(hc_sig(static_cast<float>(u)));
    \\  sum += static_cast<float>(T(static_cast<float>(sgv) * xnv[h]));
    \\}
    \\if (lane == 0) mixed_out[(size_t)row * H + j] = T(static_cast<float>(T(sum)) * static_cast<float>(T(1.0f / float(HC))));
    \\__syncthreads();
    \\}
;

var nd_kernel: ?mlx.mlx_fast_cuda_kernel = null;
var up_kernel: ?mlx.mlx_fast_cuda_kernel = null;
var engaged = false;
/// Reads served by the two launches (the tests assert the path engaged).
pub var served: u64 = 0;

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_cuda_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_cuda_kernel_new(name, in_vec, out_vec, source, HEADER, true, 0);
    if (k.ctx == null) return error.CudaKernelCreateFailed;
    return k;
}

fn launch(kernel: mlx.mlx_fast_cuda_kernel, outs: []const struct { c_int, mlx.mlx_dtype }, grid: [3]c_int, tg: c_int, tmpl: []const struct { [*:0]const u8, c_int }, dt: mlx.mlx_dtype, inputs: []const mlx.mlx_array, s: mlx.mlx_stream, res: []mlx.mlx_array) !void {
    const c = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(c);
    for (outs) |o| try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(c, &[_]c_int{o[0]}, 1, o[1]));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(c, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(c, tg, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(c, "T", dt));
    for (tmpl) |a| try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, a[0], a[1]));
    const v = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&o, kernel, v, c, s));
    if (mlx.mlx_vector_array_size(o) != res.len) return error.CudaKernelBadOutputCount;
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

/// `hc_decode2.read`'s contract on CUDA, or null outside its envelope (the caller keeps the
/// unfused chain).
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
    pend: ?hc_decode2.Pending,
) !?hc_decode2.Out {
    if (!mlx.cudaAvailable()) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    const dense = bits == 16;
    if (dense and dt != .bfloat16) return null;
    if (bits != 2 and bits != 4 and bits != 8 and !dense) return null;
    if (nw.ctx == null or mlx.mlx_array_dtype(nw) != dt) return null;
    if (!dense) inline for (.{ ds, db, us, ub }) |a| {
        if (a.ctx == null or mlx.mlx_array_dtype(a) != dt) return null;
    };
    if (iw.ctx != null and mlx.mlx_array_dtype(iw) != dt) return null;
    const vpw: c_int = @intCast(32 / bits);
    const b: c_int = @intCast(bits);
    const gs: c_int = @intCast(group_size);
    const K = hc * hidden;
    if (hc < 1 or hc > 8 or @rem(hidden, 256) != 0 or @rem(gs, 16) != 0 or @rem(hidden, gs) != 0) return null;
    const dsh = mlx.getShape(dw);
    const ush = mlx.getShape(uw);
    if (dsh.len != 2 or ush.len != 2 or dsh[1] * 32 != K * b or ush[0] != K) return null;
    const R = dsh[0];
    if (ush[1] * 32 != R * b or @rem(R, vpw) != 0 or @rem(R, ROWS_TG) != 0 or @rem(R, gs) != 0 or @rem(hidden, COLS_TG) != 0) return null;
    const inj: c_int = @intFromBool(iw.ctx != null);
    const wr: c_int = @intFromBool(pend != null);
    if (rows < 1 or rows > 16 or mlx.mlx_array_size(x) != @as(usize, @intCast(rows * K)) or mlx.mlx_array_size(nw) != @as(usize, @intCast(K))) return null;
    if (pend) |p| {
        if (mlx.mlx_array_dtype(p.out) != dt or mlx.mlx_array_dtype(p.inj) != dt) return null;
    }

    if (nd_kernel == null) nd_kernel = try makeKernel("msv_hc_read_nd", &.{ "x_in", "nw", "iw", "eps", "wo_in", "wi_in", "dw_q", "dw_s", "dw_b" }, &.{ "xn_out", "parts_out", "xs_out" }, ND_SOURCE);
    if (up_kernel == null) up_kernel = try makeKernel("msv_hc_read_up", &.{ "xn_in", "parts_in", "uw_q", "uw_s", "uw_b" }, &.{ "mixed_out", "inj_out" }, UP_SOURCE);

    const pr = R + hc;
    const groups = @divExact(R, ROWS_TG) + inj;
    const tmpl = [_]struct { [*:0]const u8, c_int }{ .{ "HC", hc }, .{ "H", hidden }, .{ "R", R }, .{ "GS", gs }, .{ "BITS", b }, .{ "INJ", inj }, .{ "WR", wr }, .{ "COLS", COLS_TG } };
    const wo = if (pend) |p| p.out else nw;
    const wi = if (pend) |p| p.inj else nw;
    // Dense weights carry no scales; the kernel never reads the operands bound in their place.
    const dsb: [4]mlx.mlx_array = if (dense) .{ nw, nw, nw, nw } else .{ ds, db, us, ub };
    var nd_out: [3]mlx.mlx_array = undefined;
    try launch(nd_kernel.?, &.{ .{ rows * K, dt }, .{ rows * hc * pr, .float32 }, .{ if (wr == 1) rows * K else 1, dt } }, .{ 256, groups * hc, 1 }, 256, &tmpl, dt, &.{ x, nw, if (inj == 1) iw else nw, eps, wo, wi, dw, dsb[0], dsb[1] }, s, &nd_out);
    defer _ = mlx.mlx_array_free(nd_out[0]);
    defer _ = mlx.mlx_array_free(nd_out[1]);
    errdefer _ = mlx.mlx_array_free(nd_out[2]);
    var upmix: [2]mlx.mlx_array = undefined;
    try launch(up_kernel.?, &.{ .{ rows * hidden, dt }, .{ rows * hc, dt } }, .{ 32 * COLS_TG, @divExact(hidden, COLS_TG), 1 }, 32 * COLS_TG, &tmpl, dt, &.{ nd_out[0], nd_out[1], uw, dsb[2], dsb[3] }, s, &upmix);
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
        log.info("[qwen4] CUDA two-launch hyper-connection read engaged: hc={d} hidden={d} lowrank={d} {d}-bit (MLX_SERVE_HC_FUSED=0 restores the unfused chain)\n", .{ hc, hidden, R, bits });
    }
    served += 1;
    return .{ .mixed = upmix[0], .inj = inj_out, .stream = stream };
}

const testing = std.testing;

fn randArr(rnd: std.Random, shape: []const c_int, scale: f32, offset: f32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(buf);
    for (buf) |*v| v.* = (rnd.float(f32) - 0.5) * scale + offset;
    const f = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, dt, s));
    return out;
}

/// The arrays of one test case; `free` releases every one.
const Bag = struct {
    items: [256]mlx.mlx_array = undefined,
    n: usize = 0,
    fn keep(b: *Bag, a: mlx.mlx_array) mlx.mlx_array {
        b.items[b.n] = a;
        b.n += 1;
        return a;
    }
    fn free(b: *Bag) void {
        for (b.items[0..b.n]) |a| _ = mlx.mlx_array_free(a);
    }
};

/// The composed hyper-connection read in dtype `dt`: stream' = x + out*inj, n = rms_norm(stream')*nw,
/// mixed = mean_s(sigmoid(up(silu(down(n)))) * n), inj = 2*sigmoid(n @ iw). `down` [R,K], `up` [K,R].
fn chain(b: *Bag, s: mlx.mlx_stream, dt: mlx.mlx_dtype, x: mlx.mlx_array, nw: mlx.mlx_array, down: mlx.mlx_array, up: mlx.mlx_array, iw: mlx.mlx_array, wo: ?mlx.mlx_array, wi: ?mlx.mlx_array, hc: c_int, h: c_int, eps: f32) ![3]mlx.mlx_array {
    const O = struct {
        fn cast(bb: *Bag, st: mlx.mlx_stream, a: mlx.mlx_array, d: mlx.mlx_dtype) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&o, a, d, st));
            return bb.keep(o);
        }
        fn bin(bb: *Bag, st: mlx.mlx_stream, comptime f: anytype, a: mlx.mlx_array, c: mlx.mlx_array) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            try mlx.check(f(&o, a, c, st));
            return bb.keep(o);
        }
        fn shape(bb: *Bag, st: mlx.mlx_stream, a: mlx.mlx_array, sh: []const c_int) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&o, a, sh.ptr, sh.len, st));
            return bb.keep(o);
        }
        fn sig(bb: *Bag, st: mlx.mlx_stream, a: mlx.mlx_array) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_sigmoid(&o, a, st));
            return bb.keep(o);
        }
        fn t(bb: *Bag, st: mlx.mlx_stream, a: mlx.mlx_array) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_transpose(&o, a, st));
            return bb.keep(o);
        }
    };
    const rows = @divExact(@as(c_int, @intCast(mlx.mlx_array_size(x))), hc * h);
    var stream = try O.cast(b, s, x, dt);
    if (wo) |o| {
        const o4 = try O.shape(b, s, try O.cast(b, s, o, dt), &.{ 1, rows, 1, h });
        const prod = try O.bin(b, s, mlx.mlx_multiply, o4, try O.cast(b, s, wi.?, dt));
        stream = try O.bin(b, s, mlx.mlx_add, try O.shape(b, s, stream, &.{ 1, rows, hc, h }), prod);
    }
    const x4 = try O.shape(b, s, stream, &.{ 1, rows, hc, h });
    var prng = std.Random.DefaultPrng.init(0);
    const ones = try O.cast(b, s, b.keep(try randArr(prng.random(), &.{h}, 0.0, 1.0, .float32, s)), dt);
    var nr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&nr, x4, ones, eps, s));
    const n4 = try O.bin(b, s, mlx.mlx_multiply, b.keep(nr), try O.cast(b, s, nw, dt));
    const nf = try O.shape(b, s, n4, &.{ 1, rows, hc * h });
    const d = try O.bin(b, s, mlx.mlx_matmul, nf, try O.t(b, s, try O.cast(b, s, down, dt)));
    const act = try O.bin(b, s, mlx.mlx_multiply, d, try O.sig(b, s, d));
    const u = try O.bin(b, s, mlx.mlx_matmul, act, try O.t(b, s, try O.cast(b, s, up, dt)));
    const prod = try O.bin(b, s, mlx.mlx_multiply, try O.sig(b, s, try O.shape(b, s, u, &.{ 1, rows, hc, h })), n4);
    var mixed = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_mean_axis(&mixed, prod, 2, false, s));
    const isg = try O.sig(b, s, try O.bin(b, s, mlx.mlx_matmul, nf, try O.cast(b, s, iw, dt)));
    var two = mlx.mlx_array_new_float(2.0);
    two = try O.cast(b, s, b.keep(two), dt);
    const inj = try O.bin(b, s, mlx.mlx_multiply, isg, two);
    return .{ b.keep(mixed), inj, try O.shape(b, s, stream, &.{ 1, rows, hc * h }) };
}

fn rmsDiff(b: *Bag, s: mlx.mlx_stream, a: mlx.mlx_array, ref: mlx.mlx_array) !f32 {
    var af = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&af, a, .float32, s));
    var rf = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&rf, ref, .float32, s));
    var rr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&rr, rf, &[_]c_int{-1}, 1, s));
    var ar = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&ar, af, &[_]c_int{-1}, 1, s));
    var d = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&d, ar, rr, s));
    var sq = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_square(&sq, d, s));
    var m = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_mean(&m, sq, false, s));
    inline for (.{ af, rf, rr, ar, d, sq }) |x| _ = b.keep(x);
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_eval(b.keep(m)));
    try mlx.check(mlx.mlx_array_item_float32(&v, m));
    return @sqrt(v);
}

/// Every element of `got` within two bf16 ulps of an O(1) operand of `ref` (the Metal read's bar): the
/// same rounding sites, only the f32 summation order differs, and the mean over streams cancels.
fn expectClose(b: *Bag, s: mlx.mlx_stream, got: mlx.mlx_array, ref: mlx.mlx_array) !void {
    const n = mlx.mlx_array_size(ref);
    try testing.expectEqual(n, mlx.mlx_array_size(got));
    var vals: [2][]const f32 = undefined;
    for ([_]mlx.mlx_array{ got, ref }, 0..) |a, i| {
        var f = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&f, a, .float32, s));
        try mlx.check(mlx.mlx_array_eval(b.keep(f)));
        vals[i] = (mlx.mlx_array_data_float32(f) orelse return error.Unreadable)[0..n];
    }
    for (vals[0], vals[1]) |g, r| {
        try testing.expect(std.math.isFinite(g));
        try testing.expect(@abs(g - r) <= 0.02 * @abs(r) + 0.008);
    }
}

test "hc_cuda: the fused read is no worse than the composed chain against f32 truth (dense bf16 and 8-bit, inject, pending write)" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const xfm = @import("transformer.zig");
    const s = mlx.gpuStream();
    const hc: c_int = 4;
    const h: c_int = 2560;
    const r: c_int = 320;
    const k = hc * h;
    const eps: f32 = 1e-6;
    for ([_]u32{ 16, 8 }) |bits| {
        for ([_]bool{ false, true }) |pending| {
            var b: Bag = .{};
            defer b.free();
            var prng = std.Random.DefaultPrng.init(0xC0DA + bits + @intFromBool(pending));
            const rnd = prng.random();
            const x = b.keep(try randArr(rnd, &.{ 1, 1, k }, 4.0, 0.0, .bfloat16, s));
            const nw = b.keep(try randArr(rnd, &.{ hc, h }, 1.0, 1.0, .bfloat16, s));
            const iw = b.keep(try randArr(rnd, &.{ k, hc }, 0.02, 0.0, .bfloat16, s));
            const wo = b.keep(try randArr(rnd, &.{ 1, 1, h }, 2.0, 0.0, .bfloat16, s));
            const wi = b.keep(try randArr(rnd, &.{ 1, 1, hc, 1 }, 1.0, 1.0, .bfloat16, s));
            var down = b.keep(try randArr(rnd, &.{ r, k }, 0.04, 0.0, .bfloat16, s));
            var up = b.keep(try randArr(rnd, &.{ k, r }, 0.2, 0.0, .bfloat16, s));
            var dq: [3]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null }, .{ .ctx = null } };
            var uq: [3]mlx.mlx_array = dq;
            if (bits == 16) {
                for ([_]*mlx.mlx_array{ &dq[0], &uq[0] }, [_]mlx.mlx_array{ down, up }) |dst, src| {
                    dst.* = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_view(dst, src, .uint32, s));
                    _ = b.keep(dst.*);
                }
            } else {
                // The chain and the truth read the dequantized weights; the fused read the packed ones.
                for ([_]*[3]mlx.mlx_array{ &dq, &uq }, [_]*mlx.mlx_array{ &down, &up }) |dst, src| {
                    var tri = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(tri);
                    try mlx.check(mlx.mlx_quantize(&tri, src.*, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, s));
                    for (dst, 0..) |*a, i| {
                        a.* = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_vector_array_get(a, tri, i));
                        _ = b.keep(a.*);
                    }
                    var deq = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_dequantize(&deq, dst[0], dst[1], dst[2], mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, .{ .value = .bfloat16, .has_value = true }, s));
                    src.* = b.keep(deq);
                }
            }
            const pend: ?xfm.HcPending = if (pending) .{ .out = wo, .inj = wi } else null;
            const before = served;
            const got = (try xfm.hcReadFused(s, x, 1, 1, nw, dq[0], dq[1], dq[2], uq[0], uq[1], uq[2], iw, eps, hc, h, bits, if (bits == 16) 16 else 64, pend)) orelse return error.HcFusedDeclined;
            _ = b.keep(got.mixed);
            _ = b.keep(got.inj);
            if (got.stream.ctx != null) _ = b.keep(got.stream);
            try testing.expectEqual(before + 1, served);
            try testing.expect((got.stream.ctx != null) == pending);
            const truth = try chain(&b, s, .float32, x, nw, down, up, iw, if (pending) wo else null, if (pending) wi else null, hc, h, eps);
            const comp = try chain(&b, s, .bfloat16, x, nw, down, up, iw, if (pending) wo else null, if (pending) wi else null, hc, h, eps);
            const names = [_][]const u8{ "mixed", "inj", "stream" };
            const fused = [_]mlx.mlx_array{ got.mixed, got.inj, got.stream };
            for (0..3) |i| {
                if (fused[i].ctx == null) continue;
                try expectClose(&b, s, fused[i], comp[i]);
                if (mlx.mlx_array_size(fused[i]) < 1024) continue;
                const ef = try rmsDiff(&b, s, fused[i], truth[i]);
                const ec = try rmsDiff(&b, s, comp[i], truth[i]);
                std.debug.print("hc_cuda bits={d} pending={} {s}: fused rms err {e:.3} chain {e:.3}\n", .{ bits, pending, names[i], ef, ec });
                try testing.expect(ef <= ec * 1.15 + 1e-7);
            }
        }
    }
}

test "hc_cuda: a multi-row read matches each row read alone, bit for bit" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const xfm = @import("transformer.zig");
    const s = mlx.gpuStream();
    const hc: c_int = 4;
    const h: c_int = 2560;
    const r: c_int = 320;
    const k = hc * h;
    const rows: c_int = 3;
    var b: Bag = .{};
    defer b.free();
    var prng = std.Random.DefaultPrng.init(0xC0DB);
    const rnd = prng.random();
    const x = b.keep(try randArr(rnd, &.{ 1, rows, k }, 4.0, 0.0, .bfloat16, s));
    const nw = b.keep(try randArr(rnd, &.{ hc, h }, 1.0, 1.0, .bfloat16, s));
    const iw = b.keep(try randArr(rnd, &.{ k, hc }, 0.02, 0.0, .bfloat16, s));
    const wo = b.keep(try randArr(rnd, &.{ 1, rows, h }, 2.0, 0.0, .bfloat16, s));
    const wi = b.keep(try randArr(rnd, &.{ 1, rows, hc, 1 }, 1.0, 1.0, .bfloat16, s));
    var dw = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_view(&dw, b.keep(try randArr(rnd, &.{ r, k }, 0.04, 0.0, .bfloat16, s)), .uint32, s));
    var uw = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_view(&uw, b.keep(try randArr(rnd, &.{ k, r }, 0.2, 0.0, .bfloat16, s)), .uint32, s));
    _ = b.keep(dw);
    _ = b.keep(uw);
    const none: mlx.mlx_array = .{ .ctx = null };
    const all = (try xfm.hcReadFused(s, x, 1, rows, nw, dw, none, none, uw, none, none, iw, 1e-6, hc, h, 16, 16, .{ .out = wo, .inj = wi })) orelse return error.HcFusedDeclined;
    _ = b.keep(all.mixed);
    _ = b.keep(all.inj);
    _ = b.keep(all.stream);
    var row: c_int = 0;
    while (row < rows) : (row += 1) {
        const Slice = struct {
            fn at(bb: *Bag, st: mlx.mlx_stream, a: mlx.mlx_array, rw: c_int) !mlx.mlx_array {
                const sh = mlx.getShape(a);
                var start: [4]c_int = @splat(0);
                var stop: [4]c_int = @splat(1);
                @memcpy(stop[0..sh.len], sh);
                start[1] = rw;
                stop[1] = rw + 1;
                var o = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(&o, a, &start, sh.len, &stop, sh.len, &[_]c_int{ 1, 1, 1, 1 }, sh.len, st));
                return bb.keep(o);
            }
        };
        const one = (try xfm.hcReadFused(s, try Slice.at(&b, s, x, row), 1, 1, nw, dw, none, none, uw, none, none, iw, 1e-6, hc, h, 16, 16, .{ .out = try Slice.at(&b, s, wo, row), .inj = try Slice.at(&b, s, wi, row) })) orelse return error.HcFusedDeclined;
        _ = b.keep(one.mixed);
        _ = b.keep(one.inj);
        _ = b.keep(one.stream);
        const pairs = [_][2]mlx.mlx_array{ .{ all.mixed, one.mixed }, .{ all.inj, one.inj }, .{ all.stream, one.stream } };
        for (pairs) |p| {
            var eq = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_equal(&eq, try Slice.at(&b, s, p[0], row), p[1], false, s));
            _ = b.keep(eq);
            var same = false;
            try mlx.check(mlx.mlx_array_eval(eq));
            try mlx.check(mlx.mlx_array_item_bool(&same, eq));
            try testing.expect(same);
        }
    }
}
