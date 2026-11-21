// dec_ffn_in's fused tail (`MLX_SERVE_DSV41_FUSE_FFN_IN`), after launch 1's body: launch 2's work in the same dispatch.
// The last of a 128-column slice's threadgroups to finish takes the slice: its 24 hc mix partials and sum of squares
// over x1 (stored as launch 2's slices store them), its sum of squares of hv, and the router's partial logits over its
// columns, gate . (norm_w * hv) without the norm's 1 / rms (atomic adds). The last slice to finish then takes what
// launch 2's last threadgroups take: the Sinkhorn split from the slices' partials, h2 = rms-norm(hv), and each row's
// scores (the logits times the rows' 1 / rms) with their top TOPK. The router reads the unrounded normed collapse, where
// launch 2 reads h2 rounded to bf16: the same math in another rounding. `facc` is persistent scratch, zero between
// dispatches: the logits [6 E], the hv sums [8], the mix partials [6 P PL] (floats), then the counters (P + 1 words,
// from word FCTR); every slot is set back to zero by the dispatch that used it. Values cross threadgroups as relaxed
// atomics: plain loads can hit another core's stale L1.
{
  constexpr int CT = NSG * NR;
  constexpr int TPS = 128 / CT;
  constexpr int P = D / 128;
  constexpr int MIX = (2 + HC) * HC;
  constexpr int PL = MIX + 1;
  constexpr int SS0 = 6 * E;
  constexpr int PARTS0 = 6 * E + 8;
  constexpr int FCTR = PARTS0 + 6 * P * PL;
  constexpr int NTH = NSG * 32;
  constexpr int NE = (E + 31) / 32;
  threadgroup float t_hw[R * 128];
  threadgroup float t_zr[R * PL];
  threadgroup bool t_last;
  device metal::atomic_float* fa = (device metal::atomic_float*)facc;
  device metal::atomic_uint* fc = (device metal::atomic_uint*)facc + FCTR;
  const device metal::atomic_uint* x1w = (const device metal::atomic_uint*)x1;
  const device metal::atomic_uint* hvw = (const device metal::atomic_uint*)hv;
  const int slice = tg / TPS;
  const int c0 = slice * 128;

  metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
  threadgroup_barrier(metal::mem_flags::mem_device);
  if (tid == 0) {
    const uint old = metal::atomic_fetch_add_explicit(fc + slice, 1u, metal::memory_order_relaxed);
    t_last = old + 1u == uint(TPS);
    if (t_last) metal::atomic_store_explicit(fc + slice, 0u, metal::memory_order_relaxed);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (!t_last) return;
  metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);

  // The slice's hv, then each row's sum of squares of it, then hv times the norm weight.
  for (int i = tid; i < R * 128; i += NTH) {
    const int r = i / 128, c = i % 128;
    const uint wd = metal::atomic_load_explicit(hvw + ((size_t)r * D + c0 + c) / 2, metal::memory_order_relaxed);
    t_hw[i] = as_type<float>((c & 1) ? (wd & 0xffff0000u) : (wd << 16));
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (int(sg) < R) {
    const int r = int(sg);
    float s2 = 0.0f;
    for (int q = 0; q < 4; ++q) {
      const float v = t_hw[r * 128 + int(lane) * 4 + q];
      s2 = metal::fma(v, v, s2);
    }
    s2 = metal::simd_sum(s2);
    if (lane == 0) metal::atomic_fetch_add_explicit(fa + SS0 + r, s2, metal::memory_order_relaxed);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int i = tid; i < R * 128; i += NTH) {
    const ushort wb = ((const device ushort*)w)[c0 + i % 128];
    t_hw[i] *= as_type<float>(uint(wb) << 16);
  }

  // The mix partials: a simdgroup a row, a lane 16 of the slice's HC x 128 stream values (copy lane / 8, 16 columns).
  if (int(sg) < R) {
    const int r = int(sg);
    const int ci = int(lane) / 8, cc = (int(lane) % 8) * 16;
    float xv[16];
    for (int q = 0; q < 8; ++q) {
      const uint wd = metal::atomic_load_explicit(x1w + (((size_t)r * HC + ci) * D + c0 + cc) / 2 + q, metal::memory_order_relaxed);
      xv[2 * q] = as_type<float>(wd << 16);
      xv[2 * q + 1] = as_type<float>(wd & 0xffff0000u);
    }
    for (int m = 0; m < PL; ++m) {
      float d = 0.0f;
      if (m < MIX) {
        const device float* fr = fn + (size_t)m * (HC * D) + ci * D + c0 + cc;
        for (int q = 0; q < 16; ++q) d = metal::fma(xv[q], fr[q], d);
      } else {
        for (int q = 0; q < 16; ++q) d = metal::fma(xv[q], xv[q], d);
      }
      d = metal::simd_sum(d);
      if (lane == 0) metal::atomic_store_explicit(fa + PARTS0 + ((size_t)r * P + slice) * PL + m, d, metal::memory_order_relaxed);
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // The router's partial logits over the slice: a simdgroup an expert at a time, a lane 4 columns.
  for (int e = int(sg); e < E; e += NSG) {
    const uint2 gw = *((const device uint2*)(gate_w + (size_t)e * D + c0) + lane);
    const float g0 = as_type<float>(gw.x << 16), g1 = as_type<float>(gw.x & 0xffff0000u);
    const float g2 = as_type<float>(gw.y << 16), g3 = as_type<float>(gw.y & 0xffff0000u);
    for (int r = 0; r < R; ++r) {
      const threadgroup float* h = t_hw + r * 128 + int(lane) * 4;
      float d = g0 * h[0];
      d = metal::fma(g1, h[1], d);
      d = metal::fma(g2, h[2], d);
      d = metal::fma(g3, h[3], d);
      d = metal::simd_sum(d);
      if (lane == 0) metal::atomic_fetch_add_explicit(fa + r * E + e, d, metal::memory_order_relaxed);
    }
  }

  metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
  threadgroup_barrier(metal::mem_flags::mem_device);
  if (tid == 0) {
    const uint old = metal::atomic_fetch_add_explicit(fc + P, 1u, metal::memory_order_relaxed);
    t_last = old + 1u == uint(P);
    if (t_last) metal::atomic_store_explicit(fc + P, 0u, metal::memory_order_relaxed);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (!t_last) return;
  metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);

  // The mixes over the slices (lanes stride them, then a tree), each partial zeroed once read.
  for (int pair = int(sg); pair < R * PL; pair += NSG) {
    const int r = pair / PL, m = pair % PL;
    float v = 0.0f;
    for (int p = int(lane); p < P; p += 32) {
      device metal::atomic_float* at = fa + PARTS0 + ((size_t)r * P + p) * PL + m;
      v += metal::atomic_load_explicit(at, metal::memory_order_relaxed);
      metal::atomic_store_explicit(at, 0.0f, metal::memory_order_relaxed);
    }
    v = metal::simd_sum(v);
    if (lane == 0) t_zr[pair] = v;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  // The Sinkhorn split, a quad a row (launch 2's).
  if (sg == 0 && int(lane) < 4 * R) {
    const int r = int(lane) / 4, i = int(lane) % 4;
    const threadgroup float* zz = t_zr + r * PL;
    const float inv = metal::precise::rsqrt(zz[MIX] / float(HC * D) + eps[0]);
    const float e1 = eps[1];
    pre[r * HC + i] = 1.0f / (1.0f + metal::exp(-(zz[i] * inv * scale[i] + base[i]))) + e1;
    post[r * HC + i] = 2.0f / (1.0f + metal::exp(-(zz[HC + i] * inv * scale[HC + i] + base[HC + i])));
    float v[HC];
    float mx = -INFINITY;
    for (int j = 0; j < HC; ++j) {
      const int m = 2 * HC + i * HC + j;
      v[j] = zz[m] * inv * scale[m] + base[m];
      mx = metal::max(mx, v[j]);
    }
    float sm = 0.0f;
    for (int j = 0; j < HC; ++j) {
      v[j] = metal::exp(v[j] - mx);
      sm += v[j];
    }
    for (int j = 0; j < HC; ++j) v[j] = v[j] / sm + e1;
    #pragma clang loop unroll(disable)
    for (int t = 0; t < 2 * ITERS - 1; ++t) {
      if (t % 2 == 0) {
        for (int j = 0; j < HC; ++j) v[j] = v[j] / (metal::quad_sum(v[j]) + e1);
      } else {
        const float rs = v[0] + v[1] + v[2] + v[3];
        for (int j = 0; j < HC; ++j) v[j] = v[j] / (rs + e1);
      }
    }
    *(device float4*)(comb + (r * HC + i) * HC) = float4(v[0], v[1], v[2], v[3]);
  }

  // Each row's 1 / rms of hv (launch 2's), h2 = T(w * T(hv * inv)), then the scores and the top TOPK, a simdgroup a row.
  threadgroup float t_inv[8];
  if (tid < R) {
    device metal::atomic_float* at = fa + SS0 + tid;
    t_inv[tid] = metal::precise::rsqrt(metal::atomic_load_explicit(at, metal::memory_order_relaxed) / float(D) + eps[0]);
    metal::atomic_store_explicit(at, 0.0f, metal::memory_order_relaxed);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int i = tid; i < R * (D / 2); i += NTH) {
    const int r = i / (D / 2), q = i % (D / 2);
    const uint hw2 = metal::atomic_load_explicit(hvw + (size_t)r * (D / 2) + q, metal::memory_order_relaxed);
    const uint nw = ((const device uint*)w)[q];
    const float a = as_type<float>(hw2 << 16), b = as_type<float>(hw2 & 0xffff0000u);
    const float wa = as_type<float>(nw << 16), wb = as_type<float>(nw & 0xffff0000u);
    const T ha = T(wa * float(T(a * t_inv[r]))), hb = T(wb * float(T(b * t_inv[r])));
    ((device uint*)h2)[(size_t)r * (D / 2) + q] = (uint(as_type<ushort>(hb)) << 16) | uint(as_type<ushort>(ha));
  }
  if (int(sg) < R) {
    const int r = int(sg);
    float sv2[NE], bv2[NE];
    for (int t = 0; t < NE; ++t) {
      const int ex = int(lane) + 32 * t;
      sv2[t] = 0.0f;
      bv2[t] = -INFINITY;
      if (ex < E) {
        device metal::atomic_float* at = fa + r * E + ex;
        const float z = metal::atomic_load_explicit(at, metal::memory_order_relaxed) * t_inv[r];
        metal::atomic_store_explicit(at, 0.0f, metal::memory_order_relaxed);
        // The sqrt-softplus score of the logit; a NaN is the lowest pick (lowest index first) and weighs NaN.
        const float y = metal::exp(-metal::abs(z));
        const float lp = y < 1e-4f ? y * (1.0f - y * (0.5f - y / 3.0f)) : metal::log(1.0f + y);
        const float s = metal::sqrt(metal::max(z, 0.0f) + lp);
        const float b = s + bias[ex];
        sv2[t] = s;
        bv2[t] = b == b ? b : -3.4e38f;
      }
    }
    float sum = 0.0f, mine_s = 0.0f;
    int mine_e = 0;
    #pragma clang loop unroll(disable)
    for (int t = 0; t < TOPK; ++t) {
      float bv = -INFINITY;
      int be = 0x7fffffff;
      for (int q = 0; q < NE; ++q) if (bv2[q] > bv) { bv = bv2[q]; be = int(lane) + 32 * q; }
      const float gv = metal::simd_max(bv);
      const int pick = metal::simd_min(bv == gv ? be : 0x7fffffff);
      float ps = 0.0f;
      for (int q = 0; q < NE; ++q) if (int(lane) + 32 * q == pick) { ps = sv2[q]; bv2[q] = -INFINITY; }
      const float s = metal::simd_sum(ps);
      if (int(lane) == t) { mine_s = s; mine_e = pick; }
      sum += s;
    }
    if (int(lane) < TOPK) {
      const float div = NORM ? sum + 1e-20f : 1.0f;
      idx[r * TOPK + int(lane)] = uint(mine_e);
      wts[r * TOPK + int(lane)] = mine_s / div * scal[0];
    }
  }
}
