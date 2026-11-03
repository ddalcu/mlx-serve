constexpr int K = 1280, D = 512, IHD = 128, NT = SGS * 32, PI = 16 / SGS, RQ = SGS * 2, UI = NI / 32, UQ = NQ / RQ;
const uint tg = threadgroup_position_in_grid.x;
const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
const uint h = lane >> 4, j = lane & 15;
threadgroup float4 xs[R * 320];
threadgroup float ps[R * SGS];
threadgroup float t_s[16];

// The last threadgroup takes the hc split, the ones before it the window rows; then each walks its units.
const int kv_l = int(G) - 2 - int(tg);
if (int(tg) == G - 1) {
  // One simdgroup a row, lane a * HC + b holding comb[a][b].
  constexpr int MIX = (2 + HC) * HC;
  const float e = eps[1];
  for (int l = int(sg); l < R; l += SGS) {
    const device float* zr = z + l * MIX;
    if (lane < HC) {
      pre[l * HC + lane] = 1.0f / (1.0f + metal::exp(-zr[lane])) + e;
      post[l * HC + lane] = 2.0f / (1.0f + metal::exp(-zr[HC + lane]));
    }
    const bool act = lane < HC * HC;
    const ushort a = act ? lane / HC : 0, b = act ? lane % HC : 0;
    float v = zr[2 * HC + a * HC + b];
    float mx = metal::simd_shuffle(v, a * HC);
    for (ushort k = 1; k < HC; ++k) mx = metal::max(mx, metal::simd_shuffle(v, a * HC + k));
    v = metal::exp(v - mx);
    float sm = 0.0f;
    for (ushort k = 0; k < HC; ++k) sm += metal::simd_shuffle(v, a * HC + k);
    v = v / sm + e;
    for (int t = 0; t < 2 * ITERS - 1; ++t) {
      const bool rows = t % 2 == 1;
      sm = 0.0f;
      for (ushort k = 0; k < HC; ++k) sm += metal::simd_shuffle(v, rows ? a * HC + k : k * HC + b);
      v = v / (sm + e);
    }
    if (act) comb[l * HC * HC + lane] = v;
  }
} else if (kv_l >= 0 && kv_l < R) {
  // Rms norm, tail rope, fp8 round trip; block b of 32 elements to simdgroup b % SGS.
  constexpr int NB = 16 / SGS;
  const device T* xr = kv + kv_l * D;
  float xv[NB];
  for (int i = 0; i < NB; ++i) {
    const int b = int(sg) + i * SGS;
    xv[i] = float(xr[32 * b + lane]);
    const float ss = metal::simd_sum(xv[i] * xv[i]);
    if (lane == 0) t_s[b] = ss;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int g = 0; g < 16; ++g) tot += t_s[g];
  const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
  for (int i = 0; i < NB; ++i) {
    const int d = 32 * (int(sg) + i * SGS) + int(lane);
    float r = float(kvn[d] * T(xv[i] * inv));
    if (d >= D - RD) {
      const float o = metal::simd_shuffle_xor(r, 1);
      const bool even = (d & 1) == 0;
      const float x1 = even ? r : o, x2 = even ? o : r;
      const float theta = float(kv_l + pos) * (1.0f / periods[(d - (D - RD)) / 2]);
      const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
      r = float(T(even ? x1 * cs - x2 * sn : x1 * sn + x2 * cs));
    }
    const float scale = dsv41_pow2ceil(metal::max(metal::simd_max(metal::abs(r)), 1e-4f) * (1.0f / 448.0f));
    r = dsv41_fp8_roundtrip(metal::clamp(r / scale, -448.0f, 448.0f)) * scale;
    kv_row[kv_l * D + d] = T(r);
  }
}

// qr = rms(qa, q_norm) of the R rows into xs: a thread takes 8 consecutive k of each row.
float inv[R];
for (int r = 0; r < R; ++r) {
  float ss = 0.0f;
  for (int v = int(tid); v < K / 8; v += NT) {
    const uint4 b = ((const device uint4*)(qa + r * K))[v];
    const uint bw[4] = {b.x, b.y, b.z, b.w};
    for (int i = 0; i < 4; ++i) {
      const float lo = as_type<float>(bw[i] << 16), hi = as_type<float>(bw[i] & 0xFFFF0000u);
      ss += lo * lo;
      ss += hi * hi;
    }
  }
  ss = metal::simd_sum(ss);
  if (lane == 0) ps[r * SGS + sg] = ss;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
for (int r = 0; r < R; ++r) {
  float tot = 0.0f;
  for (int g = 0; g < SGS; ++g) tot += ps[r * SGS + g];
  inv[r] = metal::precise::rsqrt(tot / float(K) + eps[0]);
}
for (int v = int(tid); v < K / 8; v += NT) {
  const uint4 nb = ((const device uint4*)qn)[v];
  const uint nw[4] = {nb.x, nb.y, nb.z, nb.w};
  const int s = v / 32, u = v & 1, jj = (v & 31) >> 1;
  for (int r = 0; r < R; ++r) {
    const uint4 b = ((const device uint4*)(qa + r * K))[v];
    const uint bw[4] = {b.x, b.y, b.z, b.w};
    float y[8];
    for (int i = 0; i < 4; ++i) {
      y[2 * i] = float(T(as_type<float>(nw[i] << 16) * float(T(as_type<float>(bw[i] << 16) * inv[r]))));
      y[2 * i + 1] = float(T(as_type<float>(nw[i] & 0xFFFF0000u) * float(T(as_type<float>(bw[i] & 0xFFFF0000u) * inv[r]))));
    }
    threadgroup float4* xd = xs + ((r * 5 + s) * 4 + 2 * u) * 16 + jj;
    xd[0] = float4(y[0], y[1], y[2], y[3]);
    xd[16] = float4(y[4], y[5], y[6], y[7]);
  }
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

// The units, indexer rows first: unit i of this threadgroup is i = tg, tg + G, ...
if constexpr (UI > 0) {
  for (int i = int(tg); i < UI; i += G) {
    // The indexer's q: rope the last RD of each head, then the fp4 / 32 / ue8m0 round trip over the unit's 32 rows.
    const int n_base = i * 32 + int(sg) * 2 * PI;
    dsv41_wqb_w<PI> wi;
    dsv41_wqb_load<PI>(wi, (const device uint4*)w_i + n_base * 80, (const device uchar*)s_i + n_base * 40, lane);
    float acc[PI][R];
    dsv41_wqb_dots<R, PI>(wi, xs, lane, acc);
    float vals[PI];
    float amax = 0.0f;
    for (int p = 0; p < PI; ++p) {
      float vr = 0.0f;
      for (int r = 0; r < R; ++r) {
        const float t = float(T(dsv41_wqb_sum16(acc[p][r])));
        if (int(j) == r) vr = t;
      }
      const int c = (n_base + 2 * p) % IHD;
      if (c >= IHD - RD) {
        const float o = metal::simd_shuffle_xor(vr, 16);
        const float theta = float(int(j) + pos) * (1.0f / periods_i[(c - (IHD - RD)) / 2]);
        const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
        const float x1 = h == 0 ? vr : o, x2 = h == 0 ? o : vr;
        vr = float(T(h == 0 ? x1 * cs - x2 * sn : x1 * sn + x2 * cs));
      }
      vals[p] = vr;
      amax = metal::max(amax, metal::abs(vr));
    }
    amax = metal::max(amax, metal::simd_shuffle_xor(amax, 16));
    if (lane < R) ps[lane * SGS + sg] = amax;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float blk = 0.0f;
    for (int g = 0; g < SGS; ++g) blk = metal::max(blk, ps[metal::min(int(j), R - 1) * SGS + g]);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    const float scale = dsv41_pow2ceil(metal::max(blk, 6.0f * 1.17549435e-38f) * (1.0f / 6.0f));
    for (int p = 0; p < PI; ++p) {
      const float qv = dsv41_e2m1(metal::clamp(vals[p] / scale, -6.0f, 6.0f)) * scale;
      if (int(j) < R) q_idx[int(j) * NI + n_base + 2 * p + int(h)] = float(T(qv));
    }
  }
}

// wq_b: rope the last RD of each head; the next unit's rows are read while this one's are multiplied.
const int i_q = int(tg) >= UI ? int(tg) : int(tg) + ((UI - int(tg) + int(G) - 1) / int(G)) * int(G);
if (i_q < UI + UQ) {
  dsv41_wqb_w<1> wq;
  const int n_first = (i_q - UI) * RQ + int(sg) * 2;
  dsv41_wqb_load<1>(wq, (const device uint4*)w_q + n_first * 80, (const device uchar*)s_q + n_first * 40, lane);
  for (int i = i_q; i < UI + UQ; i += G) {
    const int n_base = (i - UI) * RQ + int(sg) * 2;
    dsv41_wqb_w<1> nx;
    const bool more = i + int(G) < UI + UQ;
    if (more) {
      const int n_next = n_base + int(G) * RQ;
      dsv41_wqb_load<1>(nx, (const device uint4*)w_q + n_next * 80, (const device uchar*)s_q + n_next * 40, lane);
    }
    float acc[1][R];
    dsv41_wqb_dots<R, 1>(wq, xs, lane, acc);
    float vr = 0.0f;
    for (int r = 0; r < R; ++r) {
      const float t = float(T(dsv41_wqb_sum16(acc[0][r])));
      if (int(j) == r) vr = t;
    }
    const int c = n_base % D;
    if (c >= D - RD) {
      const float o = metal::simd_shuffle_xor(vr, 16);
      const float theta = float(int(j) + pos) * (1.0f / periods[(c - (D - RD)) / 2]);
      const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
      const float x1 = h == 0 ? vr : o, x2 = h == 0 ? o : vr;
      vr = float(T(h == 0 ? x1 * cs - x2 * sn : x1 * sn + x2 * cs));
    }
    if (int(j) < R) q[int(j) * NQ + n_base + int(h)] = T(vr);
    if (more) wq = nx;
  }
}
