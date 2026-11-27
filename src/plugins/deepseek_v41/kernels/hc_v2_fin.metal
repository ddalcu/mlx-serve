// PF_HC_V2 (prompt/hc_v2.zig): a row's hc stage from hc_v2_mix.metal's slice partials. Simdgroup 0 sums the slices'
// dots and stream squares into the 24 mixes (hc_mix's z = dot * rsqrt(mean sq + eps) * scale + base), then the split
// (hc_split.metal's statements: pre, post, the Sinkhorn comb), the mixes taken from the lanes that made them. Every
// thread rms-norms its columns of the collapse (`hraw`, rounded to T) with the slices' collapse squares:
// hc_norm's w * T(h * inv). One threadgroup of TG threads a row.
constexpr int MIX = (2 + HC) * HC;
constexpr int PW = MIX + 2;
constexpr int N = HC * D;
const int l = int(threadgroup_position_in_grid.y);
const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int L = hraw_shape[0];
float qh = 0.0f;
for (int s = 0; s < S; ++s) qh += part[(size_t(s) * L + l) * PW + MIX + 1];
const float inv = metal::precise::rsqrt(qh / float(D) + eps[0]);
if (sg == 0) {
  float qx = 0.0f;
  for (int s = 0; s < S; ++s) qx += part[(size_t(s) * L + l) * PW + MIX];
  const float rr = metal::precise::rsqrt(qx / float(N) + eps[0]);
  float zv = 0.0f;
  if (lane < uint(MIX)) {
    float a = 0.0f;
    for (int s = 0; s < S; ++s) a += part[(size_t(s) * L + l) * PW + lane];
    zv = a * rr * scale[lane] + base[lane];
    z[size_t(l) * MIX + lane] = zv;
  }
  const float e = eps[1];
  const float zpost = metal::simd_shuffle(zv, ushort((HC + lane) & 31u));
  if (lane < uint(HC)) {
    pre[l * HC + lane] = 1.0f / (1.0f + metal::exp(-zv)) + e;
    post[l * HC + lane] = 2.0f / (1.0f + metal::exp(-zpost));
  }
  // Lane a * HC + b holds comb[a][b]; each sum gathers its HC terms in the serial order.
  const bool act = lane < uint(HC * HC);
  const ushort a = act ? lane / HC : 0, b = act ? lane % HC : 0;
  float v = metal::simd_shuffle(zv, ushort(2 * HC + a * HC + b));
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
for (int c8 = int(tid); c8 < D / 8; c8 += TG) {
  const int d = c8 * 8;
  const uint4 w4 = *(const device uint4*)(hraw + size_t(l) * D + d);
  T o[8];
  for (int k = 0; k < 8; ++k) {
    const uint u = w4[k >> 1];
    const float hv = as_type<float>((k & 1) ? (u & 0xffff0000u) : (u << 16));
    o[k] = w[d + k] * T(hv * inv);
  }
  device T* dst = h + size_t(l) * D + d;
  *(device vec<T, 4>*)dst = vec<T, 4>(o[0], o[1], o[2], o[3]);
  *(device vec<T, 4>*)(dst + 4) = vec<T, 4>(o[4], o[5], o[6], o[7]);
}
