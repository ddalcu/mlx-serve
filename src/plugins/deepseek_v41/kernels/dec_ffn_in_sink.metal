// dec_ffn_in, MLX_SERVE_DSV41_FFN_IN_SINK: launch 2's mix tail as its own launch, one threadgroup. The mix threadgroups'
// partials (P slices of each row's 24 dots and sum of squares, `parts`) reduced over the slices, then the Sinkhorn
// split: launch 2's last mix threadgroup's words, read after launch 2 has finished. Its outputs are read only by the
// MoE down's hc expand, so it runs beside the MoE up. `order` (launch 2's h2) only orders it after launch 2.
constexpr int MIX = (2 + HC) * HC;
constexpr int PL = MIX + 1;
const int tid = int(thread_index_in_threadgroup);
const uint lane = thread_index_in_simdgroup;
const int sg = int(simdgroup_index_in_threadgroup);
(void)order;
threadgroup float tgm[R * P * PL + R * PL];
threadgroup float* zr = tgm + R * P * PL;
for (int i = tid; i < R * P * PL; i += 256) tgm[i] = parts[i];
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
// A simdgroup reduces whole (row, mix) columns over the slices: lanes stride them, then a tree.
#pragma clang loop unroll(disable)
for (int pair = sg; pair < R * PL; pair += 8) {
  const int r = pair / PL, m = pair % PL;
  float v = 0.0f;
  for (int p = int(lane); p < P; p += 32) v += tgm[(r * P + p) * PL + m];
  v = metal::simd_sum(v);
  if (lane == 0) zr[pair] = v;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
// The Sinkhorn split, a quad a row: lane i holds row i of the 4x4, so a row sum is in-lane and a column sum a quad_sum.
if (sg == 0 && lane < 4 * R) {
  const int r = int(lane) / 4, i = int(lane) % 4;
  const threadgroup float* zz = zr + r * PL;
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
