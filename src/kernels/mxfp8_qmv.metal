// y[M, N] = x[M, K] . W^T for an MXFP8 W: e4m3 bytes [N, K] read as u32 words, one e8m0
// exponent per 32. grid (32, N, 1), threadgroup (32, SGS, 1): a simdgroup owns one output row,
// a lane 16 bytes of it per step, decoded once and applied to every row of x.
const uint lane = thread_index_in_simdgroup;
const int n = int(thread_position_in_grid.y);
constexpr int KW = K / 4;
constexpr int KG = K / 32;
const device uint4* wrow = (const device uint4*)(w_q + (size_t)n * KW);
float acc[M];
for (int m = 0; m < M; ++m) acc[m] = 0.0f;
for (int k = int(lane) * 16; k < K; k += 512) {
  const uint4 q = wrow[k / 16];
  const uint e = uint(scales[(size_t)n * KG + k / 32]);
  // e8m0 as 2^(e - 127), times the 2^8 the e4m3 bit placement leaves out.
  const float sc = as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f;
  float wv[16];
  const uint words[4] = {q.x, q.y, q.z, q.w};
  for (int i = 0; i < 4; ++i) {
    const float2 lo = mlxserve_e4m3x2(words[i]);
    const float2 hi = mlxserve_e4m3x2(words[i] >> 8);
    wv[4 * i + 0] = lo.x;
    wv[4 * i + 1] = hi.x;
    wv[4 * i + 2] = lo.y;
    wv[4 * i + 3] = hi.y;
  }
  for (int m = 0; m < M; ++m) {
    const device T* xr = x + (size_t)m * K + k;
    float d = 0.0f;
    for (int i = 0; i < 16; ++i) d += float(xr[i]) * wv[i];
    acc[m] += d * sc;
  }
}
for (int m = 0; m < M; ++m) {
  const float v = simd_sum(acc[m]);
  if (lane == 0) y[(size_t)m * N + n] = T(v);
}
