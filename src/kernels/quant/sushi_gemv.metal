constexpr uint KC = uint(K) / 16u;
constexpr uint KB = uint(K) / 128u;
uint lane = thread_index_in_simdgroup;
uint row0 = (threadgroup_position_in_grid.x * uint(SGS) + simdgroup_index_in_threadgroup) * uint(NR);
const device uint4* wp[NR];
uint srow[NR];
float acc[M][NR];
#pragma clang loop unroll(full)
for (int r = 0; r < NR; ++r) {
  uint o = metal::min(row0 + uint(r), uint(N) - 1u);
  srow[r] = o * KB;
  wp[r] = (const device uint4*)(w + (size_t)o * (size_t)K);
  #pragma clang loop unroll(full)
  for (int m = 0; m < M; ++m) acc[m][r] = 0.0f;
}
for (uint c = lane; c < KC; c += 32u) {
  uint4 wr[NR];
  float sc[NR];
  #pragma clang loop unroll(full)
  for (int r = 0; r < NR; ++r) {
    wr[r] = wp[r][c];
    sc[r] = scales[srow[r] + (c >> 3)];
  }
  #pragma clang loop unroll(full)
  for (int i = 0; i < 4; ++i) {
    float4 wv[NR];
    #pragma clang loop unroll(full)
    for (int r = 0; r < NR; ++r) wv[r] = msv_e4m3_decode4(wr[r][i]) * sc[r];
    #pragma clang loop unroll(full)
    for (int m = 0; m < M; ++m) {
      float4 xv = float4(*((const device vec<T, 4>*)(x + (size_t)m * (size_t)K + c * 16u + uint(i) * 4u)));
      #pragma clang loop unroll(full)
      for (int r = 0; r < NR; ++r) acc[m][r] += dot(xv, wv[r]);
    }
  }
}
#pragma clang loop unroll(full)
for (int m = 0; m < M; ++m) {
  #pragma clang loop unroll(full)
  for (int r = 0; r < NR; ++r) {
    float total = simd_sum(acc[m][r]);
    uint o = row0 + uint(r);
    if (lane == 0u && o < uint(N)) y[(size_t)m * uint(N) + o] = T(total);
  }
}