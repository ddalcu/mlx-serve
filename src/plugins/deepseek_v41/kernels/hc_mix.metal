const int m0 = threadgroup_position_in_grid.x * M, l0 = threadgroup_position_in_grid.y * R;
const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int N = fn_shape[1], MIX = fn_shape[0], L = x_shape[0];
float d[R][M], ss[R];
for (int r = 0; r < R; ++r) { ss[r] = 0.0f; for (int j = 0; j < M; ++j) d[r][j] = 0.0f; }
for (int k = tid; k < N; k += TG) {
  float fv[M];
  #pragma clang loop unroll(full)
  for (int j = 0; j < M; ++j) fv[j] = fn[size_t(m0 + j) * N + k];
  #pragma clang loop unroll(full)
  for (int r = 0; r < R; ++r) {
    const float v = l0 + r < L ? float(x[size_t(l0 + r) * N + k]) : 0.0f;
    #pragma clang loop unroll(full)
    for (int j = 0; j < M; ++j) d[r][j] += v * fv[j];
    ss[r] += v * v;
  }
}
threadgroup float t_d[R][M][TG / 32], t_s[R][TG / 32];
for (int r = 0; r < R; ++r) {
  for (int j = 0; j < M; ++j) { const float v = metal::simd_sum(d[r][j]); if (lane == 0) t_d[r][j][sg] = v; }
  const float v = metal::simd_sum(ss[r]);
  if (lane == 0) t_s[r][sg] = v;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
if (tid < R * M) {
  const int r = tid / M, j = tid % M;
  float a = 0.0f, b = 0.0f;
  for (int g = 0; g < TG / 32; ++g) { a += t_d[r][j][g]; b += t_s[r][g]; }
  if (l0 + r < L) z[(l0 + r) * MIX + m0 + j] = a * metal::precise::rsqrt(b / float(N) + eps[0]) * scale[m0 + j] + base[m0 + j];
}