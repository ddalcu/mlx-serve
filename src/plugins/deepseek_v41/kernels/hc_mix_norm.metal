// hc_mix.metal's mixes and hc_norm.metal's collapse and rms norm of the same
// stream rows in one launch, so the stream comes from memory once (a row group's norm threadgroup runs beside its mix
// threadgroups, which read the same rows): threadgroups 0 .. MIX/M - 1 of a row group are hc_mix's, statement for
// statement; threadgroup MIX/M collapses its R rows with `cpre` and rms-norms them with `w`, hc_norm's statements a row
// over the same TG threads, without the split (hc_split takes it from the mixes after). H2: the same collapse normed a
// second time with `w2` into `h2` (the read-ahead's FFN norm: hc_norm again over the same rows, the same inverse).
const int mb = threadgroup_position_in_grid.x, l0 = threadgroup_position_in_grid.y * R;
const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int N = fn_shape[1], MIX = fn_shape[0], L = x_shape[0];
threadgroup float t_d[R][M][TG / 32], t_s[R][TG / 32];
if (mb < MIX / M) {
  const int m0 = mb * M;
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
} else {
  const int D = N / HC;
  float ss[R];
  for (int r = 0; r < R; ++r) {
    ss[r] = 0.0f;
    const int l = l0 + r;
    if (l >= L) continue;
    float cp[HC];
    for (int j = 0; j < HC; ++j) cp[j] = cpre[l * HC + j];
    for (int d = tid; d < D; d += TG) {
      float h = 0.0f;
      for (int j = 0; j < HC; ++j) h += cp[j] * float(x[(size_t(l) * HC + j) * D + d]);
      const T hb = T(h);
      h_out[size_t(l) * D + d] = hb;
      ss[r] += float(hb) * float(hb);
    }
  }
  for (int r = 0; r < R; ++r) {
    const float v = metal::simd_sum(ss[r]);
    if (lane == 0) t_s[r][sg] = v;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int r = 0; r < R; ++r) {
    const int l = l0 + r;
    if (l >= L) continue;
    float tot = 0.0f;
    for (int g = 0; g < TG / 32; ++g) tot += t_s[r][g];
    const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
    for (int d = tid; d < D; d += TG) {
      const size_t o = size_t(l) * D + d;
      const float hv = float(h_out[o]);
      h_out[o] = w[d] * T(hv * inv);
      if (H2 != 0) h2[o] = w2[d] * T(hv * inv);
    }
  }
}
