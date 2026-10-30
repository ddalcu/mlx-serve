if (role == 0) {
  float cp[HC];
  for (int j = 0; j < HC; ++j) cp[j] = cpre[l * HC + j];
  float ss = 0.0f;
  T hv[ND];
  #pragma clang loop unroll(full)
  for (int c = 0; c < ND; ++c) {
    float h = 0.0f;
    for (int j = 0; j < HC; ++j) h += cp[j] * v[j][c];
    hv[c] = T(h);
    ss += float(hv[c]) * float(hv[c]);
  }
  ss = metal::simd_sum(ss);
  if (lane == 0) t_a[sg] = ss;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int g = 0; g < TG / 32; ++g) tot += t_a[g];
  const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
  #pragma clang loop unroll(full)
  for (int c = 0; c < ND; ++c) {
    const int d = int(tid) + c * TG;
    if (d < D) h_out[size_t(l) * D + d] = w[d] * T(float(hv[c]) * inv);
  }
  return;
}
const int m = int(role) - 1;
const device float* fr = fn + size_t(m) * N;
float dd = 0.0f, sq = 0.0f;
#pragma clang loop unroll(full)
for (int i = 0; i < HC; ++i)
  #pragma clang loop unroll(full)
  for (int c = 0; c < ND; ++c) {
    const int d = int(tid) + c * TG;
    const float u = v[i][c];
    if (d < D) dd += u * fr[i * D + d];
    sq += u * u;
  }
dd = metal::simd_sum(dd); sq = metal::simd_sum(sq);
if (lane == 0) { t_a[sg] = dd; t_b[sg] = sq; }
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
if (tid == 0) {
  float a = 0.0f, b = 0.0f;
  for (int g = 0; g < TG / 32; ++g) { a += t_a[g]; b += t_b[g]; }
  z[l * ((2 + HC) * HC) + m] = a * metal::precise::rsqrt(b / float(N) + eps[0]) * scale[m] + base[m];
}