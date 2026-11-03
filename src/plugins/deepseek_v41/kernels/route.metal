const int e = thread_position_in_threadgroup.x, l = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int E = logits_shape[1];
threadgroup float t_s[1024], t_v[32];
threadgroup int t_i[32], t_pick[K];
float s = 0.0f, v = -INFINITY;
if (e < E) {
  const float z = logits[size_t(l) * E + e];
  const float y = metal::exp(-metal::abs(z));
  const float lp = y < 1e-4f ? y * (1.0f - y * (0.5f - y / 3.0f)) : metal::log(1.0f + y);
  s = metal::sqrt(metal::max(z, 0.0f) + lp);
  v = s + bias[(bias_shape[0] == 1 ? 0 : l) * E + e];
  t_s[e] = s;
}
for (int r = 0; r < K; ++r) {
  const float bv = metal::simd_max(v);
  const int bi = metal::simd_min(v == bv && e < E ? e : 0x7fffffff);
  if (lane == 0) { t_v[sg] = bv; t_i[sg] = bi; }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (e == 0) {
    const int nsg = (threads_per_threadgroup.x + 31) / 32;
    float best = t_v[0]; int at = t_i[0];
    for (int g = 1; g < nsg; ++g) if (t_v[g] > best || (t_v[g] == best && t_i[g] < at)) { best = t_v[g]; at = t_i[g]; }
    t_pick[r] = at;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (e == t_pick[r]) v = -INFINITY;
}
if (e == 0) {
  float sum = 0.0f;
  for (int r = 0; r < K; ++r) sum += t_s[t_pick[r]];
  const float div = NORM ? sum + 1e-20f : 1.0f;
  for (int r = 0; r < K; ++r) {
    idx[l * K + r] = uint(t_pick[r]);
    wts[l * K + r] = t_s[t_pick[r]] / div * scal[0];
  }
}