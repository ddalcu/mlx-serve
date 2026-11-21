const int l = threadgroup_position_in_grid.y, d = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int D = x_shape[1];
const size_t o = size_t(l) * D + d;
const float xv = float(x[o]);
threadgroup float t_s[32];
threadgroup float t_y[1024];
const float ss = metal::simd_sum(xv * xv);
if (lane == 0) t_s[sg] = ss;
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
float tot = 0.0f;
for (int g = 0; g < D / 32; ++g) tot += t_s[g];
const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
const T y = w[d] * T(xv * inv);
t_y[d] = float(y);
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
float r = float(y);
const int head = D - RD;
if (d >= head) {
  const int p = (d - head) / 2, b = d - (d - head) % 2;
  const float x1 = t_y[b], x2 = t_y[b + 1];
  const float theta = float(l + off) * (1.0f / periods[p]);
  const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
  r = float(T((d - head) % 2 == 0 ? x1 * cs - x2 * sn : x1 * sn + x2 * cs));
}
if (QAT) {
  const float scale = dsv41_pow2ceil(metal::max(metal::simd_max(metal::abs(r)), 1e-4f) * (1.0f / 448.0f));
  r = dsv41_fp8_roundtrip(metal::clamp(r / scale, -448.0f, 448.0f)) * scale;
}
out[o] = T(r);