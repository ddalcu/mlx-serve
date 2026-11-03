const int head = int(threadgroup_position_in_grid.x);
const int d = int(thread_position_in_threadgroup.x);
const ushort lane = ushort(thread_index_in_simdgroup);
threadgroup float wts[NS];
threadgroup float zsum[1];
if (simdgroup_index_in_threadgroup == 0) {
  constexpr int SLOTS = (NS + 31) / 32;
  float ms[SLOTS], ls[SLOTS];
  float M = -3.0e38f;
  #pragma unroll
  for (int s = 0; s < SLOTS; ++s) {
    const int j = lane + s * 32;
    const long r = (long)head * NS + j;
    ms[s] = (j < NS) ? pml[r * 2] : -3.0e38f;
    ls[s] = (j < NS) ? pml[r * 2 + 1] : 0.0f;
    if (ls[s] > 0.0f) M = metal::max(M, ms[s]);
  }
  M = simd_max(M);
  float Z = 0.0f;
  #pragma unroll
  for (int s = 0; s < SLOTS; ++s) {
    const int j = lane + s * 32;
    const float w = (ls[s] > 0.0f) ? metal::exp2(ms[s] - M) : 0.0f;
    if (j < NS) wts[j] = w;
    Z += w * ls[s];
  }
  Z = simd_sum(Z);
  if (lane == 0) zsum[0] = Z;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
const device float* pa = pacc + (long)head * NS * 256 + d;
float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
for (int j = 0; j < NS; ++j) a[j & 3] = fma(wts[j], pa[(long)j * 256], a[j & 3]);
out[(long)head * 256 + d] = T(((a[0] + a[1]) + (a[2] + a[3])) / zsum[0]);