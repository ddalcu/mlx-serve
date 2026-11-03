const uint lane = thread_index_in_simdgroup;
const int n0 = thread_position_in_grid.y * RPS, l = thread_position_in_grid.z;
const int K = x_shape[1], N = w_shape[0];
float acc[RPS];
for (int r = 0; r < RPS; ++r) acc[r] = 0.0f;
const device float* xr = x + size_t(l) * K;
for (int k = int(lane) * 4; k < K; k += 128) {
  const float4 xv = *(const device float4*)(xr + k);
  for (int r = 0; r < RPS; ++r) {
    const device auto* wr = w + size_t(n0 + r) * K + k;
    acc[r] += xv.x * float(wr[0]) + xv.y * float(wr[1]) + xv.z * float(wr[2]) + xv.w * float(wr[3]);
  }
}
for (int r = 0; r < RPS; ++r) {
  const float v = metal::simd_sum(acc[r]);
  if (lane == 0 && n0 + r < N) out[size_t(l) * N + n0 + r] = v;
}