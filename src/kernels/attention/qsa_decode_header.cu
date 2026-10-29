__device__ __forceinline__ float msv_bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float msv_bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ void msv_unpack8(uint4 r, float* f) {
  f[0] = msv_bf_lo(r.x); f[1] = msv_bf_hi(r.x); f[2] = msv_bf_lo(r.y); f[3] = msv_bf_hi(r.y);
  f[4] = msv_bf_lo(r.z); f[5] = msv_bf_hi(r.z); f[6] = msv_bf_lo(r.w); f[7] = msv_bf_hi(r.w);
}
// v[i] is this lane's partial dot for key i; each stage keeps half of the keys on the lane and
// trades the other half with the partner, leaving key lane >> 2 on the lane.
__device__ __forceinline__ float msv_reduce8(const float* v, int lane) {
  const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
  float w[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) w[i] = (b4 ? v[i + 4] : v[i]) + __shfl_xor_sync(0xffffffffu, b4 ? v[i] : v[i + 4], 16);
  float x[2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) x[i] = (b3 ? w[i + 2] : w[i]) + __shfl_xor_sync(0xffffffffu, b3 ? w[i] : w[i + 2], 8);
  float y = (b2 ? x[1] : x[0]) + __shfl_xor_sync(0xffffffffu, b2 ? x[0] : x[1], 4);
  y += __shfl_xor_sync(0xffffffffu, y, 1);
  return y + __shfl_xor_sync(0xffffffffu, y, 2);
}
