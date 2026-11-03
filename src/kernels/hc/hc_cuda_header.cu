template <bool DENSE>
__device__ __forceinline__ float hc_val(unsigned q) {
  if constexpr (DENSE) return __bfloat162float(__ushort_as_bfloat16((unsigned short)q));
  else return (float)q;
}
__device__ __forceinline__ float hc_warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}
__device__ __forceinline__ float hc_sig(float v) { return 1.0f / (1.0f + expf(-v)); }