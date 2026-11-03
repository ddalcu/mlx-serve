#include <metal_stdlib>
using namespace metal;
inline float msv_bf_lo(uint w) { return as_type<float>(w << 16); }
inline float msv_bf_hi(uint w) { return as_type<float>(w & 0xffff0000u); }
inline void msv_unpack8(uint4 r, thread float* f) {
  f[0] = msv_bf_lo(r.x); f[1] = msv_bf_hi(r.x); f[2] = msv_bf_lo(r.y); f[3] = msv_bf_hi(r.y);
  f[4] = msv_bf_lo(r.z); f[5] = msv_bf_hi(r.z); f[6] = msv_bf_lo(r.w); f[7] = msv_bf_hi(r.w);
}
// v[i] is this lane's partial dot for key i. Each stage keeps half of the keys on the lane
// and trades the other half with the partner; the key left on lane l is l >> 2.
inline float msv_reduce8(thread float* v, ushort lane) {
  const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
  float w[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) w[i] = (b4 ? v[i + 4] : v[i]) + simd_shuffle_xor(b4 ? v[i] : v[i + 4], 16);
  float x[2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) x[i] = (b3 ? w[i + 2] : w[i]) + simd_shuffle_xor(b3 ? w[i] : w[i + 2], 8);
  float y = (b2 ? x[1] : x[0]) + simd_shuffle_xor(b2 ? x[0] : x[1], 4);
  y += simd_shuffle_xor(y, 1);
  return y + simd_shuffle_xor(y, 2);
}
inline int msv_dec_pos(const device int* blk, int j, int sel_len, int tail_start, int ratio) {
  const int b = j / ratio;
  return (j < sel_len) ? (blk[b] * ratio + (j - b * ratio)) : (tail_start + (j - sel_len));
}
