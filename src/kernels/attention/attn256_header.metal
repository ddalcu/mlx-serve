#include <metal_simdgroup_matrix>

// Fragment layout mirrors MLX steel BaseMMAFrag<float,8,8>: each thread
// of a simdgroup holds 2 adjacent elements of an 8x8 tile; the hardware
// mma runs on simdgroup_float8x8 built from those elements. The 4 threads
// holding one row differ in lane bits 0 and 3 (see msv_coord).
inline short2 msv_coord(ushort lane) {
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  return short2(fn, fm);
}

inline void msv_mma(thread float2 &d, float2 a, float2 b) {
  metal::simdgroup_float8x8 D, A, B, C;
  A.thread_elements()[0] = a.x;
  A.thread_elements()[1] = a.y;
  B.thread_elements()[0] = b.x;
  B.thread_elements()[1] = b.y;
  C.thread_elements()[0] = d.x;
  C.thread_elements()[1] = d.y;
  simdgroup_multiply_accumulate(D, A, B, C);
  d.x = D.thread_elements()[0];
  d.y = D.thread_elements()[1];
}

inline float msv_row_max(float2 v) {
  float t = metal::max(v.x, v.y);
  t = metal::max(t, metal::simd_shuffle_xor(t, ushort(1)));
  t = metal::max(t, metal::simd_shuffle_xor(t, ushort(8)));
  return t;
}

inline float msv_row_sum(float2 v) {
  float t = v.x + v.y;
  t += metal::simd_shuffle_xor(t, ushort(1));
  t += metal::simd_shuffle_xor(t, ushort(8));
  return t;
}
