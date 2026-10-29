const int l = threadgroup_position_in_grid.y, n0 = threadgroup_position_in_grid.x * KT;
const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int nb = keys_shape[0];
const int seen = (meta[0] + l + 1) / meta[1];
const int end = metal::min(nb, n0 + KT);
if (n0 >= seen) {
  for (int n = n0 + int(tid); n < end; n += 128) scores[size_t(l) * nb + n] = -INFINITY;
  return;
}
constexpr int RT = IH / 8, CT = KT / 32, C4 = IHD / 4;
using T4 = metal::vec<T, 4>;
threadgroup T kt[KT * IHD];
threadgroup float wt[IH], red[4][64];
for (uint i = tid; i < KT * C4; i += 128) {
  const int n = n0 + int(i / C4);
  ((threadgroup T4*)kt)[i] = n < nb ? ((const device T4*)keys)[size_t(n) * C4 + i % C4] : T4(0);
}
if (tid < IH) wt[tid] = float(w[size_t(l) * IH + tid]) * wsc[0];
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
const device T* qr = q + size_t(l) * IH * IHD;
const threadgroup T* kr = kt + sg * (KT / 4) * IHD;
metal::simdgroup_matrix<float, 8, 8> acc[RT][CT];
#pragma clang loop unroll(full)
for (int r = 0; r < RT; ++r)
  #pragma clang loop unroll(full)
  for (int c = 0; c < CT; ++c) acc[r][c] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
for (int d = 0; d < IHD; d += 8) {
  metal::simdgroup_matrix<T, 8, 8> a[RT], b[CT];
  #pragma clang loop unroll(full)
  for (int r = 0; r < RT; ++r) metal::simdgroup_load(a[r], qr + r * 8 * IHD + d, IHD);
  #pragma clang loop unroll(full)
  for (int c = 0; c < CT; ++c) metal::simdgroup_load(b[c], kr + c * 8 * IHD + d, IHD, metal::ulong2(0, 0), true);
  #pragma clang loop unroll(full)
  for (int r = 0; r < RT; ++r)
    #pragma clang loop unroll(full)
    for (int c = 0; c < CT; ++c) metal::simdgroup_multiply_accumulate(acc[r][c], a[r], b[c], acc[r][c]);
}
// A finished tile's rows are heads: relu, weight and sum them through this simdgroup's scratch.
// Threadgroup barriers: on M1 a simdgroup barrier let lanes read the scratch before the store landed.
const int col = lane & 7, row = (lane >> 3) * 2;
float s[CT];
#pragma clang loop unroll(full)
for (int c = 0; c < CT; ++c) s[c] = 0.0f;
#pragma clang loop unroll(full)
for (int r = 0; r < RT; ++r)
  #pragma clang loop unroll(full)
  for (int c = 0; c < CT; ++c) {
    metal::simdgroup_store(acc[r][c], red[sg], 8);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float p = metal::max(red[sg][row * 8 + col], 0.0f) * wt[r * 8 + row] + metal::max(red[sg][(row + 1) * 8 + col], 0.0f) * wt[r * 8 + row + 1];
    p += metal::simd_shuffle_xor(p, 8);
    p += metal::simd_shuffle_xor(p, 16);
    s[c] += p;
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
if (lane < 8)
  for (int c = 0; c < CT; ++c) {
    const int n = n0 + int(sg) * (KT / 4) + c * 8 + col;
    if (n < nb) scores[size_t(l) * nb + n] = n < seen ? s[c] : -INFINITY;
  }