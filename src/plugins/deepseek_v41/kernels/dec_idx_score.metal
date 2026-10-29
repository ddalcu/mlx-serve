// The indexer's scores sum_h w_h relu(q_h . key_n) on NAX. A threadgroup of 4 simdgroups scores one row over a range
// of tiles (the rows of a range are adjacent threadgroups, so a tile is read from memory once); a tile of 16 keys is
// the left operand of the 8 ops of a 16 x 32 x 16 matmul against the row's 32 heads (the transposed right operand,
// kept in registers for the whole launch), so C = keys x heads and a key's score is a reduction over 4 lanes. The
// contraction axis is permuted (lane group f4 owns 8 consecutive channels a 32), which makes a lane's key loads 16
// bytes. FULL: tile t = keys 16t..16t+15. GATHER: tile t = candidate blocks cand[r][2t], cand[r][2t+1]; slot s = 8 b + e
// scores key cand[r][b] * 8 + e.
constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, metal::execution_simdgroup> op;
const uint lane = thread_index_in_simdgroup;
const int R = w_shape[0];
const int nkeys = keys_shape[0];
const int ratio = meta[1];
const int row = int(threadgroup_position_in_grid.x) % R;
const int nranges = int(threadgroups_per_grid.x) / R;
const int nstreams = int(threads_per_threadgroup.x / 32);
#if GATHER
const int ncols = int(cand_shape[1]) * 8;
#else
const int ncols = nkeys;
#endif
const ushort qid = lane >> 2;
const ushort fm = (qid & 4) | ((lane >> 1) & 3);
const ushort f4 = (qid & 2) | (lane & 1);

// The row's right operands, one per step of 16 contracted channels.
auto ct_a = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
auto ct_b0 = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
auto ct_b1 = ct_b0;
auto ct_b2 = ct_b0;
auto ct_b3 = ct_b0;
auto ct_b4 = ct_b0;
auto ct_b5 = ct_b0;
auto ct_b6 = ct_b0;
auto ct_b7 = ct_b0;
auto ct_c = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>, metal::remove_addrspace_t<decltype(ct_b0)>, float>();
#define LOAD_B(KS, CT) { \
  const int d0_ = ((KS) >> 1) * 32 + f4 * 8 + ((KS) & 1) * 4; \
  _Pragma("unroll") for (ushort tn = 0; tn < 2; ++tn) _Pragma("unroll") for (ushort i = 0; i < 2; ++i) { \
    const vec<float, 4> v = vec<float, 4>(*(const device vec<QT, 4>*)(q + (size_t(row) * 32 + tn * 16 + fm + i * 8) * 128 + d0_)); \
    _Pragma("unroll") for (ushort j = 0; j < 4; ++j) CT[tn * 8 + i * 4 + j] = bfloat(v[j]); } }
LOAD_B(0, ct_b0) LOAD_B(1, ct_b1) LOAD_B(2, ct_b2) LOAD_B(3, ct_b3) LOAD_B(4, ct_b4) LOAD_B(5, ct_b5) LOAD_B(6, ct_b6) LOAD_B(7, ct_b7)

// The weights of the lane's head columns, and how many keys the row sees.
float wv[8];
#pragma unroll
for (ushort tn = 0; tn < 2; ++tn)
  #pragma unroll
  for (ushort j = 0; j < 4; ++j) wv[tn * 4 + j] = float(w[row * 32 + tn * 16 + f4 * 4 + j]) * wsc[0];
const int seen = min((meta[0] + row + 1) / ratio, nkeys);
device float* out_row = scores + size_t(row) * ncols;

const int ntiles = (ncols + 15) / 16;
const int stride = nranges * nstreams;
for (int tile = (int(threadgroup_position_in_grid.x) / R) * nstreams + int(simdgroup_index_in_threadgroup); tile < ntiles; tile += stride) {
#if GATHER
  const int c0 = cand[row * (ncols / 8) + tile * 2], c1 = cand[row * (ncols / 8) + tile * 2 + 1];
  const int n0 = c0 * 8 + fm, n1 = c1 * 8 + fm;
  const bool v0 = c0 >= 0, v1 = c1 >= 0;
#else
  const int n0 = tile * 16 + fm, n1 = n0 + 8;
  const bool v0 = true, v1 = true;
#endif
  const device vec<bfloat, 8>* k0 = (const device vec<bfloat, 8>*)(keys + size_t(max(0, min(n0, nkeys - 1))) * 128 + f4 * 8);
  const device vec<bfloat, 8>* k1 = (const device vec<bfloat, 8>*)(keys + size_t(max(0, min(n1, nkeys - 1))) * 128 + f4 * 8);
  vec<bfloat, 8> a0[4], a1[4];
  #pragma unroll
  for (ushort c = 0; c < 4; ++c) { a0[c] = k0[c * 4]; a1[c] = k1[c * 4]; }
  #pragma unroll
  for (short e = 0; e < 16; ++e) ct_c[e] = 0.0f;
#define STEP(KS, CT_B) { \
    _Pragma("unroll") for (ushort j = 0; j < 4; ++j) { ct_a[j] = a0[(KS) >> 1][((KS) & 1) * 4 + j]; ct_a[4 + j] = a1[(KS) >> 1][((KS) & 1) * 4 + j]; } \
    op.run(ct_a, CT_B, ct_c); }
  STEP(0, ct_b0) STEP(1, ct_b1) STEP(2, ct_b2) STEP(3, ct_b3) STEP(4, ct_b4) STEP(5, ct_b5) STEP(6, ct_b6) STEP(7, ct_b7)
  // relu, the head weight, the sum over the lane's 8 head columns, then over the 4 lanes of the key row.
  float s0 = 0.0f, s1 = 0.0f;
  #pragma unroll
  for (ushort tn = 0; tn < 2; ++tn)
    #pragma unroll
    for (ushort j = 0; j < 4; ++j) {
      s0 += metal::max(ct_c[tn * 8 + j], 0.0f) * wv[tn * 4 + j];
      s1 += metal::max(ct_c[tn * 8 + 4 + j], 0.0f) * wv[tn * 4 + j];
    }
  s0 += simd_shuffle_xor(s0, 1);
  s0 += simd_shuffle_xor(s0, 8);
  s1 += simd_shuffle_xor(s1, 1);
  s1 += simd_shuffle_xor(s1, 8);
  if (f4 == 0) {
    const int m0 = tile * 16 + fm, m1 = m0 + 8;
    if (m0 < ncols) out_row[m0] = (v0 && n0 < seen) ? s0 : -INFINITY;
    if (m1 < ncols) out_row[m1] = (v1 && n1 < seen) ? s1 : -INFINITY;
  }
}
