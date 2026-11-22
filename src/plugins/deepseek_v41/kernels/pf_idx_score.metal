// A wide prompt span's indexer scores sum_h w_h relu(q_h . key_n) on NAX (MLX_SERVE_DSV41_PF_IDX_NAX), cut to the
// release's shapes: 32 heads of 128 channels, bf16 keys and queries (the fp4 round trip's output, exact in bf16).
// dec_idx_score.metal's tile op (a 16 x 32 x 16 matmul: 16 keys x 16 channels against the row's 32 heads, the heads'
// operands kept in registers for the whole launch, the contraction axis permuted so a lane's key loads are 16 bytes)
// with the prompt's mapping: a threadgroup is NSG adjacent rows (a simdgroup a row) over one range of key tiles, so
// the NSG simdgroups read each tile at once (one read from memory, the rest from the cache) and each row's queries are
// loaded once a range. The next tile's keys load while the current one contracts. A tile every row of the simdgroup
// sees none of ((p0 + row + 1) / ratio keys) is written -inf without its contraction. Ranges:
// threadgroups_per_grid.x, each ceil(tiles / ranges) adjacent tiles.
constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, metal::execution_simdgroup> op;
const uint lane = thread_index_in_simdgroup;
const int L = w_shape[0];
const int nkeys = keys_shape[0];
const int ratio = meta[1];
const int row = int(threadgroup_position_in_grid.y) * NSG + int(simdgroup_index_in_threadgroup);
if (row >= L) return;
const int ntiles = (nkeys + 15) / 16;
const int nranges = int(threadgroups_per_grid.x);
const int per = (ntiles + nranges - 1) / nranges;
const int t0 = int(threadgroup_position_in_grid.x) * per;
const int t1 = min(ntiles, t0 + per);
if (t0 >= t1) return;
const ushort qid = lane >> 2;
const ushort fm = (qid & 4) | ((lane >> 1) & 3);
const ushort f4 = (qid & 2) | (lane & 1);

// The row's right operands, one per step of 16 contracted channels.
auto ct_a = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
auto ct_b0 = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
auto ct_b1 = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
auto ct_b2 = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
auto ct_b3 = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
auto ct_b4 = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
auto ct_b5 = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
auto ct_b6 = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
auto ct_b7 = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
auto ct_c = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>, metal::remove_addrspace_t<decltype(ct_b0)>, float>();
#define LOAD_B(KS, CT) { \
  const int d0_ = ((KS) >> 1) * 32 + f4 * 8 + ((KS) & 1) * 4; \
  _Pragma("unroll") for (ushort tn = 0; tn < 2; ++tn) _Pragma("unroll") for (ushort i = 0; i < 2; ++i) { \
    const vec<bfloat, 4> v = *(const device vec<bfloat, 4>*)(q + (size_t(row) * 32 + tn * 16 + fm + i * 8) * 128 + d0_); \
    _Pragma("unroll") for (ushort j = 0; j < 4; ++j) CT[tn * 8 + i * 4 + j] = v[j]; } }
LOAD_B(0, ct_b0) LOAD_B(1, ct_b1) LOAD_B(2, ct_b2) LOAD_B(3, ct_b3) LOAD_B(4, ct_b4) LOAD_B(5, ct_b5) LOAD_B(6, ct_b6) LOAD_B(7, ct_b7)

// The weights of the lane's head columns, and how many keys the row sees.
float wv[8];
#pragma unroll
for (ushort tn = 0; tn < 2; ++tn)
  #pragma unroll
  for (ushort j = 0; j < 4; ++j) wv[tn * 4 + j] = float(w[row * 32 + tn * 16 + f4 * 4 + j]) * wsc[0];
#if DSV41_PF_IDX_PLANT
const int seen = min((meta[0] + row + 1) / ratio + 1, nkeys);
#else
const int seen = min((meta[0] + row + 1) / ratio, nkeys);
#endif
// Tiles at or past `live_end` hold no key the row sees.
const int live_end = min(t1, (seen + 15) / 16);
device float* out_row = scores + size_t(row) * nkeys;

vec<bfloat, 8> a0[4], a1[4];
#define LOAD_KEYS(TILE, X0, X1) { \
  const int n0_ = (TILE) * 16 + fm, n1_ = n0_ + 8; \
  const device vec<bfloat, 8>* k0_ = (const device vec<bfloat, 8>*)(keys + size_t(min(n0_, nkeys - 1)) * 128 + f4 * 8); \
  const device vec<bfloat, 8>* k1_ = (const device vec<bfloat, 8>*)(keys + size_t(min(n1_, nkeys - 1)) * 128 + f4 * 8); \
  _Pragma("unroll") for (ushort c = 0; c < 4; ++c) { X0[c] = k0_[c * 4]; X1[c] = k1_[c * 4]; } }
if (t0 < live_end) LOAD_KEYS(t0, a0, a1)
for (int tile = t0; tile < live_end; ++tile) {
  vec<bfloat, 8> b0[4], b1[4];
  if (tile + 1 < live_end) LOAD_KEYS(tile + 1, b0, b1)
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
    const int n0 = tile * 16 + fm, n1 = n0 + 8;
    if (n0 < nkeys) out_row[n0] = n0 < seen ? s0 : -INFINITY;
    if (n1 < nkeys) out_row[n1] = n1 < seen ? s1 : -INFINITY;
  }
  #pragma unroll
  for (ushort c = 0; c < 4; ++c) { a0[c] = b0[c]; a1[c] = b1[c]; }
}
// The range's tiles past what the row sees.
for (int n = max(t0, live_end) * 16 + int(lane); n < t1 * 16 && n < nkeys; n += 32) out_row[n] = -INFINITY;
