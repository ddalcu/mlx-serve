using BT = metal::conditional_t<metal::is_same_v<T, half>, half, float>;
constexpr uint CH = 32, TK = 128, KW = K / 16, G = K / 128, KS = K / SPLIT;
uint lane = thread_index_in_simdgroup;
uint qid = lane / 4;
uint fm = (qid & 4) + ((lane / 2) % 4);
uint fn = (qid & 2) * 2 + (lane % 2) * 2;
uint row_base = (threadgroup_position_in_grid.x * SGS + simdgroup_index_in_threadgroup) * 8 * RB;
simdgroup_float8x8 acc[RB][NB];
for (uint b = 0; b < RB; ++b) for (uint c = 0; c < NB; ++c) acc[b][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
uint kl = (fn / 2) * CH;
uint kb = (fm / 2) * CH + (fm % 2) * 8;
uint split = threadgroup_position_in_grid.y;
for (uint k0 = split * KS; k0 < (split + 1) * KS; k0 += TK) {
  uint2 words[RB];
  for (uint b = 0; b < RB; ++b) words[b] = *(const device uint2*)(w + (row_base + b * 8 + fm) * KW + (k0 + kl) / 16);
  simdgroup_float8x8 part[RB][NB];
  for (uint b = 0; b < RB; ++b) for (uint c = 0; c < NB; ++c) part[b][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint j = 0; j < CH / 2; ++j) {
    uint k = k0 + kb + (j % 8) + 16 * (j / 8);
    simdgroup_matrix<BT, 8, 8> bx[NB];
    for (uint c = 0; c < NB; ++c) {
      bx[c].thread_elements()[0] = BT(x[min(fn + 8 * c, uint(M) - 1) * K + k]);
      bx[c].thread_elements()[1] = BT(x[min(fn + 8 * c + 1, uint(M) - 1) * K + k]);
    }
    for (uint b = 0; b < RB; ++b) {
      uint u = ((j < 8 ? words[b].x : words[b].y) >> (2 * (j % 8))) & 0x00030003u;
      half2 q = as_type<half2>(u | 0x64006400u) - half2(1025.0h);
      simdgroup_half8x8 a;
      a.thread_elements()[0] = q.x;
      a.thread_elements()[1] = q.y;
      for (uint c = 0; c < NB; ++c) simdgroup_multiply_accumulate(part[b][c], a, bx[c], part[b][c]);
    }
  }
  for (uint b = 0; b < RB; ++b) {
    float sc = float(scales[(row_base + b * 8 + fm) * G + k0 / 128]);
    for (uint c = 0; c < NB; ++c) {
      acc[b][c].thread_elements()[0] += sc * part[b][c].thread_elements()[0];
      acc[b][c].thread_elements()[1] += sc * part[b][c].thread_elements()[1];
    }
  }
}
for (uint b = 0; b < RB; ++b) {
  uint row = row_base + b * 8 + fm;
  for (uint c = 0; c < NB; ++c) {
    uint m0 = fn + 8 * c;
    if (m0 < uint(M)) y[(split * M + m0) * N + row] = static_cast<OT>(acc[b][c].thread_elements()[0]);
    if (m0 + 1 < uint(M)) y[(split * M + m0 + 1) * N + row] = static_cast<OT>(acc[b][c].thread_elements()[1]);
  }
}