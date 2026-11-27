// A wide layer's FFN tail in one launch (`MLX_SERVE_DSV41_MOE_TAIL`): moe_combine.metal's unsort, k-sum and shared add
// (f32 in pair order, rounded once to T), then glm5_next/kernels/expand.metal's hc expand of the new stream from it, so
// the combined rows never go to memory. Four columns a thread.
const uint d4 = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
if (d4 >= uint(D / 4)) return;
const auto pr = post + row * HC;
const auto cb = comb + row * HC * HC;
float4 y = 0.0f;
#if TOK
// MoeParts.tok: the down rows in pair order, read in place.
for (int j = 0; j < K; ++j) y += float4(*(const device vec<T, 4>*)(dn + size_t(row * K + j) * D + 4 * d4));
#else
for (int j = 0; j < K; ++j) y += float4(*(const device vec<T, 4>*)(dn + size_t(inv[row * K + j]) * D + 4 * d4));
#endif
const float4 xv = float4(vec<T, 4>(y + float4(*(const device vec<T, 4>*)(sh + size_t(row) * D + 4 * d4))));
float4 sv[HC];
for (int j = 0; j < HC; ++j) sv[j] = float4(*(const device vec<T, 4>*)(res + (size_t(row) * HC + j) * D + 4 * d4));
for (int i = 0; i < HC; ++i) {
  float4 acc = 0.0f;
  for (int j = 0; j < HC; ++j) acc = fma(float4(cb[j * HC + i]), sv[j], acc);
  *(device vec<T, 4>*)(out + (size_t(row) * HC + i) * D + 4 * d4) = vec<T, 4>(fma(float4(pr[i]), xv, acc));
}
