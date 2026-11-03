const uint lane = thread_index_in_simdgroup;
const int e = threadgroup_position_in_grid.x * SGS + simdgroup_index_in_threadgroup;
const int l = threadgroup_position_in_grid.y;
const int E = w_shape[0], D = w_shape[1];
if (e >= E) return;
const device auto* wr = w + size_t(e) * D;
const device auto* xr = x + size_t(l) * D;
float acc = 0.0f;
for (int k = lane * 8; k < D; k += 256)
  for (int j = 0; j < 8; ++j) acc += float(xr[k + j]) * float(wr[k + j]);
acc = metal::simd_sum(acc);
if (lane == 0) logits[size_t(l) * E + e] = acc;