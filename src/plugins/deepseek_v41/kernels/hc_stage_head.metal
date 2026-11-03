const uint role = threadgroup_position_in_grid.x;
const int l = threadgroup_position_in_grid.y;
const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int D = x_shape[2], N = fn_shape[1];
threadgroup float t_a[TG / 32], t_b[TG / 32];
const device auto* xr = x + size_t(l) * HC * D;
float v[HC][ND];
for (int c = 0; c < ND; ++c) {
  const int d = int(tid) + c * TG;
  for (int i = 0; i < HC; ++i) v[i][c] = 0.0f;