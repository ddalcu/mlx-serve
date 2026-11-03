constexpr int NT = 256, NSG = NT / 32;
const uint row = threadgroup_position_in_grid.y;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const device T* x = logits + size_t(row) * V;
float v[K];
int id[K];
for (int j = 0; j < K; ++j) { v[j] = -INFINITY; id[j] = 0; }
for (int i = int(tid); i < V; i += NT) {
  float c = float(x[i]);
  if (!(c > v[K - 1])) continue;
  int ci = i;
  for (int j = 0; j < K; ++j) {
    const bool gt = c > v[j];
    const float tv = v[j]; const int ti = id[j];
    v[j] = gt ? c : tv; id[j] = gt ? ci : ti;
    c = gt ? tv : c; ci = gt ? ti : ci;
  }
}
threadgroup float sv[NSG * K];
threadgroup int si[NSG * K];
for (int r = 0; r < K; ++r) {
  const float best = simd_max(v[0]);
  const uint win = simd_min(v[0] == best ? lane : 64u);
  const int bid = simd_shuffle(id[0], ushort(win));
  if (lane == 0) { sv[sg * K + r] = best; si[sg * K + r] = bid; }
  if (lane == win) { for (int j = 0; j + 1 < K; ++j) { v[j] = v[j + 1]; id[j] = id[j + 1]; } v[K - 1] = -INFINITY; }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg != 0) return;
for (int j = 0; j < K; ++j) { v[j] = lane < NSG ? sv[lane * K + j] : -INFINITY; id[j] = lane < NSG ? si[lane * K + j] : 0; }
for (int r = 0; r < K; ++r) {
  const float best = simd_max(v[0]);
  const uint win = simd_min(v[0] == best ? lane : 64u);
  const int bid = simd_shuffle(id[0], ushort(win));
  if (lane == 0) { idx[row * K + r] = bid; val[row * K + r] = best; }
  if (lane == win) { for (int j = 0; j + 1 < K; ++j) { v[j] = v[j + 1]; id[j] = id[j + 1]; } v[K - 1] = -INFINITY; }
}