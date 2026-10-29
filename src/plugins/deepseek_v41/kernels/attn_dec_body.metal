const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int i = threadgroup_position_in_grid.y, h = threadgroup_position_in_grid.x;
const int p0 = meta[0], K = meta[2], window = meta[3], cap = meta[4], nc = meta[5];
const int H = q_shape[1];
constexpr int V = D / 32;
const int pos = p0 + i;
const int lo = metal::max(0, pos - window + 1);
const int nw = pos - lo + 1;
float qv[V], acc[V];
for (int v = 0; v < V; ++v) { qv[v] = float(q[(i * H + h) * D + lane * V + v]); acc[v] = 0.0f; }
float mx = -INFINITY, den = 0.0f;
const float scale = scal[0];
// Split sp of S takes rows [t_lo, t_hi) of the window + selected list.
const int sp = threadgroup_position_in_grid.z, chunk = (nw + K + S - 1) / S;
const int t_lo = sp * chunk, t_hi = metal::min(nw + K, t_lo + chunk);
for (int t = t_lo + sg; t < t_hi; t += NSG) {
  const device KT* row;
  if (t < nw) row = ring + size_t((lo + t) % cap) * D;
  else {
    const int c = topk[i * K + t - nw];
    if (c < 0 || c >= nc) continue;
    row = comp + size_t(c) * D;
  }
  float kv[V], dot = 0.0f;
  for (int v = 0; v < V; ++v) { kv[v] = float(row[lane * V + v]); dot += qv[v] * kv[v]; }
  const float s = metal::simd_sum(dot) * scale;
  const float nm = metal::max(mx, s);
  const float corr = metal::exp(mx - nm), p = metal::exp(s - nm);
  den = den * corr + p;
  for (int v = 0; v < V; ++v) acc[v] = acc[v] * corr + p * kv[v];
  mx = nm;
}
threadgroup float t_acc[NSG * D];
threadgroup float t_mx[NSG], t_den[NSG];
for (int v = 0; v < V; ++v) t_acc[sg * D + lane * V + v] = acc[v];
if (lane == 0) { t_mx[sg] = mx; t_den[sg] = den; }
threadgroup_barrier(metal::mem_flags::mem_threadgroup);