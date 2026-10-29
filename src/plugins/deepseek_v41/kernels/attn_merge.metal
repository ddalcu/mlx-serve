const int d = thread_position_in_threadgroup.x, h = threadgroup_position_in_grid.y, i = threadgroup_position_in_grid.z;
const int H = pm_shape[1];
const size_t base = (size_t(i) * H + h) * S;
float M = sink[h];
for (int j = 0; j < S; ++j) M = metal::max(M, pm[base + j]);
float total = metal::exp(sink[h] - M), o = 0.0f;
for (int j = 0; j < S; ++j) {
  const float w = pl[base + j] > 0.0f ? metal::exp(pm[base + j] - M) : 0.0f;
  total += pl[base + j] * w;
  o += pacc[(base + j) * D + d] * w;
}
out[(size_t(i) * H + h) * D + d] = dsv41_unrope<OT, RD, D>(OT(o / total), d, meta[0] + i, periods);