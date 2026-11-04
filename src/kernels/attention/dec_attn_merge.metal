const int d = int(thread_position_in_threadgroup.x);
const int row = int(threadgroup_position_in_grid.y);
const int NS = pm_shape[1];
float M = -3.0e38f;
for (int j = 0; j < NS; ++j) if (pl[row * NS + j] > 0.0f) M = metal::max(M, pm[row * NS + j]);
float Z = 0.0f;
float acc = 0.0f;
for (int j = 0; j < NS; ++j) {
  const float lj = pl[row * NS + j];
  if (lj > 0.0f) {
    const float e = metal::exp2(pm[row * NS + j] - M);
    Z += lj * e;
    acc += e * pacc[((long)row * NS + j) * BDV + d];
  }
}
out[(long)row * BDV + d] = T(acc / Z);