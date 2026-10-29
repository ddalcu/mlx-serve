int t = thread_position_in_grid.x;
if (t >= S) return;
const int PACK = 2 * HC + HC * HC;
const int MIX = (2 + HC) * HC;
float rsq = metal::rsqrt(ssq[t] / consts[0] + consts[1]);
const float heps = consts[2];
float pre_v[HC];
float post_v[HC];
float comb[HC * HC];
for (int j = 0; j < HC; ++j) {
  float mj = mixes[t * MIX + j] * rsq;
  pre_v[j] = 1.0f / (1.0f + metal::exp(-(mj * scale[0] + base[j]))) + heps;
  float mp = mixes[t * MIX + HC + j] * rsq;
  post_v[j] = 2.0f / (1.0f + metal::exp(-(mp * scale[1] + base[HC + j])));
}
for (int j = 0; j < HC * HC; ++j) {
  comb[j] = mixes[t * MIX + 2 * HC + j] * rsq * scale[2] + base[2 * HC + j];
}
for (int j = 0; j < HC; ++j) {
  float mx = comb[j * HC];
  for (int q = 1; q < HC; ++q) mx = metal::max(mx, comb[j * HC + q]);
  float sum = 0.0f;
  for (int q = 0; q < HC; ++q) {
    comb[j * HC + q] = metal::exp(comb[j * HC + q] - mx);
    sum += comb[j * HC + q];
  }
  for (int q = 0; q < HC; ++q) comb[j * HC + q] = comb[j * HC + q] / sum + heps;
}
for (int it = 0; it < ITERS; ++it) {
  if (it > 0) {
    for (int j = 0; j < HC; ++j) {
      float sum = 0.0f;
      for (int q = 0; q < HC; ++q) sum += comb[j * HC + q];
      for (int q = 0; q < HC; ++q) comb[j * HC + q] /= (sum + heps);
    }
  }
  for (int q = 0; q < HC; ++q) {
    float sum = 0.0f;
    for (int j = 0; j < HC; ++j) sum += comb[j * HC + q];
    for (int j = 0; j < HC; ++j) comb[j * HC + q] /= (sum + heps);
  }
}
for (int j = 0; j < HC; ++j) {
  out[t * PACK + j] = pre_v[j];
  out[t * PACK + HC + j] = post_v[j];
}
for (int j = 0; j < HC; ++j) {
  for (int q = 0; q < HC; ++q) out[t * PACK + 2 * HC + q * HC + j] = comb[j * HC + q];
}