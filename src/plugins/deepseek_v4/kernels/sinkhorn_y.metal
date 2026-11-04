int t = thread_position_in_grid.x;
int j = thread_position_in_grid.y;
if (t >= S || j >= D) return;
const int PACK = 2 * HC + HC * HC;
const int MIX = (2 + HC) * HC;
threadgroup float tg_pre[HC];
if (thread_position_in_threadgroup.y == 0) {
  float rsq = metal::rsqrt(ssq[t] / consts[0] + consts[1]);
  const float heps = consts[2];
  float pre_v[HC];
  float post_v[HC];
  float comb[HC * HC];
  for (int c = 0; c < HC; ++c) {
    float mj = mixes[t * MIX + c] * rsq;
    pre_v[c] = 1.0f / (1.0f + metal::exp(-(mj * scale[0] + base[c]))) + heps;
    float mp = mixes[t * MIX + HC + c] * rsq;
    post_v[c] = 2.0f / (1.0f + metal::exp(-(mp * scale[1] + base[HC + c])));
  }
  for (int c = 0; c < HC * HC; ++c) {
    comb[c] = mixes[t * MIX + 2 * HC + c] * rsq * scale[2] + base[2 * HC + c];
  }
  for (int c = 0; c < HC; ++c) {
    float mx = comb[c * HC];
    for (int q = 1; q < HC; ++q) mx = metal::max(mx, comb[c * HC + q]);
    float sum = 0.0f;
    for (int q = 0; q < HC; ++q) {
      comb[c * HC + q] = metal::exp(comb[c * HC + q] - mx);
      sum += comb[c * HC + q];
    }
    for (int q = 0; q < HC; ++q) comb[c * HC + q] = comb[c * HC + q] / sum + heps;
  }
  for (int it = 0; it < ITERS; ++it) {
    if (it > 0) {
      for (int c = 0; c < HC; ++c) {
        float sum = 0.0f;
        for (int q = 0; q < HC; ++q) sum += comb[c * HC + q];
        for (int q = 0; q < HC; ++q) comb[c * HC + q] /= (sum + heps);
      }
    }
    for (int q = 0; q < HC; ++q) {
      float sum = 0.0f;
      for (int c = 0; c < HC; ++c) sum += comb[c * HC + q];
      for (int c = 0; c < HC; ++c) comb[c * HC + q] /= (sum + heps);
    }
  }
  for (int c = 0; c < HC; ++c) tg_pre[c] = pre_v[c];
  if (j == 0) {
    for (int c = 0; c < HC; ++c) {
      out[t * PACK + c] = pre_v[c];
      out[t * PACK + HC + c] = post_v[c];
    }
    for (int c = 0; c < HC; ++c) {
      for (int q = 0; q < HC; ++q) out[t * PACK + 2 * HC + q * HC + c] = comb[c * HC + q];
    }
  }
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
float acc = 0.0f;
for (int c = 0; c < HC; ++c) {
  float p = tg_pre[c] * stream_in[(t * HC + c) * D + j];
  acc = acc + p;
}
y[t * D + j] = acc;