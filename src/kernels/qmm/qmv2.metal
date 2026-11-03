const int K = x_shape[x_ndim - 1];
const int KW = K / 16, KG = K / 128;
uint lane = thread_index_in_simdgroup;
int row0 = (threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup) * 4;
const device T* xp = x + lane * 16;
const device uint* wq = w + row0 * KW + lane;
const device T* sp = scales + row0 * KG + lane / 8;
const device T* bp = biases + row0 * KG + lane / 8;
float res[4] = {0.f, 0.f, 0.f, 0.f};
for (int k = 0; k < K; k += 512) {
  float2 xf[8];
  float sm = 0.f;
  for (int i = 0; i < 8; ++i) {
    xf[i] = float2(float(xp[i]), float(xp[i + 8]));
    sm += xf[i].x + xf[i].y;
  }
  for (int r = 0; r < 4; ++r) {
    half2 q[8];
    h2dec(wq[r * KW], q, 0.0h);
    res[r] += float(sp[r * KG]) * f2dot(q, xf) + float(bp[r * KG]) * sm;
  }
  wq += 32; sp += 4; bp += 4; xp += 512;
}
for (int r = 0; r < 4; ++r) {
  float v = simd_sum(res[r]);
  if (lane == 0) y[row0 + r] = static_cast<T>(v);
}