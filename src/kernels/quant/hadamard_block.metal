constexpr int S = 4;
constexpr int P = J / S;
threadgroup float tg[J * 32];
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint base = threadgroup_position_in_grid.x * (J * 32);
uint col = base % x_shape[x_ndim - 1];
float v[P];
for (int p = 0; p < P; ++p) {
  uint i = (sg * P + p) * 32 + lane;
  float e = float(x[base + i]);
  v[p] = INV ? e : e * signs[col + i];
}
for (int h = 1; h < P; h <<= 1) {
  for (int p = 0; p < P; ++p) {
    if ((p & h) == 0) { float a = v[p], b = v[p + h]; v[p] = a + b; v[p + h] = a - b; }
  }
}
for (uint m = 1; m < 32; m <<= 1) {
  float sgn = (lane & m) ? -1.0f : 1.0f;
  for (int p = 0; p < P; ++p) v[p] = fma(sgn, v[p], simd_shuffle_xor(v[p], m));
}
for (int p = 0; p < P; ++p) tg[(sg * P + p) * 32 + lane] = v[p];
threadgroup_barrier(mem_flags::mem_threadgroup);
const float scale = rsqrt(float(J * 32));
for (uint p = sg; p < P; p += S) {
  float w[S];
  for (int q = 0; q < S; ++q) w[q] = tg[(q * P + p) * 32 + lane];
  float a0 = w[0] + w[1], a1 = w[0] - w[1], a2 = w[2] + w[3], a3 = w[2] - w[3];
  w[0] = a0 + a2; w[2] = a0 - a2; w[1] = a1 + a3; w[3] = a1 - a3;
  for (int q = 0; q < S; ++q) {
    uint i = (q * P + p) * 32 + lane;
    float e = w[q] * scale;
    if (INV) e *= signs[col + i];
    y[base + i] = static_cast<T>(e);
  }
}