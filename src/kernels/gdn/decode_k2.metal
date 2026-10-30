constexpr int S = 4, P = J / S, HPS = J * 32 / (DV * S);
threadgroup float tg[J * 32];
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint base = threadgroup_position_in_grid.x * (J * 32);
for (int hh = 0; hh < HPS; ++hh) {
  uint hb = base + (sg * HPS + hh) * DV;
  float xs[4];
  float sumsq = 0.0f;
  for (int i = 0; i < 4; ++i) { xs[i] = float(y[hb + lane * 4 + i]); sumsq += xs[i] * xs[i]; }
  sumsq = simd_sum(sumsq);
  float inv = metal::precise::rsqrt(sumsq / float(DV) + eps);
  for (int i = 0; i < 4; ++i) {
    const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
    const T zv = z[hb + lane * 4 + i];
    T sy = T(1) / (T(1) + metal::exp(metal::abs(zv))); T sig = zv < T(0) ? sy : T(1) - sy;
    tg[hb - base + lane * 4 + i] = float((zv * sig) * normed) * signs[hb + lane * 4 + i];
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float v[P];
for (int p = 0; p < P; ++p) v[p] = tg[(sg * P + p) * 32 + lane];
for (int h = 1; h < P; h <<= 1)
  for (int p = 0; p < P; ++p)
    if ((p & h) == 0) { float a = v[p], b = v[p + h]; v[p] = a + b; v[p + h] = a - b; }
for (uint m = 1; m < 32; m <<= 1) {
  float sgn = (lane & m) ? -1.0f : 1.0f;
  for (int p = 0; p < P; ++p) v[p] = fma(sgn, v[p], simd_shuffle_xor(v[p], m));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int p = 0; p < P; ++p) tg[(sg * P + p) * 32 + lane] = v[p];
threadgroup_barrier(mem_flags::mem_threadgroup);
const float scale = rsqrt(float(J * 32));
for (uint p = sg; p < P; p += S) {
  float w[S];
  for (int q = 0; q < S; ++q) w[q] = tg[(q * P + p) * 32 + lane];
  float a0 = w[0] + w[1], a1 = w[0] - w[1], a2 = w[2] + w[3], a3 = w[2] - w[3];
  w[0] = a0 + a2; w[2] = a0 - a2; w[1] = a1 + a3; w[3] = a1 - a3;
  for (int q = 0; q < S; ++q) rot[base + (q * P + p) * 32 + lane] = static_cast<T>(w[q] * scale);
}