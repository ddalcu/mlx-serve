constexpr int S = 4;
constexpr int P = 8;
threadgroup float tg[1024];
const int K = x_shape[1];
const int Kg = K / 128;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint col = threadgroup_position_in_grid.x * 1024;
const uint row = threadgroup_position_in_grid.y;
const bool live = int(row) < x_shape[0];
const device T* xr = x + (size_t)(live ? row : 0) * K + col;
float v[P];
for (int p = 0; p < P; ++p) {
  const uint i = (sg * P + p) * 32 + lane;
  v[p] = live ? float(xr[i]) * float(signs[col + i]) : 0.0f;
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
const float scale = rsqrt(1024.0f);
float o[2][S];
for (int t = 0; t < 2; ++t) {
  const uint p = sg + S * t;
  float w[S];
  for (int q = 0; q < S; ++q) w[q] = tg[(q * P + p) * 32 + lane];
  float a0 = w[0] + w[1], a1 = w[0] - w[1], a2 = w[2] + w[3], a3 = w[2] - w[3];
  w[0] = a0 + a2; w[2] = a0 - a2; w[1] = a1 + a3; w[3] = a1 - a3;
  for (int q = 0; q < S; ++q) o[t][q] = float(static_cast<T>(w[q] * scale));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int t = 0; t < 2; ++t)
  for (int q = 0; q < S; ++q) tg[(q * P + sg + S * t) * 32 + lane] = o[t][q];
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int gg = 0; gg < 2; ++gg) {
  const uint gl = sg * 2 + gg;
  float e[4];
  float a = 0.0f;
  for (int i = 0; i < 4; ++i) { e[i] = tg[gl * 128 + lane * 4 + i]; a = max(a, abs(e[i])); }
  a = simd_max(a);
  const float sc = max(a / 127.0f, 1.0e-20f);
  float rs = 0.0f;
  const size_t gb = (size_t)row * K + col + gl * 128;
  for (int i = 0; i < 4; ++i) {
    const float q = clamp(rint(e[i] / sc), -127.0f, 127.0f);
    rs += q;
    const uint p = lane * 4 + i;
    xq[gb + (p & ~15u) + (p & 3u) * 4 + ((p >> 2) & 3u)] = (int8_t)q;
  }
  rs = simd_sum(rs);
  if (lane == 0) {
    const size_t gi = (size_t)row * Kg + col / 128 + gl;
    ascale[gi] = sc;
    rsum[gi] = rs;
  }
}