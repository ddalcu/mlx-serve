const int K = x_shape[1];
const int Kg = K / 128;
const uint row = threadgroup_position_in_grid.y;
const uint g = threadgroup_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;
// Rows past the input's are the tile's padding: they quantize as zeros.
const bool live = int(row) < x_shape[0];
const device T* xr = x + (size_t)(live ? row : 0) * K + g * 128;
float v[4];
float a = 0.0f;
for (int i = 0; i < 4; ++i) { v[i] = live ? float(xr[lane * 4 + i]) : 0.0f; a = max(a, abs(v[i])); }
a = simd_max(a);
const float sc = max(a / 127.0f, 1.0e-20f);
float rs = 0.0f;
for (int i = 0; i < 4; ++i) {
  const float q = clamp(rint(v[i] / sc), -127.0f, 127.0f);
  rs += q;
  // Kernel K order: position 4b + kq of a 16-block is stored at 4kq + b.
  const uint p = lane * 4 + i;
  xq[(size_t)row * K + g * 128 + (p & ~15u) + (p & 3u) * 4 + ((p >> 2) & 3u)] = (int8_t)q;
}
rs = simd_sum(rs);
if (lane == 0) {
  ascale[(size_t)row * Kg + g] = sc;
  rsum[(size_t)row * Kg + g] = rs;
}