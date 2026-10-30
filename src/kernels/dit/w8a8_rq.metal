const int K = x_shape[1];
const uint row = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int nch = K / 256;
threadgroup float red[8];
float v[CH][8];
float amax = 0.0f;
const device T* xr = x + (size_t)row * K;
for (int c = 0; c < CH; c++) {
  const int ch = int(sg) + 8 * c;
  if (ch >= nch) break;
  const int base = ch * 256 + int(lane) * 8;
  for (int p = 0; p < 8; p++) v[c][p] = float(xr[base + p]) * float(signs[base + p]);
  for (int h = 1; h < 8; h <<= 1)
    for (int p = 0; p < 8; p++) if ((p & h) == 0) { float a = v[c][p], b = v[c][p + h]; v[c][p] = a + b; v[c][p + h] = a - b; }
  for (uint m = 1; m < 32; m <<= 1) {
    const float sgn = (lane & m) ? -1.0f : 1.0f;
    for (int p = 0; p < 8; p++) v[c][p] = fma(sgn, v[c][p], simd_shuffle_xor(v[c][p], m));
  }
  for (int p = 0; p < 8; p++) { v[c][p] *= 0.0625f; amax = max(amax, abs(v[c][p])); }
}
amax = simd_max(amax);
if (lane == 0) red[sg] = amax;
threadgroup_barrier(mem_flags::mem_threadgroup);
amax = red[0];
for (int i = 1; i < 8; i++) amax = max(amax, red[i]);
const float sc = max(amax / 127.0f, 1.0e-20f);
const float inv = 1.0f / sc;
device int8_t* qr = xq + (size_t)row * K;
for (int c = 0; c < CH; c++) {
  const int ch = int(sg) + 8 * c;
  if (ch >= nch) break;
  const int base = ch * 256 + int(lane) * 8;
  char4 lo, hi;
  for (int p = 0; p < 4; p++) {
    lo[p] = char(clamp(rint(v[c][p] * inv), -127.0f, 127.0f));
    hi[p] = char(clamp(rint(v[c][p + 4] * inv), -127.0f, 127.0f));
  }
  *(device char4*)(qr + base) = lo;
  *(device char4*)(qr + base + 4) = hi;
}
if (sg == 0 && lane == 0) xs[row] = sc;