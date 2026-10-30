const uint i = thread_position_in_grid.x;
const int D = x_shape[x_ndim - 1];
const int c = i % D, row = i / ROW;
const int head = D - RD;
float r = float(x[i]);
if (c >= head) {
  const int p = (c - head) / 2;
  const size_t b = i - (c - head) % 2;
  const float x1 = float(x[b]), x2 = float(x[b + 1]);
  const float theta = scal[0] * float(row + off) * (1.0f / periods[p]);
  const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
  r = float(T((c - head) % 2 == 0 ? x1 * cs - x2 * sn : x1 * sn + x2 * cs));
}
if (QK != 0) {
  float amax = metal::abs(r);
  for (ushort m = (QK == 3 ? 8 : 16); m > 0; m >>= 1) amax = metal::max(amax, metal::simd_shuffle_xor(amax, m));
  float scale;
  if (QK == 1) scale = dsv41_pow2ceil(metal::max(amax, 1e-4f) * (1.0f / 448.0f));
  else if (QK == 2) scale = dsv41_pow2ceil(metal::max(amax, 6.0f * 1.17549435e-38f) * (1.0f / 6.0f));
  else scale = dsv41_fp8_roundtrip(metal::max(amax, 6.0f * 0.001953125f) / 6.0f);
  r = QK == 1 ? dsv41_fp8_roundtrip(metal::clamp(r / scale, -448.0f, 448.0f)) * scale
             : dsv41_e2m1(metal::clamp(r / scale, -6.0f, 6.0f)) * scale;
}
out[i] = OT(T(r));