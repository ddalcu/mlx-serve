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
  for (ushort m = (QK == 3 ? 8 : 16); m > 0; m >>= 1) amax = mlxserve_maximum(amax, metal::simd_shuffle_xor(amax, m));
  float scale;
  if (QK == 1) scale = mlxserve_mx_scale_round_up(mlxserve_maximum(amax, 1e-4f) * (1.0f / 448.0f));
  else if (QK == 2) scale = mlxserve_mx_scale_round_up(mlxserve_maximum(amax, 6.0f * 1.17549435e-38f) * (1.0f / 6.0f));
  else scale = mlxserve_fp8_e4m3_round(mlxserve_maximum(amax, 6.0f * 0.001953125f) / 6.0f);
  r = QK == 1 ? mlxserve_fp8_e4m3_round(mlxserve_clip(r / scale, -448.0f, 448.0f)) * scale
             : mlxserve_fp4_e2m1_round(mlxserve_clip(r / scale, -6.0f, 6.0f)) * scale;
}
out[i] = OT(T(r));