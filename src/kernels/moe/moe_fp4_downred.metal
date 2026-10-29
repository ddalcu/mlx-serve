const uint lane = thread_index_in_simdgroup;
const int row0 = int(thread_position_in_grid.y) * ROWS;
const int tok = int(thread_position_in_grid.z);
constexpr int KW2 = I / 16;
constexpr int KG = I / GS;
float acc[ROWS] = {0.0f};
for (int e = tok * TOPK; e < (tok + 1) * TOPK; ++e) {
  const size_t row = size_t(inds[e]) * H + row0;
  const device uint2* dw = (const device uint2*)wd_q + row * KW2 + lane;
  const float sc = float(scores[e]);
  for (int k = 0; k < I; k += 512) {
    const int kk = k + int(lane) * 16;
    if (kk < I) {
      float xr[16];
      for (int i = 0; i < 16; ++i) xr[i] = float(act[size_t(e) * I + kk + i]);
      const int gi = kk / GS;
      for (int r = 0; r < ROWS; ++r)
        acc[r] += sc * mlxserve_fp4_dot16(dw[r * KW2 + k / 16], xr) * mlxserve_fp4_scale(uint(d_scales[(row + r) * KG + gi]));
    }
  }
}
for (int r = 0; r < ROWS; ++r) {
  const float v = simd_sum(acc[r]) * 4194304.0f;
  if (lane == 0) {
    T out = T(v);
    if (SHARED != 0) {
      // The gated shared expert joins as `moe_affine4` adds it: sigmoid in T, product and sum rounded to T.
      const T gl = glog[0];
      auto sy = 1 / (1 + metal::precise::exp(metal::abs(gl)));
      const T sig = (gl < 0) ? sy : 1 - sy;
      out = T(float(out) + float(T(float(sig) * float(sdown[row0 + r]))));
    }
    y[size_t(tok) * H + row0 + r] = out;
  }
}