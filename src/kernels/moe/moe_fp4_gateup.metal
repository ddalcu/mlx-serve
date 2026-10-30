const uint lane = thread_index_in_simdgroup;
const int row0 = int(thread_position_in_grid.y) * ROWS;
const uint e = thread_position_in_grid.z;
const device auto* xt = x + size_t(e / TOPK) * K;
constexpr int KW2 = K / 16;
constexpr int KG = K / GS;
const size_t row = size_t(inds[e]) * N + row0;
const device uint2* gw = (const device uint2*)wg_q + row * KW2 + lane;
const device uint2* uw = (const device uint2*)wu_q + row * KW2 + lane;
float ag[ROWS] = {0.0f};
float au[ROWS] = {0.0f};
for (int k = 0; k < K; k += 512) {
  const int kk = k + int(lane) * 16;
  if (kk < K) {
    float xr[16];
    for (int i = 0; i < 16; ++i) xr[i] = float(xt[kk + i]);
    const int gi = kk / GS;
    for (int r = 0; r < ROWS; ++r) {
      const size_t sg = (row + r) * KG + gi;
      ag[r] += mlxserve_fp4_dot16(gw[r * KW2 + k / 16], xr) * mlxserve_fp4_scale(uint(g_scales[sg]));
      au[r] += mlxserve_fp4_dot16(uw[r * KW2 + k / 16], xr) * mlxserve_fp4_scale(uint(u_scales[sg]));
    }
  }
}
for (int r = 0; r < ROWS; ++r) {
  const float g = simd_sum(ag[r]) * 4194304.0f;
  const float u = simd_sum(au[r]) * 4194304.0f;
  if (lane == 0) {
    T gt = GSCALE != 0 ? T(T(g) * gsc[inds[e]]) : T(g);
    T ut = USCALE != 0 ? T(T(u) * usc[inds[e]]) : T(u);
    if (CLAMP != 0) {
      const float hi = float(T(float(CLAMP)));
      gt = T(min(float(gt), hi));
      ut = T(clamp(float(ut), -hi, hi));
    }
    y[size_t(e) * N + row0 + r] = (gt * sigtab[as_type<ushort>(gt)]) * ut;
  }
}