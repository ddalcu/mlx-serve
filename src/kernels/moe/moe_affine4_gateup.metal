const uint lane = thread_index_in_simdgroup;
const int row0 = int(thread_position_in_grid.y) * ROWS;
const uint e = thread_position_in_grid.z;
constexpr int KW2 = K / 16;
constexpr int KG = K / GS;
const bool sh = int(e) >= TOPK;
const size_t row = sh ? size_t(int(e) - TOPK) * N + row0 : size_t(inds[e]) * N + row0;
const device uint2* gw = (const device uint2*)(sh ? sg_q : wg_q) + row * KW2 + lane;
const device uint2* uw = (const device uint2*)(sh ? su_q : wu_q) + row * KW2 + lane;
const device T* gsc = sh ? sg_scales : g_scales;
const device T* gbi = sh ? sg_biases : g_biases;
const device T* usc = sh ? su_scales : u_scales;
const device T* ubi = sh ? su_biases : u_biases;
float ag[ROWS] = {0.0f};
float au[ROWS] = {0.0f};
for (int k = 0; k < K; k += 512) {
  const int kk = k + int(lane) * 16;
  if (kk < K) {
    float xr[16];
    float xs = 0.0f;
    for (int i = 0; i < 16; ++i) { xr[i] = float(x[kk + i]); xs += xr[i]; }
    const int gi = kk / GS;
    for (int r = 0; r < ROWS; ++r) {
      const size_t sg = (row + r) * KG + gi;
      ag[r] += float(gsc[sg]) * mlxserve_a4_dot16(gw[r * KW2 + k / 16], xr) + float(gbi[sg]) * xs;
      au[r] += float(usc[sg]) * mlxserve_a4_dot16(uw[r * KW2 + k / 16], xr) + float(ubi[sg]) * xs;
    }
  }
}
for (int r = 0; r < ROWS; ++r) {
  const float g = simd_sum(ag[r]);
  const float u = simd_sum(au[r]);
  if (lane == 0) {
    const T gt = T(g);
    y[size_t(e) * N + row0 + r] = (gt * sigtab[as_type<ushort>(gt)]) * T(u);
  }
}