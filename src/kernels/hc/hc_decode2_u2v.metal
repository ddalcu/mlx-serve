const uint lane = thread_index_in_simdgroup;
const uint tid = thread_index_in_threadgroup;
const uint sgi = simdgroup_index_in_threadgroup;
const uint part = lane & 3u;
const uint h = (lane >> 2) & 3u;
const uint cl = lane >> 4;
const uint j = threadgroup_position_in_grid.x * uint(SGS * 2) + sgi * 2u + cl;
constexpr int VPW = 32 / BITS;
constexpr int R_by_p = R / VPW;
constexpr int R_by_gs = R / GS;
constexpr int PV = R / 4;
constexpr int PU4 = PV / (4 * VPW);
constexpr int PR = R + HC;
threadgroup float act_tg[ROWS * R];
const size_t wrow = (size_t)h * (size_t)H + (size_t)j;
const device uint4* wq4 = (const device uint4*)(uw_q + wrow * (size_t)R_by_p) + part * PU4;
uint4 wv[PU4];
for (int i = 0; i < PU4; ++i) wv[i] = wq4[i];
float sc[PU4], bi[PU4];
for (int i = 0; i < PU4; ++i) {
  const int k0 = int(part) * PV + i * 4 * VPW;
  sc[i] = float(uw_s[wrow * (size_t)R_by_gs + (size_t)(k0 / GS)]);
  bi[i] = float(uw_b[wrow * (size_t)R_by_gs + (size_t)(k0 / GS)]);
}
float xnv[ROWS];
for (int r = 0; r < ROWS; ++r) xnv[r] = float(xn_in[(size_t)r * (size_t)(HC * H) + wrow]);
for (uint e = tid; e < uint(ROWS * R); e += uint(SGS * 32)) {
  const int r = int(e) / R;
  const int k = int(e) % R;
  float tsum = 0.0f;
  for (int s = 0; s < HC; ++s) tsum += parts_in[(size_t)(r * HC + s) * PR + k];
  const T v = T(tsum);
  const T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
  act_tg[e] = float(v * sig);
}
if (INJ != 0 && threadgroup_position_in_grid.x == 0 && tid < uint(ROWS * HC)) {
  const int r = int(tid) / HC;
  const int c = int(tid) % HC;
  float tsum = 0.0f;
  for (int s = 0; s < HC; ++s) tsum += parts_in[(size_t)(r * HC + s) * PR + R + c];
  const T v = T(tsum);
  const T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
  inj_out[tid] = sig * T(2.0f);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float acc[ROWS];
for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
for (int i = 0; i < PU4; ++i) {
  const int k0 = int(part) * PV + i * 4 * VPW;
  const uint words[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
  for (int wd = 0; wd < 4; ++wd) {
    for (int b = 0; b < VPW; ++b) {
      const float wf = float((words[wd] >> (b * BITS)) & ((1u << BITS) - 1u)) * sc[i] + bi[i];
      const int k = k0 + wd * VPW + b;
      for (int r = 0; r < ROWS; ++r) acc[r] += act_tg[r * R + k] * wf;
    }
  }
}
float mix[ROWS];
for (int r = 0; r < ROWS; ++r) {
  float a = acc[r];
  a += simd_shuffle_xor(a, 1);
  a += simd_shuffle_xor(a, 2);
  const T u = T(a);
  const T sgv = T(1.0f / (1.0f + metal::exp(-float(u))));
  float m = float(T(float(sgv) * xnv[r]));
  m += simd_shuffle_xor(m, 4);
  m += simd_shuffle_xor(m, 8);
  mix[r] = m;
}
if ((lane & 15u) == 0) {
  for (int r = 0; r < ROWS; ++r) mixed_out[(size_t)r * (size_t)H + (size_t)j] = T(float(T(mix[r])) * float(T(1.0f / float(HC))));
}