constexpr int GRP = HV / HK;
uint lane = thread_index_in_simdgroup;
uint dv = thread_position_in_grid.y;
uint hv = thread_position_in_grid.z;
uint hk = hv / GRP;
float st[4];
float kept[TL][4];
uint base = (hv * DV + dv) * DK + lane * 4;
for (int i = 0; i < 4; ++i) st[i] = float(state_in[base + i]);
// Row t + 1's inputs load while row t reduces.
float4 kn = *(const device float4*)(pk + hk * DK + lane * 4), qn = *(const device float4*)(pq + hk * DK + lane * 4);
float gn = pg[hv * 2], bn = pg[hv * 2 + 1], vn = pv[hv * DV + dv];
for (int t = 0; t < TL; ++t) {
  const float4 kk = kn, qq = qn;
  const float g = gn, beta = bn, vt = vn;
  if (t + 1 < TL) {
    kn = *(const device float4*)(pk + ((t + 1) * HK + hk) * DK + lane * 4);
    qn = *(const device float4*)(pq + ((t + 1) * HK + hk) * DK + lane * 4);
    gn = pg[((t + 1) * HV + hv) * 2]; bn = pg[((t + 1) * HV + hv) * 2 + 1];
    vn = pv[((t + 1) * HV + hv) * DV + dv];
  }
  const int p = parents[t];
  if (t > 0 && p != t - 1) for (int i = 0; i < 4; ++i) st[i] = kept[p][i];
  float kv_mem = 0.0f;
  for (int i = 0; i < 4; ++i) { st[i] = st[i] * g; kv_mem += st[i] * kk[i]; }
  kv_mem = simd_sum(kv_mem);
  float delta = (vt - kv_mem) * beta;
  float out = 0.0f;
  for (int i = 0; i < 4; ++i) { st[i] = st[i] + kk[i] * delta; out += st[i] * qq[i]; }
  out = simd_sum(out);
  if (lane == 0) y[(t * HV + hv) * DV + dv] = static_cast<T>(out);
  // The next token reads the state serial decoding stored (StT), not the f32.
  for (int i = 0; i < 4; ++i) st[i] = float(static_cast<StT>(st[i]));
  if (parents[5 * TL + t] != 0) for (int i = 0; i < 4; ++i) kept[t][i] = st[i];
}