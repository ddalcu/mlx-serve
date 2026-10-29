constexpr int GRP = HV / HK;
uint lane = thread_index_in_simdgroup;
uint dv = thread_position_in_grid.y;
uint hv = thread_position_in_grid.z;
uint hk = hv / GRP;
const uint tid = (hv * DV + dv) * 32 + lane;
if (tid < uint(3 * C)) conv_out[tid] = conv_in[conv_rows[tid / C] * C + tid % C];
float st[4];
uint base = (hv * DV + dv) * DK + lane * 4;
for (int i = 0; i < 4; ++i) st[i] = float(state_in[base + i]);
for (int p = 0; p < int(n_path[0]); ++p) {
  const int t = path[p];
  float kk[4];
  for (int i = 0; i < 4; ++i) kk[i] = pk[(t * HK + hk) * DK + lane * 4 + i];
  const float g = pg[(t * HV + hv) * 2], beta = pg[(t * HV + hv) * 2 + 1];
  float kv_mem = 0.0f;
  for (int i = 0; i < 4; ++i) { st[i] = st[i] * g; kv_mem += st[i] * kk[i]; }
  kv_mem = simd_sum(kv_mem);
  float delta = (pv[(t * HV + hv) * DV + dv] - kv_mem) * beta;
  for (int i = 0; i < 4; ++i) st[i] = st[i] + kk[i] * delta;
  for (int i = 0; i < 4; ++i) st[i] = float(static_cast<StT>(st[i]));
}
for (int i = 0; i < 4; ++i) state_out[base + i] = static_cast<StT>(st[i]);