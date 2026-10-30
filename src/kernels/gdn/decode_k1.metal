constexpr int NSG = NT / 32;
constexpr int RB = DV / SPLIT;       // dv rows per threadgroup
constexpr int R = RB / NSG;          // dv rows per simdgroup
constexpr int GRP = HV / HK;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint hv = threadgroup_position_in_grid.x / SPLIT;
uint part = threadgroup_position_in_grid.x % SPLIT;
uint hk = hv / GRP;
threadgroup float qs[DK], ks[DK], vs[DV];
threadgroup float gb[2];
uint row0 = part * RB + sg * R;
float st[R][4];
for (int j = 0; j < R; ++j) {
  uint base = (hv * DV + row0 + j) * DK + lane * 4;
  for (int i = 0; i < 4; ++i) st[j][i] = float(state_in[base + i]);
}
if (sg < 3) {
  uint cb = sg == 0 ? hk * DK : (sg == 1 ? HK * DK + hk * DK : 2 * HK * DK + hv * DV);
  T act[4];
  float sumsq = 0.0f;
  for (int i = 0; i < 4; ++i) {
    uint ch = cb + lane * 4 + i;
    float acc = 0.0f;
    for (int tap = 0; tap < 3; ++tap) acc += float(conv_state[tap * C + ch]) * float(conv_w[ch * 4 + tap]);
    acc += float(qkv[ch]) * float(conv_w[ch * 4 + 3]);
    const T conv = T(acc);
    T sy = T(1) / (T(1) + metal::exp(metal::abs(conv))); T sig = conv < T(0) ? sy : T(1) - sy;
    act[i] = conv * sig;
    float v = float(act[i]);
    sumsq += v * v;
  }
  if (sg < 2) {
    sumsq = simd_sum(sumsq);
    float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
    const T scale = sg == 0 ? q_scale : k_scale;
    threadgroup float* dst = sg == 0 ? qs : ks;
    for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = float(scale * T(1) * T(float(act[i]) * inv));
  } else {
    for (int i = 0; i < 4; ++i) vs[lane * 4 + i] = float(act[i]);
  }
  if (part == 0 && (sg == 2 || hv % GRP == 0)) {
    for (int i = 0; i < 4; ++i) {
      uint ch = cb + lane * 4 + i;
      conv_out[ch] = conv_state[C + ch];
      conv_out[C + ch] = conv_state[2 * C + ch];
      conv_out[2 * C + ch] = qkv[ch];
    }
  }
}
if (sg == (NSG > 3 ? 3 : 0) && lane == 31) {
  const T bv = b_in[hv];
  T by = T(1) / (T(1) + metal::exp(metal::abs(bv))); T bsig = bv < T(0) ? by : T(1) - by;
  gb[1] = float(bsig);
  const GT apd = GT(float(a_in[hv]) + float(dt_bias[hv]));
  float sp = msv_log1p(metal::precise::exp(float(apd)));
  float ea = metal::precise::exp(float(A_log[hv]));
  gb[0] = float(GT(metal::precise::exp(-(ea * sp))));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float kk[4], qq[4];
for (int i = 0; i < 4; ++i) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; }
const float g = gb[0], beta = gb[1];
for (int j = 0; j < R; ++j) {
  uint dv = row0 + j;
  float kv_mem = 0.0f;
  for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] * g; kv_mem += st[j][i] * kk[i]; }
  kv_mem = simd_sum(kv_mem);
  float delta = (vs[dv] - kv_mem) * beta;
  float out = 0.0f;
  for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] + kk[i] * delta; out += st[j][i] * qq[i]; }
  out = simd_sum(out);
  uint base = (hv * DV + dv) * DK + lane * 4;
  for (int i = 0; i < 4; ++i) state_out[base + i] = static_cast<StT>(st[j][i]);
  if (lane == 0) y[hv * DV + dv] = static_cast<T>(out);
}