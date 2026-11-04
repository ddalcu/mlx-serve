constexpr int NSG = NT / 32;
constexpr int RB = DV / SPLIT;
constexpr int R = RB / NSG;
constexpr int GRP = HV / HK;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint hv = threadgroup_position_in_grid.x / SPLIT;
uint part = threadgroup_position_in_grid.x % SPLIT;
uint hk = hv / GRP;
threadgroup float qs[TL][DK], ks[TL][DK], vs[TL][DV];
threadgroup float gb[TL][2];
uint row0 = part * RB + sg * R;
float st[R][4];
for (int j = 0; j < R; ++j) {
  uint base = (hv * DV + row0 + j) * DK + lane * 4;
  for (int i = 0; i < 4; ++i) st[j][i] = float(state_in[base + i]);
}
// one (q | k | v, token) pair per simdgroup at a time: a pair's arithmetic does not depend on which one runs it
for (int pw = int(sg); pw < 3 * TL; pw += NSG) {
  const int comp = pw % 3, t = pw / 3;
  uint cb = comp == 0 ? hk * DK : (comp == 1 ? HK * DK + hk * DK : 2 * HK * DK + hv * DV);
  {
    T act[4];
    float sumsq = 0.0f;
    for (int i = 0; i < 4; ++i) {
      uint ch = cb + lane * 4 + i;
      float acc = 0.0f;
      for (int tap = 0; tap < 4; ++tap) {
        const int w = t + tap;
        const T xv = w < 3 ? conv_state[w * C + ch] : qkv[(w - 3) * C + ch];
        acc += float(xv) * float(conv_w[ch * 4 + tap]);
      }
      const T conv = T(acc);
      T sy = T(1) / (T(1) + metal::exp(metal::abs(conv))); T sig = conv < T(0) ? sy : T(1) - sy;
      act[i] = conv * sig;
      float v = float(act[i]);
      sumsq += v * v;
    }
    if (comp < 2) {
      sumsq = simd_sum(sumsq);
      float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
      const T scale = comp == 0 ? q_scale : k_scale;
      threadgroup float* dst = comp == 0 ? qs[t] : ks[t];
      for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = float(scale * T(1) * T(float(act[i]) * inv));
    } else {
      for (int i = 0; i < 4; ++i) vs[t][lane * 4 + i] = float(act[i]);
    }
  }
}
if (sg < 3) {
  uint cb = sg == 0 ? hk * DK : (sg == 1 ? HK * DK + hk * DK : 2 * HK * DK + hv * DV);
  if (part == 0 && (sg == 2 || hv % GRP == 0)) {
    for (int i = 0; i < 4; ++i) {
      uint ch = cb + lane * 4 + i;
      for (int j = 0; j < 3; ++j) {
        int w = TL + j;
        conv_out[j * C + ch] = w < 3 ? conv_state[w * C + ch] : qkv[(w - 3) * C + ch];
      }
    }
  }
}
if (sg == (NSG > 3 ? 3 : 0) && lane == 31) {
  float ea = metal::precise::exp(float(A_log[hv]));
  for (int t = 0; t < TL; ++t) {
    const T bv = b_in[t * HV + hv];
    T by = T(1) / (T(1) + metal::exp(metal::abs(bv))); T bsig = bv < T(0) ? by : T(1) - by;
    gb[t][1] = float(bsig);
    const GT apd = GT(float(a_in[t * HV + hv]) + float(dt_bias[hv]));
    float sp = msv_log1p(metal::precise::exp(float(apd)));
    gb[t][0] = float(GT(metal::precise::exp(-(ea * sp))));
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int t = 0; t < TL; ++t) {
  float kk[4], qq[4];
  for (int i = 0; i < 4; ++i) { kk[i] = ks[t][lane * 4 + i]; qq[i] = qs[t][lane * 4 + i]; }
  const float g = gb[t][0], beta = gb[t][1];
  for (int j = 0; j < R; ++j) {
    uint dv = row0 + j;
    float kv_mem = 0.0f;
    for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] * g; kv_mem += st[j][i] * kk[i]; }
    kv_mem = simd_sum(kv_mem);
    float delta = (vs[t][dv] - kv_mem) * beta;
    float out = 0.0f;
    for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] + kk[i] * delta; out += st[j][i] * qq[i]; }
    out = simd_sum(out);