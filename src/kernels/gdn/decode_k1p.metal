constexpr int GRP = HV / HK;
uint lane = thread_index_in_simdgroup;
uint head = threadgroup_position_in_grid.y;
uint t = threadgroup_position_in_grid.z;
const int comp = head < HK ? 0 : (head < 2 * HK ? 1 : 2);
uint h = comp == 0 ? head : (comp == 1 ? head - HK : head - 2 * HK);
uint cb = comp == 0 ? h * DK : (comp == 1 ? HK * DK + h * DK : 2 * HK * DK + h * DV);
T act[4];
float sumsq = 0.0f;
for (int i = 0; i < 4; ++i) {
  uint ch = cb + lane * 4 + i;
  float acc = 0.0f;
  for (int tap = 0; tap < 4; ++tap) {
    const int w = parents[TL + t * 4 + tap];
    const T xv = w < 3 ? conv_state[w * C + ch] : qkv[(w - 3) * QS + ch];
    acc += float(xv) * float(conv_w[ch * 4 + tap]);
  }
  const T conv = T(acc);
  T sy = T(1) / (T(1) + metal::exp(metal::abs(conv))); T sig = conv < T(0) ? sy : T(1) - sy;
  act[i] = conv * sig;
  float v = float(act[i]);
  sumsq += v * v;
  conv_in[(3 + t) * C + ch] = qkv[t * QS + ch];
  for (int r = t; r < 3; r += TL) conv_in[r * C + ch] = conv_state[r * C + ch];
}
if (comp < 2) {
  sumsq = simd_sum(sumsq);
  float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
  const T scale = comp == 0 ? q_scale : k_scale;
  device float* dst = (comp == 0 ? pq : pk) + (t * HK + h) * DK;
  for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = float(scale * T(1) * T(float(act[i]) * inv));
} else {
  for (int i = 0; i < 4; ++i) pv[(t * HV + h) * DV + lane * 4 + i] = float(act[i]);
  if (lane == 0) {
    float ea = metal::precise::exp(float(A_log[h]));
    const T bv = b_in[t * ABS + BOFF + h];
    T by = T(1) / (T(1) + metal::exp(metal::abs(bv))); T bsig = bv < T(0) ? by : T(1) - by;
    pg[(t * HV + h) * 2 + 1] = float(bsig);
    const GT apd = GT(float(a_in[t * ABS + AOFF + h]) + float(dt_bias[h]));
    float sp = msv_log1p(metal::precise::exp(float(apd)));
    pg[(t * HV + h) * 2] = float(GT(metal::precise::exp(-(ea * sp))));
  }
}