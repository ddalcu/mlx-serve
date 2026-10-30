if (sg < 3 && (sg == 2 || hv % GRP == 0)) {
  uint cb = sg == 0 ? hk * DK : (sg == 1 ? HK * DK + hk * DK : 2 * HK * DK + hv * DV);
  for (int i = 0; i < 4; ++i) {
    uint ch = cb + lane * 4 + i;
    for (int w = 0; w < 3 + TL; ++w) conv_in[w * C + ch] = w < 3 ? conv_state[w * C + ch] : qkv[(w - 3) * C + ch];
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg < TL) {
  float xs[4];
  float sumsq = 0.0f;
  for (int i = 0; i < 4; ++i) { xs[i] = ys[sg][lane * 4 + i]; sumsq += xs[i] * xs[i]; }
  sumsq = simd_sum(sumsq);
  float inv = metal::precise::rsqrt(sumsq / float(DV) + eps);
  uint base = (sg * HV + hv) * DV + lane * 4;
  for (int i = 0; i < 4; ++i) {
    const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
    const T zv = z[base + i];
    T sy = T(1) / (T(1) + metal::exp(metal::abs(zv))); T sig = zv < T(0) ? sy : T(1) - sy;
    gated[base + i] = SWISH ? (zv * sig) * normed : normed * sig;
  }
}