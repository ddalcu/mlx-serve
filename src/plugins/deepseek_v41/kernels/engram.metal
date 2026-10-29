#pragma clang fp contract(off)
const uint t = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int TG = threads_per_threadgroup.x, j = threadgroup_position_in_grid.y, l = threadgroup_position_in_grid.z;
const int HC = qk_shape[0], D = qk_shape[1];
const size_t xo = (size_t(l) * HC + j) * D, ko = size_t(l) * (HC + 1) * D + size_t(j) * D, vo = size_t(l) * (HC + 1) * D + size_t(HC) * D;
threadgroup float part[3][32];
threadgroup float gate_s;
float sh = 0.0f, sk = 0.0f, sd = 0.0f;
const int blocks = D / (TG * 4), extra = D - blocks * TG * 4;
for (int b = 0; b <= blocks; ++b) {
  const int c0 = b * TG * 4 + int(t) * 4;
  const int n = b < blocks ? 4 : metal::clamp(extra - int(t) * 4, 0, 4);
  for (int i = 0; i < n; ++i) {
    const float hv = float(x[xo + c0 + i]), kk = float(kv[ko + c0 + i]);
    const float h2 = hv * hv, k2 = kk * kk, hq = hv * qk[size_t(j) * D + c0 + i];
    const float p = hq * kk;
    sh = h2 + sh;
    sk = k2 + sk;
    sd = p + sd;
  }
}
if (TG > 1) {
  sh = metal::simd_sum(sh);
  sk = metal::simd_sum(sk);
  sd = metal::simd_sum(sd);
}
if (TG > 32) {
  if (lane == 0) { part[0][sg] = sh; part[1][sg] = sk; part[2][sg] = sd; }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  const bool own = int(t) < TG / 32;
  sh = metal::simd_sum(own ? part[0][t] : 0.0f);
  sk = metal::simd_sum(own ? part[1][t] : 0.0f);
  sd = metal::simd_sum(own ? part[2][t] : 0.0f);
}
if (t == 0) {
  const float mh = sh * cst[0], mk = sk * cst[0];
  const float eh = mh + cst[1], ek = mk + cst[1];
  const float rh = metal::precise::rsqrt(eh), rk = metal::precise::rsqrt(ek);
  const float rr = rh * rk;
  const float d1 = sd * rr;
  const float dot = d1 * cst[2];
  const float ad = metal::abs(dot);
  const float mg = metal::precise::sqrt(metal::isnan(ad) || ad > 1e-6f ? ad : 1e-6f);
  const float s = dot < 0.0f ? -mg : mg;
  const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(s)));
  gate_s = s < 0.0f ? y : 1.0f - y;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
const float g = gate_s;
for (int c = t; c < D; c += TG) {
  const float gv = g * float(kv[vo + c]);
  out[xo + c] = OT(float(x[xo + c]) + gv);
}