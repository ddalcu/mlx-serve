template <int BITS, int GS, int K, int R, bool MX, typename X, typename S>
inline void dsv41_dots(const device uint32_t* w, const device S* sc, const device S* bi, thread const size_t* rows,
                       const device X* x, uint lane, thread float* acc) {
  constexpr int VPW = BITS == 3 ? 8 : 32 / BITS;
  constexpr int KW = K * BITS / 32, KG = K / GS, KP = K / VPW;
  constexpr uint MASK = (1u << BITS) - 1u;
  for (int r = 0; r < R; ++r) acc[r] = 0.0f;
  for (int pk = int(lane); pk < KP; pk += 32) {
    const int kb = pk * VPW, gi = kb / GS;
    float xp[VPW], xs = 0.0f;
    for (int i = 0; i < VPW; ++i) { const float v = float(x[kb + i]); xs += v; xp[i] = MX ? v : v * (1.0f / float(1u << (BITS * i))); }
    for (int r = 0; r < R; ++r) {
      const uint32_t q = mlxserve_qpack<BITS>(w, rows[r] * KW, pk);
      if (MX) {
        // e8m0 as 2^(e - 127), times the 2^8 the e4m3 bit placement leaves out.
        const uint e = uint(sc[rows[r] * KG + gi]);
        const float2 lo = mlxserve_e4m3x2(q), hi = mlxserve_e4m3x2(q >> 8);
        acc[r] += as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f * (xp[0] * lo.x + xp[1] * hi.x + xp[2] * lo.y + xp[3] * hi.y);
      } else {
        float dq = 0.0f;
        for (int i = 0; i < VPW; ++i) dq += xp[i] * float(q & (MASK << (BITS * i)));
        acc[r] += float(sc[rows[r] * KG + gi]) * dq + float(bi[rows[r] * KG + gi]) * xs;
      }
    }
  }
  for (int r = 0; r < R; ++r) acc[r] = metal::simd_sum(acc[r]);
}