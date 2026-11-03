  constexpr int VT = (NE + 3) / 4;
  constexpr int SIMDS = (VT + 31) / 32;
  float lane_max = -INFINITY;
  for (int g = 0; g < SIMDS; ++g) {
    int vt = g * 32 + int(lane);
    float m = -INFINITY;
    for (int i = 0; i < 4; ++i) {
      int e = vt * 4 + i;
      if (e < NE) m = metal::max(m, rk[e]);
    }
    float mg = simd_max(m);
    if (int(lane) == g) lane_max = mg;
  }
  float smax = simd_max(lane_max);
  float lane_sum = 0.0f;
  for (int g = 0; g < SIMDS; ++g) {
    int vt = g * 32 + int(lane);
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
      int e = vt * 4 + i;
      acc += (e < NE) ? fast::exp(rk[e] - smax) : 0.0f;
    }
    float pg = simd_sum(acc);
    if (int(lane) == g) lane_sum = pg;
  }
  float snorm = 1.0f / simd_sum(lane_sum);