  constexpr int G = K / 64;
  const float one = ONE[0];
  const int nb = int(threadgroup_position_in_grid.x) * (8 * NT);
  const int rb = int(threadgroup_position_in_grid.y) * (8 * RT);
  threadgroup float red[S > 1 ? S * RT * NT * 64 : 1];
  int wrow[NT];
  for (int t = 0; t < NT; t++) wrow[t] = min(nb + 8 * t + fm, N - 1);
  int xr0[RT], xr1[RT];
  for (int rt = 0; rt < RT; rt++) { xr0[rt] = min(rb + 8 * rt + fn, R - 1); xr1[rt] = min(rb + 8 * rt + fn + 1, R - 1); }
  float acc[RT][NT][2];
  for (int c = sg; c < S; c += SG) {
  for (int rt = 0; rt < RT; rt++)
    for (int t = 0; t < NT; t++) { acc[rt][t][0] = 0.0f; acc[rt][t][1] = 0.0f; }
  for (int g = c; g < G; g += S) {
    uint wv[NT][BITS / 2];
    PRAGMA_UNROLL
    for (int t = 0; t < NT; t++) {
      const device uint* wp = (const device uint*)W + size_t(wrow[t]) * (K * BITS / 32) + 2 * BITS * g + (BITS / 2) * (fn / 2);
      if (BITS == 4) { const uint2 v = *(const device uint2*)wp; wv[t][0] = v.x; wv[t][1] = v.y; }
      else if (BITS == 8) { const uint4 v = *(const device uint4*)wp; wv[t][0] = v.x; wv[t][1] = v.y; wv[t][2] = v.z; wv[t][3] = v.w; }
      else { wv[t][0] = wp[0]; wv[t][1] = wp[1]; wv[t][2] = wp[2]; }
    }