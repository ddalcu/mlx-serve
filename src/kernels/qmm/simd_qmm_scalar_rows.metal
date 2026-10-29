  constexpr int XP = 76;
  threadgroup float xs[RR][XB * XP];
  const uint lane = thread_index_in_simdgroup;
  const int tid = int(simdgroup_index_in_threadgroup) * 32 + int(lane);
  const int c = int(lane) % S;
  constexpr int SLOTS = 32 / S;
  const int n0 = (int(threadgroup_position_in_grid.x) * SGS + int(simdgroup_index_in_threadgroup)) * (SLOTS * NR)
                 + int(lane) / S;
  constexpr int G = K / 64;
  const float one = ONE[0];
  constexpr int WQ = BITS / 2;
  const device uint4* wr[NR];
  const device bfloat* sr[NR];
  const device bfloat* br[NR];
  float acc[NR][RR];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) {
    const int nn = min(n0 + SLOTS * u, N - 1);
    wr[u] = (const device uint4*)(W + size_t(nn) * (K * BITS / 32));
    sr[u] = SC + size_t(nn) * G;
    br[u] = BI + size_t(nn) * G;
    for (int r = 0; r < RR; r++) acc[u][r] = 0.0f;
  }
  uint4 nw[NR][WQ];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++)
    PRAGMA_UNROLL
    for (int q = 0; q < WQ; q++) nw[u][q] = c < G ? wr[u][WQ * c + q] : uint4(0);
  for (int b0 = 0; b0 < G; b0 += XB) {
    const int nbk = min(XB, G - b0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < RR * nbk * 8; idx += SGS * 32) {
      const int r = idx / (nbk * 8), gl = (idx / 8) % nbk, i = idx % 8;
      const uint4 v = LOAD8(r, 8 * (b0 + gl) + i);
      PRAGMA_UNROLL
      for (int s = 0; s < 8; s++) xs[r][gl * XP + 8 * s + i] = bf8(v, s) * wpre<BITS>(s);
      xs[r][gl * XP + 64 + i] = sum8(v, one);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = b0 + c; g < b0 + nbk; g += S) {
      uint wd[NR][2 * BITS];
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++)
        PRAGMA_UNROLL
        for (int q = 0; q < 2 * BITS; q++) wd[u][q] = nw[u][q / 4][q % 4];
      if (g + S < G) {
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++)
          PRAGMA_UNROLL
          for (int q = 0; q < WQ; q++) nw[u][q] = wr[u][WQ * (g + S) + q];
      }
      // The chain per row is the one-row kernel's (s, then i); the masked
      // weights are built once for every row.
      float P[NR][RR];
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++)
        for (int r = 0; r < RR; r++) P[u][r] = 0.0f;
      PRAGMA_UNROLL
      for (int s = 0; s < 8; s++) {
        float wf[NR][8];
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++)
          PRAGMA_UNROLL
          for (int i = 0; i < 8; i++) wf[u][i] = wfield<BITS>(wd[u], i, s);
        PRAGMA_UNROLL
        for (int r = 0; r < RR; r++) {
          const threadgroup float* xg = xs[r] + (g - b0) * XP;
          const float4 lo = *(const threadgroup float4*)(xg + 8 * s), hi = *(const threadgroup float4*)(xg + 8 * s + 4);
          const float xq[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
          PRAGMA_UNROLL
          for (int u = 0; u < NR; u++)
            PRAGMA_UNROLL
            for (int i = 0; i < 8; i++) P[u][r] = fma(xq[i], wf[u][i], P[u][r]);
        }
      }
      PRAGMA_UNROLL
      for (int r = 0; r < RR; r++) {
        const threadgroup float* xg = xs[r] + (g - b0) * XP;
        const float4 p0 = *(const threadgroup float4*)(xg + 64), p1 = *(const threadgroup float4*)(xg + 68);
        const float xsum = fma(fma(fma(p1.w, one, p1.z), one, fma(p1.y, one, p1.x)), one,
                               fma(fma(p0.w, one, p0.z), one, fma(p0.y, one, p0.x)));
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++) {
          acc[u][r] = fma(float(sr[u][g]), P[u][r], acc[u][r]);
          acc[u][r] = fma(float(br[u][g]), xsum, acc[u][r]);
        }
      }
    }
  }
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++)
    for (int r = 0; r < RR; r++) {
      float v = acc[u][r];
      PRAGMA_UNROLL
      for (int m = 1; m < S; m <<= 1) v = fma(simd_shuffle_xor(v, ushort(m)), one, v);
      const int n = n0 + SLOTS * u;
      if (n < N && c == 0) OUT[size_t(r) * N + n] = bfloat(v);
    }
