      }
      PRAGMA_UNROLL
      for (int t = 0; t < NT; t++) {
        simdgroup_matrix<float, 8, 8> am;
        am.thread_elements()[0] = wfield<BITS>(wv[t], 0, s);
        am.thread_elements()[1] = wfield<BITS>(wv[t], 1, s);
        PRAGMA_UNROLL
        for (int rt = 0; rt < RT; rt++) simdgroup_multiply_accumulate(P[rt][t], am, bm[rt], P[rt][t]);
      }
    }
    PRAGMA_UNROLL
    for (int t = 0; t < NT; t++) {
      const float sc = float(SC[size_t(wrow[t]) * G + g]);
      const float bi = float(BI[size_t(wrow[t]) * G + g]);
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++) {
        acc[rt][t][0] = fma(bi, xs0[rt], fma(sc, P[rt][t].thread_elements()[0], acc[rt][t][0]));
        acc[rt][t][1] = fma(bi, xs1[rt], fma(sc, P[rt][t].thread_elements()[1], acc[rt][t][1]));
      }
    }
  }
  if (S == 1) {
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++)
        for (int e = 0; e < 2; e++) {
          const int row = rb + 8 * rt + fn + e, n = nb + 8 * t + fm;
          if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(acc[rt][t][e]);
        }
    return;
  }
  for (int rt = 0; rt < RT; rt++)
    for (int t = 0; t < NT; t++)
      for (int e = 0; e < 2; e++) red[((c * RT + rt) * NT + t) * 64 + int(lane) * 2 + e] = acc[rt][t][e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int idx = sg * 32 + int(lane); idx < RT * NT * 64; idx += SG * 32) {
    float v[S];
    for (int k = 0; k < S; k++) v[k] = red[k * (RT * NT * 64) + idx];
    for (int w = 1; w < S; w *= 2)
      for (int k = 0; k + w < S; k += 2 * w) v[k] = fma(v[k + w], one, v[k]);
    const int rt = idx / (NT * 64), t = (idx / 64) % NT, l = (idx % 64) / 2, e = idx % 2;
    const int lq = l / 4;
    const int row = rb + 8 * rt + (lq & 2) * 2 + (l % 2) * 2 + e, n = nb + 8 * t + (lq & 4) + ((l / 2) % 4);
    if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(v[0]);
  }
