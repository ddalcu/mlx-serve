    // mimo_fp8_gemv (fp8res): y[m, n] = sum_k x[m, k] * (e4m3(w[n, k]) * s[n / 64, k / 128]) for M <= 8 rows (decode, the
    // MTP verify rows, a head's catch-up rows), BF16 x / y, f32 FMA. Each simdgroup owns R consecutive output rows (one
    // 64-row scale row) over one of KS K partitions; lane l reads 8 codes per row per 256-column step at
    // k = partition + 256 st + 8 l (one 128-column block: one scale per lane per step).
    // Accumulation order of every output (m, n), fixed by (KS, the text) and independent of M, R and RG:
    //   per lane: acc = fma(x, w, acc) over st ascending, then the 8 codes ascending (w = code * scale, one f32 rounding);
    //   lanes: xor butterfly 16, 8, 4, 2, 1 (every lane ends with the same sum); partitions: p = 0 .. KS - 1 in order;
    //   one rounding to BF16. Row m's output is the M = 1 call's output for that row, bit for bit (M-invariant).
    constexpr int KV = 8;
    constexpr int STEP = 32 * KV;
    constexpr int KP = K / KS;
    constexpr int SC = K / 128;
    static_assert(K % (KS * STEP) == 0, "a K partition is whole 256-column steps");
    static_assert(N % (R * RG) == 0, "N tiles by R * RG rows");
    static_assert(64 % R == 0, "a simdgroup's rows share one 64-row scale row");
    static_assert(M >= 1 && M <= 8, "decode rows");
    const int sgi = int(simdgroup_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    const int ks = sgi % KS;
    const int rg = sgi / KS;
    const int n0 = (int(threadgroup_position_in_grid.y) * RG + rg) * R;
    const int k0 = ks * KP + KV * lane;
    const device uint8_t* wb = w + size_t(n0) * size_t(K) + size_t(k0);
    const device float* sb = scales + size_t(n0 / 64) * size_t(SC) + size_t(k0 / 128);
    const device bfloat16_t* xb = x + k0;
    threadgroup float part[KS > 1 ? RG * KS * M * R : 1];
    float acc[M][R];
    for (int m = 0; m < M; m++) {
      for (int r = 0; r < R; r++) {
        acc[m][r] = 0.0f;
      }
    }
    for (int st = 0; st < KP / STEP; st++) {
      const float s256 = sb[st * (STEP / 128)] * 256.0f;
      float wv[R][KV];
      for (int r = 0; r < R; r++) {
        const uint2 q = *(const device uint2*)(wb + size_t(r) * size_t(K) + size_t(st * STEP));
        fp8r_dq4(q.x, s256, &wv[r][0]);
        fp8r_dq4(q.y, s256, &wv[r][4]);
      }
      for (int m = 0; m < M; m++) {
        const uint4 q = *(const device uint4*)(xb + size_t(m) * size_t(K) + size_t(st * STEP));
        float xv[KV];
        xv[0] = fp8r_bf_lo(q.x);
        xv[1] = fp8r_bf_hi(q.x);
        xv[2] = fp8r_bf_lo(q.y);
        xv[3] = fp8r_bf_hi(q.y);
        xv[4] = fp8r_bf_lo(q.z);
        xv[5] = fp8r_bf_hi(q.z);
        xv[6] = fp8r_bf_lo(q.w);
        xv[7] = fp8r_bf_hi(q.w);
        for (int r = 0; r < R; r++) {
          for (int i = 0; i < KV; i++) {
            acc[m][r] = fma(xv[i], wv[r][i], acc[m][r]);
          }
        }
      }
    }
    for (int m = 0; m < M; m++) {
      for (int r = 0; r < R; r++) {
        float v = acc[m][r];
        v += simd_shuffle_xor(v, ushort(16));
        v += simd_shuffle_xor(v, ushort(8));
        v += simd_shuffle_xor(v, ushort(4));
        v += simd_shuffle_xor(v, ushort(2));
        v += simd_shuffle_xor(v, ushort(1));
        acc[m][r] = v;
      }
    }
    if constexpr (KS == 1) {
      if (lane == 0) {
        for (int m = 0; m < M; m++) {
          for (int r = 0; r < R; r++) {
            y[size_t(m) * size_t(N) + size_t(n0 + r)] = static_cast<bfloat16_t>(acc[m][r]);
          }
        }
      }
    } else {
      if (lane == 0) {
        for (int m = 0; m < M; m++) {
          for (int r = 0; r < R; r++) {
            part[((rg * KS + ks) * M + m) * R + r] = acc[m][r];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (ks == 0 && lane == 0) {
        for (int m = 0; m < M; m++) {
          for (int r = 0; r < R; r++) {
            float t = part[((rg * KS) * M + m) * R + r];
            for (int p = 1; p < KS; p++) {
              t += part[((rg * KS + p) * M + m) * R + r];
            }
            y[size_t(m) * size_t(N) + size_t(n0 + r)] = static_cast<bfloat16_t>(t);
          }
        }
      }
    }
