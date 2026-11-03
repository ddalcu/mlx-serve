
    constexpr int STEP = 32 * KV;
    constexpr int KP = K / KS;
    static_assert(KV == 4 || KV == 8, "KV is 4 or 8");
    static_assert(K % KS == 0 && KP % STEP == 0, "a K partition is whole lane steps");
    static_assert(N % (R * RG) == 0, "N tiles by R * RG rows");
    static_assert(M >= 1 && M <= 8, "decode rows");
    const uint3 tid = threadgroup_position_in_grid;
    const int sgi = int(simdgroup_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    const int ks = sgi % KS;
    const int rg = sgi / KS;
    const int g = int(tid.z);
    const int n0 = (int(tid.y) * RG + rg) * R;
    const size_t row0 = size_t(g) * size_t(N) + size_t(n0);
    const int k0 = ks * KP + KV * lane;
    const device uint8_t* wb = (const device uint8_t*)w + row0 * size_t(K) + size_t(k0);
    const device uint8_t* sb = scales + row0 * size_t(K / 32) + size_t(k0 / 32);
    const device float* xb = x + g * XG + k0;
    threadgroup float part[RG * KS * M * R];
    float acc[M][R];
    for (int m = 0; m < M; m++) {
      for (int r = 0; r < R; r++) {
        acc[m][r] = 0.0f;
      }
    }
    for (int kb = 0; kb < KP / STEP; kb++) {
      float wv[R][KV];
      for (int r = 0; r < R; r++) {
        const float s256 = rcp_e8m0(uint(sb[r * (K / 32) + kb * KV])) * 256.0f;
        if constexpr (KV == 8) {
          const uint2 q = *(const device uint2*)(wb + r * K + kb * STEP);
          rcp_dq4(q.x, s256, &wv[r][0]);
          rcp_dq4(q.y, s256, &wv[r][4]);
        } else {
          const uint q = *(const device uint*)(wb + r * K + kb * STEP);
          rcp_dq4(q, s256, &wv[r][0]);
        }
      }
      for (int m = 0; m < M; m++) {
        float xv[KV];
        if constexpr (KV == 8) {
          const float4 qa = *(const device float4*)(xb + m * XS + kb * STEP);
          const float4 qb = *(const device float4*)(xb + m * XS + kb * STEP + 4);
          xv[0] = qa.x;
          xv[1] = qa.y;
          xv[2] = qa.z;
          xv[3] = qa.w;
          xv[4] = qb.x;
          xv[5] = qb.y;
          xv[6] = qb.z;
          xv[7] = qb.w;
        } else {
          const float4 q = *(const device float4*)(xb + m * XS + kb * STEP);
          xv[0] = q.x;
          xv[1] = q.y;
          xv[2] = q.z;
          xv[3] = q.w;
        }
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
            y[m * YS + g * YG + n0 + r] = static_cast<float>(acc[m][r]);
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
            y[m * YS + g * YG + n0 + r] = static_cast<float>(t);
          }
        }
      }
    }
