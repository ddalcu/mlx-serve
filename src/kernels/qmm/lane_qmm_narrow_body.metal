    float s[NF][4], bb[NF][4];
    for (int f = 0; f < NF; f++) {
#if TILED
      const device uint4* sbv = (const device uint4*)(g < KG / 2 ? ST : BT);
      const uint4 q = colok[f] ? sbv[(size_t(g % (KG / 2)) * NTOT + R0 + n0 + f * 16 + fn) / 4] : uint4(0);
#else
      const uint4 q = colok[f] ? sbv[(size_t(g) * N + n0 + f * 16 + fn) / 4] : uint4(0);
#endif
      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);
      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
    }
    auto a = tA.slice(g * GS, 0);
#if BITS != 4 && BITS != 8
    {
      // A group is one little-endian bit stream; each word takes 4 values.
      uint w[WPG + 1];
      for (int i = 0; i <= WPG; i++) w[i] = 0;
      if (n0 + lane < N) for (int i = 0; i < WPG; i++) w[i] = wcol[g * (TILED ? 32 * WPG : WPG) + i];
      for (int c = 0; c < GS / 4; c++) {
        const int bit = 4 * BITS * c, i = bit >> 5, sh = bit & 31;
        uint word = w[i] >> sh;
        if (sh + 4 * BITS > 32) word |= w[i + 1] << (32 - sh);
        word = (word & ((1u << (2 * BITS)) - 1u)) | (((word >> (2 * BITS)) & ((1u << (2 * BITS)) - 1u)) << 16);
        word = (word & ((0x10001u << BITS) - 0x10001u)) | (((word >> BITS) & ((0x10001u << BITS) - 0x10001u)) << 8);
        stage[lane * (GS / 4) + c] = word;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#elif TILED && BITS == 4
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
        (device uchar*)W + (int64_t)((R0 / 32 + threadgroup_position_in_grid.x) * KG + g) * (32 * GS / 2), dextents<int32_t, 2>(GS, 32));
#elif TILED
    tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> b(
        (device uint8_t*)W + (int64_t)((R0 / 32 + threadgroup_position_in_grid.x) * KG + g) * (32 * GS), dextents<int32_t, 2>(GS, 32));
#else
    auto b = tB.slice(g * GS, n0);
#endif
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
#if BITS != 4 && BITS != 8
    simdgroup_barrier(mem_flags::mem_threadgroup); // the op has read the stage
#endif
    for (int t = 0; t < TMR; t++) {
      const float xs0 = simd_shuffle(row_sum[t], ushort(2 * fm));
      const float xs1 = simd_shuffle(row_sum[t], ushort(2 * (fm + 8)));
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++)
          for (int j = 0; j < 4; j++) {
            const int i = f * 8 + r * 4 + j;
            C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
          }
    }
  }
  // K slices are added in slice order, one 16-row block at a time
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
  for (int t = 0; t < TMR; t++) {
    if (SK > 1) {
      if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0)
        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
      if (TMR > 1) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0)
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++) {
          const int m = rb + t * 16 + fm + 8 * r;
          const int n = n0 + f * 16 + fn;
          if (m < M && n < N)
            for (int j = 0; j < 4; j++) Y[m * N + n + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);
        }
  }
