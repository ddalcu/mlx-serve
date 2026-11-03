    auto a = tA.slice(g * GS, 0);
#if TILED
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
        (device uchar*)W + (int64_t)((R0 / 64 + threadgroup_position_in_grid.x) * KG + g) * (64 * GS / 2), dextents<int32_t, 2>(GS, 64));
#else
    auto b = tB.slice(g * GS, n0);
#endif
    op.run(a, b, P);
    for (int i = 0; i < CAP; i++) {
#if TILED
      const vec<bfloat, 2> sb = as_type<vec<bfloat, 2>>(((const device uint*)(g < KG / 2 ? ST : BT))[size_t(g % (KG / 2)) * NTOT + R0 + n0 + ecol[i]]);
#else
      const vec<bfloat, 2> sb = as_type<vec<bfloat, 2>>(sbw[size_t(g) * N + n0 + ecol[i]]);
#endif
      const float sv = float(sb[0]), bv = float(sb[1]);
      const float xs = simd_shuffle(row_sum[erow[i] >> 4], ushort(2 * (erow[i] & 15)));
      C[i] = fma(sv, P[i], fma(bv, xs, C[i]));
    }
  }
  // K slices are added in slice order, 16 outputs a thread at a time
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * 16 * 64];
  const ushort tip = ushort(thread_position_in_threadgroup.x) - slice * 64;
  if (SK > 1)
    for (int c0 = 0; c0 < CAP; c0 += 16) {
      if (slice > 0) for (int i = 0; i < 16; i++) part[((slice - 1) * 16 + i) * 64 + tip] = C[c0 + i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (slice == 0)
        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < 16; i++) C[c0 + i] += part[((s2 - 1) * 16 + i) * 64 + tip];
      if (TMR > 1) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  if (slice == 0)
    for (int i = 0; i < CAP; i++) {
      const int m = rb + erow[i], n = n0 + ecol[i];
      if (m < M) Y[m * N + n] = static_cast<bfloat>(C[i]);
    }
