  // K tile: dequantize into the SAME transposed staging the dense
  // kernel fills, so the fragment math below is untouched. The 8 dims of
  // one chunk share a quant group (GS is a multiple of 8), so the
  // scale/bias pair is read once per chunk, not per element.
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int i = tix; i < BK * (BD / 8); i += NT) {
    const int r = i >> 5;
    const int c8 = i & 31;
    const int cb = c8 * 8;
    T ev[8];
    if (r < rows_k) {
      const int pos = msv_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
      const long wrow = kq_base + (long)pos * kqS[2];
      const int g = cb / GS;
      const float sj = float(ksc[ksc_base + (long)pos * kscS[2] + (long)g]);
      const float bj = float(kbi[kbi_base + (long)pos * kbiS[2] + (long)g]);
      for (int wj = 0; wj < NW; ++wj) {
        const uint w = uint(kq[wrow + (long)(cb / VPW + wj)]);
        for (int u = 0; u < VPW; ++u) ev[wj * VPW + u] = T(float((w >> (u * BITS)) & MASKB) * sj + bj);
      }
    } else {
      for (int j = 0; j < 8; ++j) ev[j] = T(0.0f);
    }
    for (int j = 0; j < 8; ++j) Ks[(cb + j) * LDK + r] = ev[j];
  }
