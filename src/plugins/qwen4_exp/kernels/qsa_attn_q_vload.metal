  // V tile: dequantize straight into the row-major staging.
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int i = tix; i < BK * (BD / 8); i += NT) {
    const int r = i >> 5;
    const int c8 = i & 31;
    const int cb = c8 * 8;
    T ev[8];
    if (r < rows_k) {
      const int pos = msv_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
      const long wrow = vq_base + (long)pos * vqS[2];
      const int g = cb / GS;
      const float sj = float(vsc[vsc_base + (long)pos * vscS[2] + (long)g]);
      const float bj = float(vbi[vbi_base + (long)pos * vbiS[2] + (long)g]);
      for (int wj = 0; wj < NW; ++wj) {
        const uint w = uint(vq[wrow + (long)(cb / VPW + wj)]);
        for (int u = 0; u < VPW; ++u) ev[wj * VPW + u] = T(float((w >> (u * BITS)) & MASKB) * sj + bj);
      }
    } else {
      for (int j = 0; j < 8; ++j) ev[j] = T(0.0f);
    }
    for (int j = 0; j < 8; ++j) Vs[r * LDV + cb + j] = ev[j];
  }
