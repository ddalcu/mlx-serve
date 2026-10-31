  // V tile: uint4 copies (element loads when `v_vec` is false) straight
  // into the row-major staging.
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int i = tix; i < BK * (BD / 8); i += NT) {
    const int r = i >> 5;
    const int c8 = i & 31;
    uint4 w = uint4(0);
    if (r < rows_k) {
      const int pos = msv_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
      const long rb = vq_base + (long)pos * vqS[2];
      if (v_vec) {
        w = *((const device uint4*)(vq + rb) + c8);
      } else {
        T e[8];
        for (int j = 0; j < 8; ++j) e[j] = vq[rb + (long)(c8 * 8 + j) * vqS[3]];
        w = uint4(as_type<uint>(vec<T, 2>(e[0], e[1])), as_type<uint>(vec<T, 2>(e[2], e[3])), as_type<uint>(vec<T, 2>(e[4], e[5])), as_type<uint>(vec<T, 2>(e[6], e[7])));
      }
    }
    *((threadgroup uint4*)(Vs + r * LDV) + c8) = w;
  }
