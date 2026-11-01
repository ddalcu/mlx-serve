  // K tile: `msv_attn_qsa256`'s load, eight elements per uint4 (element
  // loads when `k_vec` is false), into the transposed staging.
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int i = tix; i < BK * (BD / 8); i += NT) {
    const int r = i >> 5;
    const int c8 = i & 31;
    T e[8] = {};
    if (r < rows_k) {
      const int pos = msv_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
      const long rb = kq_base + (long)pos * kqS[2];
      if (k_vec) {
        msv_unpack8(*((const device uint4*)(kq + rb) + c8), e);
      } else {
        for (int j = 0; j < 8; ++j) e[j] = kq[rb + (long)(c8 * 8 + j) * kqS[3]];
      }
    }
    const int cb = c8 * 8;
    for (int j = 0; j < 8; ++j) Ks[(cb + j) * LDK + r] = e[j];
  }
