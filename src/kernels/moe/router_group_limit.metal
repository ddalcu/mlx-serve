  for (uint g = lane; g < NG; g += 32) {
    float m1 = -INFINITY, m2 = -INFINITY;
    for (uint j = 0; j < GSZ; ++j) {
      float v = rk[g * GSZ + j];
      if (v > m1) { m2 = m1; m1 = v; } else if (v > m2) { m2 = v; }
    }
    gs[g] = m1 + m2;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  uint keep_mask = 0u;
  for (uint r = 0; r < TOPKG; ++r) {
    float best = -INFINITY;
    uint bidx = 0xFFFFFFFFu;
    for (uint g = lane; g < NG; g += 32) {
      float v = gs[g];
      if (v > best) { best = v; bidx = g; }
    }
    float gmax = simd_max(best);
    uint cand = (best == gmax) ? bidx : 0xFFFFFFFFu;
    uint gidx = metal::min(simd_min(cand), uint(NG - 1));
    keep_mask |= (1u << gidx);
    if (lane == 0) gs[gidx] = -INFINITY;
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint e = lane; e < NE; e += 32) {
    if ((keep_mask & (1u << (e / GSZ))) == 0u) rk[e] = -1.0e4f;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);