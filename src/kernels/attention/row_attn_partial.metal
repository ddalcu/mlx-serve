  const uint lane = thread_index_in_simdgroup;
  const uint sgi = simdgroup_index_in_threadgroup;
  const int g = int(sgi) / SPLIT, s = int(sgi) % SPLIT;
  const int c = int(threadgroup_position_in_grid.y);
  const int P = dims[0], W = dims[1], NCH = dims[2], MAXD = dims[3];
  const int h = int(threadgroup_position_in_grid.z) / W, w = int(threadgroup_position_in_grid.z) % W;
  constexpr int DPL = D / 32;
  const int qh = h * G + g;
  const size_t kh = size_t(h) * size_t(K_strides[1]), kp = size_t(K_strides[2]);
  const size_t vh = size_t(h) * size_t(V_strides[1]), vp = size_t(V_strides[2]);
  threadgroup float sm[G * SPLIT], sl[G * SPLIT];
  threadgroup float so[G * SPLIT][D];
  {
    const int last = P + depth[w];
    const int k0 = c * CK;
    const int k1 = min(k0 + CK, last + 1);
    float q[DPL], o[DPL];
    for (int i = 0; i < DPL; i++) {
      q[i] = float(Q[((qh * W) + w) * D + int(lane) * DPL + i]);
      o[i] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    for (int base = k0 + s; base < k1; base += SPLIT * BLK) {
      float sc[BLK];
      int rows[BLK];
      float bm = -INFINITY;
      for (int j = 0; j < BLK; j++) {
        const int pos = base + j * SPLIT;
        rows[j] = pos < k1 ? (pos < P ? pos : P + path[w * MAXD + (pos - P)]) : -1;
        float d = 0.0f;
        if (rows[j] >= 0) {
          const device bfloat* kr = K + kh + size_t(rows[j]) * kp + int(lane) * DPL;
          for (int i = 0; i < DPL; i++) d = fma(q[i], float(kr[i]), d);
        }
        sc[j] = simd_sum(d) * scale[0];
        if (rows[j] >= 0) bm = metal::max(bm, sc[j]);
      }
      const float mn = metal::max(m, bm);
      const float a = metal::exp(m - mn);
      l *= a;
      for (int i = 0; i < DPL; i++) o[i] *= a;
      for (int j = 0; j < BLK; j++) {
        if (rows[j] < 0) continue;
        const float b = metal::exp(sc[j] - mn);
        l += b;
        const device bfloat* vr = V + vh + size_t(rows[j]) * vp + int(lane) * DPL;
        for (int i = 0; i < DPL; i++) o[i] = fma(b, float(vr[i]), o[i]);
      }
      m = mn;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { sm[sgi] = m; sl[sgi] = l; }
    for (int i = 0; i < DPL; i++) so[sgi][int(lane) * DPL + i] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s == 0) {
      float mx_ = -INFINITY;
      for (int t = 0; t < SPLIT; t++) mx_ = metal::max(mx_, sm[g * SPLIT + t]);
      float lsum = 0.0f, acc[DPL];
      for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
      for (int t = 0; t < SPLIT; t++) {
        const float e = sl[g * SPLIT + t] > 0.0f ? metal::exp(sm[g * SPLIT + t] - mx_) : 0.0f;
        lsum = fma(sl[g * SPLIT + t], e, lsum);
        for (int i = 0; i < DPL; i++) acc[i] = fma(so[g * SPLIT + t][int(lane) * DPL + i], e, acc[i]);
      }
      const int slot = (qh * W + w) * NCH + c;
      if (lane == 0) { PM[slot] = mx_; PL[slot] = lsum; }
      for (int i = 0; i < DPL; i++) PO[size_t(slot) * D + int(lane) * DPL + i] = acc[i];
    }
  }