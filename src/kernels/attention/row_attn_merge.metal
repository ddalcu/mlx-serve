  const uint lane = thread_index_in_simdgroup;
  const int qh = int(threadgroup_position_in_grid.y);
  const int w = int(threadgroup_position_in_grid.z);
  const int W = dims[1], NCH = dims[2];
  constexpr int DPL = D / 32;
  const int base = (qh * W + w) * NCH;
  float mx_ = -INFINITY;
  for (int c = 0; c < NCH; c++) mx_ = metal::max(mx_, PM[base + c]);
  float lsum = 0.0f, acc[DPL];
  for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
  for (int c = 0; c < NCH; c++) {
    const float e = PL[base + c] > 0.0f ? metal::exp(PM[base + c] - mx_) : 0.0f;
    lsum = fma(PL[base + c], e, lsum);
    for (int i = 0; i < DPL; i++) acc[i] = fma(PO[size_t(base + c) * D + int(lane) * DPL + i], e, acc[i]);
  }
  for (int i = 0; i < DPL; i++) OUT[((qh * W) + w) * D + int(lane) * DPL + i] = bfloat(acc[i] / lsum);