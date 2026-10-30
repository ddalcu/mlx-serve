  const uint lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;               // node * G + g
  const int PT = dims[1], W = dims[2], NCB = dims[3], CA = dims[4], RPA = dims[5];
  const int CT = PT / CK;
  constexpr int DP = D / 32;
  const int node = r / G, g = r % G;
  float m = -INFINITY, l = 0.0f, o[DP];
  for (int i = 0; i < DP; i++) o[i] = 0.0f;
  for (int c = 0; c < CT; c++) {
    const int64_t row = ((int64_t)hk * CA + c) * RPA + g * W + node;
    const float mc = PMA[row];
    if (mc == -INFINITY) continue;
    const float lc = PLA[row];
    const float nm = max(m, mc);
    const float e1 = fast::exp(m - nm), e2 = fast::exp(mc - nm);
    l = l * e1 + lc * e2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * e1 + POA[row * D + lane * DP + i] * e2;
    m = nm;
  }
  for (int c = 0; c < NCB; c++) {
    const int64_t row = (((int64_t)hk * NCB + c) * W + node) * 16 + g;
    const float mc = PMB[row];
    if (mc == -INFINITY) continue;
    const float lc = PLB[row];
    const float nm = max(m, mc);
    const float e1 = fast::exp(m - nm), e2 = fast::exp(mc - nm);
    l = l * e1 + lc * e2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * e1 + POB[row * D + lane * DP + i] * e2;
    m = nm;
  }
  const int h = hk * G + g;
  for (int i = 0; i < DP; i++) OUT[((int64_t)h * W + node) * D + lane * DP + i] = static_cast<bfloat>(o[i] / l);
