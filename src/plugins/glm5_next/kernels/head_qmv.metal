  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int h = int(threadgroup_position_in_grid.z);
  const int r0 = (int(threadgroup_position_in_grid.y) * NSG + sg) * 4;
  constexpr int WB = K * glm_bytes_per_pack<BITS>() / glm_pack_factor<BITS>();
  constexpr int G = K / GS;
  const device uint8_t* wh = (const device uint8_t*)w + (size_t(h) * N + r0) * WB;
  const device T* sh = scales + (size_t(h) * N + r0) * G;
  const device T* bh = biases + (size_t(h) * N + r0) * G;
  const device T* xh = x + size_t(h) * K;
  float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  glm_qmv_rows<T, K, GS, BITS, 4>(wh, sh, bh, xh, lane, result);
  for (int r = 0; r < 4; r++) {
    const float v = simd_sum(result[r]);
    if (lane == 0) out[size_t(h) * N + r0 + r] = static_cast<T>(v * scale[0]);
  }