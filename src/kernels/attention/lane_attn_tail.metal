  const ushort lane = thread_index_in_simdgroup;
  const ushort tsg = simdgroup_index_in_threadgroup;           // simdgroup 0 computes, all TSG load
  const uint tid = thread_position_in_threadgroup.x;
  const uint hk = threadgroup_position_in_grid.x;              // key head
  const uint cb = threadgroup_position_in_grid.y;              // tail chunk (from the first chunk holding the window)
  const uint node = threadgroup_position_in_grid.z;
  const int P = dims[0], PT = dims[1], W = dims[2], NCB = dims[3], CA = dims[4], RPA = dims[5];
  const int depth = depths[node];
  const int nmax = P + depth + 1;                              // logical keys 0 .. P + depth
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  const int r0 = fm, r1 = fm + 8;                              // query rows: the node's heads (G of 16)
  const int n0 = r0 < G ? nmax : 0;
  const int n1 = r1 < G ? nmax : 0;
  threadgroup half myP[16 * TK];
  threadgroup bfloat KV[32 * D];                               // 32 keys at a time, or TK keys' half rows of values
  const device bfloat* kbase = (const device bfloat*)K + (int64_t)hk * K_strides[1];
  const device bfloat* vbase = (const device bfloat*)V + (int64_t)hk * V_strides[1];
  const int64_t kstep = K_strides[2], vstep = V_strides[2];
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tQ((device bfloat*)QB + ((int64_t)hk * G * W + node) * D, dextents<int32_t, 2>(D, G), array<int32_t, 2>({1, int32_t(W * D)}));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tK32(KV, dextents<int32_t, 2>(D, 32));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tVh(KV, dextents<int32_t, 2>(128, TK));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(myP, dextents<int32_t, 2>(TK, 16));
  // scores 32 keys at a time: each equals the TK-key op's bit for bit
  constexpr auto dS = matmul2d_descriptor(16, 32, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto dO = matmul2d_descriptor(16, 128, TK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dS, execution_simdgroup> opS;
  matmul2d<dO, execution_simdgroup> opO;
  auto Olo = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(tVh), float>();
  auto Ohi = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(tVh), float>();
  for (int i = 0; i < 64; i++) { Olo[i] = 0.0f; Ohi[i] = 0.0f; }
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
  const int c0 = PT / CK;
  const int c = c0 + int(cb);
  const int kbeg = max(c * CK, PT);
  const int kend = min((c + 1) * CK, nmax);
  if (tsg == 0 && cb == 0 && PT > c0 * CK) {
    const int64_t baseA = ((int64_t)hk * CA + c0) * RPA + node;   // PARTIAL's row g * W + node
    for (int q = 0; q < 16; q++) {
      const int row = fm + (q & 1) * 8;
      if (row >= G) continue;
      const device float* src = POA + (baseA + row * W) * D + (q >> 1) * 16 + fn;
      for (int j = 0; j < 4; j++) { Olo[4 * q + j] = src[j]; Ohi[4 * q + j] = src[128 + j]; }
    }
    if (r0 < G) { m0 = PMA[baseA + r0 * W]; l0 = PLA[baseA + r0 * W]; }
    if (r1 < G) { m1 = PMA[baseA + r1 * W]; l1 = PLA[baseA + r1 * W]; }
  }
  for (int kt = kbeg; kt < kend; kt += TK) {
    float sraw[TK / 2];
    for (int h = 0; h < TK / 32; h++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = tid; e < 32 * D / 8; e += 32 * TSG) {      // logical slot -> physical row
        const int row = int(e) / (D / 8), col = (int(e) % (D / 8)) * 8;
        const int q = kt + h * 32 + row;
        int phys = -1;
        if (q < P) phys = q;
        else if (q < nmax) phys = P + paths[node * MAXD + (q - P)];
        ((threadgroup vec<bfloat, 8>*)KV)[e] = phys >= 0 ? *(const device vec<bfloat, 8>*)(kbase + phys * kstep + col) : vec<bfloat, 8>(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tsg == 0) {
        auto S = opS.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK32), float>();
        opS.run(tQ, tK32, S);
        for (int i = 0; i < 16; i++) sraw[h * 16 + i] = S[i];
      }
    }
    if (tsg == 0) {
    float s[TK / 2];
    for (int i = 0; i < TK / 2; i++) {
      const int key = kt + (i >> 3) * 16 + fn + (i & 3);
      s[i] = key < ((i & 4) ? n1 : n0) ? sraw[i] * scale[0] : -INFINITY;
    }
    float x0 = -INFINITY, x1 = -INFINITY;
    for (int i = 0; i < TK / 2; i++) { if (i & 4) x1 = max(x1, s[i]); else x0 = max(x0, s[i]); }
    x0 = max(x0, simd_shuffle_xor(x0, 1)); x0 = max(x0, simd_shuffle_xor(x0, 8));
    x1 = max(x1, simd_shuffle_xor(x1, 1)); x1 = max(x1, simd_shuffle_xor(x1, 8));
    const float nm0 = max(m0, x0), nm1 = max(m1, x1);
    const float f0 = (x0 == -INFINITY) ? 1.0f : fast::exp(m0 - nm0);
    const float f1 = (x1 == -INFINITY) ? 1.0f : fast::exp(m1 - nm1);
    float p[TK / 2];
    for (int i = 0; i < TK / 2; i++) p[i] = (s[i] == -INFINITY) ? 0.0f : fast::exp(s[i] - ((i & 4) ? nm1 : nm0));
    float y0 = 0.0f, y1 = 0.0f;
    for (int b = 0; b < TK / 16; b++) {
      y0 += (p[b * 8] + p[b * 8 + 1]) + (p[b * 8 + 2] + p[b * 8 + 3]);
      y1 += (p[b * 8 + 4] + p[b * 8 + 5]) + (p[b * 8 + 6] + p[b * 8 + 7]);
    }
    y0 += simd_shuffle_xor(y0, 1); y0 += simd_shuffle_xor(y0, 8);
    y1 += simd_shuffle_xor(y1, 1); y1 += simd_shuffle_xor(y1, 8);
    if (x0 != -INFINITY) { l0 = l0 * f0 + y0; m0 = nm0; }
    if (x1 != -INFINITY) { l1 = l1 * f1 + y1; m1 = nm1; }
    for (int f = 0; f < TK / 16; f++)
      for (int i = 0; i < 4; i++) {
        myP[fm * TK + f * 16 + fn + i] = half(p[f * 8 + i]);
        myP[(fm + 8) * TK + f * 16 + fn + i] = half(p[f * 8 + 4 + i]);
      }
    for (int i = 0; i < 64; i++) { const float f = (i & 4) ? f1 : f0; Olo[i] *= f; Ohi[i] *= f; }
    }
    for (int hv = 0; hv < 2; hv++) {                           // values: TK keys x 128 columns at a time
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = tid; e < TK * 128 / 8; e += 32 * TSG) {
        const int row = int(e) / 16, col = hv * 128 + (int(e) % 16) * 8;
        const int q = kt + row;
        int phys = -1;
        if (q < P) phys = q;
        else if (q < nmax) phys = P + paths[node * MAXD + (q - P)];
        ((threadgroup vec<bfloat, 8>*)KV)[e] = phys >= 0 ? *(const device vec<bfloat, 8>*)(vbase + phys * vstep + col) : vec<bfloat, 8>(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tsg == 0) {
        if (hv == 0) opO.run(tP, tVh, Olo);
        else opO.run(tP, tVh, Ohi);
      }
    }
  }
  if (tsg != 0) return;
  const int64_t base = (((int64_t)hk * NCB + cb) * W + node) * 16;
  for (int q = 0; q < 16; q++) {
    device float* dst = PO + (base + fm + (q & 1) * 8) * D + (q >> 1) * 16 + fn;
    *(device float4*)dst = float4(Olo[4 * q], Olo[4 * q + 1], Olo[4 * q + 2], Olo[4 * q + 3]);
    *(device float4*)(dst + 128) = float4(Ohi[4 * q], Ohi[4 * q + 1], Ohi[4 * q + 2], Ohi[4 * q + 3]);
  }
  if ((lane & 9) == 0) {
    PM[base + r0] = m0; PL[base + r0] = l0;
    PM[base + r1] = m1; PL[base + r1] = l1;
  }
