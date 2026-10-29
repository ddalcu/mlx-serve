  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;
  const int tile = int(threadgroup_position_in_grid.z) * SG + sg;   // 16-row tile
  const uint hk = threadgroup_position_in_grid.x;              // key head
  const uint c = threadgroup_position_in_grid.y;               // chunk of CK keys
  const int L = dims[0], NQ = dims[2], SGA = dims[3];
  const int NCH = dims[1];
  const int RP = 16 * SGA;
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  const int r0 = tile * 16 + fm, r1 = r0 + 8;
  const int n0 = r0 < G * NQ ? L : 0;
  const int n1 = r1 < G * NQ ? L : 0;
  threadgroup half Ps[SG * 16 * TK];
  threadgroup half* myP = Ps + sg * 16 * TK;
  if (tile >= SGA) return;
  // a key head's query rows g * NQ + node, where q [1, H, NQ, D] keeps them; rows past them read as 0
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tQ((device bfloat*)Qp + (int64_t)hk * G * NQ * D, dextents<int32_t, 2>(D, G * NQ));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tK((device bfloat*)K + (int64_t)hk * K_strides[1], dextents<int32_t, 2>(D, L), array<int32_t, 2>({1, int32_t(K_strides[2])}));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tV((device bfloat*)V + (int64_t)hk * V_strides[1], dextents<int32_t, 2>(D, L), array<int32_t, 2>({1, int32_t(V_strides[2])}));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(myP, dextents<int32_t, 2>(TK, 16));
  constexpr auto dS = matmul2d_descriptor(16, TK, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto dO = matmul2d_descriptor(16, 128, TK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dS, execution_simdgroup> opS;
  matmul2d<dO, execution_simdgroup> opO;
  auto aQ = tQ.slice(0, tile * 16);
  auto bV0 = tV.slice(0, 0);
  // the op is exact up to 128 output columns: the head dimension goes in two halves
  auto Olo = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(bV0), float>();
  auto Ohi = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(bV0), float>();
  for (int i = 0; i < 64; i++) { Olo[i] = 0.0f; Ohi[i] = 0.0f; }
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
  const int kbeg = int(c) * CK;
  const int kend = min(kbeg + CK, L);
  for (int kt = kbeg; kt < kend; kt += TK) {
    auto bK = tK.slice(0, kt);
    auto S = opS.template get_destination_cooperative_tensor<decltype(aQ), decltype(bK), float>();
    opS.run(aQ, bK, S);
    float s[TK / 2];
    for (int i = 0; i < TK / 2; i++) {                         // 8 elements per 16-key block: 4 of row fm, then 4 of fm + 8
      const int key = kt + (i >> 3) * 16 + fn + (i & 3);
      s[i] = key < ((i & 4) ? n1 : n0) ? S[i] * scale[0] : -INFINITY;
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
    auto Pc = opS.template get_destination_cooperative_tensor<decltype(aQ), decltype(bK), half>();
    for (int i = 0; i < TK / 2; i++) Pc[i] = half(p[i]);
    auto Pin = opO.template get_left_input_cooperative_tensor<half, bfloat, float>(Pc);
    for (int i = 0; i < 64; i++) { const float f = (i & 4) ? f1 : f0; Olo[i] *= f; Ohi[i] *= f; }
    auto bVlo = tV.slice(0, kt);
    auto bVhi = tV.slice(128, kt);
    opO.run(Pin, bVlo, Olo);
    opO.run(Pin, bVhi, Ohi);
  }
  const int64_t base = ((int64_t)hk * NCH + c) * RP;
  // 16-bit operand layout: elements 4q..4q+3 are four consecutive columns of one row
  for (int q = 0; q < 16; q++) {
    device float* dst = PO + (base + tile * 16 + fm + (q & 1) * 8) * D + (q >> 1) * 16 + fn;
    *(device float4*)dst = float4(Olo[4 * q], Olo[4 * q + 1], Olo[4 * q + 2], Olo[4 * q + 3]);
    *(device float4*)(dst + 128) = float4(Ohi[4 * q], Ohi[4 * q + 1], Ohi[4 * q + 2], Ohi[4 * q + 3]);
  }
  if ((lane & 9) == 0) {
    PM[base + r0] = m0; PL[base + r0] = l0;
    PM[base + r1] = m1; PL[base + r1] = l1;
  }
