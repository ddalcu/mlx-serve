constexpr int BK = 32, SM = 16, NSG = 4;
const int H = qi_shape[1];
const int LQ = qi_shape[2];
const int L = ki_shape[2];
const int KVH = ki_shape[1];
const int NB = (L + BK - 1) / BK;
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int head = int(threadgroup_position_in_grid.y);
const int kvh = head / (H / KVH);
const int q0 = int(threadgroup_position_in_grid.x) * (SM * NSG) + int(sg) * SM;
tensor<device int8_t, dextents<int, 2>, tensor_inline> Qt((device int8_t*)qi + (size_t)head * LQ * 128, dextents<int, 2>(128, LQ));
tensor<device int8_t, dextents<int, 2>, tensor_inline> Kt((device int8_t*)ki + (size_t)kvh * L * 128, dextents<int, 2>(128, L));
tensor<device int8_t, dextents<int, 2>, tensor_inline> Vt((device int8_t*)vi + (size_t)kvh * L * 128, dextents<int, 2>(128, L));
constexpr auto dS = matmul2d_descriptor(SM, BK, 128, false, true, false, matmul2d_descriptor::mode::multiply);
matmul2d<dS, execution_simdgroup> opS;
constexpr auto dO = matmul2d_descriptor(SM, 128, BK, false, false, false, matmul2d_descriptor::mode::multiply);
matmul2d<dO, execution_simdgroup> opO;
auto qsl = Qt.template slice<128, SM>(0, q0);
auto ks0 = Kt.template slice<128, BK>(0, 0);
auto vs0 = Vt.template slice<128, BK>(0, 0);
auto qT = opS.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
qT.load(qsl);
auto sT = opS.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(qT)>, decltype(ks0), int32_t>();
auto pT = opO.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
auto oT = opO.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pT)>, decltype(vs0), int32_t>();
constexpr int NS = SM / 8, SC = SM * BK / 32, OC = SM * 128 / 32;
const int rb = int((lane >> 1) & 3) + 4 * int((lane >> 4) & 1);
const int cb = 4 * int(lane & 1) + 8 * int((lane >> 3) & 1);
float of[OC];
#pragma clang loop unroll(full)
for (int i = 0; i < OC; i++) of[i] = 0.0f;
float qsc[NS], m[NS], l[NS];
#pragma clang loop unroll(full)
for (int r = 0; r < NS; r++) {
  const int row = q0 + rb + 8 * r;
  qsc[r] = row < LQ ? qs[(size_t)head * LQ + row] * scale : 0.0f;
  m[r] = -INFINITY; l[r] = 0.0f;
}
const device float* kbs = ks + (size_t)kvh * NB;
const float LOG127 = 6.98868468677f;
for (int kb = 0; kb < NB; kb++) {
  const int k0 = kb * BK;
  auto ksl = Kt.template slice<128, BK>(0, k0);
  opS.run(qT, ksl, sT);
  if (k0 + BK > L) {
#pragma clang loop unroll(full)
    for (int g = 0; g < SC; g += 4) {
      const int c = k0 + cb + 16 * ((g >> 3) & 1);
#pragma clang loop unroll(full)
      for (int e = 0; e < 4; e++) if (c + e >= L) sT[g + e] = INT_MIN;
    }
  }
  int imx[NS];
#pragma clang loop unroll(full)
  for (int r = 0; r < NS; r++) imx[r] = INT_MIN;
#pragma clang loop unroll(full)
  for (int g = 0; g < SC; g++) { const int sl = ((g >> 2) & 1) + 2 * (g >> 4); imx[sl] = max(imx[sl], sT[g]); }
  const float kscale = kbs[kb];
  float c[NS], off[NS], fac[NS], bs[NS];
  int rs[NS];
#pragma clang loop unroll(full)
  for (int r = 0; r < NS; r++) {
    int t = max(imx[r], simd_shuffle_xor(imx[r], 1));
    t = max(t, simd_shuffle_xor(t, 8));
    c[r] = qsc[r] * kscale;
    const float mb = float(t) * c[r];
    const float mn = max(m[r], mb);
    fac[r] = exp2(m[r] - mn);
    bs[r] = exp2(mb - mn);
    m[r] = mn;
    off[r] = LOG127 - mb;
    rs[r] = 0;
  }
#pragma clang loop unroll(full)
  for (int g = 0; g < SC; g++) {
    const int sl = ((g >> 2) & 1) + 2 * (g >> 4);
    const int p = sT[g] == INT_MIN ? 0 : int(rint(exp2(fma(float(sT[g]), c[sl], off[sl]))));
    rs[sl] += p;
    pT[g] = int8_t(p);
  }
#pragma clang loop unroll(full)
  for (int r = 0; r < NS; r++) {
    int t = rs[r] + simd_shuffle_xor(rs[r], 1);
    t += simd_shuffle_xor(t, 8);
    l[r] = fma(l[r], fac[r], float(t) * bs[r]);
  }
  auto vsl = Vt.template slice<128, BK>(0, k0);
  opO.run(pT, vsl, oT);
#pragma clang loop unroll(full)
  for (int i = 0; i < OC; i++) { const int sl = ((i >> 2) & 1) + 2 * (i >> 6); of[i] = fma(of[i], fac[sl], float(oT[i]) * bs[sl]); }
}
#pragma clang loop unroll(full)
for (int g = 0; g < OC; g += 4) {
  const int slot = ((g >> 2) & 1) + 2 * (g >> 6);
  const int row = q0 + rb + 8 * slot;
  const int col = cb + 16 * ((g >> 3) & 7);
  if (row < LQ) {
    const float4 vc = max(*(const device float4*)(vm + (size_t)kvh * 128 + col), float4(1e-20f)) * (1.0f / (127.0f * l[slot]));
    *(device vec<T, 4>*)(o + ((size_t)row * H + head) * 128 + col) =
        vec<T, 4>(T(of[g] * vc.x), T(of[g + 1] * vc.y), T(of[g + 2] * vc.z), T(of[g + 3] * vc.w));
  }
}