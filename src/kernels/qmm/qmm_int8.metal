const int K = xq_shape[xq_ndim - 1];
const int N = w_shape[0];
const int Kg = K / 128;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int ms = int(threadgroup_position_in_grid.y) * 32;
const int ns = int(threadgroup_position_in_grid.x) * 128 + 32 * int(sg);
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    32, 32, 128, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, MPAD));
auto bT = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
thread uint32_t* bw = (thread uint32_t*)&bT;
auto tA0 = A.template slice<128, 32>(0, ms);
auto cT = op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(bT)>, int32_t>();
constexpr int CAP = 32;
const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
const int nb = ns + fn;
const int mb = ms + fm;
float acc[CAP];
#pragma clang loop unroll(full)
for (int i = 0; i < CAP; i++) acc[i] = 0.0f;
const int nl = int(((lane >> 1) & 3) + 4 * ((lane >> 4) & 1));
const uint sh = 2 * ((lane & 1) + 2 * ((lane >> 3) & 1));
const device uint4* wc[4];
#pragma clang loop unroll(full)
for (int c = 0; c < 4; c++) wc[c] = (const device uint4*)(w + (size_t)min(ns + nl + 8 * c, N - 1) * (K / 16));
const int mrow[4] = {mb, mb + 8, mb + 16, mb + 24};
for (int g = 0; g < Kg; g++) {
  uint4 wv[8];
#pragma clang loop unroll(full)
  for (int c = 0; c < 4; c++) { wv[2 * c] = wc[c][2 * g]; wv[2 * c + 1] = wc[c][2 * g + 1]; }
#pragma clang loop unroll(full)
  for (int c = 0; c < 4; c++) {
    const uint s2 = sh;
    const uint4 lo = wv[2 * c];
    const uint4 hi = wv[2 * c + 1];
    bw[c + 0] = (lo.x >> s2) & 0x03030303u; bw[c + 4] = (lo.y >> s2) & 0x03030303u;
    bw[c + 8] = (lo.z >> s2) & 0x03030303u; bw[c + 12] = (lo.w >> s2) & 0x03030303u;
    bw[c + 16] = (hi.x >> s2) & 0x03030303u; bw[c + 20] = (hi.y >> s2) & 0x03030303u;
    bw[c + 24] = (hi.z >> s2) & 0x03030303u; bw[c + 28] = (hi.w >> s2) & 0x03030303u;
  }
  auto tA = A.template slice<128, 32>(g * 128, ms);
  op.run(tA, bT, cT);
  float4 sv[2], bv[2];
  float av[4], rv[4];
#pragma clang loop unroll(full)
  for (int h = 0; h < 2; h++) {
    if (FULL) {
      sv[h] = float4(*(const device vec<ST, 4>*)(scales + (size_t)g * N + nb + 16 * h));
      bv[h] = NEG ? float4(0.0f) : float4(*(const device vec<ST, 4>*)(biases + (size_t)g * N + nb + 16 * h));
    } else {
#pragma clang loop unroll(full)
      for (int c = 0; c < 4; c++) {
        const size_t e = (size_t)g * N + min(nb + c + 16 * h, N - 1);
        sv[h][c] = float(scales[e]);
        bv[h][c] = NEG ? 0.0f : float(biases[e]);
      }
    }
  }
#pragma clang loop unroll(full)
  for (int q = 0; q < 4; q++) {
    av[q] = ascale[(size_t)mrow[q] * Kg + g];
    rv[q] = rsum[(size_t)mrow[q] * Kg + g];
    if (NEG) rv[q] *= av[q];
  }
  // Ternary packs (bias == -scale): acc += s * (as * C - as * rs).
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; i++) {
    const int c = i & 3;
    const int nh = (i >> 3) & 1;
    const int mh = ((i >> 2) & 1) | (((i >> 4) & 1) << 1);
    if (NEG) {
      acc[i] = fma(sv[nh][c], fma(av[mh], float(cT[i]), -rv[mh]), acc[i]);
    } else {
      const float t = fma(sv[nh][c], float(cT[i]), bv[nh][c] * rv[mh]);
      acc[i] = fma(av[mh], t, acc[i]);
    }
  }
}
#pragma clang loop unroll(full)
for (int i = 0; i < CAP; i += 4) {
  const int nn = nb + 16 * ((i >> 3) & 1);
  const int mm = mb + 8 * ((i >> 2) & 1) + 16 * ((i >> 4) & 1);
  if (FULL) {
    *(device vec<T, 4>*)(y + (size_t)mm * N + nn) = vec<T, 4>(acc[i], acc[i + 1], acc[i + 2], acc[i + 3]);
  } else {
#pragma clang loop unroll(full)
    for (int c = 0; c < 4; c++) if (nn + c < N) y[(size_t)mm * N + nn + c] = static_cast<T>(acc[i + c]);
  }
}