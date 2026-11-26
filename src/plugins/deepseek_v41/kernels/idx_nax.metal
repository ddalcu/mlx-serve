constexpr int WM = BM / 16;
constexpr int WN = BN / 32;
constexpr int NSG = WM * WN;
constexpr int TGN = NSG * 32;

const int L = q_shape[0];
const int nb = keys_shape[0];
const int l0 = int(threadgroup_position_in_grid.x) * BM;
const int n0 = int(threadgroup_position_in_grid.y) * BN;
const uint tid = thread_index_in_threadgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int p0 = meta[0], ratio = meta[1];
const int rows_left = metal::min(BM, L - l0);
const int max_seen = (p0 + l0 + rows_left) / ratio;
const int end = metal::min(nb, n0 + BN);
if (n0 >= max_seen) {
  for (int r = 0; r < rows_left; ++r)
    for (int n = n0 + int(tid); n < end; n += TGN) scores[size_t(l0 + r) * nb + n] = -INFINITY;
  return;
}

constexpr int C4 = IHD / 4;
using T4 = metal::vec<T, 4>;
threadgroup T kt[BN * IHD];
for (uint i = tid; i < uint(BN * C4); i += uint(TGN)) {
  const int n = n0 + int(i / C4);
  ((threadgroup T4*)kt)[i] = n < nb ? ((const device T4*)keys)[size_t(n) * C4 + i % C4] : T4(0);
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

const int sg_m = int(sg) / WN;
const int sg_n = int(sg) % WN;
const int sg_l0 = l0 + sg_m * 16;
if (sg_l0 >= L) return;

const int sg_n0 = n0 + sg_n * 32;
const int sg_rows = metal::min(16, L - sg_l0);
const ushort qid = ushort(lane >> 2);
const ushort fm = (qid & 4) | ((ushort(lane) >> 1) & 3);
const ushort fn = ((qid & 2) | (ushort(lane) & 1)) * 4;
const int r0 = sg_l0 + int(fm);
const int r1 = r0 + 8;
const int r0_safe = metal::min(r0, L - 1);
const int r1_safe = metal::min(r1, L - 1);
const float ws = wsc[0];

using ext_t = metal::dextents<int, 2>;
using str_t = metal::array<int, 2>;
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, DK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

tensor<threadgroup T, ext_t, metal::tensor_inline> B(kt + sg_n * 32 * IHD, ext_t(IHD, 32));
auto b0 = B.template slice<DK, 32>(0, 0);
tensor<device T, ext_t, metal::tensor_inline> A0(
    (device T*)q + size_t(sg_l0) * IH * IHD, ext_t(IHD, sg_rows), str_t{1, IH * IHD});
auto a0 = A0.template slice<DK, 16>(0, 0);
auto ct_c = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();

float acc[16];
#pragma clang loop unroll(full)
for (short e = 0; e < 16; ++e) acc[e] = 0.0f;

for (int h = 0; h < IH; ++h) {
  const float w0 = float(w[size_t(r0_safe) * IH + h]) * ws;
  const float w1 = float(w[size_t(r1_safe) * IH + h]) * ws;
  tensor<device T, ext_t, metal::tensor_inline> A(
      (device T*)q + (size_t(sg_l0) * IH + h) * IHD, ext_t(IHD, sg_rows), str_t{1, IH * IHD});
  #pragma clang loop unroll(full)
  for (short e = 0; e < 16; ++e) ct_c[e] = 0.0f;
  for (int k0 = 0; k0 < IHD; k0 += DK) {
    auto a = A.template slice<DK, 16>(k0, 0);
    auto b = B.template slice<DK, 32>(k0, 0);
    op.run(a, b, ct_c);
  }
  #pragma clang loop unroll(full)
  for (short j = 0; j < 4; ++j) {
    const float d0 = ct_c[j], d1 = ct_c[4 + j], d2 = ct_c[8 + j], d3 = ct_c[12 + j];
    acc[j]      += w0 * ((d0 < 0.0f) ? 0.0f : d0);
    acc[4 + j]  += w1 * ((d1 < 0.0f) ? 0.0f : d1);
    acc[8 + j]  += w0 * ((d2 < 0.0f) ? 0.0f : d2);
    acc[12 + j] += w1 * ((d3 < 0.0f) ? 0.0f : d3);
  }
}

const int seen0 = (p0 + r0 + 1) / ratio;
const int seen1 = (p0 + r1 + 1) / ratio;
#pragma clang loop unroll(full)
for (short j = 0; j < 4; ++j) {
  const int c0 = sg_n0 + int(fn) + int(j);
  const int c1 = c0 + 16;
  if (r0 < L) {
    if (c0 < nb) scores[size_t(r0) * nb + c0] = c0 < seen0 ? acc[j] : -INFINITY;
    if (c1 < nb) scores[size_t(r0) * nb + c1] = c1 < seen0 ? acc[8 + j] : -INFINITY;
  }
  if (r1 < L) {
    if (c0 < nb) scores[size_t(r1) * nb + c0] = c0 < seen1 ? acc[4 + j] : -INFINITY;
    if (c1 < nb) scores[size_t(r1) * nb + c1] = c1 < seen1 ? acc[12 + j] : -INFINITY;
  }
}
