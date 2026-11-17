const int K = xq_shape[1];
const int M = xq_shape[0];
const int N = wq_shape[0];
const int m0 = int(threadgroup_position_in_grid.y) * BM;
const int n0 = int(threadgroup_position_in_grid.x) * BN;
constexpr auto desc = matmul2d_descriptor(BM, BN, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<4>> op;
// Each tile's tensors start at its own rows (64-bit offsets): a whole-matrix tensor indexes in int and wraps past 2^31.
tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq + (size_t)m0 * K, dextents<int, 2>(K, M - m0), array<int, 2>({1, K}));
tensor<device int8_t, dextents<int, 2>, tensor_inline> B((device int8_t*)wq + (size_t)n0 * K, dextents<int, 2>(K, N - n0), array<int, 2>({1, K}));
auto a0 = A.template slice<64, BM>(0, 0);
auto b0 = B.template slice<64, BN>(0, 0);
auto cT = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0;
for (int k0 = 0; k0 < K; k0 += 64) {
  auto a = A.template slice<64, BM>(k0, 0);
  auto b = B.template slice<64, BN>(k0, 0);
  op.run(a, b, cT);
}
