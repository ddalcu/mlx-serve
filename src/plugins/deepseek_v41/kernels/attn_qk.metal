// One threadgroup: query row l against 64 of its keys (tile x): the window's (key c < W is keys_w row wp + l - W + 1 + c,
// absent below 0) or the selected compressed rows (`picks`, absent below 0), all H heads. Four simdgroups of H / 4 heads
// contract D in 64-deep steps on the tensor units, the keys staged in threadgroup memory a step ahead. Scores [L, H, NK]
// leave in f32, unscaled; absent keys score 0 here and are masked by the softmax.
constexpr int KT = 64;
constexpr int DK = 64;
threadgroup T Ks[2][KT * DK];
const int l = int(threadgroup_position_in_grid.y);
const int kt = int(threadgroup_position_in_grid.x);
const int wp = meta[0], W = meta[1], Wk = meta[2], K = meta[3];
const int H = q_shape[1];
const int D = q_shape[2];
const int NK = W + K;
const uint lid = thread_index_in_threadgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int kk = int(lid >> 1);
const int part = int(lid & 1);
const int c = kt * KT + kk;
const device T* krow;
bool live;
if (c < W) {
  const int r = wp + l - (W - 1) + c;
  live = r >= 0 && r < Wk;
  krow = kw + size_t(max(r, 0)) * D;
} else {
  const int r = picks[size_t(l) * K + (c - W)];
  live = r >= 0;
  krow = comp + size_t(max(r, 0)) * D;
}
uint4 reg[4];
auto fetch = [&](int ds) {
  const device uint4* src = (const device uint4*)(krow + ds * DK + part * 32);
#pragma clang loop unroll(full)
  for (int i = 0; i < 4; i++) reg[i] = live ? src[i] : uint4(0);
};
auto store = [&](int buf) {
  threadgroup uint4* dst = (threadgroup uint4*)(Ks[buf] + kk * DK + part * 32);
#pragma clang loop unroll(full)
  for (int i = 0; i < 4; i++) dst[i] = reg[i];
};
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    HS, KT, DK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
tensor<device T, dextents<int, 2>, tensor_inline> Q((device T*)(q + size_t(l) * H * D), dextents<int, 2>(D, H));
auto q0 = Q.template slice<DK, HS>(0, int(sg) * HS);
tensor<threadgroup T, dextents<int, 2>, tensor_inline> B0(Ks[0], dextents<int, 2>(DK, KT));
auto cT = op.template get_destination_cooperative_tensor<decltype(q0), decltype(B0), float>();
#pragma clang loop unroll(full)
for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;
const int steps = D / DK;
fetch(0);
store(0);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int ds = 0; ds < steps; ds++) {
  const int buf = ds & 1;
  const bool more = ds + 1 < steps;
  if (more) fetch(ds + 1);
  auto qa = Q.template slice<DK, HS>(ds * DK, int(sg) * HS);
  tensor<threadgroup T, dextents<int, 2>, tensor_inline> Bt(Ks[buf], dextents<int, 2>(DK, KT));
  op.run(qa, Bt, cT);
  if (more) store(buf ^ 1);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
tensor<device float, dextents<int, 2>, tensor_inline> S((device float*)(s + size_t(l) * H * NK), dextents<int, 2>(NK, H));
auto so = S.template slice<KT, HS>(kt * KT, int(sg) * HS);
cT.store(so);
