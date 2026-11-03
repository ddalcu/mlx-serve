// One threadgroup: query row l, output columns [128 x, 128 x + 128) of all H heads: the probabilities `p` [L, H, NK]
// against the same keys `attn_qk` read (now as values), 32 keys a step staged in threadgroup memory a step ahead; four
// simdgroups of H / 4 heads on the tensor units. Absent keys stage as zeros. GROUPED: the last RD channels leave
// inverse-roped at position meta[6] + l (`rope_tail`'s rotation on the f32 sums) and the heads land as `attnOut`'s
// grouped projection reads them, [G, L, H / G * D]; else [L, H, D] as summed.
constexpr int KS = 32;
constexpr int DT = 128;
threadgroup T Vs[2][KS * DT];
const int l = int(threadgroup_position_in_grid.y);
const int dt = int(threadgroup_position_in_grid.x);
const int wp = meta[0], W = meta[1], Wk = meta[2], K = meta[3];
const int H = meta[4];
const int D = meta[5];
const int NK = W + K;
const uint lid = thread_index_in_threadgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int kk = int(lid >> 2);
const int part = int(lid & 3);
uint4 reg[4];
auto fetch = [&](int ks) {
  const int c = ks * KS + kk;
  const device T* vrow;
  bool live;
  if (c < W) {
    const int r = wp + l - (W - 1) + c;
    live = r >= 0 && r < Wk;
    vrow = kw + size_t(max(r, 0)) * D;
  } else {
    const int r = picks[size_t(l) * K + (c - W)];
    live = r >= 0;
    vrow = comp + size_t(max(r, 0)) * D;
  }
  const device uint4* src = (const device uint4*)(vrow + dt * DT + part * 32);
#pragma clang loop unroll(full)
  for (int i = 0; i < 4; i++) reg[i] = live ? src[i] : uint4(0);
};
auto store = [&](int buf) {
  threadgroup uint4* dst = (threadgroup uint4*)(Vs[buf] + kk * DT + part * 32);
#pragma clang loop unroll(full)
  for (int i = 0; i < 4; i++) dst[i] = reg[i];
};
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    HS, DT, KS, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
tensor<device PT, dextents<int, 2>, tensor_inline> P((device PT*)(p + size_t(l) * H * NK), dextents<int, 2>(NK, H));
auto p0 = P.template slice<KS, HS>(0, int(sg) * HS);
tensor<threadgroup T, dextents<int, 2>, tensor_inline> B0(Vs[0], dextents<int, 2>(DT, KS));
auto cT = op.template get_destination_cooperative_tensor<decltype(p0), decltype(B0), float>();
#pragma clang loop unroll(full)
for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;
const int steps = NK / KS;
fetch(0);
store(0);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int ks = 0; ks < steps; ks++) {
  const int buf = ks & 1;
  const bool more = ks + 1 < steps;
  if (more) fetch(ks + 1);
  auto pa = P.template slice<KS, HS>(ks * KS, int(sg) * HS);
  tensor<threadgroup T, dextents<int, 2>, tensor_inline> Bt(Vs[buf], dextents<int, 2>(DT, KS));
  op.run(pa, Bt, cT);
  if (more) store(buf ^ 1);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (GROUPED) {
  const int L = meta[7], G = meta[8];
  const int hg = H / G;
  const float pos = float(meta[6] + l);
  const int head = D - RD;
#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < cT.get_capacity(); i += 2) {
    const auto ix = cT.get_multidimensional_index(i);
    const int d = dt * DT + int(ix[0]);
    const int h = int(sg) * HS + int(ix[1]);
    float a = cT[i], b = cT[i + 1];
    if (d >= head) {
      const float theta = pos * (1.0f / periods[(d - head) / 2]);
      const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
      const float ra = a * cs - b * sn;
      b = a * sn + b * cs;
      a = ra;
    }
    device OT* dst = o + (size_t(h / hg) * L + l) * size_t(hg * D) + size_t(h % hg) * D + d;
    *(device vec<OT, 2>*)dst = vec<OT, 2>(OT(a), OT(b));
  }
} else {
  auto oT = op.template get_destination_cooperative_tensor<decltype(p0), decltype(B0), OT>();
#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) oT[i] = OT(cT[i]);
  tensor<device OT, dextents<int, 2>, tensor_inline> O((device OT*)(o + size_t(l) * H * D), dextents<int, 2>(D, H));
  auto oo = O.template slice<DT, HS>(dt * DT, int(sg) * HS);
  oT.store(oo);
}
