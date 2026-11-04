// `gather_nax` over a gate and an up bank at once, then the clipped SwiGLU times each pair's weight: threadgroup (x, t)
// is tile t's 64 rows times 64 outputs of both banks; threads 0-63 decode gate rows into `wt` rows 0-63, threads
// 64-127 up rows into 64-127, simdgroups 0-1 multiply the gate's two 32-output halves and 2-3 the up's. The epilogue
// rounds each sum to T (as the two gathers would write it), then min(g, lim) and clamp(u, -lim, lim) when lim > 0,
// g * sigmoid(g) * u * wts[row] in f32, rounded once to T.
const uint tid = thread_index_in_threadgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int t = int(threadgroup_position_in_grid.y);
if (t >= tiles[E]) return;
int lo = 0, hi = E - 1;
while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (tiles[mid] <= t) lo = mid; else hi = mid - 1;
}
const int m0 = offs[lo] + (t - tiles[lo]) * 64;
const int live = min(64, offs[lo + 1] - m0);
const int a0 = min(m0, int(x_shape[0]) - 64);
const int shift = m0 - a0;
const uint n0 = threadgroup_position_in_grid.x * 64;
const device uint8_t *we = (tid < 64 ? wg : wu) + size_t(lo) * N * ROWB;
const uint dr = n0 + tid % 64;
constexpr uint KS = 2 * UNIT;
constexpr uint WS = KS + 8;
threadgroup T *wt = (threadgroup T *)buf;
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(32, 32, KS, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
tensor<device T, dextents<int, 2>, tensor_inline> A((device T *)x, dextents<int, 2>(K, int(x_shape[0])));
tensor<threadgroup T, dextents<int, 2>, tensor_inline> B(wt + 32 * sg * WS, dextents<int, 2>(KS, 32), array<int, 2>{1, WS});
auto tA0 = A.template slice<KS, 32>(0, a0);
auto tB = B.template slice<KS, 32>(0, 0);
auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(tB)>, float>();
auto cT1 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(tB)>, float>();
for (uint16_t i = 0; i < cT.get_capacity(); i++) {
    cT[i] = 0.0f;
    cT1[i] = 0.0f;
}
GgRow<T> o0 = {wt + tid * WS};
GgRow<T> o1 = {wt + tid * WS + UNIT};
for (uint u = 0; u < K / UNIT; u += 2) {
    DEQ_AT(we, dr, u / UPB, u % UPB, o0);
    DEQ_AT(we, dr, (u + 1) / UPB, (u + 1) % UPB, o1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shift < 32) {
        auto tA = A.template slice<KS, 32>(u * UNIT, a0);
        op.run(tA, tB, cT);
    }
    if (shift + live > 32) {
        auto tA1 = A.template slice<KS, 32>(u * UNIT, a0 + 32);
        op.run(tA1, tB, cT1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
threadgroup float *stage = buf;
tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(stage + 1024 * sg, dextents<int, 2>(32, 32), array<int, 2>{1, 32});
const float l = lim[0];
for (int h = 0; h < 2; h++) {
    const int r0 = max(32 * h, shift), r1 = min(32 * h + 32, shift + live);
    if (r0 >= r1) continue;
    if (h == 0) cT.store(C); else cT1.store(C);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 32 * 64; i += 128) {
        const int r = 32 * h + int(i / 64);
        const uint c = i % 64;
        if (r < r0 || r >= r1) continue;
        const uint at = (c / 32) * 1024 + uint(r - 32 * h) * 32 + c % 32;
        float g = float(T(stage[at])), v = float(T(stage[2048 + at]));
        if (l > 0.0f) { g = metal::min(g, l); v = metal::clamp(v, -l, l); }
        const float sgm = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
        out[size_t(a0 + r) * N + n0 + c] = T(g * (g < 0.0f ? sgm : 1.0f - sgm) * v * wts[a0 + r]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
