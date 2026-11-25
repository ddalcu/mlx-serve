// Inkling 2, multi-row quantized projection (2 output rows per SIMD group): reads each packed weight row ONCE via
// 128-bit uint4 loads across M = 2..8 activation rows with zero threadgroup memory and zero barriers.
// X [1, M, K] (T), WQ [N, K*BITS/32] u32, WS [N, K/GS] (S), WB [N, K/GS] (S) -> Y [1, M, N] (T).
// grid (32, N / 2, 1), threadgroup (32, SGS, 1).
const uint lane = thread_index_in_simdgroup;
const uint row0 = thread_position_in_grid.y * 2u;

constexpr uint WPR = FMT == 2 ? K / 4 : K / 8;
constexpr uint WBR = K * BITS / 8;
constexpr uint SPR = K / GS;
const device uint* wrow0 = WQ + size_t(row0) * WPR;
const device uchar* srow0 = (const device uchar*)WS + size_t(row0) * SPR;
const device uchar* arow0 = (const device uchar*)WQ + size_t(row0) * WBR;
const device S* sa0 = (const device S*)WS + size_t(row0) * SPR;
const device S* ba0 = (const device S*)WB + size_t(row0) * SPR;

float acc0[M];
float acc1[M];
#pragma unroll
for (int m = 0; m < M; ++m) {
    acc0[m] = 0.0f;
    acc1[m] = 0.0f;
}

for (uint c = lane; c < K / 16u; c += 32u) {
    float4 w0[4], w1[4];
    if (FMT == 3) {
        msv_ink2_decode16_affine<S, BITS, GS>(arow0, sa0, ba0, c, w0);
        msv_ink2_decode16_affine<S, BITS, GS>(arow0 + WBR, sa0 + SPR, ba0 + SPR, c, w1);
    } else {
        msv_ink2_decode16_fp(FMT, wrow0, srow0, c, w0);
        msv_ink2_decode16_fp(FMT, wrow0 + WPR, srow0 + SPR, c, w1);
    }
    const uint k4 = c * 4u;
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        const device vec<T, 4>* xm = (const device vec<T, 4>*)(X + size_t(m) * K);
        const float4 v0 = float4(xm[k4 + 0u]);
        const float4 v1 = float4(xm[k4 + 1u]);
        const float4 v2 = float4(xm[k4 + 2u]);
        const float4 v3 = float4(xm[k4 + 3u]);
        acc0[m] += dot(w0[0], v0) + dot(w0[1], v1) + dot(w0[2], v2) + dot(w0[3], v3);
        acc1[m] += dot(w1[0], v0) + dot(w1[1], v1) + dot(w1[2], v2) + dot(w1[3], v3);
    }
}

#pragma unroll
for (int m = 0; m < M; ++m) {
    const float y0 = simd_sum(acc0[m]);
    const float y1 = simd_sum(acc1[m]);
    if (lane == 0u) {
        Y[size_t(m) * N + row0 + 0u] = T(y0);
        Y[size_t(m) * N + row0 + 1u] = T(y1);
    }
}
