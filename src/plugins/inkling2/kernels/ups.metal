// Inkling 2, a few rows: the picked experts' gate and up projections and the soft-capped SwiGLU in one launch, the
// packed bank read in place. grid (32, I / ROWS, S * KSEL), threadgroup (32, SGS, 1): a SIMD group owns ROWS
// intermediate rows of one pick (their gate rows 2r and up rows 2r + 1 of WQ [E, 2 I, K * bits / 32] u32, scales WS
// [E, 2 I, K / GS] u8) and a lane 16 values of its row's x per sweep, kept in registers across the rows. FMT 3 is an
// affine bank (BITS bits, scales WS and biases WB [E, 2 I, K / GS] of S).
// X [S, K] (T); IDX [S * KSEL] u32 in (row, pick) order; each row's sum is rounded to T as the bank's matmul leaves
// it, then times SC[IDX[e]] with SCALED (f32 [E]). PR = { 1 / gate cap, gate cap, 1 / up cap, up cap }.
// ACT [S * KSEL, 1, I] (T) = cap_g tanh(g / cap_g) * cap_u tanh(u / cap_u) * sigmoid(g).
const uint lane = thread_index_in_simdgroup;
const uint row0 = thread_position_in_grid.y * ROWS;
const uint e = thread_position_in_grid.z;
const device T* x = X + size_t(e / KSEL) * K;
constexpr uint WPR = FMT == 2 ? K / 4 : K / 8;
constexpr uint WBR = K * BITS / 8;
constexpr uint SPR = K / GS;
const size_t base = size_t(IDX[e]) * (2 * I) + 2 * row0;
const device uint* wrow = WQ + base * WPR;
const device uchar* srow = (const device uchar*)WS + base * SPR;
const device uchar* arow = (const device uchar*)WQ + base * WBR;
const device S* sa = (const device S*)WS + base * SPR;
const device S* ba = (const device S*)WB + base * SPR;
float acc[2 * ROWS];
for (uint r = 0; r < 2 * ROWS; r++) acc[r] = 0.0f;
for (uint k = lane * 16; k < K; k += 512) {
    float xr[16];
    for (uint i = 0; i < 16; i++) xr[i] = float(x[k + i]);
    for (uint r = 0; r < 2 * ROWS; r++)
        acc[r] += FMT == 3 ? msv_ink2_affine_chunk<S, BITS, GS>(arow + r * WBR, sa + r * SPR, ba + r * SPR, k / 16, xr)
                           : msv_ink2_chunk(FMT, wrow + r * WPR, srow + r * SPR, k / 16, xr);
}
const float sc = SCALED ? SC[IDX[e]] : 1.0f;
for (uint r = 0; r < ROWS; r++) {
    const float g = float(T(simd_sum(acc[2 * r]))) * sc;
    const float u = float(T(simd_sum(acc[2 * r + 1]))) * sc;
    // The precise variants: the fast tanh is NaN at an infinity and a percent off near zero.
    const float capped = (precise::tanh(g * PR[0]) * PR[1]) * (precise::tanh(u * PR[2]) * PR[3]);
    if (lane == r) ACT[e * I + row0 + r] = T(capped / (1.0f + precise::exp(-g)));
}
