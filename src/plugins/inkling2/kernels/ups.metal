// Inkling 2, a few rows: the router's picks, the picked experts' gate and up projections and the soft-capped SwiGLU
// in one launch, the packed bank read in place. grid (32, I / ROWS, S * KSEL), threadgroup (32, SGS, 1): a SIMD group
// owns ROWS intermediate rows of one pick (their gate rows 2r and up rows 2r + 1 of WQ [E, 2 I, K * bits / 32] u32,
// scales WS [E, 2 I, K / GS] u8) and a lane 16 values of its row's x per sweep, kept in registers across the rows.
// FMT 3 is an affine bank (BITS bits, scales WS and biases WB [E, 2 I, K / GS] of S); FAST3 takes its 3-bit word path.
// With ROUTE each threadgroup ranks its row's picks as topk.metal does: SCR [S, E] f32 are the router's logits, RB [E]
// f32 the selection bias, RS [1] the route scale; a pick's threadgroup of the first rows writes its PIDX and PWTS [1,
// S, KSEL] (u32, f32), topk.metal's IDX and WTS, the pairs in (row, pick) order. Without, the picks are IDX [S * KSEL]
// u32 in that order (more rows rank in one topk launch: here every threadgroup would rank again).
// X [S, K] (T); each row's sum is rounded to T as the bank's matmul leaves it, then times SC[pick] with SCALED (f32
// [E]). PR = { 1 / gate cap, gate cap, 1 / up cap, up cap }.
// ACT [S * KSEL, 1, I] (T) = cap_g tanh(g / cap_g) * cap_u tanh(u / cap_u) * sigmoid(g).
const uint lane = thread_index_in_simdgroup;
const uint row0 = thread_position_in_grid.y * ROWS;
const uint e = thread_position_in_grid.z;
const uint tid = thread_position_in_threadgroup.x + 32 * thread_position_in_threadgroup.y;
constexpr uint TGN = 32 * SGS;
threadgroup float key[E];
threadgroup float raw[KSEL];
threadgroup uint pick[KSEL];
uint ex = 0;
if (ROUTE) {
    const uint srow0 = (e / KSEL) * E;
    float sig[(E + TGN - 1) / TGN];
    for (uint f = tid, j = 0; f < E; f += TGN, j++) {
        sig[j] = 1.0f / (1.0f + exp(-SCR[srow0 + f]));
        key[f] = sig[j] + RB[f];
    }
    // A NaN score ranks nowhere: every slot keeps an expert that exists.
    if (tid < KSEL) { pick[tid] = tid; raw[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint f = tid, j = 0; f < E; f += TGN, j++) {
        const float mine = key[f];
        uint rank = 0;
        for (uint h = 0; h < E; h++) rank += (key[h] > mine || (key[h] == mine && h < f)) ? 1u : 0u;
        if (rank < KSEL && mine == mine) { pick[rank] = f; raw[rank] = sig[j]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ex = pick[e % KSEL];
    if (thread_position_in_grid.y == 0 && tid == 0) {
        float tot = 0.0f;
        for (uint k = 0; k < KSEL; k++) tot += raw[k];
        PIDX[e] = ex;
        PWTS[e] = raw[e % KSEL] / (tot + 1e-20f) * RS[0];
    }
} else ex = IDX[e];
const device T* x = X + size_t(e / KSEL) * K;
constexpr uint WPR = FMT == 2 ? K / 4 : K / 8;
constexpr uint WBR = K * BITS / 8;
constexpr uint SPR = K / GS;
const size_t base = size_t(ex) * (2 * I) + 2 * row0;
const device uint* wrow = WQ + base * WPR;
const device uchar* srow = (const device uchar*)WS + base * SPR;
const device uchar* arow = (const device uchar*)WQ + base * WBR;
const device S* sa = (const device S*)WS + base * SPR;
const device S* ba = (const device S*)WB + base * SPR;
float acc[2 * ROWS];
for (uint r = 0; r < 2 * ROWS; r++) acc[r] = 0.0f;
for (uint k = lane * 16; k < K; k += 512) {
    const device vec<T, 4>* xp = (const device vec<T, 4>*)(x + k);
    const float4 v0 = float4(xp[0]), v1 = float4(xp[1]), v2 = float4(xp[2]), v3 = float4(xp[3]);
    float xr[16] = {
        v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w,
        v2.x, v2.y, v2.z, v2.w, v3.x, v3.y, v3.z, v3.w,
    };
    float xs[16];
    float sx = 0.0f;
    if (FMT == 3 && BITS == 3 && FAST3) { for (uint i = 0; i < 16; i++) sx += xr[i]; msv_ink2_prescale3w(xr, xs); }
    for (uint r = 0; r < 2 * ROWS; r++)
        acc[r] += FMT == 3 ? (BITS == 3 && FAST3 ? msv_ink2_affine3w<S, GS>(arow + r * WBR, sa + r * SPR, ba + r * SPR, k / 16, xs, sx)
                                        : msv_ink2_affine_chunk<S, BITS, GS>(arow + r * WBR, sa + r * SPR, ba + r * SPR, k / 16, xr))
                           : msv_ink2_chunk(FMT, wrow + r * WPR, srow + r * SPR, k / 16, xr);
}
const float sc = SCALED ? SC[ex] : 1.0f;
for (uint r = 0; r < ROWS; r++) {
    const float g = float(T(simd_sum(acc[2 * r]))) * sc;
    const float u = float(T(simd_sum(acc[2 * r + 1]))) * sc;
    // The precise variants: the fast tanh is NaN at an infinity and a percent off near zero.
    const float capped = (precise::tanh(g * PR[0]) * PR[1]) * (precise::tanh(u * PR[2]) * PR[3]);
    if (lane == r) ACT[e * I + row0 + r] = T(capped / (1.0f + precise::exp(-g)));
}
