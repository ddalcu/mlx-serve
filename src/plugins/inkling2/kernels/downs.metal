// Inkling 2, a few rows: the picked experts' down projections and their weighted sum in one launch, the packed
// bank read in place; with TAIL (one row) the MLP's short conv and the final add too. grid (32, KSEL, S * D / ROWS),
// threadgroup (32, KSEL, 1): a SIMD group owns ROWS output rows of one pick of one row (rows of WQ [E, D, I * bits /
// 32] u32, scales WS [E, D, I / GS] u8; FMT 3 an affine bank of BITS bits, scales WS and biases WB [E, D, I / GS] of S), a lane 16 values of ACT [S * KSEL, 1, I] (T) per sweep; the threadgroup
// holds every pick of those rows. Each pick's row sum is rounded to T as the bank's matmul leaves it, times
// SC[IDX[p]] with SCALED (f32 [E]) and its routing weight WTS[p] (f32); their sum in pick order, rounded to T, is
// the MLP's output XM [1, S, D]. TAIL: then the conv behind its window (the last D columns of ST [1, KT - 1, SW]
// f32) with MW [D, KT, 1] (WT), its sum rounded to WT as the conv op does, plus XM, times PR[0], rounded to T, and
// the residual RES [D] f32 added: HOUT [1, 1, D] (T), the visit's output stream.
const uint lane = thread_index_in_simdgroup;
const uint e = thread_position_in_grid.y;
const uint srow = thread_position_in_grid.z / (D / ROWS);
const uint row0 = (thread_position_in_grid.z % (D / ROWS)) * ROWS;
const uint p = srow * KSEL + e;
constexpr uint WPR = FMT == 2 ? I / 4 : I / 8;
constexpr uint WBR = I * BITS / 8;
constexpr uint SPR = I / GS;
const uint SW = ST_shape[2];
const uint OFF = SW - D;
threadgroup float part[KSEL * ROWS];
const size_t base = size_t(IDX[p]) * D + row0;
const device uint* wrow = WQ + base * WPR;
const device uchar* scales = (const device uchar*)WS + base * SPR;
const device uchar* arow = (const device uchar*)WQ + base * WBR;
const device S* sa = (const device S*)WS + base * SPR;
const device S* ba = (const device S*)WB + base * SPR;
float acc[ROWS];
for (uint r = 0; r < ROWS; r++) acc[r] = 0.0f;
for (uint k = lane * 16; k < I; k += 512) {
    float xr[16];
    for (uint i = 0; i < 16; i++) xr[i] = float(ACT[size_t(p) * I + k + i]);
    for (uint r = 0; r < ROWS; r++)
        acc[r] += FMT == 3 ? msv_ink2_affine_chunk<S, BITS, GS>(arow + r * WBR, sa + r * SPR, ba + r * SPR, k / 16, xr)
                           : msv_ink2_chunk(FMT, wrow + r * WPR, scales + r * SPR, k / 16, xr);
}
for (uint r = 0; r < ROWS; r++) {
    const float v = float(T(simd_sum(acc[r]))) * (SCALED ? SC[IDX[p]] : 1.0f) * WTS[p];
    if (lane == 0) part[e * ROWS + r] = v;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (e == 0 && lane < ROWS) {
    const uint c = row0 + lane;
    float x = 0.0f;
    for (uint k = 0; k < KSEL; k++) x += part[k * ROWS + lane];
    const T xm = T(x);
    XM[size_t(srow) * D + c] = xm;
    if (!TAIL) return;
    float y = 0.0f;
    for (uint j = 0; j < KT; j++) y += ((j + 1 < KT) ? ST[j * SW + OFF + c] : float(xm)) * float(MW[c * KT + j]);
    y = float(WT(y));
    HOUT[c] = T(RES[c] + float(T((y + float(xm)) * PR[0])));
}
