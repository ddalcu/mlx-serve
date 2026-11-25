// Inkling 2, a few rows: the picked experts' down projections and their weighted sum in one launch, the packed
// bank read in place via 128-bit uint4 loads; with TAIL (one row) the MLP's short conv and the final add too.
// grid (32, KSEL, S * D / ROWS), threadgroup (32, KSEL, 1): a SIMD group owns ROWS output rows of one pick of one row.
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
const device T* act_p = ACT + size_t(p) * I;
if (FMT == 3) {
    for (uint c = lane; c < I / 16u; c += 32u) {
        const device vec<T, 4>* ap = (const device vec<T, 4>*)(act_p + c * 16u);
        const float4 x0 = float4(ap[0]), x1 = float4(ap[1]), x2 = float4(ap[2]), x3 = float4(ap[3]);
        const float4 s4 = x0 + x1 + x2 + x3;
        const float sx = (s4.x + s4.y) + (s4.z + s4.w);
        for (uint r = 0; r < ROWS; r++)
            acc[r] += msv_ink2_affine_chunk_f4<S, BITS, GS>(arow + r * WBR, sa + r * SPR, ba + r * SPR, c, x0, x1, x2, x3, sx);
    }
} else {
    for (uint k = lane * 16; k < I; k += 512) {
        float xr[16];
        for (uint i = 0; i < 16; i++) xr[i] = float(act_p[k + i]);
        for (uint r = 0; r < ROWS; r++)
            acc[r] += msv_ink2_chunk(FMT, wrow + r * WPR, scales + r * SPR, k / 16, xr);
    }
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
