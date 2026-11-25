// Inkling 2, grouped multi-row gate+up+SwiGLU: each unique active expert reads its packed weights ONCE via 128-bit
// uint4 loads and applies them to all rows in the block that selected it, using explicit scalar register accumulators
// (zero dynamic-index stack spills) and quarter-streamed 8-bit decoding to keep peak register pressure low.
// grid (32, I, NMAX), threadgroup (32, SGS, 1).
const uint u = thread_position_in_grid.z;
const uint ex = ACT_E[u];
if (ex == 0xffffffffu) return;

const uint tid = thread_position_in_threadgroup.y * 32u + thread_index_in_simdgroup;
const uint lane = thread_index_in_simdgroup;
const uint row0 = thread_position_in_grid.y;
const uint cnt = ACT_CNT[u];

threadgroup uint sh_pairs[S_CAP];
threadgroup uint sh_rows[S_CAP];
if (tid < cnt) {
    const uint p = ACT_PAIR[u * S_CAP + tid];
    sh_pairs[tid] = p;
    sh_rows[tid] = p / KSEL;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

constexpr uint WPR = FMT == 2 ? K / 4 : K / 8;
constexpr uint WBR = K * BITS / 8;
constexpr uint SPR = K / GS;
const size_t base = size_t(ex) * (2 * I) + 2 * row0;
const device uint* wrow = WQ + base * WPR;
const device uchar* srow = (const device uchar*)WS + base * SPR;
const device uchar* arow = (const device uchar*)WQ + base * WBR;
const device S* sa = (const device S*)WS + base * SPR;
const device S* ba = (const device S*)WB + base * SPR;
const float sc = SCALED ? SC[ex] : 1.0f;

if (cnt == 1u) {
    const uint p0 = sh_pairs[0];
    const device vec<T, 4>* x0 = (const device vec<T, 4>*)(X + size_t(sh_rows[0]) * K);
    float ag0 = 0.0f, au0 = 0.0f;
    for (uint c = lane; c < K / 16u; c += 32u) {
        const uint k4 = c * 4u;
        if (FMT == 3 && BITS == 8) {
            const uint g = (c * 16u) / GS;
            const float scg = float(sa[g]), big = float(ba[g]);
            const float scu = float(sa[SPR + g]), biu = float(ba[SPR + g]);
            const uint4 qg = ((const device uint4*)arow)[c];
            const uint4 qu = ((const device uint4*)(arow + WBR))[c];
            const float4 v0 = float4(x0[k4 + 0u]);
            ag0 += dot(float4(as_type<uchar4>(qg.x)) * scg + big, v0);
            au0 += dot(float4(as_type<uchar4>(qu.x)) * scu + biu, v0);
            const float4 v1 = float4(x0[k4 + 1u]);
            ag0 += dot(float4(as_type<uchar4>(qg.y)) * scg + big, v1);
            au0 += dot(float4(as_type<uchar4>(qu.y)) * scu + biu, v1);
            const float4 v2 = float4(x0[k4 + 2u]);
            ag0 += dot(float4(as_type<uchar4>(qg.z)) * scg + big, v2);
            au0 += dot(float4(as_type<uchar4>(qu.z)) * scu + biu, v2);
            const float4 v3 = float4(x0[k4 + 3u]);
            ag0 += dot(float4(as_type<uchar4>(qg.w)) * scg + big, v3);
            au0 += dot(float4(as_type<uchar4>(qu.w)) * scu + biu, v3);
        } else {
            float4 wg[4], wu[4];
            if (FMT == 3) {
                msv_ink2_decode16_affine<S, BITS, GS>(arow, sa, ba, c, wg);
                msv_ink2_decode16_affine<S, BITS, GS>(arow + WBR, sa + SPR, ba + SPR, c, wu);
            } else {
                msv_ink2_decode16_fp(FMT, wrow, srow, c, wg);
                msv_ink2_decode16_fp(FMT, wrow + WPR, srow + SPR, c, wu);
            }
            const float4 v0 = float4(x0[k4 + 0u]), v1 = float4(x0[k4 + 1u]), v2 = float4(x0[k4 + 2u]), v3 = float4(x0[k4 + 3u]);
            ag0 += dot(wg[0], v0) + dot(wg[1], v1) + dot(wg[2], v2) + dot(wg[3], v3);
            au0 += dot(wu[0], v0) + dot(wu[1], v1) + dot(wu[2], v2) + dot(wu[3], v3);
        }
    }
    const float g = float(T(simd_sum(ag0))) * sc;
    const float uv = float(T(simd_sum(au0))) * sc;
    const float capped = (precise::tanh(g * PR[0]) * PR[1]) * (precise::tanh(uv * PR[2]) * PR[3]);
    if (lane == 0u) ACT[size_t(p0) * I + row0] = T(capped / (1.0f + precise::exp(-g)));
    return;
}

if (cnt == 2u) {
    const uint p0 = sh_pairs[0], p1 = sh_pairs[1];
    const device vec<T, 4>* x0 = (const device vec<T, 4>*)(X + size_t(sh_rows[0]) * K);
    const device vec<T, 4>* x1 = (const device vec<T, 4>*)(X + size_t(sh_rows[1]) * K);
    float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f;
    for (uint c = lane; c < K / 16u; c += 32u) {
        const uint k4 = c * 4u;
        if (FMT == 3 && BITS == 8) {
            const uint g = (c * 16u) / GS;
            const float scg = float(sa[g]), big = float(ba[g]);
            const float scu = float(sa[SPR + g]), biu = float(ba[SPR + g]);
            const uint4 qg = ((const device uint4*)arow)[c];
            const uint4 qu = ((const device uint4*)(arow + WBR))[c];
            const uint qga[4] = { qg.x, qg.y, qg.z, qg.w };
            const uint qua[4] = { qu.x, qu.y, qu.z, qu.w };
            #pragma unroll
            for (uint q = 0u; q < 4u; ++q) {
                const float4 wg = float4(as_type<uchar4>(qga[q])) * scg + big;
                const float4 wu = float4(as_type<uchar4>(qua[q])) * scu + biu;
                const float4 v0 = float4(x0[k4 + q]);
                ag0 += dot(wg, v0); au0 += dot(wu, v0);
                const float4 v1 = float4(x1[k4 + q]);
                ag1 += dot(wg, v1); au1 += dot(wu, v1);
            }
        } else {
            float4 wg[4], wu[4];
            if (FMT == 3) {
                msv_ink2_decode16_affine<S, BITS, GS>(arow, sa, ba, c, wg);
                msv_ink2_decode16_affine<S, BITS, GS>(arow + WBR, sa + SPR, ba + SPR, c, wu);
            } else {
                msv_ink2_decode16_fp(FMT, wrow, srow, c, wg);
                msv_ink2_decode16_fp(FMT, wrow + WPR, srow + SPR, c, wu);
            }
            {
                const float4 v0 = float4(x0[k4 + 0u]), v1 = float4(x0[k4 + 1u]), v2 = float4(x0[k4 + 2u]), v3 = float4(x0[k4 + 3u]);
                ag0 += dot(wg[0], v0) + dot(wg[1], v1) + dot(wg[2], v2) + dot(wg[3], v3);
                au0 += dot(wu[0], v0) + dot(wu[1], v1) + dot(wu[2], v2) + dot(wu[3], v3);
            }
            {
                const float4 v0 = float4(x1[k4 + 0u]), v1 = float4(x1[k4 + 1u]), v2 = float4(x1[k4 + 2u]), v3 = float4(x1[k4 + 3u]);
                ag1 += dot(wg[0], v0) + dot(wg[1], v1) + dot(wg[2], v2) + dot(wg[3], v3);
                au1 += dot(wu[0], v0) + dot(wu[1], v1) + dot(wu[2], v2) + dot(wu[3], v3);
            }
        }
    }
    const float g0 = float(T(simd_sum(ag0))) * sc, u0 = float(T(simd_sum(au0))) * sc;
    const float g1 = float(T(simd_sum(ag1))) * sc, u1 = float(T(simd_sum(au1))) * sc;
    if (lane == 0u) {
        const float c0 = (precise::tanh(g0 * PR[0]) * PR[1]) * (precise::tanh(u0 * PR[2]) * PR[3]);
        const float c1 = (precise::tanh(g1 * PR[0]) * PR[1]) * (precise::tanh(u1 * PR[2]) * PR[3]);
        ACT[size_t(p0) * I + row0] = T(c0 / (1.0f + precise::exp(-g0)));
        ACT[size_t(p1) * I + row0] = T(c1 / (1.0f + precise::exp(-g1)));
    }
    return;
}

if (cnt <= 4u) {
    const uint p0 = sh_pairs[0], p1 = sh_pairs[1], p2 = sh_pairs[2], p3 = cnt > 3u ? sh_pairs[3] : p2;
    const device vec<T, 4>* x0 = (const device vec<T, 4>*)(X + size_t(sh_rows[0]) * K);
    const device vec<T, 4>* x1 = (const device vec<T, 4>*)(X + size_t(sh_rows[1]) * K);
    const device vec<T, 4>* x2 = (const device vec<T, 4>*)(X + size_t(sh_rows[2]) * K);
    const device vec<T, 4>* x3 = (const device vec<T, 4>*)(X + size_t(cnt > 3u ? sh_rows[3] : sh_rows[2]) * K);
    float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f, ag2 = 0.0f, au2 = 0.0f, ag3 = 0.0f, au3 = 0.0f;
    for (uint c = lane; c < K / 16u; c += 32u) {
        const uint k4 = c * 4u;
        if (FMT == 3 && BITS == 8) {
            const uint g = (c * 16u) / GS;
            const float scg = float(sa[g]), big = float(ba[g]);
            const float scu = float(sa[SPR + g]), biu = float(ba[SPR + g]);
            const uint4 qg = ((const device uint4*)arow)[c];
            const uint4 qu = ((const device uint4*)(arow + WBR))[c];
            const uint qga[4] = { qg.x, qg.y, qg.z, qg.w };
            const uint qua[4] = { qu.x, qu.y, qu.z, qu.w };
            #pragma unroll
            for (uint q = 0u; q < 4u; ++q) {
                const float4 wg = float4(as_type<uchar4>(qga[q])) * scg + big;
                const float4 wu = float4(as_type<uchar4>(qua[q])) * scu + biu;
                const float4 v0 = float4(x0[k4 + q]); ag0 += dot(wg, v0); au0 += dot(wu, v0);
                const float4 v1 = float4(x1[k4 + q]); ag1 += dot(wg, v1); au1 += dot(wu, v1);
                const float4 v2 = float4(x2[k4 + q]); ag2 += dot(wg, v2); au2 += dot(wu, v2);
                if (cnt > 3u) { const float4 v3 = float4(x3[k4 + q]); ag3 += dot(wg, v3); au3 += dot(wu, v3); }
            }
        } else {
            float4 wg[4], wu[4];
            if (FMT == 3) {
                msv_ink2_decode16_affine<S, BITS, GS>(arow, sa, ba, c, wg);
                msv_ink2_decode16_affine<S, BITS, GS>(arow + WBR, sa + SPR, ba + SPR, c, wu);
            } else {
                msv_ink2_decode16_fp(FMT, wrow, srow, c, wg);
                msv_ink2_decode16_fp(FMT, wrow + WPR, srow + SPR, c, wu);
            }
            { const float4 v0 = float4(x0[k4 + 0u]), v1 = float4(x0[k4 + 1u]), v2 = float4(x0[k4 + 2u]), v3 = float4(x0[k4 + 3u]); ag0 += dot(wg[0], v0) + dot(wg[1], v1) + dot(wg[2], v2) + dot(wg[3], v3); au0 += dot(wu[0], v0) + dot(wu[1], v1) + dot(wu[2], v2) + dot(wu[3], v3); }
            { const float4 v0 = float4(x1[k4 + 0u]), v1 = float4(x1[k4 + 1u]), v2 = float4(x1[k4 + 2u]), v3 = float4(x1[k4 + 3u]); ag1 += dot(wg[0], v0) + dot(wg[1], v1) + dot(wg[2], v2) + dot(wg[3], v3); au1 += dot(wu[0], v0) + dot(wu[1], v1) + dot(wu[2], v2) + dot(wu[3], v3); }
            { const float4 v0 = float4(x2[k4 + 0u]), v1 = float4(x2[k4 + 1u]), v2 = float4(x2[k4 + 2u]), v3 = float4(x2[k4 + 3u]); ag2 += dot(wg[0], v0) + dot(wg[1], v1) + dot(wg[2], v2) + dot(wg[3], v3); au2 += dot(wu[0], v0) + dot(wu[1], v1) + dot(wu[2], v2) + dot(wu[3], v3); }
            if (cnt > 3u) { const float4 v0 = float4(x3[k4 + 0u]), v1 = float4(x3[k4 + 1u]), v2 = float4(x3[k4 + 2u]), v3 = float4(x3[k4 + 3u]); ag3 += dot(wg[0], v0) + dot(wg[1], v1) + dot(wg[2], v2) + dot(wg[3], v3); au3 += dot(wu[0], v0) + dot(wu[1], v1) + dot(wu[2], v2) + dot(wu[3], v3); }
        }
    }
    const float gs[4] = { float(T(simd_sum(ag0))) * sc, float(T(simd_sum(ag1))) * sc, float(T(simd_sum(ag2))) * sc, float(T(simd_sum(ag3))) * sc };
    const float us[4] = { float(T(simd_sum(au0))) * sc, float(T(simd_sum(au1))) * sc, float(T(simd_sum(au2))) * sc, float(T(simd_sum(au3))) * sc };
    const uint ps[4] = { p0, p1, p2, p3 };
    if (lane == 0u) {
        for (uint m = 0u; m < cnt; ++m) {
            const float capped = (precise::tanh(gs[m] * PR[0]) * PR[1]) * (precise::tanh(us[m] * PR[2]) * PR[3]);
            ACT[size_t(ps[m]) * I + row0] = T(capped / (1.0f + precise::exp(-gs[m])));
        }
    }
    return;
}

// Tiled 8-scalar-register passes: handles cnt = 5..16 with zero dynamic array indexing (no thread-local stack spill).
for (uint base_m = 0u; base_m < cnt; base_m += 8u) {
    const uint rem = min(cnt - base_m, 8u);
    const uint p0 = sh_pairs[base_m + 0u];
    const uint p1 = rem > 1u ? sh_pairs[base_m + 1u] : p0;
    const uint p2 = rem > 2u ? sh_pairs[base_m + 2u] : p0;
    const uint p3 = rem > 3u ? sh_pairs[base_m + 3u] : p0;
    const uint p4 = rem > 4u ? sh_pairs[base_m + 4u] : p0;
    const uint p5 = rem > 5u ? sh_pairs[base_m + 5u] : p0;
    const uint p6 = rem > 6u ? sh_pairs[base_m + 6u] : p0;
    const uint p7 = rem > 7u ? sh_pairs[base_m + 7u] : p0;
    const device vec<T, 4>* x0 = (const device vec<T, 4>*)(X + size_t(sh_rows[base_m + 0u]) * K);
    const device vec<T, 4>* x1 = (const device vec<T, 4>*)(X + size_t(rem > 1u ? sh_rows[base_m + 1u] : sh_rows[base_m + 0u]) * K);
    const device vec<T, 4>* x2 = (const device vec<T, 4>*)(X + size_t(rem > 2u ? sh_rows[base_m + 2u] : sh_rows[base_m + 0u]) * K);
    const device vec<T, 4>* x3 = (const device vec<T, 4>*)(X + size_t(rem > 3u ? sh_rows[base_m + 3u] : sh_rows[base_m + 0u]) * K);
    const device vec<T, 4>* x4 = (const device vec<T, 4>*)(X + size_t(rem > 4u ? sh_rows[base_m + 4u] : sh_rows[base_m + 0u]) * K);
    const device vec<T, 4>* x5 = (const device vec<T, 4>*)(X + size_t(rem > 5u ? sh_rows[base_m + 5u] : sh_rows[base_m + 0u]) * K);
    const device vec<T, 4>* x6 = (const device vec<T, 4>*)(X + size_t(rem > 6u ? sh_rows[base_m + 6u] : sh_rows[base_m + 0u]) * K);
    const device vec<T, 4>* x7 = (const device vec<T, 4>*)(X + size_t(rem > 7u ? sh_rows[base_m + 7u] : sh_rows[base_m + 0u]) * K);
    float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f, ag2 = 0.0f, au2 = 0.0f, ag3 = 0.0f, au3 = 0.0f;
    float ag4 = 0.0f, au4 = 0.0f, ag5 = 0.0f, au5 = 0.0f, ag6 = 0.0f, au6 = 0.0f, ag7 = 0.0f, au7 = 0.0f;
    for (uint c = lane; c < K / 16u; c += 32u) {
        const uint k4 = c * 4u;
        if (FMT == 3 && BITS == 8) {
            const uint g = (c * 16u) / GS;
            const float scg = float(sa[g]), big = float(ba[g]);
            const float scu = float(sa[SPR + g]), biu = float(ba[SPR + g]);
            const uint4 qg = ((const device uint4*)arow)[c];
            const uint4 qu = ((const device uint4*)(arow + WBR))[c];
            const uint qga[4] = { qg.x, qg.y, qg.z, qg.w };
            const uint qua[4] = { qu.x, qu.y, qu.z, qu.w };
            #pragma unroll
            for (uint q = 0u; q < 4u; ++q) {
                const float4 wg = float4(as_type<uchar4>(qga[q])) * scg + big;
                const float4 wu = float4(as_type<uchar4>(qua[q])) * scu + biu;
                { const float4 v = float4(x0[k4 + q]); ag0 += dot(wg, v); au0 += dot(wu, v); }
                if (rem > 1u) { const float4 v = float4(x1[k4 + q]); ag1 += dot(wg, v); au1 += dot(wu, v); }
                if (rem > 2u) { const float4 v = float4(x2[k4 + q]); ag2 += dot(wg, v); au2 += dot(wu, v); }
                if (rem > 3u) { const float4 v = float4(x3[k4 + q]); ag3 += dot(wg, v); au3 += dot(wu, v); }
                if (rem > 4u) { const float4 v = float4(x4[k4 + q]); ag4 += dot(wg, v); au4 += dot(wu, v); }
                if (rem > 5u) { const float4 v = float4(x5[k4 + q]); ag5 += dot(wg, v); au5 += dot(wu, v); }
                if (rem > 6u) { const float4 v = float4(x6[k4 + q]); ag6 += dot(wg, v); au6 += dot(wu, v); }
                if (rem > 7u) { const float4 v = float4(x7[k4 + q]); ag7 += dot(wg, v); au7 += dot(wu, v); }
            }
        } else {
            float4 wg[4], wu[4];
            if (FMT == 3) {
                msv_ink2_decode16_affine<S, BITS, GS>(arow, sa, ba, c, wg);
                msv_ink2_decode16_affine<S, BITS, GS>(arow + WBR, sa + SPR, ba + SPR, c, wu);
            } else {
                msv_ink2_decode16_fp(FMT, wrow, srow, c, wg);
                msv_ink2_decode16_fp(FMT, wrow + WPR, srow + SPR, c, wu);
            }
            { const float4 v0 = float4(x0[k4 + 0u]), v1 = float4(x0[k4 + 1u]), v2 = float4(x0[k4 + 2u]), v3 = float4(x0[k4 + 3u]); ag0 += dot(wg[0], v0) + dot(wg[1], v1) + dot(wg[2], v2) + dot(wg[3], v3); au0 += dot(wu[0], v0) + dot(wu[1], v1) + dot(wu[2], v2) + dot(wu[3], v3); }
            if (rem > 1u) { const float4 v0 = float4(x1[k4 + 0u]), v1 = float4(x1[k4 + 1u]), v2 = float4(x1[k4 + 2u]), v3 = float4(x1[k4 + 3u]); ag1 += dot(wg[0], v0) + dot(wg[1], v1) + dot(wg[2], v2) + dot(wg[3], v3); au1 += dot(wu[0], v0) + dot(wu[1], v1) + dot(wu[2], v2) + dot(wu[3], v3); }
            if (rem > 2u) { const float4 v0 = float4(x2[k4 + 0u]), v1 = float4(x2[k4 + 1u]), v2 = float4(x2[k4 + 2u]), v3 = float4(x2[k4 + 3u]); ag2 += dot(wg[0], v0) + dot(wg[1], v1) + dot(wg[2], v2) + dot(wg[3], v3); au2 += dot(wu[0], v0) + dot(wu[1], v1) + dot(wu[2], v2) + dot(wu[3], v3); }
            if (rem > 3u) { const float4 v0 = float4(x3[k4 + 0u]), v1 = float4(x3[k4 + 1u]), v2 = float4(x3[k4 + 2u]), v3 = float4(x3[k4 + 3u]); ag3 += dot(wg[0], v0) + dot(wg[1], v1) + dot(wg[2], v2) + dot(wg[3], v3); au3 += dot(wu[0], v0) + dot(wu[1], v1) + dot(wu[2], v2) + dot(wu[3], v3); }
            if (rem > 4u) { const float4 v0 = float4(x4[k4 + 0u]), v1 = float4(x4[k4 + 1u]), v2 = float4(x4[k4 + 2u]), v3 = float4(x4[k4 + 3u]); ag4 += dot(wg[0], v0) + dot(wg[1], v1) + dot(wg[2], v2) + dot(wg[3], v3); au4 += dot(wu[0], v0) + dot(wu[1], v1) + dot(wu[2], v2) + dot(wu[3], v3); }
            if (rem > 5u) { const float4 v0 = float4(x5[k4 + 0u]), v1 = float4(x5[k4 + 1u]), v2 = float4(x5[k4 + 2u]), v3 = float4(x5[k4 + 3u]); ag5 += dot(wg[0], v0) + dot(wg[1], v1) + dot(wg[2], v2) + dot(wg[3], v3); au5 += dot(wu[0], v0) + dot(wu[1], v1) + dot(wu[2], v2) + dot(wu[3], v3); }
            if (rem > 6u) { const float4 v0 = float4(x6[k4 + 0u]), v1 = float4(x6[k4 + 1u]), v2 = float4(x6[k4 + 2u]), v3 = float4(x6[k4 + 3u]); ag6 += dot(wg[0], v0) + dot(wg[1], v1) + dot(wg[2], v2) + dot(wg[3], v3); au6 += dot(wu[0], v0) + dot(wu[1], v1) + dot(wu[2], v2) + dot(wu[3], v3); }
            if (rem > 7u) { const float4 v0 = float4(x7[k4 + 0u]), v1 = float4(x7[k4 + 1u]), v2 = float4(x7[k4 + 2u]), v3 = float4(x7[k4 + 3u]); ag7 += dot(wg[0], v0) + dot(wg[1], v1) + dot(wg[2], v2) + dot(wg[3], v3); au7 += dot(wu[0], v0) + dot(wu[1], v1) + dot(wu[2], v2) + dot(wu[3], v3); }
        }
    }
    const float gs[8] = {
        float(T(simd_sum(ag0))) * sc, float(T(simd_sum(ag1))) * sc, float(T(simd_sum(ag2))) * sc, float(T(simd_sum(ag3))) * sc,
        float(T(simd_sum(ag4))) * sc, float(T(simd_sum(ag5))) * sc, float(T(simd_sum(ag6))) * sc, float(T(simd_sum(ag7))) * sc,
    };
    const float us[8] = {
        float(T(simd_sum(au0))) * sc, float(T(simd_sum(au1))) * sc, float(T(simd_sum(au2))) * sc, float(T(simd_sum(au3))) * sc,
        float(T(simd_sum(au4))) * sc, float(T(simd_sum(au5))) * sc, float(T(simd_sum(au6))) * sc, float(T(simd_sum(au7))) * sc,
    };
    const uint ps[8] = { p0, p1, p2, p3, p4, p5, p6, p7 };
    if (lane == 0u) {
        for (uint m = 0u; m < rem; ++m) {
            const float capped = (precise::tanh(gs[m] * PR[0]) * PR[1]) * (precise::tanh(us[m] * PR[2]) * PR[3]);
            ACT[size_t(ps[m]) * I + row0] = T(capped / (1.0f + precise::exp(-gs[m])));
        }
    }
}
