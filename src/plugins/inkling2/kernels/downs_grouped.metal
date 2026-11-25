// Inkling 2, grouped multi-row down projection (2 output rows per SIMD group): each unique active expert reads its
// packed down weights ONCE via 128-bit uint4 loads and reuses every ACT vector across two adjacent output rows, using
// explicit scalar register accumulators (zero dynamic-index stack spills) for every cnt = 1..S_CAP.
// grid (32, D / 2, NMAX), threadgroup (32, SGS, 1).
const uint u = thread_position_in_grid.z;
const uint ex = ACT_E[u];
if (ex == 0xffffffffu) return;

const uint tid = thread_position_in_threadgroup.y * 32u + thread_index_in_simdgroup;
const uint lane = thread_index_in_simdgroup;
const uint row0 = thread_position_in_grid.y * 2u;
const uint cnt = ACT_CNT[u];

threadgroup uint sh_pairs[S_CAP];
if (tid < cnt) sh_pairs[tid] = ACT_PAIR[u * S_CAP + tid];
threadgroup_barrier(mem_flags::mem_threadgroup);

constexpr uint WPR = FMT == 2 ? I / 4 : I / 8;
constexpr uint WBR = I * BITS / 8;
constexpr uint SPR = I / GS;
const size_t base = size_t(ex) * D + row0;
const device uint* wrow0 = WQ + base * WPR;
const device uchar* srow0 = (const device uchar*)WS + base * SPR;
const device uchar* arow0 = (const device uchar*)WQ + base * WBR;
const device S* sa0 = (const device S*)WS + base * SPR;
const device S* ba0 = (const device S*)WB + base * SPR;

if (cnt == 1u) {
    const uint p0 = sh_pairs[0];
    const device vec<T, 4>* a0 = (const device vec<T, 4>*)(ACT + size_t(p0) * I);
    float acc0 = 0.0f, bcc0 = 0.0f;
    for (uint c = lane; c < I / 16u; c += 32u) {
        const uint k4 = c * 4u;
        if (FMT == 3 && BITS == 8) {
            const uint g = (c * 16u) / GS;
            const float sc0 = float(sa0[g]), bi0 = float(ba0[g]);
            const float sc1 = float(sa0[SPR + g]), bi1 = float(ba0[SPR + g]);
            const uint4 q0 = ((const device uint4*)arow0)[c];
            const uint4 q1 = ((const device uint4*)(arow0 + WBR))[c];
            const float4 v0 = float4(a0[k4 + 0u]);
            acc0 += dot(float4(as_type<uchar4>(q0.x)) * sc0 + bi0, v0);
            bcc0 += dot(float4(as_type<uchar4>(q1.x)) * sc1 + bi1, v0);
            const float4 v1 = float4(a0[k4 + 1u]);
            acc0 += dot(float4(as_type<uchar4>(q0.y)) * sc0 + bi0, v1);
            bcc0 += dot(float4(as_type<uchar4>(q1.y)) * sc1 + bi1, v1);
            const float4 v2 = float4(a0[k4 + 2u]);
            acc0 += dot(float4(as_type<uchar4>(q0.z)) * sc0 + bi0, v2);
            bcc0 += dot(float4(as_type<uchar4>(q1.z)) * sc1 + bi1, v2);
            const float4 v3 = float4(a0[k4 + 3u]);
            acc0 += dot(float4(as_type<uchar4>(q0.w)) * sc0 + bi0, v3);
            bcc0 += dot(float4(as_type<uchar4>(q1.w)) * sc1 + bi1, v3);
        } else {
            float4 w0[4], w1[4];
            if (FMT == 3) {
                msv_ink2_decode16_affine<S, BITS, GS>(arow0, sa0, ba0, c, w0);
                msv_ink2_decode16_affine<S, BITS, GS>(arow0 + WBR, sa0 + SPR, ba0 + SPR, c, w1);
            } else {
                msv_ink2_decode16_fp(FMT, wrow0, srow0, c, w0);
                msv_ink2_decode16_fp(FMT, wrow0 + WPR, srow0 + SPR, c, w1);
            }
            const float4 v0 = float4(a0[k4 + 0u]), v1 = float4(a0[k4 + 1u]), v2 = float4(a0[k4 + 2u]), v3 = float4(a0[k4 + 3u]);
            acc0 += dot(w0[0], v0) + dot(w0[1], v1) + dot(w0[2], v2) + dot(w0[3], v3);
            bcc0 += dot(w1[0], v0) + dot(w1[1], v1) + dot(w1[2], v2) + dot(w1[3], v3);
        }
    }
    const float v0 = simd_sum(acc0), v1 = simd_sum(bcc0);
    if (lane == 0u) {
        DN[size_t(p0) * D + row0 + 0u] = T(v0);
        DN[size_t(p0) * D + row0 + 1u] = T(v1);
    }
    return;
}

if (cnt == 2u) {
    const uint p0 = sh_pairs[0], p1 = sh_pairs[1];
    const device vec<T, 4>* a0 = (const device vec<T, 4>*)(ACT + size_t(p0) * I);
    const device vec<T, 4>* a1 = (const device vec<T, 4>*)(ACT + size_t(p1) * I);
    float acc0 = 0.0f, acc1 = 0.0f, bcc0 = 0.0f, bcc1 = 0.0f;
    for (uint c = lane; c < I / 16u; c += 32u) {
        const uint k4 = c * 4u;
        if (FMT == 3 && BITS == 8) {
            const uint g = (c * 16u) / GS;
            const float sc0 = float(sa0[g]), bi0 = float(ba0[g]);
            const float sc1 = float(sa0[SPR + g]), bi1 = float(ba0[SPR + g]);
            const uint4 q0 = ((const device uint4*)arow0)[c];
            const uint4 q1 = ((const device uint4*)(arow0 + WBR))[c];
            const uint q0a[4] = { q0.x, q0.y, q0.z, q0.w };
            const uint q1a[4] = { q1.x, q1.y, q1.z, q1.w };
            #pragma unroll
            for (uint q = 0u; q < 4u; ++q) {
                const float4 w0 = float4(as_type<uchar4>(q0a[q])) * sc0 + bi0;
                const float4 w1 = float4(as_type<uchar4>(q1a[q])) * sc1 + bi1;
                const float4 v0 = float4(a0[k4 + q]); acc0 += dot(w0, v0); bcc0 += dot(w1, v0);
                const float4 v1 = float4(a1[k4 + q]); acc1 += dot(w0, v1); bcc1 += dot(w1, v1);
            }
        } else {
            float4 w0[4], w1[4];
            if (FMT == 3) {
                msv_ink2_decode16_affine<S, BITS, GS>(arow0, sa0, ba0, c, w0);
                msv_ink2_decode16_affine<S, BITS, GS>(arow0 + WBR, sa0 + SPR, ba0 + SPR, c, w1);
            } else {
                msv_ink2_decode16_fp(FMT, wrow0, srow0, c, w0);
                msv_ink2_decode16_fp(FMT, wrow0 + WPR, srow0 + SPR, c, w1);
            }
            { const float4 v0 = float4(a0[k4 + 0u]), v1 = float4(a0[k4 + 1u]), v2 = float4(a0[k4 + 2u]), v3 = float4(a0[k4 + 3u]); acc0 += dot(w0[0], v0) + dot(w0[1], v1) + dot(w0[2], v2) + dot(w0[3], v3); bcc0 += dot(w1[0], v0) + dot(w1[1], v1) + dot(w1[2], v2) + dot(w1[3], v3); }
            { const float4 v0 = float4(a1[k4 + 0u]), v1 = float4(a1[k4 + 1u]), v2 = float4(a1[k4 + 2u]), v3 = float4(a1[k4 + 3u]); acc1 += dot(w0[0], v0) + dot(w0[1], v1) + dot(w0[2], v2) + dot(w0[3], v3); bcc1 += dot(w1[0], v0) + dot(w1[1], v1) + dot(w1[2], v2) + dot(w1[3], v3); }
        }
    }
    const float va0 = simd_sum(acc0), va1 = simd_sum(acc1);
    const float vb0 = simd_sum(bcc0), vb1 = simd_sum(bcc1);
    if (lane == 0u) {
        DN[size_t(p0) * D + row0 + 0u] = T(va0); DN[size_t(p0) * D + row0 + 1u] = T(vb0);
        DN[size_t(p1) * D + row0 + 0u] = T(va1); DN[size_t(p1) * D + row0 + 1u] = T(vb1);
    }
    return;
}

if (cnt <= 4u) {
    const uint p0 = sh_pairs[0], p1 = sh_pairs[1], p2 = sh_pairs[2], p3 = cnt > 3u ? sh_pairs[3] : p2;
    const device vec<T, 4>* a0 = (const device vec<T, 4>*)(ACT + size_t(p0) * I);
    const device vec<T, 4>* a1 = (const device vec<T, 4>*)(ACT + size_t(p1) * I);
    const device vec<T, 4>* a2 = (const device vec<T, 4>*)(ACT + size_t(p2) * I);
    const device vec<T, 4>* a3 = (const device vec<T, 4>*)(ACT + size_t(p3) * I);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f, bcc2 = 0.0f, bcc3 = 0.0f;
    for (uint c = lane; c < I / 16u; c += 32u) {
        const uint k4 = c * 4u;
        if (FMT == 3 && BITS == 8) {
            const uint g = (c * 16u) / GS;
            const float sc0 = float(sa0[g]), bi0 = float(ba0[g]);
            const float sc1 = float(sa0[SPR + g]), bi1 = float(ba0[SPR + g]);
            const uint4 q0 = ((const device uint4*)arow0)[c];
            const uint4 q1 = ((const device uint4*)(arow0 + WBR))[c];
            const uint q0a[4] = { q0.x, q0.y, q0.z, q0.w };
            const uint q1a[4] = { q1.x, q1.y, q1.z, q1.w };
            #pragma unroll
            for (uint q = 0u; q < 4u; ++q) {
                const float4 w0 = float4(as_type<uchar4>(q0a[q])) * sc0 + bi0;
                const float4 w1 = float4(as_type<uchar4>(q1a[q])) * sc1 + bi1;
                { const float4 v = float4(a0[k4 + q]); acc0 += dot(w0, v); bcc0 += dot(w1, v); }
                { const float4 v = float4(a1[k4 + q]); acc1 += dot(w0, v); bcc1 += dot(w1, v); }
                { const float4 v = float4(a2[k4 + q]); acc2 += dot(w0, v); bcc2 += dot(w1, v); }
                if (cnt > 3u) { const float4 v = float4(a3[k4 + q]); acc3 += dot(w0, v); bcc3 += dot(w1, v); }
            }
        } else {
            float4 w[4];
            if (FMT == 3) msv_ink2_decode16_affine<S, BITS, GS>(arow0, sa0, ba0, c, w);
            else msv_ink2_decode16_fp(FMT, wrow0, srow0, c, w);
            { acc0 += dot(w[0], float4(a0[k4 + 0u])) + dot(w[1], float4(a0[k4 + 1u])) + dot(w[2], float4(a0[k4 + 2u])) + dot(w[3], float4(a0[k4 + 3u])); }
            { acc1 += dot(w[0], float4(a1[k4 + 0u])) + dot(w[1], float4(a1[k4 + 1u])) + dot(w[2], float4(a1[k4 + 2u])) + dot(w[3], float4(a1[k4 + 3u])); }
            { acc2 += dot(w[0], float4(a2[k4 + 0u])) + dot(w[1], float4(a2[k4 + 1u])) + dot(w[2], float4(a2[k4 + 2u])) + dot(w[3], float4(a2[k4 + 3u])); }
            if (cnt > 3u) { acc3 += dot(w[0], float4(a3[k4 + 0u])) + dot(w[1], float4(a3[k4 + 1u])) + dot(w[2], float4(a3[k4 + 2u])) + dot(w[3], float4(a3[k4 + 3u])); }
            if (FMT == 3) msv_ink2_decode16_affine<S, BITS, GS>(arow0 + WBR, sa0 + SPR, ba0 + SPR, c, w);
            else msv_ink2_decode16_fp(FMT, wrow0 + WPR, srow0 + SPR, c, w);
            { bcc0 += dot(w[0], float4(a0[k4 + 0u])) + dot(w[1], float4(a0[k4 + 1u])) + dot(w[2], float4(a0[k4 + 2u])) + dot(w[3], float4(a0[k4 + 3u])); }
            { bcc1 += dot(w[0], float4(a1[k4 + 0u])) + dot(w[1], float4(a1[k4 + 1u])) + dot(w[2], float4(a1[k4 + 2u])) + dot(w[3], float4(a1[k4 + 3u])); }
            { bcc2 += dot(w[0], float4(a2[k4 + 0u])) + dot(w[1], float4(a2[k4 + 1u])) + dot(w[2], float4(a2[k4 + 2u])) + dot(w[3], float4(a2[k4 + 3u])); }
            if (cnt > 3u) { bcc3 += dot(w[0], float4(a3[k4 + 0u])) + dot(w[1], float4(a3[k4 + 1u])) + dot(w[2], float4(a3[k4 + 2u])) + dot(w[3], float4(a3[k4 + 3u])); }
        }
    }
    const float vas[4] = { simd_sum(acc0), simd_sum(acc1), simd_sum(acc2), simd_sum(acc3) };
    const float vbs[4] = { simd_sum(bcc0), simd_sum(bcc1), simd_sum(bcc2), simd_sum(bcc3) };
    const uint ps[4] = { p0, p1, p2, p3 };
    if (lane == 0u) {
        for (uint m = 0u; m < cnt; ++m) {
            DN[size_t(ps[m]) * D + row0 + 0u] = T(vas[m]);
            DN[size_t(ps[m]) * D + row0 + 1u] = T(vbs[m]);
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
    const device vec<T, 4>* a0 = (const device vec<T, 4>*)(ACT + size_t(p0) * I);
    const device vec<T, 4>* a1 = (const device vec<T, 4>*)(ACT + size_t(p1) * I);
    const device vec<T, 4>* a2 = (const device vec<T, 4>*)(ACT + size_t(p2) * I);
    const device vec<T, 4>* a3 = (const device vec<T, 4>*)(ACT + size_t(p3) * I);
    const device vec<T, 4>* a4 = (const device vec<T, 4>*)(ACT + size_t(p4) * I);
    const device vec<T, 4>* a5 = (const device vec<T, 4>*)(ACT + size_t(p5) * I);
    const device vec<T, 4>* a6 = (const device vec<T, 4>*)(ACT + size_t(p6) * I);
    const device vec<T, 4>* a7 = (const device vec<T, 4>*)(ACT + size_t(p7) * I);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f, acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f, bcc2 = 0.0f, bcc3 = 0.0f, bcc4 = 0.0f, bcc5 = 0.0f, bcc6 = 0.0f, bcc7 = 0.0f;
    for (uint c = lane; c < I / 16u; c += 32u) {
        const uint k4 = c * 4u;
        if (FMT == 3 && BITS == 8) {
            const uint g = (c * 16u) / GS;
            const float sc0 = float(sa0[g]), bi0 = float(ba0[g]);
            const float sc1 = float(sa0[SPR + g]), bi1 = float(ba0[SPR + g]);
            const uint4 q0 = ((const device uint4*)arow0)[c];
            const uint4 q1 = ((const device uint4*)(arow0 + WBR))[c];
            const uint q0a[4] = { q0.x, q0.y, q0.z, q0.w };
            const uint q1a[4] = { q1.x, q1.y, q1.z, q1.w };
            #pragma unroll
            for (uint q = 0u; q < 4u; ++q) {
                const float4 w0 = float4(as_type<uchar4>(q0a[q])) * sc0 + bi0;
                const float4 w1 = float4(as_type<uchar4>(q1a[q])) * sc1 + bi1;
                { const float4 v = float4(a0[k4 + q]); acc0 += dot(w0, v); bcc0 += dot(w1, v); }
                if (rem > 1u) { const float4 v = float4(a1[k4 + q]); acc1 += dot(w0, v); bcc1 += dot(w1, v); }
                if (rem > 2u) { const float4 v = float4(a2[k4 + q]); acc2 += dot(w0, v); bcc2 += dot(w1, v); }
                if (rem > 3u) { const float4 v = float4(a3[k4 + q]); acc3 += dot(w0, v); bcc3 += dot(w1, v); }
                if (rem > 4u) { const float4 v = float4(a4[k4 + q]); acc4 += dot(w0, v); bcc4 += dot(w1, v); }
                if (rem > 5u) { const float4 v = float4(a5[k4 + q]); acc5 += dot(w0, v); bcc5 += dot(w1, v); }
                if (rem > 6u) { const float4 v = float4(a6[k4 + q]); acc6 += dot(w0, v); bcc6 += dot(w1, v); }
                if (rem > 7u) { const float4 v = float4(a7[k4 + q]); acc7 += dot(w0, v); bcc7 += dot(w1, v); }
            }
        } else {
            float4 w[4];
            if (FMT == 3) msv_ink2_decode16_affine<S, BITS, GS>(arow0, sa0, ba0, c, w);
            else msv_ink2_decode16_fp(FMT, wrow0, srow0, c, w);
            { acc0 += dot(w[0], float4(a0[k4 + 0u])) + dot(w[1], float4(a0[k4 + 1u])) + dot(w[2], float4(a0[k4 + 2u])) + dot(w[3], float4(a0[k4 + 3u])); }
            if (rem > 1u) { acc1 += dot(w[0], float4(a1[k4 + 0u])) + dot(w[1], float4(a1[k4 + 1u])) + dot(w[2], float4(a1[k4 + 2u])) + dot(w[3], float4(a1[k4 + 3u])); }
            if (rem > 2u) { acc2 += dot(w[0], float4(a2[k4 + 0u])) + dot(w[1], float4(a2[k4 + 1u])) + dot(w[2], float4(a2[k4 + 2u])) + dot(w[3], float4(a2[k4 + 3u])); }
            if (rem > 3u) { acc3 += dot(w[0], float4(a3[k4 + 0u])) + dot(w[1], float4(a3[k4 + 1u])) + dot(w[2], float4(a3[k4 + 2u])) + dot(w[3], float4(a3[k4 + 3u])); }
            if (rem > 4u) { acc4 += dot(w[0], float4(a4[k4 + 0u])) + dot(w[1], float4(a4[k4 + 1u])) + dot(w[2], float4(a4[k4 + 2u])) + dot(w[3], float4(a4[k4 + 3u])); }
            if (rem > 5u) { acc5 += dot(w[0], float4(a5[k4 + 0u])) + dot(w[1], float4(a5[k4 + 1u])) + dot(w[2], float4(a5[k4 + 2u])) + dot(w[3], float4(a5[k4 + 3u])); }
            if (rem > 6u) { acc6 += dot(w[0], float4(a6[k4 + 0u])) + dot(w[1], float4(a6[k4 + 1u])) + dot(w[2], float4(a6[k4 + 2u])) + dot(w[3], float4(a6[k4 + 3u])); }
            if (rem > 7u) { acc7 += dot(w[0], float4(a7[k4 + 0u])) + dot(w[1], float4(a7[k4 + 1u])) + dot(w[2], float4(a7[k4 + 2u])) + dot(w[3], float4(a7[k4 + 3u])); }
            if (FMT == 3) msv_ink2_decode16_affine<S, BITS, GS>(arow0 + WBR, sa0 + SPR, ba0 + SPR, c, w);
            else msv_ink2_decode16_fp(FMT, wrow0 + WPR, srow0 + SPR, c, w);
            { bcc0 += dot(w[0], float4(a0[k4 + 0u])) + dot(w[1], float4(a0[k4 + 1u])) + dot(w[2], float4(a0[k4 + 2u])) + dot(w[3], float4(a0[k4 + 3u])); }
            if (rem > 1u) { bcc1 += dot(w[0], float4(a1[k4 + 0u])) + dot(w[1], float4(a1[k4 + 1u])) + dot(w[2], float4(a1[k4 + 2u])) + dot(w[3], float4(a1[k4 + 3u])); }
            if (rem > 2u) { bcc2 += dot(w[0], float4(a2[k4 + 0u])) + dot(w[1], float4(a2[k4 + 1u])) + dot(w[2], float4(a2[k4 + 2u])) + dot(w[3], float4(a2[k4 + 3u])); }
            if (rem > 3u) { bcc3 += dot(w[0], float4(a3[k4 + 0u])) + dot(w[1], float4(a3[k4 + 1u])) + dot(w[2], float4(a3[k4 + 2u])) + dot(w[3], float4(a3[k4 + 3u])); }
            if (rem > 4u) { bcc4 += dot(w[0], float4(a4[k4 + 0u])) + dot(w[1], float4(a4[k4 + 1u])) + dot(w[2], float4(a4[k4 + 2u])) + dot(w[3], float4(a4[k4 + 3u])); }
            if (rem > 5u) { bcc5 += dot(w[0], float4(a5[k4 + 0u])) + dot(w[1], float4(a5[k4 + 1u])) + dot(w[2], float4(a5[k4 + 2u])) + dot(w[3], float4(a5[k4 + 3u])); }
            if (rem > 6u) { bcc6 += dot(w[0], float4(a6[k4 + 0u])) + dot(w[1], float4(a6[k4 + 1u])) + dot(w[2], float4(a6[k4 + 2u])) + dot(w[3], float4(a6[k4 + 3u])); }
            if (rem > 7u) { bcc7 += dot(w[0], float4(a7[k4 + 0u])) + dot(w[1], float4(a7[k4 + 1u])) + dot(w[2], float4(a7[k4 + 2u])) + dot(w[3], float4(a7[k4 + 3u])); }
        }
    }
    const float vas[8] = {
        simd_sum(acc0), simd_sum(acc1), simd_sum(acc2), simd_sum(acc3),
        simd_sum(acc4), simd_sum(acc5), simd_sum(acc6), simd_sum(acc7),
    };
    const float vbs[8] = {
        simd_sum(bcc0), simd_sum(bcc1), simd_sum(bcc2), simd_sum(bcc3),
        simd_sum(bcc4), simd_sum(bcc5), simd_sum(bcc6), simd_sum(bcc7),
    };
    const uint ps[8] = { p0, p1, p2, p3, p4, p5, p6, p7 };
    if (lane == 0u) {
        for (uint m = 0u; m < rem; ++m) {
            DN[size_t(ps[m]) * D + row0 + 0u] = T(vas[m]);
            DN[size_t(ps[m]) * D + row0 + 1u] = T(vbs[m]);
        }
    }
}
