// Inkling 2, a few rows: the down projection of the picks grouped by expert (group.metal), a tile of up to BM (4 or 8) picks reading
// the expert's rows once. grid (32 * MAX_TILES, D / 4, 1), threadgroup (32, SGS, 1): each half of a SIMD group (16 lanes) owns 2 output rows.
const uint tile = threadgroup_position_in_grid.x;
const uint exp_id = TEXP[tile];
if (exp_id == 0xFFFFFFFFu) return;
const uint cnt = TCNT[tile];
const uint lane = thread_index_in_simdgroup;
const uint sub = lane & 15u;
const uint half_sg = lane >> 4;
const uint row0 = (thread_position_in_grid.y * 2u + half_sg) * 2u;
const uint tbase = tile * uint(BM);
const uint p0 = TPICK[tbase + 0u];
const uint p1 = (cnt > 1u) ? TPICK[tbase + 1u] : p0;
const uint p2 = (cnt > 2u) ? TPICK[tbase + 2u] : p0;
const uint p3 = (cnt > 3u) ? TPICK[tbase + 3u] : p0;
uint p4 = p0, p5 = p0, p6 = p0, p7 = p0;
if constexpr (BM > 4) {
    p4 = (cnt > 4u) ? TPICK[tbase + 4u] : p0;
    p5 = (cnt > 5u) ? TPICK[tbase + 5u] : p0;
    p6 = (cnt > 6u) ? TPICK[tbase + 6u] : p0;
    p7 = (cnt > 7u) ? TPICK[tbase + 7u] : p0;
}

constexpr uint WPR = FMT == 2 ? I / 4 : I / 8;
constexpr uint WBR = I * BITS / 8;
constexpr uint SPR = I / GS;
const size_t base = size_t(exp_id) * D + row0;
const device uint* wrow0 = WQ + base * WPR;
const device uint* wrow1 = wrow0 + WPR;
const device uchar* srow0 = (const device uchar*)WS + base * SPR;
const device uchar* srow1 = srow0 + SPR;
const device uchar* arow0 = (const device uchar*)WQ + base * WBR;
const device uchar* arow1 = arow0 + WBR;
const device S* sa0 = (const device S*)WS + base * SPR;
const device S* sa1 = sa0 + SPR;
const device S* ba0 = (const device S*)WB + base * SPR;
const device S* ba1 = ba0 + SPR;

#define SUM16(v) do { \
    v += simd_shuffle_xor(v, 8); \
    v += simd_shuffle_xor(v, 4); \
    v += simd_shuffle_xor(v, 2); \
    v += simd_shuffle_xor(v, 1); \
} while (0)

if (cnt == 1u) {
    float a0 = 0.0f, a1 = 0.0f;
    const device T* act0 = ACT + size_t(p0) * I;
    for (uint k = sub * 16u; k < I; k += 256u) {
        const device vec<T, 4>* xp = (const device vec<T, 4>*)(act0 + k);
        const float4 v0 = float4(xp[0]), v1 = float4(xp[1]), v2 = float4(xp[2]), v3 = float4(xp[3]);
        float xr[16] = {
            v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w,
            v2.x, v2.y, v2.z, v2.w, v3.x, v3.y, v3.z, v3.w,
        };
        const uint c = k >> 4;
        if constexpr (FMT == 3) {
            a0 += msv_ink2_affine_chunk<S, BITS, GS>(arow0, sa0, ba0, c, xr);
            a1 += msv_ink2_affine_chunk<S, BITS, GS>(arow1, sa1, ba1, c, xr);
        } else {
            a0 += msv_ink2_chunk(FMT, wrow0, srow0, c, xr);
            a1 += msv_ink2_chunk(FMT, wrow1, srow1, c, xr);
        }
    }
    SUM16(a0); SUM16(a1);
    if (sub == 0u) {
        DN[size_t(p0) * D + row0] = T(a0);
        DN[size_t(p0) * D + row0 + 1u] = T(a1);
    }
    return;
}

float a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f, a04 = 0.0f, a05 = 0.0f, a06 = 0.0f, a07 = 0.0f;
float a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f, a14 = 0.0f, a15 = 0.0f, a16 = 0.0f, a17 = 0.0f;

for (uint k = sub * 16u; k < I; k += 256u) {
    const uint c = k >> 4;
    half4 w00, w01, w02, w03, w10, w11, w12, w13;
    float s0 = 0.0f, s1 = 0.0f, b0 = 0.0f, b1 = 0.0f;
    if (FMT == 3) {
        msv_ink2_decode16_affine_v4<S, BITS, GS>(arow0, sa0, ba0, c, w00, w01, w02, w03, s0, b0);
        msv_ink2_decode16_affine_v4<S, BITS, GS>(arow1, sa1, ba1, c, w10, w11, w12, w13, s1, b1);
    } else {
        msv_ink2_decode16_v4(FMT, wrow0, srow0, c, w00, w01, w02, w03, s0);
        msv_ink2_decode16_v4(FMT, wrow1, srow1, c, w10, w11, w12, w13, s1);
    }
    const float4 f00 = float4(w00), f01 = float4(w01), f02 = float4(w02), f03 = float4(w03);
    const float4 f10 = float4(w10), f11 = float4(w11), f12 = float4(w12), f13 = float4(w13);

    #define ACC_DN(PICK_IDX, A0, A1) do { \
        const device vec<T, 4>* xp = (const device vec<T, 4>*)(ACT + size_t(PICK_IDX) * I + k); \
        const float4 x0 = float4(xp[0]), x1 = float4(xp[1]), x2 = float4(xp[2]), x3 = float4(xp[3]); \
        const float d0 = dot(x0, f00) + dot(x1, f01) + dot(x2, f02) + dot(x3, f03); \
        const float d1 = dot(x0, f10) + dot(x1, f11) + dot(x2, f12) + dot(x3, f13); \
        if (FMT == 3) { \
            const float sx = dot(x0 + x1 + x2 + x3, float4(1.0f)); \
            A0 += d0 * s0 + b0 * sx; \
            A1 += d1 * s1 + b1 * sx; \
        } else { \
            A0 += d0 * s0; \
            A1 += d1 * s1; \
        } \
    } while (0)

    ACC_DN(p0, a00, a10);
    if (cnt > 1u) ACC_DN(p1, a01, a11);
    if (cnt > 2u) ACC_DN(p2, a02, a12);
    if (cnt > 3u) ACC_DN(p3, a03, a13);
    if constexpr (BM > 4) {
        if (cnt > 4u) ACC_DN(p4, a04, a14);
        if (cnt > 5u) ACC_DN(p5, a05, a15);
        if (cnt > 6u) ACC_DN(p6, a06, a16);
        if (cnt > 7u) ACC_DN(p7, a07, a17);
    }
    #undef ACC_DN
}

SUM16(a00); SUM16(a10);
if (cnt > 1u) { SUM16(a01); SUM16(a11); }
if (cnt > 2u) { SUM16(a02); SUM16(a12); }
if (cnt > 3u) { SUM16(a03); SUM16(a13); }
if constexpr (BM > 4) {
    if (cnt > 4u) { SUM16(a04); SUM16(a14); }
    if (cnt > 5u) { SUM16(a05); SUM16(a15); }
    if (cnt > 6u) { SUM16(a06); SUM16(a16); }
    if (cnt > 7u) { SUM16(a07); SUM16(a17); }
}
#undef SUM16

if (sub < cnt) {
    float r0v, r1v;
    uint pk;
    if constexpr (BM > 4) {
        switch (sub) {
            case 0u: r0v = a00; r1v = a10; pk = p0; break;
            case 1u: r0v = a01; r1v = a11; pk = p1; break;
            case 2u: r0v = a02; r1v = a12; pk = p2; break;
            case 3u: r0v = a03; r1v = a13; pk = p3; break;
            case 4u: r0v = a04; r1v = a14; pk = p4; break;
            case 5u: r0v = a05; r1v = a15; pk = p5; break;
            case 6u: r0v = a06; r1v = a16; pk = p6; break;
            default: r0v = a07; r1v = a17; pk = p7; break;
        }
    } else {
        r0v = (sub == 0u) ? a00 : ((sub == 1u) ? a01 : ((sub == 2u) ? a02 : a03));
        r1v = (sub == 0u) ? a10 : ((sub == 1u) ? a11 : ((sub == 2u) ? a12 : a13));
        pk = (sub == 0u) ? p0 : ((sub == 1u) ? p1 : ((sub == 2u) ? p2 : p3));
    }
    DN[size_t(pk) * D + row0] = T(r0v);
    DN[size_t(pk) * D + row0 + 1u] = T(r1v);
}
