// Inkling 2, a few rows: the down projection of the picks grouped by expert (group.metal), a tile of up to 4 picks reading
// the expert's rows once. grid (32 * N, D / 4, 1), threadgroup (32, SGS, 1): each half of a SIMD group (16 lanes) owns 2 output rows.
const uint tile = threadgroup_position_in_grid.x;
const uint exp_id = TEXP[tile];
if (exp_id == 0xFFFFFFFFu) return;
const uint cnt = TCNT[tile];
const uint lane = thread_index_in_simdgroup;
const uint sub = lane & 15u;
const uint half_sg = lane >> 4;
const uint row0 = (thread_position_in_grid.y * 2u + half_sg) * 2u;
const uint p0 = TPICK[tile * 4u + 0u];
const uint p1 = (cnt > 1u) ? TPICK[tile * 4u + 1u] : p0;
const uint p2 = (cnt > 2u) ? TPICK[tile * 4u + 2u] : p0;
const uint p3 = (cnt > 3u) ? TPICK[tile * 4u + 3u] : p0;

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

float a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
float a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;

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
    #undef ACC_DN
}

#define SUM16(v) do { \
    v += simd_shuffle_xor(v, 8); \
    v += simd_shuffle_xor(v, 4); \
    v += simd_shuffle_xor(v, 2); \
    v += simd_shuffle_xor(v, 1); \
} while (0)

SUM16(a00); SUM16(a10);
if (cnt > 1u) { SUM16(a01); SUM16(a11); }
if (cnt > 2u) { SUM16(a02); SUM16(a12); }
if (cnt > 3u) { SUM16(a03); SUM16(a13); }
#undef SUM16

if (sub < cnt) {
    const float r0v = (sub == 0u) ? a00 : ((sub == 1u) ? a01 : ((sub == 2u) ? a02 : a03));
    const float r1v = (sub == 0u) ? a10 : ((sub == 1u) ? a11 : ((sub == 2u) ? a12 : a13));
    const uint pk = (sub == 0u) ? p0 : ((sub == 1u) ? p1 : ((sub == 2u) ? p2 : p3));
    DN[size_t(pk) * D + row0] = T(r0v);
    DN[size_t(pk) * D + row0 + 1u] = T(r1v);
}
