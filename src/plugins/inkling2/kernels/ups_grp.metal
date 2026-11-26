// Inkling 2, a few rows: the gate_up projection and soft-capped SwiGLU of the picks grouped by expert (group.metal): a tile of
// up to 4 picks of one expert reads that expert's rows once. grid (32 * N, I / 2, 1), threadgroup (32, SGS, 1): a tile is a
// threadgroup along x, and each half of a SIMD group (16 lanes, 16 values of x a sweep) owns one intermediate row.
const uint tile = threadgroup_position_in_grid.x;
const uint exp_id = TEXP[tile];
if (exp_id == 0xFFFFFFFFu) return;
const uint cnt = TCNT[tile];
const uint lane = thread_index_in_simdgroup;
const uint sub = lane & 15u;
const uint half_sg = lane >> 4;
const uint row0 = thread_position_in_grid.y * 2u + half_sg;
const uint p0 = TPICK[tile * 4u + 0u];
const uint p1 = (cnt > 1u) ? TPICK[tile * 4u + 1u] : p0;
const uint p2 = (cnt > 2u) ? TPICK[tile * 4u + 2u] : p0;
const uint p3 = (cnt > 3u) ? TPICK[tile * 4u + 3u] : p0;
const uint r0 = p0 / KSEL;
const uint r1 = p1 / KSEL;
const uint r2 = p2 / KSEL;
const uint r3 = p3 / KSEL;

constexpr uint WPR = FMT == 2 ? K / 4 : K / 8;
constexpr uint WBR = K * BITS / 8;
constexpr uint SPR = K / GS;
const size_t base = size_t(exp_id) * (2 * I) + 2 * row0;
const device uint* wrow_g = WQ + base * WPR;
const device uint* wrow_u = wrow_g + WPR;
const device uchar* srow_g = (const device uchar*)WS + base * SPR;
const device uchar* srow_u = srow_g + SPR;
const device uchar* arow_g = (const device uchar*)WQ + base * WBR;
const device uchar* arow_u = arow_g + WBR;
const device S* sa_g = (const device S*)WS + base * SPR;
const device S* sa_u = sa_g + SPR;
const device S* ba_g = (const device S*)WB + base * SPR;
const device S* ba_u = ba_g + SPR;

#define SUM16(v) do { \
    v += simd_shuffle_xor(v, 8); \
    v += simd_shuffle_xor(v, 4); \
    v += simd_shuffle_xor(v, 2); \
    v += simd_shuffle_xor(v, 1); \
} while (0)

if (cnt == 1u) {
    float ag0 = 0.0f, au0 = 0.0f;
    const device T* x0p = X + size_t(r0) * K;
    for (uint k = sub * 16u; k < K; k += 256u) {
        const device vec<T, 4>* xp = (const device vec<T, 4>*)(x0p + k);
        const float4 v0 = float4(xp[0]), v1 = float4(xp[1]), v2 = float4(xp[2]), v3 = float4(xp[3]);
        float xr[16] = {
            v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w,
            v2.x, v2.y, v2.z, v2.w, v3.x, v3.y, v3.z, v3.w,
        };
        const uint c = k >> 4;
        if constexpr (FMT == 3) {
            ag0 += msv_ink2_affine_chunk<S, BITS, GS>(arow_g, sa_g, ba_g, c, xr);
            au0 += msv_ink2_affine_chunk<S, BITS, GS>(arow_u, sa_u, ba_u, c, xr);
        } else {
            ag0 += msv_ink2_chunk(FMT, wrow_g, srow_g, c, xr);
            au0 += msv_ink2_chunk(FMT, wrow_u, srow_u, c, xr);
        }
    }
    SUM16(ag0); SUM16(au0);
    if (sub == 0u) {
        const float sc = SCALED ? SC[exp_id] : 1.0f;
        const float g = float(T(ag0)) * sc;
        const float u = float(T(au0)) * sc;
        const float capped = (precise::tanh(g * PR[0]) * PR[1]) * (precise::tanh(u * PR[2]) * PR[3]);
        ACT[size_t(p0) * I + row0] = T(capped / (1.0f + precise::exp(-g)));
    }
    return;
}

float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;

for (uint k = sub * 16u; k < K; k += 256u) {
    const uint c = k >> 4;
    half4 wg0, wg1, wg2, wg3, wu0, wu1, wu2, wu3;
    float sg = 0.0f, su = 0.0f, bg = 0.0f, bu = 0.0f;
    if (FMT == 3) {
        msv_ink2_decode16_affine_v4<S, BITS, GS>(arow_g, sa_g, ba_g, c, wg0, wg1, wg2, wg3, sg, bg);
        msv_ink2_decode16_affine_v4<S, BITS, GS>(arow_u, sa_u, ba_u, c, wu0, wu1, wu2, wu3, su, bu);
    } else {
        msv_ink2_decode16_v4(FMT, wrow_g, srow_g, c, wg0, wg1, wg2, wg3, sg);
        msv_ink2_decode16_v4(FMT, wrow_u, srow_u, c, wu0, wu1, wu2, wu3, su);
    }
    const float4 fg0 = float4(wg0), fg1 = float4(wg1), fg2 = float4(wg2), fg3 = float4(wg3);
    const float4 fu0 = float4(wu0), fu1 = float4(wu1), fu2 = float4(wu2), fu3 = float4(wu3);

    #define ACC_UP(ROW_IDX, AG, AU) do { \
        const device vec<T, 4>* xp = (const device vec<T, 4>*)(X + size_t(ROW_IDX) * K + k); \
        const float4 x0 = float4(xp[0]), x1 = float4(xp[1]), x2 = float4(xp[2]), x3 = float4(xp[3]); \
        const float dg = dot(x0, fg0) + dot(x1, fg1) + dot(x2, fg2) + dot(x3, fg3); \
        const float du = dot(x0, fu0) + dot(x1, fu1) + dot(x2, fu2) + dot(x3, fu3); \
        if (FMT == 3) { \
            const float sx = dot(x0 + x1 + x2 + x3, float4(1.0f)); \
            AG += dg * sg + bg * sx; \
            AU += du * su + bu * sx; \
        } else { \
            AG += dg * sg; \
            AU += du * su; \
        } \
    } while (0)

    ACC_UP(r0, ag0, au0);
    if (cnt > 1u) ACC_UP(r1, ag1, au1);
    if (cnt > 2u) ACC_UP(r2, ag2, au2);
    if (cnt > 3u) ACC_UP(r3, ag3, au3);
    #undef ACC_UP
}

SUM16(ag0); SUM16(au0);
if (cnt > 1u) { SUM16(ag1); SUM16(au1); }
if (cnt > 2u) { SUM16(ag2); SUM16(au2); }
if (cnt > 3u) { SUM16(ag3); SUM16(au3); }
#undef SUM16

const float sc = SCALED ? SC[exp_id] : 1.0f;
if (sub < cnt) {
    const float raw_g = (sub == 0u) ? ag0 : ((sub == 1u) ? ag1 : ((sub == 2u) ? ag2 : ag3));
    const float raw_u = (sub == 0u) ? au0 : ((sub == 1u) ? au1 : ((sub == 2u) ? au2 : au3));
    const uint pk = (sub == 0u) ? p0 : ((sub == 1u) ? p1 : ((sub == 2u) ? p2 : p3));
    const float g = float(T(raw_g)) * sc;
    const float u = float(T(raw_u)) * sc;
    const float capped = (precise::tanh(g * PR[0]) * PR[1]) * (precise::tanh(u * PR[2]) * PR[3]);
    ACT[size_t(pk) * I + row0] = T(capped / (1.0f + precise::exp(-g)));
}
