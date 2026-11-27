// Inkling 2's drafter, one Markov step: quantized 8-bit group-GS fold over the vocab and each threadgroup's share of
// the pick. FQ [R, V] uint8, FS [R/GS, V] half, FQB [R/GS, V] half (quant bias), FB [V] (T, fold bias),
// BASE [1, block - 1, V] f32, P [3 R] f32, AT [1] int32, CAP [1] f32, LPARAMS [4] u32 (when LAW=1).
// Quantizing FOLD along R halves its footprint (~54.7 MB) so it fits in SLC across all 7 Markov steps.
// When LAW=1, writes SCALED [1, VOCAB] f32 = capped / temp and draws via exact Gumbel-max into (GM, PI).
constexpr uint NG = R / GS;
threadgroup float lat[R];
threadgroup float lsum[NG];
threadgroup float rm[TG / 32];
threadgroup uint ri[TG / 32];
threadgroup float rs[TG / 32];
threadgroup float rgm[TG / 32];
constexpr uint NONE = 0xffffffffu;
const uint t = thread_position_in_threadgroup.x;
const uint tg = threadgroup_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = t / 32;
const uint V = FQ_shape[1];
const size_t at = size_t(AT[0]);
for (uint r = t; r < R; r += TG) lat[r] = float(T(metal::fast::tanh(P[2 * R + r])));
threadgroup_barrier(mem_flags::mem_threadgroup);
if (t < NG) {
    float s = 0.0f;
    const uint base = t * GS;
    for (uint k = 0; k < GS; ++k) s += lat[base + k];
    lsum[t] = s;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint v0 = (tg * TG + t) * VPT;
float4 acc0 = float4(0.0f);
float4 acc1 = float4(0.0f);
if (v0 < VOCAB) {
    for (uint g = 0; g < NG; ++g) {
        float4 g0 = float4(0.0f);
        float4 g1 = float4(0.0f);
        const uint rbase = g * GS;
        for (uint k = 0; k < GS; ++k) {
            const float lv = lat[rbase + k];
            const device uchar4* fp = (const device uchar4*)(FQ + size_t(rbase + k) * V + v0);
            g0 = fma(float4(fp[0]), lv, g0);
            g1 = fma(float4(fp[1]), lv, g1);
        }
        const device half4* sp = (const device half4*)(FS + size_t(g) * V + v0);
        const device half4* bp = (const device half4*)(FQB + size_t(g) * V + v0);
        const float ls = lsum[g];
        acc0 = fma(g0, float4(sp[0]), fma(float4(bp[0]), ls, acc0));
        acc1 = fma(g1, float4(sp[1]), fma(float4(bp[1]), ls, acc1));
    }
}
float acc[8] = { acc0.x, acc0.y, acc0.z, acc0.w, acc1.x, acc1.y, acc1.z, acc1.w };
float lg[8];
const float cap = CAP[0];
const float inv_cap = cap > 0 ? 1.0f / cap : 0.0f;
float m = -INFINITY;
float gm = -INFINITY;
uint mi = NONE;

if constexpr (LAW) {
    const float inv_temp = as_type<float>(LPARAMS[0]);
    const ulong seed = (ulong(LPARAMS[2]) << 32) | ulong(LPARAMS[1]);
    const ulong step_mix = seed ^ (ulong(at) * 0x9E3779B97F4A7C15ul);
    if (v0 < VOCAB) {
        const device float4* bp = (const device float4*)(BASE + at * V + v0);
        const device vec<T, 4>* fbp = (const device vec<T, 4>*)(FB + v0);
        const float4 b0 = bp[0], b1 = bp[1];
        const vec<T, 4> fb0 = fbp[0], fb1 = fbp[1];
        const float bv[8] = { b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w };
        const T fbv[8] = { fb0[0], fb0[1], fb0[2], fb0[3], fb1[0], fb1[1], fb1[2], fb1[3] };
        for (uint j = 0; j < 8; ++j) {
            const uint v = v0 + j;
            if (v < VOCAB) {
                const float raw = bv[j] + float(T(T(acc[j]) + fbv[j]));
                const float sc = (cap > 0 ? cap * metal::fast::tanh(raw * inv_cap) : raw) * inv_temp;
                lg[j] = sc;
                SCALED[v] = sc;
                m = max(m, sc);
                ulong k = step_mix ^ (ulong(v) * 0xBF58476D1CE4E5B9ul);
                k = (k ^ (k >> 30)) * 0xBF58476D1CE4E5B9ul;
                k = (k ^ (k >> 27)) * 0x94D049BB133111EBul;
                k ^= (k >> 31);
                const float u = (float(uint(k >> 41)) + 0.5f) * (1.0f / 8388608.0f);
                const float gval = sc - metal::log(-metal::log(u));
                if (mi == NONE || msv_ink2_better(gval, v, gm, mi)) {
                    gm = gval;
                    mi = v;
                }
            } else {
                lg[j] = -INFINITY;
            }
        }
    } else {
        for (uint j = 0; j < 8; ++j) lg[j] = -INFINITY;
    }
    m = simd_max(m);
    for (ushort off = 16; off > 0; off >>= 1) {
        const float ogm = simd_shuffle_xor(gm, off);
        const uint oi = simd_shuffle_xor(mi, off);
        if (oi != NONE && (mi == NONE || msv_ink2_better(ogm, oi, gm, mi))) {
            gm = ogm;
            mi = oi;
        }
    }
    if (lane == 0) {
        rm[sg] = m;
        rgm[sg] = gm;
        ri[sg] = mi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float top_m = rm[0];
    float top_gm = rgm[0];
    uint at_gm = ri[0];
    for (uint g = 1; g < TG / 32; ++g) {
        top_m = max(top_m, rm[g]);
        if (ri[g] != NONE && (at_gm == NONE || msv_ink2_better(rgm[g], ri[g], top_gm, at_gm))) {
            top_gm = rgm[g];
            at_gm = ri[g];
        }
    }
    float s = 0.0f;
    if (v0 < VOCAB) {
        for (uint j = 0; j < 8; ++j) {
            if (v0 + j < VOCAB) s += metal::exp(lg[j] - top_m);
        }
    }
    s = simd_sum(s);
    if (lane == 0) rs[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float tot = 0.0f;
        for (uint g = 0; g < TG / 32; ++g) tot += rs[g];
        PM[tg] = top_m;
        PI[tg] = at_gm;
        PS[tg] = tot;
        GM[tg] = top_gm;
    }
    return;
}

if (v0 < VOCAB) {
    const device float4* bp = (const device float4*)(BASE + at * V + v0);
    const device vec<T, 4>* fbp = (const device vec<T, 4>*)(FB + v0);
    const float4 b0 = bp[0], b1 = bp[1];
    const vec<T, 4> fb0 = fbp[0], fb1 = fbp[1];
    const float bv[8] = { b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w };
    const T fbv[8] = { fb0[0], fb0[1], fb0[2], fb0[3], fb1[0], fb1[1], fb1[2], fb1[3] };
    for (uint j = 0; j < 8; ++j) {
        const uint v = v0 + j;
        lg[j] = v < VOCAB ? bv[j] + float(T(T(acc[j]) + fbv[j])) : -INFINITY;
        if (v < VOCAB && (mi == NONE || msv_ink2_better(lg[j], v, m, mi))) {
            m = lg[j];
            mi = v;
        }
    }
} else {
    for (uint j = 0; j < 8; ++j) lg[j] = -INFINITY;
}
for (ushort off = 16; off > 0; off >>= 1) {
    const float om = simd_shuffle_xor(m, off);
    const uint oi = simd_shuffle_xor(mi, off);
    if (oi != NONE && (mi == NONE || msv_ink2_better(om, oi, m, mi))) {
        m = om;
        mi = oi;
    }
}
if (lane == 0) {
    rm[sg] = m;
    ri[sg] = mi;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float top = rm[0];
uint at_top = ri[0];
for (uint g = 1; g < TG / 32; ++g) {
    if (ri[g] != NONE && (at_top == NONE || msv_ink2_better(rm[g], ri[g], top, at_top))) {
        top = rm[g];
        at_top = ri[g];
    }
}
const float ctop = cap > 0 ? cap * metal::fast::tanh(top * inv_cap) : top;
float s = 0.0f;
if (v0 < VOCAB) {
    for (uint j = 0; j < 8; ++j) {
        if (v0 + j < VOCAB) s += metal::fast::exp((cap > 0 ? cap * metal::fast::tanh(lg[j] * inv_cap) : lg[j]) - ctop);
    }
}
s = simd_sum(s);
if (lane == 0) rs[sg] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (t == 0) {
    float tot = 0.0f;
    for (uint g = 0; g < TG / 32; ++g) tot += rs[g];
    PM[tg] = top;
    PI[tg] = at_top;
    PS[tg] = tot;
}
