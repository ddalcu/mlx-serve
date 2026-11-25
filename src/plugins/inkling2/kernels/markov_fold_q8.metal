// Inkling 2's drafter, one Markov step over coalesced per-vocab-row Q8 fold (64 vocab rows per threadgroup):
// FQ [V, R] int8, FSC [V] f32, BASE [1, steps, V] f32 (fold_bias pre-added), P [3 R] f32, AT [1] i32, CAP [1] f32.
// grid (ceil(VOCAB / 64) * 256, 1, 1), threadgroup (256, 1, 1): 8 SIMD groups x 8 contiguous vocab rows = 64 rows/TG.
threadgroup float sh_lat[R];
threadgroup float sh_lg[64];
threadgroup float rm[8];
threadgroup uint ri[8];
threadgroup float rs[2];
constexpr uint NONE = 0xffffffffu;
const uint t = thread_position_in_threadgroup.x;
const uint tg = threadgroup_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = t >> 5;
const uint V = FQ_shape[0];
const size_t at = size_t(AT[0]);

for (uint r = t; r < R; r += 256) sh_lat[r] = float(T(metal::precise::tanh(P[2 * R + r])));
threadgroup_barrier(mem_flags::mem_threadgroup);

const device float* brow = BASE + at * V;
float m = -INFINITY;
uint mi = NONE;

for (uint pass = 0u; pass < 2u; ++pass) {
    const uint v0 = tg * 64u + pass * 32u + sg * 4u;
    const bool full4 = (v0 + 3u) < VOCAB;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (full4) {
        const device char4* q0 = (const device char4*)(FQ + size_t(v0 + 0u) * R);
        const device char4* q1 = (const device char4*)(FQ + size_t(v0 + 1u) * R);
        const device char4* q2 = (const device char4*)(FQ + size_t(v0 + 2u) * R);
        const device char4* q3 = (const device char4*)(FQ + size_t(v0 + 3u) * R);
        for (uint k = lane; k < R / 4u; k += 32u) {
            const float4 xv = ((const threadgroup float4*)sh_lat)[k];
            a0 += dot(float4(q0[k]), xv);
            a1 += dot(float4(q1[k]), xv);
            a2 += dot(float4(q2[k]), xv);
            a3 += dot(float4(q3[k]), xv);
        }
    } else if (v0 < VOCAB) {
        for (uint k = lane; k < R / 4u; k += 32u) {
            const float4 xv = ((const threadgroup float4*)sh_lat)[k];
            a0 += dot(float4(((const device char4*)(FQ + size_t(v0) * R))[k]), xv);
            if (v0 + 1u < VOCAB) a1 += dot(float4(((const device char4*)(FQ + size_t(v0 + 1u) * R))[k]), xv);
            if (v0 + 2u < VOCAB) a2 += dot(float4(((const device char4*)(FQ + size_t(v0 + 2u) * R))[k]), xv);
        }
    }
    const float ds[4] = { simd_sum(a0), simd_sum(a1), simd_sum(a2), simd_sum(a3) };
    if (lane == 0u) {
        for (uint j = 0u; j < 4u; ++j) {
            const uint v = v0 + j;
            float lg = -INFINITY;
            if (v < VOCAB) {
                lg = brow[v] + ds[j] * FSC[v];
                if (mi == NONE || msv_ink2_better(lg, v, m, mi)) {
                    m = lg;
                    mi = v;
                }
            }
            sh_lg[pass * 32u + sg * 4u + j] = lg;
        }
    }
}

if (lane == 0u) {
    rm[sg] = m;
    ri[sg] = mi;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (t < 64u) {
    float top = rm[0];
    uint at_top = ri[0];
    for (uint g = 1u; g < 8u; ++g) {
        if (ri[g] != NONE && (at_top == NONE || msv_ink2_better(rm[g], ri[g], top, at_top))) {
            top = rm[g];
            at_top = ri[g];
        }
    }
    const float cap = CAP[0];
    const float ctop = cap > 0 ? cap * metal::precise::tanh(top / cap) : top;
    const uint v = tg * 64u + t;
    float ex = 0.0f;
    if (v < VOCAB) {
        const float lv = sh_lg[t];
        ex = metal::precise::exp((cap > 0 ? cap * metal::precise::tanh(lv / cap) : lv) - ctop);
    }
    const float s = simd_sum(ex);
    if (lane == 0u) rs[sg] = s;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (t == 0u) {
    float top = rm[0];
    uint at_top = ri[0];
    for (uint g = 1u; g < 8u; ++g) {
        if (ri[g] != NONE && (at_top == NONE || msv_ink2_better(rm[g], ri[g], top, at_top))) {
            top = rm[g];
            at_top = ri[g];
        }
    }
    PM[tg] = top;
    PI[tg] = at_top;
    PS[tg] = rs[0] + rs[1];
}
