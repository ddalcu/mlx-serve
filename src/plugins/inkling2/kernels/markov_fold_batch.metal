// Inkling 2's drafter, batched Markov step for B lanes: reads FOLD [R, V] ONCE per NB lanes.
// P [B, 3 R] f32, FOLD [R, V] (T), FB [V] (T), BASE [B, steps, V] f32, AT [1] int32, CAP [1] f32.
// grid (groups * TG, ceil(B / NB), 1), threadgroup (TG, 1, 1), VPT columns per thread.
// SG_OUT == 0: PM, PI, PS [B, groups] (threadgroup top-1 + exp sum for greedy markov_pick_batch).
// SG_OUT != 0: PM, PI [B, groups * (TG / 32) * 4] (exact top-4 per simdgroup for sampled markov_pick_sample).
threadgroup float lat[NB][R];
threadgroup float rm[TG / 32];
threadgroup uint ri[TG / 32];
threadgroup float rs[TG / 32];
constexpr uint NONE = 0xffffffffu;
const uint t = thread_position_in_threadgroup.x;
const uint tg = threadgroup_position_in_grid.x;
const uint bg = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
const uint sg = t / 32u;
const uint B = P_shape[0];
const uint b0 = bg * uint(NB);
const uint nb = min(uint(NB), B - b0);
const uint V = FOLD_shape[1];
const uint STEPS = BASE_shape[1];
const size_t at = size_t(AT[0]);
const uint groups = (VOCAB + TG * VPT - 1u) / (TG * VPT);

for (uint lb = 0u; lb < nb; ++lb) {
    const size_t pbase = size_t(b0 + lb) * (3u * R) + 2u * R;
    for (uint r = t; r < R; r += TG) lat[lb][r] = float(T(metal::precise::tanh(P[pbase + r])));
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const uint v0 = (tg * TG + t) * VPT;
float acc[NB][VPT];
#pragma unroll
for (uint lb = 0u; lb < uint(NB); ++lb) {
    #pragma unroll
    for (uint j = 0u; j < uint(VPT); ++j) acc[lb][j] = 0.0f;
}
if (v0 < VOCAB) {
    for (uint r = 0u; r < R; ++r) {
        const vec<T, VPT> f = *(const device vec<T, VPT>*)(FOLD + size_t(r) * V + v0);
        #pragma unroll
        for (uint lb = 0u; lb < uint(NB); ++lb) {
            if (lb < nb) {
                const float lr = lat[lb][r];
                #pragma unroll
                for (uint j = 0u; j < uint(VPT); ++j) acc[lb][j] += lr * float(f[j]);
            }
        }
    }
}

float fbv[VPT];
#pragma unroll
for (uint j = 0u; j < uint(VPT); ++j) {
    const uint v = v0 + j;
    fbv[j] = v < VOCAB ? float(FB[v]) : 0.0f;
}
const float cap = CAP[0];

for (uint lb = 0u; lb < nb; ++lb) {
    const uint b = b0 + lb;
    const size_t base_off = (size_t(b) * STEPS + at) * V;
    float lane_acc[VPT];
    #pragma unroll
    for (uint j = 0u; j < uint(VPT); ++j) lane_acc[j] = 0.0f;
    #pragma unroll
    for (uint u = 0u; u < uint(NB); ++u) {
        if (u == lb) {
            #pragma unroll
            for (uint j = 0u; j < uint(VPT); ++j) lane_acc[j] = acc[u][j];
        }
    }
    float lg[VPT];
    #pragma unroll
    for (uint j = 0u; j < uint(VPT); ++j) {
        const uint v = v0 + j;
        lg[j] = v < VOCAB ? BASE[base_off + v] + float(T(T(lane_acc[j]) + T(fbv[j]))) : -INFINITY;
    }

    if constexpr (SG_OUT != 0) {
        const uint sgs = TG / 32u;
        float prev_m = INFINITY;
        uint prev_i = NONE;
        for (uint k = 0u; k < 4u; ++k) {
            float km = -INFINITY;
            uint ki = NONE;
            #pragma unroll
            for (uint j = 0u; j < uint(VPT); ++j) {
                const uint v = v0 + j;
                if (v >= VOCAB) continue;
                if (k > 0u && !msv_ink2_better(prev_m, prev_i, lg[j], v)) continue;
                if (ki == NONE || msv_ink2_better(lg[j], v, km, ki)) {
                    km = lg[j];
                    ki = v;
                }
            }
            for (ushort off = 16; off > 0; off >>= 1) {
                const float om = simd_shuffle_xor(km, off);
                const uint oi = simd_shuffle_xor(ki, off);
                if (oi != NONE && (ki == NONE || msv_ink2_better(om, oi, km, ki))) {
                    km = om;
                    ki = oi;
                }
            }
            if (lane == 0u) {
                const size_t out_idx = size_t(b) * (size_t(groups) * sgs * 4u) + (size_t(tg) * sgs + sg) * 4u + k;
                PM[out_idx] = km;
                PI[out_idx] = ki;
            }
            prev_m = km;
            prev_i = ki;
        }
        continue;
    }

    float m = -INFINITY;
    uint mi = NONE;
    #pragma unroll
    for (uint j = 0u; j < uint(VPT); ++j) {
        const uint v = v0 + j;
        if (v < VOCAB && (mi == NONE || msv_ink2_better(lg[j], v, m, mi))) {
            m = lg[j];
            mi = v;
        }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        const float om = simd_shuffle_xor(m, off);
        const uint oi = simd_shuffle_xor(mi, off);
        if (oi != NONE && (mi == NONE || msv_ink2_better(om, oi, m, mi))) {
            m = om;
            mi = oi;
        }
    }
    if (lane == 0u) {
        rm[sg] = m;
        ri[sg] = mi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float top = rm[0];
    uint at_top = ri[0];
    for (uint g = 1u; g < TG / 32u; ++g) {
        if (ri[g] != NONE && (at_top == NONE || msv_ink2_better(rm[g], ri[g], top, at_top))) {
            top = rm[g];
            at_top = ri[g];
        }
    }
    const float ctop = cap > 0 ? cap * metal::precise::tanh(top / cap) : top;
    float s = 0.0f;
    #pragma unroll
    for (uint j = 0u; j < uint(VPT); ++j) {
        if (v0 + j < VOCAB) s += metal::precise::exp((cap > 0 ? cap * metal::precise::tanh(lg[j] / cap) : lg[j]) - ctop);
    }
    s = simd_sum(s);
    if (lane == 0u) rs[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0u) {
        float tot = 0.0f;
        for (uint g = 0u; g < TG / 32u; ++g) tot += rs[g];
        const size_t out_idx = size_t(b) * groups + tg;
        PM[out_idx] = top;
        PI[out_idx] = at_top;
        PS[out_idx] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
