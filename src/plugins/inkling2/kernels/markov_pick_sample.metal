// Inkling 2's drafter, batched sampled Markov pick for B lanes: grid (256, B, 1), threadgroup (256, 1, 1).
// Finds top-K candidates from PM/PI [B, NT] (msv_ink2_better order), applies softcap + temperature softmax over top-K
// and matching top_k / min_p / top_p filtering from LPARAMS [B, 4], draws one token per lane (argmax when temp <= 0
// or top_k == 1), emits proposal (CANDS, PROBS, DRAW), and updates RNN state NST.
// Inputs: PM [B, NT] f32, PI [B, NT] u32, P [B, 3 R] f32, ST [B, 1, R] (T), CAP [1] f32, LPARAMS [B, 4] f32, SEEDS [B, 2] u32, AT [1] i32.
// Outputs: TOKN [B] i32, CANDS [B, K] i32, PROBS [B, K] f32, DRAW [B] u8, NST [B, 1, R] (T).
threadgroup float rm[8];
threadgroup uint ri[8];
threadgroup float top_v[K];
threadgroup uint top_i[K];
threadgroup float tg_m;
threadgroup uint tg_i;
constexpr uint NONE = 0xffffffffu;
const uint t = thread_position_in_threadgroup.x;
const uint b = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
const uint sg = t / 32u;
const uint NT = PM_shape[1];
const size_t base = size_t(b) * NT;

float prev_m = INFINITY;
uint prev_i = NONE;
for (uint k = 0u; k < uint(K); ++k) {
    float m = -INFINITY;
    uint mi = NONE;
    for (uint i = t; i < NT; i += 256u) {
        const uint id = PI[base + i];
        if (id == NONE) continue;
        const float val = PM[base + i];
        if (k > 0u && !msv_ink2_better(prev_m, prev_i, val, id)) continue;
        if (mi == NONE || msv_ink2_better(val, id, m, mi)) {
            m = val;
            mi = id;
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
    if (t == 0u) {
        float top = rm[0];
        uint at_top = ri[0];
        for (uint g = 1u; g < 8u; ++g) {
            if (ri[g] != NONE && (at_top == NONE || msv_ink2_better(rm[g], ri[g], top, at_top))) {
                top = rm[g];
                at_top = ri[g];
            }
        }
        top_v[k] = top;
        top_i[k] = at_top;
        tg_m = top;
        tg_i = at_top;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prev_m = tg_m;
    prev_i = tg_i;
    if (prev_i == NONE) {
        if (t == 0u) {
            for (uint j = k + 1u; j < uint(K); ++j) { top_v[j] = -INFINITY; top_i[j] = NONE; }
        }
        break;
    }
}

if (t == 0u) {
    const float cap = CAP[0];
    const float temp = LPARAMS[b * 4u + 0u];
    const uint top_k = uint(LPARAMS[b * 4u + 1u]);
    const float top_p = LPARAMS[b * 4u + 2u];
    const float min_p = LPARAMS[b * 4u + 3u];
    const size_t kbase = size_t(b) * uint(K);
    uint nv = 0u;
    float c[K];
    for (uint k = 0u; k < uint(K); ++k) {
        if (top_i[k] == NONE) {
            CANDS[kbase + k] = -1;
            PROBS[kbase + k] = 0.0f;
            c[k] = -INFINITY;
        } else {
            CANDS[kbase + k] = int(top_i[k]);
            c[k] = cap > 0.0f ? cap * metal::precise::tanh(top_v[k] / cap) : top_v[k];
            nv = k + 1u;
        }
    }
    if (nv == 0u) {
        TOKN[b] = 0;
        DRAW[b] = uchar(0);
    } else if (temp <= 0.0f || top_k == 1u) {
        for (uint k = 0u; k < nv; ++k) PROBS[kbase + k] = (k == 0u) ? 1.0f : 0.0f;
        TOKN[b] = int(top_i[0]);
        DRAW[b] = uchar(0);
    } else {
        const float inv_t = 1.0f / temp;
        float max_s = c[0] * inv_t;
        for (uint k = 1u; k < nv; ++k) max_s = max(max_s, c[k] * inv_t);
        float q[K];
        float sum_w = 0.0f;
        for (uint k = 0u; k < nv; ++k) {
            q[k] = metal::precise::exp(c[k] * inv_t - max_s);
            sum_w += q[k];
        }
        const float inv_sum = 1.0f / sum_w;
        for (uint k = 0u; k < nv; ++k) q[k] *= inv_sum;
        uint keep = nv;
        if (top_k > 0u && top_k < keep) keep = top_k;
        if (min_p > 0.0f) {
            const float cut = min_p * q[0];
            uint mkeep = 1u;
            while (mkeep < keep && q[mkeep] >= cut) ++mkeep;
            keep = mkeep;
        }
        if (top_p > 0.0f && top_p < 1.0f) {
            float cum = 0.0f;
            uint pkeep = 0u;
            while (pkeep < keep) {
                cum += q[pkeep];
                ++pkeep;
                if (cum >= top_p) break;
            }
            keep = max(1u, pkeep);
        }
        float kept_sum = 0.0f;
        for (uint k = 0u; k < keep; ++k) kept_sum += q[k];
        const float inv_kept = 1.0f / kept_sum;
        for (uint k = 0u; k < nv; ++k) {
            q[k] = (k < keep) ? (q[k] * inv_kept) : 0.0f;
            PROBS[kbase + k] = q[k];
        }
        const ulong seed = (ulong(SEEDS[b * 2u + 1u]) << 32) | ulong(SEEDS[b * 2u + 0u]);
        ulong h = seed ^ (0x9e3779b97f4a7c15ul * ulong(uint(AT[0]) + 1u));
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ul;
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebul;
        h = h ^ (h >> 31);
        const float u = (float(h >> 41) + 0.5f) * 0x1p-23f;
        float acc = 0.0f;
        uint di = keep - 1u;
        for (uint k = 0u; k < keep; ++k) {
            acc += q[k];
            if (acc >= u) { di = k; break; }
        }
        TOKN[b] = int(top_i[di]);
        DRAW[b] = uchar(di);
    }
}

const size_t pbase = size_t(b) * (3u * R);
const size_t sbase = size_t(b) * R;
for (uint r = t; r < R; r += 256u) {
    const float x = P[pbase + r];
    const float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
    const T keep = T(x < 0 ? y : 1.0f - y);
    const T cand = T(metal::precise::tanh(P[pbase + R + r]));
    NST[sbase + r] = T(T(keep * ST[sbase + r]) + T(T(T(1.0f) - keep) * cand));
}
