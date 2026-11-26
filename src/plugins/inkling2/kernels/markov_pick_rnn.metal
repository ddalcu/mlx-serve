// Inkling 2's drafter: fused markov_pick (step i-1) + markov_rnn (step i) in one launch, eliminating the strictly
// serial GPU barrier between adjacent Markov steps. grid (DO_RNN ? 32 * 3 * R : 256), threadgroup (256).
// When LAW=1, reduces Gumbel-max (GM, PI) for TOKN and softmax (PM, PS) for CONF + LSE; when DO_RNN=0, exits after pick.
threadgroup float rm[8];
threadgroup uint ri[8];
threadgroup float rs[8];
threadgroup T st_s[R];
threadgroup T prev_s[R];
constexpr uint NONE = 0xffffffffu;
const uint t = thread_position_in_threadgroup.x;
const uint tg = threadgroup_position_in_grid.x;
const uint row = thread_position_in_grid.x / 32;
const uint lane = thread_index_in_simdgroup;
const uint sg = t / 32;
const uint NT = PM_shape[0];
float m = -INFINITY;
uint mi = NONE;
if constexpr (LAW) {
    for (uint i = t; i < NT; i += 256) {
        if (PI[i] != NONE && (mi == NONE || msv_ink2_better(GM[i], PI[i], m, mi))) {
            m = GM[i];
            mi = PI[i];
        }
    }
} else {
    for (uint i = t; i < NT; i += 256) {
        if (PI[i] != NONE && (mi == NONE || msv_ink2_better(PM[i], PI[i], m, mi))) {
            m = PM[i];
            mi = PI[i];
        }
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
if (lane == 0) {
    rm[sg] = m;
    ri[sg] = mi;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float top = rm[0];
uint at_top = ri[0];
for (uint g = 1; g < 8; ++g) {
    if (ri[g] != NONE && (at_top == NONE || msv_ink2_better(rm[g], ri[g], top, at_top))) {
        top = rm[g];
        at_top = ri[g];
    }
}
if (tg == 0) {
    if constexpr (LAW) {
        float pm = -INFINITY;
        for (uint i = t; i < NT; i += 256) {
            if (PI[i] != NONE) pm = max(pm, PM[i]);
        }
        pm = simd_max(pm);
        if (lane == 0) rm[sg] = pm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float top_m = rm[0];
        for (uint g = 1; g < 8; ++g) top_m = max(top_m, rm[g]);
        float s = 0.0f;
        for (uint i = t; i < NT; i += 256) {
            if (PI[i] != NONE) s += PS[i] * metal::exp(PM[i] - top_m);
        }
        s = simd_sum(s);
        if (lane == 0) rs[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            float tot = 0.0f;
            for (uint g = 0; g < 8; ++g) tot += rs[g];
            TOKN[0] = int(at_top);
            CONF[0] = 1.0f / tot;
            LSE[0] = top_m + metal::log(tot);
        }
    } else {
        const float cap = CAP[0];
        const float ctop = cap > 0 ? cap * metal::precise::tanh(top / cap) : top;
        float s = 0.0f;
        for (uint i = t; i < NT; i += 256) {
            if (PI[i] != NONE) s += PS[i] * metal::precise::exp((cap > 0 ? cap * metal::precise::tanh(PM[i] / cap) : PM[i]) - ctop);
        }
        s = simd_sum(s);
        if (lane == 0) rs[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            float tot = 0.0f;
            for (uint g = 0; g < 8; ++g) tot += rs[g];
            TOKN[0] = int(at_top);
            CONF[0] = 1.0f / tot;
        }
    }
}
if constexpr (!DO_RNN) return;

for (uint r = t; r < R; r += 256) {
    const float x = P[r];
    const float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
    const T keep = T(x < 0 ? y : 1.0f - y);
    const T cand = T(metal::precise::tanh(P[R + r]));
    const T nst = T(T(keep * ST[r]) + T(T(T(1.0f) - keep) * cand));
    st_s[r] = nst;
    prev_s[r] = PREV[size_t(at_top) * R + r];
    if (tg == 0) NST[r] = nst;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

constexpr uint IN = 2 * R + 2 * H;
const uint at = uint(AT[0]);
const device T* w = W + size_t(row) * IN;
const device T* rp = ROWS + size_t(at) * H;
float acc = 0.0f;
if constexpr ((R % 4 == 0) && (H % 4 == 0)) {
    for (uint c = lane * 4; c < R; c += 128) {
        const float4 w0 = float4(*(const device vec<T, 4>*)(w + c));
        const float4 x0 = float4(*(const threadgroup vec<T, 4>*)(st_s + c));
        acc += dot(w0, x0);
        const float4 w1 = float4(*(const device vec<T, 4>*)(w + R + c));
        const float4 x1 = float4(*(const threadgroup vec<T, 4>*)(prev_s + c));
        acc += dot(w1, x1);
    }
    for (uint c = lane * 4; c < H; c += 128) {
        const float4 w2 = float4(*(const device vec<T, 4>*)(w + 2 * R + c));
        const float4 x2 = float4(*(const device vec<T, 4>*)(rp + c));
        acc += dot(w2, x2);
        const float4 w3 = float4(*(const device vec<T, 4>*)(w + 2 * R + H + c));
        const float4 x3 = float4(*(const device vec<T, 4>*)(NEW + c));
        acc += dot(w3, x3);
    }
} else {
    for (uint c = lane; c < R; c += 32) acc += float(w[c]) * float(st_s[c]);
    for (uint c = lane; c < R; c += 32) acc += float(w[R + c]) * float(prev_s[c]);
    for (uint c = lane; c < H; c += 32) acc += float(w[2 * R + c]) * float(rp[c]);
    for (uint c = lane; c < H; c += 32) acc += float(w[2 * R + H + c]) * float(NEW[c]);
}
acc = simd_sum(acc);
if (lane == 0) PN[row] = float(T(T(acc) + B[row]));
