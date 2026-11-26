// Inkling 2's drafter, batched Markov step pick for B lanes: grid (256, B, 1), threadgroup (256, 1, 1).
// PM, PI, PS [B, groups] f32 / u32 / f32; P [B, 3 R] f32; ST [B, 1, R] (T); CAP [1] f32.
// Outputs: TOKN [B] int32, CONF [B] f32, NST [B, 1, R] (T).
threadgroup float rm[8];
threadgroup uint ri[8];
threadgroup float rs[8];
constexpr uint NONE = 0xffffffffu;
const uint t = thread_position_in_threadgroup.x;
const uint b = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
const uint sg = t / 32u;
const uint NT = PM_shape[1];
const size_t base = size_t(b) * NT;
float m = -INFINITY;
uint mi = NONE;
for (uint i = t; i < NT; i += 256u) {
    if (PI[base + i] != NONE && (mi == NONE || msv_ink2_better(PM[base + i], PI[base + i], m, mi))) {
        m = PM[base + i];
        mi = PI[base + i];
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
for (uint g = 1u; g < 8u; ++g) {
    if (ri[g] != NONE && (at_top == NONE || msv_ink2_better(rm[g], ri[g], top, at_top))) {
        top = rm[g];
        at_top = ri[g];
    }
}
const float cap = CAP[0];
const float ctop = cap > 0 ? cap * metal::precise::tanh(top / cap) : top;
float s = 0.0f;
for (uint i = t; i < NT; i += 256u) {
    if (PI[base + i] != NONE) s += PS[base + i] * metal::precise::exp((cap > 0 ? cap * metal::precise::tanh(PM[base + i] / cap) : PM[base + i]) - ctop);
}
s = simd_sum(s);
if (lane == 0u) rs[sg] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (t == 0u) {
    float tot = 0.0f;
    for (uint g = 0u; g < 8u; ++g) tot += rs[g];
    TOKN[b] = int(at_top);
    CONF[b] = 1.0f / tot;
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
