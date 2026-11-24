// Inkling 2's drafter, one Markov step: the token at the argmax in MLX's order over markov_fold's
// groups' PM, PI, PS [groups] f32 / u32 / f32, its probability exp(max - logsumexp) of the capped logits (CAP [1] f32,
// markov_fold's) = 1 / its sum, and the RNN's next
// state from P [3 R] f32 and ST [R] (T) as the chain rounds it: keep = T(sigmoid(P[r])), NST = T(T(keep ST) +
// T(T(1 - keep) T(tanh(P[R + r])))). One threadgroup of 256 threads; TOKN [1] int32, CONF [1] f32, NST [R] T.
threadgroup float rm[8];
threadgroup uint ri[8];
threadgroup float rs[8];
constexpr uint NONE = 0xffffffffu;
const uint t = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = t / 32;
const uint NT = PM_shape[0];
float m = -INFINITY;
uint mi = NONE;
for (uint i = t; i < NT; i += 256) {
    if (PI[i] != NONE && (mi == NONE || msv_ink2_better(PM[i], PI[i], m, mi))) {
        m = PM[i];
        mi = PI[i];
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
for (uint g = 1; g < 8; g++) {
    if (ri[g] != NONE && (at_top == NONE || msv_ink2_better(rm[g], ri[g], top, at_top))) {
        top = rm[g];
        at_top = ri[g];
    }
}
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
    for (uint g = 0; g < 8; g++) tot += rs[g];
    TOKN[0] = int(at_top);
    CONF[0] = 1.0f / tot;
}
// MLX's sigmoid: 1 / (1 + exp(|x|)), mirrored for x >= 0.
for (uint r = t; r < R; r += 256) {
    const float x = P[r];
    const float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
    const T keep = T(x < 0 ? y : 1.0f - y);
    const T cand = T(metal::precise::tanh(P[R + r]));
    NST[r] = T(T(keep * ST[r]) + T(T(T(1.0f) - keep) * cand));
}
