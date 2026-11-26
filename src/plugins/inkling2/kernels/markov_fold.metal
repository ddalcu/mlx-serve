// Inkling 2's drafter, one Markov step: the latent's bias over the vocab and each threadgroup's share of the pick.
// latent = T(tanh(P[2 R + r])); logit[v] = BASE[AT, v] + float(T(T(latent . FOLD[:, v]) + FB[v])) for v < VOCAB, as
// the matmul and the bias add round it. FOLD [R, V] and FB [V] (T), BASE [1, block - 1, V] f32, P [3 R] f32, AT [1]
// int32. A threadgroup of TG threads takes VPT columns a thread: PM and PI its argmax in MLX's order
// (markov_header.metal), PS its sum of exp(c(logit) - c(PM)) under the soft cap c(x) = CAP tanh(x / CAP) (CAP [1]
// f32; 0 = none), as the drafter's probability reads the head's capped logits.
threadgroup float lat[R];
threadgroup float rm[TG / 32];
threadgroup uint ri[TG / 32];
threadgroup float rs[TG / 32];
constexpr uint NONE = 0xffffffffu;
const uint t = thread_position_in_threadgroup.x;
const uint tg = threadgroup_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = t / 32;
const uint V = FOLD_shape[1];
const size_t at = size_t(AT[0]);
for (uint r = t; r < R; r += TG) lat[r] = float(T(metal::precise::tanh(P[2 * R + r])));
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint v0 = (tg * TG + t) * VPT;
float acc[VPT];
for (uint j = 0; j < VPT; j++) acc[j] = 0.0f;
if (v0 < VOCAB) for (uint r = 0; r < R; r++) {
    const vec<T, VPT> f = *(const device vec<T, VPT>*)(FOLD + size_t(r) * V + v0);
    for (uint j = 0; j < VPT; j++) acc[j] += lat[r] * float(f[j]);
}
float lg[VPT];
float m = -INFINITY;
uint mi = NONE;
for (uint j = 0; j < VPT; j++) {
    const uint v = v0 + j;
    lg[j] = v < VOCAB ? BASE[at * V + v] + float(T(T(acc[j]) + FB[v])) : -INFINITY;
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
if (lane == 0) {
    rm[sg] = m;
    ri[sg] = mi;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float top = rm[0];
uint at_top = ri[0];
for (uint g = 1; g < TG / 32; g++) {
    if (ri[g] != NONE && (at_top == NONE || msv_ink2_better(rm[g], ri[g], top, at_top))) {
        top = rm[g];
        at_top = ri[g];
    }
}
const float cap = CAP[0];
const float ctop = cap > 0 ? cap * metal::precise::tanh(top / cap) : top;
float s = 0.0f;
for (uint j = 0; j < VPT; j++) {
    if (v0 + j < VOCAB) s += metal::precise::exp((cap > 0 ? cap * metal::precise::tanh(lg[j] / cap) : lg[j]) - ctop);
}
s = simd_sum(s);
if (lane == 0) rs[sg] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (t == 0) {
    float tot = 0.0f;
    for (uint g = 0; g < TG / 32; g++) tot += rs[g];
    PM[tg] = top;
    PI[tg] = at_top;
    PS[tg] = tot;
}
