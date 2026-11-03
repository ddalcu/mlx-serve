constexpr int K = HC * D;
constexpr int MIX = (2 + HC) * HC;
constexpr int SL = K / NT;
constexpr float EPS = EPS_INT * 1e-9;
const uint g = threadgroup_position_in_grid.x;
const uint row = threadgroup_position_in_grid.y;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = tid % 32;
const uint sg = tid / 32;
float acc[MIX + 1];
for (int r = 0; r <= MIX; ++r) acc[r] = 0.0f;
for (int k = int(g) * SL + int(tid); k < int(g + 1) * SL; k += TPG) {
    const float hv = HVAL(k);
    STORE_H(k, hv);
    acc[MIX] = fma(hv, hv, acc[MIX]);
    for (int r = 0; r < MIX; ++r) acc[r] = fma(hv, fn[r * K + k], acc[r]);
}
constexpr int NSG = TPG / 32;
threadgroup float red[NSG][MIX + 1];
threadgroup float tot[MIX + 1];
threadgroup bool last;
for (int r = 0; r <= MIX; ++r) {
    const float v = simd_sum(acc[r]);
    if (lane == 0) red[sg][r] = v;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// Partials cross threadgroups through L2: plain loads can hit another core's stale L1.
if (tid <= uint(MIX)) {
    float t = 0.0f;
    for (int q = 0; q < NSG; ++q) t += red[q][tid];
    atomic_store_explicit((device atomic_float*)parts + (row * NT + g) * (MIX + 1) + tid, t, memory_order_relaxed);
}
atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
threadgroup_barrier(mem_flags::mem_device);
if (tid == 0) {
    const uint old = atomic_fetch_add_explicit((device atomic_uint*)counter + row, 1u, memory_order_relaxed);
    last = ((old + 1u) % uint(NT)) == 0u;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (!last) return;
atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
threadgroup float all[NT * (MIX + 1)];
for (uint q = tid; q < uint(NT * (MIX + 1)); q += TPG) all[q] = atomic_load_explicit((device atomic_float*)parts + row * NT * (MIX + 1) + q, memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid <= uint(MIX)) {
    float t = 0.0f;
    for (int q = 0; q < NT; ++q) t += all[q * (MIX + 1) + tid];
    tot[tid] = t;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float inv = metal::precise::rsqrt(tot[MIX] / float(K) + eps[0]);
threadgroup float mix_n[MIX];
if (tid < uint(MIX)) mix_n[tid] = tot[tid] * inv;
float p[HC];
for (int j = 0; j < HC; ++j) {
    const float z = tot[j] * inv * scale[0] + base[j];
    p[j] = 1.0f / (1.0f + metal::fast::exp(-z)) + EPS;
}
constexpr int D4 = D / 4;
constexpr int CPT = (D4 + TPG - 1) / TPG;
float4 cf[CPT];
float s2 = 0.0f;
for (int u = 0; u < CPT; ++u) {
    cf[u] = 0.0f;
    const int d4 = u * TPG + int(tid);
    if (d4 < D4) {
        const int d = 4 * d4;
        float4 xs[HC];
        for (int j = 0; j < HC; ++j) xs[j] = TAIL4(j * D + d);
        const float4 c = fma(float4(p[0]), xs[0], fma(float4(p[1]), xs[1], fma(float4(p[2]), xs[2], float4(p[3]) * xs[3])));
        cf[u] = float4(vec<T, 4>(c));
        s2 += dot(cf[u], cf[u]);
    }
}
s2 = simd_sum(s2);
threadgroup float red2[NSG];
if (lane == 0) red2[sg] = s2;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    const float t = simd_sum(lane < uint(NSG) ? red2[lane] : 0.0f);
    if (lane == 0) red2[0] = t;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float inv2 = metal::precise::rsqrt(red2[0] / float(D) + eps[1]);
for (int u = 0; u < CPT; ++u) {
    const int d4 = u * TPG + int(tid);
    if (d4 < D4) ((device vec<T, 4>*)(normed + row * D))[d4] = ((const device vec<T, 4>*)nw)[d4] * vec<T, 4>(cf[u] * inv2);
}
if (tid == 0) msv_hc_gates1<ITERS>((threadgroup const float*)mix_n, base, scale[1], scale[2], EPS, post + row * HC, comb + row * HC * HC);