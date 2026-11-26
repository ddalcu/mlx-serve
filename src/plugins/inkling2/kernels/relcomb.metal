// Inkling 2: relsplit's splits combined and times the output gate.
// grid (32, L * HQ, 1), threadgroup (32, 1, 1): 1 SIMD group per (row, head) shares split weights and streams float4.
// PART [NS, L * HQ, D] f32, ML [NS, L * HQ, 2] f32, GATE [L * HQ * D] f32 (sigmoid). OUT [1, L, HQ * D] (T).
const uint lane = thread_index_in_simdgroup;
const uint lh = threadgroup_position_in_grid.y;
const uint NS = PART_shape[0];
const uint LH = PART_shape[1];
float m = -INFINITY;
for (uint s = lane; s < NS; s += 32u) m = max(m, ML[(s * LH + lh) * 2u]);
m = simd_max(m);
float den = 0.0f;
threadgroup float sw[128];
for (uint s = lane; s < NS; s += 32u) {
    const float w = precise::exp2(ML[(s * LH + lh) * 2u] - m);
    den += w * ML[(s * LH + lh) * 2u + 1u];
    if (s < 128u) sw[s] = w;
}
const float den_sum = simd_sum(den);
const uint cached = min(NS, 128u);
simdgroup_barrier(mem_flags::mem_threadgroup);
const uint D4 = D / 4u;
for (uint d4 = lane; d4 < D4; d4 += 32u) {
    const uint col = d4 * 4u;
    float4 num = float4(0.0f);
    for (uint s = 0u; s < cached; s++) {
        num += sw[s] * (*((const device float4*)(PART + (size_t(s) * LH + lh) * D + col)));
    }
    for (uint s = cached; s < NS; s++) {
        const float w = precise::exp2(ML[(s * LH + lh) * 2u] - m);
        num += w * (*((const device float4*)(PART + (size_t(s) * LH + lh) * D + col)));
    }
    const float4 g = *((const device float4*)(GATE + size_t(lh) * D + col));
    const vec<T, 4> q = vec<T, 4>(num / den_sum);
    *((device vec<T, 4>*)(OUT + size_t(lh) * D + col)) = vec<T, 4>(float4(q) * g);
}
