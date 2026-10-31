// One threadgroup of 128 threads per token: u = silu(gate) * content (GC holds gate then content), delta = u @ final_map
// (3 per head, f32 FMAs), then the angle delta[head, axis(p)] * freq[p] * pi of each rotation pair, out as cos and sin.
threadgroup float u[1024];
threadgroup float delta[192];
const uint row = threadgroup_position_in_grid.x;
const uint lid = thread_position_in_threadgroup.x;
const uint K = GC_shape[1] / 2, D = 3 * uint(H), P = 32;
for (uint k = lid; k < K; k += 128) {
    const float g = GC[size_t(row) * 2 * K + k];
    u[k] = g / (1.0f + metal::precise::exp(-g)) * GC[size_t(row) * 2 * K + K + k];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint j = lid; j < D; j += 128) {
    float acc = 0;
    for (uint k = 0; k < K; k++) acc = metal::fma(u[k], WF[k * D + j], acc);
    delta[j] = acc;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint o = lid; o < uint(H) * P; o += 128) {
    const uint h = o / P, p = o % P;
    const float df = delta[h * 3 + AXIS[p]] * FQ[p];
    const float a = df * M_PI_F;
    OC[size_t(row) * H * P + o] = metal::precise::cos(a);
    OS[size_t(row) * H * P + o] = metal::precise::sin(a);
}
