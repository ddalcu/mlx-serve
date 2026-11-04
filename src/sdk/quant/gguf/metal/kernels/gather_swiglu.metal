const uint lid = thread_index_in_simdgroup;
const uint r0 = (threadgroup_position_in_grid.y * simdgroups_per_threadgroup + simdgroup_index_in_threadgroup) * RPT;
const uint nu = K / UNIT;
const uint z = threadgroup_position_in_grid.z;
const device uint8_t *wg_z = wg + size_t(ids[z]) * N * ROWB;
const device uint8_t *wu_z = wu + size_t(ids[z]) * N * ROWB;
const device T *x_z = x + size_t(z / XDIV) * K;
float acc[2][RPT] = {{0}};
for (uint i = lid; i < RPT * nu; i += threads_per_simdgroup) {
    const uint r = i / nu;
    const uint u = i % nu;
    if (r0 + r >= N) break;
    GgDot<T, 1, K> g = {x_z + u * UNIT, {0}};
    DEQ_AT(wg_z, r0 + r, u / UPB, u % UPB, g);
    acc[0][r] += g.acc[0].x + g.acc[0].y + g.acc[0].z + g.acc[0].w;
    GgDot<T, 1, K> v = {x_z + u * UNIT, {0}};
    DEQ_AT(wu_z, r0 + r, u / UPB, u % UPB, v);
    acc[1][r] += v.acc[0].x + v.acc[0].y + v.acc[0].z + v.acc[0].w;
}
for (uint r = 0; r < RPT; r++) {
    // Each projection rounds to T as its own gather writes it, then MLX's SwiGLU chain (kernels/mlp/swiglu.metal).
    const T g = T(simd_sum(acc[0][r]));
    const T v = T(simd_sum(acc[1][r]));
    if (lid == 0 && r0 + r < N) {
        const T act = g * sigtab[as_type<ushort>(g)];
        out[size_t(z) * N + r0 + r] = act * v;
    }
}
