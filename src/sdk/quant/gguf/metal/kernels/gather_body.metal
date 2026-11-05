// One activation row against RPT weight rows of one expert: each lane walks every 32nd unit and loads that unit of
// the activation once for all RPT rows (the matvec body reloads it per row). Rows of whole 32-unit strides only: the
// matvec body's flat walk keeps every lane busy on the rest.
const uint lid = thread_index_in_simdgroup;
const uint r0 = (threadgroup_position_in_grid.y * simdgroups_per_threadgroup + simdgroup_index_in_threadgroup) * RPT;
const uint nu = K / UNIT;
const uint z = threadgroup_position_in_grid.z;
const device T *xz = GG_X;
float acc[RPT] = {0};
for (uint u = lid; u < nu; u += threads_per_simdgroup) {
    GgDotR o;
    for (uint i = 0; i < UNIT / 4; i++) o.x[i] = float4(*(const device vec<T, 4> *)(xz + u * UNIT + 4 * i));
    for (uint r = 0; r < RPT; r++) {
        if (r0 + r >= N) break;
        o.acc = 0.0f;
        DEQ_AT(GG_W, r0 + r, u / UPB, u % UPB, o);
        acc[r] += o.acc.x + o.acc.y + o.acc.z + o.acc.w;
    }
}
for (uint r = 0; r < RPT; r++) {
    const float total = simd_sum(acc[r]);
    if (lid == 0 && r0 + r < N) out[size_t(z) * N + r0 + r] = T(total);
}
