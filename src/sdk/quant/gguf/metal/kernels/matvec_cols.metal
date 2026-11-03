const uint lid = thread_index_in_simdgroup;
const uint r0 = (threadgroup_position_in_grid.y * simdgroups_per_threadgroup + simdgroup_index_in_threadgroup) * RPT;
const uint nu = K / UNIT;
const uint z = threadgroup_position_in_grid.z;
float acc[RPT][M] = {{0}};
for (uint u = lid; u < nu; u += threads_per_simdgroup) {
    float4 xr[M * (UNIT / 4)];
    for (uint m = 0; m < M; m++) {
        const device T *xm = GG_X + m * K + u * UNIT;
        for (uint i = 0; i < UNIT / 4; i++) xr[m * (UNIT / 4) + i] = float4(xm[4 * i], xm[4 * i + 1], xm[4 * i + 2], xm[4 * i + 3]);
    }
    for (uint r = 0; r < RPT; r++) {
        if (r0 + r >= N) break;
        GgDotRegs<M, UNIT / 4> o = {xr, {0}};
        DEQ_AT(GG_W, r0 + r, u / UPB, u % UPB, o);
        for (uint m = 0; m < M; m++) acc[r][m] += o.acc[m].x + o.acc[m].y + o.acc[m].z + o.acc[m].w;
    }
}
for (uint r = 0; r < RPT; r++) {
    for (uint m = 0; m < M; m++) {
        const float total = simd_sum(acc[r][m]);
        if (lid == 0 && r0 + r < N) out[size_t(z * M + m) * N + r0 + r] = T(total);
    }
}
