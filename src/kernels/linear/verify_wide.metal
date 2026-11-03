uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint kl = lane % 8, r = threadgroup_position_in_grid.y * 8 + sg * 4 + lane / 8;
uint v0 = threadgroup_position_in_grid.x * NV;
uint row = min(r, uint(N - 1));
device const uchar* wr = reinterpret_cast<device const uchar*>(w) + size_t(row) * K;
float result[NV] = {0};
for (uint g = kl; g < K / 64; g += 8) {
    float scale = float(sc[size_t(row) * (K / 64) + g]);
    float bias = float(bi[size_t(row) * (K / 64) + g]);
    #pragma unroll
    for (uint chunk = 0; chunk < 8; ++chunk) {
        uint k0 = g * 64 + chunk * 8;
        float dq[8];
        #pragma unroll
        for (uint i = 0; i < 8; ++i) dq[i] = scale * float(wr[k0 + i]) + bias;
        #pragma unroll
        for (uint v = 0; v < NV; ++v) {
            uint input_row = min(v0 + v, uint(M - 1));
            float acc = 0;
            #pragma unroll
            for (uint i = 0; i < 8; ++i) acc += float(x[size_t(input_row) * K + k0 + i]) * dq[i];
            result[v] += acc;
        }
    }
}
for (uint v = 0; v < NV; ++v) {
    result[v] += simd_shuffle_down(result[v], 4);
    result[v] += simd_shuffle_down(result[v], 2);
    result[v] += simd_shuffle_down(result[v], 1);
}
if (kl == 0 && r < N) {
    for (uint v = 0; v < NV; ++v)
        if (v0 + v < M) y[size_t(v0 + v) * N + r] = T(result[v]);
}