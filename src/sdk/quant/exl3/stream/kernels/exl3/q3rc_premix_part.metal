
    // q3rc_premix_part: part[p, v, n] = sum over columns c of partition p of w[n, c] * x[v, c]
    constexpr int SG = 8, RS = N / SG, MAXV = 8, STEPS = KP / 128;
    static_assert(N % SG == 0, "rows must tile the 8 simdgroups");
    static_assert(K % KP == 0 && KP % 128 == 0, "partition must tile K by 128-column lane steps");
    const int M = x_shape[0];
    const int p = int(threadgroup_position_in_grid.z);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    const int n0 = sg * RS;
    const int c0 = p * KP + 4 * lane;
    float acc[MAXV][RS];
    for (int v = 0; v < MAXV; ++v) {
        for (int r = 0; r < RS; ++r) {
            acc[v][r] = 0.0f;
        }
    }
    for (int j = 0; j < STEPS; ++j) {
        const int c = c0 + 128 * j;
        float4 wv[RS];
        for (int r = 0; r < RS; ++r) {
            wv[r] = float4(*(const device packed_float4*)(w + int64_t(n0 + r) * K + c));
        }
        for (int v = 0; v < MAXV; ++v) {
            if (v < M) {
                const float4 xv = float4(*(const device packed_float4*)(x + int64_t(v) * K + c));
                for (int r = 0; r < RS; ++r) {
                    acc[v][r] = fma(wv[r].x, xv.x, acc[v][r]);
                    acc[v][r] = fma(wv[r].y, xv.y, acc[v][r]);
                    acc[v][r] = fma(wv[r].z, xv.z, acc[v][r]);
                    acc[v][r] = fma(wv[r].w, xv.w, acc[v][r]);
                }
            }
        }
    }
    for (int v = 0; v < MAXV; ++v) {
        if (v < M) {
            for (int r = 0; r < RS; ++r) {
                const float t = simd_sum(acc[v][r]);
                if (lane == 0) {
                    part[(int64_t(p) * M + v) * N + n0 + r] = t;
                }
            }
        }
    }
