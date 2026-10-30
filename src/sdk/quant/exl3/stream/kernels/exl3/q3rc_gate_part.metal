
    // q3rc_gate_part: part[p, v, n] = sum over columns c of partition p of w[n, c] * x[v, c]
    constexpr int R = 2, SG = 8, MAXV = 8, STEPS = KP / 128;
    static_assert(K % KP == 0 && KP % 128 == 0, "partition must tile K by 128-column lane steps");
    static_assert(N % R == 0, "rows per simdgroup must tile N");
    const int M = x_shape[0];
    const int nb = int(threadgroup_position_in_grid.y);
    const int p = int(threadgroup_position_in_grid.z);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    const int n0 = nb * (SG * R) + sg * R;
    if (n0 >= N) {
        return;                                     // whole simdgroup past the last row block
    }
    const int c0 = p * KP + 4 * lane;
    float acc[MAXV][R];
    for (int v = 0; v < MAXV; ++v) {
        for (int r = 0; r < R; ++r) {
            acc[v][r] = 0.0f;
        }
    }
    for (int j = 0; j < STEPS; ++j) {
        const int c = c0 + 128 * j;
        float wv[R][4];
        for (int r = 0; r < R; ++r) {
            const packed_ushort4 u = *(const device packed_ushort4*)(w + int64_t(n0 + r) * K + c);  // 4 bf16
            wv[r][0] = as_type<float>(uint(u[0]) << 16);
            wv[r][1] = as_type<float>(uint(u[1]) << 16);
            wv[r][2] = as_type<float>(uint(u[2]) << 16);
            wv[r][3] = as_type<float>(uint(u[3]) << 16);
        }
        for (int v = 0; v < MAXV; ++v) {
            if (v < M) {
                const float4 xv = float4(*(const device packed_float4*)(x + int64_t(v) * K + c));
                for (int r = 0; r < R; ++r) {
                    acc[v][r] = fma(wv[r][0], xv.x, acc[v][r]);
                    acc[v][r] = fma(wv[r][1], xv.y, acc[v][r]);
                    acc[v][r] = fma(wv[r][2], xv.z, acc[v][r]);
                    acc[v][r] = fma(wv[r][3], xv.w, acc[v][r]);
                }
            }
        }
    }
    for (int v = 0; v < MAXV; ++v) {
        if (v < M) {
            for (int r = 0; r < R; ++r) {
                const float t = simd_sum(acc[v][r]);
                if (lane == 0) {
                    part[(int64_t(p) * M + v) * N + n0 + r] = t;
                }
            }
        }
    }
