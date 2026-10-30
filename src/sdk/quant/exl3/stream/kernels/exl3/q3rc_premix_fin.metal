
    // q3rc_premix_fin: out[v, n] = the P partials of (v, n) summed in partition order
    const int M = part_shape[1];
    const int n = int(thread_position_in_grid.x);
    const int v = int(thread_position_in_grid.y);
    if (n < N && v < M) {
        float s = part[int64_t(v) * N + n];
        for (int p = 1; p < P; ++p) {
            s += part[(int64_t(p) * M + v) * N + n];
        }
        out[v * N + n] = s;
    }
