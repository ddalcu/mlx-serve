
    // q3rc_router_tail: one threadgroup per row, thread n = expert n
    const int M = part_shape[1];
    const int row = int(threadgroup_position_in_grid.y);
    const int n = int(thread_position_in_threadgroup.x);
    threadgroup float key_sh[N];
    threadgroup float w_sh[TOPK];
    float g = part[int64_t(row) * N + n];
    for (int p = 1; p < P; ++p) {
        g += part[(int64_t(p) * M + row) * N + n];
    }
    g = g / 1.000000000e+00f;
    const float sp = q3rc_logaddexp(g, 0.0f);
    const float score = metal::precise::sqrt(sp);
    const float biased = score + bias[n];
    key_sh[n] = biased;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // rank = experts that precede n in the stock order (biased descending, ties by index, NaN last)
    const bool mn = metal::isnan(biased);
    int rank = 0;
    for (int j = 0; j < N; ++j) {
        const float o = key_sh[j];
        const bool on = metal::isnan(o);
        const bool before = mn ? (!on || j < n) : (!on && (o > biased || (o == biased && j < n)));
        rank += before ? 1 : 0;
    }
    if (rank < TOPK) {
        indices[row * TOPK + rank] = n;
        w_sh[rank] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (n == 0) {
        float s = 0.0f;
        for (int j = 0; j < TOPK; ++j) {
            s = w_sh[j] + s;                         // row_reduce_small: total = op(val, total), in order
        }
        const float d = s + 1.000000000e-20f;
        for (int j = 0; j < TOPK; ++j) {
            weights[row * TOPK + j] = (w_sh[j] / d) * 1.500000000e+00f;
        }
    }
