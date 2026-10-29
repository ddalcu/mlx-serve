
    // _rmsnorm(_hc_pre(h, pre), w, eps): the J-term collapse in col_reduce_small order (lsize.y = J <= 8: lane j
    // holds pre_j * h_j + 0, then t = s_0; t = s_j + t), cast to T (the stream dtype), the row_reduce_simple
    // partition of the squares over D, mean multiply, + eps, precise::rsqrt, xf * r, w * (.), cast to T.
    const int lid = int(thread_position_in_threadgroup.x);
    const int ls = int(threads_per_threadgroup.x);
    const uint nsg = simdgroups_per_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int z = int(threadgroup_position_in_grid.y);
    const int S = x_shape[1];
    const int J = x_shape[2];
    const int Dd = x_shape[3];
    const int bi = z / S;
    const int si = z - bi * S;
    const device T* xz = x + bi * x_strides[0] + si * x_strides[1];
    const int64_t sj = x_strides[2];
    const int64_t sd = x_strides[3];
    const int64_t pb = bi * pre_strides[0] + si * pre_strides[1];
    const int64_t sp = pre_strides[2];
    const int64_t sw = w_strides[0];
    threadgroup float red[32];
    threadgroup float rb[1];
    const int blocks = Dd / (ls * 4);
    const int extra = Dd - blocks * ls * 4;
    const int index = lid * 4;
    auto col = [&](int d) -> float {
        float t = metal::fma(q3pf_ld(pre, pb), q3pf_ld(xz, d * sd), -0.0f) + 0.0f;
        for (int j = 1; j < J; j++) {
            t = (metal::fma(q3pf_ld(pre, pb + j * sp), q3pf_ld(xz, j * sj + d * sd), -0.0f) + 0.0f) + t;
        }
        return static_cast<float>(static_cast<T>(t));
    };
    float yv[MAXE];
    int nv = 0;
    float tot = 0.0f;
    int e = index;
    for (int b = 0; b < blocks; b++) {
        for (int i = 0; i < 4; i++) {
            const float v = col(e + i);
            yv[nv++] = v;
            tot = metal::fma(v, v, -0.0f) + tot;
        }
        e += ls * 4;
    }
    if (index + 4 <= extra) {
        for (int i = 0; i < 4; i++) {
            const float v = col(e + i);
            yv[nv++] = v;
            tot = metal::fma(v, v, -0.0f) + tot;
        }
    } else {
        for (int i = 0; index + i < extra; i++) {
            const float v = col(e + i);
            yv[nv++] = v;
            tot = metal::fma(v, v, -0.0f) + tot;
        }
    }
    tot = simd_sum(tot);
    if (nsg > 1) {
        if (lane == 0) {
            red[sg] = tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tot = simd_sum(uint(lid) < nsg ? red[lid] : 0.0f);
    }
    if (lid == 0) {
        rb[0] = metal::precise::rsqrt(metal::fma(tot, inv[0], -0.0f) + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rr = rb[0];
    device T* orow = y + int64_t(z) * Dd;
    nv = 0;
    e = index;
    for (int b = 0; b < blocks; b++) {
        for (int i = 0; i < 4; i++) {
            const int d = e + i;
            orow[d] = static_cast<T>(metal::fma(q3pf_ld(w, d * sw), metal::fma(yv[nv++], rr, -0.0f), -0.0f));
        }
        e += ls * 4;
    }
    if (index + 4 <= extra) {
        for (int i = 0; i < 4; i++) {
            const int d = e + i;
            orow[d] = static_cast<T>(metal::fma(q3pf_ld(w, d * sw), metal::fma(yv[nv++], rr, -0.0f), -0.0f));
        }
    } else {
        for (int i = 0; index + i < extra; i++) {
            const int d = e + i;
            orow[d] = static_cast<T>(metal::fma(q3pf_ld(w, d * sw), metal::fma(yv[nv++], rr, -0.0f), -0.0f));
        }
    }
