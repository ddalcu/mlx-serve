
    // rsqrt(mean(square(flat), -1) + eps) of DecoderLayer._mixes: row_reduce_simple<float, float, Sum> partition over
    // the flattened [J * D] row (N_READS 4, blocks of ls*4, tail at lid*4; simd_sum; lanes < nsg), then the stock
    // mean multiply (sum * f32(1/numel)), + eps, precise::rsqrt.  One threadgroup per row, ls = MLX
    // threadgroup_size_from_row_size(J * D).
    const int lid = int(thread_position_in_threadgroup.x);
    const int ls = int(threads_per_threadgroup.x);
    const uint nsg = simdgroups_per_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int z = int(threadgroup_position_in_grid.y);
    const int S = x_shape[1];
    const int J = x_shape[2];
    const int Dd = x_shape[3];
    const int K = J * Dd;
    const int bi = z / S;
    const int si = z - bi * S;
    const device T* xz = x + bi * x_strides[0] + si * x_strides[1];
    const int64_t sj = x_strides[2];
    const int64_t sd = x_strides[3];
    threadgroup float red[32];
    const int blocks = K / (ls * 4);
    const int extra = K - blocks * ls * 4;
    const int index = lid * 4;
    auto sq = [&](int e) -> float {
        const int j = e / Dd;
        const int d = e - j * Dd;
        const float v = q3pf_ld(xz, j * sj + d * sd);
        return metal::fma(v, v, -0.0f);
    };
    float tot = 0.0f;
    int e = index;
    for (int b = 0; b < blocks; b++) {
        for (int i = 0; i < 4; i++) {
            tot = sq(e + i) + tot;
        }
        e += ls * 4;
    }
    if (index + 4 <= extra) {
        for (int i = 0; i < 4; i++) {
            tot = sq(e + i) + tot;
        }
    } else {
        for (int i = 0; index + i < extra; i++) {
            tot = sq(e + i) + tot;
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
        r[z] = metal::precise::rsqrt(metal::fma(tot, inv[0], -0.0f) + eps[0]);
    }
