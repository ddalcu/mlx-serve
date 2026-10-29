
    // q3_attnfuse_softmax: one threadgroup per (query row z, head h); row_reduce_simple partition.
    constexpr int NR = 4;
    const int lid = int(thread_position_in_threadgroup.x);
    const int ls = int(threads_per_threadgroup.x);
    const uint nsg = simdgroups_per_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int h = int(threadgroup_position_in_grid.y);
    const int z = int(threadgroup_position_in_grid.z);
    const int S = qk_shape[1];
    const int H = qk_shape[2];
    const int K = qk_shape[3];
    const int bi = z / S;
    const int si = z - bi * S;
    const device float* srow = qk + bi * qk_strides[0] + si * qk_strides[1] + h * qk_strides[2];
    const int64_t sst = qk_strides[3];
    const device bool* vrow = valid + bi * valid_strides[0] + si * valid_strides[1];
    const int64_t vst = valid_strides[2];
    const float scale = scl[0];
    const float snk = sink[h * sink_strides[2]];
    device float* erow = ex + (int64_t(z) * H + h) * K;
    threadgroup float red[32];
    threadgroup float mbc[1];
    const int blocks = K / (ls * NR);
    const int extra = K - blocks * ls * NR;
    const int index = lid * NR;

    // pass 1: row max (per_thread_row_reduce<Max>, then threadgroup_reduce<Max>)
    float tot = Limits<float>::min;
    int j = index;
    for (int b = 0; b < blocks; b++) {
        for (int i = 0; i < NR; i++) {
            tot = q3af_max_op(q3af_val(srow, sst, vrow, vst, j + i, scale), tot);
        }
        j += ls * NR;
    }
    if (index + NR <= extra) {
        for (int i = 0; i < NR; i++) {
            tot = q3af_max_op(q3af_val(srow, sst, vrow, vst, j + i, scale), tot);
        }
    } else {
        for (int i = 0; index + i < extra; i++) {
            tot = q3af_max_op(q3af_val(srow, sst, vrow, vst, j + i, scale), tot);
        }
    }
    tot = q3af_simd_max(tot);
    if (nsg > 1) {
        if (lane == 0) {
            red[sg] = tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tot = q3af_simd_max(uint(lid) < nsg ? red[lid] : Limits<float>::min);
    }
    if (lid == 0) {
        mbc[0] = q3af_maximum(tot, snk);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m = mbc[0];

    // pass 2: ex = exp(v - m) written once, row sum (per_thread_row_reduce<Sum>, threadgroup_reduce<Sum>)
    float acc = 0.0f;
    j = index;
    for (int b = 0; b < blocks; b++) {
        for (int i = 0; i < NR; i++) {
            const float e = metal::precise::exp(q3af_val(srow, sst, vrow, vst, j + i, scale) - m);
            erow[j + i] = e;
            acc = e + acc;
        }
        j += ls * NR;
    }
    if (index + NR <= extra) {
        for (int i = 0; i < NR; i++) {
            const float e = metal::precise::exp(q3af_val(srow, sst, vrow, vst, j + i, scale) - m);
            erow[j + i] = e;
            acc = e + acc;
        }
    } else {
        for (int i = 0; index + i < extra; i++) {
            const float e = metal::precise::exp(q3af_val(srow, sst, vrow, vst, j + i, scale) - m);
            erow[j + i] = e;
            acc = e + acc;
        }
    }
    acc = simd_sum(acc);
    if (nsg > 1) {
        if (lane == 0) {
            red[sg] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = simd_sum(uint(lid) < nsg ? red[lid] : 0.0f);
    }
    if (lid == 0) {
        denom[int64_t(z) * H + h] = acc + metal::precise::exp(snk - m);
    }
