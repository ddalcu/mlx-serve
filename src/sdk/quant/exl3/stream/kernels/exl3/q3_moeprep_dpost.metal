
    const float SCALE = as_type<float>(1035273459u);
    uint tid = thread_position_in_threadgroup.x;
    uint blk = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint c0 = blk * 128u + tid;
    uint c1 = c0 + 64u;

    threadgroup float B[128];
    uint slot = ids[row * ids_strides[0]];
    B[tid] = zd[row * zd_strides[0] + c0 * zd_strides[1]];
    B[tid + 64u] = zd[row * zd_strides[0] + c1 * zd_strides[1]];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h = 1u; h < 128u; h <<= 1u) {
        uint k = tid & (h - 1u);
        uint j = ((tid - k) << 1u) + k;
        float a = B[j]; float b = B[j + h];
        B[j] = a + b; B[j + h] = a - b;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    out[row * 5120u + c0] = (B[tid] * SCALE) * float(rd[slot * rd_strides[0] + c0 * rd_strides[1]]);
    out[row * 5120u + c1] = (B[tid + 64u] * SCALE) * float(rd[slot * rd_strides[0] + c1 * rd_strides[1]]);
