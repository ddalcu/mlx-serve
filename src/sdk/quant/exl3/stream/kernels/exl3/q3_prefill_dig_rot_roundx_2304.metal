#pragma METAL fp contract(off)
    const float SCALE = as_type<float>(1035273459u);
    uint l = thread_index_in_simdgroup;
    uint blk = threadgroup_position_in_grid.x * 6u + simdgroup_index_in_threadgroup;
    uint r = threadgroup_position_in_grid.y;
    uint c0 = blk * 128u + l;
    int64_t src = (int64_t) r * act_strides[0] + (int64_t) c0;
    int64_t rb = (int64_t) slots[rhs[r]] * rin_strides[0];
    float w[4];
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
        float a = float(act[src + (int64_t) (32u * q)]);
        float s = float(rin[rb + (int64_t) (c0 + 32u * q) * rin_strides[1]]);
        w[q] = a * s;
    }
    _Pragma("clang loop unroll(full)")
    for (uint h = 1u; h < 32u; h <<= 1u) {
        bool upper = (l & h) != 0u;
        _Pragma("clang loop unroll(full)")
        for (uint q = 0u; q < 4u; ++q) {
            float o = simd_shuffle_xor(w[q], (ushort) h);
            w[q] = upper ? (o - w[q]) : (w[q] + o);
        }
    }
    float a0 = w[0] + w[1];
    float a1 = w[0] - w[1];
    float a2 = w[2] + w[3];
    float a3 = w[2] - w[3];
    w[0] = a0 + a2;
    w[2] = a0 - a2;
    w[1] = a1 + a3;
    w[3] = a1 - a3;
    int64_t dst = (int64_t) r * 2304 + (int64_t) c0;
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
        out[dst + (int64_t) (32u * q)] = static_cast<half>(w[q] * SCALE);
    }
