#pragma METAL fp contract(off)
    const float SCALE = as_type<float>(1035273459u);
    uint l = thread_index_in_simdgroup;
    uint blk = threadgroup_position_in_grid.x * 6u + simdgroup_index_in_threadgroup;
    uint r = threadgroup_position_in_grid.y;
    uint c0 = blk * 128u + l;
    uint p = threadgroup_position_in_grid.z;
    const device float* act = p == 0u ? act_g : act_u;
    const constant int64_t* act_strides = p == 0u ? act_g_strides : act_u_strides;
    const device float16_t* rout = p == 0u ? rout_g : rout_u;
    const constant int64_t* rout_strides = p == 0u ? rout_g_strides : rout_u_strides;
    device float* out = p == 0u ? out_g : out_u;
    int64_t src = (int64_t) r * act_strides[0] + (int64_t) c0;
    float w[4];
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
        w[q] = float(act[src + (int64_t) (32u * q)]);
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
    int64_t rb = (int64_t) slots[rhs[r]] * rout_strides[0];
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
        float ro = float(rout[rb + (int64_t) (c0 + 32u * q) * rout_strides[1]]);
        out[dst + (int64_t) (32u * q)] = (w[q] * SCALE) * ro;
    }
