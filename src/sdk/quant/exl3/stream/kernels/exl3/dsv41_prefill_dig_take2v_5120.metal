#pragma METAL fp contract(off)
    // q3_prefill_dig_rot_take2_5120 retuned (mlx-serve native, 2026-09-30): its arithmetic element for element, one
    // simdgroup per (row, 128-block) writing BOTH outputs from one act load. Lane l holds block elements 4l .. 4l + 3.
    // The 128-point butterflies run in the lane text's bit order, element bit 0 first and bit 6 last (bits 0 and 1 in
    // registers, bits 2 .. 6 across lanes), each pair as (lower + upper, lower - upper): the same f32 values throughout.
    const float SCALE = as_type<float>(1035273459u);
    uint l = thread_index_in_simdgroup;
    uint blk = threadgroup_position_in_grid.x * 8u + simdgroup_index_in_threadgroup;
    uint r = threadgroup_position_in_grid.y;
    uint c0 = blk * 128u + 4u * l;
    int64_t src = (int64_t) ridx[r] * act_strides[0] + (int64_t) c0;
    int64_t slot = (int64_t) slots[rhs[r]];
    int64_t rg = slot * rin_g_strides[0] + (int64_t) c0 * rin_g_strides[1];
    int64_t ru = slot * rin_u_strides[0] + (int64_t) c0 * rin_u_strides[1];
    float wg[4];
    float wu[4];
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
        float a = float(act[src + (int64_t) q]);
        wg[q] = a * float(rin_g[rg + (int64_t) q * rin_g_strides[1]]);
        wu[q] = a * float(rin_u[ru + (int64_t) q * rin_u_strides[1]]);
    }
    {
        float a0 = wg[0] + wg[1];
        float a1 = wg[0] - wg[1];
        float a2 = wg[2] + wg[3];
        float a3 = wg[2] - wg[3];
        wg[0] = a0 + a2;
        wg[2] = a0 - a2;
        wg[1] = a1 + a3;
        wg[3] = a1 - a3;
    }
    {
        float a0 = wu[0] + wu[1];
        float a1 = wu[0] - wu[1];
        float a2 = wu[2] + wu[3];
        float a3 = wu[2] - wu[3];
        wu[0] = a0 + a2;
        wu[2] = a0 - a2;
        wu[1] = a1 + a3;
        wu[3] = a1 - a3;
    }
    _Pragma("clang loop unroll(full)")
    for (uint h = 1u; h < 32u; h <<= 1u) {
        bool upper = (l & h) != 0u;
        _Pragma("clang loop unroll(full)")
        for (uint q = 0u; q < 4u; ++q) {
            float og = simd_shuffle_xor(wg[q], (ushort) h);
            float ou = simd_shuffle_xor(wu[q], (ushort) h);
            wg[q] = upper ? (og - wg[q]) : (wg[q] + og);
            wu[q] = upper ? (ou - wu[q]) : (wu[q] + ou);
        }
    }
    int64_t dst = (int64_t) r * 5120 + (int64_t) c0;
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
        out_g[dst + (int64_t) q] = static_cast<half>(wg[q] * SCALE);
        out_u[dst + (int64_t) q] = static_cast<half>(wu[q] * SCALE);
    }
