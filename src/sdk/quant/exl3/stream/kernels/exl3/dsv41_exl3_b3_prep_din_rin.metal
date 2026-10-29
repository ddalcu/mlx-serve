#define BANKED_rn(c) (bk == 0u ? rn0[slot * rn0_strides[0] + (c) * rn0_strides[1]] : (bk == 1u ? rn1[slot * rn1_strides[0] + (c) * rn1_strides[1]] : rn2[slot * rn2_strides[0] + (c) * rn2_strides[1]]))

    const float SCALE = as_type<float>(1035273459u);
    uint tid = thread_position_in_threadgroup.x;
    uint blk = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint c0 = blk * 128u + tid;
    uint c1 = c0 + 64u;

    threadgroup float B[128];
    uint slot_ = ids[row * ids_strides[0]]; uint bk = slot_ >> 24u; uint slot = slot_ & 0xFFFFFFu;
    B[tid] = hid[row * hid_strides[0] + c0 * hid_strides[1]] * float(BANKED_rn(c0));
    B[tid + 64u] = hid[row * hid_strides[0] + c1 * hid_strides[1]] * float(BANKED_rn(c1));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h = 1u; h < 128u; h <<= 1u) {
        uint k = tid & (h - 1u);
        uint j = ((tid - k) << 1u) + k;
        float a = B[j]; float b = B[j + h];
        B[j] = a + b; B[j + h] = a - b;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    out[row * 2304u + c0] = B[tid] * SCALE;
    out[row * 2304u + c1] = B[tid + 64u] * SCALE;
