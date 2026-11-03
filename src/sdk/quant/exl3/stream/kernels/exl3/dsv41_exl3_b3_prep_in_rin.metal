#define BANKED_rg(c) (bk == 0u ? rg0[slot * rg0_strides[0] + (c) * rg0_strides[1]] : (bk == 1u ? rg1[slot * rg1_strides[0] + (c) * rg1_strides[1]] : rg2[slot * rg2_strides[0] + (c) * rg2_strides[1]]))
#define BANKED_ru(c) (bk == 0u ? ru0[slot * ru0_strides[0] + (c) * ru0_strides[1]] : (bk == 1u ? ru1[slot * ru1_strides[0] + (c) * ru1_strides[1]] : ru2[slot * ru2_strides[0] + (c) * ru2_strides[1]]))

    const float SCALE = as_type<float>(1035273459u);
    uint tid = thread_position_in_threadgroup.x;
    uint blk = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint c0 = blk * 128u + tid;
    uint c1 = c0 + 64u;

    threadgroup float BG[128];
    threadgroup float BU[128];
    uint slot_ = ids[row * ids_strides[0]]; uint bk = slot_ >> 24u; uint slot = slot_ & 0xFFFFFFu;
    uint t = uint(tok[row * tok_strides[0]]);
    float x0 = float(x[t * x_strides[0] + c0 * x_strides[1]]);
    float x1 = float(x[t * x_strides[0] + c1 * x_strides[1]]);
    BG[tid] = x0 * float(BANKED_rg(c0));
    BG[tid + 64u] = x1 * float(BANKED_rg(c1));
    BU[tid] = x0 * float(BANKED_ru(c0));
    BU[tid + 64u] = x1 * float(BANKED_ru(c1));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
    threadgroup float* B = BG;

    for (uint h = 1u; h < 128u; h <<= 1u) {
        uint k = tid & (h - 1u);
        uint j = ((tid - k) << 1u) + k;
        float a = B[j]; float b = B[j + h];
        B[j] = a + b; B[j + h] = a - b;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    }
    {
    threadgroup float* B = BU;

    for (uint h = 1u; h < 128u; h <<= 1u) {
        uint k = tid & (h - 1u);
        uint j = ((tid - k) << 1u) + k;
        float a = B[j]; float b = B[j + h];
        B[j] = a + b; B[j + h] = a - b;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    }
    xg[row * 5120u + c0] = BG[tid] * SCALE;
    xg[row * 5120u + c1] = BG[tid + 64u] * SCALE;
    xu[row * 5120u + c0] = BU[tid] * SCALE;
    xu[row * 5120u + c1] = BU[tid + 64u] * SCALE;
