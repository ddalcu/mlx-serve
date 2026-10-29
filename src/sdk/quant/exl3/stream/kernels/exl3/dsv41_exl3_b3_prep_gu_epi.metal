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
    BG[tid] = zg[row * zg_strides[0] + c0 * zg_strides[1]];
    BG[tid + 64u] = zg[row * zg_strides[0] + c1 * zg_strides[1]];
    BU[tid] = zu[row * zu_strides[0] + c0 * zu_strides[1]];
    BU[tid + 64u] = zu[row * zu_strides[0] + c1 * zu_strides[1]];
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

    {
        uint e = tid; uint col = blk * 128u + e;
        float g = BG[e] * SCALE;
        float rgv = float(BANKED_rg(col));
        float gm = g * rgv;
        float gate = metal::isnan(gm) ? gm : (gm < 10.0f ? gm : 10.0f);
        float u = BU[e] * SCALE;
        float ruv = float(BANKED_ru(col));
        float um = u * ruv;
        float up0 = metal::isnan(um) ? um : (um > -10.0f ? um : -10.0f);
        float up = metal::isnan(up0) ? up0 : (up0 < 10.0f ? up0 : 10.0f);
        float y = 1 / (1 + metal::exp(metal::abs(gate)));
        float sg = (gate < 0) ? y : 1 - y;
        float sl = gate * sg;
        out[row * 2304u + col] = sl * up;
    }

    {
        uint e = tid + 64u; uint col = blk * 128u + e;
        float g = BG[e] * SCALE;
        float rgv = float(BANKED_rg(col));
        float gm = g * rgv;
        float gate = metal::isnan(gm) ? gm : (gm < 10.0f ? gm : 10.0f);
        float u = BU[e] * SCALE;
        float ruv = float(BANKED_ru(col));
        float um = u * ruv;
        float up0 = metal::isnan(um) ? um : (um > -10.0f ? um : -10.0f);
        float up = metal::isnan(up0) ? up0 : (up0 < 10.0f ? up0 : 10.0f);
        float y = 1 / (1 + metal::exp(metal::abs(gate)));
        float sg = (gate < 0) ? y : 1 - y;
        float sl = gate * sg;
        out[row * 2304u + col] = sl * up;
    }

