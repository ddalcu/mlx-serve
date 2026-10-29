
    const float SCALE = as_type<float>(1035273459u);
    uint tid = thread_position_in_threadgroup.x;
    uint blk = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint c0 = blk * 128u + tid;
    uint c1 = c0 + 64u;

    threadgroup float BG[128];
    threadgroup float BU[128];
    uint slot = ids[row * ids_strides[0]];
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
        float rgv = float(rg[slot * rg_strides[0] + col * rg_strides[1]]);
        float gm = g * rgv;
        float gate = metal::isnan(gm) ? gm : (gm < 10.0f ? gm : 10.0f);
        float u = BU[e] * SCALE;
        float ruv = float(ru[slot * ru_strides[0] + col * ru_strides[1]]);
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
        float rgv = float(rg[slot * rg_strides[0] + col * rg_strides[1]]);
        float gm = g * rgv;
        float gate = metal::isnan(gm) ? gm : (gm < 10.0f ? gm : 10.0f);
        float u = BU[e] * SCALE;
        float ruv = float(ru[slot * ru_strides[0] + col * ru_strides[1]]);
        float um = u * ruv;
        float up0 = metal::isnan(um) ? um : (um > -10.0f ? um : -10.0f);
        float up = metal::isnan(up0) ? up0 : (up0 < 10.0f ? up0 : 10.0f);
        float y = 1 / (1 + metal::exp(metal::abs(gate)));
        float sg = (gate < 0) ? y : 1 - y;
        float sl = gate * sg;
        out[row * 2304u + col] = sl * up;
    }

