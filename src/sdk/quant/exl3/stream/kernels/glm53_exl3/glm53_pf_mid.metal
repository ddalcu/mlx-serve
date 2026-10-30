
    // GLM-5.3 EXL3 prompt, stage 2: each wave row's SwiGLU middle per mini, f16, through DeepSeek-V4.1's rotation helpers
    // (exl3/header_dig2_x.metal: dig2_widen of the gate and up, the SwiGLU unclamped, dig2_round with the down's rin): the
    // down GEMM's A rows. One simdgroup per (128-block of the mini's intermediate, row x mini). rt: per wave row (x row,
    // slot row).
    const uint lane = thread_index_in_simdgroup;
    const uint ib = threadgroup_position_in_grid.x;
    const uint pm = threadgroup_position_in_grid.y;
    const uint i = pm >> 2u;
    const size_t rb = (size_t(rt[2u * i + 1u]) * 4u + (pm & 3u)) * uint(MINI);
    const uint c0 = ib * 128u + lane;
    const size_t src = size_t(pm) * uint(MINI) + c0;
    float g[4];
    float u[4];
    float h[4];
    dig2_widen(zg, src, og, rb, c0, lane, g);
    dig2_widen(zu, src, ou, rb, c0, lane, u);
    for (uint q = 0u; q < 4u; ++q) h[q] = (g[q] * Dig2Sigmoid()(g[q])) * u[q];
    dig2_round(h, di, rb, c0, hm, src, lane);
