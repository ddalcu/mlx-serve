
    // GLM-5.3 EXL3 decode, stage 3, per (128-block of the output, pair): the SwiGLU middle of the pair's four minis
    // (t128(t128(zg) rout_g * scale) and t128(zu) rout_u * scale, silu(g) u, t128(h rin_d) * scale), the down GEMV of each
    // mini over that middle, and the output block sum_r t128(zd_r) * scale * rout_d[r] times the pair's score, f32.
    // Eight simdgroups: the middle's 4 x MINI/128 blocks among them, then simdgroup s the output tile s of the block for
    // every mini, then simdgroup 0 the block's rotation and the minis' sum. pt: per pair (x row, slot packed bank << 24 |
    // row, routed position, 0); DEV: the call's routed positions are the pairs, pt their packed slots; sc: the router's
    // weights by routed position. A pair skips at slot 0xFFFFFFFF or score 0: no read, its output block zero.
    constexpr uint IT = uint(MINI) / 16u;
    constexpr uint OT = uint(HID) / 16u;
    constexpr uint NW = 8u * uint(K);
    constexpr uint NIB = uint(MINI) / 128u;
    threadgroup float mid[4u * uint(MINI)];
    threadgroup float zd[4u * 128u];
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x;
    const uint p = threadgroup_position_in_grid.y;
    const uint packed = DEV ? pt[p] : pt[4u * p + 1u];
    const uint pos = DEV ? p : pt[4u * p + 2u];
    if (packed == 0xffffffffu || sc[pos] == 0.0f) {
        if (sg == 0u) {
            device float* yp = y + size_t(p) * uint(HID) + blk * 128u + lane;
            yp[0] = 0.0f;
            yp[32] = 0.0f;
            yp[64] = 0.0f;
            yp[96] = 0.0f;
        }
        return;
    }
    const uint bank = packed >> 24u;
    const size_t srow = size_t(packed & 0xffffffu) * 4u;
    const device half* rg = bank == 0u ? og0 : (bank == 1u ? og1 : og2);
    const device half* ru = bank == 0u ? ou0 : (bank == 1u ? ou1 : ou2);
    const device half* rd = bank == 0u ? di0 : (bank == 1u ? di1 : di2);
    for (uint job = sg; job < 4u * NIB; job += 8u) {
        const uint r = job / NIB;
        const uint ib = job - r * NIB;
        const uint c0 = ib * 128u + lane;
        const size_t zb = (size_t(p) * 4u + r) * uint(MINI) + c0;
        const size_t rb = (srow + r) * uint(MINI);
        const size_t vb = rb + c0;
        float g[4];
        float u[4];
        dig2_widen(zg, zb, rg, rb, c0, lane, g);
        dig2_widen(zu, zb, ru, rb, c0, lane, u);
        float4 h;
        for (uint q = 0u; q < 4u; ++q) h[q] = (g[q] * Dig2Sigmoid()(g[q])) * u[q];
        const float4 v = g53_fwht128(h * float4(float(rd[vb]), float(rd[vb + 32u]), float(rd[vb + 64u]), float(rd[vb + 96u])), ushort(lane)) * G53_HSC;
        threadgroup float* mb = mid + r * uint(MINI) + ib * 128u + g53_lane_slot(lane);
        mb[0] = v.x;
        mb[32] = v.y;
        mb[64] = v.z;
        mb[96] = v.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device ushort* code = bank == 0u ? cd0 : (bank == 1u ? cd1 : cd2);
    const uint tn = blk * 8u + sg;
    const g53_lane_plan pl = g53_plan<K>(lane);
    for (uint r = 0u; r < 4u; ++r) {
        const device uint* tw = reinterpret_cast<const device uint*>(code) + ((srow + r) * (IT * OT) + tn) * NW;
        const threadgroup float4* xr = reinterpret_cast<const threadgroup float4*>(mid + r * uint(MINI)) + (lane & 3u);
        float acc[8];
        for (uint m = 0u; m < 8u; ++m) acc[m] = 0.0f;
        #pragma clang loop unroll_count(4)
        for (uint tk = 0u; tk < IT; ++tk) {
            half w8[8];
            g53_lane_weights<K>(tw + size_t(tk) * (OT * NW), pl, w8);
            g53_fma8(acc, xr[tk * 4u], w8);
        }
        const float2 c = g53_cols(acc);
        if ((lane & 3u) == 0u) {
            zd[r * 128u + sg * 16u + (lane >> 2u)] = c.x;
            zd[r * 128u + sg * 16u + (lane >> 2u) + 8u] = c.y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        const device half* ro = bank == 0u ? do0 : (bank == 1u ? do1 : do2);
        float4 yb = float4(0.0f);
        for (uint r = 0u; r < 4u; ++r) {
            const threadgroup float* zr = zd + r * 128u + lane;
            const float4 t = g53_fwht128(float4(zr[0], zr[32], zr[64], zr[96]), ushort(lane)) * G53_HSC;
            const size_t ob = (srow + r) * uint(HID) + blk * 128u + lane;
            yb += t * float4(float(ro[ob]), float(ro[ob + 32u]), float(ro[ob + 64u]), float(ro[ob + 96u]));
        }
        yb *= sc[pos];
        device float* yp = y + size_t(p) * uint(HID) + blk * 128u + lane;
        yp[0] = yb.x;
        yp[32] = yb.y;
        yp[64] = yb.z;
        yp[96] = yb.w;
    }
