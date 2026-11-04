
    // GLM-5.3 EXL3 decode, stage 1: each routed pair's gate and up inputs for its four minis, f16 (t128(x[tok] * rin),
    // DeepSeek-V4.1's dig2_round), stored in GEMV lane order. One simdgroup per (128-block, pair x mini, projection). pt: per pair (x row, slot packed
    // bank << 24 | row, routed position, 0); DEV: the call's routed positions are the pairs, pt their packed slots (x row
    // position / TOPK). A pair skips (no read, nothing written) at slot 0xFFFFFFFF or score 0. A pair's mini r is slot row
    // 4 row + r of its bank.
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x;
    const uint pm = threadgroup_position_in_grid.y;
    const uint proj = threadgroup_position_in_grid.z;
    const uint p = pm >> 2u;
    const uint tok = DEV ? p / uint(TOPK) : pt[4u * p];
    const uint packed = DEV ? pt[p] : pt[4u * p + 1u];
    const uint pos = DEV ? p : pt[4u * p + 2u];
    if (packed == 0xffffffffu || sc[pos] == 0.0f) return;
    const uint bank = packed >> 24u;
    const size_t mrow = size_t(packed & 0xffffffu) * 4u + (pm & 3u);
    const size_t xb = size_t(tok) * uint(HID) + blk * 128u + lane;
    float h[4];
    for (uint q = 0u; q < 4u; ++q) h[q] = float(x[xb + 32u * q]);
    const device half* rin = proj == 0u ? (bank == 0u ? rg0 : (bank == 1u ? rg1 : rg2)) : (bank == 0u ? ru0 : (bank == 1u ? ru1 : ru2));
    dig2_round(h, rin, mrow * uint(HID), blk * 128u + lane, proj == 0u ? ug : uu, size_t(pm) * uint(HID) + blk * 128u + g53_lane_slot(lane), lane);
