#define GG_TILE(GR) {
    const uint lid = thread_index_in_threadgroup;
    const uint r0 = threadgroup_position_in_grid.x * TILE_ROWS;
    const uint m0 = threadgroup_position_in_grid.y * TILE_TOKENS;
    simdgroup_float8x8 acc[GR][TILE_ROWS / 8];
    for (uint t = 0; t < GR; t++)
        for (uint g = 0; g < TILE_ROWS / 8; g++) acc[t][g] = simdgroup_float8x8(0.0f);
    for (uint u = 0; u < K / UNIT; u++) {
        if (lid < TILE_ROWS) {
            GgTile o = {tile + (lid / 8) * 8 * UNIT, lid % 8};
            DEQ_AT(GG_W, r0 + lid, u / UPB, u % UPB, o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < UNIT; c += 8) {
            for (uint t = 0; t < GR; t++) {
                if (t < GG_T0 || t >= GG_T1) continue;
                simdgroup_float8x8 xm;
                simdgroup_load(xm, x + size_t(m0 + 8 * t) * K + u * UNIT + c, K);
                for (uint g = 0; g < TILE_ROWS / 8; g++) {
                    simdgroup_float8x8 wm;
                    simdgroup_load(wm, tile + (g * UNIT + c) * 8, 8);
                    simdgroup_multiply_accumulate(acc[t][g], xm, wm, acc[t][g]);
                }
            }
        }
    }
    for (uint t = 0; t < GR; t++) {
        if (t < GG_T0 || t >= GG_T1) continue;
        for (uint g = 0; g < TILE_ROWS / 8; g++) GG_STORE(t, g);
    }
}