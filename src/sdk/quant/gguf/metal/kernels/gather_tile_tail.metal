    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (m0 + lid < m_end && ids[m0 + lid] == e)
        for (uint c = 0; c < TILE_ROWS; c++) out[size_t(m0 + lid) * N + r0 + c] = stage[lid * TILE_ROWS + c];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}