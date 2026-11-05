threadgroup half4 gg_grid[{d}];
for (uint i = simdgroup_index_in_threadgroup * threads_per_simdgroup + thread_index_in_simdgroup; i < {d}; i += simdgroups_per_threadgroup * threads_per_simdgroup) {{
    const uint64_t e = {s}[i];
    gg_grid[2 * i] = half4(gg_grid4(uint32_t(e)));
    gg_grid[2 * i + 1] = half4(gg_grid4(uint32_t(e >> 32)));
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
#undef DEQ_AT
#define DEQ_AT(w, row, blk, u, o) deq_{s}_h((w) + (size_t(row) * NB + (blk)) * BB, u, o, gg_grid)
