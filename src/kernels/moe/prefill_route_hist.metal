threadgroup atomic_uint hist[E];
const uint lane = thread_index_in_threadgroup;
const uint block = threadgroup_position_in_grid.x;
for (uint e = lane; e < E; e += 256) atomic_store_explicit(&hist[e], 0u, memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint i = block * 256 + lane;
if (i < uint(ids_shape[0]) && ids[i] < E) atomic_fetch_add_explicit(&hist[ids[i]], 1u, memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint e = lane; e < E; e += 256) counts[block * E + e] = atomic_load_explicit(&hist[e], memory_order_relaxed);