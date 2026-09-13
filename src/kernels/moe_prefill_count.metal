// Exact expert grouping; stability within an expert is unnecessary.
threadgroup atomic_uint hist[512];
uint t=thread_index_in_threadgroup,g=threadgroup_position_in_grid.x;
atomic_store_explicit(&hist[t],0u,memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
uint n=indices_shape[0];
for(uint i=g*1024u+t;i<min(n,(g+1u)*1024u);i+=512u)
  atomic_fetch_add_explicit(&hist[indices[i]],1u,memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
counts[g*512u+t]=atomic_load_explicit(&hist[t],memory_order_relaxed);
