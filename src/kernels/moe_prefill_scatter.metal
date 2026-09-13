uint t=thread_index_in_threadgroup,g=threadgroup_position_in_grid.x,n=indices_shape[0];
threadgroup atomic_uint used[512];
atomic_store_explicit(&used[t],0u,memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=g*1024u+t;i<min(n,(g+1u)*1024u);i+=512u){
  uint e=indices[i],rank=atomic_fetch_add_explicit(&used[e],1u,memory_order_relaxed);
  uint at=offsets[g*512u+e]+rank;
  order[at]=i;inverse[i]=at;sorted_ids[at]=e;
}
