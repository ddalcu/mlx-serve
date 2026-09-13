// Prefix the fixed 512-bin histograms; no CPU count readback.
uint e=thread_index_in_threadgroup,lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
uint groups=counts_shape[0],total=0;
for(uint g=0;g<groups;g++)total+=counts[g*512u+e];
threadgroup uint totals[16];
uint local=simd_prefix_exclusive_sum(total);
uint sum=simd_sum(total);
if(lane==0)totals[sg]=sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
uint cursor=local;
for(uint g=0;g<sg;g++)cursor+=totals[g];
for(uint g=0;g<groups;g++){offsets[g*512u+e]=cursor;cursor+=counts[g*512u+e];}
