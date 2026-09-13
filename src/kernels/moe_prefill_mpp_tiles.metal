uint e=thread_index_in_threadgroup,n=sorted_shape[0],lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
uint lo=0,hi=n;
while(lo<hi){uint mid=(lo+hi)/2;if(sorted[mid]<e)lo=mid+1;else hi=mid;}
uint start=lo;hi=n;
while(lo<hi){uint mid=(lo+hi)/2;if(sorted[mid]<=e)lo=mid+1;else hi=mid;}
uint rows=lo-start,nt=(rows+63u)/64u;
uint base=simd_prefix_exclusive_sum(nt),total=simd_sum(nt);
threadgroup uint totals[16];
if(lane==0)totals[sg]=total;
for(uint i=e;i<(n/64u+512u)*3u;i+=512)tiles[i]=0;
threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
for(uint g=0;g<sg;g++)base+=totals[g];
for(uint b=0;b<nt;b++){uint at=(base+b)*3u;tiles[at]=start;tiles[at+1]=rows;tiles[at+2]=b;}
