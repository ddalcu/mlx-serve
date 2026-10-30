const uint first=uint(threadgroup_position_in_grid.y);
const uint group_lane=uint(thread_index_in_simdgroup);
const uint group_eid=uint(slots[first]);
uint rank=0u, partner=first;
for(uint b=0u;b<uint(NSLOTS);b+=32u) {
 const bool hit=b+group_lane<uint(NSLOTS) && uint(slots[b+group_lane])==group_eid;
 const uint mask=uint(static_cast<simd_vote::vote_t>(simd_ballot(hit)));
 if(b+32u<=first) rank+=popcount(mask);
 else if(b<=first) rank+=popcount(mask & ((1u<<(first-b))-1u));
 const uint low=uint(max(int(first+1u)-int(b),0));
 const uint after=low>=32u?0u:(mask & (0xffffffffu<<low));
 if(partner==first && after!=0u) partner=b+ctz(after);
}
if((rank&1u)!=0u) return;
threadgroup float partial[SERIAL_REDUCTION?1024:2048];