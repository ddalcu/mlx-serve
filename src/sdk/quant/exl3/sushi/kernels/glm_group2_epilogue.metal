if constexpr(SERIAL_REDUCTION) {
 for(uint j=0u;j<2u;++j) {
  for(uint si=0u;si<8u;++si) partial[sg*256u+pos[si]]=acc[j][si];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lid<16u) {
   float sum=0.0f;
   for(uint r=0u;r<16u;++r) for(uint g=0u;g<SGS;++g) sum+=partial[g*256u+r*16u+lid];
   const uint member=j==0u?first:partner;
   y[size_t(member)*uint(ODIM)+ot*TILE+lid]=half(exl3_out(sum));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
} else {
 for(uint j=0u;j<2u;++j) for(uint si=0u;si<8u;++si) partial[j*1024u+sg*256u+pos[si]]=acc[j][si];
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(lid<32u) {
  const uint j=lid/16u,col=lid%16u;
  float sum=0.0f;
  for(uint r=0u;r<16u;++r) for(uint g=0u;g<SGS;++g) sum+=partial[j*1024u+g*256u+r*16u+col];
  const uint member=j==0u?first:partner;
  y[size_t(member)*uint(ODIM)+ot*TILE+col]=half(exl3_out(sum));
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
}