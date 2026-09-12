// Partition two sorted selections into private-0, private-1 and shared blocks.
constexpr int CAP=KB+1;
threadgroup int bucket[3*CAP];
threadgroup int counts[3];
const int gi=threadgroup_position_in_grid.x, bb=threadgroup_position_in_grid.z;
const int lane=thread_index_in_threadgroup;
const int qL=blocks_shape[1], s=gi*2, kL=kvlen;
const int p0=kL-qL+s, p1=p0+1;
const int c0=(p0+1)/RATIO, c1=(p1+1)/RATIO;
const int n0=min(c0,KB), n1=s+1<qL?min(c1,KB):0;
const int t0=(p0+1)%RATIO!=0, t1=s+1<qL && (p1+1)%RATIO!=0;
const device int* b0=blocks+bb*blocks_strides[0]+s*blocks_strides[1];
const device int* b1=b0+blocks_strides[1];
if(lane==0) {
  int i=0,j=0,a0=0,a1=0,a2=0;
  while(i<n0+t0 || j<n1+t1) {
    const int a=i<n0 ? b0[i*blocks_strides[2]] : (i<n0+t0?c0:2147483647);
    const int b=j<n1 ? b1[j*blocks_strides[2]] : (j<n1+t1?c1:2147483647);
    if(a==b) { bucket[2*CAP+a2++]=a; ++i; ++j; }
    else if(a<b) { bucket[a0++]=a; ++i; }
    else { bucket[CAP+a1++]=b; ++j; }
  }
  counts[0]=a0; counts[1]=a1; counts[2]=a2;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const int end0=(counts[0]*RATIO+31)/32;
const int end1=end0+(counts[1]*RATIO+31)/32;
const int end2=end1+(counts[2]*RATIO+31)/32;
const int NG=(qL+1)/2;
device int* pos=tilepos+((long)bb*NG+gi)*NT*8;
device int* masks=tilemask+((long)bb*NG+gi)*NT;
for(int it=lane;it<NT;it+=32)masks[it]=it<end0?1:(it<end1?2:(it<end2?3:0));
for(int i=lane;i<NT*8;i+=32) {
  int it=i/8;
  int b=it<end0?0:(it<end1?1:2);
  int start=b==0?0:(b==1?end0:end1);
  int r=i-start*8;
  pos[i]=it<end2 && r<counts[b] ? bucket[b*CAP+r]*RATIO : -1;
}
