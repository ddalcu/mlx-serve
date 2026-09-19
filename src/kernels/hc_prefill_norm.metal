// Adapted from Apple's rms_single_row; preserve its four-value reduction and BF16 rounding.
const int row=threadgroup_position_in_grid.x;
const int h=row%HC, token=row/HC;
const int tid=thread_index_in_threadgroup, lane=thread_index_in_simdgroup;
const int sg=simdgroup_index_in_threadgroup;
threadgroup float partial[32];
threadgroup float inv;
float xv[4], acc=0;
for(int i=0;i<4;i++) {
  T val=x[(long)row*H+tid*4+i];
  if(WR) {
    T delta=T(float(wo[(long)token*H+tid*4+i])*float(wi[token*HC+h]));
    val=T(float(val)+float(delta));
    stream[(long)row*H+tid*4+i]=val;
  }
  xv[i]=float(val); acc+=xv[i]*xv[i];
}
acc=simd_sum(acc);
if(sg==0)partial[lane]=0;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(lane==0)partial[sg]=acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0) { acc=simd_sum(partial[lane]); if(lane==0)inv=metal::precise::rsqrt(acc/float(H)+eps); }
threadgroup_barrier(mem_flags::mem_threadgroup);
for(int i=0;i<4;i++) {
  int col=h*H+tid*4+i;
  T val=T(float(T(xv[i]*inv))*float(w[col]));
  normed[(long)row*H+tid*4+i]=val;
}
