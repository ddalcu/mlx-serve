using namespace metal;
using namespace mpp::tensor_ops;
constexpr int BN=128,BK=64,LD=72;
const int K=x_rep_shape[2],N=w_shape[1];
const int ti=threadgroup_position_in_grid.y;
const int start=tiles[ti*3],rows=tiles[ti*3+1],block=tiles[ti*3+2];
if(rows==0)return;
const int expert=indices_input[start];
const int sg=simdgroup_index_in_threadgroup,tid=thread_index_in_threadgroup;
const int m0=start+block*BM+(sg/2)*32;
const int n0=threadgroup_position_in_grid.x*BN+(sg%2)*64;
const long wb=(long)expert*N*(K/8),sb=(long)expert*N*(K/64);
threadgroup T weights[BN*LD];
auto X=tensor<device T,dextents<int,2>,tensor_inline>((device T*)x_rep,dextents<int,2>(K,x_rep_shape[0]));
auto W=tensor<threadgroup T,dextents<int,2>,tensor_inline>(weights,dextents<int,2>(LD,BN));
constexpr auto desc=matmul2d_descriptor(32,64,64,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc,execution_simdgroup> op;
auto total=op.template get_destination_cooperative_tensor<decltype(X),decltype(W),float>();
_Pragma("clang loop unroll(full)")
for(int i=0;i<64;i++)total[i]=0;
for(int k=0;k<K;k+=BK) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int i=tid;i<BN*8;i+=WM*2*32) {
    int row=i/8,col=(i%8)*8,combined=threadgroup_position_in_grid.x*BN+row;
    int rn=combined/2;bool up=(combined&1)!=0;
    uint codes=up?up_w[wb+(long)rn*(K/8)+(k+col)/8]:w[wb+(long)rn*(K/8)+(k+col)/8];
    float sc=float(up?up_scales[sb+(long)rn*(K/64)+k/64]:scales[sb+(long)rn*(K/64)+k/64]);
    float bi=float(up?up_biases[sb+(long)rn*(K/64)+k/64]:biases[sb+(long)rn*(K/64)+k/64]);
    _Pragma("clang loop unroll(full)")
    for(int j=0;j<8;j++)weights[row*LD+col+j]=T(sc*float((codes>>(4*j))&15)+bi);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto x=X.template slice<64,32>(k,m0);
  auto weight=W.template slice<64,64>(0,(sg%2)*64);
  if(m0<start+rows)op.run(x,weight,total);
}
_Pragma("clang loop unroll(full)")
for(int i=0;i<64;i+=2) {
  auto coord=total.get_multidimensional_index(i),next=total.get_multidimensional_index(i+1);
  const int row=m0+coord[1],col=(n0+coord[0])/2;
  if(row<start+rows) {
    if(next[0]!=coord[0]+1 || next[1]!=coord[1]){out[(long)row*N+col]=T(NAN);return;}
    T gate=T(total[i]),up=T(total[i+1]);
    T act=T(float(gate)*float(sigtab[as_type<ushort>(gate)]));
    out[(long)row*N+col]=T(float(act)*float(up));
  }
}
