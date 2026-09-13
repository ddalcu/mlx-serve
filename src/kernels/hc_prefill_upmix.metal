using namespace metal;
using namespace mpp::tensor_ops;
const int M=x_shape[0],K=x_shape[1],H=normed_shape[2],N=H*4;
const int sg=simdgroup_index_in_threadgroup,tid=thread_index_in_threadgroup;
const int n0=threadgroup_position_in_grid.x*64+(sg%2)*32;
const int m0=threadgroup_position_in_grid.y*64+(sg/2)*32;
auto A=tensor<device T,dextents<int,2>,tensor_inline>((device T*)x,dextents<int,2>(K,M));
auto W=tensor<device T,dextents<int,2>,tensor_inline>((device T*)w,dextents<int,2>(K,N));
constexpr auto desc=matmul2d_descriptor(32,32,64,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc,execution_simdgroup> op;
auto total=op.template get_destination_cooperative_tensor<decltype(A),decltype(W),float>();
_Pragma("clang loop unroll(full)")
for(int i=0;i<32;i++)total[i]=0;
for(int k=0;k<K;k+=64) {
  auto a=A.template slice<64,32>(k,m0);
  auto w=W.template slice<64,32>(k,n0);
  op.run(a,w,total);
}
// Four neighboring HC columns share one lane on this cooperative layout.
_Pragma("clang loop unroll(full)")
for(int i=0;i<32;i+=4) {
  auto index=total.get_multidimensional_index(i);
  const int col=(n0+index[0])/4,row=m0+index[1];
  T value=T(0);
  _Pragma("clang loop unroll(full)")
  for(int h=0;h<4;h++) {
    auto next=total.get_multidimensional_index(i+h);
    if(next[0]!=index[0]+h || next[1]!=index[1]) {
      out[(long)row*H+col]=T(NAN);
      return;
    }
    T up=T(total[i+h]);
    T product=T(float(sigtab[as_type<ushort>(up)])*float(normed[((long)row*4+h)*H+col]));
    value=h==0?product:T(float(product)+float(value));
  }
  out[(long)row*H+col]=T(float(value)*0.25f);
}
