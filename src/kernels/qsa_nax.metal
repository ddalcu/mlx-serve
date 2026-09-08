// Sparse QSA for bf16 D=256, GQA12, on G17.
// Three bf16 terms retain float32 softmax precision under relaxed NAX MMA.
// D and block IDs are contiguous; KV rows are aligned for uint4 loads.
using namespace mlx::steel;
constexpr int BD = 256, LD = 264, TDH = 8, TK = BK / 16;
const int qL=q_shape[2], kL=k_shape[2], Hq=q_shape[1], Hk=k_shape[1];
const int gqa=Hq/Hk, KB=blocks_shape[2];
const int s=threadgroup_position_in_grid.x;
const int hk=threadgroup_position_in_grid.y;
const int bb=threadgroup_position_in_grid.z;
const ushort warp=simdgroup_index_in_threadgroup;
const ushort lane=thread_index_in_simdgroup;
const int tid=thread_index_in_threadgroup;
const int p=kL-qL+s, complete=(p+1)/RATIO;
const int sel_len=min(complete,KB)*RATIO, tail_start=complete*RATIO;
const int L=sel_len+p+1-tail_start;
const device int* blk=blocks+(long)bb*blocks_strides[0]+(long)s*blocks_strides[1];
const device T* Kp=k+bb*k_strides[0]+hk*k_strides[1];
const device T* Vp=v+bb*v_strides[0]+hk*v_strides[1];
const device T* Qp=q+bb*q_strides[0]+(long)(hk*gqa)*q_strides[1]+(long)s*q_strides[2]+warp*128;
threadgroup T KV[BK*LD];
threadgroup float exchange[2][512];
using ST=NAXTile<float,1,TK>;
using OT=NAXTile<float,1,TDH>;
OT O; O.clear();
NAXTile<T,1,1> Q[TDH];
STEEL_PRAGMA_UNROLL
for (short d=0;d<TDH;++d) Q[d].load_rows(Qp+d*16, int(q_strides[1]), short(gqa));
const short2 coord=BaseNAXFrag::get_coord();
float2 max_score=float2(-3e38f),sum_score=float2(0.0f);
const float scale=scl[0]*1.44269504088896340736f;
for(int t0=0;t0<L;t0+=BK) {
  const int rows=min(BK,L-t0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int i=tid;i<BK*32;i+=64) {
    int r=i>>5,c=i&31; uint4 value=uint4(0);
    if(r<rows) {
      int pos=msv_qsa_pos(blk,t0+r,sel_len,tail_start,RATIO);
      value=*((const device uint4*)(Kp+(long)pos*k_strides[2])+c);
    }
    *((threadgroup uint4*)(KV+r*LD)+c)=value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ST S; S.clear();
  STEEL_PRAGMA_UNROLL
  for(short ik=0;ik<TK;ik+=2) {
    STEEL_PRAGMA_UNROLL
    for(short d=0;d<TDH;++d) {
      NAXTile<T,2,1> K;
      K.template load<T,LD,1>(KV+ik*16*LD+warp*128+d*16);
      BaseNAXFrag::mma(S.frag_at(0,ik),S.frag_at(0,ik+1),Q[d].frag_at(0,0),
        metal::false_type{},K.frag_at(0,0),K.frag_at(1,0),metal::true_type{});
    }
    STEEL_PRAGMA_UNROLL
    for(short i=0;i<8;++i) {
      exchange[warp][lane*16+i]=S.frag_at(0,ik)[i];
      exchange[warp][lane*16+8+i]=S.frag_at(0,ik+1)[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for(short i=0;i<8;++i) {
      S.frag_at(0,ik)[i]+=exchange[1-warp][lane*16+i];
      S.frag_at(0,ik+1)[i]+=exchange[1-warp][lane*16+8+i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  STEEL_PRAGMA_UNROLL
  for(short ik=0;ik<TK;++ik) {
    STEEL_PRAGMA_UNROLL
    for(short i=0;i<8;++i) {
      S.frag_at(0,ik)[i]*=scale;
      if(ik*16+coord.x+(i%4)>=rows) S.frag_at(0,ik)[i]=-INFINITY;
    }
  }
  float2 new_max=max_score;
  S.template row_reduce<QsaMax>(new_max);
  S.template row_bin_op<QsaExpSub>(new_max);
  float2 factor=metal::exp2(max_score-new_max);
  max_score=new_max;
  sum_score*=factor;
  S.template row_reduce<QsaSum>(sum_score);
  O.template row_bin_op<QsaMul>(factor);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int i=tid;i<BK*32;i+=64) {
    int r=i>>5,c=i&31; uint4 value=uint4(0);
    if(r<rows) {
      int pos=msv_qsa_pos(blk,t0+r,sel_len,tail_start,RATIO);
      value=*((const device uint4*)(Vp+(long)pos*v_strides[2])+c);
    }
    *((threadgroup uint4*)(KV+r*LD)+c)=value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  STEEL_PRAGMA_UNROLL
  for(short d=0;d<TDH;d+=2) {
    STEEL_PRAGMA_UNROLL
    for(short ik=0;ik<TK;++ik) {
      NAXTile<T,1,2> V;
      V.template load<T,LD,1>(KV+ik*16*LD+warp*128+d*16);
      NAXTile<T,1,1> Shi, Slo, Stail;
      for(short j=0;j<8;++j) {
        Shi.frag_at(0,0)[j]=T(S.frag_at(0,ik)[j]);
        float residual=S.frag_at(0,ik)[j]-float(Shi.frag_at(0,0)[j]);
        Slo.frag_at(0,0)[j]=T(residual);
        Stail.frag_at(0,0)[j]=T(residual-float(Slo.frag_at(0,0)[j]));
      }
      BaseNAXFrag::mma(O.frag_at(0,d),O.frag_at(0,d+1),Shi.frag_at(0,0),
        metal::false_type{},V.frag_at(0,0),V.frag_at(0,1),metal::false_type{});
      BaseNAXFrag::mma(O.frag_at(0,d),O.frag_at(0,d+1),Slo.frag_at(0,0),
        metal::false_type{},V.frag_at(0,0),V.frag_at(0,1),metal::false_type{});
      BaseNAXFrag::mma(O.frag_at(0,d),O.frag_at(0,d+1),Stail.frag_at(0,0),
        metal::false_type{},V.frag_at(0,0),V.frag_at(0,1),metal::false_type{});
    }
  }
}
float2 inv=1.0f/sum_score;
O.template row_bin_op<QsaMul>(inv);
device T* Op=out+(((long)bb*Hq+hk*gqa)*qL+s)*BD+warp*128;
O.store_rows(Op,qL*BD,short(gqa));
