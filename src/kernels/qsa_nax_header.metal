// Copyright © 2023-2024 Apple Inc. Adapted from MLX steel/attn/nax.h.
// Only the cooperative fragments used by sparse attention are retained.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
namespace mlx { namespace steel {
struct BaseNAXFrag {
  template<typename U> using dtype_frag_t = metal::vec<U,8>;
  static short2 get_coord() {
    ushort lane=__metal_get_thread_index_in_simdgroup(ushort());
    short qid=lane>>2;
    return short2(((qid&2)|(lane&1))*4, (qid&4)|((lane>>1)&3));
  }
  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  inline static constexpr void mma(
      thread dtype_frag_t<CType>& Cn0,
      thread dtype_frag_t<CType>& Cn1,
      const thread dtype_frag_t<AType>& A,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& Bn0,
      const thread dtype_frag_t<BType>& Bn1,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>,
        CType>();

    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 8; i++) {
      ct_a[i] = A[i];
    }

    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 8; i++) {
      ct_b[i] = Bn0[i];
      ct_b[8 + i] = Bn1[i];
    }

    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 8; i++) {
      ct_c[i] = Cn0[i];
      ct_c[8 + i] = Cn1[i];
    }

    gemm_op.run(ct_a, ct_b, ct_c);

    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 8; i++) {
      Cn0[i] = ct_c[i];
      Cn1[i] = ct_c[8 + i];
    }
  }

};
template<typename T,short R,short C> struct NAXTile {
  using frag_type=metal::vec<T,8>;
  frag_type data[R*C];
  thread frag_type& frag_at(short r,short c) thread { return data[r*C+c]; }
  void clear() thread {
    STEEL_PRAGMA_UNROLL
    for(short j=0;j<R*C;j++)data[j]=frag_type(0);
  }
  template<typename U,int LD,int STRIDE> void load(const threadgroup U* p) thread {
    short2 xy=BaseNAXFrag::get_coord();
    STEEL_PRAGMA_UNROLL
    for(short r=0;r<R;r++) {
      STEEL_PRAGMA_UNROLL
      for(short c=0;c<C;c++) {
        STEEL_PRAGMA_UNROLL
        for(short j=0;j<8;j++)data[r*C+c][j]=T(p[(r*16+xy.y+(j/4)*8)*LD+(c*16+xy.x+j%4)*STRIDE]);
      }
    }
  }
  template<typename U> void load_rows(const device U* p,int ld,short rows) thread {
    short2 xy=BaseNAXFrag::get_coord();
    STEEL_PRAGMA_UNROLL
    for(short r=0;r<R;r++) {
      STEEL_PRAGMA_UNROLL
      for(short c=0;c<C;c++) {
        STEEL_PRAGMA_UNROLL
        for(short j=0;j<8;j++) {
          int row=r*16+xy.y+(j/4)*8;
          data[r*C+c][j]=row<rows ? T(p[row*ld+c*16+xy.x+j%4]) : T(0);
        }
      }
    }
  }
  template<typename U> void store_rows(device U* p,int ld,short rows) const thread {
    short2 xy=BaseNAXFrag::get_coord();
    STEEL_PRAGMA_UNROLL
    for(short r=0;r<R;r++) {
      STEEL_PRAGMA_UNROLL
      for(short c=0;c<C;c++) {
        STEEL_PRAGMA_UNROLL
        for(short j=0;j<8;j++) {
          int row=r*16+xy.y+(j/4)*8;
          if(row<rows)p[row*ld+c*16+xy.x+j%4]=U(data[r*C+c][j]);
        }
      }
    }
  }
  template<typename Op> void row_reduce(thread metal::vec<T,2*R>& values) const thread {
    STEEL_PRAGMA_UNROLL
    for(short r=0;r<R;r++) {
      STEEL_PRAGMA_UNROLL
      for(short c=0;c<C;c++) {
        STEEL_PRAGMA_UNROLL
        for(short row=0;row<2;row++) {
          short j=row*4;
          T v=Op::apply(Op::apply(data[r*C+c][j],data[r*C+c][j+1]),Op::apply(data[r*C+c][j+2],data[r*C+c][j+3]));
          v=Op::apply(v,metal::simd_shuffle_xor(v,ushort(1)));
          v=Op::apply(v,metal::simd_shuffle_xor(v,ushort(8)));
          values[r*2+row]=Op::apply(values[r*2+row],v);
        }
      }
    }
  }
  template<typename Op> void row_bin_op(thread metal::vec<T,2*R>& values) thread {
    STEEL_PRAGMA_UNROLL
    for(short r=0;r<R;r++) {
      STEEL_PRAGMA_UNROLL
      for(short c=0;c<C;c++) {
        STEEL_PRAGMA_UNROLL
        for(short j=0;j<8;j++)data[r*C+c][j]=Op::apply(data[r*C+c][j],values[r*2+j/4]);
      }
    }
  }
};
}}
inline int msv_qsa_pos(const device int* blk,int vi,int sel_len,int tail_start,int ratio) {
  const int b=vi/ratio;
  return vi<sel_len ? blk[b]*ratio+vi-b*ratio : tail_start+vi-sel_len;
}
struct QsaMax { template<typename T> static T apply(T a,T b) { return metal::max(a,b); } };
struct QsaSum { template<typename T> static T apply(T a,T b) { return a+b; } };
struct QsaMul { template<typename T> static T apply(T a,T b) { return a*b; } };
struct QsaExpSub { template<typename T> static T apply(T a,T b) { return metal::exp2(a-b); } };
