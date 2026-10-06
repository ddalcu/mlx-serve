// SPDX-License-Identifier: Apache-2.0
// Ported from oMLX (jundot/omlx) omlx/patches/m5_gather_qmm_nax.py @ d6b2b92.
// Copyright © 2025 Apple Inc. Adapted from MLX steel/gemm/nax.h.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")
namespace mlx { namespace steel {
struct BaseNAXFrag {
  static short2 get_coord() {
    ushort lane=__metal_get_thread_index_in_simdgroup(ushort());
    short qid=lane>>2;
    return short2(((qid&2)|(lane&1))*4, (qid&4)|((lane>>1)&3));
  }
  template <typename CType,typename AType,typename BType,bool ta=false,bool tb=false>
  inline static constexpr void mma(
      thread metal::vec<CType,8>& Cn0,thread metal::vec<CType,8>& Cn1,
      const thread metal::vec<AType,8>& A,metal::bool_constant<ta>,
      const thread metal::vec<BType,8>& Bn0,const thread metal::vec<BType,8>& Bn1,
      metal::bool_constant<tb>) {
    constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(
        16,32,16,ta,tb,true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
    auto a=op.template get_left_input_cooperative_tensor<AType,BType,CType>();
    auto b=op.template get_right_input_cooperative_tensor<AType,BType,CType>();
    auto c=op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,CType>();
    STEEL_PRAGMA_UNROLL
    for(short i=0;i<8;i++){a[i]=A[i];b[i]=Bn0[i];b[8+i]=Bn1[i];c[i]=Cn0[i];c[8+i]=Cn1[i];}
    op.run(a,b,c);
    STEEL_PRAGMA_UNROLL
    for(short i=0;i<8;i++){Cn0[i]=c[i];Cn1[i]=c[8+i];}
  }
};
template<typename T,short R,short C> struct NAXTile {
  using frag_type=metal::vec<T,8>;
  STEEL_CONST short kTileRows=R;
  STEEL_CONST short kTileCols=C;
  frag_type data[R*C];
  thread frag_type& frag_at(short r,short c) thread {return data[r*C+c];}
  void clear() thread {
    STEEL_PRAGMA_UNROLL
    for(short i=0;i<R*C;i++)data[i]=frag_type(0);
  }
  template<typename U,int LD,int STRIDE> void load(const threadgroup U* p) thread {
    short2 xy=BaseNAXFrag::get_coord();
    STEEL_PRAGMA_UNROLL
    for(short r=0;r<R;r++){
      STEEL_PRAGMA_UNROLL
      for(short c=0;c<C;c++){
        STEEL_PRAGMA_UNROLL
        for(short j=0;j<8;j++)data[r*C+c][j]=T(p[(r*16+xy.y+(j/4)*8)*LD+(c*16+xy.x+j%4)*STRIDE]);
      }
    }
  }
  template<typename U> void load(const device U* p,int ld) thread {
    short2 xy=BaseNAXFrag::get_coord();
    STEEL_PRAGMA_UNROLL
    for(short r=0;r<R;r++){
      STEEL_PRAGMA_UNROLL
      for(short c=0;c<C;c++){
        STEEL_PRAGMA_UNROLL
        for(short j=0;j<8;j++)data[r*C+c][j]=T(p[(r*16+xy.y+(j/4)*8)*ld+c*16+xy.x+j%4]);
      }
    }
  }
  template<typename U> void load_safe(const device U* p,int ld,short2 dims) thread {
    short2 xy=BaseNAXFrag::get_coord();
    STEEL_PRAGMA_UNROLL
    for(short r=0;r<R;r++){
      STEEL_PRAGMA_UNROLL
      for(short c=0;c<C;c++){
        STEEL_PRAGMA_UNROLL
        for(short j=0;j<8;j++){
          int row=r*16+xy.y+(j/4)*8,col=c*16+xy.x+j%4;
          data[r*C+c][j]=(row<dims.y && col<dims.x)?T(p[row*ld+col]):T(0);
        }
      }
    }
  }
  template<typename U> void store(device U* p,int ld) const thread {
    short2 xy=BaseNAXFrag::get_coord();
    STEEL_PRAGMA_UNROLL
    for(short r=0;r<R;r++){
      STEEL_PRAGMA_UNROLL
      for(short c=0;c<C;c++){
        STEEL_PRAGMA_UNROLL
        for(short j=0;j<8;j++)p[(r*16+xy.y+(j/4)*8)*ld+c*16+xy.x+j%4]=U(data[r*C+c][j]);
      }
    }
  }
  template<typename U> void store_safe(device U* p,int ld,short2 dims) const thread {
    short2 xy=BaseNAXFrag::get_coord();
    STEEL_PRAGMA_UNROLL
    for(short r=0;r<R;r++){
      STEEL_PRAGMA_UNROLL
      for(short c=0;c<C;c++){
        STEEL_PRAGMA_UNROLL
        for(short j=0;j<8;j++){
          int row=r*16+xy.y+(j/4)*8,col=c*16+xy.x+j%4;
          if(row<dims.y && col<dims.x)p[row*ld+col]=U(data[r*C+c][j]);
        }
      }
    }
  }
};
template<typename C,typename A,typename B,bool ta,bool tb>
void tile_matmad_nax(thread C& d,thread A& a,metal::bool_constant<ta>,thread B& b,metal::bool_constant<tb>){
  static_assert(!ta && tb && C::kTileCols%2==0,"gather tile geometry");
  STEEL_PRAGMA_UNROLL
  for(short m=0;m<C::kTileRows;m++){
    STEEL_PRAGMA_UNROLL
    for(short n=0;n<C::kTileCols;n+=2){
      STEEL_PRAGMA_UNROLL
      for(short k=0;k<A::kTileCols;k++){
        BaseNAXFrag::mma(d.frag_at(m,n),d.frag_at(m,n+1),a.frag_at(m,k),
            metal::bool_constant<false>{},b.frag_at(n,k),b.frag_at(n+1,k),metal::bool_constant<true>{});
      }
    }
  }
}
template<typename F> void dispatch_bool(bool v,F f){if(v)f(metal::true_type{});else f(metal::false_type{});}
}}

using namespace metal;
using namespace mlx::steel;

namespace omlx_gqmm {

STEEL_CONST int kBN = 64;
STEEL_CONST int kWN = 2;
STEEL_CONST short kSM = 32;
STEEL_CONST short kSN = kBN / kWN;
STEEL_CONST short kSK = 32;
STEEL_CONST short kTM = kSM / 16;
STEEL_CONST short kTN = kSN / 16;
STEEL_CONST short kTK = kSK / 16;

// Tile geometry: BM rows in BM / 32 row simdgroups times kWN column
// simdgroups, K steps BK deep. kLT loader threads dequantize the kBN x BK
// weight tile, each kVPT consecutive values of one weight row: every
// thread when they split the tile evenly, else the largest power of two
// below the thread count (96-row tiles: 128 of 192).
template <int BM, int BK>
struct Geo {
  STEEL_CONST int kBM = BM;
  STEEL_CONST int kBK = BK;
  STEEL_CONST int kWM = BM / kSM;
  STEEL_CONST int kThreads = kWM * kWN * 32;
  STEEL_CONST int kLT = (kThreads & (kThreads - 1)) == 0
      ? kThreads
      : (kThreads > 256 ? 256 : (kThreads > 128 ? 128 : 64));
  STEEL_CONST int kVPT = kBN * BK / kLT;
  STEEL_CONST int kTPR = BK / kVPT;
  static_assert(BM % kSM == 0 && BK % kSK == 0, "tile geometry");
  static_assert(kTPR >= 1 && kTPR * kVPT == BK, "loader split");
};

// Affine: w = scale * q + bias computed in fp32 and rounded once to T, as
// mlx's dequantize() does (scale * q is exact in fp32).
template <typename T, int GS, int BITS>
struct AffineQ {
  using WT = T;
  STEEL_CONST int kBits = BITS;
  STEEL_CONST int kGroup = GS;
  const device T* scales;
  const device T* biases;

  struct P {
    float s;
    float b;
  };

  METAL_FUNC void advance(const size_t n) thread {
    scales += n;
    biases += n;
  }
  METAL_FUNC P params(const int g) const thread {
    return P{float(scales[g]), float(biases[g])};
  }
  METAL_FUNC static WT dq(thread const P& p, const uint32_t q) {
    return static_cast<WT>(p.s * float(q) + p.b);
  }
};

// MXFP4: e2m1 nibbles under an e8m0 scale per 32, no biases (the bias pointer
// is a dummy). Decoded as MLX's fp4_e2m1 / fp8_e8m0 do; s * v is exact in fp32.
template <typename T, int GS>
struct Fp4Q {
  using WT = T;
  STEEL_CONST int kBits = 4;
  STEEL_CONST int kGroup = GS;
  const device uint8_t* scales;
  const device T* biases;

  struct P {
    float s;
  };

  METAL_FUNC void advance(const size_t n) thread {
    scales += n;
  }
  METAL_FUNC P params(const int g) const thread {
    const uint b = scales[g];
    return P{as_type<float>(b == 0u ? 0x400000u : (b << 23))};
  }
  METAL_FUNC static WT dq(thread const P& p, const uint32_t q) {
    const float v = float(as_type<half>(ushort((q & 0x7u) << 9))) * 16384.0f;
    return static_cast<WT>(p.s * ((q & 0x8u) ? -v : v));
  }
};

// Weight-tile loader: loader thread lid owns row lid / kTPR of the
// kBN x BK tile and the kVPT values from column (lid % kTPR) * kVPT, in
// kNG chunks that each lie in one quantization group. fetch() reads the
// packed words and group parameters of one K step, store() dequantizes
// them into threadgroup memory (row stride BKP). The *_tail variants
// cover a K tail of k_valid (a multiple of 32) columns and never touch a
// word or group at or past it.
template <typename Q, typename G, bool PAIR = false>
struct TileLoader {
  using WT = typename Q::WT;
  using P = typename Q::P;
  STEEL_CONST int kBits = Q::kBits;
  STEEL_CONST int kVPT = G::kVPT;
  STEEL_CONST int kWords = kVPT * kBits / 32;
  STEEL_CONST int kPer = 32 / kBits;
  STEEL_CONST uint32_t kMask = (1u << kBits) - 1u;
  STEEL_CONST int kGV = kVPT < Q::kGroup ? kVPT : Q::kGroup;
  STEEL_CONST int kNG = kVPT / kGV;
  STEEL_CONST int kWPG = kGV * kBits / 32;
  STEEL_CONST int kBKP = G::kBK + 16 / sizeof(WT);
  static_assert(kWords * 32 == kVPT * kBits, "whole words per thread");
  static_assert(kWPG >= 1 && kNG * kWPG == kWords, "group split");

  const device uint32_t* src;
  Q q;
  const short row;
  const short col;
  uint32_t raw[kWords];
  P p[kNG];

  METAL_FUNC TileLoader(
      const device uint8_t* w_tile,
      const int K,
      thread const Q& q_,
      const uint lid) thread
      : q(q_),
        row(short(lid / G::kTPR)),
        col(short((lid % G::kTPR) * kVPT)) {
    const size_t w_off = size_t(row);
    src = (const device uint32_t*)(w_tile + w_off * (K * kBits / 8) +
                                   col * kBits / 8);
    q.advance(w_off * (K / Q::kGroup));
  }

  METAL_FUNC TileLoader(
      const device uint8_t* gate_tile,
      const device uint8_t* up_tile,
      const int K,
      thread const Q& gate_q,
      thread const Q& up_q,
      const uint lid) thread
      : q(gate_q),
        row(short(lid / G::kTPR)),
        col(short((lid % G::kTPR) * kVPT)) {
    const size_t w_off = PAIR ? size_t(((row >> 5) << 4) + (row & 15)) : size_t(row);
    const bool use_up = PAIR && ((row >> 4) & 1);
    q = use_up ? up_q : gate_q;
    src = (const device uint32_t*)((use_up ? up_tile : gate_tile) + w_off * (K * kBits / 8) +
                                   col * kBits / 8);
    q.advance(w_off * (K / Q::kGroup));
  }

  METAL_FUNC void fetch(const int kb) thread {
    const device uint32_t* ptr = src + kb * (G::kBK * kBits / 32);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      raw[i] = ptr[i];
    }
    STEEL_PRAGMA_UNROLL
    for (short g = 0; g < kNG; g++) {
      p[g] = q.params((kb * G::kBK + col + g * kGV) / Q::kGroup);
    }
  }

  METAL_FUNC void fetch_tail(const int kb, const int k_valid) thread {
    const device uint32_t* ptr = src + kb * (G::kBK * kBits / 32);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      if (col + i * kPer < k_valid) {
        raw[i] = ptr[i];
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short g = 0; g < kNG; g++) {
      if (col + g * kGV < k_valid) {
        p[g] = q.params((kb * G::kBK + col + g * kGV) / Q::kGroup);
      }
    }
  }

  METAL_FUNC void store_words(threadgroup WT* Ws, const int k_valid) const
      thread {
    threadgroup WT* dst = Ws + row * kBKP + col;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      if (col + i * kPer < k_valid) {
        vec<WT, kPer> v;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kPer; j++) {
          v[j] = Q::dq(p[i / kWPG], (raw[i] >> (kBits * j)) & kMask);
        }
        *(threadgroup vec<WT, kPer>*)(dst + i * kPer) = v;
      }
    }
  }

  METAL_FUNC void store(threadgroup WT* Ws) const thread {
    store_words(Ws, G::kBK);
  }

  METAL_FUNC void zero(threadgroup WT* Ws) const thread {
    threadgroup WT* dst = Ws + row * kBKP + col;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kVPT; i++) {
      dst[i] = WT(0);
    }
  }
};

// One 32-deep sub-step of a simdgroup's 32 x 32 block: full row blocks run
// tile_matmad_nax; partial ones skip the 16-row fragments without rows
// (the tensor ops of the others are the ones tile_matmad_nax issues).
template <typename T, typename WT, int BKP, bool FULL>
METAL_FUNC void sub_step(
    thread NAXTile<float, kTM, kTN>& Dtile,
    const device T* xn,
    const threadgroup WT* ws,
    const int K,
    const short sgp_sm) {
  NAXTile<WT, kTN, kTK> Btile;
  if constexpr (FULL) {
    NAXTile<T, kTM, kTK> Atile;

    volatile int compiler_barrier;

    Atile.load(xn, K);
    Btile.template load<WT, BKP, 1>(ws);

    tile_matmad_nax(
        Dtile,
        Atile,
        metal::bool_constant<false>{},
        Btile,
        metal::bool_constant<true>{});

    (void)compiler_barrier;
  } else {
    Btile.template load<WT, BKP, 1>(ws);
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < kTM; mm++) {
      if (mm * 16 < sgp_sm) {
        NAXTile<T, 1, kTK> Arow;
        Arow.load_safe(xn + mm * 16 * K, K, short2(kSK, sgp_sm - mm * 16));
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < kTN; nn += 2) {
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < kTK; kk++) {
            BaseNAXFrag::mma(
                Dtile.frag_at(mm, nn),
                Dtile.frag_at(mm, nn + 1),
                Arow.frag_at(0, kk),
                metal::bool_constant<false>{},
                Btile.frag_at(nn, kk),
                Btile.frag_at(nn + 1, kk),
                metal::bool_constant<true>{});
          }
        }
      }
    }
  }
}

// seg: mlx's segmented sorted gather kernel (affine_gather_qmm_rhs_seg_nax /
// fp_gather_qmm_rhs_seg_nax): one single-expert BM x kBN tile per
// threadgroup, the weight tile of each BK-deep K step dequantized into
// threadgroup memory between two barriers. K tail (K % BK, a multiple of
// 32): only its sub-steps run. N tail: weight rows past N are zero, stores
// are bounded.

template <
    typename T,
    typename Q,
    typename G,
    bool ALIGN_N,
    bool ALIGN_K>
METAL_FUNC void gather_seg(
    const device T* x,
    const device uint8_t* w,
    thread Q& q,
    const uint4 desc,
    const int y_col,
    device T* y,
    const int N,
    const int K,
    threadgroup typename Q::WT* Ws,
    const uint sgid,
    const uint lane) {
  using WT = typename Q::WT;
  constexpr int BKP = G::kBK + 16 / sizeof(WT);
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int rows = int(desc.z);

  const int K_w = K * Q::kBits / 8;
  const int K_g = K / Q::kGroup;
  const int K_it = K / G::kBK;
  const short tgp_bn = ALIGN_N ? short(kBN) : short(min(kBN, N - y_col));
  const int k_remain = K - K_it * G::kBK;
  // First weight row of this output tile.
  const int w_col = y_col;

  const size_t w_row = size_t(expert) * N + w_col;
  q.advance(w_row * K_g);
  TileLoader<Q, G> loader(
      w + w_row * K_w, K, q, sgid * 32 + lane);
  const bool loads = G::kLT == G::kThreads || sgid * 32 + lane < uint(G::kLT);
  const bool row_live = ALIGN_N || loader.row < tgp_bn;

  x += size_t(row_start) * K;
  y += size_t(row_start) * N + w_col;

  const short tm = kSM * short(sgid / kWN);
  const short tn = kSN * short(sgid % kWN);
  const short sgp_sm = short(min(int(kSM), max(0, rows - int(tm))));
  const short sgp_sn =
      ALIGN_N ? kSN : short(min(int(kSN), max(0, N - (y_col + tn))));
  const bool sg_active = sgp_sm > 0;

  NAXTile<float, kTM, kTN> Dtile;
  Dtile.clear();
  const device T* xn = x + tm * K;
  const threadgroup WT* ws = Ws + tn * BKP;

  dispatch_bool(sgp_sm == kSM, [&](auto kAlignedM) {
    for (int k = 0; k < K_it; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (loads) {
        if (row_live) {
          loader.fetch(k);
          loader.store(Ws);
        } else {
          loader.zero(Ws);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < G::kBK; kk1 += kSK) {
        if (sg_active) {
          sub_step<T, WT, BKP, kAlignedM.value>(
              Dtile, xn + kk1, ws + kk1, K, sgp_sm);
        }
      }
      xn += G::kBK;
    }

    if (!ALIGN_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (loads) {
        if (row_live) {
          loader.fetch_tail(K_it, k_remain);
          loader.store_words(Ws, k_remain);
        } else {
          loader.zero(Ws);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < k_remain; kk1 += kSK) {
        if (sg_active) {
          sub_step<T, WT, BKP, kAlignedM.value>(
              Dtile, xn + kk1, ws + kk1, K, sgp_sm);
        }
      }
    }

    if (kAlignedM.value && sgp_sn == kSN) {
      Dtile.store(y + tm * N + tn, N);
    } else if (sg_active) {
      Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
    }
  });
}

// db: the same tiles and arithmetic with double-buffered 64-deep weight
// tiles: the packed words of step k + 1 are fetched before the tensor ops
// of step k and dequantized into the other buffer after them, so each K
// step has a single barrier. Activation fragments are read straight from
// device memory (rows past the tile are clamped to its last row and never
// stored) and 16-row fragments without rows of the tile are skipped.
// Requires K % 64 == 0 and N % 64 == 0.
template <typename T, typename Q, typename G>
METAL_FUNC void gather_db(
    const device T* x,
    const device uint8_t* w,
    thread Q& q,
    const uint4 desc,
    const int y_col,
    device T* y,
    const int N,
    const int K,
    threadgroup typename Q::WT* Ws,
    const uint sgid,
    const uint lane) {
  using WT = typename Q::WT;
  static_assert(G::kBK == 64, "db runs 64-deep K steps");
  constexpr int BKP = G::kBK + 16 / sizeof(WT);
  constexpr int kTile = kBN * BKP;
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int tile_rows = int(desc.z);

  const int K_w = K * Q::kBits / 8;
  const int K_g = K / Q::kGroup;
  const int K_it = K / G::kBK;
  const int w_col = y_col;

  const size_t w_row = size_t(expert) * N + w_col;
  q.advance(w_row * K_g);
  TileLoader<Q, G> loader(
      w + w_row * K_w, K, q, sgid * 32 + lane);
  const bool loads = G::kLT == G::kThreads || sgid * 32 + lane < uint(G::kLT);

  const int m0 = kSM * int(sgid / kWN);
  const int rows = min(int(kSM), tile_rows - m0);
  const device T* xs = x + size_t(row_start + max(0, min(m0, tile_rows - 1))) * K;

  const short2 sc = BaseNAXFrag::get_coord();
  int x_off[kTM][2];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kTM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = min(int(i * 16 + sc.y + h * 8), max(rows, 1) - 1);
      x_off[i][h] = r * K + sc.x;
    }
  }
  const short m_frags = rows > 0 ? short((rows + 15) / 16) : short(0);
  const threadgroup WT* wsg = Ws + (sgid % kWN) * kSN * BKP;

  NAXTile<float, kTM, kTN> D;
  D.clear();

  if (loads) {
    loader.fetch(0);
    loader.store(Ws);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int kb = 0; kb < K_it; kb++) {
    const bool more = kb + 1 < K_it;
    if (more && loads) {
      loader.fetch(kb + 1);
    }
    const threadgroup WT* wb = wsg + (kb & 1) * kTile;
    STEEL_PRAGMA_UNROLL
    for (short kk1 = 0; kk1 < G::kBK; kk1 += kSK) {
      NAXTile<WT, kTN, 2> Btile;
      Btile.template load<WT, BKP, 1>(wb + kk1);
      const int k = kb * G::kBK + kk1;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kTM; i++) {
        if (i < m_frags) {
          NAXTile<T, 1, 2> Atile;
          STEEL_PRAGMA_UNROLL
          for (short h = 0; h < 2; h++) {
            const device T* xp = xs + x_off[i][h] + k;
            const vec<T, 4> a0 = *(const device vec<T, 4>*)(xp);
            const vec<T, 4> a1 = *(const device vec<T, 4>*)(xp + 16);
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; c++) {
              Atile.frag_at(0, 0)[h * 4 + c] = a0[c];
              Atile.frag_at(0, 1)[h * 4 + c] = a1[c];
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < 2; kk++) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < kTN; j += 2) {
              BaseNAXFrag::mma(
                  D.frag_at(i, j),
                  D.frag_at(i, j + 1),
                  Atile.frag_at(0, kk),
                  metal::bool_constant<false>{},
                  Btile.frag_at(j, kk),
                  Btile.frag_at(j + 1, kk),
                  metal::bool_constant<true>{});
            }
          }
        }
      }
    }
    if (more && loads) {
      loader.store(Ws + ((kb + 1) & 1) * kTile);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  device T* yb = y + size_t(row_start + m0) * N + y_col + kSN * (sgid % kWN);
  if (rows >= kSM) D.store(yb, N);
  else if (rows > 0) D.store_safe(yb, N, short2(kSN, short(rows)));
}

// The row tile and output column of this threadgroup. GX == 0: grid
// (columns, tiles) as mlx lays it out. GX > 0: tile t on grid x t % GX and
// (t / GX, column) on y, so all threadgroups of a row tile share one x
// coordinate and a tile's columns run GX threadgroups apart.
template <int GX>
METAL_FUNC bool tile_of(
    const device uint32_t* tiles,
    const uint tile_count,
    const uint3 tid,
    const int N,
    thread uint4& desc,
    thread int& y_col) {
  uint t;
  uint c;
  if constexpr (GX > 0) {
    const uint n_cols = uint((N + kBN - 1) / kBN);
    t = (tid.y / n_cols) * GX + tid.x;
    c = tid.y % n_cols;
  } else {
    t = tid.y;
    c = tid.x;
  }
  if (t >= tile_count) {
    return false;
  }
  desc = *((const device uint4*)tiles + t);
  y_col = int(c) * kBN;
  return true;
}

} // namespace omlx_gqmm

using namespace metal;

// Cuts the sorted rows into (row_start, expert, rows, 0) tiles of at most
// BM rows of one expert, expert-major (the tile order of mlx's segmented
// gather_qmm). One threadgroup: the run bounds of every expert are found in
// parallel over the rows, then a threadgroup scan of the per-expert tile
// counts gives each expert's first tile. At most max_tiles tiles are
// written (a guard for unsorted input, which the contract excludes).
template <int BM>
METAL_FUNC void msv_gqmm_tile_scan(
    const device uint32_t* idx,
    const constant int* params,
    device uint32_t* tiles,
    device uint32_t* tile_count,
    threadgroup uint32_t* run_start,
    threadgroup uint32_t* run_end,
    threadgroup uint32_t* simd_tot,
    const uint lid,
    const uint tg_size,
    const uint sg,
    const uint lane) {
  const int M = params[0];
  const int E = params[1];
  const uint32_t max_tiles = uint32_t(params[2]);
  for (int e = int(lid); e < E; e += int(tg_size)) {
    run_start[e] = 0;
    run_end[e] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Each thread walks 4 consecutive rows per step, with their neighbours
  // (0xffffffff past either end, never a valid expert).
  for (int g0 = 4 * int(lid); g0 < M; g0 += 4 * int(tg_size)) {
    const int cnt = min(4, M - g0);
    uint32_t v[6];
    v[0] = g0 > 0 ? idx[g0 - 1] : 0xffffffffu;
    for (int j = 0; j < 4; j++) {
      v[j + 1] = j < cnt ? idx[g0 + j] : 0xffffffffu;
    }
    v[5] = g0 + 4 < M ? idx[g0 + 4] : 0xffffffffu;
    for (int j = 0; j < cnt; j++) {
      const uint32_t e = v[j + 1];
      if (e < uint32_t(E)) {
        if (v[j] != e) {
          run_start[e] = uint32_t(g0 + j);
        }
        if (v[j + 2] != e) {
          run_end[e] = uint32_t(g0 + j + 1);
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_simd = (tg_size + 31) / 32;
  uint32_t running = 0;
  for (int base = 0; base < E; base += int(tg_size)) {
    const int e = base + int(lid);
    uint32_t start = 0;
    uint32_t cnt = 0;
    if (e < E) {
      start = run_start[e];
      const uint32_t end = run_end[e];
      cnt = end > start ? end - start : 0;
    }
    const uint32_t nt = (cnt + BM - 1) / BM;
    const uint32_t local = simd_prefix_exclusive_sum(nt);
    const uint32_t stot = simd_sum(nt);
    if (lane == 0) {
      simd_tot[sg] = stot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint32_t prefix = 0;
    uint32_t total = 0;
    for (uint s = 0; s < n_simd; s++) {
      const uint32_t v = simd_tot[s];
      prefix += (s < sg) ? v : 0;
      total += v;
    }
    const uint32_t off = running + prefix + local;
    for (uint32_t j = 0; j < nt && off + j < max_tiles; j++) {
      const uint32_t r = start + j * BM;
      *((device uint4*)tiles + off + j) =
          uint4(r, uint32_t(e), min(uint32_t(BM), start + cnt - r), 0);
    }
    running += total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) {
    tile_count[0] = min(running, max_tiles);
  }
}
