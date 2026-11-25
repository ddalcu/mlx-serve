// SPDX-License-Identifier: Apache-2.0
// The staged sorted gather: MLX's staged quantized matmul (fp_qmm_t_nax_staged_impl) over the scan's single-expert
// tiles, with K and the output width as compile-time constants.
namespace omlx_gqmm {

// GQMM_SCAT (PF_DOWN_TOKORD, the msv_gqmm_staged_scat build only): the plain (EPI 0) tile's rows go to their pairs'
// slots instead of their sorted places. Sorted row p is token rmap[p]'s route j, where j is the slot of the tile's
// expert among that token's GQMM_SCAT_K picks ridx[tok * K ..]; the row lands at y row tok * K + j, so the output is in
// pair (token) order and a combine reads it in place. Each value is the same T(D) as D.store writes; only the row
// address differs. A token's picks are distinct experts, so every slot is written once.
#if GQMM_SCAT
#define GQMM_SCAT_ARG , const device uint* ridx
#define GQMM_SCAT_PASS , ridx
template <typename T, short TM, short TN, int NW, int KR, typename RMap>
METAL_FUNC void scat_store(
    thread const NAXTile<float, TM, TN>& D,
    device T* y,
    RMap rmap,
    const device uint* ridx,
    const uint expert,
    const int base,
    const int rows,
    const int col0) {
  const short2 xy = BaseNAXFrag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < TM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = i * 16 + h * 8 + int(xy.y);
      if (r < rows) {
        const uint tok = uint(rmap[base + r]);
        uint j = 0;
        STEEL_PRAGMA_UNROLL
        for (int k = 0; k < KR; k++) {
          if (ridx[tok * uint(KR) + uint(k)] == expert) j = uint(k);
        }
        device T* yr = y + size_t(tok * uint(KR) + j) * NW + col0 + xy.x;
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < TN; c++) {
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < 4; e++) yr[c * 16 + e] = T(D.data[i * TN + c][h * 4 + e]);
        }
      }
    }
  }
}
#else
#define GQMM_SCAT_ARG
#define GQMM_SCAT_PASS
#endif

// BM rows of one expert by BN weight rows over WM x WN simdgroups; each simdgroup holds BM / WM rows by BN / WN weight
// rows. Every thread loads its share of the BN x BK weight tile (kVPT values of one weight row each).
template <int BM, int WM, int WN, int BN = kBN, int BK = 64>
struct GeoS {
  STEEL_CONST int kBM = BM;
  STEEL_CONST int kBN = BN;
  STEEL_CONST int kBK = BK;
  STEEL_CONST int kKS = BK / 16;
  STEEL_CONST int kWM = WM;
  STEEL_CONST int kWN = WN;
  STEEL_CONST int kSM = BM / WM;
  STEEL_CONST int kSN = BN / WN;
  STEEL_CONST int kThreads = WM * WN * 32;
  STEEL_CONST int kLT = kThreads;
  STEEL_CONST int kVPT = BN * BK / kLT;
  STEEL_CONST int kTPR = BK / kVPT;
  static_assert((kThreads & (kThreads - 1)) == 0 && kThreads >= 64 && kThreads <= 512, "64..512 loader threads, a power of two");
  static_assert(BM % WM == 0 && kSM % 16 == 0 && kSN % 32 == 0, "whole 16-row fragments, column pairs");
  static_assert(BK % 16 == 0 && kVPT >= 8 && kTPR * kVPT == BK, "whole k16 blocks, whole packed words a loader thread");
};

// The activation fragments of one KS x 16-deep step: a[s][i][h] holds the 4 values of k16 block s a lane feeds row
// fragment i's half h. Fragments without rows of the tile are not read.
template <typename T, short TM, short KS>
METAL_FUNC void load_a_staged(
    thread vec<T, 4> (&a)[KS][TM][2],
    const device T* x,
    const thread uint (&off)[TM][2],
    const int k,
    const short m_frags) {
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < TM; i++) {
    if (i < m_frags) {
      STEEL_PRAGMA_UNROLL
      for (short h = 0; h < 2; h++) {
        STEEL_PRAGMA_UNROLL
        for (short s = 0; s < KS; s++) {
          a[s][i][h] = *(const device vec<T, 4>*)(x + off[i][h] + k + s * 16);
        }
      }
    }
  }
}

// One tile: the packed words of step k + 1 and its activations are read before the tensor ops of step k; the words
// are dequantized into the other weight buffer after them, so each step has one barrier. MAP reads x's rows through
// rmap; EPI != 0 is the paired gate/up kernel's epilogue (store_act), one 16-column pair at a time. The k16 blocks of
// every output run in order, as the seg and db schedules run them. `RMap`: MLX binds an array under 8 elements in
// `constant`, so the row map's address space is the caller's.
template <typename T, typename Q, typename G, int NW, int KD, int EPI, bool MAP, int ESC, typename RMap>
METAL_FUNC void gather_staged(
    const device T* x,
    RMap rmap,
    const device uint8_t* w,
    const device uint8_t* up_w,
    thread Q& q,
    thread Q& up_q,
    const uint4 desc,
    const int y_col,
    device T* y,
    threadgroup typename Q::WT* Ws,
    const uint sgid,
    const uint lane,
    const device T* sigtab,
    const int up_off,
    const device T* gsc,
    const device T* usc,
    const device float* rw,
    const float lim GQMM_SCAT_ARG) {
  using WT = typename Q::WT;
  constexpr bool kPair = EPI != 0;
  constexpr int BK = G::kBK;
  constexpr int BKP = BK + 16 / sizeof(WT);
  constexpr int kTile = G::kBN * BKP;
  constexpr int KS = G::kKS;
  constexpr int SM = G::kSM;
  constexpr int SN = G::kSN;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr int K_it = KD / BK;
  constexpr int K_w = KD * Q::kBits / 8;
  constexpr int K_g = KD / Q::kGroup;
  constexpr int half_n = NW / 2;
  static_assert(KD % BK == 0 && NW % G::kBN == 0, "whole BK-deep steps over whole BN-column tiles");
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int tile_rows = int(desc.z);
  const int w_col = kPair ? y_col / 2 : y_col;

  const size_t w_row = size_t(expert) * (kPair ? half_n + up_off : NW) + w_col;
  q.advance(w_row * K_g);
  up_q.advance((w_row + up_off) * K_g);
  TileLoader<Q, G, kPair> loader(
      w + w_row * K_w, up_w + (w_row + up_off) * K_w, KD, q, up_q, sgid * 32 + lane);

  const int m0 = SM * int(sgid / G::kWN);
  const int tn = SN * int(sgid % G::kWN);
  const int rows = min(SM, tile_rows - m0);
  const short m_frags = rows > 0 ? short((rows + 15) / 16) : short(0);
  const short2 sc = BaseNAXFrag::get_coord();
  // Rows past the tile read its last row and are never stored.
  uint a_off[TM][2];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < TM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = min(m0 + i * 16 + h * 8 + int(sc.y), tile_rows - 1);
      a_off[i][h] = (MAP ? rmap[row_start + r] : uint(row_start + r)) * uint(KD) + uint(sc.x);
    }
  }

  NAXTile<float, TM, TN> D;
  D.clear();
  vec<T, 4> a_cur[KS][TM][2];
  loader.fetch(0);
  loader.store(Ws);
  load_a_staged<T, TM, KS>(a_cur, x, a_off, 0, m_frags);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const threadgroup WT* wsg = Ws + (tn + sc.y) * BKP + sc.x;

  for (int kb = 0; kb < K_it; kb++) {
    const bool more = kb + 1 < K_it;
    if (more) {
      loader.fetch(kb + 1);
    }
    vec<T, 4> a_next[KS][TM][2];
    if (more) {
      load_a_staged<T, TM, KS>(a_next, x, a_off, (kb + 1) * BK, m_frags);
    }
    if (m_frags > 0) {
      const threadgroup WT* ws = wsg + (kb & 1) * kTile;
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < BK / 16; s++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TN; j += 2) {
          const threadgroup WT* p0 = ws + (j * 16) * BKP + s * 16;
          const threadgroup WT* p1 = p0 + 16 * BKP;
          vec<WT, 8> b0;
          vec<WT, 8> b1;
          STEEL_PRAGMA_UNROLL
          for (short h = 0; h < 2; h++) {
            const vec<WT, 4> u0 = *(const threadgroup vec<WT, 4>*)(p0 + h * 8 * BKP);
            const vec<WT, 4> u1 = *(const threadgroup vec<WT, 4>*)(p1 + h * 8 * BKP);
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; c++) {
              b0[h * 4 + c] = u0[c];
              b1[h * 4 + c] = u1[c];
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < TM; i++) {
            if (i < m_frags) {
              vec<T, 8> a;
              STEEL_PRAGMA_UNROLL
              for (short c = 0; c < 4; c++) {
                a[c] = a_cur[s][i][0][c];
                a[4 + c] = a_cur[s][i][1][c];
              }
              BaseNAXFrag::mma(
                  D.frag_at(i, j),
                  D.frag_at(i, j + 1),
                  a,
                  metal::bool_constant<false>{},
                  b0,
                  b1,
                  metal::bool_constant<true>{});
            }
          }
        }
      }
    }
    if (more) {
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < KS; s++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TM; i++) {
          a_cur[s][i][0] = a_next[s][i][0];
          a_cur[s][i][1] = a_next[s][i][1];
        }
      }
      loader.store(Ws + ((kb + 1) & 1) * kTile);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (rows <= 0) {
    return;
  }
  if constexpr (kPair) {
    device T* yb = y + size_t(row_start + m0) * half_n + w_col + tn / 2;
    STEEL_PRAGMA_UNROLL
    for (short p = 0; p < TN / 2; p++) {
      store_act<T, EPI, ESC>(D, yb + 16 * p, half_n, rows, sigtab, gsc, usc, expert, rw + row_start + m0, lim, p);
    }
  } else {
#if GQMM_SCAT
    scat_store<T, TM, TN, NW, GQMM_SCAT_K>(D, y, rmap, ridx, expert, row_start + m0, rows, y_col + tn);
#else
    device T* yb = y + size_t(row_start + m0) * NW + y_col + tn;
    if (rows >= SM) {
      D.store(yb, NW);
    } else {
      D.store_safe(yb, NW, short2(SN, short(rows)));
    }
#endif
  }
}

// AS == 1: the same tile with its activations staged through threadgroup memory: each thread holds its share of the
// next step's BM x 64 activations (16-byte loads) and weight words in registers while the tensor ops of this step read
// the shared buffers, then two barriers fence their refill. Every simdgroup of a row band reads its fragments from
// the one copy, so 32-row simdgroups fit their registers.
template <typename T, typename Q, typename G, int NW, int KD, int EPI, bool MAP, int ESC, typename RMap>
METAL_FUNC void gather_staged_tg(
    const device T* x,
    RMap rmap,
    const device uint8_t* w,
    const device uint8_t* up_w,
    thread Q& q,
    thread Q& up_q,
    const uint4 desc,
    const int y_col,
    device T* y,
    threadgroup typename Q::WT* Ws,
    threadgroup T* As,
    const uint sgid,
    const uint lane,
    const device T* sigtab,
    const int up_off,
    const device T* gsc,
    const device T* usc,
    const device float* rw,
    const float lim GQMM_SCAT_ARG) {
  using WT = typename Q::WT;
  constexpr bool kPair = EPI != 0;
  constexpr int BK = G::kBK;
  constexpr int BKP = BK + 16 / sizeof(WT);
  constexpr int AKP = BK + 16 / sizeof(T);
  constexpr int SM = G::kSM;
  constexpr int SN = G::kSN;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr int K_it = KD / BK;
  constexpr int K_w = KD * Q::kBits / 8;
  constexpr int K_g = KD / Q::kGroup;
  constexpr int half_n = NW / 2;
  // Each thread's 8-value chunks of the BM x 64 activation tile.
  constexpr int kAL = G::kBM * (BK / 8) / G::kThreads;
  static_assert(KD % BK == 0 && NW % G::kBN == 0, "whole BK-deep steps over whole BN-column tiles");
  static_assert(kAL >= 1 && kAL * G::kThreads == G::kBM * (BK / 8), "whole activation chunks per thread");
  static_assert(BK == 64, "the activation chunks are 8 a 64-deep row");
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int tile_rows = int(desc.z);
  const int w_col = kPair ? y_col / 2 : y_col;

  const size_t w_row = size_t(expert) * (kPair ? half_n + up_off : NW) + w_col;
  q.advance(w_row * K_g);
  up_q.advance((w_row + up_off) * K_g);
  const uint lid = sgid * 32 + lane;
  TileLoader<Q, G, kPair> loader(
      w + w_row * K_w, up_w + (w_row + up_off) * K_w, KD, q, up_q, lid);

  // Chunk c of this thread: tile row (lid + c * threads) / 8, columns ((lid + c * threads) % 8) * 8; rows past the
  // tile read its last row and are never stored.
  uint a_src[kAL];
  STEEL_PRAGMA_UNROLL
  for (short c = 0; c < kAL; c++) {
    const int idx = int(lid) + c * G::kThreads;
    const int r = min(idx / 8, tile_rows - 1);
    a_src[c] = (MAP ? rmap[row_start + r] : uint(row_start + r)) * uint(KD) + uint((idx % 8) * 8);
  }
  vec<T, 8> a_raw[kAL];

  const int m0 = SM * int(sgid / G::kWN);
  const int tn = SN * int(sgid % G::kWN);
  const int rows = min(SM, tile_rows - m0);
  const short m_frags = rows > 0 ? short((rows + 15) / 16) : short(0);
  const short2 sc = BaseNAXFrag::get_coord();

  NAXTile<float, TM, TN> D;
  D.clear();
  loader.fetch(0);
  STEEL_PRAGMA_UNROLL
  for (short c = 0; c < kAL; c++) {
    a_raw[c] = *(const device vec<T, 8>*)(x + a_src[c]);
  }
  loader.store(Ws);
  STEEL_PRAGMA_UNROLL
  for (short c = 0; c < kAL; c++) {
    const int idx = int(lid) + c * G::kThreads;
    *(threadgroup vec<T, 8>*)(As + (idx / 8) * AKP + (idx % 8) * 8) = a_raw[c];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const threadgroup WT* wsg = Ws + (tn + sc.y) * BKP + sc.x;
  const threadgroup T* asg = As + (m0 + sc.y) * AKP + sc.x;

  for (int kb = 0; kb < K_it; kb++) {
    const bool more = kb + 1 < K_it;
    if (more) {
      loader.fetch(kb + 1);
      STEEL_PRAGMA_UNROLL
      for (short c = 0; c < kAL; c++) {
        a_raw[c] = *(const device vec<T, 8>*)(x + a_src[c] + (kb + 1) * BK);
      }
    }
    if (m_frags > 0) {
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < BK / 16; s++) {
        vec<T, 8> a[TM];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TM; i++) {
          const threadgroup T* pa = asg + (i * 16) * AKP + s * 16;
          const vec<T, 4> v0 = *(const threadgroup vec<T, 4>*)(pa);
          const vec<T, 4> v1 = *(const threadgroup vec<T, 4>*)(pa + 8 * AKP);
          STEEL_PRAGMA_UNROLL
          for (short c = 0; c < 4; c++) {
            a[i][c] = v0[c];
            a[i][4 + c] = v1[c];
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TN; j += 2) {
          const threadgroup WT* p0 = wsg + (j * 16) * BKP + s * 16;
          const threadgroup WT* p1 = p0 + 16 * BKP;
          vec<WT, 8> b0;
          vec<WT, 8> b1;
          STEEL_PRAGMA_UNROLL
          for (short h = 0; h < 2; h++) {
            const vec<WT, 4> u0 = *(const threadgroup vec<WT, 4>*)(p0 + h * 8 * BKP);
            const vec<WT, 4> u1 = *(const threadgroup vec<WT, 4>*)(p1 + h * 8 * BKP);
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; c++) {
              b0[h * 4 + c] = u0[c];
              b1[h * 4 + c] = u1[c];
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < TM; i++) {
            if (i < m_frags) {
              BaseNAXFrag::mma(
                  D.frag_at(i, j),
                  D.frag_at(i, j + 1),
                  a[i],
                  metal::bool_constant<false>{},
                  b0,
                  b1,
                  metal::bool_constant<true>{});
            }
          }
        }
      }
    }
    if (more) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader.store(Ws);
      STEEL_PRAGMA_UNROLL
      for (short c = 0; c < kAL; c++) {
        const int idx = int(lid) + c * G::kThreads;
        *(threadgroup vec<T, 8>*)(As + (idx / 8) * AKP + (idx % 8) * 8) = a_raw[c];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  if (rows <= 0) {
    return;
  }
  if constexpr (kPair) {
    device T* yb = y + size_t(row_start + m0) * half_n + w_col + tn / 2;
    STEEL_PRAGMA_UNROLL
    for (short p = 0; p < TN / 2; p++) {
      store_act<T, EPI, ESC>(D, yb + 16 * p, half_n, rows, sigtab, gsc, usc, expert, rw + row_start + m0, lim, p);
    }
  } else {
#if GQMM_SCAT
    scat_store<T, TM, TN, NW, GQMM_SCAT_K>(D, y, rmap, ridx, expert, row_start + m0, rows, y_col + tn);
#else
    device T* yb = y + size_t(row_start + m0) * NW + y_col + tn;
    if (rows >= SM) {
      D.store(yb, NW);
    } else {
      D.store_safe(yb, NW, short2(SN, short(rows)));
    }
#endif
  }
}
} // namespace omlx_gqmm
