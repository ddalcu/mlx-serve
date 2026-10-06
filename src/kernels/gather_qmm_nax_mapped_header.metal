// SPDX-License-Identifier: Apache-2.0
// Ported from oMLX (jundot/omlx) omlx/patches/m5_gather_qmm_nax.py @ d6b2b92.
namespace omlx_gqmm {
template <typename T, short R>
METAL_FUNC void load_a_map(
    thread NAXTile<T, R, kTK>& A,
    const device T* xk,
    const thread uint (&a_off)[kTM][2],
    const short i0) {
  STEEL_PRAGMA_UNROLL
  for (short r = 0; r < R; r++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const device T* xp = xk + a_off[i0 + r][h];
      STEEL_PRAGMA_UNROLL
      for (short kk = 0; kk < kTK; kk++) {
        const vec<T, 4> v = *(const device vec<T, 4>*)(xp + kk * 16);
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; c++) {
          A.frag_at(r, kk)[h * 4 + c] = v[c];
        }
      }
    }
  }
}

template <typename T, typename WT, int BKP, bool FULL>
METAL_FUNC void sub_step_map(
    thread NAXTile<float, kTM, kTN>& Dtile,
    const device T* xk,
    const thread uint (&a_off)[kTM][2],
    const threadgroup WT* ws,
    const short sgp_sm) {
  NAXTile<WT, kTN, kTK> Btile;
  if constexpr (FULL) {
    NAXTile<T, kTM, kTK> Atile;

    volatile int compiler_barrier;

    load_a_map<T, kTM>(Atile, xk, a_off, 0);
    Btile.template load<WT, BKP, 1>(ws);

    tile_matmad_nax(
        Dtile,
        Atile,
        metal::bool_constant<false>{},
        Btile,
        metal::bool_constant<true>{});

    (void)compiler_barrier;
  } else {
    const short2 sc = BaseNAXFrag::get_coord();
    Btile.template load<WT, BKP, 1>(ws);
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < kTM; mm++) {
      if (mm * 16 < sgp_sm) {
        NAXTile<T, 1, kTK> Arow;
        load_a_map<T, 1>(Arow, xk, a_off, mm);
        STEEL_PRAGMA_UNROLL
        for (short h = 0; h < 2; h++) {
          if (mm * 16 + h * 8 + sc.y >= sgp_sm) {
            STEEL_PRAGMA_UNROLL
            for (short kk = 0; kk < kTK; kk++) {
              STEEL_PRAGMA_UNROLL
              for (short c = 0; c < 4; c++) {
                Arow.frag_at(0, kk)[h * 4 + c] = T(0);
              }
            }
          }
        }
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

METAL_FUNC void map_rows(
    thread uint (&a_off)[kTM][2],
    const device uint32_t* rmap,
    const int row_start,
    const int tile_rows,
    const int m0,
    const int K) {
  const short2 sc = BaseNAXFrag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kTM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = min(m0 + i * 16 + h * 8 + int(sc.y), tile_rows - 1);
      a_off[i][h] = rmap[row_start + r] * uint(K) + uint(sc.x);
    }
  }
}

template <typename T, int EPI, typename DTile>
METAL_FUNC void store_act(thread DTile& D, device T* y, const int ld,
                          const int rows, const device T* sigtab);

template <
    typename T,
    typename Q,
    typename G,
    bool ALIGN_N,
    bool ALIGN_K,
    int EPI = 0,
    bool MAP = false>
METAL_FUNC void gather_mapped_seg(
    const device T* x,
    const device uint32_t* rmap,
    const device uint8_t* w,
    const device uint8_t* up_w,
    thread Q& q,
    thread Q& up_q,
    const uint4 desc,
    const int y_col,
    device T* y,
    const int N,
    const int K,
    threadgroup typename Q::WT* Ws,
    const uint sgid,
    const uint lane,
    const device T* sigtab) {
  using WT = typename Q::WT;
  constexpr bool kPair = EPI != 0;
  static_assert(!kPair || ALIGN_N, "paired gate/up tiles are full");
  constexpr int BKP = G::kBK + 16 / sizeof(WT);
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int rows = int(desc.z);

  const int K_w = K * Q::kBits / 8;
  const int K_g = K / Q::kGroup;
  const int K_it = K / G::kBK;
  const short tgp_bn = ALIGN_N ? short(kBN) : short(min(kBN, N - y_col));
  const int k_remain = K - K_it * G::kBK;
  const int half_n = N / 2;
  const int w_col = kPair ? y_col / 2 : y_col;
  const int ldy = kPair ? half_n : N;

  const size_t w_row = size_t(expert) * (kPair ? half_n : N) + w_col;
  q.advance(w_row * K_g);
  up_q.advance(w_row * K_g);
  TileLoader<Q, G, kPair> loader(
      w + w_row * K_w, up_w + w_row * K_w, K, q, up_q, sgid * 32 + lane);
  const bool loads = G::kLT == G::kThreads || sgid * 32 + lane < uint(G::kLT);
  const bool row_live = ALIGN_N || loader.row < tgp_bn;

  if constexpr (!MAP) {
    x += size_t(row_start) * K;
  }
  y += size_t(row_start) * ldy + w_col;

  const short tm = kSM * short(sgid / kWN);
  const short tn = kSN * short(sgid % kWN);
  const short sgp_sm = short(min(int(kSM), max(0, rows - int(tm))));
  const short sgp_sn =
      ALIGN_N ? kSN : short(min(int(kSN), max(0, N - (y_col + tn))));
  const bool sg_active = sgp_sm > 0;
  uint a_off[kTM][2];
  if constexpr (MAP) {
    map_rows(a_off, rmap, row_start, rows, int(tm), K);
  }

  NAXTile<float, kTM, kTN> Dtile;
  Dtile.clear();
  const device T* xn = MAP ? x : x + tm * K;
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
          if constexpr (MAP) {
            sub_step_map<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, a_off, ws + kk1, sgp_sm);
          } else {
            sub_step<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, ws + kk1, K, sgp_sm);
          }
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
          if constexpr (MAP) {
            sub_step_map<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, a_off, ws + kk1, sgp_sm);
          } else {
            sub_step<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, ws + kk1, K, sgp_sm);
          }
        }
      }
    }

    if constexpr (kPair) {
      if (sg_active) {
        store_act<T, EPI>(
            Dtile, y + tm * ldy + tn / 2, ldy, int(sgp_sm), sigtab);
      }
    } else if (kAlignedM.value && sgp_sn == kSN) {
      Dtile.store(y + tm * N + tn, N);
    } else if (sg_active) {
      Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
    }
  });
}

template <typename T, typename Q, typename G, int EPI = 0, bool MAP = false>
METAL_FUNC void gather_mapped_db(
    const device T* x,
    const device uint32_t* rmap,
    const device uint8_t* w,
    const device uint8_t* up_w,
    thread Q& q,
    thread Q& up_q,
    const uint4 desc,
    const int y_col,
    device T* y,
    const int N,
    const int K,
    threadgroup typename Q::WT* Ws,
    const uint sgid,
    const uint lane,
    const device T* sigtab) {
  using WT = typename Q::WT;
  static_assert(G::kBK == 64, "db runs 64-deep K steps");
  constexpr bool kPair = EPI != 0;
  constexpr int BKP = G::kBK + 16 / sizeof(WT);
  constexpr int kTile = kBN * BKP;
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int tile_rows = int(desc.z);

  const int K_w = K * Q::kBits / 8;
  const int K_g = K / Q::kGroup;
  const int K_it = K / G::kBK;
  const int half_n = N / 2;
  const int w_col = kPair ? y_col / 2 : y_col;

  const size_t w_row = size_t(expert) * (kPair ? half_n : N) + w_col;
  q.advance(w_row * K_g);
  up_q.advance(w_row * K_g);
  TileLoader<Q, G, kPair> loader(
      w + w_row * K_w, up_w + w_row * K_w, K, q, up_q, sgid * 32 + lane);
  const bool loads = G::kLT == G::kThreads || sgid * 32 + lane < uint(G::kLT);

  const int m0 = kSM * int(sgid / kWN);
  const int rows = min(int(kSM), tile_rows - m0);
  const device T* xs = MAP
      ? x
      : x + size_t(row_start + max(0, min(m0, tile_rows - 1))) * K;

  const short2 sc = BaseNAXFrag::get_coord();
  metal::conditional_t<MAP, uint, int> x_off[kTM][2];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kTM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = min(int(i * 16 + sc.y + h * 8), max(rows, 1) - 1);
      if constexpr (MAP) {
        x_off[i][h] = rmap[row_start + min(m0 + r, tile_rows - 1)] * uint(K) +
            uint(sc.x);
      } else {
        x_off[i][h] = r * K + sc.x;
      }
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

  if constexpr (kPair) {
    if (rows > 0) {
      store_act<T, EPI>(
          D,
          y + size_t(row_start + m0) * half_n + w_col + (kSN / 2) * (sgid % kWN),
          half_n,
          rows,
          sigtab);
    }
  } else {
    device T* yb = y + size_t(row_start + m0) * N + y_col + kSN * (sgid % kWN);
    if (rows >= kSM) {
      D.store(yb, N);
    } else if (rows > 0) {
      D.store_safe(yb, N, short2(kSN, short(rows)));
    }
  }
}


// EPI 1: silu(g) * u. EPI = 1 + L: the gate clamped above at L and u to [-L, L] first
// (a `swiglu_limit`), on the rounded values like the split path's ops.
template <typename T, int EPI, typename DTile>
METAL_FUNC void store_act(
    thread DTile& D, device T* y, const int ld, const int rows,
    const device T* sigtab) {
  const short2 sc = BaseNAXFrag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < DTile::kTileRows; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = i * 16 +
          h * 8 + sc.y;
      if (r < rows) {
        vec<T, 4> v;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; j++) {
          const short e = h * 4 + j;
          T g = T(D.frag_at(i, 0)[e]);
          T u = T(D.frag_at(i, 1)[e]);
          if constexpr (EPI > 1) {
            g = metal::min(g, T(EPI - 1));
            u = metal::max(metal::min(u, T(EPI - 1)), T(1 - EPI));
          }
          const T a = T(g * sigtab[as_type<ushort>(g)]);
          v[j] = T(a * u);
        }
        *(device vec<T, 4>*)(y + size_t(r) * ld + sc.x) = v;
      }
    }
  }
}
} // namespace omlx_gqmm
