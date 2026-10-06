// SPDX-License-Identifier: Apache-2.0
// Ported from oMLX (jundot/omlx) omlx/utils/nax_attention.py @ 5dcfe243: MLX's
// steel_attention_nax.h (Copyright (c) 2024-25 Apple Inc., MIT) with a separate
// value head dim, key-range passes and the head-dim split. The bool-mask arm is
// dropped and a sliding WINDOW bound added (block skip + in-tile mask).
namespace omlx_nax {

// Scalar parameters of one attention call (MLX's AttnParams without the
// strides, which the kernel reads from the inputs' own stride vectors).
struct AttnParams {
  int B;
  int H;
  int D;
  int qL;
  int kL;
  int gqa_factor;
  float scale;
  int NQ;
  int NK;
  int NQ_aligned;
  int NK_aligned;
  int qL_rem;
  int kL_rem;
  int qL_off;
  // Key blocks [kb_begin, kb_end) of this dispatch (key-range passes).
  int kb_begin;
  int kb_end;
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

// MLX's attention_nax with a value head dim BDV <= BD. The plumbing
// differs: strides come from the inputs, the output is written as
// [B, qL, H, BDV] rows (MLX's SDPA output layout), function constants are
// template arguments, and the mask column stride is honoured. Beyond that:
//
// * Key-range passes. One dispatch covers key blocks [kb_begin, kb_end).
//   Unless FIRST, the online-softmax state of every query row (fp32 O
//   accumulator, running max, running sum) is loaded from Sin; unless LAST it
//   is stored to Sout instead of normalising and writing the output. Every
//   row thus runs exactly the operations of one full dispatch, in the same
//   order (bit-identical output). Within a dispatch the threadgroups in
//   flight only stream one key slice of their KV head, which stays on chip;
//   over one long dispatch they drift apart and, once the KV head no longer
//   fits in the caches, each re-streams it from DRAM.
// * The query tile stays in registers instead of being re-read from
//   memory for every key block, and each row max is reduced over all of the
//   row's score fragments before the cross-lane shuffles (max is exact, so
//   the grouping does not matter).
// * WN = 2 splits the head dims over simdgroup pairs, as MLX's
//   attention_nax_dsplit does for 256-wide heads: each simdgroup of a pair
//   computes Q @ K.T over half of the query/key dims and P @ V for half of
//   the value dims, the pair adds its partial scores through threadgroup
//   memory, and both run the softmax on the full score tile. Half the O
//   accumulator and query registers per thread; the scores become the sum
//   of two fp32 partial dot products (a summation-order change).
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int BDV,
    int WM,
    int WN,
    bool align_Q,
    bool align_K,
    bool do_causal,
    int WINDOW,
    bool has_sinks,
    bool FIRST,
    bool LAST,
    typename AccumType,
    typename StridePtr,
    typename SinkPtr,
    typename StatePtr>
METAL_FUNC void attention_nax_bdv(
    const device T* Q,
    const device T* K,
    const device T* V,
    device T* O,
    const device AttnParams* params,
    StridePtr q_str,
    StridePtr k_str,
    StridePtr v_str,
    SinkPtr sinks,
    StatePtr Sin,
    device float* Sout,
    threadgroup AccumType* xchg,
    uint simd_group_id,
    uint simd_lane_id,
    uint3 tid) {
  ulong3 tidl{tid.x, tid.y, tid.z};

  const int64_t Q_strides[3] = {q_str[0], q_str[1], q_str[2]};
  const int64_t K_strides[3] = {k_str[0], k_str[1], k_str[2]};
  const int64_t V_strides[3] = {v_str[0], v_str[1], v_str[2]};
  const int64_t O_strides[3] = {
      int64_t(params->qL) * params->H * BDV, BDV, int64_t(params->H) * BDV};

  Q += tidl.z * Q_strides[0] + // Batch
      tidl.y * Q_strides[1] + // Head
      tidl.x * BQ * Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * K_strides[0] + // Batch
      kv_head_idx * K_strides[1]; // Head

  V += tidl.z * V_strides[0] + // Batch
      kv_head_idx * V_strides[1]; // Head

  O += tidl.z * O_strides[0] + // Batch
      tidl.y * O_strides[1] + // Head
      tidl.x * BQ * O_strides[2]; // Sequence

  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short kU = 16;

  // WM groups of 16 query rows; each group's WN simdgroups split the head
  // dims (WN = 2: MLX's attention_nax_dsplit scheme; see the note above).
  static_assert(BQ == WM * kU, "One 16-row fragment per row group");
  static_assert(WN == 1 || WN == 2, "Head dims split over 1 or 2 simdgroups");

  // Q seq frags per warp
  constexpr int TQ = 1;
  // HeadDim frags of this simdgroup
  constexpr int TD = BD / kU / WN;
  // Value head dim frags of this simdgroup
  constexpr int TDV = BDV / kU / WN;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TD * kU * WN == BD, "The head dim must split evenly");
  static_assert(TDV % 2 == 0, "P@V accumulates output fragments in pairs");
  using otile_t = NAXTile<AccumType, TQ, TDV>;
  otile_t Otile;

  Otile.clear();

  // Prepare mma tile offsets: rows of this row group, columns of this
  // simdgroup's share of the head dims.
  const short row_group = simd_group_id / WN;
  const short d_part = simd_group_id % WN;
  const short tm = kU * TQ * row_group;
  Q += tm * int(Q_strides[2]) + d_part * (BD / WN);
  K += d_part * (BD / WN);
  V += d_part * (BDV / WN);
  O += d_part * (BDV / WN);

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Init row reduction variables
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  // Online-softmax state rows of this simdgroup in the pass buffers:
  // [B, H, NQ * BQ] rows of BDV + 2 floats (O accumulator, max, sum).
  constexpr int kSW = BDV + 2;
  const int64_t srow0 = (int64_t(tid.z) * params->H + tid.y) *
          (int64_t(params->NQ) * BQ) +
      int64_t(tid.x) * BQ + tm;

  if constexpr (FIRST) {
    // Init to -Inf
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = Limits<AccumType>::finite_min;
    }

    if (has_sinks) {
      OMLX_NAX_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
        sum_score[i] = 1;
      }
    }
  } else {
    // Resume the previous pass (rows past a query tail are never output).
    Otile.load(Sin + srow0 * kSW + d_part * (BDV / WN), kSW);
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      const int64_t r = srow0 + sm + i * otile_t::kFragRowsJump;
      max_score[i] = Sin[r * kSW + BDV];
      sum_score[i] = Sin[r * kSW + BDV + 1];
    }
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }
  // Sliding window: query position p sees keys (p - WINDOW, p], so a tile
  // starts at the first block its first row can see. Per tile, not per row
  // group: the key loop holds threadgroup barriers.
  const int kb_min_window = WINDOW > 0 ? max(0, int(tid.x) * BQ + params->qL_off - WINDOW + 1) / BK : 0;

  const bool is_last_bq = int(tid.x) == (params->NQ_aligned);
  const bool is_last_q = is_last_bq;

  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  // This dispatch's key blocks.
  const int kb_lo = max(params->kb_begin, kb_min_window);
  const int kb_hi = min(kb_lim, params->kb_end);
  K += int64_t(kb_lo) * BK * K_strides[2];
  V += int64_t(kb_lo) * BK * V_strides[2];

  // Query fragments, loaded once and kept in registers.
  NAXTile<T, TQ, TD> Qreg;
  if (!align_Q && is_last_q) {
    Qreg.load_rows(Q, int(Q_strides[2]), lim_rows_q);
  } else {
    Qreg.load(Q, int(Q_strides[2]));
  }

  // Loop over KV seq length
  for (int kb = kb_lo; kb < kb_hi; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));

    // Do S = Q @ K.T
    using stile_t = NAXTile<AccumType, TQ, TK>;
    stile_t Stile;

    Stile.clear();

    OMLX_NAX_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      OMLX_NAX_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
        auto qk_step = [&](short id) {
          NAXTile<T, 1, 1> Qtile;
          NAXTile<T, 2, 1> Ktile;

          const int K_load_off = ik * kU * int(K_strides[2]) + id * kU;

          Qtile.frag_at(0, 0) = Qreg.frag_at(iq, id);

          if (!align_K && is_last_k) {
            Ktile.load_rows(
                K + K_load_off, int(K_strides[2]), lim_rows_k - ik * kU);
          } else {
            Ktile.load(K + K_load_off, int(K_strides[2]));
          }

          stile_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qtile.frag_at(0, 0),
              metal::false_type{},
              Ktile.frag_at(0, 0),
              Ktile.frag_at(1, 0),
              metal::true_type{});
        };
        if constexpr (WN == 2) {
          // The head-dim split kernel keeps static fragment indices.
          OMLX_NAX_UNROLL
          for (short id = 0; id < TD; id++) {
            qk_step(id);
          }
        } else {
#pragma clang loop unroll_count(4)
          for (short id = 0; id < TD; id++) {
            qk_step(id);
          }
        }

        if constexpr (WN == 2) {
          // Add the peer's partial scores (its half of the head dims).
          constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;
          threadgroup AccumType* slot =
              xchg + (row_group * WN + d_part) * (2 * kEPF * 32);
          const threadgroup AccumType* peer =
              xchg + (row_group * WN + 1 - d_part) * (2 * kEPF * 32);
          thread auto& s0 = Stile.frag_at(iq, ik);
          thread auto& s1 = Stile.frag_at(iq, ik + 1);
          const short base = short(simd_lane_id) * (2 * kEPF);
          OMLX_NAX_UNROLL
          for (short i = 0; i < kEPF; i++) {
            slot[base + i] = s0[i];
            slot[base + kEPF + i] = s1[i];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          OMLX_NAX_UNROLL
          for (short i = 0; i < kEPF; i++) {
            s0[i] += peer[base + i];
            s1[i] += peer[base + kEPF + i];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    }

    // Scale S
    OMLX_NAX_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      OMLX_NAX_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;

          thread auto& fg = Stile.frag_at(iq, ik);

          OMLX_NAX_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            OMLX_NAX_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      OMLX_NAX_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);

          OMLX_NAX_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            OMLX_NAX_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r =
                  base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // Mask keys that fell out of the sliding window
    if (WINDOW > 0) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      OMLX_NAX_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);

          OMLX_NAX_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            OMLX_NAX_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r =
                  base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (c <= r - WINDOW) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // Do softmax

    // Temp variables
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max: all of a row's fragments first, then across lanes (the
    // grouping is free, max is exact).
    OMLX_NAX_UNROLL
    for (short iq = 0; iq < TQ; ++iq) {
      OMLX_NAX_UNROLL
      for (short ii = 0; ii < stile_t::kFragThrRows; ++ii) {
        const short loc0 = ii * stile_t::kFragThrCols;
        AccumType m = Stile.frag_at(iq, 0)[loc0];
        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ++ik) {
          OMLX_NAX_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; ++jj) {
            m = metal::max(m, Stile.frag_at(iq, ik)[loc0 + jj]);
          }
        }
        m = metal::max(m, simd_shuffle_xor(m, ushort(1)));
        m = metal::max(m, simd_shuffle_xor(m, ushort(8)));
        const short r = iq * stile_t::kFragThrRows + ii;
        new_max[r] = metal::max(new_max[r], m);
      }
    }

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    // Row Sum
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Stile.template row_reduce<SumOp>(sum_score);

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // Do O = P @ V
    OMLX_NAX_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      OMLX_NAX_UNROLL
      for (short id = 0; id < TDV; id += 2) {
        if constexpr (BDV == 128 && WN == 1) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }

        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          NAXTile<T, 1, 2> Vtile;

          const int V_load_off = ik * kU * int(V_strides[2]) + id * kU;

          if (!align_K && is_last_k) {
            Vtile.load_rows(
                V + V_load_off, int(V_strides[2]), lim_rows_k - ik * kU);
          } else {
            Vtile.load(V + V_load_off, int(V_strides[2]));
          }

          otile_t::NAXFrag_t::mma(
              Otile.frag_at(iq, id),
              Otile.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    // Prepare for next iteration
    K += BK * int(K_strides[2]);
    V += BK * int(V_strides[2]);
  }

  threadgroup_barrier(mem_flags::mem_none);

  if constexpr (!LAST) {
    // Hand the row state to the next pass.
    Otile.store(Sout + srow0 * kSW + d_part * (BDV / WN), kSW);
    if (sn == 0 && d_part == 0) {
      OMLX_NAX_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        const int64_t r = srow0 + sm + i * otile_t::kFragRowsJump;
        Sout[r * kSW + BDV] = max_score[i];
        Sout[r * kSW + BDV + 1] = sum_score[i];
      }
    }
    return;
  }

  // Normalize output

  metal::vec<AccumType, kRowsPT> rcp;
  OMLX_NAX_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

  Otile.template row_bin_op<MulOp>(rcp);

  // Store results
  O += tm * int(O_strides[2]);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;

    Otile.store_rows(O, int(O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(O_strides[2]));
  }
}

} // namespace omlx_nax
