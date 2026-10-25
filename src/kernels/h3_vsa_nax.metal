// SPDX-License-Identifier: Apache-2.0
// Block-sparse VSA attention on the tensor units: `attention_nax_bdv_qt`'s loop (nax_attention.metal,
// MLX's steel_attention_nax.h via oMLX) walking each video query tile's selected key tiles instead of a
// contiguous key range. Head-major [H, n_tiles * TILE, 128] tiles; padding rows are zero, and a key
// tile's columns past its valid count are masked. Rows past a query tile's valid count are written
// but never read.
namespace omlx_nax {

// Index arrays bind as device or constant pointers by their size, hence the pointer parameters.
template <typename T, int WM, int TILE, typename SelPtr, typename VbsPtr, typename MetaPtr>
METAL_FUNC void h3_vsa_nax(
    const device T* Q,
    const device T* K,
    const device T* V,
    device T* O,
    SelPtr sel,
    VbsPtr vbs,
    MetaPtr meta,
    float scale,
    uint simd_group_id,
    uint simd_lane_id,
    uint3 tid) {
  constexpr int BQ = 16 * WM;
  constexpr int BK = 32;
  constexpr int BD = 128;
  constexpr short kU = 16;
  constexpr int TD = BD / kU;
  constexpr short TK = BK / kU;
  using AccumType = float;

  const int n_prefix = meta[0];
  const int n_video = meta[1];
  const int k_sel = meta[2];
  const int n_tiles = meta[3];
  const int q_tile = int(tid.x) / (TILE / BQ);
  const int q_sub = int(tid.x) % (TILE / BQ);
  const int head = int(tid.y);
  const int qt = n_prefix + q_tile;
  if (q_sub * BQ >= vbs[qt]) return; // a block of padding rows, uniform per threadgroup

  const int64_t plane = int64_t(head) * n_tiles * TILE;
  const short tm = kU * short(simd_group_id);

  using otile_t = NAXTile<AccumType, 1, TD>;
  otile_t Otile;
  Otile.clear();
  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  constexpr short kRowsPT = otile_t::kRowsPerThread;
  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};
  OMLX_NAX_UNROLL
  for (short i = 0; i < kRowsPT; ++i) max_score[i] = Limits<AccumType>::finite_min;
  const float scale2 = scale * 1.44269504089f;

  NAXTile<T, 1, TD> Qreg;
  Qreg.load(Q + (plane + int64_t(qt) * TILE + q_sub * BQ + tm) * BD, BD);

  const int64_t picks = (int64_t(head) * n_video + q_tile) * k_sel;
  for (int s = 0; s < k_sel; s++) {
    const int kt = sel[picks + s];
    const int kv = vbs[kt];
    for (int b0 = 0; b0 < kv; b0 += BK) {
      const int lim = kv - b0;
      const device T* Kb = K + (plane + int64_t(kt) * TILE + b0) * BD;
      const device T* Vb = V + (plane + int64_t(kt) * TILE + b0) * BD;

      using stile_t = NAXTile<AccumType, 1, TK>;
      stile_t Stile;
      Stile.clear();
      OMLX_NAX_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
#pragma clang loop unroll_count(4)
        for (short id = 0; id < TD; id++) {
          NAXTile<T, 1, 1> Qtile;
          NAXTile<T, 2, 1> Ktile;
          Qtile.frag_at(0, 0) = Qreg.frag_at(0, id);
          Ktile.load(Kb + ik * kU * BD + id * kU, BD);
          stile_t::NAXFrag_t::mma(
              Stile.frag_at(0, ik), Stile.frag_at(0, ik + 1), Qtile.frag_at(0, 0), metal::false_type{},
              Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
        }
      }
      OMLX_NAX_UNROLL
      for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) Stile.elems()[ii] *= scale2;

      if (lim < BK) {
        constexpr auto neg_inf = Limits<AccumType>::finite_min;
        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;
          thread auto& fg = Stile.frag_at(0, ik);
          OMLX_NAX_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            OMLX_NAX_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (col_pos + jj) < lim ? fg[loc] : neg_inf;
            }
          }
        }
      }

      metal::vec<AccumType, kRowsPT> new_max;
      metal::vec<AccumType, kRowsPT> factor;
      OMLX_NAX_UNROLL
      for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
      OMLX_NAX_UNROLL
      for (short ii = 0; ii < stile_t::kFragThrRows; ++ii) {
        const short loc0 = ii * stile_t::kFragThrCols;
        AccumType m = Stile.frag_at(0, 0)[loc0];
        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ++ik) {
          OMLX_NAX_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; ++jj) m = metal::max(m, Stile.frag_at(0, ik)[loc0 + jj]);
        }
        m = metal::max(m, simd_shuffle_xor(m, ushort(1)));
        m = metal::max(m, simd_shuffle_xor(m, ushort(8)));
        new_max[ii] = metal::max(new_max[ii], m);
      }
      Stile.template row_bin_op<ExpSubOp>(new_max);
      OMLX_NAX_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        factor[i] = fast::exp2(max_score[i] - new_max[i]);
        max_score[i] = new_max[i];
        sum_score[i] = sum_score[i] * factor[i];
      }
      Stile.template row_reduce<SumOp>(sum_score);
      Otile.template row_bin_op<MulOp>(factor);
      simdgroup_barrier(mem_flags::mem_none);

      OMLX_NAX_UNROLL
      for (short id = 0; id < TD; id += 2) {
        if (id == 4) threadgroup_barrier(mem_flags::mem_none);
        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          NAXTile<T, 1, 2> Vtile;
          Vtile.load(Vb + ik * kU * BD + id * kU, BD);
          otile_t::NAXFrag_t::mma(
              Otile.frag_at(0, id), Otile.frag_at(0, id + 1), Stile.frag_at(0, ik), metal::false_type{},
              Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  OMLX_NAX_UNROLL
  for (short i = 0; i < kRowsPT; ++i) rcp[i] = 1.f / sum_score[i];
  Otile.template row_bin_op<MulOp>(rcp);
  Otile.store(O + ((int64_t(head) * n_video + q_tile) * TILE + q_sub * BQ + tm) * BD, BD);
}

} // namespace omlx_nax
