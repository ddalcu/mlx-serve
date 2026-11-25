// SPDX-License-Identifier: Apache-2.0
using Q = metal::conditional_t<MX == 1, omlx_gqmm::Fp4Q<T, GS>, omlx_gqmm::AffineQ<T, GS, BITS>>;
using G = omlx_gqmm::GeoS<BM, WM, WN, BN, BK>;
using WT = typename Q::WT;
threadgroup WT Ws[(AS == 1 ? 1 : 2) * BN * (BK + 16 / sizeof(WT))];
threadgroup T As[AS == 1 ? BM * (BK + 16 / sizeof(T)) : 1];
uint4 desc;
int y_col;
if (!omlx_gqmm::tile_of<GX, BN>(tiles, tile_count[0], threadgroup_position_in_grid, NW, desc, y_col)) return;
Q q{scales, biases};
Q uq{up_scales, up_biases};
if constexpr (AS == 1) {
  omlx_gqmm::gather_staged_tg<T, Q, G, NW, KD, EPI, MAP, ESC>(x, rmap, (const device uint8_t*)w,
      (const device uint8_t*)up_w, q, uq, desc, y_col, y, Ws, As, simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup, sigtab, params[2], gsc, usc, rw, as_type<float>(params[3]) GQMM_SCAT_PASS);
} else {
  (void)As;
  omlx_gqmm::gather_staged<T, Q, G, NW, KD, EPI, MAP, ESC>(x, rmap, (const device uint8_t*)w,
      (const device uint8_t*)up_w, q, uq, desc, y_col, y, Ws, simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup, sigtab, params[2], gsc, usc, rw, as_type<float>(params[3]) GQMM_SCAT_PASS);
}
