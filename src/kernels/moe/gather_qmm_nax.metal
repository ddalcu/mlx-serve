// SPDX-License-Identifier: Apache-2.0
// Ported from oMLX (jundot/omlx) omlx/patches/m5_gather_qmm_nax.py @ d6b2b92.
using Q = metal::conditional_t<MX == 1, omlx_gqmm::Fp4Q<T, GS>, omlx_gqmm::AffineQ<T, GS, BITS>>;
using G = omlx_gqmm::Geo<BM, BK>;
using WT = typename Q::WT;
constexpr int BKP = BK + 16 / sizeof(WT);
threadgroup WT Ws[(SCHED == 1 ? 2 : 1) * omlx_gqmm::kBN * BKP + PAD / sizeof(WT)];
uint4 desc;
int y_col;
if (!omlx_gqmm::tile_of<GX>(tiles, tile_count[0], threadgroup_position_in_grid, params[0], desc, y_col)) return;
Q q{scales, biases};
if constexpr (SCHED == 1) {
  omlx_gqmm::gather_db<T,Q,G>(x,(const device uint8_t*)w,q,desc,y_col,y,params[0],params[1],Ws,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
} else {
  omlx_gqmm::gather_seg<T,Q,G,ALIGN_N,ALIGN_K>(x,(const device uint8_t*)w,q,desc,y_col,y,params[0],params[1],Ws,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
}
