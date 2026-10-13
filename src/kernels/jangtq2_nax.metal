// SPDX-License-Identifier: Apache-2.0
// Ported from vMLX (jjang-ai/vmlx) vmlx_engine/jangh/kernels.py _NAX_IMPL @ 5f007e6d: the generic
// sorted kernel, without the expert-tile and Hadamard-32 epilogue arms.

using namespace mlx::steel;
METAL_FUNC float tq_act(float g, float u, float lim) {
  if (lim > 0.0f) { g = metal::min(g, lim); u = metal::clamp(u, -lim, lim); }
  return (g / (1.0f + metal::fast::exp(-g))) * u;
}
// FUSED=false: y = x W^T for one weight.  FUSED=true: y = act(x Wg^T, x Wu^T).
template <typename T, int bits, bool FUSED>
METAL_FUNC void tq_gather_qmm_nax(
    const device T* x, const device uint32_t* wg, const device half* sg, const device uint32_t* wu, const device half* su,
    const device uint32_t* indices, device T* y, const int M, const int N, const int K, const float lim,
    threadgroup T* Wg, threadgroup T* Wu, uint3 tid, uint simd_group_id, uint simd_lane_id) {
  constexpr int BM = 64, BK = 64, BN = 64, WM = 2, WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using loader_w_t = TQBlockLoader<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE, bits>;
  const int K_w = K * bytes_per_pack / pack_factor; const int K_it = K / BK;
  const size_t stride_w = size_t(N) * K_w;
  const int y_row = tid.y * BM;
  const int row_end = M;
  const int y_col = tid.x * BN;
  const short tgp_bm = short(min(BM, row_end - y_row));
  const short tgp_bn = short(min(BN, N - y_col));
  auto wgl = (const device uint8_t*)wg; auto wul = (const device uint8_t*)wu;
  x += size_t(y_row) * K; y += size_t(y_row) * N + y_col;
  wgl += size_t(y_col) * K_w; if (FUSED) wul += size_t(y_col) * K_w;
  constexpr short SM = BM / WM, SN = BN / WN, SK = 32;
  constexpr short TM = SM / 16, TN = SN / 16, TK = SK / 16;
  const short tm = SM * (simd_group_id / WN); const short tn = SN * (simd_group_id % WN);
  const short sgp_sm = short(min(int(SM), max(0, row_end - (y_row + tm))));
  const short sgp_sn = short(min(int(SN), max(0, N - (y_col + tn))));
  uint32_t index; short offset; uint32_t index_next = indices[y_row]; short offset_next = 0; int n = 0;
  while (n < tgp_bm) {
    n++; offset = offset_next; index = index_next; offset_next = tgp_bm;
    for (; n < tgp_bm; n++) { if (indices[y_row + n] != index) { offset_next = n; index_next = indices[y_row + n]; break; } }
    threadgroup_barrier(mem_flags::mem_none);
    NAXTile<float, TM, TN> Gt; Gt.clear();
    NAXTile<float, TM, TN> Ut; if (FUSED) Ut.clear();
    const device T* xn = x + tm * K;
    thread loader_w_t lg(wgl + index * stride_w, sg + size_t(index) * N + y_col, K, Wg, simd_group_id, simd_lane_id);
    thread loader_w_t lu(FUSED ? wul + index * stride_w : wgl, FUSED ? su + size_t(index) * N + y_col : sg, K,
                         FUSED ? Wu : Wg, simd_group_id, simd_lane_id);
    const bool full_n = (tgp_bn == BN);
    // A simdgroup with no rows in this expert's segment skips the MMA but still loads its share of the
    // weight tile and joins every barrier.
    const short m_lo_lim = min(int(sgp_sm), max(0, offset - tm));
    const short m_hi_lim = min(int(sgp_sm), max(0, offset_next - tm));
    const bool sg_active = (m_hi_lim > m_lo_lim) && (sgp_sn > 0);
    dispatch_bool(sgp_sm == SM, [&](auto kAlignedM) {
      for (int k = 0; k < K_it; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (full_n) { lg.load_unsafe(); if (FUSED) lu.load_unsafe(); }
        else { lg.load_safe(short2(BK, tgp_bn)); if (FUSED) lu.load_safe(short2(BK, tgp_bn)); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_active) {
          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            NAXTile<T, TM, TK> Atile; NAXTile<T, TN, TK> Bg;
            volatile int compiler_barrier;
            if constexpr (kAlignedM.value) Atile.load(xn + kk1, K); else Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
            Bg.template load<T, BK_padded, 1>(Wg + tn * BK_padded + kk1);
            tile_matmad_nax(Gt, Atile, metal::bool_constant<false>{}, Bg, metal::bool_constant<true>{});
            if (FUSED) {
              NAXTile<T, TN, TK> Bu;
              Bu.template load<T, BK_padded, 1>(Wu + tn * BK_padded + kk1);
              tile_matmad_nax(Ut, Atile, metal::bool_constant<false>{}, Bu, metal::bool_constant<true>{});
            }
            (void)compiler_barrier;
          }
        }
        xn += BK; lg.next(); if (FUSED) lu.next();
      }
    });
    if (FUSED) {
      for (short i = 0; i < decltype(Gt)::kNumFrags; i++)
        for (short e = 0; e < decltype(Gt)::kElemsPerFrag; e++)
          Gt.val_frags[i][e] = tq_act(Gt.val_frags[i][e], Ut.val_frags[i][e], lim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_active) {
      if (m_lo_lim == 0 && m_hi_lim == SM && sgp_sn == SN) Gt.store(y + tm * N + tn, N);
      else Gt.store_slice(y + tm * N + tn, N, short2(0, m_lo_lim), short2(sgp_sn, m_hi_lim));
    }
  }
}
