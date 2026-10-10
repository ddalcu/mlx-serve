// SPDX-License-Identifier: Apache-2.0
// Ported from vMLX (jjang-ai/vmlx) vmlx_engine/jangh/kernels.py _STEEL_IMPL @ 5f007e6d.

template <typename T, int bits>
METAL_FUNC void tq_gather_qmm_steel(
    const device T* x, const device uint32_t* wg, const device half* sg, const device uint32_t* indices, device T* y,
    const int M, const int N, const int K, threadgroup T* Xs, threadgroup T* Ws,
    uint3 tid, uint simd_group_id, uint simd_lane_id) {
  // MLX affine_gather_qmm_rhs (non-NAX, transpose=true) with the TQ tile loader. Tiles: 16x32x32, 1x2 simdgroups.
  constexpr int BM = 16, BN = 32, BK = 32, WM = 1, WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = TQBlockLoader<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE, bits>;
  const int K_w = K * bytes_per_pack / pack_factor; const int K_it = K / BK;
  const size_t stride_w = size_t(N) * K_w;
  const int y_row = tid.y * BM; const int y_col = tid.x * BN;
  const short tgp_bm = short(min(BM, M - y_row));
  const short tgp_bn = short(min(BN, N - y_col));
  auto wl = (const device uint8_t*)wg;
  x += size_t(y_row) * K; y += size_t(y_row) * N + y_col; wl += size_t(y_col) * K_w;
  uint32_t index; short offset; uint32_t index_next = indices[y_row]; short offset_next = 0; int n = 0;
  while (n < tgp_bm) {
    n++; offset = offset_next; index = index_next; offset_next = tgp_bm;
    for (; n < tgp_bm; n++) { if (indices[y_row + n] != index) { offset_next = n; index_next = indices[y_row + n]; break; } }
    threadgroup_barrier(mem_flags::mem_none);
    thread mma_t mma_op(simd_group_id, simd_lane_id);
    thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
    thread loader_w_t loader_w(wl + index * stride_w, sg + size_t(index) * N + y_col, K, Ws, simd_group_id, simd_lane_id);
    if (tgp_bm == BM && tgp_bn == BN) gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    else if (tgp_bn == BN) gemm_loop_unaligned<false, true, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, (short)BK);
    else if (tgp_bm == BM) gemm_loop_unaligned<true, false, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, (short)BK);
    else gemm_loop_unaligned<false, false, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, (short)BK);
    if (offset_next - offset == BM && tgp_bn == BN) mma_op.store_result(y, N);
    else mma_op.store_result_slice(y, N, short2(0, offset), short2(tgp_bn, offset_next));
  }
}
