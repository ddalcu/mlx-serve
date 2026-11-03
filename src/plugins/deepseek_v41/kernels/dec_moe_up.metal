// dec_moe_up: routed and shared gate and up, clamped SwiGLU times the routing weight, each distinct expert read once.
// One simdgroup per (output column, slot): slots 0 .. TOPK * R - 1 are the (row, k) pairs, slot TOPK * R is the shared
// expert (every row). A column's slots are adjacent in the grid, which spreads the heavy shared tiles over the launch.
const uint lane = thread_index_in_simdgroup;
const uint sg = thread_position_in_grid.x / 32;
// The threadgroup's simdgroups decode through one table, built before any of them leaves.
threadgroup vec<float, BITS == 2 ? 4 : 2> lut[BITS == 2 ? 256 : 64];
dec_up_table<BITS>(lut, thread_index_in_threadgroup, threads_per_threadgroup.x);
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint u = sg % uint(TOPK * R + 1);
const uint n_col = sg / uint(TOPK * R + 1);
if (u == uint(TOPK * R)) {
  float acc[R][2];
  dec_up_shared<R, K, N, T>(sw, sscales, n_col, x, lane, acc);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) out[(r * (TOPK + 1) + TOPK) * N + n_col] = dec_up_act<T>(acc[r][0], acc[r][1], lim[0], 1.0f);
  }
  return;
}
const uint e = inds[u];
ulong pairs = 1ul << u;
if (R > 1) {
  // The simdgroup of an expert's first pair serves every row that picked it; the others have nothing to do.
  const uint ea = lane < uint(TOPK * R) ? uint(inds[lane]) : 0xFFFFFFFFu;
  const uint eb = lane + 32 < uint(TOPK * R) ? uint(inds[lane + 32]) : 0xFFFFFFFFu;
  const ulong same = ulong(static_cast<simd_vote::vote_t>(simd_ballot(ea == e))) | (ulong(static_cast<simd_vote::vote_t>(simd_ballot(eb == e))) << 32);
  if ((same & ((1ul << u) - 1ul)) != 0ul) return;
  pairs = same;
}
while (pairs != 0ul) {
  const uint p = uint(ctz(pairs));
  pairs &= pairs - 1ul;
  if (R > 1 && pairs != 0ul) {  // a second row picked this expert: one decode serves both
    const uint q = uint(ctz(pairs));
    pairs &= pairs - 1ul;
    const uint xrow[2] = {p / TOPK, q / TOPK};
    float acc[2][2];
    dec_up_routed<BITS, K, N, 2, T>(w, scales, biases, e * (2 * N) + n_col, x, xrow, lane, acc, lut);
    if (lane == 0) {
      out[(xrow[0] * (TOPK + 1) + p % TOPK) * N + n_col] = dec_up_act<T>(acc[0][0], acc[0][1], lim[0], wts[p]);
      out[(xrow[1] * (TOPK + 1) + q % TOPK) * N + n_col] = dec_up_act<T>(acc[1][0], acc[1][1], lim[0], wts[q]);
    }
  } else {
    const uint xrow[1] = {p / TOPK};
    float acc[1][2];
    dec_up_routed<BITS, K, N, 1, T>(w, scales, biases, e * (2 * N) + n_col, x, xrow, lane, acc, lut);
    if (lane == 0) out[(xrow[0] * (TOPK + 1) + p % TOPK) * N + n_col] = dec_up_act<T>(acc[0][0], acc[0][1], lim[0], wts[p]);
  }
}
