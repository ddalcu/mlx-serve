// dec_moe_fp4_up: routed (mxfp4) and shared (mxfp8) gate and up, clamped SwiGLU times the routing weight. One simdgroup
// per (output column, slot): slots 0 .. TOPK * R - 1 are the (row, pick) pairs, slot TOPK * R the shared expert (every
// row). A column's slots are adjacent in the grid, which spreads the heavy shared tiles over the launch.
const uint lane = thread_index_in_simdgroup;
const uint sg = thread_position_in_grid.x / 32;
const uint u = sg % uint(TOPK * R + 1);
const uint n_col = sg / uint(TOPK * R + 1);
if (n_col >= uint(N)) return;
if (u == uint(TOPK * R)) {
  float acc[R][2];
  dec_up_shared<R, K, N, T>(sw, sscales, n_col, x, lane, acc);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) out[(r * (TOPK + 1) + TOPK) * N + n_col] = dec_up_act<T>(acc[r][0], acc[r][1], lim[0], 1.0f);
  }
  return;
}
const uint r = u / TOPK;
float acc[2];
dec_fp4_up_routed<K, N, T>(w, scales, size_t(inds[u]) * (2 * N) + n_col, x + size_t(r) * K, lane, acc);
if (lane == 0) out[(r * (TOPK + 1) + u % TOPK) * N + n_col] = dec_up_act<T>(acc[0], acc[1], lim[0], wts[u]);
