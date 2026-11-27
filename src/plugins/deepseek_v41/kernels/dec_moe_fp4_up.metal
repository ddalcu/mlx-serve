// dec_moe_fp4_up: routed (fp4, groups of GS) and shared (mxfp8) gate and up, clamped SwiGLU times the routing weight.
// ModelOpt's per-expert scales where their flags are set: GSCALE / USCALE on the gate / up sum before the clamp, DSCALE
// (the down scale) on the routing weight. One simdgroup per (output column, slot): slots 0 .. TOPK * R - 1 are the
// (row, pick) pairs, slot TOPK * R the shared expert (every row). A column's slots are adjacent in the grid, which
// spreads the heavy shared tiles over the launch. US (`MLX_SERVE_DSV41_FP4_UP_SPLIT`): the shared expert takes two
// simdgroups, a column's first two (one threadgroup: TOPK * R + 2 is even), the gate's and the up's row each, which
// meet in threadgroup memory; the routed slots follow. DW (`MLX_SERVE_DSV41_FP4_DEC_WORD`): the routed codes decoded a
// word at a time.
const uint lane = thread_index_in_simdgroup;
const uint sg = thread_position_in_grid.x / 32;
constexpr uint SH = US != 0 ? 2 : 1;
const uint u = sg % uint(TOPK * R + SH);
const uint n_col = sg / uint(TOPK * R + SH);
threadgroup float shu[US != 0 ? R : 1];
if (n_col >= uint(N)) return;
if constexpr (US != 0) {
  if (u < 2) {
    float a[R];
    dec_fp4_up_shared_row<R, K, T>(sw, sscales, n_col + u * uint(N), x, lane, a);
    if (u == 1 && lane == 0) {
#pragma unroll
      for (int r = 0; r < R; ++r) shu[r] = a[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (u == 0 && lane == 0) {
#pragma unroll
      for (int r = 0; r < R; ++r) out[(r * (TOPK + 1) + TOPK) * N + n_col] = dec_up_act<T>(a[r], shu[r], lim[0], 1.0f);
    }
    return;
  }
} else if (u == uint(TOPK * R)) {
  float acc[R][2];
  dec_up_shared<R, K, N, T>(sw, sscales, n_col, x, lane, acc);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) out[(r * (TOPK + 1) + TOPK) * N + n_col] = dec_up_act<T>(acc[r][0], acc[r][1], lim[0], 1.0f);
  }
  return;
}
const uint slot = u - (SH - 1) * 2;
const uint r = slot / TOPK;
const uint e = inds[slot];
float acc[2];
dec_fp4_up_routed<K, N, GS, FR, T, DW>(w, scales, size_t(e) * (2 * N) + n_col, x + size_t(r) * K, lane, acc);
if (lane == 0) {
  dec_fp4_expert_scales<GSCALE, USCALE, T>(acc, gsc, usc, e);
  const float wt = DSCALE != 0 ? wts[slot] * float(dsc[e]) : wts[slot];
  out[(r * (TOPK + 1) + slot % TOPK) * N + n_col] = dec_up_act<T>(acc[0], acc[1], lim[0], wt);
}
