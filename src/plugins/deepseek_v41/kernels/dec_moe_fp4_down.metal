// dec_moe_fp4_down: the routed (fp4, groups of GS) and shared (mxfp8) down projections of one decode row, the combine
// and the hc expand into the next stream. A threadgroup is NC adjacent output columns of one row r: simdgroup j < TOPK
// is the row's j-th pick, simdgroup TOPK the shared expert; lane l sums column l % NC over units l / NC, + 32 / NC...
// (the columns' rows are adjacent in the bank). The slot sums are added in that order and rounded to T where
// `hcExpand` reads y. DW (`MLX_SERVE_DSV41_FP4_DEC_WORD`): the routed codes decoded a word at a time.
#if DSV41_DOWN_ORDER
// `MLX_SERVE_DSV41_MOE_DOWN_ORDER`: a 1-D grid whose adjacent threadgroups are one column block's rows, so the rows
// read the shared expert's and any common pick's weight rows back to back.
const uint rows_ = threadgroups_per_grid.x / uint(N / NC);
const uint d0 = (threadgroup_position_in_grid.x / rows_) * NC;
#if DSV41_DOWN_ORDER_PLANT
const uint r = threadgroup_position_in_grid.x / uint(N / NC);
#else
const uint r = threadgroup_position_in_grid.x % rows_;
#endif
#else
const uint d0 = threadgroup_position_in_grid.x * NC;
const uint r = threadgroup_position_in_grid.y;
#endif
const uint sgi = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint c = lane % NC;
threadgroup float part[NC * (TOPK + 1)];
float acc;
if (sgi == uint(TOPK)) {
  acc = dec_fp4_down_shared<K, FR, 32 / NC, T>(sw, sscales, d0 + c, x + size_t(r * (TOPK + 1) + TOPK) * K, lane / NC);
} else {
  acc = dec_fp4_down_routed<K, GS, FR, 32 / NC, T, DW>(w, scales, size_t(inds[r * TOPK + sgi]) * N + d0 + c, x + size_t(r * (TOPK + 1) + sgi) * K, lane / NC);
}
#pragma unroll
for (uint o = 16; o >= uint(NC); o >>= 1) acc += simd_shuffle_xor(acc, o);
if (lane < uint(NC)) part[lane * (TOPK + 1) + sgi] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sgi == 0 && lane < uint(NC * HC)) {
  const uint cc = lane / HC;
  const uint h = lane % HC;
  const uint d = d0 + cc;
  float s = 0.0f;
  for (int j = 0; j <= TOPK; ++j) s += part[cc * (TOPK + 1) + j];
  const float y = float(T(s));
  float a = 0.0f;
  for (int j = 0; j < HC; ++j) a = metal::fma(comb[(r * HC + j) * HC + h], float(res[(r * HC + j) * N + d]), a);
  out[(r * HC + h) * N + d] = T(metal::fma(post[r * HC + h], y, a));
}
