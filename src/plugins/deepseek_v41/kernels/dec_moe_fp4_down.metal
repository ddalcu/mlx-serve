// dec_moe_fp4_down: the routed (mxfp4) and shared (mxfp8) down projections of one decode row, the combine and the hc
// expand into the next stream. A threadgroup is one (output column d, row r): simdgroup j < TOPK is the row's j-th pick,
// simdgroup TOPK the shared expert; the slot sums are added in that order and rounded to T where `hcExpand` reads y.
const uint d = threadgroup_position_in_grid.x;
const uint r = threadgroup_position_in_grid.y;
const uint sgi = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
threadgroup float part[TOPK + 1];
float acc;
if (sgi == uint(TOPK)) {
  acc = dec_fp4_down_shared<K, T>(sw, sscales, d, x + size_t(r * (TOPK + 1) + TOPK) * K, lane);
} else {
  acc = dec_fp4_down_routed<K, T>(w, scales, size_t(inds[r * TOPK + sgi]) * N + d, x + size_t(r * (TOPK + 1) + sgi) * K, lane);
}
acc = simd_sum(acc);
if (lane == 0) part[sgi] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sgi == 0 && lane < uint(HC)) {
  float s = 0.0f;
  for (int j = 0; j <= TOPK; ++j) s += part[j];
  const float y = float(T(s));
  const uint h = lane;
  float a = 0.0f;
  for (int j = 0; j < HC; ++j) a = metal::fma(comb[(r * HC + j) * HC + h], float(res[(r * HC + j) * N + d]), a);
  out[(r * HC + h) * N + d] = T(metal::fma(post[r * HC + h], y, a));
}
