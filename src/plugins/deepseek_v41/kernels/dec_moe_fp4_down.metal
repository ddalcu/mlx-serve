// dec_moe_fp4_down: the routed (mxfp4) and shared (mxfp8) down projections of one decode row, the combine and the hc
// expand into the next stream. A threadgroup is NC adjacent output columns of one row r: simdgroup j < TOPK is the row's
// j-th pick, simdgroup TOPK the shared expert; lane l sums column l % NC over units l / NC, + 32 / NC... (the columns'
// rows are adjacent in the bank). The slot sums are added in that order and rounded to T where `hcExpand` reads y.
const uint d0 = threadgroup_position_in_grid.x * NC;
const uint r = threadgroup_position_in_grid.y;
const uint sgi = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint c = lane % NC;
threadgroup float part[NC * (TOPK + 1)];
float acc;
if (sgi == uint(TOPK)) {
  acc = dec_fp4_down_shared<K, T>(sw, sscales, d0 + c, x + size_t(r * (TOPK + 1) + TOPK) * K, lane / NC, 32 / NC);
} else {
  acc = dec_fp4_down_routed<K, T>(w, scales, size_t(inds[r * TOPK + sgi]) * N + d0 + c, x + size_t(r * (TOPK + 1) + sgi) * K, lane / NC, 32 / NC);
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
