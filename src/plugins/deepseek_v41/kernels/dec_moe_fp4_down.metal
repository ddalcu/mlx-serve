// dec_moe_fp4_down: the routed (mxfp4) and shared (mxfp8) down projections of the decode rows, the combine and the hc
// expand into the next stream. A threadgroup is 2 NC adjacent output columns of one row, the R rows of a column group
// adjacent in the grid (threadgroup t is row t % R of columns t / R * 2 NC ..) so the rows' reads of the shared expert
// and of the experts they share come from the cache. Simdgroup j < TOPK reads the row's j-th pick for both column
// halves (dec_fp4_down_rows); simdgroups TOPK and TOPK + 1 the shared expert (twice the bytes a unit), a half each. The
// slot sums are added in that order and rounded to T where `hcExpand` reads y.
const uint t = threadgroup_position_in_grid.x;
const uint r = t % uint(R);
const uint d0 = t / uint(R) * (2 * NC);
const uint sgi = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint c = lane % NC;
threadgroup float part[2 * NC * (TOPK + 1)];
if (sgi >= uint(TOPK)) {
  const uint hf = sgi - TOPK;
  float a = dec_fp4_down_shared<K, T>(sw, sscales, d0 + hf * NC + c, x + size_t(r * (TOPK + 1) + TOPK) * K, lane / NC, 32 / NC);
#pragma unroll
  for (uint o = 16; o >= uint(NC); o >>= 1) a += simd_shuffle_xor(a, o);
  if (lane < uint(NC)) part[(hf * NC + lane) * (TOPK + 1) + TOPK] = a;
} else {
  float acc[2];
  dec_fp4_down_rows<K, 2, T>(w, scales, size_t(inds[r * TOPK + sgi]) * N + d0 + c, NC, x + size_t(r * (TOPK + 1) + sgi) * K, lane / NC, 32 / NC, acc);
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
    for (uint o = 16; o >= uint(NC); o >>= 1) acc[hf] += simd_shuffle_xor(acc[hf], o);
    if (lane < uint(NC)) part[(hf * NC + lane) * (TOPK + 1) + sgi] = acc[hf];
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = sgi * 32 + lane; i < uint(2 * NC * HC); i += 32 * (TOPK + 2)) {
  const uint cc = i / HC;
  const uint h = i % HC;
  const uint d = d0 + cc;
  float s = 0.0f;
  for (int j = 0; j <= TOPK; ++j) s += part[cc * (TOPK + 1) + j];
  const float y = float(T(s));
  float a = 0.0f;
  for (int j = 0; j < HC; ++j) a = metal::fma(comb[(r * HC + j) * HC + h], float(res[(r * HC + j) * N + d]), a);
  out[(r * HC + h) * N + d] = T(metal::fma(post[r * HC + h], y, a));
}
