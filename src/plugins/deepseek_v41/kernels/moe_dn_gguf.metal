// `dsv41_moe_dn` over a GGUF expert bank `w` [experts, N, ROWB] (32-weight units), a z per decode row: threadgroup (y, l)
// owns RB output rows of row l and simdgroup s computes slot s of them (TOPK = the shared expert, affine or mxfp8),
// each slot's down against its own activation row (`moe_gu_gguf`'s layout) and rounded as the gather writes it; then a
// thread per (output row, stream) sums the slots in order and writes the hc expand.
const uint lane = thread_index_in_simdgroup;
const uint s = simdgroup_index_in_threadgroup;
const uint d0 = threadgroup_position_in_grid.y * RB;
const uint l = threadgroup_position_in_grid.z;
constexpr uint NU = K / UNIT;
threadgroup float part[(TOPK + 1) * RB];
if (s < TOPK) {
  const uint p = l * TOPK + s;
  const device uint8_t* we = w + size_t(inds[p]) * N * ROWB;
  float acc[RB];
  for (int r = 0; r < RB; ++r) acc[r] = 0.0f;
  for (uint i = lane; i < RB * NU; i += 32) {
    const uint r = i / NU, u = i % NU;
    GgDot<T, 1, K> o = {x + size_t(p) * K + u * UNIT, {0}};
    DEQ_AT(we, d0 + r, u / UPB, u % UPB, o);
    acc[r] += o.acc[0].x + o.acc[0].y + o.acc[0].z + o.acc[0].w;
  }
  for (int r = 0; r < RB; ++r) {
    const float v = metal::simd_sum(acc[r]);
    if (lane == 0) part[s * RB + r] = float(T(v));
  }
} else if (s == TOPK) {
  float a[RB];
  size_t rows[RB];
  for (int r = 0; r < RB; ++r) rows[r] = d0 + r;
  dsv41_dots<SBITS, SGS, K, RB, SMX>(sw, sscales, sbiases, rows, x + size_t(threadgroups_per_grid.z * TOPK + l) * K, lane, a);
  if (lane == 0) for (int r = 0; r < RB; ++r) part[TOPK * RB + r] = float(T(a[r]));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint t = thread_index_in_threadgroup;
if (t < RB * HC) {
  const uint r = t / HC, i = t % HC, d = d0 + r;
  float tot = 0.0f;
  for (int e = 0; e < TOPK; ++e) tot += part[e * RB + r];
  const float y = float(T(tot + part[TOPK * RB + r]));
  float acc = 0.0f;
  for (int j = 0; j < HC; ++j) acc = metal::fma(comb[(l * HC + j) * HC + i], float(res[(l * HC + j) * N + d]), acc);
  out[(l * HC + i) * N + d] = T(metal::fma(post[l * HC + i], y, acc));
}
