// dec_moe_fp4_down_rows: dec_moe_fp4_down over all R rows in one threadgroup, so a weight row two rows read streams
// once. A threadgroup is NC adjacent output columns of every row. Its work items are the distinct routed experts of the
// R x TOPK picks (in first-pick order) and the shared expert (last); simdgroup j takes items j, j + TOPK + 1... and
// adds each unit into every row that reads it. Each (row, pick) sum lands in its own slot, and a row's slots are added
// in pick order, shared last, as the one-row launch adds them. A row's picks are distinct (top-k ids).
constexpr int NS = R * TOPK;
constexpr uint NSG = uint(TOPK + 1);
const uint d0 = threadgroup_position_in_grid.x * NC;
const uint sgi = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = thread_index_in_threadgroup;
const uint c = lane % NC;
threadgroup float part[NC * R * (TOPK + 1)];
threadgroup uint lead_e[NS];
threadgroup uchar lead_j[NS * R];
threadgroup bool is_lead[NS];
threadgroup uint n_lead;
// Pick s leads when no earlier pick holds its expert.
if (tid < uint(NS)) {
  const uint e = inds[tid];
  bool lead = true;
  for (uint s = 0; s < tid; ++s) lead = lead && inds[s] != e;
  is_lead[tid] = lead;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < uint(NS) && is_lead[tid]) {
  uint at = 0;
  for (uint s = 0; s < tid; ++s) at += is_lead[s] ? 1u : 0u;
  const uint e = inds[tid];
  lead_e[at] = e;
  for (int r = 0; r < R; ++r) {
    uchar j = 0xFF;
    for (int p = 0; p < TOPK; ++p)
      if (inds[r * TOPK + p] == e) j = uchar(p);
    lead_j[at * R + r] = j;
  }
}
if (tid == 0) {
  uint n = 0;
  for (int s = 0; s < NS; ++s) n += is_lead[s] ? 1u : 0u;
  n_lead = n;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint nl = n_lead;
const uint items = nl + 1;
for (uint it = sgi; it < items; it += NSG) {
  float acc[R];
  if (it == nl) {
    dec_fp4_down_shared_rows<K, FR, R, TOPK, T>(sw, sscales, d0 + c, x, lane / NC, 32 / NC, acc);
  } else {
    dec_fp4_down_routed_rows<K, GS, FR, R, TOPK, T, DW>(w, scales, size_t(lead_e[it]) * N + d0 + c, x, lead_j + it * R, lane / NC, 32 / NC, acc);
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float a = acc[r];
#pragma unroll
    for (uint o = 16; o >= uint(NC); o >>= 1) a += simd_shuffle_xor(a, o);
    const uint j = it == nl ? uint(TOPK) : uint(lead_j[it * R + r]);
    if (lane < uint(NC) && j != 0xFFu) part[(lane * R + r) * (TOPK + 1) + j] = a;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < uint(NC * HC * R)) {
  const uint r = tid / uint(NC * HC);
  const uint cc = (tid % uint(NC * HC)) / HC;
  const uint h = tid % HC;
  const uint d = d0 + cc;
  float s = 0.0f;
  for (int j = 0; j <= TOPK; ++j) s += part[(cc * R + r) * (TOPK + 1) + j];
  const float y = float(T(s));
  float a = 0.0f;
  for (int j = 0; j < HC; ++j) a = metal::fma(comb[(r * HC + j) * HC + h], float(res[(r * HC + j) * N + d]), a);
  out[(r * HC + h) * N + d] = T(metal::fma(post[r * HC + h], y, a));
}
