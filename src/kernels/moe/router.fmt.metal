uint row = thread_position_in_grid.y;
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;

threadgroup float rk[NE];
{s}
threadgroup uint sel[NK];
{s}
size_t rbase = (size_t)row * (size_t)NE;
for (uint e = tid; e < NE; e += TG) {{
{s}
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
{s}
// Selection runs on simdgroup 0 only: K rounds of "max, then mask".
// simd_min over the candidate indices makes the tie-break the LOWEST
// expert id, so the pick is a pure function of the keys.
if (tid < 32) {{
{s}
{s}
  // The lane's strided slice of the keys lives in registers for the K rounds: a round is
  // a register max, two simd reductions and a register mask, no threadgroup traffic.
  constexpr uint EPL = (NE + 31) / 32;
  float mine[EPL];
  for (uint i = 0; i < EPL; ++i) {{
    uint e = lane + 32 * i;
    mine[i] = (e < NE) ? rk[e] : -INFINITY;
  }}
  for (uint r = 0; r < NK; ++r) {{
    float best = -INFINITY;
    uint bslot = 0xFFFFFFFFu;
    for (uint i = 0; i < EPL; ++i) {{
      if (mine[i] > best) {{ best = mine[i]; bslot = i; }}
    }}
    float gmax = simd_max(best);
    uint cand = (best == gmax) ? lane + 32 * bslot : 0xFFFFFFFFu;
    uint gidx = metal::min(simd_min(cand), uint(NE - 1));
    if (gidx % 32 == lane) mine[gidx / 32] = -INFINITY;
    if (lane == 0) sel[r] = gidx;
  }}
  simdgroup_barrier(mem_flags::mem_threadgroup);

  float w = 0.0f;
  uint myidx = 0;
  if (lane < NK) {{
    myidx = sel[lane];
{s}
  }}
  // MLX's `row_reduce_small` owns an 8-wide row in ONE thread and
  // folds it in ASCENDING order in the reduce's accumulator type —
  // which for a bf16 input is bf16, rounding after every add. A simd
  // butterfly over the same values is a different number.
  float tot = 0.0f;
{s}
{s}
  if (lane < NK) {{
    inds[(size_t)row * (size_t)NK + lane] = myidx;
    scores[(size_t)row * (size_t)NK + lane] = TOUT(w);
  }}
}}
{s}