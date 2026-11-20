auto lane = thread_index_in_simdgroup;
uint n = thread_position_in_grid.y;      // output row within the expert
// grid.z: a (batch row, top-k slot) pair; ONE_ROW (the single-row decode) has one row, so z is the slot.
uint combo = thread_position_in_grid.z;
uint e = ONE_ROW ? combo : combo % uint(TOPK);
uint row = ONE_ROW ? 0u : combo / uint(TOPK);

int K = int(K_size);
int N = int(N_size);
constexpr int VPW = mlxserve_vpw<BITS>();
int K_by_p = K / VPW;
int K_by_w = K * BITS / 32;
int K_by_gs = K / GS;
uint mask = (1u << BITS) - 1u;

uint eid = inds[(size_t)row * (size_t)TOPK + e];
size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_w + (size_t)n * (size_t)K_by_w;
size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
size_t xrow = (size_t)row * (size_t)K;

// Every lane's packed words are issued up front (KP = K_by_p is a
// template constant): −0.7% per forward in idle same-boot pairs vs
// the runtime-count loop. Same per-lane accumulation order.
constexpr int IT = (KP + 31) / 32;
mlxserve_pack_t<BITS> pg[IT];
mlxserve_pack_t<BITS> pu[IT];
for (int i = 0; i < IT; ++i) {{
  int pack = int(lane) + 32 * i;
  pg[i] = (pack < KP) ? mlxserve_qpack<BITS>(wg_q, wbase, pack) : 0u;
  pu[i] = (pack < KP) ? mlxserve_qpack<BITS>(wu_q, wbase, pack) : 0u;
}}
// Two independent 4-way accumulator sets: the gate chain and the up
// chain never wait on each other.
float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
for (int i = 0; i < IT; ++i) {{
  int pack = int(lane) + 32 * i;
  if (pack >= KP) break;
  mlxserve_pack_t<BITS> packed_g = pg[i];
  mlxserve_pack_t<BITS> packed_u = pu[i];
  int k_base = pack * VPW;
  int gi = k_base / GS;
{s}
}}
float acc_g = simd_sum((g0 + g1) + (g2 + g3));
float acc_u = simd_sum((u0 + u1) + (u2 + u3));
if (lane == 0) {{
  // Round where the split path's gathers write T(acc); each expert's scale is the split path's rounded multiply.
  T gt = T(acc_g);
  T ut = T(acc_u);
  if constexpr (GSCALE != 0) gt = T(gt * gsc[eid]);
  if constexpr (USCALE != 0) ut = T(ut * usc[eid]);
  if constexpr (LIMIT > 0) {{
    // clampedSwiGLU's minimum and clip on the rounded values; a NaN compares false and stays NaN, as MLX keeps it.
    const T lim = T(LIMIT);
    const T nlim = T(-LIMIT);
    gt = gt > lim ? lim : gt;
    ut = ut < nlim ? nlim : ut;
    ut = ut > lim ? lim : ut;
  }}
  T sig = sigtab[as_type<ushort>(gt)];
  y[(((size_t)row * (size_t)TOPK) + e) * (size_t)N + (size_t)n] = (gt * sig) * ut;
}}
