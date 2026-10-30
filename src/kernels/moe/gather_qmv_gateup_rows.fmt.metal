auto lane = thread_index_in_simdgroup;
uint n = thread_position_in_grid.y;
uint combo = thread_position_in_grid.z;
uint e = combo % uint(TOPK);
uint row = combo / uint(TOPK);

int K = int(K_size);
int N = int(N_size);
constexpr int VPW = (BITS == 3) ? 8 : 32 / BITS;
int K_by_p = K / VPW;
int K_by_w = K * BITS / 32;
int K_by_gs = K / GS;
uint mask = (1u << BITS) - 1u;

uint eid = inds[(size_t)row * (size_t)TOPK + e];
size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_w + (size_t)n * (size_t)K_by_w;
size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
size_t xrow = (size_t)row * (size_t)K;

constexpr int IT = (KP + 31) / 32;
uint32_t pg[IT];
uint32_t pu[IT];
for (int i = 0; i < IT; ++i) {{
  int pack = int(lane) + 32 * i;
  pg[i] = (pack < KP) ? mlxserve_qpack<BITS>(wg_q, wbase, pack) : 0u;
  pu[i] = (pack < KP) ? mlxserve_qpack<BITS>(wu_q, wbase, pack) : 0u;
}}
float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
for (int i = 0; i < IT; ++i) {{
  int pack = int(lane) + 32 * i;
  if (pack >= KP) break;
  uint32_t packed_g = pg[i];
  uint32_t packed_u = pu[i];
  int k_base = pack * VPW;
  int gi = k_base / GS;
{s}
}}
float acc_g = simd_sum((g0 + g1) + (g2 + g3)){s};
float acc_u = simd_sum((u0 + u1) + (u2 + u3)){s};
if (lane == 0) {{
  T gt = {s};
  T ut = T(acc_u);
  T sig = sigtab[as_type<ushort>(gt)];
  y[(((size_t)row * (size_t)TOPK) + e) * (size_t)N + (size_t)n] = (gt * sig) * ut;
}}