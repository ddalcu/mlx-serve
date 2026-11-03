auto lane = thread_index_in_simdgroup;
uint slot = simdgroup_index_in_threadgroup;
uint tile = threadgroup_position_in_grid.x;
uint row = threadgroup_position_in_grid.y;

int K = int(K_size);
int N = int(N_size);
constexpr int VPW = (BITS == 3) ? 8 : 32 / BITS;
int K_by_w = K * BITS / 32;
int K_by_gs = K / GS;
uint mask = (1u << BITS) - 1u;

uint eid = inds[(size_t)row * (size_t)TOPK + slot];
size_t xoff = ((size_t)row * (size_t)TOPK + (size_t)slot) * (size_t)K;
threadgroup T slot_vals[uint(TOPK) * uint(ROWS)];
uint r = lane / uint(LPR);
uint sub = lane % uint(LPR);
uint n = tile * uint(ROWS) + r;
size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_w + (size_t)n * (size_t)K_by_w;
size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
constexpr int IT = (KP + LPR - 1) / LPR;
uint32_t pw[IT];
for (int i = 0; i < IT; ++i) {{
  int pack = int(sub) + LPR * i;
  pw[i] = (pack < KP) ? mlxserve_qpack<BITS>(w_q, wbase, pack) : 0u;
}}
for (int i = 0; i < IT; ++i) {{
  int pack = int(sub) + LPR * i;
  if (pack >= KP) break;
  uint32_t packed = pw[i];
  int k_base = pack * VPW;
  int gi = k_base / GS;
{s}
}}
float acc = (a0 + a1) + (a2 + a3);
for (uint d = uint(LPR) / 2u; d >= 1u; d >>= 1u) acc += simd_shuffle_xor(acc, ushort(d));
acc = acc{s};
if (sub == 0u) {{
  slot_vals[slot * uint(ROWS) + r] = T(acc);
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (slot == 0 && lane < uint(ROWS)) {{
  T total = T(0.0f);
  for (uint s2 = 0; s2 < uint(TOPK); ++s2) {{
    T p = slot_vals[s2 * uint(ROWS) + lane] * scores[(size_t)row * (size_t)TOPK + s2];
    total = total + p;
  }}
  y[(size_t)row * (size_t)N + (size_t)tile * (size_t)ROWS + (size_t)lane] = total;
}}