auto lane = thread_index_in_simdgroup;
uint n = thread_position_in_grid.y;      // output row within the expert
uint e = thread_position_in_grid.z;      // top-K slot

int K = int(K_size);
int N = int(N_size);
constexpr int VPW = (BITS == 3) ? 8 : 32 / BITS;  // values per pack: a word, or 3 bytes at 3-bit
int K_by_p = K / VPW;                    // packs per row
int K_by_w = K * BITS / 32;              // words per row
int K_by_gs = K / GS;                    // quant groups per row
uint mask = (1u << BITS) - 1u;

uint eid = inds[e];                      // bank row for this slot
size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_w + (size_t)n * (size_t)K_by_w;
size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
size_t xoff = {s};

// Four independent accumulators so the per-lane FMA chain is not one
// serial dependency; VPW is 16/8/4 for bits 2/4/8, always a multiple of 4.
float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
for (int pack = int(lane); pack < K_by_p; pack += 32) {{
  uint32_t packed = mlxserve_qpack<BITS>(w_q, wbase, pack);
  int k_base = pack * VPW;
  int gi = k_base / GS;                  // GS >= VPW, so one group per word
{s}
}}
float acc = simd_sum((a0 + a1) + (a2 + a3)){s};
if (lane == 0) {{
  float o = float(T(acc));
  if (ACT2) {{ o = metal::max(o, 0.0f); o = o * o; }}
  y[(size_t)e * (size_t)N + (size_t)n] = T(o);
}}