auto sg = simdgroup_index_in_threadgroup;
auto lane = thread_index_in_simdgroup;
auto tg_n = threadgroup_position_in_grid.y;

int K = int(K_size);
int N = int(N_size);
int K_by_p = K / 8;
int K_bytes = (K * BITS) / 8;
int K_by_gs = K / GS;
int n0 = (int(tg_n) * NSG + int(sg)) * {d};
if (n0 + {d} >= N) {{ return; }}

float acc[{d}];
{s}

using Vec8 = vec<T, 8>;
const device Vec8 *xv = (const device Vec8*)x;

for (int pack = int(lane); pack < K_by_p; pack += 32) {{
  int k_base = pack * 8;
  int gi = k_base / GS;
{s}
}}

{s}

if (lane < {d}) {{
  int j = int(lane) / {d};
  int row = int(lane) - j * {d};
  y[row * N + n0 + j] = T(acc[int(lane)]);
}}