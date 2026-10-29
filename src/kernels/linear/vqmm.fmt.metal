auto part = simdgroup_index_in_threadgroup;
auto lane = thread_index_in_simdgroup;
auto tg_n = threadgroup_position_in_grid.y;

int K = int(K_size);
int N = int(N_size);
int K_by_p = K / 8;
int K_bytes = (K * BITS) / 8;
int K_by_gs = K / GS;
int per_part = K_by_p / K_PARTS;
int n0 = int(tg_n) * {d};
int p_start = int(part) * per_part;
int p_end = (int(part) == K_PARTS - 1) ? K_by_p : p_start + per_part;

float acc[{d}];
{s}

using Vec8 = vec<T, 8>;
const device Vec8 *xv = (const device Vec8*)x;

for (int pack = p_start + int(lane); pack < p_end; pack += 32) {{
  int k_base = pack * 8;
  int gi = k_base / GS;
{s}
}}

{s}

threadgroup float partials[K_PARTS * {d}];
if (lane == 0) {{
  _Pragma("unroll")
  for (int i = 0; i < {d}; ++i) {{
    partials[int(part) * {d} + i] = acc[i];
  }}
}}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (part == 0) {{
  for (int i = int(lane); i < {d}; i += 32) {{
    float total = 0.0f;
    _Pragma("unroll")
    for (int p = 0; p < K_PARTS; ++p) {{
      total += partials[p * {d} + i];
    }}
    int j = i / {d};
    int row = i - j * {d};
    y[row * N + n0 + j] = T(total);
  }}
}}