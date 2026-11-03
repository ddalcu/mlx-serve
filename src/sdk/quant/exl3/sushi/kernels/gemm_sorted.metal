threadgroup float partial[2 * 256];
uint ot = uint(threadgroup_position_in_grid.x);
uint win = uint(threadgroup_position_in_grid.y);
uint sg = uint(simdgroup_index_in_threadgroup);
uint lane = uint(thread_index_in_simdgroup);
uint lid = uint(thread_index_in_threadgroup);
constexpr uint TILE = 16u;
constexpr uint N = uint(NHW);
constexpr uint PACKED_HW = N;
constexpr uint PACKED_W = N / 2u;
constexpr uint IT = uint(IDIM) / TILE;
constexpr uint OT = uint(ODIM) / TILE;
const uint start = wstarts[win];
const uint n = wnlive[win];
if (n == 0u || n > uint(WIN)) return;
const uint pg = sg >> 1u;
const uint th = sg & 1u;
const uint first = th * 128u + lane * 4u;
uint pos[4];
uint irow[4];
for (uint s = 0u; s < 4u; s++) {
  const uint src = first + s;
  const uint ln = src >> 3u;
  const uint sl = src & 7u;
  const uint prow = (ln & 3u) * 2u;
  const uint col0 = ln >> 2u;
  uint p;
  switch (sl) {
    case 0u: p = prow * 16u + col0; break;
    case 1u: p = (prow + 1u) * 16u + col0; break;
    case 2u: p = (prow + 8u) * 16u + col0; break;
    case 3u: p = (prow + 9u) * 16u + col0; break;
    case 4u: p = prow * 16u + col0 + 8u; break;
    case 5u: p = (prow + 1u) * 16u + col0 + 8u; break;
    case 6u: p = (prow + 8u) * 16u + col0 + 8u; break;
    default: p = (prow + 9u) * 16u + col0 + 8u; break;
  }
  pos[s] = p;
  irow[s] = p >> 4u;
}
uint row = start;
const uint end = start + n;
while (row < end) {
const uint run0 = row;
const uint eid = uint(eids[row]);
uint run_end = row + 1u;
while (run_end < end && uint(eids[run_end]) == eid) run_end++;
const uint nlive = run_end - row;
float4 acc[8][4];
for (uint g = 0u; g < 8u; g++) {
  acc[g][0] = float4(0.0f);
  acc[g][1] = float4(0.0f);
  acc[g][2] = float4(0.0f);
  acc[g][3] = float4(0.0f);
}
  for (uint tk = pg; tk < IT; tk += 2u) {
    const device uint* words = (const device uint*)(trellis + ((((size_t)eid * (size_t)IT + tk) * (size_t)OT + ot) * PACKED_HW));
    const exl3_win wlo_v = exl3_pair_window(first, N);
    const exl3_win whi_v = exl3_pair_window(first + 2u, N);
    const ulong m0 = ((ulong)words[wlo_v.i0] << 32) | (ulong)words[wlo_v.i1];
    const uint f0 = uint(m0 >> wlo_v.sh);
    const ulong m1 = ((ulong)words[whi_v.i0] << 32) | (ulong)words[whi_v.i1];
    const uint f1 = uint(m1 >> whi_v.sh);
    const uint2 lo = uint2((f0 >> wlo_v.fresh) & 0xffffu, f0 & 0xffffu);
    const uint2 hi = uint2((f1 >> whi_v.fresh) & 0xffffu, f1 & 0xffffu);
    const float2 wlo = exl3_decode2(lo);
    const float2 whi = exl3_decode2(hi);
    const float4 wt = float4(wlo.x, wlo.y, whi.x, whi.y);
    const uint ib = tk * TILE;
    for (uint g = 0u; g < 8u; g++) {
      if (g * 4u >= nlive) break;
      const uint base_r = run0 + g * 4u;
      for (uint s = 0u; s < 4u; s++) {
        const size_t col = (size_t)(ib + irow[s]);
        float4 a = float4(0.0f);
        a.x = float(x[(size_t)base_r * (size_t)(IDIM) + col]);
        if (g * 4u + 1u < nlive) a.y = float(x[(size_t)(base_r + 1u) * (size_t)(IDIM) + col]);
        if (g * 4u + 2u < nlive) a.z = float(x[(size_t)(base_r + 2u) * (size_t)(IDIM) + col]);
        if (g * 4u + 3u < nlive) a.w = float(x[(size_t)(base_r + 3u) * (size_t)(IDIM) + col]);
        acc[g][s] = fma(a, float4(wt[s]), acc[g][s]);
      }
    }
  }
  for (uint g = 0u; g < 8u; g++) {
    for (uint r = 0u; r < 4u; r++) {
      const uint rr = g * 4u + r;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint s = 0u; s < 4u; s++) {
        partial[pg * 256u + pos[s]] = acc[g][s][r];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lid < 16u && rr < nlive) {
        float sum = 0.0f;
        for (uint pr = 0u; pr < 16u; pr++) {
          const uint p = pr * 16u + lid;
          sum += partial[p] + partial[256u + p];
        }
        y[(size_t)(run0 + rr) * (size_t)(ODIM) + ot * TILE + lid] = half(exl3_out(sum));
      }
    }
  }
  row = run_end;
}