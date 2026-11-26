// Inkling 2: cuts sorted expert indices `idx` [M] u32 into (row_start, expert, rows, 0) tiles of at most BM rows
// of one expert, expert-major. `params` = { M, E, max_tiles }. grid (256, 1, 1), threadgroup (256, 1, 1).
const int M = params[0];
const int E = params[1];
const uint max_tiles = uint(params[2]);
const uint lid = thread_index_in_threadgroup;
const uint tg_size = threads_per_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
threadgroup uint run_start[MAXE];
threadgroup uint run_end[MAXE];
threadgroup uint simd_tot[32];
for (int e = int(lid); e < E; e += int(tg_size)) {
  run_start[e] = 0;
  run_end[e] = 0;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int g0 = 4 * int(lid); g0 < M; g0 += 4 * int(tg_size)) {
  const int cnt = min(4, M - g0);
  uint v[6];
  v[0] = g0 > 0 ? idx[g0 - 1] : 0xffffffffu;
  for (int j = 0; j < 4; ++j) v[j + 1] = j < cnt ? idx[g0 + j] : 0xffffffffu;
  v[5] = g0 + 4 < M ? idx[g0 + 4] : 0xffffffffu;
  for (int j = 0; j < cnt; ++j) {
    const uint e = v[j + 1];
    if (e < uint(E)) {
      if (v[j] != e) run_start[e] = uint(g0 + j);
      if (v[j + 2] != e) run_end[e] = uint(g0 + j + 1);
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint n_simd = (tg_size + 31) / 32;
uint running = 0;
for (int base = 0; base < E; base += int(tg_size)) {
  const int e = base + int(lid);
  uint start = 0;
  uint cnt = 0;
  if (e < E) {
    start = run_start[e];
    const uint end = run_end[e];
    cnt = end > start ? end - start : 0;
  }
  const uint nt = (cnt + BM - 1) / BM;
  const uint local = simd_prefix_exclusive_sum(nt);
  const uint stot = simd_sum(nt);
  if (lane == 0) simd_tot[sg] = stot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint prefix = 0;
  uint total = 0;
  for (uint i = 0; i < n_simd; ++i) {
    const uint val = simd_tot[i];
    prefix += (i < sg) ? val : 0;
    total += val;
  }
  const uint off = running + prefix + local;
  for (uint j = 0; j < nt && off + j < max_tiles; ++j) {
    const uint r = start + j * BM;
    *((device uint4*)tiles + off + j) = uint4(r, uint(e), min(uint(BM), start + cnt - r), 0);
  }
  running += total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (lid == 0) tile_count[0] = min(running, max_tiles);
