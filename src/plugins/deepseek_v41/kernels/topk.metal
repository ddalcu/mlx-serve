const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int l = threadgroup_position_in_grid.y, n = x_shape[1];
const auto xr = x + size_t(l) * n;
threadgroup atomic_uint hist[256];
threadgroup uint t_sel[2], t_part[TG / 32];
uint prefix = 0, krem = K;
for (int pass = 0; pass < 4; ++pass) {
  const int shift = 24 - 8 * pass;
  const uint hmask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
  for (uint b = tid; b < 256; b += TG) atomic_store_explicit(&hist[b], 0u, metal::memory_order_relaxed);
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int i = int(tid); i < n; i += TG) {
    const uint key = dsv41_key(xr[i]);
    if ((key & hmask) == prefix) atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, metal::memory_order_relaxed);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (sg == 0) {
    // Bins 255..0 in lane order of 8 bins each: the digit where the count from the top reaches krem.
    uint c[8], tot = 0;
    for (int j = 0; j < 8; ++j) { c[j] = atomic_load_explicit(&hist[255 - (lane * 8 + j)], metal::memory_order_relaxed); tot += c[j]; }
    const uint before = metal::simd_prefix_exclusive_sum(tot);
    const bool mine = before < krem && before + tot >= krem;
    if (mine) {
      uint acc = before;
      int j = 0;
      for (; j < 8; ++j) { if (acc + c[j] >= krem) break; acc += c[j]; }
      t_sel[0] = prefix | (uint(255 - (lane * 8 + j)) << shift);
      t_sel[1] = krem - acc;
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  prefix = t_sel[0];
  krem = t_sel[1];
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}
// Contiguous index chunks a thread, so the output keeps index order.
const int C = (n + TG - 1) / TG, i0 = int(tid) * C, i1 = metal::min(n, i0 + C);
uint gt = 0, eq = 0;
for (int i = i0; i < i1; ++i) { const uint key = dsv41_key(xr[i]); gt += key > prefix; eq += key == prefix; }
// Exclusive scans across the threadgroup: ties before this chunk, then picks before it.
uint s = metal::simd_prefix_exclusive_sum(eq);
if (lane == 31) t_part[sg] = s + eq;
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
uint off = 0;
for (uint g = 0; g < sg; ++g) off += t_part[g];
const uint eq_before = s + off;
const uint take = eq_before >= krem ? 0u : metal::min(eq, krem - eq_before);
const uint sel = gt + take;
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
s = metal::simd_prefix_exclusive_sum(sel);
if (lane == 31) t_part[sg] = s + sel;
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
off = 0;
for (uint g = 0; g < sg; ++g) off += t_part[g];
uint pos = s + off, taken = 0;
for (int i = i0; i < i1; ++i) {
  const uint key = dsv41_key(xr[i]);
  if (key > prefix || (key == prefix && taken < take)) {
    taken += key == prefix;
    out[size_t(l) * K + pos++] = i;
  }
}