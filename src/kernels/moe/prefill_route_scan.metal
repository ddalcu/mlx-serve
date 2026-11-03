threadgroup uint totals[E];
const uint lane = thread_index_in_threadgroup;
const uint nb = uint(counts_shape[0]) / E;
for (uint e = lane; e < E; e += 256) {
  uint total = 0;
  for (uint b = 0; b < nb; ++b) total += counts[b * E + e];
  totals[e] = total;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint e = lane; e < E; e += 256) {
  uint run = 0;
  for (uint j = 0; j < e; ++j) run += totals[j];
  for (uint b = 0; b < nb; ++b) {
    offsets[b * E + e] = run;
    run += counts[b * E + e];
  }
}