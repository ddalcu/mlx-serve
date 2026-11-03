const uint tid = uint(thread_index_in_threadgroup);
const uint n = uint(count[0]);
const uint capacity = (n + uint(WIN) - 1u) / uint(WIN) + uint(EXPERTS);
threadgroup uint counts[uint(EXPERTS)];
for (uint i = tid; i < capacity; i += uint(EXPERTS)) { starts[i] = 0u; nlives[i] = 0u; }
for (uint i = tid; i < n; i += uint(EXPERTS)) inverse[uint(order[i])] = i;
uint lo = 0u, hi = n;
while (lo < hi) { uint m = (lo + hi) >> 1u; if (uint(eids[m]) < tid) lo = m + 1u; else hi = m; }
const uint first = lo;
hi = n;
while (lo < hi) { uint m = (lo + hi) >> 1u; if (uint(eids[m]) <= tid) lo = m + 1u; else hi = m; }
const uint length = lo - first;
counts[tid] = (length + uint(WIN) - 1u) / uint(WIN);
threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
uint offset = 0u;
for (uint e = 0u; e < tid; e++) offset += counts[e];
for (uint i = 0u; i < counts[tid]; i++) {
  starts[offset + i] = first + i * uint(WIN);
  nlives[offset + i] = min(uint(WIN), length - i * uint(WIN));
}