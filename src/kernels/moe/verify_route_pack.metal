threadgroup uint values[320];
const uint n = uint(ids_shape[0]);
const uint lane = thread_index_in_threadgroup;
for (uint i = lane; i < n; i += 128) values[i] = ids[i];
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = lane; i < n; i += 128) {
  const uint key = values[i];
  uint rank = 0;
  for (uint j = 0; j < n; ++j) {
    const uint other = values[j];
    rank += uint(other < key || (other == key && j < i));
  }
  inverse[i] = rank;
  sorted[rank] = key;
  lhs[rank] = i / 10;
}