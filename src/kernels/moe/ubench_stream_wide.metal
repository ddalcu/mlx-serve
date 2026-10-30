uint sg = thread_position_in_grid.x / 32;
auto lane = thread_index_in_simdgroup;
const device uint4* p = (const device uint4*)a + (size_t)sg * (size_t)SPAN4;
uint acc = 0;
if (STRIDED) {
  // grid-stride: at each step the whole GPU reads one contiguous slab
  const device uint4* q = (const device uint4*)a;
  size_t gid = thread_position_in_grid.x;
  size_t nth = (size_t)NSG * 32;
  for (size_t i = gid; i < nth * (size_t)SPAN4 / 32 ; i += nth * (size_t)UNROLL) {
    uint4 v[UNROLL];
    for (int u = 0; u < UNROLL; ++u) v[u] = q[i + nth * (size_t)u];
    for (int u = 0; u < UNROLL; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  out[sg] = simd_sum(acc);
  return;
}
for (size_t i = lane; i < (size_t)SPAN4; i += (size_t)(32 * UNROLL)) {
  uint4 v[UNROLL];
  for (int u = 0; u < UNROLL; ++u) v[u] = p[i + (size_t)(32 * u)];
  for (int u = 0; u < UNROLL; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
}
out[sg] = simd_sum(acc);