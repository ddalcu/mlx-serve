threadgroup float partial[4 * 256];
threadgroup half prepared[uint(IDIM)];
{
 const uint slot = threadgroup_position_in_grid.y;
 const uint eid = uint(slots[slot]);
 const ushort lane = thread_index_in_simdgroup;
 for (uint block = simdgroup_index_in_threadgroup; block < uint(IDIM)/128u; block += 4u) {
  const uint base = block * 128u;
  const size_t xb = (size_t)slot * uint(IDIM) + base;
  const size_t sb = (size_t)eid * uint(IDIM) + base;
