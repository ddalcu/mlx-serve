  const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * N);
  const device uint *wp = trellis_e + ((size_t)(tk0 + sg) * (size_t)OT + ot) * PACKED_W;
  for (uint c0 = tk0; c0 < tk1; c0 += CHUNK / TILE) {
    const uint c1 = min(c0 + CHUNK / TILE, tk1);
    const uint nb = (c1 - c0) * TILE / 128u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
