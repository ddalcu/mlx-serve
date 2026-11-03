#pragma clang loop unroll_count(2)
for (uint tk = 0u; tk < IT; tk++) {
  const device uint *words = wp;
  wp += (size_t)OT * PACKED_W;
