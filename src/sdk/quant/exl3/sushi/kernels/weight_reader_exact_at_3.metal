const uint lane = thread_position_in_grid.x % 32u;
const uint tile = thread_position_in_grid.x / 32u;
const nfrag f = nax_wfrag_k<uint(NHW)>((const device uint *)trellis + tile * (uint(NHW) / 2u), lane);
for (uint j = 0u; j < 8u; j++) result[tile * 256u + lane * 8u + j] = as_type<ushort>(f[j]);