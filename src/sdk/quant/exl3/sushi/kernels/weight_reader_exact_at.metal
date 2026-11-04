const uint lane = thread_position_in_grid.x % 32u;
const uint tile = thread_position_in_grid.x / 32u;
half2 p[4];
smat_group<uint(NHW)>((const device uint *)trellis + tile * (uint(NHW) / 2u), lane, p);
for (uint j = 0u; j < 4u; j++) {
  result[tile * 256u + lane * 8u + 2u * j] = as_type<ushort>(p[j].x);
  result[tile * 256u + lane * 8u + 2u * j + 1u] = as_type<ushort>(p[j].y);
}