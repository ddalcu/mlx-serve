constexpr uint N = uint(NHW);
const uint lane = thread_position_in_grid.x % 32u;
const uint tile = thread_position_in_grid.x / 32u;
const device uint *words = (const device uint *)trellis + tile * (uint(NHW) / 2u);
const auto bits = exl3_lane<uint(NHW)>(words, lane);
for (uint j = 0u; j < 8u; j += 2u) {
  const uint2 cw = uint2(exl3_lane_word(bits, exl3_lane_sh(uint(NHW), j)), exl3_lane_word(bits, exl3_lane_sh(uint(NHW), j + 1u))) & uint2(0xffffu);
  const half2 h = exl3_pairh(cw);
  result[tile * 256u + lane * 8u + j] = as_type<ushort>(h.x);
  result[tile * 256u + lane * 8u + j + 1u] = as_type<ushort>(h.y);
}