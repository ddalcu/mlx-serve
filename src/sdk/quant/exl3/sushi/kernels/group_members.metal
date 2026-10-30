const uint eid = uint(slots[first]);
ulong same = 0ul;
for (uint b = 0u; b < uint(NSLOTS); b += 32u) {
  const bool hit = b + lane < uint(NSLOTS) && uint(slots[b + lane]) == eid;
  same |= (ulong)static_cast<simd_vote::vote_t>(simd_ballot(hit)) << b;
}
if (popcount(same & ((1ul << first) - 1ul)) % uint(GROUP) != 0u) return;
ulong rest = same >> first;
uint members[GROUP];
uint m = 0u;
for (uint j = 0u; j < uint(GROUP); j++) {
  members[j] = first + (rest != 0ul ? uint(ctz(rest)) : 0u);
  m += rest != 0ul ? 1u : 0u;
  rest &= rest - 1ul;
}
const uint prow = (lane & 3u) * 2u;
const uint pcol = lane >> 2u;
const uint sh[8] = {exl3_lane_sh(N, 0u), exl3_lane_sh(N, 1u), exl3_lane_sh(N, 2u), exl3_lane_sh(N, 3u), exl3_lane_sh(N, 4u), exl3_lane_sh(N, 5u), exl3_lane_sh(N, 6u), exl3_lane_sh(N, 7u)};
