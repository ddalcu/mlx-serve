for (uint u = 0u; u < 2u; u++) {
  for (uint o = 0u; o < uint(OTPT); o++) {
    const device uint *words = wp + u * SGS * OT * PACKED_W + o * PACKED_W;
    merged[u][o] = exl3_lane<N>(words, lane);
  }
}
for (uint u = 0u; u < 2u; u++) {
