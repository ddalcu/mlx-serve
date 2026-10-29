    ins[j][0] = float(q[prow]);
    ins[j][1] = float(q[prow + 1u]);
    ins[j][2] = float(q[prow + 8u]);
    ins[j][3] = float(q[prow + 9u]);
  }
  for (uint o = 0u; o < uint(OTPT); o++) {
    for (uint p = 0u; p < 4u; p++) {
      const uint2 cw = uint2(exl3_lane_word(merged[u][o], sh[p * 2u]), exl3_lane_word(merged[u][o], sh[p * 2u + 1u])) & uint2(0xffffu);
      const float2 w = exl3_decode2(cw);
