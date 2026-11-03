  for (uint o = 0u; o < uint(OTPT); o++) {
    float clo = (acc[j][o][0] + acc[j][o][1]) + (acc[j][o][2] + acc[j][o][3]);
    float chi = (acc[j][o][4] + acc[j][o][5]) + (acc[j][o][6] + acc[j][o][7]);
    clo += simd_shuffle_xor(clo, 1u);
    chi += simd_shuffle_xor(chi, 1u);
    clo += simd_shuffle_xor(clo, 2u);
    chi += simd_shuffle_xor(chi, 2u);
    if ((lane & 3u) == 0u) {
      partial[((j * uint(OTPT) + o) * SGS + sg) * 16u + pcol] = clo;
      partial[((j * uint(OTPT) + o) * SGS + sg) * 16u + pcol + 8u] = chi;
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid < 16u * uint(OTPT) * m) {
  const uint j = lid / (16u * uint(OTPT));
  const uint o = (lid >> 4u) % uint(OTPT);
  float sum = 0.0f;
  for (uint g = 0u; g < SGS; g++) {
    sum += partial[((j * uint(OTPT) + o) * SGS + g) * 16u + (lid & 15u)];
  }
