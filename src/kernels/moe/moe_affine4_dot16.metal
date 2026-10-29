inline float mlxserve_a4_dot16(uint2 w, thread const float* x) {
  float a = 0.0f, b = 0.0f;
  for (int j = 0; j < 8; ++j) {
    a += x[j] * float((w.x >> (4 * j)) & 0xFu);
    b += x[8 + j] * float((w.y >> (4 * j)) & 0xFu);
  }
  return a + b;
}