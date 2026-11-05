inline float mlxserve_a4_dot16(uint2 w, thread const float* x) {
  float a = 0.0f, b = 0.0f;
  for (int j = 0; j < 8; ++j) {
    a += x[j] * float((w.x >> (4 * j)) & 0xFu);
    b += x[8 + j] * float((w.y >> (4 * j)) & 0xFu);
  }
  return a + b;
}

inline float mlxserve_a3_dot16(const device ushort* wp, thread const float* x) {
  const uint u0 = wp[0], u1 = wp[1], u2 = wp[2];
  const uint w0 = u0 | (u1 << 16);
  const uint w1 = (u1 >> 8) | (u2 << 8);
  float a = 0.0f, b = 0.0f;
  for (int j = 0; j < 8; ++j) {
    a += x[j] * float((w0 >> (3 * j)) & 0x7u);
    b += x[8 + j] * float((w1 >> (3 * j)) & 0x7u);
  }
  return a + b;
}