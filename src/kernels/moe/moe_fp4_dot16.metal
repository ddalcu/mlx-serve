inline float mlxserve_fp4_dot16(uint2 w, thread const float* x) {
  float a = 0.0f, b = 0.0f;
  for (int j = 0; j < 4; ++j) {
    const float2 p = mlxserve_e2m1x2(w.x, j);
    const float2 q = mlxserve_e2m1x2(w.y, j);
    a += x[j] * p.x + x[j + 4] * p.y;
    b += x[8 + j] * q.x + x[12 + j] * q.y;
  }
  return a + b;
}