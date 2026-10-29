  float sjg = mlxserve_e4m3(uint(g_scales[gbase + (size_t)gi]));
  float sju = mlxserve_e4m3(uint(u_scales[gbase + (size_t)gi]));
  float dg[4], du[4];
  size_t xi = xrow + (size_t)k_base;
  for (int j = 0; j < 4; ++j) {
    const float xa = float(x[xi + j]);
    const float xb = float(x[xi + j + 4]);
    const float2 pg2 = mlxserve_e2m1x2(packed_g, j);
    const float2 pu2 = mlxserve_e2m1x2(packed_u, j);
    dg[j] = xa * pg2.x + xb * pg2.y;
    du[j] = xa * pu2.x + xb * pu2.y;
  }
  float dg0 = dg[0], dg1 = dg[1], dg2 = dg[2], dg3 = dg[3];
  float du0 = du[0], du1 = du[1], du2 = du[2], du3 = du[3];
  g0 += dg0 * sjg; g1 += dg1 * sjg; g2 += dg2 * sjg; g3 += dg3 * sjg;
  u0 += du0 * sju; u1 += du1 * sju; u2 += du2 * sju; u3 += du3 * sju;