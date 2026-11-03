  float sj = mlxserve_e4m3(uint(scales[gbase + (size_t)gi]));
  size_t xi = xoff + (size_t)k_base;
  float2 p0 = mlxserve_e2m1x2(packed, 0), p1 = mlxserve_e2m1x2(packed, 1);
  float2 p2 = mlxserve_e2m1x2(packed, 2), p3 = mlxserve_e2m1x2(packed, 3);
  float d0 = float(x[xi + 0]) * p0.x + float(x[xi + 4]) * p0.y;
  float d1 = float(x[xi + 1]) * p1.x + float(x[xi + 5]) * p1.y;
  float d2 = float(x[xi + 2]) * p2.x + float(x[xi + 6]) * p2.y;
  float d3 = float(x[xi + 3]) * p3.x + float(x[xi + 7]) * p3.y;
  a0 += d0 * sj; a1 += d1 * sj; a2 += d2 * sj; a3 += d3 * sj;