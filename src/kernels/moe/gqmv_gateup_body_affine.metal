  float sjg = float(g_scales[gbase + (size_t)gi]);
  float bjg = float(g_biases[gbase + (size_t)gi]);
  float sju = float(u_scales[gbase + (size_t)gi]);
  float bju = float(u_biases[gbase + (size_t)gi]);
  for (int ki = 0; ki < VPW; ki += 4) {
    size_t xi = xrow + (size_t)(k_base + ki);
    uint32_t qg = packed_g >> (ki * BITS);
    uint32_t qu = packed_u >> (ki * BITS);
    float x0 = float(x[xi + 0]);
    float x1 = float(x[xi + 1]);
    float x2 = float(x[xi + 2]);
    float x3 = float(x[xi + 3]);
    g0 += x0 * (float((qg >> (0 * BITS)) & mask) * sjg + bjg);
    g1 += x1 * (float((qg >> (1 * BITS)) & mask) * sjg + bjg);
    g2 += x2 * (float((qg >> (2 * BITS)) & mask) * sjg + bjg);
    g3 += x3 * (float((qg >> (3 * BITS)) & mask) * sjg + bjg);
    u0 += x0 * (float((qu >> (0 * BITS)) & mask) * sju + bju);
    u1 += x1 * (float((qu >> (1 * BITS)) & mask) * sju + bju);
    u2 += x2 * (float((qu >> (2 * BITS)) & mask) * sju + bju);
    u3 += x3 * (float((qu >> (3 * BITS)) & mask) * sju + bju);
  }