  float sj = float(scales[gbase + (size_t)gi]);
  float bj = float(biases[gbase + (size_t)gi]);
  for (int ki = 0; ki < VPW; ki += 4) {
    size_t xi = xoff + (size_t)(k_base + ki);
    uint32_t q = packed >> (ki * BITS);
    a0 += float(x[xi + 0]) * (float((q >> (0 * BITS)) & mask) * sj + bj);
    a1 += float(x[xi + 1]) * (float((q >> (1 * BITS)) & mask) * sj + bj);
    a2 += float(x[xi + 2]) * (float((q >> (2 * BITS)) & mask) * sj + bj);
    a3 += float(x[xi + 3]) * (float((q >> (3 * BITS)) & mask) * sj + bj);
  }