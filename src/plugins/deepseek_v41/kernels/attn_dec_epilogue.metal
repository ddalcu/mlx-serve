float M = sink[h];
for (int g = 0; g < NSG; ++g) M = metal::max(M, t_mx[g]);
float total = metal::exp(sink[h] - M);
float w[NSG];
for (int g = 0; g < NSG; ++g) { w[g] = t_den[g] > 0.0f ? metal::exp(t_mx[g] - M) : 0.0f; total += t_den[g] * w[g]; }
for (int d = sg * 32 + lane; d < D; d += NSG * 32) {
  float o = 0.0f;
  for (int g = 0; g < NSG; ++g) o += t_acc[g * D + d] * w[g];
  out[(i * H + h) * D + d] = dsv41_unrope<OT, RD, D>(OT(o / total), d, p0 + i, periods);
}