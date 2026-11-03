float M = -INFINITY, l = 0.0f, w[NSG];
for (int g = 0; g < NSG; ++g) M = metal::max(M, t_mx[g]);
for (int g = 0; g < NSG; ++g) { w[g] = t_den[g] > 0.0f ? metal::exp(t_mx[g] - M) : 0.0f; l += t_den[g] * w[g]; }
const size_t base = (size_t(i) * H + h) * S + sp;
for (int d = sg * 32 + lane; d < D; d += NSG * 32) {
  float o = 0.0f;
  for (int g = 0; g < NSG; ++g) o += t_acc[g * D + d] * w[g];
  pacc[base * D + d] = o;
}
if (sg == 0 && lane == 0) { pm[base] = M; pl[base] = l; }