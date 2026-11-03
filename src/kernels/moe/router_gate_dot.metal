if (tid >= 32) {
  float ga = 0.0f;
  for (uint k = tid - 32; k < uint(GK); k += TG - 32) ga += float(gx[(size_t)row * (size_t)GK + k]) * float(gw[k]);
  ga = simd_sum(ga);
  if (lane == 0) gp[tid / 32] = ga;
}