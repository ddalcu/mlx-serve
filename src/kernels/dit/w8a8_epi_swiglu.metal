const int NO = N / 2;
for (uint16_t i = 0; i < cT.get_capacity(); i += 4) {
  auto ids = cT.get_multidimensional_index(i);
  const int n = n0 + ids[0], m = m0 + ids[1];
  if (m >= M || n >= N) continue;
  const float xsm = xs[m];
  const float g0 = float(cT[i]) * xsm * ws[n], u0 = float(cT[i + 1]) * xsm * ws[n + 1];
  const float g1 = float(cT[i + 2]) * xsm * ws[n + 2], u1 = float(cT[i + 3]) * xsm * ws[n + 3];
  *(device vec<T, 2>*)(y + (size_t)m * NO + n / 2) = vec<T, 2>(T(g0 / (1.0f + exp(-g0)) * u0), T(g1 / (1.0f + exp(-g1)) * u1));
}