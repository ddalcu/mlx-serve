  if (d < D) {
    const auto pr = spost + l * HC;
    const auto cb = scomb + l * HC * HC;
    float sv[HC];
    for (int j = 0; j < HC; ++j) sv[j] = float(xr[j * D + d]);
    const float yv = float(y[size_t(l) * D + d]);
    for (int i = 0; i < HC; ++i) {
      float acc = 0.0f;
      for (int j = 0; j < HC; ++j) acc = metal::fma(cb[j * HC + i], sv[j], acc);
      const T o = T(metal::fma(pr[i], yv, acc));
      v[i][c] = float(o);
      if (role == 0) x1[(size_t(l) * HC + i) * D + d] = o;
    }
  }
}