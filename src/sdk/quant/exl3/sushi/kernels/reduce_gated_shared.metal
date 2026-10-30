  const T lg = gate[0];
  const T ex = T(metal::precise::exp(float(metal::abs(lg))));
  const T rc = T(1.0f / float(T(1.0f + float(ex))));
  const float sig = float(lg < T(0.0f) ? rc : T(1.0f - float(rc)));
  y[yb + lane] = T(float(T(a0)) + float(T(sig * float(shared[yb + lane]))));
  y[yb + lane + 32u] = T(float(T(a1)) + float(T(sig * float(shared[yb + lane + 32u]))));
  y[yb + lane + 64u] = T(float(T(a2)) + float(T(sig * float(shared[yb + lane + 64u]))));
  y[yb + lane + 96u] = T(float(T(a3)) + float(T(sig * float(shared[yb + lane + 96u]))));