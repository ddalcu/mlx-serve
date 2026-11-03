  float l = float(logits[rbase + e]);
  float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(l)));
  float sg = (l < 0.0f) ? sy : (1.0f - sy);